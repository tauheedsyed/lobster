"""Host-owned, expiring request authentication; no discovery or authority minting."""

import asyncio
import math
import time
from concurrent.futures import Future
from threading import RLock

from lobster.core.exceptions import InferenceRefused, reraise_inference_refusal


def canonical_refusal(category, retryable, default="auth"):
    if (
        type(category) is str
        and category
        in {"context", "auth", "policy", "budget", "provider", "accounting"}
        and type(retryable) is bool
    ):
        return category, retryable
    return default, False


def refusal_spec(error, default="auth"):
    """Copy only canonical primitives, never mutable exception attributes."""
    try:
        reraise_inference_refusal(error)
    except InferenceRefused as refusal:
        return canonical_refusal(refusal.category, refusal.retryable, default)
    return default, False


def _unwrap(result):
    if type(result) is tuple:
        raise InferenceRefused(result[0], retryable=result[1]) from None
    return result


class RequestAuth:
    """Immutable ephemeral header value; redacted and nonserializable."""

    __slots__ = ("_value", "expires_at", "generation")

    def __init__(self, value, *, expires_at, generation=1):
        if (
            type(value) is not str
            or not 0 < len(value) <= 4096
            or any(not 0x21 <= ord(char) <= 0x7E for char in value)
            or type(expires_at) is not float
            or not math.isfinite(expires_at)
            or not 0 < expires_at < 253402300800
            or type(generation) is not int
            or generation <= 0
        ):
            raise InferenceRefused("auth")
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "generation", generation)

    def __setattr__(self, name, value):
        raise TypeError("Request authentication is immutable")

    @property
    def value(self):
        return self._value

    def __repr__(self):
        return "RequestAuth(<ephemeral>)"

    def __reduce__(self):
        raise TypeError("Request authentication cannot be serialized")


class ExpiringRequestAuth:
    """One supplier-owned refresh shared across threads/async callers.

    The host owns fixed-origin control transport and response scope validation.
    Both trusted callbacks MUST have transport deadlines <= refresh_timeout;
    arbitrary blocking Python cannot be forcibly cancelled. Async renewal also
    has an enforced timeout. Cancelling a waiter does not cancel the renewal;
    close drains it, erases the current grant, and forbids replacement install.
    Failure state contains only category/retryability, never exception objects.
    """

    def __init__(
        self,
        *,
        initial,
        refresh,
        arefresh,
        refresh_ahead=120.0,
        max_lifetime=300.0,
        renewal_deadline=None,
        refresh_timeout=10.0,
        clock=time.time,
    ):
        if (
            not callable(refresh)
            or not callable(arefresh)
            or not callable(clock)
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in (refresh_ahead, max_lifetime, refresh_timeout)
            )
            or not 0 < refresh_ahead < max_lifetime
            or not 0 < refresh_timeout <= 10
        ):
            raise InferenceRefused("context")
        if renewal_deadline is not None and (
            type(renewal_deadline) is not float
            or not math.isfinite(renewal_deadline)
            or not 0 < renewal_deadline < 253402300800
        ):
            raise InferenceRefused("context")
        self._deadline, self._timeout = renewal_deadline, refresh_timeout
        self._refresh, self._arefresh, self._clock = refresh, arefresh, clock
        self._ahead, self._lifetime = refresh_ahead, max_lifetime
        self._lock = RLock()
        self._closed = False
        self._refusal = None
        self._future = None
        self._renewal_task = None
        self._closing = None
        self._current = self._validate(initial)

    def __repr__(self):
        return "ExpiringRequestAuth(<owned>)"

    def __reduce__(self):
        raise TypeError("Request authentication suppliers cannot be serialized")

    def _validate(self, value):
        now = self._clock()
        if (
            not isinstance(value, RequestAuth)
            or not 0 < value.expires_at - now <= self._lifetime
            or (self._deadline is not None and value.expires_at > self._deadline)
        ):
            raise InferenceRefused("auth")
        return value

    def _begin(self):
        with self._lock:
            if self._refusal is not None:
                return self._refusal, None, False
            if self._closed:
                return ("auth", False), None, False
            if self._future is not None:
                return self._current, self._future, False
            now = self._clock()
            if self._current.expires_at <= now:
                self._refusal = ("auth", False)
                return self._refusal, None, False
            if (
                self._current.expires_at - now > self._ahead
                or self._current.expires_at == self._deadline
            ):
                return self._current, None, False
            self._future = Future()
            return self._current, self._future, True

    def _complete(self, future, value=None, failure=None):
        with self._lock:
            if failure is None:
                try:
                    value = self._validate(value)
                    if (
                        self._closed
                        or (
                            value.expires_at - self._clock() <= self._ahead
                            and value.expires_at != self._deadline
                        )
                        or value.value == self._current.value
                        or value.generation <= self._current.generation
                    ):
                        failure = ("auth", False)
                except Exception as error:
                    failure = refusal_spec(error)
            if failure is not None:
                self._refusal = failure
                result = failure
            else:
                self._current = value
                result = value
            # Shared futures never hold callback/SDK exception graphs.
            future.set_result(result)
            self._future = None
        return result

    def get(self):
        current, future, leader = self._begin()
        if future is None:
            return _unwrap(current)
        if leader:
            value, failure = None, None
            started = time.monotonic()
            try:
                value = self._refresh(current)
            except BaseException as error:
                failure = refusal_spec(error)
            if time.monotonic() - started > self._timeout:
                failure = ("auth", False)
            result = self._complete(future, value, failure)
        else:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                result = future.result()
            else:
                # Never deadlock the loop which owns an async refresh.
                raise InferenceRefused("context")
        return _unwrap(result)

    async def aget(self):
        current, future, leader = self._begin()
        if future is None:
            return _unwrap(current)
        if leader:
            self._renewal_task = asyncio.create_task(self._renew(future, current))
        # This waiter may cancel immediately; renewal is supplier-owned and
        # other consumers/close still observe its one shared result.
        result = await asyncio.shield(asyncio.wrap_future(future))
        return _unwrap(result)

    async def _renew(self, future, current):
        value, failure = None, None
        try:
            value = await asyncio.wait_for(self._arefresh(current), self._timeout)
        except BaseException as error:
            failure = refusal_spec(error)
        self._complete(future, value, failure)
        # Do not retain the replacement grant as a completed supplier task result.
        return None

    def stop(self):
        with self._lock:
            self._closed = True

    async def _erase(self):
        try:
            with self._lock:
                future = self._future
            if future is not None:
                await asyncio.wrap_future(future)
        finally:
            with self._lock:
                self._current = None
                self._renewal_task = None

    async def aclose(self):
        self.stop()
        with self._lock:
            if self._closing is None:
                self._closing = asyncio.create_task(self._erase())
            task = self._closing
        cancelled = False
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                cancelled = True
                if task.done():
                    task.result()
                    break
        if cancelled:
            raise asyncio.CancelledError
