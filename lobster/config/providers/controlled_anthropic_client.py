"""Explicit native Messages clients with request admission and owned streams."""

import asyncio
import json
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from threading import RLock

import anthropic
from langchain_anthropic import ChatAnthropic
from pydantic import PrivateAttr

from lobster.config.inference_context import (
    InferenceContext,
    NativeRequestView,
    RequestDescriptor,
    UsageOutcome,
)
from lobster.core.exceptions import InferenceRefused, reraise_inference_refusal


class _Attempt:
    def __init__(self):
        self.permit = None
        self.admitted = False
        self.issued = False
        self.finalized = False
        self.raw = None
        self.stream = None
        self.counts = {}
        self.model = None
        self.finish = None
        self.stopped = False
        self.invalid_usage = False
        self.lock = RLock()
        self.asynchronous = False
        self.beta = False

    def observe_message(self, message):
        self.model = message.model
        self.observe_usage(message.usage)
        self.observe_finish(message.stop_reason)

    def observe_finish(self, finish):
        if type(finish) is str and finish in {
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "pause_turn",
            "refusal",
            "model_context_window_exceeded",
        }:
            self.finish = finish

    def observe_usage(self, usage):
        if usage is None:
            return
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            value = getattr(usage, key, None)
            if value is None:
                continue
            if type(value) is not int or value < 0:
                self.invalid_usage = True
                continue
            self.counts[key] = value
        creation = getattr(usage, "cache_creation", None)
        if creation is not None:
            for key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
                value = getattr(creation, key, None)
                if value is not None:
                    if type(value) is not int or value < 0:
                        self.invalid_usage = True
                    else:
                        self.counts[key] = value

    def observe_event(self, event):
        if event.type == "message_start":
            self.observe_message(event.message)
        elif event.type == "message_delta":
            self.observe_usage(event.usage)
            self.observe_finish(event.delta.stop_reason)
        elif event.type == "message_stop":
            self.stopped = True

    def outcome(self, kind):
        quality = "unknown"
        if (
            not self.invalid_usage
            and "input_tokens" in self.counts
            and "output_tokens" in self.counts
        ):
            quality = "complete" if kind == "completed" else "partial"
        return UsageOutcome(
            raw_usage=tuple(self.counts.items()),
            finish_reason=self.finish,
            actual_model_id=self.model,
            completeness=quality,
            request_issued=self.issued,
            outcome=kind,
        )


def _provider_error(error):
    if isinstance(error, Exception):
        reraise_inference_refusal(error)
        raise InferenceRefused("provider") from None
    raise error


class _Messages:
    def __init__(self, owner, asynchronous=False, beta=False):
        self.owner, self.asynchronous, self.beta = owner, asynchronous, beta
        self.with_raw_response = self

    def __getattr__(self, name):
        raise InferenceRefused("context")

    def create(self, **payload):
        if self.asynchronous:
            return self.owner.acreate(payload, self.beta)
        return self.owner.create(payload, self.beta)


class _ClientView:
    def __init__(self, owner, asynchronous=False):
        self.messages = _Messages(owner, asynchronous)
        self.beta = _BetaView(owner, asynchronous)

    def __getattr__(self, name):
        raise InferenceRefused("context")


class _BetaView:
    def __init__(self, owner, asynchronous):
        self.messages = _Messages(owner, asynchronous, True)

    def __getattr__(self, name):
        raise InferenceRefused("context")


class ControlledAnthropicClient:
    """Own explicit sync/async SDK clients; other client operations refuse.

    Inject ``sync_view`` and ``async_view`` into ControlledChatAnthropic. The
    model's stream scopes compensate for the pinned model's missing close-finally.
    """

    def __init__(
        self,
        *,
        context: InferenceContext,
        component,
        authorizer,
        validate_request,
        describe_request,
        api_key,
        base_url,
        http_client,
        http_async_client,
    ):
        if (
            not isinstance(context, InferenceContext)
            or context.provider != "anthropic"
            or not api_key
            or not base_url.startswith("https://")
        ):
            raise InferenceRefused("context")
        self.context, self.component = context, component
        self.authorizer = authorizer
        self.validate_request, self.describe_request = (
            validate_request,
            describe_request,
        )
        self.endpoint = base_url.rstrip("/") + "/v1/messages"
        self._api_key = api_key
        self._attempt = ContextVar(f"native-attempt-{id(self)}", default=None)
        self._scope = ContextVar(f"native-scope-{id(self)}", default=None)
        self._lock = RLock()
        self._attempt_ids = set()
        self._pending = set()
        self._closed = False
        http_client.event_hooks.setdefault("request", []).append(self._before_http)
        http_async_client.event_hooks.setdefault("request", []).append(
            self._abefore_http
        )
        self._sync = anthropic.Client(
            api_key=api_key, base_url=base_url, http_client=http_client, max_retries=0
        )
        self._async = anthropic.AsyncClient(
            api_key=api_key,
            base_url=base_url,
            http_client=http_async_client,
            max_retries=0,
        )
        self.sync_view, self.async_view = _ClientView(self), _ClientView(self, True)

    def __repr__(self):
        return "ControlledAnthropicClient(<owned>)"

    def _descriptor(self, request):
        attempt = self._attempt.get()
        if self._closed or attempt is None or attempt.admitted:
            raise InferenceRefused("context")
        expected_endpoint = self.endpoint + ("?beta=true" if attempt.beta else "")
        if (
            str(request.url) != expected_endpoint
            or request.method != "POST"
            or request.headers.get("x-api-key") != self._api_key
        ):
            raise InferenceRefused("context")
        payload = json.loads(request.content)
        view = NativeRequestView(
            "Messages",
            str(request.url),
            {
                **payload,
                "betas": tuple(
                    filter(None, request.headers.get("anthropic-beta", "").split(","))
                ),
            },
        )
        self.validate_request(view)
        descriptor = self.describe_request(view)
        if not isinstance(descriptor, RequestDescriptor):
            raise InferenceRefused("context")
        identity = (
            descriptor.provider,
            descriptor.model_id,
            descriptor.component,
            descriptor.binding_id,
            descriptor.binding_version,
        )
        expected = (
            self.context.provider,
            self.context.model_id,
            self.component,
            self.context.binding_id,
            self.context.binding_version,
        )
        bound = payload.get("max_tokens")
        if (
            identity != expected
            or payload.get("model") != self.context.model_id
            or type(bound) is not int
            or not 0 < bound <= self.context.max_output_tokens
            or descriptor.max_output_tokens != self.context.max_output_tokens
            or descriptor.deadline_at <= time.time()
        ):
            raise InferenceRefused("context")
        with self._lock:
            if descriptor.attempt_id in self._attempt_ids:
                raise InferenceRefused("context")
            self._attempt_ids.add(descriptor.attempt_id)
        return attempt, descriptor

    def _before_http(self, request):
        attempt, descriptor = self._descriptor(request)
        attempt.permit = self.authorizer.before_request(descriptor)
        attempt.admitted = True
        if descriptor.deadline_at <= time.time():
            raise InferenceRefused("context")
        attempt.issued = True

    async def _abefore_http(self, request):
        attempt, descriptor = self._descriptor(request)
        attempt.permit = await self.authorizer.abefore_request(descriptor)
        attempt.admitted = True
        if descriptor.deadline_at <= time.time():
            raise InferenceRefused("context")
        attempt.issued = True

    def _start(self, asynchronous=False):
        with self._lock:
            if self._closed or self._attempt.get() is not None:
                raise InferenceRefused("context")
            attempt = _Attempt()
            attempt.asynchronous = asynchronous
            self._pending.add(attempt)
        scope = self._scope.get()
        if scope is not None:
            scope.append(attempt)
        return attempt

    def finalize(self, attempt, kind):
        with attempt.lock:
            if attempt.finalized:
                return
            attempt.finalized = True
        try:
            if attempt.stream is not None:
                attempt.stream.close()
            elif attempt.raw is not None:
                attempt.raw.close()
        finally:
            with self._lock:
                self._pending.discard(attempt)
            if attempt.admitted:
                self.authorizer.after_request(attempt.permit, attempt.outcome(kind))

    async def afinalize(self, attempt, kind):
        with attempt.lock:
            if attempt.finalized:
                return
            attempt.finalized = True

        async def finish():
            try:
                if attempt.stream is not None:
                    await attempt.stream.close()
                elif attempt.raw is not None:
                    await attempt.raw.close()
            finally:
                with self._lock:
                    self._pending.discard(attempt)
                if attempt.admitted:
                    await self.authorizer.aafter_request(
                        attempt.permit, attempt.outcome(kind)
                    )

        task = asyncio.create_task(finish())
        cancelled = False
        while True:
            try:
                result = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                cancelled = True
                if task.done():
                    result = task.result()
                    break
        if cancelled:
            raise asyncio.CancelledError
        return result

    def create(self, payload, beta):
        attempt = self._start()
        attempt.beta = beta
        token = self._attempt.set(attempt)
        try:
            client = self._sync.beta if beta else self._sync
            attempt.raw = client.messages.with_raw_response.create(**payload)
            return _RawResponse(self, attempt, bool(payload.get("stream")))
        except BaseException as error:
            self.finalize(
                attempt, "refused" if isinstance(error, InferenceRefused) else "failed"
            )
            _provider_error(error)
        finally:
            self._attempt.reset(token)

    async def acreate(self, payload, beta):
        attempt = self._start(True)
        attempt.beta = beta
        token = self._attempt.set(attempt)
        try:
            client = self._async.beta if beta else self._async
            attempt.raw = await client.messages.with_raw_response.create(**payload)
            return _AsyncRawResponse(self, attempt, bool(payload.get("stream")))
        except BaseException as error:
            kind = (
                "cancelled"
                if isinstance(error, asyncio.CancelledError)
                else "refused" if isinstance(error, InferenceRefused) else "failed"
            )
            await self.afinalize(attempt, kind)
            _provider_error(error)
        finally:
            self._attempt.reset(token)

    @contextmanager
    def stream_scope(self):
        attempts = []
        token = self._scope.set(attempts)
        try:
            yield
        finally:
            try:
                self._scope.reset(token)
            except ValueError:
                # An abandoned generator may be closed in another task/context.
                # Ownership cleanup must still run, irrespective of token reset.
                self._scope.set(None)
            for attempt in attempts:
                self.finalize(attempt, "cancelled")

    @asynccontextmanager
    async def astream_scope(self):
        attempts = []
        token = self._scope.set(attempts)
        try:
            yield
        finally:
            try:
                self._scope.reset(token)
            except ValueError:
                self._scope.set(None)
            for attempt in attempts:
                await self.afinalize(attempt, "cancelled")

    def close(self):
        with self._lock:
            if any(not item.finalized and item.raw is None for item in self._pending):
                raise InferenceRefused("context")
            self._closed = True
        error = None
        try:
            for attempt in list(self._pending):
                if not attempt.asynchronous:
                    try:
                        self.finalize(attempt, "cancelled")
                    except BaseException as caught:
                        error = caught
        finally:
            self._sync.close()
        if error is not None:
            raise error

    async def aclose(self):
        with self._lock:
            if any(not item.finalized and item.raw is None for item in self._pending):
                raise InferenceRefused("context")
            self._closed = True
        error = None
        try:
            for attempt in list(self._pending):
                if attempt.asynchronous:
                    try:
                        await self.afinalize(attempt, "cancelled")
                    except BaseException as caught:
                        error = caught
        finally:
            await self._async.close()
        if error is not None:
            raise error


class _RawResponse:
    def __init__(self, owner, attempt, streaming):
        self.owner, self.attempt, self.streaming = owner, attempt, streaming
        self.parsed = None

    def __getattr__(self, name):
        return getattr(self.attempt.raw, name)

    def parse(self, **kwargs):
        if self.parsed is not None:
            return self.parsed
        try:
            value = self.attempt.raw.parse(**kwargs)
            if self.streaming:
                self.attempt.stream = value
                self.parsed = _Stream(self.owner, self.attempt, value)
            else:
                self.attempt.observe_message(value)
                self.owner.finalize(self.attempt, "completed")
                self.parsed = value
            return self.parsed
        except BaseException as error:
            self.owner.finalize(self.attempt, "failed")
            _provider_error(error)

    def close(self):
        self.owner.finalize(self.attempt, "cancelled")


class _AsyncRawResponse(_RawResponse):
    async def parse(self, **kwargs):
        if self.parsed is not None:
            return self.parsed
        try:
            value = await self.attempt.raw.parse(**kwargs)
            if self.streaming:
                self.attempt.stream = value
                self.parsed = _AsyncStream(self.owner, self.attempt, value)
            else:
                self.attempt.observe_message(value)
                await self.owner.afinalize(self.attempt, "completed")
                self.parsed = value
            return self.parsed
        except BaseException as error:
            await self.owner.afinalize(self.attempt, "failed")
            _provider_error(error)

    async def close(self):
        await self.owner.afinalize(self.attempt, "cancelled")


class _Stream:
    def __init__(self, owner, attempt, stream):
        self.owner, self.attempt = owner, attempt
        self.iterator = iter(stream)

    def __iter__(self):
        return self

    def __next__(self):
        if self.attempt.finalized:
            raise StopIteration
        try:
            event = next(self.iterator)
            self.attempt.observe_event(event)
            return event
        except StopIteration:
            self.owner.finalize(
                self.attempt, "completed" if self.attempt.stopped else "failed"
            )
            if not self.attempt.stopped:
                raise InferenceRefused("provider") from None
            raise
        except BaseException as error:
            self.owner.finalize(self.attempt, "failed")
            _provider_error(error)

    def close(self):
        self.owner.finalize(self.attempt, "cancelled")


class _AsyncStream(_Stream):
    def __init__(self, owner, attempt, stream):
        self.owner, self.attempt = owner, attempt
        self.iterator = stream.__aiter__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.attempt.finalized:
            raise StopAsyncIteration
        try:
            event = await anext(self.iterator)
            self.attempt.observe_event(event)
            return event
        except StopAsyncIteration:
            await self.owner.afinalize(
                self.attempt, "completed" if self.attempt.stopped else "failed"
            )
            if not self.attempt.stopped:
                raise InferenceRefused("provider") from None
            raise
        except BaseException as error:
            kind = (
                "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
            )
            await self.owner.afinalize(self.attempt, kind)
            _provider_error(error)

    async def close(self):
        await self.owner.afinalize(self.attempt, "cancelled")


class ControlledChatAnthropic(ChatAnthropic):
    """Real native chat model with explicit stream-finally ownership."""

    _request_owner = PrivateAttr(default=None)

    def attach_request_owner(self, owner):
        if (
            not isinstance(owner, ControlledAnthropicClient)
            or self.model != owner.context.model_id
            or self.cache is not False
            or self.max_retries != 0
        ):
            raise InferenceRefused("context")
        self._request_owner = owner
        self.__dict__["_client"] = owner.sync_view
        self.__dict__["_async_client"] = owner.async_view
        return self

    def _create(self, payload):
        if self._request_owner is None:
            raise InferenceRefused("context")
        return super()._create(payload)

    async def _acreate(self, payload):
        if self._request_owner is None:
            raise InferenceRefused("context")
        return await super()._acreate(payload)

    def _stream(self, *args, **kwargs):
        with self._request_owner.stream_scope():
            yield from super()._stream(*args, **kwargs)

    async def _astream(self, *args, **kwargs):
        stream = super()._astream(*args, **kwargs)
        try:
            async for chunk in stream:
                yield chunk
        finally:
            await stream.aclose()

    async def astream(self, *args, **kwargs):
        async with self._request_owner.astream_scope():
            stream = super().astream(*args, **kwargs)
            try:
                async for chunk in stream:
                    yield chunk
            finally:
                await stream.aclose()

    async def ainvoke(self, *args, **kwargs):
        async with self._request_owner.astream_scope():
            # The base model uses gather: cancelling it can hide an accounting
            # refusal from a child. Drain the protected operation before reporting
            # cancellation, allowing finalizer refusal to take precedence.
            task = asyncio.create_task(super().ainvoke(*args, **kwargs))
            cancelled = False
            while True:
                try:
                    result = await asyncio.shield(task)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    if task.done():
                        result = task.result()
                        break
            if cancelled:
                raise asyncio.CancelledError
            return result
