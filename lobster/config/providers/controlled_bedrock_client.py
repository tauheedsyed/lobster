"""Controlled native Converse requests with one host finalizer per attempt.

The host supplies an explicitly constructed SDK client, request validator and
nonsecret descriptor builder. No credentials, endpoints or policy are resolved
from the environment. Other SDK operations are deliberately unavailable.
"""

import asyncio
import json
import time
from contextvars import ContextVar
from threading import RLock, local
from urllib.parse import quote

from langchain_aws import ChatBedrockConverse
from langchain_core.runnables.config import run_in_executor
from pydantic import PrivateAttr

from lobster.config.inference_context import (
    InferenceContext,
    NativeRequestView,
    RequestDescriptor,
    UsageOutcome,
)
from lobster.core.exceptions import InferenceRefused, reraise_inference_refusal

_USAGE_FIELDS = (
    "inputTokens",
    "outputTokens",
    "totalTokens",
    "cacheReadInputTokens",
    "cacheWriteInputTokens",
)
_FINISH_REASONS = frozenset(
    {
        "end_turn",
        "tool_use",
        "max_tokens",
        "stop_sequence",
        "guardrail_intervened",
        "content_filtered",
        "model_context_window_exceeded",
    }
)


class _Attempt:
    def __init__(self):
        self.permit = None
        self.admitted = False
        self.issued = False
        self.finalized = False
        self.usage = ()
        self.usage_valid = False
        self.finish = None
        self.stopped = False
        self.lock = RLock()

    def observe(self, usage, finish=None):
        if type(finish) is str and finish in _FINISH_REASONS:
            self.finish = finish
        if not isinstance(usage, dict):
            return
        pairs = []
        for name in _USAGE_FIELDS:
            if name in usage:
                count = usage[name]
                if type(count) is not int or count < 0:
                    self.usage_valid = False
                    return
                pairs.append((name, count))
        # Retain cache evidence, but unknown/malformed TTLs never get a guessed
        # price or complete-usage claim. The host reconciles unknown outcomes.
        details_valid = True
        if "cacheDetails" in usage:
            details = usage["cacheDetails"]
            details_valid = isinstance(details, list)
            counts = {"5m": 0, "1h": 0}
            if details_valid:
                for detail in details:
                    if not isinstance(detail, dict):
                        details_valid = False
                        break
                    ttl, count = detail.get("ttl"), detail.get("inputTokens")
                    if (
                        type(ttl) is not str
                        or ttl not in counts
                        or type(count) is not int
                        or count < 0
                    ):
                        details_valid = False
                        break
                    counts[ttl] += count
                if details_valid:
                    pairs.extend(
                        (f"ephemeral_{ttl}_input_tokens", count)
                        for ttl, count in counts.items()
                    )
        self.usage = tuple(pairs)
        self.usage_valid = (
            details_valid and "inputTokens" in usage and "outputTokens" in usage
        )


class ControlledBedrockClient:
    """Expose only Converse; admission occurs at actual native before-send.

    One wrapper owns one SDK client and its event hooks. SDK retries must be
    disabled before injection. Sharing an inner client between wrappers refuses.
    """

    def __init__(
        self,
        *,
        client,
        context: InferenceContext,
        component: str,
        authorizer,
        validate_request,
        describe_request,
    ):
        if (
            not isinstance(context, InferenceContext)
            or context.provider != "bedrock"
            or type(component) is not str
            or not component
        ):
            raise InferenceRefused("context")
        if (
            not callable(validate_request)
            or not callable(describe_request)
            or not callable(getattr(authorizer, "before_request", None))
            or not callable(getattr(authorizer, "after_request", None))
        ):
            raise InferenceRefused("context")
        self._client = client
        self._context = context
        self._component = component
        self._authorizer = authorizer
        self._validate_request = validate_request
        self._describe_request = describe_request
        self._local = local()
        self._lock = RLock()
        self._closed = False
        self._streams = set()
        self._attempt_ids = set()
        self._active = 0
        try:
            retries = client.meta.config.retries
            endpoint = client.meta.endpoint_url
        except Exception:
            raise InferenceRefused("context") from None
        if retries.get("total_max_attempts") != 1 or not endpoint.startswith(
            "https://"
        ):
            raise InferenceRefused("context")
        self._endpoint = endpoint.rstrip("/")
        self._hook_id = f"controlled-inference-{id(self)}"
        client.meta.events.register(
            "before-send.bedrock-runtime.*", self._before_send, unique_id=self._hook_id
        )

    def __repr__(self):
        return "ControlledBedrockClient(<owned>)"

    def __getattr__(self, name):
        # Includes CountTokens, InvokeModel and control/profile requests. Never
        # leak a raw client method that could bypass its request ownership.
        raise InferenceRefused("context")

    @property
    def meta(self):
        return self._client.meta

    def _before_send(self, request, **kwargs):
        attempt = getattr(self._local, "attempt", None)
        operation = getattr(self._local, "operation", None)
        if attempt is None or attempt.admitted or self._closed:
            raise InferenceRefused("context")
        suffix = "converse-stream" if operation == "ConverseStream" else "converse"
        expected = (
            f"{self._endpoint}/model/{quote(self._context.model_id, safe='')}/{suffix}"
        )
        if request.url != expected or request.method != "POST":
            raise InferenceRefused("context")
        try:
            parameters = {**json.loads(request.body), "modelId": self._context.model_id}
        except (ValueError, TypeError):
            raise InferenceRefused("context") from None
        view = NativeRequestView(operation, request.url, parameters)
        self._validate_request(view)
        descriptor = self._describe_request(view)
        if not isinstance(descriptor, RequestDescriptor):
            raise InferenceRefused("context")
        expected_identity = (
            self._context.provider,
            self._context.model_id,
            self._component,
            self._context.binding_id,
            self._context.binding_version,
        )
        actual_identity = (
            descriptor.provider,
            descriptor.model_id,
            descriptor.component,
            descriptor.binding_id,
            descriptor.binding_version,
        )
        bound = parameters.get("inferenceConfig", {}).get("maxTokens")
        if (
            actual_identity != expected_identity
            or descriptor.max_output_tokens != self._context.max_output_tokens
            or type(bound) is not int
            or not 0 < bound <= self._context.max_output_tokens
            or descriptor.deadline_at <= time.time()
        ):
            raise InferenceRefused("context")
        with self._lock:
            if descriptor.attempt_id in self._attempt_ids:
                raise InferenceRefused("context")
            self._attempt_ids.add(descriptor.attempt_id)
        attempt.permit = self._authorizer.before_request(descriptor)
        attempt.admitted = True
        if descriptor.deadline_at <= time.time():
            raise InferenceRefused("context")
        # Transport may fail after this hook. Conservatively treat the request as
        # possibly issued; no SDK receipt can prove it was not accepted remotely.
        attempt.issued = True

    def _finalize(self, attempt, outcome):
        with attempt.lock:
            if attempt.finalized:
                return
            attempt.finalized = True
        if not attempt.admitted:
            return
        quality = "unknown"
        if attempt.usage_valid:
            quality = "complete" if outcome == "completed" else "partial"
        self._authorizer.after_request(
            attempt.permit,
            UsageOutcome(
                raw_usage=attempt.usage,
                finish_reason=attempt.finish,
                completeness=quality,
                request_issued=attempt.issued,
                outcome=outcome,
                actual_model_id=self._context.model_id,
            ),
        )

    def _call(self, operation, parameters):
        with self._lock:
            if self._closed:
                raise InferenceRefused("context")
            self._active += 1
        attempt = _Attempt()
        if getattr(self._local, "attempt", None) is not None:
            with self._lock:
                self._active -= 1
            raise InferenceRefused("context")
        self._local.attempt = attempt
        self._local.operation = operation
        stream_transferred = False
        outcome = "failed"
        try:
            method = (
                self._client.converse_stream
                if operation == "ConverseStream"
                else self._client.converse
            )
            response = method(**parameters)
            if not attempt.admitted:
                raise InferenceRefused("context")
            if operation == "ConverseStream":
                stream = _ControlledStream(self, attempt, response["stream"])
                with self._lock:
                    self._streams.add(stream)
                response = {**response, "stream": stream}
                stream_transferred = True
            else:
                attempt.observe(response.get("usage"), response.get("stopReason"))
                outcome = "completed"
            return response
        except BaseException as error:
            if isinstance(error, InferenceRefused):
                outcome = "refused"
            if isinstance(error, Exception):
                reraise_inference_refusal(error)
                raise InferenceRefused("provider") from None
            raise
        finally:
            self._local.attempt = None
            self._local.operation = None
            with self._lock:
                self._active -= 1
            if not stream_transferred:
                self._finalize(attempt, outcome)

    def converse(self, **parameters):
        return self._call("Converse", parameters)

    def converse_stream(self, **parameters):
        return self._call("ConverseStream", parameters)

    def close(self):
        with self._lock:
            if self._active:
                raise InferenceRefused("context")
            if self._closed:
                return
            self._closed = True
            streams = list(self._streams)
        error = None
        for stream in streams:
            try:
                stream.close()
            except BaseException as caught:
                error = caught
        try:
            self._client.close()
        finally:
            # Leave the deny hook installed on the now-closed SDK client: retained
            # raw aliases must not acquire an unguarded request path on disposal.
            if error is not None:
                raise error


class _ControlledStream:
    def __init__(self, owner, attempt, stream):
        self._owner = owner
        self._attempt = attempt
        self._stream = stream
        self._iterator = iter(stream)
        self._closed = False
        self._close_lock = RLock()

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            event = next(self._iterator)
            if any(name.endswith("Exception") for name in event):
                raise InferenceRefused("provider")
            if "messageStop" in event:
                self._attempt.stopped = True
                self._attempt.observe(None, event["messageStop"].get("stopReason"))
            if "metadata" in event:
                self._attempt.observe(event["metadata"].get("usage"))
            return event
        except StopIteration:
            self._finish("completed" if self._attempt.stopped else "failed")
            if not self._attempt.stopped:
                raise InferenceRefused("provider") from None
            raise
        except BaseException as error:
            self._finish("failed")
            if isinstance(error, Exception):
                reraise_inference_refusal(error)
                raise InferenceRefused("provider") from None
            raise

    def _finish(self, outcome):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        with self._owner._lock:
            self._owner._streams.discard(self)
        try:
            self._stream.close()
        finally:
            self._owner._finalize(self._attempt, outcome)

    def close(self):
        self._finish("cancelled")


async def _drain_operation(awaitable):
    """Wait for a protected provider/finalizer operation before reporting cancel."""
    task = asyncio.ensure_future(awaitable)
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


class ControlledChatBedrockConverse(ChatBedrockConverse):
    """Real native model whose async executor work drains before cancellation."""

    _async_stream_scope = PrivateAttr(
        default_factory=lambda: ContextVar("native-bedrock-stream", default=None)
    )

    def _check_owner(self):
        if (
            not isinstance(self.client, ControlledBedrockClient)
            or self.client._context.model_id != self.model_id
            or self.cache is not False
        ):
            raise InferenceRefused("context")

    def _generate(self, *args, **kwargs):
        self._check_owner()
        return super()._generate(*args, **kwargs)

    def _stream(self, *args, **kwargs):
        self._check_owner()
        yield from super()._stream(*args, **kwargs)

    async def agenerate(self, *args, **kwargs):
        self._check_owner()
        return await _drain_operation(super().agenerate(*args, **kwargs))

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        # The base async implementation never closes its sync iterator and can
        # leave next() running in an executor. Close only after that work drains.
        iterator = self._stream(
            messages, stop, run_manager.get_sync() if run_manager else None, **kwargs
        )
        scope = self._async_stream_scope.get()
        if scope is not None:
            scope.append(iterator)
        done = object()
        try:
            while True:
                item = await _drain_operation(
                    run_in_executor(None, next, iterator, done)
                )
                if item is done:
                    break
                yield item
        finally:
            await _drain_operation(run_in_executor(None, iterator.close))

    async def astream(self, *args, **kwargs):
        self._check_owner()
        stream = super().astream(*args, **kwargs)
        iterators = []
        token = self._async_stream_scope.set(iterators)
        try:
            async for chunk in stream:
                yield chunk
        finally:
            try:
                await stream.aclose()
            finally:
                try:
                    self._async_stream_scope.reset(token)
                except ValueError:
                    self._async_stream_scope.set(None)
                for iterator in iterators:
                    await _drain_operation(run_in_executor(None, iterator.close))
