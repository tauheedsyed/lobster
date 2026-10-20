"""Controlled clients exercise real native serialization and synthetic HTTP."""

import asyncio
import json
import pickle  # nosec B403 # dumps only tests serialization refusal; no untrusted loads.
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from lobster.config.inference_context import (
    InferenceContext,
    NativeRequestView,
    RequestDescriptor,
    UsageOutcome,
)
from lobster.config.providers.controlled_bedrock_client import (
    ControlledBedrockClient,
    ControlledChatBedrockConverse,
)
from lobster.config.providers.gateway_bedrock_stub import StubBedrockControlClient
from lobster.core.exceptions import InferenceRefused
from tests.unit.config.test_native_request_hooks import (
    BEDROCK_MODEL,
    HookProbe,
    bedrock_model,
)

pytestmark = pytest.mark.no_auto_config


class Authorizer:
    def __init__(self):
        self.descriptors = []
        self.outcomes = []
        self.refusal = None
        self.finalization_refusal = None

    def before_request(self, descriptor):
        self.descriptors.append(descriptor)
        if self.refusal is not None:
            raise self.refusal
        return object()

    def after_request(self, permit, outcome):
        self.outcomes.append((permit, outcome))
        if self.finalization_refusal is not None:
            raise self.finalization_refusal


@pytest.fixture
def controlled():
    probe = HookProbe()
    model = bedrock_model(probe)
    context = InferenceContext(
        "bedrock", BEDROCK_MODEL, "synthetic-binding", "version-a", 200000, 137
    )
    authorizer = Authorizer()
    views = []

    def validate(view):
        views.append(view)
        assert view.parameters["modelId"] == context.model_id

    def describe(view):
        return RequestDescriptor(
            context.provider,
            context.model_id,
            "test-component",
            context.binding_id,
            context.binding_version,
            str(uuid4()),
            str(uuid4()),
            context.max_output_tokens,
            1024,
            time.time() + 60,
        )

    owner = ControlledBedrockClient(
        client=model.client,
        context=context,
        component="test-component",
        authorizer=authorizer,
        validate_request=validate,
        describe_request=describe,
    )
    model = ControlledChatBedrockConverse(
        model_id=BEDROCK_MODEL,
        max_tokens=context.max_output_tokens,
        client=owner,
        bedrock_client=StubBedrockControlClient(),
        cache=False,
    )
    yield model, owner, probe, authorizer, views
    owner.close()


@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_native_admission_and_exactly_once_finalization(controlled, mode):
    model, owner, probe, authorizer, views = controlled
    if mode == "invoke":
        model.invoke("hello")
    elif mode == "ainvoke":
        await model.ainvoke("hello")
    elif mode == "stream":
        assert list(model.stream("hello"))
    else:
        assert [chunk async for chunk in model.astream("hello")]
    assert (
        len(views)
        == len(authorizer.descriptors)
        == len(authorizer.outcomes)
        == len(probe.sent)
        == 1
    )
    outcome = authorizer.outcomes[0][1]
    assert outcome.outcome == "completed"
    assert outcome.completeness == "complete"
    assert outcome.request_issued
    assert dict(outcome.raw_usage)["inputTokens"] == 4
    assert dict(outcome.raw_usage)["outputTokens"] == 1
    assert outcome.actual_model_id == BEDROCK_MODEL
    if "stream" in mode:
        assert probe.closed == 1
    owner.close()
    assert len(authorizer.outcomes) == 1


@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_refusal_before_actual_send_has_no_provider_or_receipt(controlled, mode):
    model, _, probe, authorizer, _ = controlled
    refusal = InferenceRefused("budget")
    authorizer.refusal = refusal
    with pytest.raises(InferenceRefused) as caught:
        if mode == "invoke":
            model.invoke("hello")
        elif mode == "ainvoke":
            await model.ainvoke("hello")
        elif mode == "stream":
            list(model.stream("hello"))
        else:
            _ = [chunk async for chunk in model.astream("hello")]
    assert caught.value is refusal
    assert len(authorizer.descriptors) == 1
    assert not probe.sent
    assert not authorizer.outcomes


@pytest.mark.parametrize("mode", ["stream", "astream"])
@pytest.mark.asyncio
async def test_early_stream_close_finalizes_unknown_usage_and_closes_provider(
    controlled, mode
):
    model, _, probe, authorizer, _ = controlled
    stream = getattr(model, mode)("hello")
    if mode == "stream":
        next(stream)
        stream.close()
    else:
        await anext(stream)
        await stream.aclose()
    assert len(authorizer.outcomes) == 1
    assert authorizer.outcomes[0][1].outcome == "cancelled"
    assert authorizer.outcomes[0][1].completeness == "unknown"
    assert probe.closed == 1


@pytest.mark.parametrize("mode", ["invoke", "stream"])
def test_accounting_failure_overrides_success_and_always_closes(controlled, mode):
    model, _, probe, authorizer, _ = controlled
    refusal = InferenceRefused("accounting")
    authorizer.finalization_refusal = refusal
    with pytest.raises(InferenceRefused) as caught:
        if mode == "invoke":
            model.invoke("hello")
        else:
            list(model.stream("hello"))
    assert caught.value is refusal
    assert len(authorizer.outcomes) == 1
    if mode == "stream":
        assert probe.closed == 1


def test_sdk_500_is_issued_unknown_and_not_retried(controlled):
    model, _, probe, authorizer, _ = controlled
    probe.status = 500
    with pytest.raises(InferenceRefused) as caught:
        model.invoke("hello")
    assert caught.value.category == "provider"
    assert "Synthetic failure" not in str(caught.value)
    assert (
        len(authorizer.descriptors) == len(authorizer.outcomes) == len(probe.sent) == 1
    )
    outcome = authorizer.outcomes[0][1]
    assert outcome.outcome == "failed"
    assert outcome.request_issued
    assert outcome.completeness == "unknown"


def test_real_native_bound_identity_mismatch_refuses_before_admit(controlled):
    model, _, probe, authorizer, _ = controlled
    with pytest.raises(InferenceRefused):
        model.invoke("hello", modelId="different-model")
    assert not authorizer.descriptors
    assert not probe.sent


def test_real_output_override_cannot_exceed_frozen_bound(controlled):
    model, _, probe, authorizer, _ = controlled
    with pytest.raises(InferenceRefused):
        model.invoke("hello", max_tokens=9999)
    assert not authorizer.descriptors
    assert not probe.sent


@pytest.mark.parametrize(
    "operation",
    ["count_tokens", "invoke_model", "get_foundation_model", "get_inference_profile"],
)
def test_control_apis_refuse_without_transport(controlled, operation):
    _, owner, probe, authorizer, _ = controlled
    with pytest.raises(InferenceRefused):
        getattr(owner, operation)(modelId=BEDROCK_MODEL)
    assert not probe.sent
    assert not authorizer.descriptors


def test_cached_native_model_in_raw_threads_is_still_authorized(controlled):
    model, _, probe, authorizer, _ = controlled
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(model.invoke, ["first", "second"]))
    assert (
        len(authorizer.outcomes) == len(authorizer.descriptors) == len(probe.sent) == 2
    )
    assert len({item.attempt_id for item in authorizer.descriptors}) == 2


def test_owned_client_close_finalizes_abandoned_stream(controlled):
    _, owner, probe, authorizer, _ = controlled
    response = owner.converse_stream(
        modelId=BEDROCK_MODEL,
        messages=[{"role": "user", "content": [{"text": "hello"}]}],
        inferenceConfig={"maxTokens": 137},
    )
    next(response["stream"])
    owner.close()
    assert probe.closed == 1
    assert len(authorizer.outcomes) == 1
    with pytest.raises(InferenceRefused):
        owner.converse(modelId=BEDROCK_MODEL)


def test_native_view_is_readonly_ephemeral_and_redacted():
    source = {"messages": [{"text": "synthetic-private-prompt"}]}
    view = NativeRequestView("Converse", "https://example.invalid/converse", source)
    source["messages"][0]["text"] = "changed"
    assert view.parameters["messages"][0]["text"] == "synthetic-private-prompt"
    with pytest.raises(TypeError):
        view.parameters["messages"][0]["text"] = "changed"
    assert "private-prompt" not in repr(view)
    with pytest.raises(TypeError):
        pickle.dumps(view)


@pytest.mark.parametrize(
    "raw", [(("secret", "fake"),), (("inputTokens", -1),), (("inputTokens", True),)]
)
def test_usage_rejects_untrusted_nonusage_fields(raw):
    with pytest.raises(InferenceRefused):
        UsageOutcome(raw_usage=raw)


def test_unknown_native_ttl_keeps_counts_but_requires_reconciliation(controlled):
    from botocore.awsrequest import AWSResponse

    from tests.unit.config.test_native_request_hooks import FakeRawResponse

    model, owner, probe, authorizer, _ = controlled
    original = owner._client._endpoint.http_session.send

    def unexpected(request):
        response = original(request)
        payload = json.loads(response.raw.payload)
        payload["usage"]["cacheDetails"] = [{"ttl": "unexpected", "inputTokens": 2}]
        return AWSResponse(
            request.url,
            200,
            response.headers,
            FakeRawResponse(json.dumps(payload).encode(), probe),
        )

    owner._client._endpoint.http_session.send = unexpected
    model.invoke("hello")
    assert len(probe.sent) == len(authorizer.outcomes) == 1
    outcome = authorizer.outcomes[0][1]
    assert outcome.completeness == "unknown"
    assert dict(outcome.raw_usage)["inputTokens"] == 4
    assert "cache_ttl" not in dict(outcome.raw_usage)


def test_midstream_error_finalizes_once_and_closes_provider(controlled):
    _, owner, probe, authorizer, _ = controlled
    response = owner.converse_stream(
        modelId=BEDROCK_MODEL,
        messages=[{"role": "user", "content": [{"text": "hello"}]}],
        inferenceConfig={"maxTokens": 137},
    )
    stream = response["stream"]

    def fail():
        yield {
            "contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "partial"}}
        }
        raise OSError("synthetic transport failure")

    stream._iterator = fail()
    next(stream)
    with pytest.raises(InferenceRefused):
        next(stream)
    stream.close()
    assert probe.closed == 1
    assert len(authorizer.outcomes) == 1
    assert authorizer.outcomes[0][1].outcome == "failed"
    assert authorizer.outcomes[0][1].completeness == "unknown"


def test_raw_client_alias_is_still_gated_outside_owned_request(controlled):
    _, owner, probe, authorizer, _ = controlled
    raw = owner._client
    with pytest.raises(InferenceRefused):
        raw.converse(
            modelId=BEDROCK_MODEL,
            messages=[{"role": "user", "content": [{"text": "hello"}]}],
            inferenceConfig={"maxTokens": 137},
        )
    assert not probe.sent
    assert not authorizer.descriptors


def test_descriptor_reused_attempt_id_is_refused_before_second_admission(controlled):
    model, owner, probe, authorizer, _ = controlled
    original = owner._describe_request
    descriptor = None

    def repeat(view):
        nonlocal descriptor
        if descriptor is None:
            descriptor = original(view)
        return descriptor

    owner._describe_request = repeat
    model.invoke("first")
    with pytest.raises(InferenceRefused):
        model.invoke("second")
    assert (
        len(authorizer.descriptors) == len(authorizer.outcomes) == len(probe.sent) == 1
    )


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), 0.0, 253402300800.0])
def test_descriptor_deadline_is_finite_bounded(deadline):
    with pytest.raises(InferenceRefused):
        RequestDescriptor(
            "bedrock",
            BEDROCK_MODEL,
            "component",
            "binding",
            "version",
            str(uuid4()),
            str(uuid4()),
            137,
            1024,
            deadline,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("recording_failure", [False, True])
async def test_async_executor_finalizer_drains_before_cancel_is_reported(
    controlled, recording_failure
):
    model, _, _, authorizer, _ = controlled
    started, release = threading.Event(), threading.Event()
    refusal = InferenceRefused("accounting")

    def finish(permit, outcome):
        authorizer.outcomes.append((permit, outcome))
        started.set()
        assert release.wait(2)
        if recording_failure:
            raise refusal

    authorizer.after_request = finish
    task = asyncio.create_task(model.ainvoke("hello"))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    if recording_failure:
        with pytest.raises(InferenceRefused) as caught:
            await task
        assert caught.value is refusal
    else:
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(authorizer.outcomes) == 1


@pytest.mark.asyncio
async def test_async_executor_request_drains_before_cancel_is_reported(controlled):
    model, owner, probe, authorizer, _ = controlled
    started, release = threading.Event(), threading.Event()
    original = owner._client._endpoint.http_session.send

    def send(request):
        started.set()
        assert release.wait(2)
        return original(request)

    owner._client._endpoint.http_session.send = send
    task = asyncio.create_task(model.ainvoke("hello"))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert not authorizer.outcomes
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(probe.sent) == len(authorizer.outcomes) == 1


@pytest.mark.asyncio
async def test_async_executor_stream_read_drains_and_closes_before_cancel(controlled):
    from botocore.awsrequest import AWSResponse

    from tests.unit.config.test_native_request_hooks import (
        FakeRawResponse,
        encode_event,
    )

    model, owner, probe, authorizer, _ = controlled
    started, release = threading.Event(), threading.Event()
    original = owner._client._endpoint.http_session.send

    class Body(FakeRawResponse):
        def stream(self, amt=None, decode_content=False):
            yield encode_event("messageStart", {"role": "assistant"})
            yield encode_event(
                "contentBlockDelta",
                {"contentBlockIndex": 0, "delta": {"text": "partial"}},
            )
            started.set()
            assert release.wait(2)
            yield encode_event("messageStop", {"stopReason": "end_turn"})
            yield encode_event(
                "metadata",
                {"usage": {"inputTokens": 4, "outputTokens": 1, "totalTokens": 5}},
            )

    def send(request):
        response = original(request)
        return AWSResponse(
            request.url, response.status_code, response.headers, Body(b"", probe)
        )

    owner._client._endpoint.http_session.send = send

    async def consume():
        return [chunk async for chunk in model.astream("hello")]

    task = asyncio.create_task(consume())
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert not authorizer.outcomes
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert probe.closed == 1
    assert len(authorizer.outcomes) == 1
    assert authorizer.outcomes[0][1].outcome == "cancelled"
