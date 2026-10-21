"""Real Messages/SSE SDK serialization with owned lifecycle, offline."""

import asyncio
import time
from uuid import uuid4

import pytest

from tests.unit.config._native_inference_helpers import (
    ANTHROPIC_MODEL,
    AsyncAuthorizer,
    HookProbe,
    anthropic_model,
    require_anthropic,
)

# This module contains only real Anthropic SDK/model cases.
require_anthropic()

from lobster.config.inference_context import InferenceContext, RequestDescriptor
from lobster.config.providers.controlled_anthropic_client import (
    ControlledAnthropicClient,
    ControlledChatAnthropic,
)
from lobster.core.exceptions import InferenceRefused

pytestmark = pytest.mark.no_auto_config


@pytest.fixture
def controlled():
    probe = HookProbe()
    native = anthropic_model(probe)
    context = InferenceContext(
        "anthropic", ANTHROPIC_MODEL, "synthetic-binding", "version-a", 200000, 137
    )
    authorizer = AsyncAuthorizer()
    views = []

    def validate(view):
        views.append(view)
        assert view.parameters["model"] == context.model_id

    def describe(view):
        return RequestDescriptor(
            context.provider,
            context.model_id,
            "test-component",
            context.binding_id,
            context.binding_version,
            str(uuid4()),
            str(uuid4()),
            137,
            1024,
            time.time() + 60,
        )

    owner = ControlledAnthropicClient(
        context=context,
        component="test-component",
        authorizer=authorizer,
        validate_request=validate,
        describe_request=describe,
        api_key="fake-anthropic-key",
        base_url="https://api.anthropic.com",
        http_client=native._client._client,
        http_async_client=native._async_client._client,
    )
    model = ControlledChatAnthropic(
        model=ANTHROPIC_MODEL,
        api_key="fake-anthropic-key",
        base_url="https://api.anthropic.com",
        max_tokens=137,
        cache=False,
        max_retries=0,
    ).attach_request_owner(owner)
    yield model, owner, probe, authorizer, views
    owner.close()
    asyncio.run(owner.aclose())


@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_complete_request_and_stream_finalizes_once(controlled, mode):
    model, _, probe, authorizer, views = controlled
    if mode == "invoke":
        model.invoke("hello")
    elif mode == "ainvoke":
        await model.ainvoke("hello")
    elif mode == "stream":
        list(model.stream("hello"))
    else:
        _ = [chunk async for chunk in model.astream("hello")]
    assert (
        len(probe.sent)
        == len(authorizer.descriptors)
        == len(authorizer.outcomes)
        == len(views)
        == 1
    )
    result = authorizer.outcomes[0][1]
    assert result.outcome == "completed"
    assert result.completeness == "complete"
    assert dict(result.raw_usage)["input_tokens"] == 4
    assert dict(result.raw_usage)["output_tokens"] == 1
    assert result.actual_model_id == ANTHROPIC_MODEL


@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_refusal_before_send_never_creates_provider_request(controlled, mode):
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
    assert not probe.sent
    assert not authorizer.outcomes


@pytest.mark.parametrize("mode", ["stream", "astream"])
@pytest.mark.asyncio
async def test_early_close_pinned_model_missing_finally_is_corrected(controlled, mode):
    model, _, probe, authorizer, _ = controlled
    stream = getattr(model, mode)("hello")
    if mode == "stream":
        next(stream)
        stream.close()
    else:
        await anext(stream)
        await stream.aclose()
    assert len(authorizer.outcomes) == len(probe.sent) == 1
    assert authorizer.outcomes[0][1].outcome == "cancelled"
    assert authorizer.outcomes[0][1].completeness == "partial"
    assert probe.sent[0] is not None


@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_finalizer_failure_cannot_be_hidden_by_success(controlled, mode):
    model, _, _, authorizer, _ = controlled
    refusal = InferenceRefused("accounting")
    authorizer.finalization_refusal = refusal
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
    assert len(authorizer.outcomes) == 1


def test_real_controlled_client_count_tokens_is_refused(controlled):
    _, owner, probe, authorizer, _ = controlled
    for client in (owner.sync_view, owner.async_view):
        with pytest.raises(InferenceRefused):
            client.messages.count_tokens(model=ANTHROPIC_MODEL)
    assert not probe.sent
    assert not authorizer.descriptors


def test_sdk500_refuses_without_retry_or_provider_message(controlled):
    model, _, probe, authorizer, _ = controlled
    probe.status = 500
    with pytest.raises(InferenceRefused) as caught:
        model.invoke("hello")
    assert caught.value.category == "provider"
    assert "Synthetic failure" not in str(caught.value)
    assert len(probe.sent) == len(authorizer.outcomes) == 1
    assert authorizer.outcomes[0][1].request_issued
    assert authorizer.outcomes[0][1].completeness == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("recording_failure", [False, True])
async def test_cancel_waits_for_durable_async_finalizer(controlled, recording_failure):
    model, _, _, authorizer, _ = controlled
    started, release = asyncio.Event(), asyncio.Event()
    refusal = InferenceRefused("accounting")

    async def finalize(permit, outcome):
        authorizer.outcomes.append((permit, outcome))
        started.set()
        await release.wait()
        if recording_failure:
            raise refusal

    authorizer.aafter_request = finalize
    task = asyncio.create_task(model.ainvoke("hello"))
    await asyncio.wait_for(started.wait(), 1)
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
async def test_cancel_during_native_sse_read_closes_and_finalizes(controlled):
    import httpx2

    model, owner, _, authorizer, _ = controlled
    reading, closed = asyncio.Event(), asyncio.Event()

    class Body(httpx2.AsyncByteStream):
        async def __aiter__(self):
            yield b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":4,"output_tokens":0}}}\n\n'
            reading.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    async def send(request):
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Body()
        )

    owner._async._client._transport = httpx2.MockTransport(send)

    async def consume():
        return [chunk async for chunk in model.astream("hello")]

    task = asyncio.create_task(consume())
    await asyncio.wait_for(reading.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert len(authorizer.outcomes) == 1
    assert authorizer.outcomes[0][1].outcome == "cancelled"
    assert authorizer.outcomes[0][1].request_issued


@pytest.mark.parametrize("mode", ["invoke", "stream"])
def test_native_beta_header_is_validated_and_finalized(controlled, mode):
    model, _, _, authorizer, views = controlled
    if mode == "invoke":
        model.invoke("hello", betas=["synthetic-beta"])
    else:
        list(model.stream("hello", betas=["synthetic-beta"]))
    assert "synthetic-beta" in views[0].parameters["betas"]
    assert len(authorizer.outcomes) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_native_async_beta_messages(controlled, mode):
    model, _, _, authorizer, views = controlled
    if mode == "ainvoke":
        await model.ainvoke("hello", betas=["synthetic-beta"])
    else:
        _ = [chunk async for chunk in model.astream("hello", betas=["synthetic-beta"])]
    assert "synthetic-beta" in views[0].parameters["betas"]
    assert len(authorizer.outcomes) == 1


def test_header_credential_override_refuses_before_issue(controlled):
    model, _, probe, authorizer, _ = controlled
    with pytest.raises(InferenceRefused):
        model.invoke("hello", extra_headers={"x-api-key": "different-synthetic-key"})
    assert not probe.sent
    assert not authorizer.descriptors


def test_model_without_owned_clients_never_falls_back():
    model = ControlledChatAnthropic(
        model=ANTHROPIC_MODEL,
        api_key="fake-anthropic-key",
        max_tokens=137,
        cache=False,
        max_retries=0,
    )
    with pytest.raises(InferenceRefused):
        model.invoke("hello")


def test_native_output_override_refuses_before_issue(controlled):
    model, _, probe, authorizer, _ = controlled
    with pytest.raises(InferenceRefused):
        model.invoke("hello", max_tokens=9999)
    assert not probe.sent
    assert not authorizer.descriptors


def test_disposal_closes_all_abandoned_streams_even_when_first_receipt_fails(
    controlled,
):
    _, owner, _, authorizer, _ = controlled
    streams = []
    for _ in range(2):
        raw = owner.sync_view.messages.with_raw_response.create(
            model=ANTHROPIC_MODEL,
            max_tokens=137,
            stream=True,
            messages=[{"role": "user", "content": "hello"}],
        )
        stream = raw.parse()
        next(stream)
        streams.append(stream)
    authorizer.finalization_refusal = InferenceRefused("accounting")
    with pytest.raises(InferenceRefused):
        owner.close()
    assert len(authorizer.outcomes) == 2
    assert not owner._pending
    assert all(stream.attempt.stream.response.is_closed for stream in streams)
