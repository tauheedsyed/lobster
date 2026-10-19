"""Native SDK injection seams with real model serialization and fake HTTP.

These are request-interception feasibility tests, not an authorization or
accounting implementation. A host must still supply its own bound factory.
"""

import json
import socket
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote
from zlib import crc32

import pytest

anthropic = pytest.importorskip(
    "anthropic",
    minversion="1.7.0",
    reason="Native middleware tests require the Anthropic provider extra with middleware support",
)
boto3 = pytest.importorskip(
    "boto3", reason="Native request tests require the Bedrock provider extra"
)
httpx2 = pytest.importorskip(
    "httpx2", reason="Native middleware tests require the SDK HTTP transport"
)
pytest.importorskip(
    "langchain_aws", reason="Native request tests require the Bedrock provider extra"
)
pytest.importorskip(
    "langchain_anthropic",
    reason="Native request tests require the Anthropic provider extra",
)

from botocore.awsrequest import AWSResponse
from botocore.config import Config

from lobster.config.providers.anthropic_provider import AnthropicProvider
from lobster.config.providers.bedrock_builder import build_bedrock_converse
from lobster.config.providers.gateway_bedrock_stub import StubBedrockControlClient
from lobster.core.exceptions import InferenceRefused

ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


class HookProbe:
    """Record native request-hook execution; never implements policy branches."""

    def __init__(self):
        self.hooks = []
        self.sent = []
        self.error = None
        self.status = 200
        self.closed = 0

    def before(self, request, **kwargs):
        self.hooks.append(request)
        if self.error is not None:
            raise self.error


class FakeRawResponse:
    def __init__(self, payload, probe):
        self.payload = payload
        self.probe = probe

    def stream(self, amt=None, decode_content=False):
        yield self.payload

    def close(self):
        self.probe.closed += 1


def encode_event(kind, payload):
    """Encode an AWS EventStream fixture, including the real wire CRCs."""
    headers = b""
    for name, value in (
        (":message-type", "event"),
        (":event-type", kind),
        (":content-type", "application/json"),
    ):
        name, value = name.encode(), value.encode()
        headers += (
            bytes([len(name)]) + name + b"\x07" + struct.pack(">H", len(value)) + value
        )
    body = json.dumps(payload).encode()
    prelude = struct.pack(">II", 16 + len(headers) + len(body), len(headers))
    message = prelude + struct.pack(">I", crc32(prelude)) + headers + body
    return message + struct.pack(">I", crc32(message))


def bedrock_model(probe):
    runtime = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="fake-access-id",
        aws_secret_access_key="fake-access-secret",
        config=Config(retries={"total_max_attempts": 1}),
    )
    runtime.meta.events.register("before-send.bedrock-runtime.*", probe.before)

    def send(request):
        probe.sent.append(request)
        if probe.status != 200:
            payload = json.dumps({"message": "Synthetic failure"}).encode()
            headers = {
                "content-type": "application/json",
                "x-amzn-errortype": "InternalServerException",
            }
        elif request.url.endswith("converse-stream"):
            events = [
                ("messageStart", {"role": "assistant"}),
                (
                    "contentBlockDelta",
                    {"contentBlockIndex": 0, "delta": {"text": "ok"}},
                ),
                ("contentBlockStop", {"contentBlockIndex": 0}),
                ("messageStop", {"stopReason": "end_turn"}),
                (
                    "metadata",
                    {
                        "usage": {
                            "inputTokens": 4,
                            "outputTokens": 1,
                            "totalTokens": 5,
                        },
                        "metrics": {"latencyMs": 1},
                    },
                ),
            ]
            payload = b"".join(encode_event(kind, value) for kind, value in events)
            headers = {"content-type": "application/vnd.amazon.eventstream"}
        else:
            payload = json.dumps(
                {
                    "output": {
                        "message": {"role": "assistant", "content": [{"text": "ok"}]}
                    },
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 4, "outputTokens": 1, "totalTokens": 5},
                    "metrics": {"latencyMs": 1},
                }
            ).encode()
            headers = {"content-type": "application/json"}
        return AWSResponse(
            request.url, probe.status, headers, FakeRawResponse(payload, probe)
        )

    runtime._endpoint.http_session.send = send
    return build_bedrock_converse(
        model_id=BEDROCK_MODEL,
        max_tokens=137,
        client=runtime,
        bedrock_client=StubBedrockControlClient(),
        cache=False,
    )


def anthropic_model(probe):
    message = {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": ANTHROPIC_MODEL,
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 4, "output_tokens": 1},
    }

    def send(request):
        probe.sent.append(request)
        if probe.status != 200:
            return httpx2.Response(
                probe.status,
                json={
                    "type": "error",
                    "error": {"type": "api_error", "message": "Synthetic failure"},
                },
            )
        if json.loads(request.content).get("stream"):
            start = {
                **message,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 4, "output_tokens": 0},
            }
            events = [
                {"type": "message_start", "message": start},
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "ok"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
                {"type": "message_stop"},
            ]
            stream = "".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in events
            )
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=stream.encode(),
            )
        return httpx2.Response(200, json=message)

    def before(request, next_handler):
        probe.before(request)
        return next_handler(request)

    async def abefore(request, next_handler):
        probe.before(request)
        return await next_handler(request)

    model = AnthropicProvider().create_chat_model(
        ANTHROPIC_MODEL,
        api_key="fake-anthropic-key",
        max_tokens=137,
        base_url="https://api.anthropic.com",
        max_retries=0,
        cache=False,
    )
    # The host supplies controlled native clients before exposing the model.
    # Do not put http_client in model_kwargs: it is not a Messages body field.
    model.__dict__["_client"] = anthropic.Client(
        api_key="fake-anthropic-key",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(send)),
        middleware=[before],
    )
    model.__dict__["_async_client"] = anthropic.AsyncClient(
        api_key="fake-anthropic-key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(send)),
        middleware=[abefore],
    )
    return model


@pytest.fixture(autouse=True)
def isolate_transport_and_configuration(monkeypatch):
    import dotenv

    def no_network(*args, **kwargs):
        raise AssertionError("Unexpected network access")

    def no_dotenv(*args, **kwargs):
        return False

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(dotenv, "load_dotenv", no_dotenv)
    if settings_module := sys.modules.get("lobster.config.settings"):
        monkeypatch.setattr(settings_module, "load_dotenv", no_dotenv)


@pytest.fixture(params=["bedrock", "anthropic"])
def native_model(request):
    probe = HookProbe()
    model = (
        bedrock_model(probe) if request.param == "bedrock" else anthropic_model(probe)
    )
    return model, probe, request.param


@pytest.mark.parametrize("operation", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_native_hook_covers_every_execution_mode(native_model, operation):
    model, probe, provider = native_model
    if operation == "invoke":
        model.invoke("hello")
    elif operation == "ainvoke":
        await model.ainvoke("hello")
    elif operation == "stream":
        list(model.stream("hello"))
    else:
        _ = [chunk async for chunk in model.astream("hello")]
    assert len(probe.hooks) == len(probe.sent) == 1
    request = probe.sent[0]
    payload = json.loads(request.body if provider == "bedrock" else request.content)
    if provider == "bedrock":
        assert BEDROCK_MODEL in unquote(request.url)
        assert payload["inferenceConfig"]["maxTokens"] == 137
        if "stream" in operation:
            assert probe.closed == 1
    else:
        assert payload["model"] == ANTHROPIC_MODEL
        assert payload["max_tokens"] == 137
        assert request.headers["x-api-key"] == "fake-anthropic-key"


@pytest.mark.parametrize("operation", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_native_hook_refuses_before_transport(native_model, operation):
    model, probe, _ = native_model
    refusal = InferenceRefused("budget")
    probe.error = refusal
    with pytest.raises(InferenceRefused) as caught:
        if operation == "invoke":
            model.invoke("hello")
        elif operation == "ainvoke":
            await model.ainvoke("hello")
        elif operation == "stream":
            list(model.stream("hello"))
        else:
            _ = [chunk async for chunk in model.astream("hello")]
    assert caught.value is refusal
    assert len(probe.hooks) == 1
    assert probe.sent == []


def test_native_sdk_does_not_retry_500(native_model):
    model, probe, _ = native_model
    probe.status = 500
    with pytest.raises(Exception):
        model.invoke("hello")
    assert len(probe.hooks) == len(probe.sent) == 1


def test_cached_helper_model_still_passes_native_hook_in_thread(native_model):
    from lobster.agents.data_expert.assistant import DataExpertAssistant

    model, probe, _ = native_model
    helper = DataExpertAssistant()
    helper._llm = model
    helper.detect_modality({}, "GSE1")  # Exercise and retain the same model instance.
    refusal = InferenceRefused("policy")
    probe.error = refusal
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(InferenceRefused) as caught:
            executor.submit(helper.extract_strategy_config, {}, "GSE1").result()
    assert caught.value is refusal
    assert len(probe.hooks) == 2
    assert len(probe.sent) == 1


def test_legacy_bulk_entry_preserves_native_refusal(native_model, tmp_path):
    from types import SimpleNamespace

    from lobster.services.data_access.geo_fallback_service import GEOFallbackService

    model, probe, _ = native_model
    refusal = InferenceRefused("policy")
    probe.error = refusal
    service = SimpleNamespace(
        data_manager=SimpleNamespace(metadata_store={}),
        cache_dir=tmp_path,
        console=None,
        geo_downloader=None,
        geo_parser=None,
        fetch_metadata_only=lambda _: model.invoke("hello"),
    )
    with pytest.raises(InferenceRefused) as caught:
        GEOFallbackService(service).download_bulk_dataset("GSE1")
    assert caught.value is refusal
    assert len(probe.hooks) == 1
    assert probe.sent == []


def test_legacy_sample_entry_does_not_retry_native_refusal(
    native_model, monkeypatch, tmp_path
):
    from types import SimpleNamespace

    import pandas as pd

    from lobster.agents.data_expert.assistant import DataExpertAssistant
    from lobster.services.data_access import geo_fallback_service
    from lobster.services.data_access.geo.concatenation import SampleConcatenator

    model, probe, _ = native_model
    refusal = InferenceRefused("budget")
    probe.error = refusal
    monkeypatch.setattr(DataExpertAssistant, "llm", property(lambda self: model))
    gsm = SimpleNamespace(metadata={}, table=pd.DataFrame([[1, 2]]))
    monkeypatch.setattr(
        geo_fallback_service, "pre_download_soft_file", lambda *a, **k: None
    )
    monkeypatch.setattr(geo_fallback_service.GEOparse, "get_GEO", lambda **k: gsm)
    service = SimpleNamespace(
        data_manager=SimpleNamespace(metadata_store={}, workspace_path=tmp_path),
        cache_dir=tmp_path,
        console=None,
        geo_downloader=None,
        geo_parser=None,
    )
    service._store_single_sample_as_modality = SampleConcatenator(
        service
    )._store_single_sample_as_modality
    with pytest.raises(InferenceRefused) as caught:
        geo_fallback_service.GEOFallbackService(service).download_single_cell_sample(
            "GSM1"
        )
    assert caught.value is refusal
    assert len(probe.hooks) == 1
    assert probe.sent == []
