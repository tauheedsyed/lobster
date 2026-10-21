"""Gateway NDJSON termination and cleanup through real HTTPX and model parsing."""

import json
from uuid import uuid4

import httpx
import pytest
from botocore.exceptions import ClientError

from lobster.config.providers.gateway_bedrock_client import GatewayBedrockClient
from lobster.config.providers.gateway_bedrock_stub import StubBedrockControlClient

EVENTS = [
    {"messageStart": {"role": "assistant"}},
    {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "partial"}}},
    {"contentBlockStop": {"contentBlockIndex": 0}},
    {"messageStop": {"stopReason": "end_turn"}},
    {
        "metadata": {
            "usage": {"inputTokens": 2, "outputTokens": 1, "totalTokens": 3},
            "metrics": {"latencyMs": 1},
        }
    },
]


@pytest.fixture
def gateway_transport(monkeypatch):
    clients, streams = [], []
    original = httpx.Client

    class Lines(httpx.SyncByteStream):
        def __init__(self, lines, failure):
            self.lines, self.failure, self.closes = lines, failure, 0

        def __iter__(self):
            for line in self.lines:
                yield (line + "\n").encode()
            if self.failure is not None:
                raise self.failure

        def close(self):
            self.closes += 1

    def build(lines, failure=None):
        stream = Lines(lines, failure)
        streams.append(stream)

        class CountedClient(original):
            def __init__(self, **kwargs):
                super().__init__(
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(200, stream=stream)
                    ),
                    trust_env=False,
                    **kwargs,
                )
                self.closes = 0

            def close(self):
                self.closes += 1
                super().close()

        def create(**kwargs):
            client = CountedClient(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(httpx, "Client", create)
        return (
            GatewayBedrockClient("https://example.invalid", lambda: uuid4().hex),
            stream,
            clients,
        )

    yield build
    # Keep genuinely RED pre-fix tests from leaving their synthetic clients open.
    for client in clients:
        if not client.is_closed:
            client.close()


def _body(client):
    return client.converse_stream(modelId="synthetic", messages=[])["stream"]


def test_complete_stream_preserves_events_skips_json_noise_and_closes_once(
    gateway_transport,
):
    lines = [
        "",
        "not-json",
        *map(json.dumps, EVENTS),
        "[DONE]",
        json.dumps({"must_not_be_read": True}),
    ]
    gateway, transport, clients = gateway_transport(lines)
    stream = _body(gateway)
    assert list(stream) == EVENTS
    assert transport.closes == clients[0].closes == 1
    stream.close()
    stream.close()
    with pytest.raises(StopIteration):
        next(stream)
    assert transport.closes == clients[0].closes == 1


@pytest.mark.parametrize("events", [[], EVENTS[:2], EVENTS])
def test_cut_stream_raises_incomplete_even_after_message_stop(
    gateway_transport, events
):
    gateway, transport, clients = gateway_transport(list(map(json.dumps, events)))
    stream = _body(gateway)
    assert [next(stream) for _ in events] == events
    with pytest.raises(ClientError) as caught:
        next(stream)
    assert caught.value.response["Error"]["Code"] == "StreamIncomplete"
    assert caught.value.operation_name == "ConverseStream"
    assert transport.closes == clients[0].closes == 1
    stream.close()
    assert transport.closes == clients[0].closes == 1


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ReadError("synthetic stream failure"),
        KeyboardInterrupt("synthetic interruption"),
    ],
)
def test_mid_iteration_exception_closes_both_owners(gateway_transport, failure):
    gateway, transport, clients = gateway_transport([json.dumps(EVENTS[0])], failure)
    stream = _body(gateway)
    assert next(stream) == EVENTS[0]
    with pytest.raises(type(failure)) as caught:
        next(stream)
    assert caught.value is failure
    assert transport.closes == clients[0].closes == 1
    stream.close()
    assert transport.closes == clients[0].closes == 1


def test_explicit_close_is_idempotent_before_consumption(gateway_transport):
    gateway, transport, clients = gateway_transport(list(map(json.dumps, EVENTS)))
    stream = _body(gateway)
    stream.close()
    stream.close()
    with pytest.raises(StopIteration):
        next(stream)
    assert transport.closes == clients[0].closes == 1


@pytest.mark.parametrize("events", [EVENTS[:2], EVENTS])
def test_actual_chat_bedrock_stream_propagates_incomplete(gateway_transport, events):
    ChatBedrockConverse = pytest.importorskip(
        "langchain_aws", reason="Native model parsing requires lobster-ai[bedrock]"
    ).ChatBedrockConverse
    gateway, transport, clients = gateway_transport(list(map(json.dumps, events)))
    model = ChatBedrockConverse(
        model_id="anthropic.claude-sonnet-4-5-20250929-v1:0",
        client=gateway,
        bedrock_client=StubBedrockControlClient(),
        region_name="us-east-1",
        cache=False,
    )
    chunks = []
    with pytest.raises(ClientError) as caught:
        for chunk in model.stream("hello"):
            chunks.append(chunk)
    assert any("partial" in str(chunk.content) for chunk in chunks)
    assert caught.value.response["Error"]["Code"] == "StreamIncomplete"
    assert transport.closes == clients[0].closes == 1
