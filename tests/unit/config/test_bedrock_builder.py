"""Exercise output bounds at the real Converse serialization boundary."""

import socket
from copy import deepcopy

import pytest

pytest.importorskip(
    "langchain_aws",
    reason="Converse transport tests require the Bedrock provider extra",
)

from lobster.config.providers.bedrock_builder import build_bedrock_converse
from lobster.config.providers.gateway_bedrock_stub import StubBedrockControlClient

MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


class FakeRuntime:
    """In-memory provider transport; model conversion remains real."""

    def __init__(self):
        self.requests = []

    def converse(self, **request):
        self.requests.append(deepcopy(request))
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 4, "outputTokens": 1, "totalTokens": 5},
            "metrics": {"latencyMs": 1},
        }

    def converse_stream(self, **request):
        self.requests.append(deepcopy(request))
        return {
            "stream": iter(
                [
                    {"messageStart": {"role": "assistant"}},
                    {
                        "contentBlockDelta": {
                            "contentBlockIndex": 0,
                            "delta": {"text": "ok"},
                        }
                    },
                    {"contentBlockStop": {"contentBlockIndex": 0}},
                    {"messageStop": {"stopReason": "end_turn"}},
                    {
                        "metadata": {
                            "usage": {
                                "inputTokens": 4,
                                "outputTokens": 1,
                                "totalTokens": 5,
                            },
                            "metrics": {"latencyMs": 1},
                        }
                    },
                ]
            )
        }


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Unexpected network access")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def model_and_runtime():
    runtime = FakeRuntime()
    model = build_bedrock_converse(
        model_id=MODEL_ID,
        temperature=0.5,
        max_tokens=137,
        client=runtime,
        bedrock_client=StubBedrockControlClient(),
        cache=False,
    )
    return model, runtime


def assert_request(runtime):
    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request["modelId"] == MODEL_ID
    assert request["inferenceConfig"]["maxTokens"] == 137
    assert request["inferenceConfig"]["temperature"] == 0.5


def test_invoke_serializes_max_tokens(model_and_runtime):
    model, runtime = model_and_runtime
    assert model.invoke("hello").content == "ok"
    assert_request(runtime)


@pytest.mark.asyncio
async def test_ainvoke_serializes_max_tokens(model_and_runtime):
    model, runtime = model_and_runtime
    assert (await model.ainvoke("hello")).content == "ok"
    assert_request(runtime)


def test_stream_serializes_max_tokens(model_and_runtime):
    model, runtime = model_and_runtime
    assert "".join(chunk.text for chunk in model.stream("hello")) == "ok"
    assert_request(runtime)


@pytest.mark.asyncio
async def test_astream_serializes_max_tokens(model_and_runtime):
    model, runtime = model_and_runtime
    chunks = [chunk async for chunk in model.astream("hello")]
    assert "".join(chunk.text for chunk in chunks) == "ok"
    assert_request(runtime)


def test_tool_binding_preserves_output_bound(model_and_runtime):
    model, runtime = model_and_runtime
    bound = model.bind_tools(
        [
            {
                "name": "sample",
                "description": "A sample tool",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
    )
    bound.invoke("hello")
    assert_request(runtime)
    assert runtime.requests[0]["toolConfig"]["tools"][0]["toolSpec"]["name"] == "sample"
