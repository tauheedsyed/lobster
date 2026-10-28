"""Opt-in cache points at real native serialization and gateway HTTP seams.

Transport responses use ordinary uncached synthetic usage. These are not evidence
of cache hits, billing completeness, or live provider support.
"""

import json
import socket
from copy import deepcopy

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from lobster.config.providers.bedrock_builder import build_bedrock_converse
from lobster.config.providers.gateway_bedrock_stub import StubBedrockControlClient
from tests.unit.config._native_inference_helpers import require_bedrock
from tests.unit.config.test_controlled_bedrock_client import controlled  # noqa: F401

pytestmark = pytest.mark.no_auto_config

MODELS = [
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "global.anthropic.claude-sonnet-5-5",
]
POINT = {"cachePoint": {"type": "default"}}


@tool
def echo(value: str) -> str:
    """Return a synthetic tool value."""
    return value


class Runtime:
    """Real model conversion, deterministic in-memory provider transport."""

    def __init__(self, tool_loop=False):
        self.requests = []
        self.tool_loop = tool_loop

    def response(self, request):
        self.requests.append(deepcopy(request))
        first = self.tool_loop and len(self.requests) == 1
        content = (
            [
                {
                    "toolUse": {
                        "toolUseId": "call-1",
                        "name": "echo",
                        "input": {"value": "fixture"},
                    }
                }
            ]
            if first
            else [{"text": "ok"}]
        )
        return {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": "tool_use" if first else "end_turn",
            "usage": {"inputTokens": 4, "outputTokens": 1, "totalTokens": 5},
            "metrics": {"latencyMs": 1},
        }

    def converse(self, **request):
        return self.response(request)

    def events(self, response):
        yield {"messageStart": {"role": "assistant"}}
        content = response["output"]["message"]["content"][0]
        if "toolUse" in content:
            use = content["toolUse"]
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {
                        "toolUse": {"toolUseId": use["toolUseId"], "name": use["name"]}
                    },
                }
            }
            delta = {"toolUse": {"input": json.dumps(use["input"])}}
        else:
            delta = {"text": content["text"]}
        yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": delta}}
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": response["stopReason"]}}
        yield {"metadata": {"usage": response["usage"], "metrics": response["metrics"]}}

    def converse_stream(self, **request):
        return {"stream": self.events(self.response(request))}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    require_bedrock()
    monkeypatch.delenv("LOBSTER_BEDROCK_PROMPT_CACHE", raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("Unexpected network access")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def model_for(runtime, model_id=MODELS[0], **kwargs):
    return build_bedrock_converse(
        model_id=model_id,
        client=runtime,
        bedrock_client=StubBedrockControlClient(),
        cache=False,
        max_tokens=137,
        temperature=0.5,
        **kwargs,
    )


def points(request):
    found = []
    for index, block in enumerate(request.get("system", [])):
        if "cachePoint" in block:
            found.append((f"system[{index}]", block))
    for index, block in enumerate(request.get("toolConfig", {}).get("tools", [])):
        if "cachePoint" in block:
            found.append((f"tools[{index}]", block))
    for number, message in enumerate(request.get("messages", [])):
        for index, block in enumerate(message["content"]):
            if "cachePoint" in block:
                found.append((f"msg{number}({message['role']})[{index}]", block))
    return found


def assert_no_unknown_cache_keys(value):
    if isinstance(value, dict):
        assert not {"cache_control", "cacheControl"} & value.keys()
        for item in value.values():
            assert_no_unknown_cache_keys(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_unknown_cache_keys(item)


def assert_points(request, locations):
    found = points(request)
    assert [location for location, _ in found] == locations
    assert len(found) <= 4
    assert all(block == POINT for _, block in found)  # default only; no TTL
    assert_no_unknown_cache_keys(request)
    assert request["inferenceConfig"]["maxTokens"] == 137
    # The library's existing Sonnet 5.5 profile drops unsupported temperature.
    if request["modelId"] == MODELS[2]:
        assert "temperature" not in request["inferenceConfig"]
    else:
        assert request["inferenceConfig"]["temperature"] == 0.5


@pytest.mark.parametrize("flag", [None, "0", "true", " 1", "yes"])
@pytest.mark.parametrize("model_id", [MODELS[0], "amazon.nova-lite-v1:0"])
def test_strict_off_gate_is_plain_and_has_no_points(monkeypatch, flag, model_id):
    from langchain_aws import ChatBedrockConverse

    if flag is not None:
        monkeypatch.setenv("LOBSTER_BEDROCK_PROMPT_CACHE", flag)
    runtime = Runtime()
    model = model_for(runtime, model_id)
    assert type(model) is ChatBedrockConverse
    assert model.invoke("fixture").text == "ok"
    assert_points(runtime.requests[0], [])


def test_non_anthropic_stays_plain_even_when_enabled(monkeypatch):
    from langchain_aws import ChatBedrockConverse

    monkeypatch.setenv("LOBSTER_BEDROCK_PROMPT_CACHE", "1")
    runtime = Runtime()
    model = model_for(runtime, "amazon.nova-lite-v1:0")
    assert type(model) is ChatBedrockConverse
    model.invoke("fixture")
    assert_points(runtime.requests[0], [])


@pytest.mark.parametrize("model_id", MODELS)
def test_enabled_anthropic_gate_and_native_request(monkeypatch, model_id):
    monkeypatch.setenv("LOBSTER_BEDROCK_PROMPT_CACHE", "1")
    runtime = Runtime()
    model = model_for(runtime, model_id)
    assert type(model).__name__ == "CachingChatBedrockConverse"
    assert model.invoke("fixture").text == "ok"
    assert_points(runtime.requests[0], ["msg0(user)[1]"])


def without_points(value):
    if isinstance(value, list):
        return [
            without_points(item)
            for item in value
            if not (isinstance(item, dict) and "cachePoint" in item)
        ]
    if isinstance(value, dict):
        return {key: without_points(item) for key, item in value.items()}
    return value


@pytest.mark.parametrize("model_id", MODELS)
def test_opt_in_keeps_entire_plain_request_except_points(monkeypatch, model_id):
    plain_runtime = Runtime()
    model_for(plain_runtime, model_id).bind_tools([echo]).invoke("fixture")
    monkeypatch.setenv("LOBSTER_BEDROCK_PROMPT_CACHE", "1")
    runtime = Runtime()
    model_for(runtime, model_id).bind_tools([echo]).invoke("fixture")
    assert points(runtime.requests[0])
    assert without_points(runtime.requests[0]) == plain_runtime.requests[0]


def agent_for(model, prompt="Synthetic system"):
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(model=model, tools=[echo], prompt=prompt)


async def run_agent(agent, mode):
    state = {"messages": [HumanMessage(content="fixture")]}
    if mode == "ainvoke":
        output = await agent.ainvoke(state)
    elif mode == "stream":
        outputs = list(agent.stream(state, stream_mode="values"))
        output = outputs[-1]
    else:
        output = agent.invoke(state)
    assert output["messages"][-1].text == "ok"


def assert_loop_requests(runtime):
    assert len(runtime.requests) == 2
    assert_points(runtime.requests[0], ["system[1]", "tools[1]", "msg0(user)[1]"])
    assert_points(
        runtime.requests[1],
        ["system[1]", "tools[1]", "msg1(assistant)[1]", "msg2(user)[1]"],
    )


@pytest.mark.parametrize("model_id", [MODELS[0], MODELS[2]])
@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream"])
@pytest.mark.asyncio
async def test_graph_rebinding_preserves_points(monkeypatch, model_id, mode):
    monkeypatch.setenv("LOBSTER_BEDROCK_PROMPT_CACHE", "1")
    runtime = Runtime(tool_loop=True)
    await run_agent(agent_for(model_for(runtime, model_id)), mode)
    assert_loop_requests(runtime)


@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_model_native_async_and_stream_hooks(monkeypatch, mode):
    monkeypatch.setenv("LOBSTER_BEDROCK_PROMPT_CACHE", "1")
    runtime = Runtime()
    model = model_for(runtime)
    messages = [
        SystemMessage(content="Synthetic system"),
        HumanMessage(content="fixture"),
    ]
    if mode == "ainvoke":
        assert (await model.ainvoke(messages)).text == "ok"
    elif mode == "astream":
        chunks = [chunk async for chunk in model.astream(messages)]
        assert "".join(chunk.text for chunk in chunks) == "ok"
    elif mode == "stream":
        assert "".join(chunk.text for chunk in model.stream(messages)) == "ok"
    else:
        assert model.invoke(messages).text == "ok"
    assert_points(runtime.requests[0], ["system[1]", "msg0(user)[1]"])


@pytest.mark.parametrize("mode", ["invoke", "stream"])
def test_explicit_none_preserves_caller_control(monkeypatch, mode):
    monkeypatch.setenv("LOBSTER_BEDROCK_PROMPT_CACHE", "1")
    runtime = Runtime()
    model = model_for(runtime)
    if mode == "stream":
        list(model.stream("fixture", cache_control=None))
    else:
        model.invoke("fixture", cache_control=None)
    assert_points(runtime.requests[0], [])


def test_manual_point_preserved_and_counts_against_budget(monkeypatch):
    monkeypatch.setenv("LOBSTER_BEDROCK_PROMPT_CACHE", "1")
    runtime = Runtime()
    prompt = SystemMessage(content=[{"text": "Synthetic system"}, deepcopy(POINT)])
    model_for(runtime).bind_tools([echo]).invoke(
        [prompt, HumanMessage(content="fixture")]
    )
    assert_points(runtime.requests[0], ["system[1]", "tools[1]", "msg0(user)[1]"])
    assert prompt.content == [{"text": "Synthetic system"}, POINT]


def test_four_manual_points_leave_no_automatic_budget(monkeypatch):
    monkeypatch.setenv("LOBSTER_BEDROCK_PROMPT_CACHE", "1")
    runtime = Runtime()
    prompt = SystemMessage(
        content=[{"text": "Synthetic system"}, deepcopy(POINT), deepcopy(POINT)]
    )
    user = HumanMessage(content=[{"text": "fixture"}, deepcopy(POINT), deepcopy(POINT)])
    model_for(runtime).bind_tools([echo]).invoke([prompt, user])
    assert_points(
        runtime.requests[0],
        ["system[1]", "system[2]", "msg0(user)[1]", "msg0(user)[2]"],
    )


@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_controlled_client_is_not_opted_in(monkeypatch, request, mode):
    monkeypatch.setenv("LOBSTER_BEDROCK_PROMPT_CACHE", "1")
    model, _, _, _, views = request.getfixturevalue("controlled")
    assert all(
        cls.__name__ != "CachingChatBedrockConverse" for cls in type(model).__mro__
    )
    if mode == "ainvoke":
        await model.ainvoke("fixture")
    elif mode == "astream":
        _ = [chunk async for chunk in model.astream("fixture")]
    elif mode == "stream":
        list(model.stream("fixture"))
    else:
        model.invoke("fixture")
    assert views
    for view in views:
        assert not points(view.parameters)
        assert_no_unknown_cache_keys(view.parameters)


@pytest.fixture
def gateway(monkeypatch):
    from lobster.config import credentials
    from lobster.config.providers.omics_os_provider import OmicsOSProvider

    monkeypatch.setattr(credentials, "get_api_key", lambda: "synthetic-token")
    runtime = Runtime(tool_loop=True)
    posts = []
    original_client = httpx.Client

    def handle(request):
        posts.append(
            (
                request.method,
                request.url.path,
                request.headers.get("X-Omics-Gateway-Contract"),
            )
        )
        response = runtime.response(json.loads(request.content))
        if request.url.path.endswith("converse-stream"):
            content = (
                "\n".join(json.dumps(event) for event in runtime.events(response))
                + "\n[DONE]\n"
            )
            return httpx.Response(
                200, content=content, headers={"content-type": "application/x-ndjson"}
            )
        return httpx.Response(200, json=response)

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original_client(transport=transport, **kwargs)
    )
    return OmicsOSProvider(), runtime, posts


@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream"])
@pytest.mark.parametrize("flag", [None, "0", "1"])
@pytest.mark.asyncio
async def test_gateway_actual_provider_and_http_json(monkeypatch, gateway, mode, flag):
    if flag is not None:
        monkeypatch.setenv("LOBSTER_BEDROCK_PROMPT_CACHE", flag)
    provider, runtime, posts = gateway
    model = provider.create_chat_model(
        MODELS[0],
        endpoint="https://app.omics-os.com",
        cache=False,
        max_tokens=137,
        temperature=0.5,
    )
    await run_agent(agent_for(model), mode)
    assert len(posts) == 2
    assert all(
        method == "POST"
        and path == "/api/v1/gateway/bedrock/converse"
        and contract == "bedrock-converse-v1"
        for method, path, contract in posts
    )
    if flag == "1":
        assert_loop_requests(runtime)
    else:
        from langchain_aws import ChatBedrockConverse

        assert type(model) is ChatBedrockConverse
        for request in runtime.requests:
            assert_points(request, [])


@pytest.mark.parametrize("mode", ["stream", "astream"])
@pytest.mark.asyncio
async def test_gateway_stream_json_and_events(monkeypatch, gateway, mode):
    monkeypatch.setenv("LOBSTER_BEDROCK_PROMPT_CACHE", "1")
    provider, runtime, posts = gateway
    runtime.tool_loop = False
    model = provider.create_chat_model(
        MODELS[0],
        endpoint="https://app.omics-os.com",
        cache=False,
        max_tokens=137,
        temperature=0.5,
    )
    messages = [
        SystemMessage(content="Synthetic system"),
        HumanMessage(content="fixture"),
    ]
    chunks = (
        [chunk async for chunk in model.astream(messages)]
        if mode == "astream"
        else list(model.stream(messages))
    )
    assert "".join(chunk.text for chunk in chunks) == "ok"
    assert posts == [
        ("POST", "/api/v1/gateway/bedrock/converse-stream", "bedrock-converse-v1")
    ]
    assert_points(runtime.requests[0], ["system[1]", "msg0(user)[1]"])


@pytest.mark.parametrize(
    "alias,expected",
    [("claude-sonnet-4-5-20250929", True), ("unresolved-alias", False)],
)
def test_gateway_gate_uses_resolved_model_id(monkeypatch, gateway, alias, expected):
    from langchain_aws import ChatBedrockConverse

    monkeypatch.setenv("LOBSTER_BEDROCK_PROMPT_CACHE", "1")
    provider, runtime, _ = gateway
    runtime.tool_loop = False
    model = provider.create_chat_model(
        alias,
        endpoint="https://app.omics-os.com",
        provider="anthropic",
        cache=False,
        max_tokens=137,
        temperature=0.5,
    )
    assert ("anthropic." in model.model_id) is expected
    assert (type(model) is not ChatBedrockConverse) is expected
    model.invoke("fixture")
    assert_points(runtime.requests[0], ["msg0(user)[1]"] if expected else [])
