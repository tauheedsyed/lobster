"""Real graph/helper binding propagation and run-boundary isolation, offline."""

import asyncio
import importlib
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from lobster.agents.graph import create_bioinformatics_graph
from lobster.config.agent_registry import (
    get_agent_registry_config,
    import_agent_factory,
)
from lobster.config.inference_context import INFERENCE_BINDING_VERSION, InferenceContext
from lobster.config.inference_runtime import (
    InferenceBoundGraph,
    capture_inference_binding,
)
from lobster.config.providers.bedrock_builder import build_bedrock_converse
from lobster.config.providers.gateway_bedrock_stub import StubBedrockControlClient
from lobster.core.exceptions import InferenceRefused
from lobster.core.runtime.data_manager import DataManagerV2

pytestmark = pytest.mark.no_auto_config

AGENTS = [
    "research_agent",
    "data_expert_agent",
    "metadata_assistant",
    "transcriptomics_expert",
    "annotation_expert",
    "de_analysis_expert",
]


class NativeRuntime:
    def __init__(self):
        self.requests = []
        self.refusal = None

    def converse(self, **request):
        if self.refusal is not None:
            raise self.refusal
        self.requests.append(request)
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "{}"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 4, "outputTokens": 1, "totalTokens": 5},
            "metrics": {"latencyMs": 1},
        }

    def converse_stream(self, **request):
        raise AssertionError("Use synchronous native responses in lifecycle tests")


class Factory:
    capability_version = INFERENCE_BINDING_VERSION

    def __init__(self):
        self.calls = []
        self.runtime = NativeRuntime()

    def create(self, *, component, context):
        self.calls.append((component, context))
        return build_bedrock_converse(
            model_id=context.model_id,
            max_tokens=context.max_output_tokens,
            client=self.runtime,
            bedrock_client=StubBedrockControlClient(),
            cache=False,
            disable_streaming=True,
        )


@pytest.fixture
def context():
    return InferenceContext(
        "bedrock",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "synthetic-binding",
        "version-a",
        200_000,
        4096,
    )


@pytest.fixture
def no_ambient(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Ambient generation resolution is forbidden")

    from lobster.agents import context_management, graph

    monkeypatch.setattr(graph, "get_settings", forbidden)
    monkeypatch.setattr(context_management, "resolve_context_window", forbidden)
    for name in AGENTS:
        config = get_agent_registry_config(name)
        assert config is not None, f"Required agent package {name} must be installed"
        factory = import_agent_factory(config.factory_function)
        monkeypatch.setattr(
            importlib.import_module(factory.__module__), "get_settings", forbidden
        )
    for module in (
        "lobster.agents.data_expert.assistant",
        "lobster.services.metadata.metadata_validation_service",
    ):
        monkeypatch.setattr(importlib.import_module(module), "get_settings", forbidden)


def build(manager, context, factory, **kwargs):
    return create_bioinformatics_graph(
        manager,
        subscription_tier="enterprise",
        enabled_agents=AGENTS,
        inference_context=context,
        model_factory=factory,
        store=InMemoryStore(),
        **kwargs,
    )


def helpers():
    from lobster.agents.data_expert.assistant import DataExpertAssistant
    from lobster.services.metadata.metadata_validation_service import (
        MetadataValidationService,
    )

    return DataExpertAssistant, MetadataValidationService


def test_real_graph_supervisor_specialists_and_legacy_helpers(
    context, no_ambient, tmp_path
):
    manager = DataManagerV2(workspace_path=tmp_path)
    factory = Factory()
    graph, metadata = build(manager, context, factory)
    assert metadata.agent_count == 6
    assert {component for component, _ in factory.calls} == {"supervisor", *AGENTS}
    assert all(captured is context for _, captured in factory.calls)
    graph.invoke({"messages": [("user", "hello")]})
    for cls in helpers():
        helper = cls(data_manager=manager)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(helper.llm.invoke, "helper input").result()
    assert {component for component, _ in factory.calls} == {
        "supervisor",
        *AGENTS,
        "data_expert_assistant",
        "metadata_validation_service",
    }
    assert len(factory.runtime.requests) == 3
    assert all(
        request["modelId"] == context.model_id for request in factory.runtime.requests
    )
    assert all(
        request["inferenceConfig"]["maxTokens"] == context.max_output_tokens
        for request in factory.runtime.requests
    )


@pytest.mark.parametrize(
    "mode", ["invoke", "ainvoke", "stream", "astream", "astream_events"]
)
def test_real_graph_modes_use_native_bound_model(mode, context, no_ambient, tmp_path):
    factory = Factory()
    graph, _ = build(DataManagerV2(workspace_path=tmp_path), context, factory)
    payload = {"messages": [("user", "hello")]}
    if mode == "invoke":
        graph.invoke(payload)
    elif mode == "stream":
        assert list(graph.stream(payload))
    else:

        async def execute():
            if mode == "ainvoke":
                return await graph.ainvoke(payload)
            return [chunk async for chunk in getattr(graph, mode)(payload)]

        assert asyncio.run(execute())
    assert len(factory.runtime.requests) == 1


@pytest.mark.parametrize("cls_index", [0, 1])
def test_cached_and_uncached_helpers_refuse_after_identical_rebuild(
    cls_index, context, no_ambient, tmp_path
):
    manager = DataManagerV2(workspace_path=tmp_path)
    first, second = Factory(), Factory()
    old_graph, _ = build(manager, context, first)
    cached = helpers()[cls_index](manager)
    uninitialized = helpers()[cls_index](manager)
    old_model = cached.llm
    new_graph, _ = build(manager, context, second)
    assert cached._llm is None
    for helper in (cached, uninitialized):
        with pytest.raises(InferenceRefused):
            _ = helper.llm
    with pytest.raises(InferenceRefused):
        old_graph.invoke({"messages": [("user", "obsolete graph")]})
    fresh = helpers()[cls_index](manager)
    assert fresh.llm is not old_model
    new_graph.invoke({"messages": [("user", "current graph")]})
    assert not first.runtime.requests
    assert len(second.runtime.requests) == 1


@pytest.mark.parametrize("cls_index", [0, 1])
def test_helper_partial_and_conflicting_pairs_refuse(cls_index, context, tmp_path):
    manager = DataManagerV2(workspace_path=tmp_path)
    factory = Factory()
    cls = helpers()[cls_index]
    with pytest.raises(InferenceRefused):
        cls(manager, inference_context=context)
    capture_inference_binding(manager, context, factory)
    with pytest.raises(InferenceRefused):
        cls(
            manager,
            inference_context=replace(context, binding_version="version-b"),
            model_factory=factory,
        )
    with pytest.raises(InferenceRefused):
        cls(manager, inference_context=context, model_factory=Factory())


def test_two_managers_never_share_binding(context, no_ambient, tmp_path):
    first_manager = DataManagerV2(workspace_path=tmp_path / "one")
    second_manager = DataManagerV2(workspace_path=tmp_path / "two")
    first, second = Factory(), Factory()
    capture_inference_binding(first_manager, context, first)
    other = replace(context, binding_id="second-binding")
    capture_inference_binding(second_manager, other, second)

    def invoke(manager):
        return helpers()[0](manager).llm.invoke("helper input")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(invoke, [first_manager, second_manager]))
    assert first.calls[0][1] is context
    assert second.calls[0][1] is other
    assert len(first.runtime.requests) == len(second.runtime.requests) == 1


def test_runtime_binding_is_not_serializable(context, tmp_path):
    binding = capture_inference_binding(
        DataManagerV2(workspace_path=tmp_path), context, Factory()
    )
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(binding)


@pytest.mark.parametrize("part", ["context", "factory"])
def test_partial_graph_pair_refuses_before_model_construction(
    part, context, no_ambient, tmp_path
):
    factory = Factory()
    with pytest.raises(InferenceRefused):
        create_bioinformatics_graph(
            DataManagerV2(workspace_path=tmp_path),
            subscription_tier="enterprise",
            **(
                {"inference_context": context}
                if part == "context"
                else {"model_factory": factory}
            ),
        )
    assert not factory.calls


@pytest.mark.parametrize("failure", ["unsupported", "import", "execution", "wrapped"])
def test_installed_broken_plugin_cannot_be_pruned(
    failure, context, tmp_path, monkeypatch
):
    from lobster.agents import graph

    refusal = InferenceRefused("budget")

    def plugin(
        data_manager,
        callback_handler,
        agent_name,
        inference_context=None,
        model_factory=None,
    ):
        if failure == "wrapped":
            raise RuntimeError("synthetic private provider message") from refusal
        raise ValueError("synthetic private provider message")

    def unsupported(**kwargs):
        raise AssertionError("Unsupported plugin must never be executed")

    def load(name):
        if failure == "import":
            raise ImportError("synthetic private provider message")
        return unsupported if failure == "unsupported" else plugin

    monkeypatch.setattr(graph, "import_agent_factory", load)
    config = SimpleNamespace(
        factory_function="synthetic:plugin", name="plugin", child_agents=[]
    )
    with pytest.raises(InferenceRefused) as caught:
        graph._create_agents_single_pass(
            {"plugin": config},
            {},
            DataManagerV2(workspace_path=tmp_path),
            None,
            None,
            "enterprise",
            None,
            None,
            tmp_path,
            context,
            Factory(),
        )
    assert "synthetic private" not in str(caught.value)
    if failure == "wrapped":
        assert caught.value is refusal


def test_live_stream_blocks_replacement_until_closed(context, no_ambient, tmp_path):
    manager = DataManagerV2(workspace_path=tmp_path)
    graph, _ = build(manager, context, Factory())
    stream = graph.stream({"messages": [("user", "hello")]})
    next(stream)
    with pytest.raises(InferenceRefused):
        build(manager, context, Factory())
    stream.close()
    build(manager, context, Factory())


def test_checkpoint_resume_preserves_original_lease(context, tmp_path):
    class State(TypedDict):
        value: str

    def pause(state):
        return {"value": interrupt("confirm")}

    manager = DataManagerV2(workspace_path=tmp_path)
    factory = Factory()
    binding = capture_inference_binding(manager, context, factory)
    workflow = StateGraph(State)
    workflow.add_node("pause", pause)
    workflow.add_edge(START, "pause")
    workflow.add_edge("pause", END)
    graph = InferenceBoundGraph(workflow.compile(checkpointer=InMemorySaver()), binding)
    config = {"configurable": {"thread_id": "synthetic-thread"}}
    result = graph.invoke({"value": "pending"}, config)
    assert result["__interrupt__"]
    with pytest.raises(InferenceRefused):
        capture_inference_binding(manager, context, Factory(), replace=True)
    assert graph.invoke(Command(resume="accepted"), config)["value"] == "accepted"
    capture_inference_binding(manager, context, Factory(), replace=True)


@pytest.mark.parametrize("operation", ["modality", "strategy", "metadata"])
def test_actual_helper_generation_refusal_survives_executor(
    operation, context, no_ambient, tmp_path
):
    manager = DataManagerV2(workspace_path=tmp_path)
    factory = Factory()
    capture_inference_binding(manager, context, factory)
    refusal = InferenceRefused("budget")
    factory.runtime.refusal = refusal
    data_helper = helpers()[0](manager)
    metadata_helper = helpers()[1](manager)
    operations = {
        "modality": lambda: data_helper.detect_modality(
            {"title": "RNA expression"}, "GSE12345"
        ),
        "strategy": lambda: data_helper.extract_strategy_config(
            {"title": "RNA expression"}, "GSE12345"
        ),
        "metadata": lambda: metadata_helper.validate_dataset_metadata(
            {"samples": {}}, "GSE12345", ["condition"]
        ),
    }
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(InferenceRefused) as caught:
            pool.submit(operations[operation]).result()
    assert caught.value is refusal
    assert not factory.runtime.requests


def test_cached_queue_preparer_is_invalidated_and_fresh_service_uses_new_binding(
    context, no_ambient, tmp_path
):
    from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer

    manager = DataManagerV2(workspace_path=tmp_path)
    first, second = Factory(), Factory()
    build(manager, context, first)
    preparer = GEOQueuePreparer(manager)
    helper = preparer._get_data_expert()
    helper.llm.invoke("helper input")
    build(manager, context, second)
    with pytest.raises(InferenceRefused):
        preparer._get_data_expert().extract_strategy_config({}, "GSE12345")
    fresh = GEOQueuePreparer(manager)
    fresh._get_data_expert().llm.invoke("fresh helper input")
    assert len(first.runtime.requests) == len(second.runtime.requests) == 1


@pytest.mark.parametrize("cls_index", [0, 1])
def test_unbound_helper_cannot_downgrade_after_owner_gets_binding(
    cls_index, context, tmp_path
):
    manager = DataManagerV2(workspace_path=tmp_path)
    helper = helpers()[cls_index](manager)
    capture_inference_binding(manager, context, Factory())
    with pytest.raises(InferenceRefused):
        _ = helper.llm


def test_bound_owner_rejects_ambient_graph(context, no_ambient, tmp_path):
    manager = DataManagerV2(workspace_path=tmp_path)
    capture_inference_binding(manager, context, Factory())
    with pytest.raises(InferenceRefused):
        create_bioinformatics_graph(manager, subscription_tier="enterprise")
