"""Every official compiled specialist issues through owned native clients."""

import asyncio
import importlib
import json
import time
from uuid import uuid4

import pytest

from lobster.config.agent_registry import (
    get_agent_registry_config,
    import_agent_factory,
)
from lobster.config.inference_context import (
    INFERENCE_BINDING_VERSION,
    InferenceContext,
    RequestDescriptor,
)
from lobster.config.providers.controlled_anthropic_client import (
    ControlledAnthropicClient,
    ControlledChatAnthropic,
)
from lobster.config.providers.controlled_bedrock_client import (
    ControlledBedrockClient,
    ControlledChatBedrockConverse,
)
from lobster.config.providers.gateway_bedrock_stub import StubBedrockControlClient
from lobster.core.exceptions import InferenceRefused
from lobster.core.runtime.data_manager import DataManagerV2
from tests.unit.config.test_controlled_anthropic_client import AsyncAuthorizer
from tests.unit.config.test_native_request_hooks import (
    ANTHROPIC_MODEL,
    BEDROCK_MODEL,
    HookProbe,
    anthropic_model,
    bedrock_model,
)
from tests.unit.config.test_specialist_inference_binding import (
    pytest_generate_tests as pytest_generate_tests,
)

pytestmark = pytest.mark.no_auto_config


class NativeFactory:
    capability_version = INFERENCE_BINDING_VERSION

    def __init__(self):
        self.records = []

    def create(self, *, component, context):
        probe = HookProbe()
        authorizer = AsyncAuthorizer()
        views = []

        def validate(view):
            views.append(view)

        def describe(view):
            return RequestDescriptor(
                context.provider,
                context.model_id,
                component,
                context.binding_id,
                context.binding_version,
                str(uuid4()),
                str(uuid4()),
                context.max_output_tokens,
                16000,
                time.time() + 60,
            )

        kwargs = dict(
            context=context,
            component=component,
            authorizer=authorizer,
            validate_request=validate,
            describe_request=describe,
        )
        if context.provider == "bedrock":
            native = bedrock_model(probe)
            owner = ControlledBedrockClient(client=native.client, **kwargs)
            model = ControlledChatBedrockConverse(
                model_id=context.model_id,
                max_tokens=context.max_output_tokens,
                client=owner,
                bedrock_client=StubBedrockControlClient(),
                cache=False,
            )
        else:
            native = anthropic_model(probe)
            owner = ControlledAnthropicClient(
                **kwargs,
                api_key="fake-anthropic-key",
                base_url="https://api.anthropic.com",
                http_client=native._client._client,
                http_async_client=native._async_client._client,
            )
            model = ControlledChatAnthropic(
                model=context.model_id,
                max_tokens=context.max_output_tokens,
                api_key="fake-anthropic-key",
                base_url="https://api.anthropic.com",
                max_retries=0,
                cache=False,
            ).attach_request_owner(owner)
        self.records.append((component, context, owner, probe, authorizer, views))
        return model

    def close(self):
        for _, context, owner, *_ in self.records:
            owner.close()
            if context.provider == "anthropic":
                asyncio.run(owner.aclose())


@pytest.mark.parametrize("provider", ["bedrock", "anthropic"])
def test_official_compiled_agent_native_request_and_outward_refusal(
    name, provider, tmp_path, monkeypatch
):
    config = get_agent_registry_config(name)
    assert config is not None, f"Required official package {name} is not installed"
    constructor = import_agent_factory(config.factory_function)
    module = importlib.import_module(constructor.__module__)

    def no_settings():
        raise AssertionError(
            "Supplied construction consulted ambient generation settings"
        )

    monkeypatch.setattr(module, "get_settings", no_settings)
    context = InferenceContext(
        provider,
        BEDROCK_MODEL if provider == "bedrock" else ANTHROPIC_MODEL,
        "synthetic-binding",
        "version-a",
        200000,
        137,
    )
    factory = NativeFactory()
    try:
        agent = constructor(
            data_manager=DataManagerV2(workspace_path=tmp_path),
            inference_context=context,
            model_factory=factory,
        )
        assert len(factory.records) == 1
        _, actual, _, probe, authorizer, views = factory.records[0]
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "synthetic input"}]}
        )
        assert result["messages"][-1].content == "ok"
        assert actual is context
        assert (
            len(probe.sent)
            == len(authorizer.descriptors)
            == len(authorizer.outcomes)
            == len(views)
            == 1
        )
        assert authorizer.descriptors[0].binding_id == context.binding_id
        assert authorizer.descriptors[0].model_id == context.model_id
        assert authorizer.outcomes[0][1].completeness == "complete"
        refusal = InferenceRefused("accounting")
        authorizer.refusal = refusal
        with pytest.raises(InferenceRefused) as caught:
            agent.invoke(
                {"messages": [{"role": "user", "content": "synthetic refused input"}]}
            )
        assert caught.value is refusal
        assert len(authorizer.descriptors) == 2
        assert len(probe.sent) == len(authorizer.outcomes) == 1
    finally:
        factory.close()


@pytest.mark.parametrize("deny_child", [False, True])
def test_real_supervisor_delegation_preserves_child_native_refusal(
    deny_child, tmp_path
):
    from botocore.awsrequest import AWSResponse
    from langgraph.store.memory import InMemoryStore

    from lobster.agents.graph import create_bioinformatics_graph
    from tests.unit.config.test_native_request_hooks import FakeRawResponse

    context = InferenceContext(
        "bedrock", BEDROCK_MODEL, "synthetic-binding", "version-a", 200000, 137
    )
    factory = NativeFactory()
    try:
        graph, _ = create_bioinformatics_graph(
            DataManagerV2(workspace_path=tmp_path),
            subscription_tier="enterprise",
            enabled_agents=["research_agent"],
            store=InMemoryStore(),
            inference_context=context,
            model_factory=factory,
        )
        supervisor = next(r for r in factory.records if r[0] == "supervisor")
        child = next(r for r in factory.records if r[0] == "research_agent")
        _, _, owner, probe, _, _ = supervisor
        send = owner._client._endpoint.http_session.send

        def reply(request):
            response = send(request)
            if len(probe.sent) != 1:
                return response
            payload = json.loads(request.body)
            tool = next(
                t["toolSpec"]["name"]
                for t in payload["toolConfig"]["tools"]
                if "research" in t["toolSpec"]["name"]
            )
            response.raw.close()
            body = json.dumps(
                {
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "toolUse": {
                                        "toolUseId": "synthetic-call",
                                        "name": tool,
                                        "input": {"task_description": "synthetic task"},
                                    }
                                }
                            ],
                        }
                    },
                    "stopReason": "tool_use",
                    "usage": {"inputTokens": 4, "outputTokens": 1, "totalTokens": 5},
                    "metrics": {"latencyMs": 1},
                }
            ).encode()
            return AWSResponse(
                request.url,
                200,
                {"content-type": "application/json"},
                FakeRawResponse(body, probe),
            )

        owner._client._endpoint.http_session.send = reply
        refusal = InferenceRefused("accounting")
        if deny_child:
            child[4].refusal = refusal
            with pytest.raises(InferenceRefused) as caught:
                graph.invoke(
                    {
                        "messages": [
                            {"role": "user", "content": "synthetic delegated input"}
                        ]
                    }
                )
            assert caught.value is refusal
            assert len(supervisor[4].outcomes) == 1
            assert len(child[4].descriptors) == 1
            assert not child[3].sent
            assert not child[4].outcomes
        else:
            result = graph.invoke(
                {"messages": [{"role": "user", "content": "synthetic delegated input"}]}
            )
            assert result["messages"][-1].content == "ok"
            assert len(supervisor[4].outcomes) == len(supervisor[3].sent) == 2
            assert len(child[4].outcomes) == len(child[3].sent) == 1
            assert all(r[1] is context for r in factory.records)
    finally:
        factory.close()


@pytest.mark.parametrize("provider", ["bedrock", "anthropic"])
def test_full_graph_supervisor_uses_owned_native_request(provider, tmp_path):
    from langgraph.store.memory import InMemoryStore

    from lobster.agents.graph import create_bioinformatics_graph
    from lobster.config.agent_registry import get_worker_agents

    context = InferenceContext(
        provider,
        BEDROCK_MODEL if provider == "bedrock" else ANTHROPIC_MODEL,
        "synthetic-binding",
        "version-a",
        200000,
        137,
    )
    factory = NativeFactory()
    try:
        graph, metadata = create_bioinformatics_graph(
            DataManagerV2(workspace_path=tmp_path),
            subscription_tier="enterprise",
            enabled_agents=list(get_worker_agents()),
            store=InMemoryStore(),
            inference_context=context,
            model_factory=factory,
        )
        assert metadata.agent_count == 22
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "synthetic input"}]}
        )
        assert result["messages"][-1].content == "ok"
        records = [r for r in factory.records if r[4].descriptors]
        assert len(records) == 1
        component, actual, _, probe, authorizer, _ = records[0]
        assert component == "supervisor"
        assert actual is context
        assert len(authorizer.outcomes) == len(probe.sent) == 1
        refusal = InferenceRefused("budget")
        authorizer.refusal = refusal
        with pytest.raises(InferenceRefused) as caught:
            graph.invoke(
                {"messages": [{"role": "user", "content": "synthetic refused input"}]}
            )
        assert caught.value is refusal
        assert len(probe.sent) == len(authorizer.outcomes) == 1
    finally:
        factory.close()


@pytest.mark.parametrize("provider", ["bedrock", "anthropic"])
@pytest.mark.parametrize("helper_name", ["data", "metadata"])
def test_bound_helpers_issue_owned_requests_and_preserve_refusals(
    provider, helper_name, tmp_path
):
    from lobster.agents.data_expert.assistant import DataExpertAssistant
    from lobster.config.inference_runtime import capture_inference_binding
    from lobster.services.metadata.metadata_validation_service import (
        MetadataValidationService,
    )

    context = InferenceContext(
        provider,
        BEDROCK_MODEL if provider == "bedrock" else ANTHROPIC_MODEL,
        "synthetic-binding",
        "version-a",
        200000,
        137,
    )
    factory = NativeFactory()
    manager = DataManagerV2(workspace_path=tmp_path)
    capture_inference_binding(manager, context, factory)
    if helper_name == "data":
        helper = DataExpertAssistant(data_manager=manager)

        def generate():
            return helper.detect_modality({}, "GSE1")

    else:
        helper = MetadataValidationService(data_manager=manager)

        def generate():
            return helper.validate_dataset_metadata(
                {"samples": {"GSM1": {"characteristics_ch1": ["disease: healthy"]}}},
                "GSE1",
                ["disease"],
            )

    try:
        generate()
        assert len(factory.records) == 1
        _, actual, _, probe, authorizer, _ = factory.records[0]
        assert actual is context
        assert len(authorizer.outcomes) == len(probe.sent) == 1
        refusal = InferenceRefused("policy")
        authorizer.refusal = refusal
        with pytest.raises(InferenceRefused) as caught:
            generate()
        assert caught.value is refusal
        assert len(probe.sent) == len(authorizer.outcomes) == 1
    finally:
        factory.close()
