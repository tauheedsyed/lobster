"""Installed specialist factories pass supplied bindings to real model construction."""

import ast
import importlib
from pathlib import Path

import pytest

pytest.importorskip(
    "langchain_aws", reason="Native binding tests require the Bedrock provider extra"
)

from lobster.config.agent_registry import (
    get_agent_registry_config,
    import_agent_factory,
)
from lobster.config.inference_context import INFERENCE_BINDING_VERSION, InferenceContext
from lobster.config.providers.bedrock_builder import build_bedrock_converse
from lobster.config.providers.gateway_bedrock_stub import StubBedrockControlClient
from lobster.core.runtime.data_manager import DataManagerV2

pytestmark = pytest.mark.no_auto_config


class Runtime:
    def converse(self, **kwargs):
        raise AssertionError("Construction must not issue inference")

    def converse_stream(self, **kwargs):
        raise AssertionError("Construction must not issue inference")


class Factory:
    capability_version = INFERENCE_BINDING_VERSION

    def __init__(self):
        self.calls = []

    def create(self, *, component, context):
        self.calls.append((component, context))
        return build_bedrock_converse(
            model_id=context.model_id,
            max_tokens=context.max_output_tokens,
            client=Runtime(),
            bedrock_client=StubBedrockControlClient(),
            cache=False,
        )


@pytest.mark.parametrize(
    "name",
    [
        "research_agent",
        "data_expert_agent",
        "metadata_assistant",
        "transcriptomics_expert",
        "annotation_expert",
        "de_analysis_expert",
    ],
)
def test_installed_factory_uses_supplied_binding_without_settings(
    name, tmp_path, monkeypatch
):
    config = get_agent_registry_config(name)
    assert config is not None, f"Required test package for {name} must be installed"
    factory_function = import_agent_factory(config.factory_function)
    module = importlib.import_module(factory_function.__module__)

    def no_settings():
        raise AssertionError("Strict specialist consulted ambient settings")

    monkeypatch.setattr(module, "get_settings", no_settings)
    context = InferenceContext(
        provider="bedrock",
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        binding_id="synthetic-binding",
        binding_version="version-a",
        context_window=200_000,
        max_output_tokens=4096,
    )
    bound = Factory()
    manager = DataManagerV2(workspace_path=tmp_path)
    agent = factory_function(
        data_manager=manager, inference_context=context, model_factory=bound
    )
    assert agent is not None
    assert len(bound.calls) == 1
    assert bound.calls[0][1] is context


def test_all_repository_specialist_signatures_support_explicit_pair():
    root = Path(__file__).resolve().parents[3]
    found = []
    for path in sorted((root / "packages").rglob("*.py")):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef):
                continue
            arguments = {
                arg.arg for arg in function.args.args + function.args.kwonlyargs
            }
            if (
                "provider_override" not in arguments
                or "model_override" not in arguments
            ):
                continue
            if not any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "create_llm"
                for node in ast.walk(function)
            ):
                continue
            assert {"inference_context", "model_factory"} <= arguments, str(path)
            found.append((path, function.name))
    assert len(found) == 22
