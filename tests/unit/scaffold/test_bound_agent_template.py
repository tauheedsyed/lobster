"""Rendered production factory preserves explicit bindings, not just syntax."""

import importlib.util
import sys
from types import ModuleType

import pytest
from langgraph.prebuilt.chat_agent_executor import AgentState

from lobster.config.inference_context import InferenceContext
from lobster.core.exceptions import InferenceRefused
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.scaffold.generators.agent import _render_template
from tests.unit.config.test_specialist_inference_binding import Factory

pytestmark = pytest.mark.no_auto_config


def test_rendered_factory_builds_real_bound_model_without_settings(
    tmp_path, monkeypatch
):
    domain = "synthetic_binding"
    package = f"lobster.agents.{domain}"
    modules = {
        package: ModuleType(package),
        f"{package}.shared_tools": ModuleType(f"{package}.shared_tools"),
        f"{package}.state": ModuleType(f"{package}.state"),
        f"{package}.prompts": ModuleType(f"{package}.prompts"),
    }
    modules[f"{package}.shared_tools"].create_shared_tools = lambda manager: []
    modules[f"{package}.state"].SyntheticState = AgentState
    modules[f"{package}.prompts"].create_synthetic_expert_prompt = (
        lambda: "Synthetic test"
    )
    for key, module in modules.items():
        monkeypatch.setitem(sys.modules, key, module)
    source = _render_template(
        "agent.py.j2",
        {
            "agent_name": "synthetic_expert",
            "display_name": "Synthetic Expert",
            "description": "Synthetic analysis",
            "domain": domain,
            "handoff_description": "Synthetic task",
            "tier": "free",
            "has_children": False,
            "children": [],
            "state_class": "SyntheticState",
            "package_name": "lobster-synthetic-binding",
        },
    )
    path = tmp_path / "generated_agent.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("synthetic_generated_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def no_settings():
        raise AssertionError("Supplied binding consulted ambient settings")

    monkeypatch.setattr(module, "get_settings", no_settings)
    context = InferenceContext(
        "bedrock",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "synthetic-binding",
        "version-a",
        200000,
        4096,
    )
    bound = Factory()
    manager = DataManagerV2(workspace_path=tmp_path)
    with pytest.raises(InferenceRefused):
        module.synthetic_expert(data_manager=manager, inference_context=context)
    assert not bound.calls
    agent = module.synthetic_expert(
        data_manager=manager, inference_context=context, model_factory=bound
    )
    assert agent is not None
    assert bound.calls == [("synthetic_expert", context)]
