"""Alternate delegation cannot bypass a supplied runtime binding."""

from types import SimpleNamespace

import pytest

from lobster.config.inference_context import INFERENCE_BINDING_VERSION, InferenceContext
from lobster.config.inference_runtime import capture_inference_binding
from lobster.core.exceptions import InferenceRefused
from lobster.services.execution import sdk_delegation_service as sdk


class Factory:
    capability_version = INFERENCE_BINDING_VERSION

    def create(self, *, component, context):
        raise AssertionError("Bound delegation must refuse before model construction")


pytestmark = pytest.mark.no_auto_config


@pytest.fixture
def service(tmp_path, monkeypatch):
    # Construction does not use the optional SDK; protocol fixtures below own it.
    monkeypatch.setattr(sdk, "SDK_AVAILABLE", True)
    manager = SimpleNamespace(workspace_path=tmp_path, list_modalities=lambda: [])
    return sdk.SDKDelegationService(manager)


@pytest.mark.parametrize("operation", ["delegate", "_async_delegate"])
@pytest.mark.asyncio
async def test_supplied_binding_refuses_before_any_sdk_objects(
    service, operation, monkeypatch
):
    context = InferenceContext(
        "bedrock",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "synthetic-binding",
        "version-a",
        200000,
        137,
    )
    capture_inference_binding(service.data_manager, context, Factory())

    def forbidden(*args, **kwargs):
        raise AssertionError("Supplied delegation built an unowned SDK object")

    for name in ("create_sdk_mcp_server", "ClaudeAgentOptions", "ClaudeSDKClient"):
        monkeypatch.setattr(sdk, name, forbidden)
    monkeypatch.setattr(service, "create_lobster_sdk_tools", forbidden)
    with pytest.raises(InferenceRefused) as caught:
        if operation == "delegate":
            service.delegate("synthetic task")
        else:
            await service._async_delegate("synthetic task")
    assert caught.value.category == "context"


def test_legacy_delegation_preserves_sdk_flow_and_provenance(service, monkeypatch):
    calls = []
    monkeypatch.setattr(sdk, "sdk_tool", lambda *a, **k: lambda fn: fn)

    def server(**kwargs):
        calls.append(("server", len(kwargs["tools"])))
        return SimpleNamespace()

    def options(**kwargs):
        calls.append(("options", kwargs["model"]))
        return SimpleNamespace(**kwargs)

    class Client:
        def __init__(self, *, options):
            calls.append(("client", options.max_turns))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            calls.append(("close", None))

        async def query(self, prompt):
            calls.append(("query", prompt))

        async def receive_response(self):
            yield SimpleNamespace(content=[SimpleNamespace(text="synthetic result")])

    monkeypatch.setattr(sdk, "create_sdk_mcp_server", server)
    monkeypatch.setattr(sdk, "ClaudeAgentOptions", options)
    monkeypatch.setattr(sdk, "ClaudeSDKClient", Client)
    result, stats, ir = service.delegate("synthetic task")
    assert result == "synthetic result"
    assert stats["success"]
    assert ir.operation == "sdk_delegation"
    assert [name for name, _ in calls] == [
        "server",
        "options",
        "client",
        "query",
        "close",
    ]
    assert calls[0][1] == 3


def test_legacy_unavailable_sdk_still_reports_missing_optional_dependency(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sdk, "SDK_AVAILABLE", False)
    with pytest.raises(sdk.SDKDelegationError, match="not available"):
        sdk.SDKDelegationService(SimpleNamespace(workspace_path=tmp_path))
