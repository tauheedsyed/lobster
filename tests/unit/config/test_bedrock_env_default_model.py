"""LOBSTER_BEDROCK_MODEL_ID sets the Bedrock default model.

Before 1.1.433 nothing in the engine read this variable, so a deployment that
set it silently ran the catalog default. It is the provider-default layer:
an explicit model and the workspace/global configs still win over it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lobster.config.providers.bedrock_provider import BedrockProvider
from lobster.core.config_resolver import ConfigResolver

ENV = "LOBSTER_BEDROCK_MODEL_ID"
PINNED = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def _catalog_default() -> str:
    return next(m.name for m in BedrockProvider.KNOWN_MODELS if m.is_default)


@pytest.fixture
def isolated_resolver(tmp_path):
    """A resolver with no workspace or global provider config."""
    with patch(
        "lobster.core.config_resolver.GlobalProviderConfig.exists", return_value=False
    ):
        yield ConfigResolver(workspace_path=tmp_path)


def test_env_sets_default_model(monkeypatch):
    monkeypatch.setenv(ENV, PINNED)
    assert BedrockProvider().get_default_model() == PINNED


def test_env_is_stripped(monkeypatch):
    monkeypatch.setenv(ENV, f"  {PINNED}\n")
    assert BedrockProvider().get_default_model() == PINNED


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unset_or_blank_env_keeps_catalog_default(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert BedrockProvider().get_default_model() == _catalog_default()


def test_resolver_reports_env_model_as_provider_default(monkeypatch, isolated_resolver):
    monkeypatch.setenv(ENV, PINNED)
    model, source = isolated_resolver.resolve_model(
        agent_name="supervisor", provider="bedrock"
    )
    assert (model, source) == (PINNED, "provider default (bedrock)")


def test_explicit_model_beats_env(monkeypatch, isolated_resolver):
    monkeypatch.setenv(ENV, PINNED)
    model, source = isolated_resolver.resolve_model(
        agent_name="supervisor",
        provider="bedrock",
        runtime_override="global.anthropic.claude-haiku-5-5",
    )
    assert model == "global.anthropic.claude-haiku-5-5"
    assert source == "runtime flag --model"


def test_global_config_beats_env(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, PINNED)
    resolver = ConfigResolver(workspace_path=tmp_path)
    resolver.global_config = type(
        "Cfg", (), {"get_model_for_provider": lambda self, p: "from-global-config"}
    )()
    with patch(
        "lobster.core.config_resolver.GlobalProviderConfig.exists", return_value=True
    ):
        model, source = resolver.resolve_model(
            agent_name="supervisor", provider="bedrock"
        )
    assert model == "from-global-config"
    assert source.startswith("global user config")


def test_env_does_not_leak_into_other_providers(monkeypatch):
    from lobster.config.providers import get_provider

    monkeypatch.setenv(ENV, PINNED)
    anthropic = get_provider("anthropic")
    assert anthropic is not None
    assert anthropic.get_default_model() != PINNED
