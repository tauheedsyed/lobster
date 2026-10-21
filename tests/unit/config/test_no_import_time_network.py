"""Importing the settings module must not touch the network."""

import json
import subprocess  # nosec B404 # runs the interpreter on a fixed probe
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

_PROBE = textwrap.dedent("""
    import json, socket

    attempts = []  # egress attempts: DNS lookups and connections
    creations = []  # local socket creations (e.g. urllib3's IPv6 probe)

    class _NoSocket:
        def __init__(self, *args, **kwargs):
            creations.append("socket()")
            raise OSError("network disabled by test")

    def _blocked(name):
        def _raise(*args, **kwargs):
            attempts.append(name)
            raise OSError("network disabled by test")
        return _raise

    socket.socket = _NoSocket
    socket.create_connection = _blocked("create_connection")
    socket.getaddrinfo = _blocked("getaddrinfo")
    socket.gethostbyname = _blocked("gethostbyname")

    from lobster.config import settings

    print(json.dumps({
        "attempts": attempts,
        "creations": creations,
        "pricing_models": len(settings.MODEL_PRICING),
        "openrouter_priced": "anthropic/claude-sonnet-4-5" in settings.MODEL_PRICING,
    }))
    """)


def test_settings_import_makes_no_network_attempt():
    result = subprocess.run(  # nosec B603 # fixed argv, no shell, no untrusted input
        [sys.executable, "-B", "-c", _PROBE],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["attempts"] == [], "import attempted DNS/connect"
    # Pricing is still populated, from the static catalogs.
    assert report["pricing_models"] > 0
    assert report["openrouter_priced"] is True


def test_pricing_collection_does_not_call_live_catalog(monkeypatch):
    from lobster.config.providers.openrouter_provider import OpenRouterProvider
    from lobster.config.providers.registry import ProviderRegistry

    def _fail(self):
        raise AssertionError("live catalog fetched while collecting pricing")

    monkeypatch.setattr(OpenRouterProvider, "_models_cache", None)
    monkeypatch.setattr(OpenRouterProvider, "list_models", _fail)

    pricing = ProviderRegistry.get_all_models_with_pricing()

    assert "anthropic/claude-sonnet-4-5" in pricing
    assert pricing["anthropic/claude-sonnet-4-5"]["input_per_million"] == 3.0


def test_openrouter_static_models_is_network_free_and_priced(monkeypatch):
    import httpx

    from lobster.config.providers.openrouter_provider import OpenRouterProvider

    def _fail(*args, **kwargs):
        raise AssertionError("network used by static_models()")

    monkeypatch.setattr(httpx, "get", _fail)
    models = OpenRouterProvider().static_models()

    assert models and all(m.provider == "openrouter" for m in models)
    assert any(m.input_cost_per_million for m in models)


def test_live_catalog_is_still_fetched_on_explicit_use(monkeypatch):
    import httpx

    from lobster.config.providers.openrouter_provider import OpenRouterProvider

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {
                        "id": "vendor/synthetic-model",
                        "name": "Synthetic",
                        "context_length": 1000,
                        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    }
                ]
            }

    calls = []
    monkeypatch.setattr(OpenRouterProvider, "_models_cache", None)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: calls.append(a) or _Response())

    models = OpenRouterProvider().list_models()

    assert calls, "explicit list_models() should still use the live catalog"
    assert [m.name for m in models] == ["vendor/synthetic-model"]
    monkeypatch.setattr(OpenRouterProvider, "_models_cache", None)


def test_hosted_vendor_prefixed_models_are_never_free():
    from lobster.utils.callbacks import TokenTrackingCallback

    tracker = TokenTrackingCallback(session_id="t", pricing_config={})
    for model in (
        "meta-llama/llama-4-scout",
        "qwen/qwen3-32b",
        "google/gemma-3-27b-it",
        "mistralai/mistral-medium-3",
        "deepseek/deepseek-r1",
        "microsoft/phi-4",
    ):
        assert tracker._calculate_cost(model, 1000, 1000) is None, model
    # Local and explicitly free models are still free.
    for model in ("llama3:8b", "qwen2.5-7b", "ollama/llama3.3", "vendor/x-model:free"):
        assert tracker._calculate_cost(model, 1000, 1000) == 0.0, model


def test_live_catalog_models_are_priced_after_explicit_listing(monkeypatch):
    import httpx

    from lobster.config import settings as cfg
    from lobster.config.providers.openrouter_provider import OpenRouterProvider
    from lobster.utils.callbacks import TokenTrackingCallback

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {
                        "id": "vendor/synthetic-llama-9b",
                        "name": "Synthetic",
                        "context_length": 1000,
                        "pricing": {"prompt": "0.000002", "completion": "0.000004"},
                    }
                ]
            }

    model = "vendor/synthetic-llama-9b"
    monkeypatch.setattr(OpenRouterProvider, "_models_cache", None)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Response())
    tracker = TokenTrackingCallback(session_id="t", pricing_config=cfg.MODEL_PRICING)
    try:
        assert tracker._calculate_cost(model, 1_000_000, 1_000_000) is None

        OpenRouterProvider().list_models()

        assert tracker._calculate_cost(model, 1_000_000, 1_000_000) == pytest.approx(
            6.0
        )
        assert cfg.calculate_token_cost(1_000_000, 0, model) == pytest.approx(2.0)
    finally:
        cfg.MODEL_PRICING.pop(model, None)


def test_live_merge_keeps_curated_prices():
    from lobster.config import settings as cfg
    from lobster.config.providers.base_provider import ModelInfo

    before = dict(cfg.MODEL_PRICING["anthropic/claude-sonnet-4-5"])
    cfg.register_model_pricing(
        [
            ModelInfo(
                name="anthropic/claude-sonnet-4-5",
                display_name="x",
                description="x",
                provider="openrouter",
                context_window=1,
                input_cost_per_million=99.0,
                output_cost_per_million=99.0,
            )
        ]
    )
    assert cfg.MODEL_PRICING["anthropic/claude-sonnet-4-5"] == before
