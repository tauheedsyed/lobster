"""Fixtures for the pricing tests.

Everything here is synthetic: invented provider models and made-up rates, so no test
asserts a real market price. The network is blocked for every test in this directory;
the generator is exercised through injected fetchers and local files only.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import socket
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from lobster.pricing import PriceBook, schema

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "update_pricing.py"
FETCHED_AT = "2026-10-24T23:15:51Z"

ABSENT = object()  # a model entry that carries no ``cost`` key at all


@pytest.fixture(autouse=True)
def _network_is_blocked(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("pricing tests must not touch the network")

    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture(scope="session")
def update_pricing():
    """The generator module, imported by path (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("update_pricing", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _model(model_id: str, cost: Any = ABSENT, **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": model_id,
        "name": f"Name of {model_id}",
        "description": "free-form text that must never reach the snapshot",
        "limit": {"context": 128000, "output": 4096},
        "modalities": {"input": ["text"], "output": ["text"]},
        "release_date": "2026-01-01",
    }
    if cost is not ABSENT:
        entry["cost"] = cost
    entry.update(extra)
    return entry


def _provider(provider_id: str, models: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": provider_id,
        "name": provider_id.title(),
        "env": ["EXAMPLE_API_KEY"],
        "doc": "https://example.invalid/docs",
        "models": dict(models),
    }


TIERED_COST = {
    "input": 0.1,
    "output": 0.5,
    "cache_read": 0.01,
    "cache_write": 0.125,
    "tiers": [
        {
            "input": 0.5,
            "output": 2.5,
            "cache_read": 0.05,
            "cache_write": 0.625,
            "tier": {"type": "context", "size": 100000},
        }
    ],
}
LONG_CONTEXT_COST = {
    "input": 4,
    "output": 20,
    "reasoning": 20,
    "input_audio": 2.5,
    "context_over_200k": {"input": 8, "output": 30},
}


def modelsdev_payload() -> dict[str, Any]:
    """A small models.dev-shaped payload covering every case the generator handles."""
    models = {
        "anthropic": {
            "acme-sonnet": {
                "input": 3,
                "output": 15,
                "cache_read": 0.3,
                "cache_write": 3.75,
            },
            "acme-haiku": TIERED_COST,
            "acme-free": {"input": 0, "output": 0},
            "acme-nocost": ABSENT,
        },
        "amazon-bedrock": {
            "global.acme.sonnet-v1:0": {"input": 3, "output": 15},
            "us.acme.sonnet-v1:0": {"input": 3.3, "output": 16.5},
            "acme.sonnet-v1:0": {"input": 3.6, "output": 18},
            "eu.acme.long-v1:0": LONG_CONTEXT_COST,
        },
        "google": {
            "acme-flash": {
                "input": None,
                "output": 5,
                "cache_read": None,
                "cache_write": None,
                "reasoning": None,
            },
            "acme-flash-empty": {},
            "acme-flash-null": None,
        },
        "openai": {
            "shared-model": {"input": 1, "output": 2},
            "acme-gpt": {"input": 2.5, "output": 10, "cache_read": 1.25},
            "acme-image": ABSENT,
        },
        "azure": {"shared-model": {"input": 1.5, "output": 3}},
        "nebius": {"acme/open-model": {"input": 0.2, "output": 0.6}},
        "openrouter": {
            "openai/shared-model": {"input": 1.1, "output": 2.2},
            "acme/router-free": {"input": 0, "output": 0},
            "~acme/alias-latest": {"input": 2, "output": 10},
        },
    }
    payload = {
        provider: _provider(
            provider,
            {
                mid: _model(mid, cost if cost is ABSENT else copy.deepcopy(cost))
                for mid, cost in entries.items()
            },
        )
        for provider, entries in models.items()
    }
    # Providers that exist upstream but are not supported: must be ignored.
    for ignored in (
        "google-vertex",
        "azure-cognitive-services",
        "ollama-cloud",
        "deepseek",
    ):
        payload[ignored] = _provider(
            ignored,
            {
                "shared-model": _model("shared-model", {"input": 99, "output": 99}),
                "acme-flash": _model("acme-flash", {"input": 99, "output": 99}),
            },
        )
    return payload


@pytest.fixture
def payload() -> dict[str, Any]:
    return modelsdev_payload()


@pytest.fixture
def payload_bytes(payload) -> bytes:
    return json.dumps(payload).encode("utf-8")


@pytest.fixture
def snapshot(update_pricing, payload_bytes) -> dict[str, Any]:
    return update_pricing.build_snapshot(payload_bytes, FETCHED_AT)


def _override_entry(**changes: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "provider": "bedrock",
        "model_id": "global.acme.manual-v1:0",
        "cost": {"input": 7, "output": 21},
        "source_url": "https://example.invalid/pricing",
        "checked_at": "2026-10-20",
        "reviewer": "reviewer-handle",
    }
    entry.update(changes)
    return entry


def _overrides_document(*entries: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": schema.SCHEMA_VERSION, "overrides": list(entries)}


@pytest.fixture
def override_entry() -> Callable[..., dict[str, Any]]:
    """Factory for one valid ``overrides.json`` entry, with fields replaced by kwargs."""
    return _override_entry


@pytest.fixture
def overrides_document() -> Callable[..., dict[str, Any]]:
    """Factory for an ``overrides.json`` document holding the given entries."""
    return _overrides_document


@pytest.fixture
def make_book() -> Callable[..., PriceBook]:
    """Build a PriceBook from ``{engine_provider: {model_id: cost}}``.

    A ``None`` cost stores the model entry without any ``cost`` key.
    """

    def build(
        providers: Mapping[str, Mapping[str, Any]],
        overrides: list[Mapping[str, Any]] | None = None,
    ) -> PriceBook:
        document = {
            "schema_version": schema.SCHEMA_VERSION,
            "snapshot_id": "20261024T231551Z-abcdef012345",
            "fetched_at": FETCHED_AT,
            "unit": schema.RATE_UNIT,
            "source": {
                "name": schema.SOURCE_NAME,
                "url": schema.SOURCE_URL,
                "sha256": "abcdef012345" * 5 + "abcd",
            },
            "providers": {
                engine: {
                    "modelsdev_provider": schema.PROVIDER_MAP[engine],
                    "source_model_count": len(models),
                    "models": {
                        mid: ({} if cost is None else {"cost": cost})
                        for mid, cost in models.items()
                    },
                }
                for engine, models in providers.items()
            },
        }
        return PriceBook.from_documents(
            schema.loads_json(json.dumps(document)),
            schema.loads_json(json.dumps(_overrides_document(*(overrides or [])))),
        )

    return build


_MODEL = (
    'ModelInfo(name="{name}", display_name="x", description="d", provider="{provider}")'
)


def _catalog(attribute: str, *models: tuple[str, str], indent: str = "    ") -> str:
    rows = "".join(
        f"{indent}    {_MODEL.format(name=name, provider=provider)},\n"
        for name, provider in models
    )
    return f"{indent}{attribute} = [\n{rows}{indent}]\n"


PROVIDER_SOURCES = {
    "anthropic_provider.py": "class AnthropicProvider:\n"
    + _catalog(
        "KNOWN_MODELS",
        ("acme-sonnet", "anthropic"),
        ("acme-unlisted", "anthropic"),
    ),
    # Annotated class attribute, positional arguments, a duplicate name, a list that
    # is not a catalog and a catalog-looking name inside a function: only the first
    # three distinct names of KNOWN_MODELS count.
    "bedrock_provider.py": (
        "from typing import ClassVar, List\n\n"
        "class BedrockProvider:\n"
        "    KNOWN_MODELS: ClassVar[List[ModelInfo]] = [\n"
        '        ModelInfo("global.acme.sonnet-v1:0", "x", "d", "bedrock"),\n'
        '        ModelInfo(name="us.acme.sonnet-v1:0", display_name="x", '
        'description="d", provider="bedrock"),\n'
        '        ModelInfo(name="eu.acme.sonnet-v1:0", display_name="x", '
        'description="d", provider="bedrock"),\n'
        '        ModelInfo(name="us.acme.sonnet-v1:0", display_name="again", '
        'description="d", provider="bedrock"),\n'
        "    ]\n"
        + _catalog("OTHER_MODELS", ("ignored-not-a-catalog", "bedrock"))
        + "\n    def helper(self):\n"
        + _catalog(
            "KNOWN_MODELS", ("ignored-local-variable", "bedrock"), indent="        "
        )
    ),
    # Entries that are not plain literals are reported, never guessed.
    "gemini_provider.py": (
        'NAME = "dynamic"\n\n'
        "class GeminiProvider:\n"
        "    KNOWN_MODELS = [\n"
        '        ModelInfo(name="acme-flash", display_name="x", description="d", '
        'provider="gemini"),\n'
        '        ModelInfo(name=NAME, display_name="x", description="d", '
        'provider="gemini"),\n'
        '        ModelInfo(name=f"x-{NAME}", display_name="x", description="d", '
        'provider="gemini"),\n'
        '        ModelInfo(name="no-provider-literal", display_name="x", '
        'description="d", provider=NAME),\n'
        '        make_model("not-a-model-info-call"),\n'
        "    ]\n"
    ),
    # Module-level curated list, as OpenRouter's fallback catalog is written.
    "openrouter_provider.py": "from typing import List\n\n"
    + _catalog(
        "_FALLBACK_MODELS: List[ModelInfo]",
        ("openai/shared-model", "openrouter"),
        ("acme/not-in-snapshot", "openrouter"),
        indent="",
    ),
    "omics_os_provider.py": "class OmicsProvider:\n"
    + _catalog("KNOWN_MODELS", ("global.acme.sonnet-v1:0", "omics-os")),
    "ollama_provider.py": "class OllamaProvider:\n    pass\n",
    "newvendor_provider.py": "class NewVendorProvider:\n"
    + _catalog("KNOWN_MODELS", ("nv-1", "newvendor")),
    # A catalog that is assembled rather than written out cannot be read statically.
    "composed_provider.py": (
        "class ComposedProvider:\n"
        "    KNOWN_MODELS = BASE_MODELS + [\n"
        '        ModelInfo(name="late", display_name="x", description="d", provider="openai"),\n'
        "    ]\n"
    ),
    # Importing this file would raise: the catalog reader must only parse it.
    "never_imported_provider.py": (
        'raise RuntimeError("provider modules must not be imported")\n'
        + _catalog("KNOWN_MODELS", ("boom-model", "openai"), indent="")
    ),
    # Not a *_provider.py file: ignored even though it defines a catalog.
    "registry.py": "class Registry:\n"
    + _catalog("KNOWN_MODELS", ("ignored-registry", "anthropic")),
}


@pytest.fixture
def providers_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "providers"
    directory.mkdir()
    for name, source in PROVIDER_SOURCES.items():
        (directory / name).write_text(source, encoding="utf-8")
    return directory
