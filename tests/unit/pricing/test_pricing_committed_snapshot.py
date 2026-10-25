"""The committed data files: shape, provenance, canonical form and packaging."""

from __future__ import annotations

import fnmatch
import importlib.resources
import json
import re
import tomllib
from pathlib import Path

import pytest

import lobster.pricing as pricing
from lobster.pricing import schema

pytestmark = pytest.mark.no_auto_config

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_PROVIDERS = [
    "anthropic",
    "azure",
    "bedrock",
    "gemini",
    "nebius",
    "openai",
    "openrouter",
]


def _read(name: str) -> str:
    return (
        importlib.resources.files("lobster.pricing").joinpath(name).read_text("utf-8")
    )


@pytest.fixture(scope="module")
def prices_text() -> str:
    return _read("prices.json")


@pytest.fixture(scope="module")
def prices(prices_text) -> dict:
    return schema.loads_json(prices_text)


@pytest.fixture(scope="module")
def models(prices):
    """Every ``(engine_provider, model_id, entry)`` in the snapshot."""
    return [
        (engine, model_id, entry)
        for engine, block in prices["providers"].items()
        for model_id, entry in block["models"].items()
    ]


def test_data_files_ship_with_the_package():
    package_dir = Path(pricing.__file__).parent
    for name in ("prices.json", "overrides.json", "coverage.json"):
        assert (package_dir / name).is_file(), name

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    patterns = pyproject["tool"]["setuptools"]["package-data"]["lobster"]
    for name in ("prices.json", "overrides.json", "coverage.json"):
        assert any(fnmatch.fnmatch(f"pricing/{name}", p) for p in patterns), name


def test_document_shape(prices):
    assert list(prices) == [
        "schema_version",
        "snapshot_id",
        "fetched_at",
        "unit",
        "source",
        "providers",
    ]
    assert prices["schema_version"] == schema.SCHEMA_VERSION == 1
    assert prices["unit"] == "USD per 1M tokens"
    assert list(prices["providers"]) == ENGINE_PROVIDERS
    for engine, block in prices["providers"].items():
        assert list(block) == ["modelsdev_provider", "source_model_count", "models"]
        assert block["modelsdev_provider"] == schema.PROVIDER_MAP[engine]
        assert 0 < len(block["models"]) <= block["source_model_count"]
        assert list(block["models"]) == sorted(block["models"])


def test_every_model_is_a_verbatim_valid_cost_object(models):
    assert len(models) > 100
    for engine, model_id, entry in models:
        assert list(entry) == ["cost"], (engine, model_id)
        schema.validate_cost(entry["cost"], f"{engine}/{model_id}")


def test_provenance_is_complete_and_self_consistent(prices):
    source = prices["source"]

    assert list(source) == [
        "name",
        "url",
        "retrieved_via",
        "sha256",
        "size_bytes",
        "license",
        "attribution",
        "license_text",
    ]
    assert source["name"] == "models.dev" and source["url"] == schema.SOURCE_URL
    assert source["retrieved_via"] in {"https", "file"}
    assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
    assert isinstance(source["size_bytes"], int) and source["size_bytes"] > 0
    assert schema.normalize_fetched_at(prices["fetched_at"]) == prices["fetched_at"]
    assert prices["snapshot_id"] == schema.derive_snapshot_id(
        source["sha256"], prices["fetched_at"]
    )


def test_full_mit_attribution_is_embedded(prices):
    source = prices["source"]

    assert source["license"] == "MIT"
    assert source["license_text"] == schema.MODELS_DEV_LICENSE_TEXT
    assert "Copyright (c) 2025 models.dev" in source["license_text"]
    assert "Permission is hereby granted" in source["license_text"]
    assert "models.dev" in source["attribution"]


def test_the_raw_source_is_not_committed(prices, prices_text):
    # The whole models.dev payload is megabytes; the snapshot keeps costs only.
    assert len(prices_text.encode("utf-8")) < 1_000_000
    assert len(prices_text.encode("utf-8")) < prices["source"]["size_bytes"] / 4


def test_file_is_in_the_exact_form_the_generator_writes(
    update_pricing, prices, prices_text
):
    """No hand edits and no formatting drift: regenerate-and-diff stays clean."""
    assert update_pricing.render_json(prices, compact_depth=4) == prices_text


def test_lookup_serves_every_committed_model_exactly(models, prices_text):
    plain = json.loads(prices_text)["providers"]
    upstream = schema.PROVIDER_MAP

    for engine, model_id, _ in models:
        rates = pricing.get_rates(engine, model_id)
        assert rates is not None, (engine, model_id)
        assert rates.provider == upstream[engine] and rates.model_id == model_id
        assert rates.source == "models.dev" and not rates.manual
        # Verbatim, and keyed per provider (a shared id never crosses providers).
        assert rates.cost == plain[engine]["models"][model_id]["cost"]
        assert pricing.get_rates(upstream[engine], model_id) == rates
        for field in schema.RATE_FIELDS:
            raw = rates.cost.get(field)
            typed = getattr(rates, field)
            assert (typed is None) == (raw is None), (engine, model_id, field)
            if raw is not None:
                assert typed == schema.loads_json(json.dumps(raw))


def test_snapshot_identity_matches_the_file(prices):
    info = pricing.SNAPSHOT

    assert info.snapshot_id == prices["snapshot_id"]
    assert info.fetched_at == prices["fetched_at"]
    assert info.source_sha256 == prices["source"]["sha256"]
    assert info.source_url == schema.SOURCE_URL
    assert info.schema_version == schema.SCHEMA_VERSION


def test_non_market_providers_price_nothing_even_for_real_model_ids(models):
    for _, model_id, _ in models:
        assert pricing.get_rates("ollama", model_id) is None
        assert pricing.get_rates("omics-os", model_id) is None


def test_bedrock_route_prefixes_are_not_stripped_in_real_data(models):
    bedrock_ids = {model_id for engine, model_id, _ in models if engine == "bedrock"}
    prefixes = ("global.", "us.", "eu.", "apac.", "au.", "jp.")
    routed = [m for m in bedrock_ids if m.startswith(prefixes)]
    assert routed, "expected routed (global./us./eu./...) ids in the Bedrock snapshot"
    for model_id in routed:
        bare = model_id.split(".", 1)[1]
        if bare not in bedrock_ids:
            assert pricing.get_rates("bedrock", bare) is None, bare


def test_overrides_file_is_valid():
    schema.validate_overrides(schema.loads_json(_read("overrides.json")))


def test_coverage_report_is_consistent_with_the_snapshot(prices):
    coverage = json.loads(_read("coverage.json"))

    assert coverage["snapshot_id"] == prices["snapshot_id"]
    assert list(coverage["market_providers"]) == ENGINE_PROVIDERS
    assert set(coverage["excluded_providers"]) == {"ollama", "omics-os"}
    assert all(
        info["market_provider"] is False
        for info in coverage["excluded_providers"].values()
    )
    totals = {"engine_models": 0, "priced": 0, "unpriced": 0}
    for provider, info in coverage["market_providers"].items():
        priced = info["priced_by_snapshot"] + info["priced_by_override"]
        assert priced + len(info["unpriced"]) == info["engine_models"], provider
        for name in info["unpriced"]:  # reported gaps really have no rate
            assert pricing.get_rates(provider, name) is None, (provider, name)
        totals["engine_models"] += info["engine_models"]
        totals["priced"] += priced
        totals["unpriced"] += len(info["unpriced"])
    assert coverage["totals"] == totals
    assert coverage["unresolved_catalog_entries"] == []
