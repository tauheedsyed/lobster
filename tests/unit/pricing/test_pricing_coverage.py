"""Coverage report of the engine's static model catalogs against the snapshot.

Catalogs are read with ``ast`` (never imported), ``ollama`` / ``omics-os`` are reported
as non-market providers rather than as gaps, and the report is deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lobster.pricing import schema

pytestmark = pytest.mark.no_auto_config

REPO_ROOT = Path(__file__).resolve().parents[3]
FETCHED_AT = "2026-10-24T23:15:51Z"

EXPECTED_CATALOGS = {
    "providers": {
        "anthropic": {
            "models": ["acme-sonnet", "acme-unlisted"],
            "sources": ["anthropic_provider.py:KNOWN_MODELS"],
        },
        "bedrock": {
            "models": [
                "eu.acme.sonnet-v1:0",
                "global.acme.sonnet-v1:0",
                "us.acme.sonnet-v1:0",
            ],
            "sources": ["bedrock_provider.py:KNOWN_MODELS"],
        },
        "gemini": {
            "models": ["acme-flash"],
            "sources": ["gemini_provider.py:KNOWN_MODELS"],
        },
        "newvendor": {
            "models": ["nv-1"],
            "sources": ["newvendor_provider.py:KNOWN_MODELS"],
        },
        "omics-os": {
            "models": ["global.acme.sonnet-v1:0"],
            "sources": ["omics_os_provider.py:KNOWN_MODELS"],
        },
        "openai": {
            "models": ["boom-model"],
            "sources": ["never_imported_provider.py:KNOWN_MODELS"],
        },
        "openrouter": {
            "models": ["acme/not-in-snapshot", "openai/shared-model"],
            "sources": ["openrouter_provider.py:_FALLBACK_MODELS"],
        },
    },
    "unresolved": [
        "composed_provider.py:KNOWN_MODELS",  # whole catalog, not a list literal
        "gemini_provider.py:KNOWN_MODELS[1]",
        "gemini_provider.py:KNOWN_MODELS[2]",
        "gemini_provider.py:KNOWN_MODELS[3]",
        "gemini_provider.py:KNOWN_MODELS[4]",
    ],
}


@pytest.fixture
def generated_files(update_pricing, tmp_path, payload_bytes, providers_dir):
    """Run the CLI on the synthetic capture; each call returns the files it wrote."""
    capture = tmp_path / "capture.json"
    capture.write_bytes(payload_bytes)

    def run():
        out = tmp_path / f"out-{len(list(tmp_path.glob('out-*')))}"
        out.mkdir()
        code = update_pricing.main(
            ["--source", str(capture), "--fetched-at", FETCHED_AT]
            + ["--output-dir", str(out), "--providers-dir", str(providers_dir)]
        )
        assert code == 0
        return {p.name: p.read_bytes() for p in out.iterdir()}

    return run


def test_catalogs_are_read_statically(update_pricing, providers_dir):
    catalogs = update_pricing.extract_catalogs(providers_dir)

    assert catalogs == EXPECTED_CATALOGS
    # ``ollama`` has no static catalog; the registry file and the helper-local and
    # non-catalog lists are not provider catalogs and were skipped.
    assert "ollama" not in catalogs["providers"]
    names = {n for info in catalogs["providers"].values() for n in info["models"]}
    assert (
        not {"ignored-registry", "ignored-local-variable", "ignored-not-a-catalog"}
        & names
    )


def test_unreadable_catalog_sources_fail_loudly(update_pricing, tmp_path):
    with pytest.raises(update_pricing.SnapshotError, match="no \\*_provider.py"):
        update_pricing.extract_catalogs(tmp_path)
    (tmp_path / "bad_provider.py").write_text("KNOWN_MODELS = [", encoding="utf-8")
    with pytest.raises(update_pricing.SnapshotError, match="bad_provider.py"):
        update_pricing.extract_catalogs(tmp_path)


def test_coverage_report_is_exact_and_deterministic(
    update_pricing, snapshot, providers_dir
):
    catalogs = update_pricing.extract_catalogs(providers_dir)

    report = update_pricing.build_coverage(snapshot, (), catalogs)

    assert report == update_pricing.build_coverage(snapshot, (), catalogs)
    assert report["snapshot_id"] == snapshot["snapshot_id"]
    assert report["market_providers"] == {
        "anthropic": {
            "sources": ["anthropic_provider.py:KNOWN_MODELS"],
            "engine_models": 2,
            "priced_by_snapshot": 1,
            "priced_by_override": 0,
            "unpriced": ["acme-unlisted"],
        },
        "azure": {
            "sources": [],
            "engine_models": 0,
            "priced_by_snapshot": 0,
            "priced_by_override": 0,
            "unpriced": [],
        },
        "bedrock": {
            "sources": ["bedrock_provider.py:KNOWN_MODELS"],
            "engine_models": 3,
            "priced_by_snapshot": 2,
            "priced_by_override": 0,
            "unpriced": ["eu.acme.sonnet-v1:0"],  # the snapshot has eu...long, not this
        },
        "gemini": {
            "sources": ["gemini_provider.py:KNOWN_MODELS"],
            "engine_models": 1,
            "priced_by_snapshot": 1,
            "priced_by_override": 0,
            "unpriced": [],
        },
        "nebius": {
            "sources": [],
            "engine_models": 0,
            "priced_by_snapshot": 0,
            "priced_by_override": 0,
            "unpriced": [],
        },
        "openai": {
            "sources": ["never_imported_provider.py:KNOWN_MODELS"],
            "engine_models": 1,
            "priced_by_snapshot": 0,
            "priced_by_override": 0,
            "unpriced": ["boom-model"],
        },
        "openrouter": {
            "sources": ["openrouter_provider.py:_FALLBACK_MODELS"],
            "engine_models": 2,
            "priced_by_snapshot": 1,
            "priced_by_override": 0,
            "unpriced": ["acme/not-in-snapshot"],
        },
    }
    assert report["totals"] == {"engine_models": 9, "priced": 5, "unpriced": 4}
    assert report["unresolved_catalog_entries"] == EXPECTED_CATALOGS["unresolved"]
    assert report["unmapped_providers"] == {"newvendor": {"engine_models": 1}}
    assert report["overrides"] == {"count": 0, "shadowing_snapshot": []}


def test_non_market_providers_are_flagged_and_never_counted_as_gaps(
    update_pricing, snapshot, providers_dir
):
    report = update_pricing.build_coverage(
        snapshot, (), update_pricing.extract_catalogs(providers_dir)
    )

    assert set(report["excluded_providers"]) == {"ollama", "omics-os"}
    for info in report["excluded_providers"].values():
        assert info["market_provider"] is False
        assert "not a market provider" in info["reason"]
    assert report["excluded_providers"]["omics-os"]["engine_models"] == 1
    assert report["excluded_providers"]["ollama"]["engine_models"] == 0
    # Their models appear nowhere in the market view or the totals.
    assert "omics-os" not in report["market_providers"]
    assert report["totals"]["engine_models"] == sum(
        info["engine_models"] for info in report["market_providers"].values()
    )


def test_overrides_count_as_covered_and_shadowing_is_flagged(
    update_pricing, snapshot, providers_dir, overrides_document, override_entry
):
    overrides = schema.validate_overrides(
        overrides_document(
            override_entry(provider="bedrock", model_id="eu.acme.sonnet-v1:0"),
            override_entry(provider="bedrock", model_id="us.acme.sonnet-v1:0"),
            override_entry(provider="openai", model_id="boom-model"),
        )
    )

    report = update_pricing.build_coverage(
        snapshot, overrides, update_pricing.extract_catalogs(providers_dir)
    )

    bedrock = report["market_providers"]["bedrock"]
    assert bedrock["priced_by_override"] == 2 and bedrock["priced_by_snapshot"] == 1
    assert bedrock["unpriced"] == []
    assert report["market_providers"]["openai"]["priced_by_override"] == 1
    assert report["totals"] == {"engine_models": 9, "priced": 7, "unpriced": 2}
    # ``us.acme.sonnet-v1:0`` is also in the snapshot: the override hides it.
    assert report["overrides"] == {
        "count": 3,
        "shadowing_snapshot": ["bedrock/us.acme.sonnet-v1:0"],
    }


def test_written_coverage_file_is_byte_stable(update_pricing, generated_files):
    first, second = generated_files(), generated_files()

    assert first["coverage.json"] == second["coverage.json"]
    report = json.loads(first["coverage.json"])
    assert list(report["market_providers"]) == sorted(schema.PROVIDER_MAP)
    assert report["description"].endswith("are not inspected.")


def test_real_engine_catalogs_are_statically_readable(update_pricing):
    """Guards the AST pass against provider modules drifting to non-literal catalogs."""
    catalogs = update_pricing.extract_catalogs(
        REPO_ROOT / "lobster" / "config" / "providers"
    )

    assert catalogs["unresolved"] == []
    for provider in ("anthropic", "bedrock", "openai", "openrouter"):
        assert catalogs["providers"][provider]["models"], provider
    assert catalogs["providers"]["omics-os"]["models"]
