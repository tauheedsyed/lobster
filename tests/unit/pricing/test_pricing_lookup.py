"""Lookup contract of ``lobster.pricing``.

Exact provider aliases, exact model ids, ``None`` for anything unknown (never zero),
null/zero/absent handling, verbatim conditional pricing and offline operation.
"""

from __future__ import annotations

import json
import subprocess  # nosec B404 # runs the interpreter on a fixed probe
import sys
from decimal import Decimal
from pathlib import Path

import pytest

import lobster.pricing as pricing
from lobster.pricing import PriceBook, PricingDataError, Rates, schema

pytestmark = pytest.mark.no_auto_config

REPO_ROOT = Path(__file__).resolve().parents[3]

EXPECTED_ALIASES = {
    "anthropic": "anthropic",
    "azure": "azure",
    "bedrock": "amazon-bedrock",
    "gemini": "google",
    "nebius": "nebius",
    "openai": "openai",
    "openrouter": "openrouter",
}


def test_supported_provider_mapping_is_exactly_the_agreed_set():
    assert dict(schema.PROVIDER_MAP) == EXPECTED_ALIASES
    assert set(schema.NON_MARKET_PROVIDERS) == {"ollama", "omics-os"}
    assert not set(schema.NON_MARKET_PROVIDERS) & set(EXPECTED_ALIASES)


@pytest.mark.parametrize("engine,upstream", EXPECTED_ALIASES.items())
def test_engine_and_models_dev_ids_resolve_to_the_same_rates(
    make_book, engine, upstream
):
    book = make_book({engine: {"model-x": {"input": 1, "output": 2}}})

    by_engine = book.get_rates(engine, "model-x")
    by_upstream = book.get_rates(upstream, "model-x")

    assert by_engine is not None
    assert by_engine == by_upstream
    assert by_engine.provider == upstream  # always the canonical models.dev id


@pytest.mark.parametrize(
    "provider",
    [
        "ollama",
        "omics-os",
        "ollama-cloud",
        "google-vertex",
        "google-vertex-anthropic",
        "azure-cognitive-services",
        "deepseek",
        "",
        " anthropic",
        "anthropic ",
        "Anthropic",
        "ANTHROPIC",
        "Amazon-Bedrock",
        None,
        7,
    ],
)
def test_unsupported_or_inexact_provider_ids_return_none(make_book, provider):
    book = make_book(
        {engine: {"model-x": {"input": 1, "output": 2}} for engine in EXPECTED_ALIASES}
    )

    assert book.get_rates(provider, "model-x") is None


def test_non_market_providers_have_no_rates_in_the_packaged_data():
    for provider in ("omics-os", "ollama"):
        assert pricing.get_rates(provider, "any-model") is None


def test_bedrock_routes_are_distinct_and_never_normalised(make_book):
    book = make_book(
        {
            "bedrock": {
                "global.acme.sonnet-v1:0": {"input": 3, "output": 15},
                "us.acme.sonnet-v1:0": {"input": 3.3, "output": 16.5},
            }
        }
    )

    assert book.get_rates("bedrock", "global.acme.sonnet-v1:0").input == Decimal("3")
    assert book.get_rates("bedrock", "us.acme.sonnet-v1:0").input == Decimal("3.3")
    for lookalike in (
        "eu.acme.sonnet-v1:0",  # a route nobody priced
        "acme.sonnet-v1:0",  # the prefix-stripped form
        "global.acme.sonnet-v1",  # the version suffix stripped
        "GLOBAL.acme.sonnet-v1:0",
        " global.acme.sonnet-v1:0",
        "global.acme.sonnet-v1:0 ",
        "",
    ):
        assert book.get_rates("bedrock", lookalike) is None, lookalike


def test_same_model_id_is_isolated_per_provider(make_book):
    book = make_book(
        {
            "openai": {"shared": {"input": 1, "output": 2}},
            "azure": {"shared": {"input": 1.5, "output": 3}},
            "openrouter": {"openai/shared": {"input": 1.1, "output": 2.2}},
        }
    )

    assert book.get_rates("openai", "shared").input == Decimal("1")
    assert book.get_rates("azure", "shared").input == Decimal("1.5")
    assert book.get_rates("openrouter", "openai/shared").input == Decimal("1.1")
    # No provider falls back to another provider's entry or to a re-prefixed key.
    assert book.get_rates("anthropic", "shared") is None
    assert book.get_rates("openrouter", "shared") is None
    assert book.get_rates("openai", "openai/shared") is None


def test_unknown_model_and_non_string_model_return_none(make_book):
    book = make_book({"openai": {"known": {"input": 1, "output": 2}}})

    assert book.get_rates("openai", "unknown") is None
    assert book.get_rates("openai", None) is None
    assert book.get_rates("openai", 5) is None
    assert book.get_rates("anthropic", "known") is None  # provider block absent


def test_missing_null_and_zero_are_three_different_things(make_book):
    book = make_book(
        {
            "gemini": {
                "free": {"input": 0, "output": 0.0, "cache_read": 0},
                "partial": {
                    "input": None,
                    "output": 5,
                    "cache_read": None,
                    "reasoning": None,
                },
                "output-only": {"output": 5},
                "no-cost-key": None,
                "empty-cost": {},
            }
        }
    )

    free = book.get_rates("gemini", "free")
    assert free.input == 0 and free.input is not None
    assert free.output == 0 and free.output is not None
    assert free.cache_read == 0 and free.cache_read is not None
    assert free.cache_write is None and free.reasoning is None

    partial = book.get_rates("gemini", "partial")
    assert partial.input is None and partial.cache_read is None
    assert partial.cache_write is None and partial.reasoning is None
    assert partial.output == Decimal(5)
    # The raw object still tells ``null`` and "absent" apart.
    assert "input" in partial.cost and partial.cost["input"] is None
    assert "cache_write" not in partial.cost

    assert book.get_rates("gemini", "output-only").input is None
    assert book.get_rates("gemini", "no-cost-key") is None
    assert book.get_rates("gemini", "empty-cost") is None


def test_rates_are_exact_decimals_and_cost_is_plain_json(make_book):
    book = make_book(
        {"openai": {"m": {"input": 0.1, "output": 3, "cache_read": 0.075}}}
    )

    rates = book.get_rates("openai", "m")

    assert isinstance(rates, Rates)
    assert rates.input == Decimal("0.1") and isinstance(rates.input, Decimal)
    assert rates.output == Decimal(3) and isinstance(rates.output, Decimal)
    assert rates.cache_read == Decimal("0.075")
    assert rates.input * 3 == Decimal("0.3")  # exact, unlike 0.1 * 3 in binary floats
    # ``cost`` holds plain JSON types (what json.load gives), so it serialises as is.
    assert rates.cost == {"input": 0.1, "output": 3, "cache_read": 0.075}
    assert json.loads(json.dumps(rates.cost)) == rates.cost


def test_conditional_pricing_is_preserved_and_never_resolved(make_book):
    tiered = {
        "input": 0.1,
        "output": 0.5,
        "tiers": [
            {"input": 0.5, "output": 2.5, "tier": {"type": "context", "size": 100000}}
        ],
        "context_over_200k": {"input": 0.4, "output": 2.0},
        "input_audio": 1.5,
    }
    book = make_book(
        {"anthropic": {"tiered": tiered, "flat": {"input": 1, "output": 2}}}
    )

    rates = book.get_rates("anthropic", "tiered")

    # Typed fields are the flat base rates: no tier is chosen on anyone's behalf.
    assert (rates.input, rates.output) == (Decimal("0.1"), Decimal("0.5"))
    assert rates.cost == tiered  # complete and verbatim, including audio
    assert rates.has_conditional_rates is True
    assert book.get_rates("anthropic", "flat").has_conditional_rates is False


def test_returned_cost_is_a_private_copy(make_book):
    book = make_book(
        {"openai": {"m": {"input": 1, "output": 2, "tiers": [{"input": 3}]}}}
    )

    first = book.get_rates("openai", "m")
    first.cost["input"] = 999
    first.cost["tiers"].append({"input": 1000})

    second = book.get_rates("openai", "m")
    assert second.cost == {"input": 1, "output": 2, "tiers": [{"input": 3}]}
    assert second.input == Decimal(1)


def test_rates_are_hashable_and_comparable(make_book):
    book = make_book({"openai": {"m": {"input": 1, "output": 2}, "n": {"input": 1}}})

    first, again = book.get_rates("openai", "m"), book.get_rates("openai", "m")

    assert first == again and hash(first) == hash(again)
    assert first != book.get_rates("openai", "n")
    assert len({first, again, book.get_rates("openai", "n")}) == 2


def test_manual_override_wins_and_is_labelled(make_book, override_entry):
    book = make_book(
        {
            "bedrock": {
                "global.acme.sonnet-v1:0": {"input": 3, "output": 15},
                "global.acme.other-v1:0": {"input": 1, "output": 2},
            }
        },
        overrides=[
            override_entry(
                provider="bedrock",
                model_id="global.acme.sonnet-v1:0",
                cost={"input": 4, "output": 16},
            ),
            override_entry(
                provider="bedrock",
                model_id="global.acme.manual-v1:0",
                cost={"input": 7, "output": 21, "cache_read": 0},
            ),
        ],
    )

    replaced = book.get_rates("amazon-bedrock", "global.acme.sonnet-v1:0")
    assert replaced.input == Decimal(4) and replaced.manual
    assert replaced.source == "override"

    gap_fill = book.get_rates("bedrock", "global.acme.manual-v1:0")
    assert gap_fill.manual and gap_fill.cache_read == 0
    assert gap_fill.cost == {"input": 7, "output": 21, "cache_read": 0}

    untouched = book.get_rates("bedrock", "global.acme.other-v1:0")
    assert not untouched.manual and untouched.source == "models.dev"
    # An override is keyed exactly too: no prefix or provider leakage.
    assert book.get_rates("bedrock", "us.acme.manual-v1:0") is None
    assert book.get_rates("anthropic", "global.acme.manual-v1:0") is None


def test_malformed_rate_in_data_raises_instead_of_pricing_wrongly(make_book):
    book = make_book({"openai": {"bad": {"input": "3", "output": 2}}})

    with pytest.raises(PricingDataError, match="openai/bad: cost.input"):
        book.get_rates("openai", "bad")


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d.pop("source"), "'source' and 'providers'"),
        (lambda d: d["source"].pop("sha256"), "missing"),
        (lambda d: d["providers"].update(ollama={}), "unsupported provider"),
        (
            lambda d: d["providers"]["openai"].update(modelsdev_provider="azure"),
            "malformed provider block",
        ),
        (lambda d: d["providers"]["openai"].pop("models"), "malformed provider block"),
    ],
)
def test_book_rejects_documents_it_cannot_interpret(
    update_pricing, snapshot, mutate, match
):
    document = schema.loads_json(update_pricing.render_json(snapshot, compact_depth=4))
    mutate(document)

    with pytest.raises(PricingDataError, match=match):
        PriceBook.from_documents(document)


def test_snapshot_end_to_end_from_generator_output(update_pricing, snapshot):
    """Generator output -> text -> ``PriceBook`` is a closed, loss-free loop."""
    text = update_pricing.render_json(snapshot, compact_depth=4)
    book = PriceBook.from_documents(schema.loads_json(text))

    assert book.snapshot.snapshot_id == snapshot["snapshot_id"]
    assert book.snapshot.fetched_at == "2026-10-24T23:15:51Z"
    assert book.snapshot.source_url == "https://models.dev/api.json"
    assert book.snapshot.source_sha256 == snapshot["source"]["sha256"]
    assert book.snapshot.schema_version == 1
    assert book.get_rates("google", "acme-flash").output == Decimal(5)
    assert book.get_rates("gemini", "acme-flash-empty") is None
    assert book.get_rates("openai", "acme-image") is None
    assert book.get_rates("bedrock", "us.acme.sonnet-v1:0").input == Decimal("3.3")


def test_lookups_work_with_the_network_disabled():
    """Packaged data answers offline (the autouse fixture makes any connect fail)."""
    book = pricing.default_price_book()

    assert book.snapshot.snapshot_id
    assert pricing.SNAPSHOT is book.snapshot
    provider, models = next(iter(book._models.items()))
    model_id = next(iter(models))
    assert pricing.get_rates(provider, model_id) is not None
    assert pricing.get_rates(provider, model_id + "-not-a-model") is None


def test_snapshot_attribute_is_lazy_and_unknown_attributes_still_fail():
    assert pricing.SNAPSHOT.schema_version == schema.SCHEMA_VERSION
    with pytest.raises(AttributeError):
        getattr(pricing, "NOT_A_THING")


_IMPORT_PROBE = """
import json, sys
before = set(sys.modules)
import lobster.pricing as pricing
new = set(sys.modules) - before
foreign = sorted(
    name.split(".")[0] for name in new
    if name.split(".")[0] not in sys.stdlib_module_names
    and name.split(".")[0] != "lobster"
)
print(json.dumps({
    "lobster": sorted(name for name in new if name.split(".")[0] == "lobster"),
    "third_party": foreign,
    "loaded": pricing.default_price_book.cache_info().currsize,
}))
"""


def test_importing_the_package_is_pure():
    """Stdlib only, no other lobster module, and no data file read until first use."""
    result = subprocess.run(  # nosec B603 # fixed argv, no shell, no untrusted input
        [sys.executable, "-B", "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["loaded"] == 0
    assert report["third_party"] == []
    assert set(report["lobster"]) == {
        "lobster",
        "lobster.pricing",
        "lobster.pricing.schema",
        "lobster.version",
    }
