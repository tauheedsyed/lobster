"""Synthetic cache-cost unit probes, not live capture or billing evidence."""

from decimal import Decimal

import pytest

from lobster.pricing import SNAPSHOT, for_context, get_rates
from lobster.pricing.usage import NormalizedUsage

MODEL = "global.anthropic.claude-sonnet-5-5"


def decide(usage=None, **kwargs):
    from lobster.pricing.cache_cost import price_cache_usage

    if usage is None:
        usage = NormalizedUsage(20, 5, 7, 3, None, "inclusive", True)
    args = dict(
        provider="amazon_bedrock",
        model_id=MODEL,
        usage_metadata={
            "input_token_details": {
                "cache_read": 7,
                "cache_creation": 3,
                "ephemeral_5m_input_tokens": 3,
            }
        },
        raw_usage=None,
    )
    args.update(kwargs)
    return price_cache_usage(usage, **args)


@pytest.mark.parametrize(
    "model",
    [
        MODEL,
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    ],
)
def test_prices_exact_routed_catalog_without_double_counting(model):
    result = decide(model_id=model)
    rates = for_context(get_rates("bedrock", model), 20)
    expected = (
        10 * rates.input
        + 5 * rates.output
        + 7 * rates.cache_read
        + 3 * rates.cache_write
    ) / Decimal(1_000_000)
    assert result.status == "catalog"
    assert isinstance(result.cost, Decimal)
    assert result.cost == expected
    assert result.snapshot == {
        "snapshot_id": SNAPSHOT.snapshot_id,
        "source_sha256": SNAPSHOT.source_sha256,
        "overrides_sha256": SNAPSHOT.overrides_sha256,
    }
    assert 10 + 7 + 3 == 20


@pytest.mark.parametrize("read,write", [(None, None), (0, 0), (None, 0), (0, None)])
def test_absent_and_zero_cache_decisions_do_not_engage(read, write):
    usage = NormalizedUsage(20, 5, read, write, None, "inclusive", True)
    result = decide(usage, usage_metadata={"input_token_details": {}}, raw_usage=None)
    assert result.status == "legacy"
    assert result.reason is None


def test_read_without_cache_details_is_priceable():
    result = decide(
        NormalizedUsage(20, 5, 7, 0, None, "inclusive", True),
        usage_metadata={"input_token_details": {"cache_read": 7}},
        raw_usage={
            "inputTokens": 13,
            "outputTokens": 5,
            "cacheReadInputTokens": 7,
            "cacheWriteInputTokens": 0,
            "totalTokens": 25,
        },
    )
    assert result.status == "catalog"


def test_native_five_minute_write_details():
    result = decide(
        usage_metadata=None,
        raw_usage={
            "cacheReadInputTokens": 7,
            "cacheWriteInputTokens": 3,
            "cacheDetails": [{"ttl": "5m", "inputTokens": 3}],
        },
    )
    assert result.status == "catalog"


@pytest.mark.parametrize(
    "provider", [None, "", "unknown", "bedrock", "omics-os", "anthropic"]
)
def test_unproved_provider_labels_fall_back_without_model_inference(provider):
    result = decide(provider=provider)
    assert result.status == "legacy_fallback"
    assert result.reason == "pricing_identity_unknown"


def test_unknown_exact_model_is_not_stripped_or_fuzzy_matched():
    result = decide(model_id="unresolved-sonnet-alias")
    assert result.status == "legacy_fallback"
    assert result.reason == "rates_unknown"


def test_incomplete_or_negative_decomposition_falls_back():
    for usage in [
        NormalizedUsage(None, 5, 7, 3, None, "unknown", False),
        NormalizedUsage(9, 5, 7, 3, None, "inclusive", True),
        NormalizedUsage(20, 5, -7, 3, None, "inclusive", True),
        NormalizedUsage(20, 5, True, 3, None, "inclusive", True),
    ]:
        result = decide(usage)
        assert result.status == "legacy_fallback"
        assert result.reason == "cache_usage_invalid"


def test_unknown_write_ttl_falls_back_not_zero():
    result = decide(usage_metadata={"input_token_details": {"cache_creation": 3}})
    assert result.status == "legacy_fallback"
    assert result.reason == "write_ttl_unreported"


@pytest.mark.parametrize(
    "details",
    [
        None,
        {},
        [{"ttl": "5m", "inputTokens": -3}],
        [{"ttl": "5m", "inputTokens": 2}],
        [{"ttl": "5m", "inputTokens": True}],
        [{"ttl": "unexpected", "inputTokens": 3}],
    ],
)
def test_malformed_or_conflicting_native_ttl_falls_back(details):
    result = decide(
        usage_metadata=None,
        raw_usage={"cacheWriteInputTokens": 3, "cacheDetails": details},
    )
    assert result.status == "legacy_fallback"


@pytest.mark.parametrize("native", [True, False])
def test_one_hour_write_explicitly_requests_legacy_fallback(native):
    result = decide(
        usage_metadata=(
            None
            if native
            else {
                "input_token_details": {
                    "cache_creation": 3,
                    "ephemeral_1h_input_tokens": 3,
                }
            }
        ),
        raw_usage=(
            {
                "cacheWriteInputTokens": 3,
                "cacheDetails": [{"ttl": "1h", "inputTokens": 3}],
            }
            if native
            else None
        ),
    )
    assert result.status == "legacy_fallback"
    assert result.cost is None
    assert result.reason == "write_ttl_unsupported"


def test_conflicting_ttl_observations_never_price_five_minute():
    result = decide(
        raw_usage={
            "cacheWriteInputTokens": 3,
            "cacheDetails": [{"ttl": "5m", "inputTokens": 2}],
        }
    )
    assert result.status == "legacy_fallback"


@pytest.mark.parametrize("raw_value", [None, -1, True, "7"])
def test_invalid_reported_cache_counter_is_not_absent(raw_value):
    usage = NormalizedUsage(None, None, None, None, None, None, False)
    result = decide(
        usage, usage_metadata=None, raw_usage={"cacheReadInputTokens": raw_value}
    )
    assert result.status == "legacy_fallback"


def test_context_tiers_receive_inclusive_input_and_exact_route(monkeypatch):
    from lobster.pricing import cache_cost

    seen = []
    original = cache_cost.for_context

    def context(rates, logical_input):
        seen.append(logical_input)
        return original(rates, logical_input)

    monkeypatch.setattr(cache_cost, "for_context", context)
    assert decide().status == "catalog"
    assert seen == [20]


def test_catalog_failure_is_a_flag_not_a_new_zero(monkeypatch):
    from lobster.pricing import PricingDataError, cache_cost

    def unknown(*args):
        raise PricingDataError("synthetic invalid catalog")

    monkeypatch.setattr(cache_cost, "get_rates", unknown)
    result = decide()
    assert result.status == "legacy_fallback"
    assert result.reason == "rates_unknown"


def test_missing_nonzero_rate_falls_back(monkeypatch):
    from dataclasses import replace

    from lobster.pricing import cache_cost

    original = get_rates("bedrock", MODEL)
    monkeypatch.setattr(
        cache_cost, "get_rates", lambda *_: replace(original, cache_read=None)
    )
    assert decide().reason == "rates_unknown"


def test_explicit_zero_rate_is_known_not_missing(monkeypatch):
    from dataclasses import replace

    from lobster.pricing import cache_cost

    original = get_rates("bedrock", MODEL)
    monkeypatch.setattr(
        cache_cost, "get_rates", lambda *_: replace(original, cache_read=Decimal(0))
    )
    assert decide().status == "catalog"
