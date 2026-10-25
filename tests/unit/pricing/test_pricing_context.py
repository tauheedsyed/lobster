"""Context tiers are exclusive thresholds and apply to the whole request."""

from decimal import Decimal

import pytest

from lobster.pricing import PricingDataError, for_context, get_rates

pytestmark = pytest.mark.no_auto_config


def test_live_haiku_tier_boundary():
    rates = get_rates("amazon-bedrock", "global.anthropic.claude-haiku-5-5")
    assert rates.provider == "amazon-bedrock"
    assert rates.tiers[0].context_size == 100000
    assert rates.input == Decimal("0.1")
    assert for_context(rates, 99999) is rates
    assert for_context(rates, 100000) is rates
    selected = for_context(rates, 100001)
    assert selected.input == Decimal("0.5")
    assert selected.output == Decimal("2.5")
    assert selected.tiers == ()
    assert rates.input == Decimal("0.1")  # No mutation.


def test_legacy_threshold_fallback_and_reviewed_override():
    rates = get_rates("google", "gemini-3-pro-preview")
    assert rates.manual
    assert rates.cache_read is None and rates.cache_write is None
    assert rates.input == Decimal(2)
    assert for_context(rates, 200000) is rates
    assert for_context(rates, 200001).input == Decimal(4)
    assert for_context(rates, 200001).output == Decimal(18)


def test_explicit_tiers_win_and_highest_exceeded_threshold_selected(make_book):
    rates = make_book(
        {
            "openai": {
                "m": {
                    "input": 1,
                    "output": 2,
                    "context_over_200k": {"input": 99, "output": 99},
                    "tiers": [
                        {
                            "input": 3,
                            "output": 4,
                            "tier": {"type": "context", "size": 300000},
                        },
                        {
                            "input": 2,
                            "output": 3,
                            "tier": {"type": "context", "size": 100000},
                        },
                    ],
                }
            }
        }
    ).get_rates("openai", "m")
    assert for_context(rates, 100000).input == Decimal(1)
    assert for_context(rates, 200001).input == Decimal(2)
    assert for_context(rates, 300000).input == Decimal(2)
    assert for_context(rates, 300001).input == Decimal(3)


@pytest.mark.parametrize("count", [True, -1, 1.0, None, "100"])
def test_reject_noninteger_or_negative_context(count):
    rates = get_rates("amazon-bedrock", "global.anthropic.claude-haiku-5-5")
    with pytest.raises(ValueError):
        for_context(rates, count)


def test_unknown_tier_condition_not_silently_assumed_base(make_book):
    rates = make_book(
        {
            "openai": {
                "m": {
                    "input": 1,
                    "output": 2,
                    "tiers": [{"input": 3, "tier": {"type": "unqualified", "size": 1}}],
                }
            }
        }
    ).get_rates("openai", "m")
    with pytest.raises(PricingDataError, match="unsupported"):
        for_context(rates, 100)


def test_missing_tier_output_stays_unknown(make_book):
    rates = make_book(
        {
            "openai": {
                "m": {
                    "input": 1,
                    "output": 2,
                    "tiers": [{"input": 3, "tier": {"type": "context", "size": 10}}],
                }
            }
        }
    ).get_rates("openai", "m")
    assert for_context(rates, 11).output is None
