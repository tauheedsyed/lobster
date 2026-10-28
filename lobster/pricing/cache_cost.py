"""Cache-aware list-price decisions; unresolved dimensions retain legacy cost.

Only the SDK provider label observed on direct/gateway Bedrock is accepted.
No vendor inference, market-rate fallback, SDK import, request or rate revision.
"""

from collections.abc import Mapping
from decimal import Decimal
from typing import NamedTuple

from lobster.pricing import PricingDataError, default_price_book, for_context, get_rates
from lobster.pricing.usage import NormalizedUsage, _normalize_observations


class CacheCostDecision(NamedTuple):
    status: str
    cost: Decimal | None
    reason: str | None
    snapshot: dict[str, str] | None


LEGACY = CacheCostDecision("legacy", None, None, None)
_READ = (
    "cache_read",
    "cacheReadInputTokens",
    "cacheReadInputTokenCount",
    "cache_read_input_tokens",
)
_WRITE = (
    "cache_creation",
    "cache_write",
    "cacheWriteInputTokens",
    "cacheWriteInputTokenCount",
    "cache_creation_input_tokens",
    "cache_write_input_tokens",
)
_SPLIT = ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")


def _valid_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _maps(metadata, raw):
    maps = []
    if isinstance(metadata, Mapping):
        details = metadata.get("input_token_details")
        if isinstance(details, Mapping):
            maps.append(details)
    if isinstance(raw, Mapping):
        maps.append(raw)
        creation = raw.get("cache_creation")
        if isinstance(creation, Mapping):
            maps.append(creation)
    return maps


def _active(usage, maps):
    if any(value not in (None, 0) for value in (usage.cache_read, usage.cache_write)):
        return True
    for data in maps:
        for key in (*_READ, *_WRITE, *_SPLIT):
            if key not in data or isinstance(data[key], Mapping):
                continue
            if not _valid_count(data[key]) or data[key] != 0:
                return True
    return False


def cache_observations_active(metadata, raw):
    """Only nonzero/invalid reported cache dimensions engage the new seam."""
    return _active(
        NormalizedUsage(None, None, None, None, None, None, False), _maps(metadata, raw)
    )


def normalize_cache_observations(metadata, raw, provider):
    """Same pure counts/bounds; the callback emits one typed fallback warning.

    Avoid a second generic normalization warning for active cache anomalies.
    This does not change public normalize_usage or any inactive observation.
    """
    result = _normalize_observations(metadata, raw, provider)
    if result.complete and (
        result.logical_input is None
        or result.output_incl_reasoning is None
        or (result.cache_read or 0) + (result.cache_write or 0) > result.logical_input
        or (result.reasoning or 0) > result.output_incl_reasoning
    ):
        result = result._replace(complete=False)
    return (
        result
        if result.complete
        else result._replace(logical_input=None, output_incl_reasoning=None)
    )


def _fallback(reason):
    return CacheCostDecision("legacy_fallback", None, reason, None)


def _write_ttl(write, maps, raw):
    if not write:
        return None
    totals = []
    one_hour = False
    if isinstance(raw, Mapping) and "cacheDetails" in raw:
        details = raw["cacheDetails"]
        if not isinstance(details, list) or not details:
            return "write_ttl_invalid"
        total = 0
        for detail in details:
            if not isinstance(detail, Mapping):
                return "write_ttl_invalid"
            count = detail.get("inputTokens")
            ttl = detail.get("ttl")
            if not _valid_count(count) or ttl not in ("5m", "1h"):
                return "write_ttl_invalid"
            one_hour |= ttl == "1h" and count > 0
            total += count
        totals.append(total)
    for data in maps:
        if any(key in data for key in _SPLIT):
            values = [data.get(key, 0) for key in _SPLIT]
            if not all(_valid_count(value) for value in values):
                return "write_ttl_invalid"
            one_hour |= values[1] > 0
            totals.append(sum(values))
    if any(total != write for total in totals):
        return "write_ttl_conflict"
    if one_hour:
        return "write_ttl_unsupported"
    return None if totals else "write_ttl_unreported"


def price_cache_usage(
    usage: NormalizedUsage, *, provider, model_id, usage_metadata=None, raw_usage=None
) -> CacheCostDecision:
    """Return a replacement only when normalized counts, identity and rates agree.

    Missing/zero cache counts do not engage this path at all. Unknown decisions
    carry a content-free code and request the exact legacy result, not zero.
    Explicit positive 1h writes retain legacy cost with a flag: the catalog
    only has a 5m write rate, so that rate must not be applied to 1h evidence.
    """
    maps = _maps(usage_metadata, raw_usage)
    if not _active(usage, maps):
        return LEGACY
    read = 0 if usage.cache_read is None else usage.cache_read
    write = 0 if usage.cache_write is None else usage.cache_write
    values = (usage.logical_input, usage.output_incl_reasoning, read, write)
    if (
        not usage.complete
        or not all(_valid_count(value) for value in values)
        or read + write > usage.logical_input
        or (
            usage.reasoning is not None
            and (
                not _valid_count(usage.reasoning)
                or usage.reasoning > usage.output_incl_reasoning
            )
        )
    ):
        return _fallback("cache_usage_invalid")
    # Only this label is produced by the real SDK on both approved paths.
    if provider != "amazon_bedrock" or not isinstance(model_id, str):
        return _fallback("pricing_identity_unknown")
    reason = _write_ttl(write, maps, raw_usage)
    if reason:
        return _fallback(reason)
    try:
        rates = get_rates("bedrock", model_id)
        if rates is not None:
            rates = for_context(rates, usage.logical_input)
            snapshot = default_price_book().snapshot
    except (PricingDataError, OSError, UnicodeError, OverflowError):
        return _fallback("rates_unknown")
    if rates is None:
        return _fallback("rates_unknown")
    parts = (
        (usage.logical_input - read - write, rates.input),
        (usage.output_incl_reasoning, rates.output),
        (read, rates.cache_read),
        (write, rates.cache_write),
    )
    if any(
        count and (not isinstance(rate, Decimal) or not rate.is_finite() or rate < 0)
        for count, rate in parts
    ):
        return _fallback("rates_unknown")
    cost = sum(
        (Decimal(count) * rate for count, rate in parts if count), Decimal(0)
    ) / Decimal(1_000_000)
    return CacheCostDecision(
        "catalog",
        cost,
        None,
        {
            "snapshot_id": snapshot.snapshot_id,
            "source_sha256": snapshot.source_sha256,
            "overrides_sha256": snapshot.overrides_sha256,
        },
    )
