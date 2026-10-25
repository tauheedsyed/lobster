"""Schema, constants and validation shared by the price lookup and its generator.

Stdlib only and free of I/O. ``lobster.pricing`` (runtime lookup) and
``scripts/update_pricing.py`` (generator) both import from here, so what the
generator writes is, by construction, what the lookup accepts.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

SCHEMA_VERSION = 1

SOURCE_NAME = "models.dev"
SOURCE_URL = "https://models.dev/api.json"
SOURCE_LICENSE = "MIT"

# models.dev quotes every rate in USD per one million tokens.
RATE_UNIT = "USD per 1M tokens"

# Engine provider id -> models.dev provider id. These seven are the only providers
# read from the source; every other models.dev provider is ignored.
PROVIDER_MAP: Mapping[str, str] = MappingProxyType(
    {
        "anthropic": "anthropic",
        "azure": "azure",
        "bedrock": "amazon-bedrock",
        "gemini": "google",
        "nebius": "nebius",
        "openai": "openai",
        "openrouter": "openrouter",
    }
)

# Engine providers that are deliberately not priced from a market price list.
NON_MARKET_PROVIDERS: Mapping[str, str] = MappingProxyType(
    {
        "ollama": "runs local models; not a market provider",
        "omics-os": "routes through the Omics-OS Cloud gateway; not a market provider",
    }
)

# Typed rate fields exposed by ``Rates``. ``input_audio`` / ``output_audio`` are also
# range-checked, but only the raw cost object carries them.
RATE_FIELDS = ("input", "output", "cache_read", "cache_write", "reasoning")
_CHECKED_RATE_KEYS = (*RATE_FIELDS, "input_audio", "output_audio")

# No real list price is anywhere near one dollar per token; a larger value is a data
# or unit error and must not reach a cost estimate.
MAX_RATE = Decimal(1_000_000)

# Verbatim copy of the LICENSE file of the models.dev repository (MIT). It is embedded
# in the generated snapshot because the data file is the only thing that always ships
# together with the derived prices.
MODELS_DEV_LICENSE_TEXT = """\
MIT License

Copyright (c) 2025 models.dev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

_PROVIDER_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        **{upstream: engine for engine, upstream in PROVIDER_MAP.items()},
        **{engine: engine for engine in PROVIDER_MAP},
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class PricingDataError(ValueError):
    """Pricing data (snapshot, override or source payload) violates the schema."""


def resolve_provider(provider: object) -> str | None:
    """Return the engine provider id for an engine *or* models.dev id, else ``None``.

    Matching is exact (case-sensitive). Providers without a market price list
    (``ollama``, ``omics-os``) and every unmapped models.dev provider resolve to ``None``.
    """
    if not isinstance(provider, str):
        return None
    return _PROVIDER_ALIASES.get(provider)


def loads_json(text: str) -> Any:
    """Parse JSON with exact decimals, rejecting ``NaN``/``Infinity`` and duplicate keys.

    Non-integer numbers become :class:`~decimal.Decimal`, so a price such as ``0.1`` is
    held exactly rather than as the nearest binary float.
    """
    try:
        return json.loads(
            text,
            parse_float=Decimal,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (ValueError, ArithmeticError, RecursionError) as exc:
        raise PricingDataError(f"invalid JSON: {exc}") from exc


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def normalize_fetched_at(value: datetime | str) -> str:
    """Return ``value`` as a UTC ``YYYY-MM-DDTHH:MM:SSZ`` string (whole seconds).

    Raises ``TypeError``/``ValueError`` for anything that is not a timezone-aware
    ISO-8601 timestamp; a missing timezone is never assumed.
    """
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError as exc:
            raise ValueError(f"not an ISO-8601 timestamp: {value!r}") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError("fetched_at must be a datetime or an ISO-8601 string")
    if parsed.tzinfo is None:
        raise ValueError("fetched_at must carry a timezone (for example ...Z)")
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def derive_snapshot_id(source_sha256: str, fetched_at: datetime | str) -> str:
    """Build the snapshot id from the raw source hash and the fetch time.

    ``<UTC fetch time as YYYYMMDDTHHMMSSZ>-<first 12 hex digits of the source sha256>``
    """
    if not isinstance(source_sha256, str) or not _SHA256_RE.fullmatch(source_sha256):
        raise PricingDataError("source sha256 must be 64 lowercase hex digits")
    stamp = normalize_fetched_at(fetched_at).replace("-", "").replace(":", "")
    return f"{stamp}-{source_sha256[:12]}"


def validate_cost(cost: Any, where: str = "cost") -> None:
    """Check a cost object without altering or interpreting it.

    Rate fields (including those inside ``context_over_200k`` and each ``tiers`` entry)
    must be a number in ``[0, MAX_RATE]`` or ``null``. Everything else is left alone:
    the object is preserved verbatim and conditional pricing is never resolved here.
    """
    if not isinstance(cost, Mapping) or not cost:
        raise PricingDataError(f"{where}: cost must be a non-empty object")
    _check_rates(cost, where)
    nested = cost.get("context_over_200k")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise PricingDataError(f"{where}.context_over_200k: must be an object")
        _check_rates(nested, f"{where}.context_over_200k")
    tiers = cost.get("tiers")
    if tiers is not None:
        if not isinstance(tiers, list):
            raise PricingDataError(f"{where}.tiers: must be a list")
        sizes: set[int] = set()
        for index, tier in enumerate(tiers):
            if not isinstance(tier, Mapping):
                raise PricingDataError(f"{where}.tiers[{index}]: must be an object")
            _check_rates(tier, f"{where}.tiers[{index}]")
            size = context_tier_size(tier, f"{where}.tiers[{index}]")
            if size is not None:
                if size in sizes:
                    raise PricingDataError(f"{where}: duplicate context tier threshold")
                sizes.add(size)


def context_tier_size(tier: Mapping, where: str = "tier") -> int | None:
    """Validate supported context thresholds identically at build and lookup time."""
    condition = tier.get("tier")
    if not isinstance(condition, Mapping) or condition.get("type") != "context":
        return None  # Unsupported conditions remain verbatim; for_context refuses.
    size = condition.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise PricingDataError(f"{where}: context tier size must be a positive integer")
    return size


def _check_rates(mapping: Mapping[str, Any], where: str) -> None:
    for key in _CHECKED_RATE_KEYS:
        if key not in mapping:
            continue
        value = mapping[key]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            raise PricingDataError(
                f"{where}.{key}: expected a number or null, got {type(value).__name__}"
            )
        finite = (
            value.is_finite() if isinstance(value, Decimal) else math.isfinite(value)
        )
        if not finite or value < 0 or value > MAX_RATE:
            raise PricingDataError(
                f"{where}.{key}: {value} is outside 0..{MAX_RATE} {RATE_UNIT}"
            )


@dataclass(frozen=True)
class Override:
    """A hand-reviewed rate for one exact ``(provider, model_id)``."""

    provider: str
    model_id: str
    cost: Mapping[str, Any]
    source_url: str
    checked_at: str
    reviewer: str
    note: str | None = None


_OVERRIDES_KEYS = frozenset({"schema_version", "description", "overrides"})
_OVERRIDE_REQUIRED = (
    "provider",
    "model_id",
    "cost",
    "source_url",
    "checked_at",
    "reviewer",
)
_OVERRIDE_OPTIONAL = ("note",)


def validate_overrides(document: Any) -> tuple[Override, ...]:
    """Validate an ``overrides.json`` document and return its entries.

    Every entry must name a supported market provider by its *engine* id and an exact
    model id, carry a schema-valid ``cost`` object, and record where the rate came from
    (``source_url``, https), when it was checked (``checked_at``, ``YYYY-MM-DD``) and
    who reviewed it (``reviewer``). Duplicate ``(provider, model_id)`` pairs, unknown
    fields and unsupported providers are rejected. ``None`` means "no overrides".
    """
    if document is None:
        return ()
    if not isinstance(document, Mapping):
        raise PricingDataError("overrides document must be an object")
    unknown = sorted(set(document) - _OVERRIDES_KEYS)
    if unknown:
        raise PricingDataError(f"overrides document: unknown field(s) {unknown}")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise PricingDataError(f"overrides schema_version must be {SCHEMA_VERSION}")
    entries = document.get("overrides")
    if not isinstance(entries, list):
        raise PricingDataError("overrides.overrides must be a list")
    parsed: list[Override] = []
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        override = _validate_override(entry, f"overrides[{index}]")
        key = (override.provider, override.model_id)
        if key in seen:
            raise PricingDataError(
                f"overrides[{index}]: duplicate entry for {key[0]}/{key[1]}"
            )
        seen.add(key)
        parsed.append(override)
    return tuple(parsed)


def _validate_override(entry: Any, where: str) -> Override:
    if not isinstance(entry, Mapping):
        raise PricingDataError(f"{where}: must be an object")
    missing = [key for key in _OVERRIDE_REQUIRED if key not in entry]
    if missing:
        raise PricingDataError(f"{where}: missing required field(s) {missing}")
    unknown = sorted(set(entry) - set(_OVERRIDE_REQUIRED) - set(_OVERRIDE_OPTIONAL))
    if unknown:
        raise PricingDataError(f"{where}: unknown field(s) {unknown}")

    provider = entry["provider"]
    if not isinstance(provider, str):
        raise PricingDataError(f"{where}.provider: must be a string")
    if provider in NON_MARKET_PROVIDERS:
        raise PricingDataError(
            f"{where}.provider: {provider!r} is not a market provider"
        )
    if provider not in PROVIDER_MAP:
        raise PricingDataError(
            f"{where}.provider: {provider!r} is not a supported engine provider id "
            f"(expected one of {sorted(PROVIDER_MAP)})"
        )
    validate_cost(entry["cost"], f"{where}.cost")
    note = entry.get("note")
    return Override(
        provider=provider,
        model_id=_clean_text(entry["model_id"], f"{where}.model_id"),
        cost=entry["cost"],
        source_url=_https_url(entry["source_url"], f"{where}.source_url"),
        checked_at=_iso_date(entry["checked_at"], f"{where}.checked_at"),
        reviewer=_clean_text(entry["reviewer"], f"{where}.reviewer"),
        note=None if note is None else _clean_text(note, f"{where}.note"),
    )


def _clean_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PricingDataError(
            f"{where}: must be a non-empty string without surrounding whitespace"
        )
    return value


def _https_url(value: Any, where: str) -> str:
    text = _clean_text(value, where)
    try:
        parts = urlsplit(text)
        host = parts.hostname
    except ValueError:
        host = None
        parts = None
    if (
        parts is None
        or parts.scheme != "https"
        or not host
        or any(char.isspace() for char in text)
    ):
        raise PricingDataError(f"{where}: must be an https:// URL with a host")
    return text


def _iso_date(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _DATE_RE.fullmatch(value):
        raise PricingDataError(f"{where}: must be a YYYY-MM-DD date")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise PricingDataError(f"{where}: {value!r} is not a real date") from exc
    return value
