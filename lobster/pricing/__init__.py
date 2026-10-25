"""Offline LLM list-price lookup.

    >>> from lobster.pricing import SNAPSHOT, get_rates
    >>> rates = get_rates("bedrock", "<exact model id>")  # Rates | None

Rates come from ``prices.json``, a snapshot of https://models.dev/api.json that
``scripts/update_pricing.py`` generates and that is committed and reviewed like source.
``overrides.json`` holds hand-reviewed gaps, explicitly labelled as manual.
Nothing is fetched at run time and importing this package performs no I/O: the data
files are read on first use.

Contract
--------
* Exact keys. ``get_rates`` takes an engine provider id (``anthropic``, ``azure``,
  ``bedrock``, ``gemini``, ``nebius``, ``openai``, ``openrouter``) or its models.dev
  id (``amazon-bedrock`` for ``bedrock``, ``google`` for ``gemini``) and the exact
  model id. Nothing is normalised or stripped, so Bedrock ``global.`` / ``us.`` /
  ``eu.`` routes and the same model under different providers stay distinct.
* ``None`` means unknown, never zero: unknown provider, unknown model, a model without
  a cost entry, and the non-market providers ``ollama`` and ``omics-os``. There is no
  fallback to a similar model or a default price. A rate of ``0`` is a real zero.
* Rates are USD per one million tokens, as :class:`~decimal.Decimal`. A rate the
  source does not record is ``None``.
* ``for_context`` selects the highest supported context tier strictly exceeded by
  the logical input count. It never computes cost or changes commercial tariffs.
  The complete, unchanged cost object remains in ``Rates.cost``.

Price data derives from models.dev (MIT License, Copyright (c) 2025 models.dev). The
full license text is embedded in ``prices.json`` under ``source.license_text``.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import importlib.resources
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from lobster.pricing import schema
from lobster.pricing.schema import (
    PROVIDER_MAP,
    SCHEMA_VERSION,
    Override,
    PricingDataError,
    resolve_provider,
)
from lobster.version import __version__

__all__ = [
    "PriceBook",
    "PricingDataError",
    "Rates",
    "SNAPSHOT",
    "SnapshotInfo",
    "Snapshot",
    "Tier",
    "default_price_book",
    "for_context",
    "get_rates",
]

RateSource = Literal["models.dev", "override"]
_FROM_SNAPSHOT: RateSource = "models.dev"
_FROM_OVERRIDE: RateSource = "override"


@dataclass(frozen=True)
class SnapshotInfo:
    """Identity of the committed price snapshot."""

    snapshot_id: str
    source_url: str
    source_sha256: str  # sha256 of the raw models.dev response
    fetched_at: str  # UTC, ``YYYY-MM-DDTHH:MM:SSZ``
    schema_version: int = SCHEMA_VERSION
    overrides_sha256: str = ""
    lobster_version: str = __version__


Snapshot = SnapshotInfo


@dataclass(frozen=True)
class Tier:
    """Whole-request rates above an exclusive logical-input threshold."""

    context_size: int
    rates: Rates


@dataclass(frozen=True)
class Rates:
    """List rates for one exact ``(provider, model_id)``, in USD per 1M tokens.

    ``input`` / ``output`` / ``cache_read`` / ``cache_write`` / ``reasoning`` are
    ``None`` when the source records no value (absent or ``null``); an explicit ``0``
    is kept as ``Decimal(0)``. ``cost`` is a private copy of the complete cost object
    exactly as stored (``tiers``, ``context_over_200k``, audio rates and all); read it
    for anything the typed fields do not cover.
    """

    provider: str  # canonical models.dev id, whichever alias was asked for
    model_id: str
    input: Decimal | None
    output: Decimal | None
    cache_read: Decimal | None
    cache_write: Decimal | None
    reasoning: Decimal | None
    cost: dict[str, Any] = field(default_factory=dict, hash=False)
    source: RateSource = "models.dev"
    tiers: tuple[Tier, ...] = ()

    @property
    def manual(self) -> bool:
        """True when the rate is a hand-reviewed override, not models.dev data."""
        return self.source == _FROM_OVERRIDE

    @property
    def has_conditional_rates(self) -> bool:
        """True when ``cost`` holds conditional pricing the typed rates do not apply.

        A token-count estimate from the typed rates alone can then under- or
        over-state the real charge for the affected requests.
        """
        return any(
            isinstance(value, (dict, list)) and bool(value)
            for value in self.cost.values()
        )


class PriceBook:
    """An immutable set of rates built from a snapshot and its overrides.

    ``get_rates`` and ``SNAPSHOT`` use the packaged data. Build one directly from
    documents to test or to price against another snapshot.
    """

    def __init__(
        self,
        snapshot: SnapshotInfo,
        models: Mapping[str, Mapping[str, Any]],
        overrides: tuple[Override, ...] = (),
    ) -> None:
        self.snapshot = snapshot
        self._models = copy.deepcopy(dict(models))
        self._overrides = {
            (o.provider, o.model_id): replace(o, cost=copy.deepcopy(dict(o.cost)))
            for o in overrides
        }

    @classmethod
    def from_documents(
        cls, prices: Any, overrides: Any = None, *, overrides_sha256: str = ""
    ) -> PriceBook:
        """Build from parsed ``prices.json`` / ``overrides.json`` documents.

        Parse with :func:`lobster.pricing.schema.loads_json` so numbers stay exact.
        """
        if not isinstance(prices, Mapping):
            raise PricingDataError("prices document must be an object")
        if prices.get("schema_version") != SCHEMA_VERSION:
            raise PricingDataError(
                f"unsupported prices schema_version {prices.get('schema_version')!r}; "
                f"this lobster.pricing reads version {SCHEMA_VERSION}"
            )
        source = prices.get("source")
        providers = prices.get("providers")
        if not isinstance(source, Mapping) or not isinstance(providers, Mapping):
            raise PricingDataError("prices document needs 'source' and 'providers'")
        try:
            snapshot = SnapshotInfo(
                snapshot_id=prices["snapshot_id"],
                schema_version=prices["schema_version"],
                fetched_at=prices["fetched_at"],
                source_url=source["url"],
                source_sha256=source["sha256"],
                overrides_sha256=overrides_sha256,
            )
        except KeyError as exc:
            raise PricingDataError(f"prices document is missing {exc}") from exc

        models: dict[str, Mapping[str, Any]] = {}
        for engine_id, block in providers.items():
            upstream = PROVIDER_MAP.get(engine_id)
            if upstream is None:
                raise PricingDataError(f"unsupported provider {engine_id!r} in prices")
            if (
                not isinstance(block, Mapping)
                or block.get("modelsdev_provider") != upstream
                or not isinstance(block.get("models"), Mapping)
            ):
                raise PricingDataError(f"malformed provider block {engine_id!r}")
            models[engine_id] = block["models"]
        return cls(snapshot, models, schema.validate_overrides(overrides))

    def get_rates(self, provider: str, model_id: str) -> Rates | None:
        """Return the rates for ``provider`` / ``model_id``, or ``None`` if unknown.

        A reviewed override for the exact pair wins over the snapshot. Anything not
        found returns ``None``; no other key is ever tried.
        """
        engine_id = resolve_provider(provider)
        if engine_id is None or not isinstance(model_id, str):
            return None
        override = self._overrides.get((engine_id, model_id))
        if override is not None:
            return _build_rates(engine_id, model_id, override.cost, _FROM_OVERRIDE)
        entry = self._models.get(engine_id, {}).get(model_id)
        if not isinstance(entry, Mapping):
            return None
        return _build_rates(engine_id, model_id, entry.get("cost"), _FROM_SNAPSHOT)


def _build_rates(
    provider: str,
    model_id: str,
    cost: Any,
    source: RateSource,
    *,
    include_tiers: bool = True,
) -> Rates | None:
    if not isinstance(cost, Mapping) or not cost:
        return None
    schema.validate_cost(cost, f"{provider}/{model_id}: cost")

    def rate(name: str) -> Decimal | None:
        return _rate(cost.get(name), provider, model_id, name)

    return Rates(
        provider=PROVIDER_MAP.get(provider, provider),
        model_id=model_id,
        input=rate("input"),
        output=rate("output"),
        cache_read=rate("cache_read"),
        cache_write=rate("cache_write"),
        reasoning=rate("reasoning"),
        cost=_to_json_types(cost),
        source=source,
        tiers=_context_tiers(provider, model_id, cost, source) if include_tiers else (),
    )


def _context_tiers(
    provider: str, model_id: str, cost: Mapping[str, Any], source: RateSource
) -> tuple[Tier, ...]:
    tiers = []
    if "tiers" in cost:
        for entry in cost.get("tiers") or []:
            size = schema.context_tier_size(entry)
            if size is None:
                continue
            rates = _build_rates(provider, model_id, entry, source, include_tiers=False)
            if rates is None:
                raise PricingDataError("context tier has no rates")
            tiers.append(Tier(size, rates))
    elif cost.get("context_over_200k") is not None:
        rates = _build_rates(
            provider, model_id, cost["context_over_200k"], source, include_tiers=False
        )
        if rates is not None:
            tiers.append(Tier(200_000, rates))
    if len({tier.context_size for tier in tiers}) != len(tiers):
        raise PricingDataError("duplicate context tier threshold")
    return tuple(sorted(tiers, key=lambda tier: tier.context_size))


def for_context(rates: Rates, input_tokens: int) -> Rates:
    """Select whole-request rates using logical input, with STRICT ``>`` boundaries.

    Explicit tiers take precedence over legacy context_over_200k. Unsupported
    conditions are refused, not guessed. This is rate selection, not accounting.
    """
    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens < 0
    ):
        raise ValueError("input_tokens must be a nonnegative integer")
    if rates.cost.get("tiers") and len(rates.cost["tiers"]) != len(rates.tiers):
        raise PricingDataError("unsupported pricing tier condition")
    selected = rates
    for tier in rates.tiers:
        if input_tokens > tier.context_size:
            selected = tier.rates
    return selected


def _rate(value: Any, provider: str, model_id: str, name: str) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise PricingDataError(
            f"{provider}/{model_id}: cost.{name} is {type(value).__name__}, not a number"
        )
    if isinstance(value, float):  # a document parsed without ``loads_json``
        value = repr(value)
    rate = Decimal(value)
    if not rate.is_finite() or rate < 0:
        raise PricingDataError(f"{provider}/{model_id}: cost.{name} is {rate}")
    return rate


def _to_json_types(value: Any) -> Any:
    """Copy a parsed cost tree back to plain JSON types (``Decimal`` -> ``float``)."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Mapping):
        return {key: _to_json_types(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_types(item) for item in value]
    return value


@functools.cache
def default_price_book() -> PriceBook:
    """The packaged ``prices.json`` + ``overrides.json``, loaded once on first use."""
    root = importlib.resources.files(__name__)
    overrides_bytes = root.joinpath("overrides.json").read_bytes()
    return PriceBook.from_documents(
        schema.loads_json(root.joinpath("prices.json").read_text(encoding="utf-8")),
        schema.loads_json(overrides_bytes.decode("utf-8")),
        overrides_sha256=hashlib.sha256(overrides_bytes).hexdigest(),
    )


def get_rates(provider: str, model_id: str) -> Rates | None:
    """Return packaged list rates for ``provider`` / ``model_id``, or ``None``.

    See the module docstring for the contract. Pure and offline.
    """
    return default_price_book().get_rates(provider, model_id)


if TYPE_CHECKING:
    SNAPSHOT: SnapshotInfo


def __getattr__(name: str) -> Any:
    # ``SNAPSHOT`` is resolved lazily so that importing this package never reads files.
    if name == "SNAPSHOT":
        return default_price_book().snapshot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
