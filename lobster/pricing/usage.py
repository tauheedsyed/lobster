"""Pure token normalization; no rates, billing, SDK imports or request-coverage claim.

Raw Anthropic/Bedrock input excludes cache. OpenAI/Gemini input includes it.
LangChain metadata is version-dependent: AWS 1.3 keeps exclusive input whereas
AWS 1.9/Anthropic 1.7 include cache. A consistent total resolves that distinction;
otherwise counts remain incomplete unless matching native evidence resolves it.
"""

import logging
from collections.abc import Mapping
from typing import Any, NamedTuple


class NormalizedUsage(NamedTuple):
    logical_input: int | None
    output_incl_reasoning: int | None
    cache_read: int | None
    cache_write: int | None
    reasoning: int | None
    basis: str | None
    complete: bool


UNKNOWN = NormalizedUsage(None, None, None, None, None, None, False)
_TTL_KEYS = ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("unknown or invalid token count")
    return value


def _counter(data: Mapping, *names: str) -> int | None:
    values = [_count(data[name]) for name in names if name in data]
    if values and any(value != values[0] for value in values):
        raise ValueError("conflicting count aliases")
    return values[0] if values else None


def _nullable_counter(data: Mapping, *names: str) -> int | None:
    """Unknown subset details do not invalidate an independently inclusive total."""
    known = {key: data[key] for key in names if key in data and data[key] is not None}
    value = _counter(known, *names)
    return None if any(key in data and data[key] is None for key in names) else value


def _details(data: Mapping, key: str) -> Mapping:
    value = data.get(key, {})
    if value is None:
        return {}  # Optional decomposition, not an unknown aggregate counter.
    if not isinstance(value, Mapping):
        raise ValueError("invalid or unknown detail object")
    return value


def _creation(data: Mapping, *names: str, sdk: bool = False) -> int | None:
    aggregate = _counter(data, *names)
    split = [_count(data[key]) for key in _TTL_KEYS if key in data]
    if not split:
        return aggregate
    total = sum(split)
    # LC 1.9/Anthropic 1.7 replace the generic value with 0 to avoid
    # adding both the aggregate and TTL components in their usage merger.
    if aggregate is not None and aggregate != total and not (sdk and aggregate == 0):
        raise ValueError("cache creation aggregate disagrees with TTL split")
    return total


def _finish(
    input_count,
    output,
    read,
    write,
    reasoning,
    basis,
    total=None,
    *,
    allow_cache_excluded_total=False,
):
    if input_count is None or output is None:
        return UNKNOWN
    cached = (read or 0) + (write or 0)
    if reasoning is not None and reasoning > output:
        raise ValueError("reasoning exceeds inclusive output")
    if basis == "inclusive" and cached > input_count:
        raise ValueError("cache exceeds inclusive input")
    logical = input_count if basis == "inclusive" else input_count + cached
    if total is not None and total != logical + output:
        if not allow_cache_excluded_total or total != input_count + output:
            raise ValueError("inconsistent total tokens")
    return NormalizedUsage(logical, output, read, write, reasoning, basis, True)


def _metadata(data: Mapping) -> NormalizedUsage:
    try:
        input_count = _counter(data, "input_tokens")
        output = _counter(data, "output_tokens")
        if input_count is None or output is None:
            return UNKNOWN
        inputs = _details(data, "input_token_details")
        outputs = _details(data, "output_token_details")
        read = _counter(inputs, "cache_read")
        write = _creation(inputs, "cache_creation", "cache_write", sdk=True)
        reasoning = _counter(outputs, "reasoning")
        total = _counter(data, "total_tokens")
        cached = (read or 0) + (write or 0)
        if reasoning is not None and reasoning > output:
            raise ValueError("reasoning exceeds inclusive output")
        if cached:
            if total == input_count + output and cached <= input_count:
                basis = "inclusive"
            elif total == input_count + output + cached:
                basis = "exclusive"
            else:
                return NormalizedUsage(
                    None, output, read, write, reasoning, "unknown", False
                )
        else:
            basis = "inclusive"  # Equivalent when no cache counts are present.
        return _finish(input_count, output, read, write, reasoning, basis, total)
    except (ValueError, TypeError):
        return UNKNOWN


def _raw(data: Mapping, provider: str | None) -> NormalizedUsage:
    try:
        families = [
            "inputTokens" in data,
            "prompt_token_count" in data or "promptTokenCount" in data,
            "prompt_tokens" in data,
            "input_tokens" in data,
        ]
        if sum(families) != 1:
            return UNKNOWN
        if "inputTokens" in data:  # Native Bedrock Converse, not SDK metadata.
            input_count = _counter(data, "inputTokens")
            output = _counter(data, "outputTokens")
            read = _counter(data, "cacheReadInputTokens", "cacheReadInputTokenCount")
            write = _counter(data, "cacheWriteInputTokens", "cacheWriteInputTokenCount")
            reasoning = _nullable_counter(data, "reasoningTokens")
            thinking = _nullable_counter(
                _details(data, "output_tokens_details"), "thinking_tokens"
            )
            if reasoning is not None and thinking is not None and reasoning != thinking:
                raise ValueError("conflicting reasoning counts")
            if thinking is not None:
                reasoning = thinking
            total = _counter(data, "totalTokens")
            if total is None:
                return UNKNOWN
            return _finish(
                input_count, output, read, write, reasoning, "exclusive", total
            )
        if "prompt_token_count" in data or "promptTokenCount" in data:
            input_count = _counter(data, "prompt_token_count", "promptTokenCount")
            candidates = _counter(
                data, "candidates_token_count", "candidatesTokenCount"
            )
            reasoning = _counter(data, "thoughts_token_count", "thoughtsTokenCount")
            output = None if candidates is None else candidates + (reasoning or 0)
            read = _nullable_counter(
                data, "cached_content_token_count", "cachedContentTokenCount"
            )
            return _finish(
                input_count,
                output,
                read,
                None,
                reasoning,
                "inclusive",
                _counter(data, "total_token_count", "totalTokenCount"),
            )
        if "prompt_tokens" in data:
            input_count = _counter(data, "prompt_tokens")
            output = _counter(data, "completion_tokens")
            native_cache = any(
                key in data
                for key in ("cache_read_input_tokens", "cache_write_input_tokens")
            )
            if native_cache and "prompt_tokens_details" in data:
                return UNKNOWN  # Incompatible cache-exclusive/inclusive families.
            if native_cache:
                # ChatBedrock llm_output['usage'] has exclusive prompt counts.
                read = _counter(data, "cache_read_input_tokens")
                write = _counter(data, "cache_write_input_tokens")
                basis = "exclusive"
            else:
                read = _nullable_counter(
                    _details(data, "prompt_tokens_details"), "cached_tokens"
                )
                write, basis = None, "inclusive"
            reasoning = _nullable_counter(
                _details(data, "completion_tokens_details"), "reasoning_tokens"
            )
            return _finish(
                input_count,
                output,
                read,
                write,
                reasoning,
                basis,
                _counter(data, "total_tokens"),
                allow_cache_excluded_total=(basis == "exclusive"),
            )
        if "input_tokens" in data:
            input_count = _counter(data, "input_tokens")
            output = _counter(data, "output_tokens")
            native = any(
                key in data
                for key in (
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                    "cache_write_input_tokens",
                    "cache_creation",
                )
            )
            output_details = _details(data, "output_tokens_details")
            openai_fields = (
                "input_tokens_details" in data or "reasoning_tokens" in output_details
            )
            if native and openai_fields:
                return UNKNOWN
            openai = openai_fields or (
                not native and provider in {"openai", "azure", "openrouter", "nebius"}
            )
            if openai:
                read = _nullable_counter(
                    _details(data, "input_tokens_details"), "cached_tokens"
                )
                write, basis = None, "inclusive"
                reasoning = _nullable_counter(
                    _details(data, "output_tokens_details"), "reasoning_tokens"
                )
            else:
                read = _counter(data, "cache_read_input_tokens")
                write = _counter(
                    data, "cache_creation_input_tokens", "cache_write_input_tokens"
                )
                if "cache_creation" in data:
                    if data["cache_creation"] is None and write is None:
                        raise ValueError("unknown exclusive cache creation")
                    creation = _details(data, "cache_creation")
                    if set(creation) - set(_TTL_KEYS):
                        raise ValueError("unsupported cache creation detail")
                    split = _creation(creation)
                    if split is not None:
                        if write is not None and write != split:
                            raise ValueError("cache creation total conflict")
                        write = split
                reasoning = _counter(data, "reasoning_tokens")
                thinking = _counter(
                    _details(data, "output_tokens_details"), "thinking_tokens"
                )
                if (
                    reasoning is not None
                    and thinking is not None
                    and reasoning != thinking
                ):
                    raise ValueError("conflicting reasoning counts")
                if thinking is not None:
                    reasoning = thinking
                basis = "exclusive"
            return _finish(
                input_count,
                output,
                read,
                write,
                reasoning,
                basis,
                _counter(data, "total_tokens"),
            )
        return UNKNOWN
    except (ValueError, TypeError):
        return UNKNOWN


def _normalize_observations(
    usage_metadata=None, raw_usage=None, provider=None
) -> NormalizedUsage:
    """Normalize one request's FINAL usage, never repeated streaming deltas.

    Returns logical input, reasoning-inclusive output, cache read/write, reasoning,
    input basis and completeness. Explicit null/invalid counts are unknown, not 0.
    Omitted optional detail counters remain None. ``provider`` describes transport
    semantics, not a pricing identity, entitlement or invocation permission.
    If both observations are supplied, contradictions/unknown raw evidence remain
    incomplete. Matching native evidence can resolve version-ambiguous metadata.
    """
    metadata = (
        _metadata(usage_metadata) if isinstance(usage_metadata, Mapping) else None
    )
    raw = _raw(raw_usage, provider) if isinstance(raw_usage, Mapping) else None
    if metadata is None:
        return raw if raw is not None else UNKNOWN
    if raw is None:
        return metadata
    if not raw.complete:
        return metadata._replace(complete=False)
    if metadata.complete:
        if (metadata.logical_input, metadata.output_incl_reasoning) != (
            raw.logical_input,
            raw.output_incl_reasoning,
        ):
            return metadata._replace(complete=False)
        for field in ("cache_read", "cache_write", "reasoning"):
            a, b = getattr(metadata, field), getattr(raw, field)
            if a is not None and b is not None and a != b:
                return metadata._replace(complete=False)
        return metadata._replace(
            cache_read=(
                metadata.cache_read
                if metadata.cache_read is not None
                else raw.cache_read
            ),
            cache_write=(
                metadata.cache_write
                if metadata.cache_write is not None
                else raw.cache_write
            ),
            reasoning=(
                metadata.reasoning if metadata.reasoning is not None else raw.reasoning
            ),
        )
    if (
        metadata.basis == "unknown"
        and metadata.output_incl_reasoning == raw.output_incl_reasoning
    ):
        cached = (raw.cache_read or 0) + (raw.cache_write or 0)
        observed = usage_metadata.get("input_tokens")
        if observed in {raw.logical_input, raw.logical_input - cached}:
            for field in ("cache_read", "cache_write", "reasoning"):
                a, b = getattr(metadata, field), getattr(raw, field)
                if a is not None and b is not None and a != b:
                    return metadata
            return raw._replace(
                cache_read=(
                    raw.cache_read
                    if raw.cache_read is not None
                    else metadata.cache_read
                ),
                cache_write=(
                    raw.cache_write
                    if raw.cache_write is not None
                    else metadata.cache_write
                ),
                reasoning=(
                    raw.reasoning if raw.reasoning is not None else metadata.reasoning
                ),
            )
    return metadata


def normalize_usage(
    usage_metadata=None, raw_usage=None, provider=None
) -> NormalizedUsage:
    """Normalize final counts; warn, never guess, when token evidence is incomplete.

    Unknown logical totals are None. This never establishes physical-attempt
    coverage or pricing/billing semantics, and never logs response content.
    Completeness covers logical totals, not every optional billing dimension.
    """
    result = _normalize_observations(usage_metadata, raw_usage, provider)
    # Independently valid observations can form an impossible merged decomposition.
    # Apply the bounds to logical totals, regardless of the original input basis.
    if result.complete and (
        result.logical_input is None
        or result.output_incl_reasoning is None
        or (result.cache_read or 0) + (result.cache_write or 0) > result.logical_input
        or (result.reasoning or 0) > result.output_incl_reasoning
    ):
        result = result._replace(complete=False)
    if not result.complete:
        logging.getLogger(__name__).warning(
            "Usage normalization incomplete: missing, invalid, conflicting or ambiguous token evidence"
        )
        return result._replace(logical_input=None, output_incl_reasoning=None)
    return result
