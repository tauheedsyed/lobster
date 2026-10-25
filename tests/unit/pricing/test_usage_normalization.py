"""Provider-shaped final usage, independent of SDK version and billing policy."""

import pytest

from lobster.pricing.usage import normalize_usage

pytestmark = pytest.mark.no_auto_config


def lc(input_count, total, read=9000, write=2000):
    return {
        "input_tokens": input_count,
        "output_tokens": 100,
        "total_tokens": total,
        "input_token_details": {"cache_read": read, "cache_creation": write},
        "output_token_details": {"reasoning": 40},
    }


@pytest.mark.parametrize(
    "input_count,total,basis", [(1000, 12100, "exclusive"), (12000, 12100, "inclusive")]
)
def test_aws_exclusive_and_inclusive_metadata_variants(input_count, total, basis):
    result = normalize_usage(usage_metadata=lc(input_count, total), provider="bedrock")
    assert result == (12000, 100, 9000, 2000, 40, basis, True)


def test_native_converse_same_logical_input_as_control_plane():
    result = normalize_usage(
        raw_usage={
            "inputTokens": 1000,
            "outputTokens": 100,
            "cacheReadInputTokens": 9000,
            "cacheWriteInputTokens": 2000,
            "totalTokens": 12100,
        },
        provider="bedrock",
    )
    assert result.logical_input == 12000
    assert result.output_incl_reasoning == 100
    assert result.cache_read == 9000 and result.cache_write == 2000
    assert result.complete


def test_native_converse_count_aliases_match_or_conflict():
    usage = {
        "inputTokens": 1000,
        "outputTokens": 100,
        "cacheReadInputTokens": 9000,
        "cacheReadInputTokenCount": 9000,
        "cacheWriteInputTokenCount": 2000,
        "totalTokens": 12100,
    }
    assert normalize_usage(raw_usage=usage).logical_input == 12000
    usage["cacheReadInputTokenCount"] = 1
    assert not normalize_usage(raw_usage=usage).complete


def test_native_anthropic_creation_aggregate_not_added_twice():
    usage = {
        "input_tokens": 1000,
        "output_tokens": 100,
        "cache_read_input_tokens": 9000,
        "cache_creation_input_tokens": 2000,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 500,
            "ephemeral_1h_input_tokens": 1500,
        },
    }
    result = normalize_usage(raw_usage=usage, provider="anthropic")
    assert result.logical_input == 12000 and result.cache_write == 2000
    assert result.complete


def test_sdk_ttl_sentinel_zero_is_replaced_by_split_not_added_twice():
    usage = lc(12000, 12100, write=0)
    usage["input_token_details"].update(
        ephemeral_5m_input_tokens=500, ephemeral_1h_input_tokens=1500
    )
    result = normalize_usage(usage_metadata=usage)
    assert result.logical_input == 12000 and result.cache_write == 2000
    assert result.complete


@pytest.mark.parametrize(
    "key", ["cache_read_input_tokens", "cache_creation_input_tokens"]
)
def test_unknown_exclusive_cache_is_incomplete(key, caplog):
    usage = {
        "input_tokens": 1000,
        "output_tokens": 100,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    usage[key] = None
    result = normalize_usage(raw_usage=usage)
    assert result.logical_input is None and not result.complete
    assert "incomplete" in caplog.text


def test_null_exclusive_creation_without_aggregate_is_unknown():
    result = normalize_usage(
        raw_usage={"input_tokens": 1000, "output_tokens": 100, "cache_creation": None}
    )
    assert not result.complete and result.logical_input is None


@pytest.mark.parametrize("total", [None, 0, 13000])
def test_cached_metadata_ambiguous_total_is_unknown(total, caplog):
    usage = lc(1000, total)
    result = normalize_usage(usage_metadata=usage)
    assert result.logical_input is None and not result.complete
    assert "incomplete" in caplog.text


def test_matching_native_evidence_can_resolve_sdk_total_artifact():
    metadata = lc(12000, 1100)  # SDK added cache to input, retained older raw total.
    raw = {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "total_tokens": 1100,
        "cache_read_input_tokens": 9000,
        "cache_write_input_tokens": 2000,
        "completion_tokens_details": {"reasoning_tokens": 40},
    }
    result = normalize_usage(metadata, raw, "bedrock")
    assert result.logical_input == 12000 and result.complete


def test_unknown_raw_counters_cannot_be_erased_by_sdk_zero():
    result = normalize_usage(
        lc(1000, 1100, 0, 0),
        {
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": 0,
        },
    )
    assert result.logical_input is None and not result.complete


def test_contradictory_observers_not_added_or_certified_complete():
    result = normalize_usage(
        lc(12000, 12100),
        {
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_read_input_tokens": 8000,
            "cache_creation_input_tokens": 2000,
        },
    )
    assert result.logical_input is None and not result.complete


@pytest.mark.parametrize("camel", [False, True])
def test_gemini_candidates_exclude_thoughts_but_prompt_includes_cache(camel):
    usage = (
        {
            "promptTokenCount": 1000,
            "candidatesTokenCount": 60,
            "thoughtsTokenCount": 40,
            "cachedContentTokenCount": 800,
            "totalTokenCount": 1100,
        }
        if camel
        else {
            "prompt_token_count": 1000,  # nosec B105 - token-count fixture, not a credential
            "candidates_token_count": 60,  # nosec B105 - token-count fixture, not a credential
            "thoughts_token_count": 40,  # nosec B105 - token-count fixture, not a credential
            "cached_content_token_count": 800,  # nosec B105 - token-count fixture, not a credential
            "total_token_count": 1100,  # nosec B105 - token-count fixture, not a credential
        }
    )
    result = normalize_usage(raw_usage=usage, provider="gemini")
    assert result == (1000, 100, 800, None, 40, "inclusive", True)


def test_openai_reasoning_already_in_completion_count():
    result = normalize_usage(
        raw_usage={
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "prompt_tokens_details": {"cached_tokens": 800},
            "completion_tokens_details": {"reasoning_tokens": 40},
        },
        provider="openai",
    )
    assert result == (1000, 100, 800, None, 40, "inclusive", True)


def test_openai_responses_inclusive_input_and_output():
    result = normalize_usage(
        raw_usage={
            "input_tokens": 1000,
            "output_tokens": 100,
            "total_tokens": 1100,
            "input_tokens_details": {"cached_tokens": 800},
            "output_tokens_details": {"reasoning_tokens": 40},
        },
        provider="openai",
    )
    assert result == (1000, 100, 800, None, 40, "inclusive", True)


def test_null_optional_openai_detail_objects_do_not_erase_inclusive_totals():
    result = normalize_usage(
        raw_usage={
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "prompt_tokens_details": None,
            "completion_tokens_details": None,
        },
        provider="openai",
    )
    assert result == (1000, 100, None, None, None, "inclusive", True)


@pytest.mark.parametrize("bad", [True, -1, 1.5, "1000"])
def test_invalid_aggregate_count_never_guessed(bad):
    assert not normalize_usage(
        raw_usage={"input_tokens": bad, "output_tokens": 100}
    ).complete


@pytest.mark.parametrize("hint", ["azure", "openai", "openrouter"])
def test_native_field_family_wins_over_provider_hint(hint):
    result = normalize_usage(
        raw_usage={
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_read_input_tokens": 9000,
            "cache_creation_input_tokens": 2000,
        },
        provider=hint,
    )
    assert (
        result.logical_input == 12000
        and result.complete
        and result.basis == "exclusive"
    )


def test_conflicting_native_field_families_are_unknown():
    result = normalize_usage(
        raw_usage={
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_read_input_tokens": 9000,
            "input_tokens_details": {"cached_tokens": 9000},
        },
        provider="azure",
    )
    assert result.logical_input is None and not result.complete


@pytest.mark.parametrize("total", [1100, None])
def test_native_converse_requires_full_logical_total(total):
    result = normalize_usage(
        raw_usage={
            "inputTokens": 1000,
            "outputTokens": 100,
            "cacheReadInputTokens": 9000,
            "cacheWriteInputTokens": 2000,
            "totalTokens": total,
        }
    )
    assert result.logical_input is None and not result.complete


def test_compatible_native_detail_counters_supplement_metadata():
    result = normalize_usage(
        {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100},
        {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "prompt_tokens_details": {"cached_tokens": 800},
            "completion_tokens_details": {"reasoning_tokens": 40},
        },
    )
    assert result == (1000, 100, 800, None, 40, "inclusive", True)


@pytest.mark.parametrize(
    "native_key", ["cache_read_input_tokens", "cache_write_input_tokens"]
)
@pytest.mark.parametrize(
    "details", [{"cached_tokens": 700}, {"cached_tokens": 800}, None]
)
def test_mixed_chatbedrock_and_openai_cache_families_are_unknown(
    native_key, details, caplog
):
    result = normalize_usage(
        raw_usage={
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            native_key: 800,
            "prompt_tokens_details": details,
        },
        provider="bedrock",
    )
    assert result.logical_input is None and result.output_incl_reasoning is None
    assert not result.complete and "incomplete" in caplog.text


@pytest.mark.parametrize("metadata_total", [110, 1])
def test_merged_native_cache_counters_cannot_exceed_logical_input(
    metadata_total, caplog
):
    result = normalize_usage(
        {
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": metadata_total,
            "input_token_details": {"cache_read": 80},
        },
        {"input_tokens": 20, "output_tokens": 10, "cache_creation_input_tokens": 80},
        provider="anthropic",
    )
    assert result.cache_read == 80 and result.cache_write == 80
    assert result.logical_input is None and result.output_incl_reasoning is None
    assert not result.complete and "incomplete" in caplog.text


@pytest.mark.parametrize("metadata_total", [110, 1])
@pytest.mark.parametrize("write", [20, 21])
def test_merged_cache_bounds_apply_to_both_observation_paths(metadata_total, write):
    result = normalize_usage(
        {
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": metadata_total,
            "input_token_details": {"cache_creation": write},
        },
        {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_tokens_details": {"cached_tokens": 80},
        },
        provider="openai",
    )
    assert result.cache_read == 80 and result.cache_write == write
    assert result.complete is (write == 20)
    assert result.logical_input == (100 if write == 20 else None)
    assert result.output_incl_reasoning == (10 if write == 20 else None)


def test_unknown_shape_warns_and_returns_no_logical_totals(caplog):
    result = normalize_usage(raw_usage={"some_other_units": 100})
    assert result.logical_input is result.output_incl_reasoning is None
    assert not result.complete and "incomplete" in caplog.text
