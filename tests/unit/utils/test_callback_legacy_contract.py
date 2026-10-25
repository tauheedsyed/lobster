"""Exact pre-snapshot contract: real price catalog and every pre-existing field.

The golden corpus was generated from the unchanged pre-snapshot implementation,
not approximated or recomputed with custom prices. New observational fields are
intentionally projected out; every original field must remain byte/value exact.
"""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import lobster.utils.callbacks as callbacks
from lobster.config.settings import MODEL_PRICING

pytestmark = pytest.mark.no_auto_config


def project(new, old):
    if isinstance(old, dict):
        return {key: project(new[key], value) for key, value in old.items()}
    if isinstance(old, list):
        assert len(new) == len(old)
        return [project(a, b) for a, b in zip(new, old)]
    return new


def test_exact_real_price_catalog_and_all_legacy_fields(monkeypatch):
    contract = json.loads(
        Path(__file__).with_name("legacy_callback_contract.json").read_text()
    )
    assert MODEL_PRICING == contract["MODEL_PRICING"]
    model = contract["model"]

    class FixedClock:
        @staticmethod
        def now():
            return datetime(2026, 1, 1)

    monkeypatch.setattr(callbacks, "datetime", FixedClock)
    tracker = callbacks.TokenTrackingCallback("same", MODEL_PRICING)
    tracker._valid_agents = {"supervisor"}

    def raw(usage, key="usage", name=model):
        return SimpleNamespace(
            generations=[[]], llm_output={key: usage, "model_name": name}
        )

    def metadata(usage):
        message = SimpleNamespace(
            usage_metadata=usage, response_metadata={"model_name": model}
        )
        return SimpleNamespace(
            generations=[[SimpleNamespace(message=message)]], llm_output={}
        )

    cases = [
        raw({"input_tokens": 1000, "output_tokens": 100}),
        raw(
            {
                "input_tokens": 1000,
                "output_tokens": 100,
                "cache_read_input_tokens": 9000,
                "cache_creation_input_tokens": 2000,
            }
        ),
        metadata(
            {
                "input_tokens": 1000,
                "output_tokens": 100,
                "total_tokens": 12100,
                "input_token_details": {"cache_read": 9000, "cache_creation": 2000},
                "output_token_details": {"reasoning": 40},
            }
        ),
        metadata(
            {
                "input_tokens": 12000,
                "output_tokens": 100,
                "total_tokens": 12100,
                "input_token_details": {"cache_read": 9000, "cache_creation": 2000},
                "output_token_details": {"reasoning": 40},
            }
        ),
        raw(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1100,
                "prompt_tokens_details": None,
                "completion_tokens_details": None,
            },
            key="token_usage",
        ),
        raw(
            {
                "input_tokens": 1000,
                "output_tokens": 100,
                "cache_read_input_tokens": None,
                "cache_creation_input_tokens": 0,
            }
        ),
        raw({"input_tokens": 1000, "output_tokens": 100}, name="unpriced-model"),
        SimpleNamespace(generations=[[]], llm_output={}),
    ]

    def assert_snapshot(expected):
        assert (
            project(tracker.get_usage_summary(), expected["summary"])
            == expected["summary"]
        )
        assert tracker.get_latest_cost() == expected["latest"]
        assert tracker.get_minimal_summary() == expected["minimal"]
        assert tracker.get_verbose_summary() == expected["verbose"]

    for index, result in enumerate(cases):
        tracker.on_chat_model_start(
            {},
            [],
            run_id=f"r{index}",
            run_name="supervisor",
            metadata={"provider": "bedrock"},
        )
        tracker.on_llm_end(result, run_id=f"r{index}")
        assert_snapshot(contract["snapshots"][index])
    tracker.on_llm_error(RuntimeError("mock"), run_id="failed")
    tracker.on_llm_end(cases[0], run_id="duplicate")
    tracker.on_llm_end(cases[0], run_id="duplicate")
    assert_snapshot(contract["after_error_duplicate"])
