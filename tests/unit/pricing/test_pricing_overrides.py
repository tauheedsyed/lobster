"""Validation of ``overrides.json``: every manual rate must be traceable.

An override needs an https ``source_url``, a real ``checked_at`` date and a named
``reviewer``; it may only target a supported market provider by its engine id.
"""

from __future__ import annotations

import pytest

from lobster.pricing import PriceBook, PricingDataError, schema

pytestmark = pytest.mark.no_auto_config


def test_empty_and_absent_overrides_are_valid(overrides_document):
    assert schema.validate_overrides(overrides_document()) == ()
    assert schema.validate_overrides(None) == ()


def test_valid_entry_round_trips_every_field():
    document = schema.loads_json(
        '{"schema_version": 1, "description": "notes", "overrides": [{'
        '"provider": "gemini", "model_id": "acme-flash", '
        '"cost": {"input": 0.25, "output": 1.5, "cache_read": null}, '
        '"source_url": "https://example.invalid/pricing", '
        '"checked_at": "2026-10-20", "reviewer": "reviewer-handle", '
        '"note": "not in models.dev yet"}]}'
    )

    (entry,) = schema.validate_overrides(document)

    assert (entry.provider, entry.model_id) == ("gemini", "acme-flash")
    assert entry.source_url == "https://example.invalid/pricing"
    assert (entry.checked_at, entry.reviewer) == ("2026-10-20", "reviewer-handle")
    assert entry.note == "not in models.dev yet"
    assert entry.cost["cache_read"] is None
    assert str(entry.cost["input"]) == "0.25"  # parsed exactly


@pytest.mark.parametrize(
    "field",
    ["provider", "model_id", "cost", "source_url", "checked_at", "reviewer"],
)
def test_each_required_field_is_required(overrides_document, override_entry, field):
    entry = override_entry()
    del entry[field]

    with pytest.raises(PricingDataError, match=f"missing required field.*{field}"):
        schema.validate_overrides(overrides_document(entry))


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"provider": "omics-os"}, "not a market provider"),
        ({"provider": "ollama"}, "not a market provider"),
        ({"provider": "google"}, "not a supported engine provider id"),  # alias
        ({"provider": "amazon-bedrock"}, "not a supported engine provider id"),
        ({"provider": "Bedrock"}, "not a supported engine provider id"),
        ({"provider": "somewhere-else"}, "not a supported engine provider id"),
        ({"provider": 3}, "provider: must be a string"),
        ({"model_id": ""}, "model_id"),
        ({"model_id": " padded"}, "model_id"),
        ({"model_id": None}, "model_id"),
        ({"source_url": "http://example.invalid/pricing"}, "https"),
        ({"source_url": "https://"}, "https"),
        ({"source_url": "ftp://example.invalid/x"}, "https"),
        ({"source_url": "javascript:alert(1)"}, "https"),
        ({"source_url": "https://exa mple.invalid/x"}, "https"),
        ({"source_url": "https://[::1/x"}, "https"),
        ({"source_url": ""}, "source_url"),
        ({"source_url": None}, "source_url"),
        ({"checked_at": "2026-13-01"}, "not a real date"),
        ({"checked_at": "2026-02-30"}, "not a real date"),
        ({"checked_at": "10/20/2026"}, "YYYY-MM-DD"),
        ({"checked_at": "20261020"}, "YYYY-MM-DD"),
        ({"checked_at": "2026-10-20T00:00:00Z"}, "YYYY-MM-DD"),
        ({"checked_at": ""}, "YYYY-MM-DD"),
        ({"checked_at": None}, "YYYY-MM-DD"),
        ({"reviewer": ""}, "reviewer"),
        ({"reviewer": "   "}, "reviewer"),
        ({"reviewer": " someone"}, "reviewer"),
        ({"reviewer": None}, "reviewer"),
        ({"note": ""}, "note"),
        ({"unexpected": 1}, "unknown field"),
        ({"cost": {}}, "non-empty object"),
        ({"cost": []}, "non-empty object"),
        ({"cost": {"input": "7"}}, "expected a number or null"),
        ({"cost": {"input": True}}, "expected a number or null"),
        ({"cost": {"output": -1}}, "outside 0"),
        ({"cost": {"output": 10_000_000}}, "outside 0"),
        ({"cost": {"input": 1, "tiers": "none"}}, "tiers: must be a list"),
        ({"cost": {"input": 1, "tiers": [3]}}, r"tiers\[0\]: must be an object"),
        ({"cost": {"input": 1, "tiers": [{"output": -2}]}}, r"tiers\[0\]\.output"),
        (
            {"cost": {"input": 1, "context_over_200k": 7}},
            "context_over_200k: must be an object",
        ),
    ],
)
def test_invalid_entries_are_rejected(
    overrides_document, override_entry, changes, match
):
    with pytest.raises(PricingDataError, match=match):
        schema.validate_overrides(overrides_document(override_entry(**changes)))


def test_duplicate_provider_model_pairs_are_rejected(
    overrides_document, override_entry
):
    first = override_entry()
    second = override_entry(cost={"input": 9, "output": 9})

    with pytest.raises(PricingDataError, match="duplicate entry for bedrock/"):
        schema.validate_overrides(overrides_document(first, second))
    # The same model id under another provider is a different key.
    other = override_entry(provider="openrouter")
    assert len(schema.validate_overrides(overrides_document(first, other))) == 2


@pytest.mark.parametrize(
    "document,match",
    [
        ([], "must be an object"),
        ({"schema_version": 2, "overrides": []}, "schema_version"),
        ({"overrides": []}, "schema_version"),
        ({"schema_version": 1}, "must be a list"),
        ({"schema_version": 1, "overrides": {}}, "must be a list"),
        ({"schema_version": 1, "overrides": ["x"]}, "must be an object"),
        ({"schema_version": 1, "overrides": [], "extra": 1}, "unknown field"),
    ],
)
def test_invalid_documents_are_rejected(document, match):
    with pytest.raises(PricingDataError, match=match):
        schema.validate_overrides(document)


def test_book_refuses_to_load_with_an_invalid_override(
    make_book, overrides_document, override_entry
):
    with pytest.raises(PricingDataError, match="reviewer"):
        make_book(
            {"openai": {"m": {"input": 1}}}, overrides=[override_entry(reviewer="")]
        )
    with pytest.raises(PricingDataError, match="unknown field"):
        PriceBook.from_documents(
            {
                "schema_version": 1,
                "snapshot_id": "s",
                "fetched_at": "f",
                "source": {"url": "u", "sha256": "h"},
                "providers": {},
            },
            {"schema_version": 1, "overrides": [], "unexpected": True},
        )


def test_loader_rejects_nan_and_duplicate_keys():
    for text in ('{"a": NaN}', '{"a": Infinity}', '{"a": 1, "a": 2}', "{", ""):
        with pytest.raises(PricingDataError):
            schema.loads_json(text)
