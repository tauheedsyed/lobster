"""Tests for the HITL component mapper: context -> decision fields."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from lobster.services.interaction.component_mapper import map_question
from lobster.services.interaction.component_schemas import AskUserContext


def _only_field(result):
    assert isinstance(result, AskUserContext)
    assert len(result.fields) == 1
    return result.fields[0]


# --- Fields given directly ---


def test_fields_context_is_validated_and_kept():
    context = {
        "fields": [
            {"key": "source", "type": "choice", "options": ["SRC-A", "SRC-B"]},
        ]
    }
    field = _only_field(map_question("Which source?", context=context))
    assert field.key == "source"
    assert [option.id for option in field.options] == ["SRC-A", "SRC-B"]


def test_invalid_fields_context_raises_for_the_caller():
    with pytest.raises(ValidationError, match="duplicate ids"):
        map_question(
            "Which source?",
            context={"fields": [{"key": "s", "type": "choice", "options": ["A", "A"]}]},
        )


def test_fields_context_skips_the_llm():
    llm = MagicMock()
    map_question("Q?", context={"fields": [{"key": "a", "type": "text"}]}, llm=llm)
    llm.with_structured_output.assert_not_called()


# --- Hint-style context ---


def test_options_list_maps_to_choice():
    field = _only_field(
        map_question(
            "Which normalization method?",
            context={"options": ["DESeq2", "CPM", "TMM"]},
        )
    )
    assert field.type == "choice"
    assert [option.id for option in field.options] == ["DESeq2", "CPM", "TMM"]


def test_dict_options_keep_their_ids():
    """Stable ids survive; the old mapper flattened them to labels."""
    field = _only_field(
        map_question(
            "Which donor?",
            context={
                "options": [
                    {"id": "S01", "label": "Donor A"},
                    {"id": "S05", "label": "Donor A"},
                ]
            },
        )
    )
    assert [(o.id, o.label) for o in field.options] == [
        ("S01", "Donor A"),
        ("S05", "Donor A"),
    ]


def test_clusters_map_to_text_per_cluster():
    clusters = [
        {"id": 0, "size": 500, "markers": ["CD3D", "CD8A"]},
        {"id": 1, "size": 200, "markers": ["MS4A1", "CD79A"]},
    ]
    field = _only_field(
        map_question("Annotate clusters", context={"clusters": clusters})
    )
    assert field.type == "text"
    assert [item.id for item in field.for_each] == ["0", "1"]
    assert field.for_each[0].details == "500 cells · CD3D, CD8A"


def test_threshold_context_maps_to_slider():
    field = _only_field(
        map_question(
            "Set p-value threshold",
            context={"min": 0.001, "max": 0.1, "default": 0.05, "unit": "p-value"},
        )
    )
    assert (field.type, field.view) == ("number", "slider")
    assert (field.min, field.max, field.default, field.unit) == (
        0.001,
        0.1,
        0.05,
        "p-value",
    )


def test_threshold_default_out_of_range_is_dropped():
    field = _only_field(map_question("Set cutoff", context={"min": 0, "max": 0.1}))
    assert field.default is None


def test_threshold_keyword_in_question_maps_to_slider():
    field = _only_field(map_question("What log2FC cutoff should I use?"))
    assert field.view == "slider"


def test_confirm_pattern_maps_to_yes_no_choice():
    field = _only_field(map_question("Do you want to proceed with batch correction?"))
    assert [option.id for option in field.options] == ["yes", "no"]


def test_confirm_default_propagated():
    field = _only_field(map_question("Continue?", context={"default": True}))
    assert field.default == {"id": "yes"}


def test_empty_options_falls_through():
    field = _only_field(map_question("Choose one", context={"options": []}))
    assert field.type == "text"


def test_fallback_is_free_text():
    field = _only_field(map_question("What cell types are you expecting?"))
    assert (field.type, field.question) == (
        "text",
        "What cell types are you expecting?",
    )


# --- LLM path ---


def _mock_llm(result=None, error=None):
    llm = MagicMock()
    structured = llm.with_structured_output.return_value
    if error is not None:
        structured.invoke.side_effect = error
    else:
        structured.invoke.return_value = result
    return llm


def test_rules_win_over_llm():
    llm = _mock_llm()
    map_question("Which?", context={"options": ["A", "B"]}, llm=llm)
    map_question("Set cutoff", context={"min": 0.0, "max": 1.0}, llm=llm)
    llm.with_structured_output.assert_not_called()


def test_ambiguous_question_uses_llm():
    formatted = AskUserContext(
        fields=[{"key": "strategy", "type": "choice", "options": ["drop", "impute"]}]
    )
    llm = _mock_llm(result=formatted)
    result = map_question("How should we handle the missing values?", llm=llm)
    assert result is formatted
    llm.with_structured_output.assert_called_once_with(AskUserContext)
    prompt = llm.with_structured_output.return_value.invoke.call_args[0][0]
    assert "How should we handle the missing values?" in prompt


def test_llm_error_falls_back_to_free_text():
    llm = _mock_llm(error=RuntimeError("LLM unavailable"))
    field = _only_field(map_question("Some ambiguous question here", llm=llm))
    assert (field.type, field.question) == ("text", "Some ambiguous question here")


def test_ambiguous_question_without_llm_is_free_text():
    field = _only_field(map_question("How should we handle the missing values?"))
    assert field.type == "text"
