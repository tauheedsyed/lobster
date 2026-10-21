"""Host-resolved model intent verification, with no configuration resolution."""

from dataclasses import FrozenInstanceError

import pytest

from lobster.config.inference_context import (
    InferenceContext,
    InferenceModelSelection,
    resolve_inference_model,
)
from lobster.core.exceptions import InferenceRefused

pytestmark = pytest.mark.no_auto_config


@pytest.fixture
def context():
    return InferenceContext(
        "bedrock",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "synthetic-binding",
        "version-a",
        200000,
        4096,
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_resolution_is_the_exact_host_pair(context, explicit):
    selection = (
        InferenceModelSelection(context.provider, context.model_id)
        if explicit
        else None
    )
    assert resolve_inference_model(context, selection) == (
        context.provider,
        context.model_id,
    )


def test_explicit_selection_is_immutable(context):
    selection = InferenceModelSelection(context.provider, context.model_id)
    with pytest.raises(FrozenInstanceError):
        selection.model_id = "different"
    assert not hasattr(selection, "__dict__")


@pytest.mark.parametrize("field", ["provider", "model_id"])
@pytest.mark.parametrize("value", [None, 3, "", " ", " padded", "padded ", "x" * 513])
def test_selection_rejects_invalid_identity_fields(context, field, value):
    options = {"provider": context.provider, "model_id": context.model_id, field: value}
    with pytest.raises(InferenceRefused) as error:
        InferenceModelSelection(**options)
    assert error.value.category == "context"


@pytest.mark.parametrize(
    "selection",
    [
        InferenceModelSelection("anthropic", "claude-sonnet-4-5-20250929"),
        InferenceModelSelection("bedrock", "claude-sonnet-4-5-20250929"),
        {
            "provider": "bedrock",
            "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        },
        "inherit",
        False,
    ],
)
def test_no_alias_dict_or_provider_conversion(context, selection):
    with pytest.raises(InferenceRefused) as error:
        resolve_inference_model(context, selection)
    assert error.value.category == "context"


def test_resolution_requires_a_supplied_context():
    with pytest.raises(InferenceRefused):
        resolve_inference_model(None)
