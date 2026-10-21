"""Strict supplied binding resolution without registry or ambient credentials."""

import os
from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from lobster.config.inference_context import (
    INFERENCE_BINDING_VERSION,
    InferenceContext,
    validate_inference_binding,
)
from lobster.config.llm_factory import create_llm
from lobster.core.exceptions import InferenceRefused

pytestmark = pytest.mark.no_auto_config


class ScopedModel(BaseChatModel):
    """In-memory transport for resolver tests, not a copied resolution branch."""

    model_id: str
    requests: list = []

    @property
    def _llm_type(self):
        return "test-scoped-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append((self.model_id, messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


class Factory:
    capability_version = INFERENCE_BINDING_VERSION

    def __init__(self):
        self.contexts = []

    def create(self, *, component, context):
        self.contexts.append((component, context))
        return ScopedModel(model_id=context.model_id, cache=False)


def context(**overrides):
    fields = dict(
        provider="example",
        model_id="model-a",
        binding_id="opaque-binding",
        binding_version="generation-a",
        context_window=200_000,
        max_output_tokens=4096,
        generation_options=(("temperature", 0.5),),
    )
    fields.update(overrides)
    return InferenceContext(**fields)


@pytest.fixture(autouse=True)
def no_ambient_resolution(monkeypatch):
    from lobster.config import providers
    from lobster.core.config_resolver import ConfigResolver

    def refuse(*args, **kwargs):
        raise AssertionError("Strict binding accessed ambient resolution")

    monkeypatch.setattr(ConfigResolver, "get_instance", refuse)
    monkeypatch.setattr(providers, "get_provider", refuse)


def test_strict_factory_receives_exact_context_without_ambient_reads():
    supplied = context()
    factory = Factory()
    before = dict(os.environ)
    model = create_llm(
        "specialist",
        {"temperature": 1.0, "max_tokens": 999},
        inference_context=supplied,
        model_factory=factory,
    )
    assert model.invoke("hello").content == "ok"
    assert factory.contexts == [("specialist", supplied)]
    assert model.model_id == supplied.model_id
    assert model.cache is False
    assert dict(os.environ) == before


def test_context_is_frozen_and_nonsecret_serializable():
    supplied = context(
        generation_options=(
            ("thinking", (("type", "enabled"), ("budget_tokens", 1024))),
        )
    )
    with pytest.raises(FrozenInstanceError):
        supplied.model_id = "other"
    assert asdict(supplied)["binding_version"] == "generation-a"


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", ""),
        ("model_id", " "),
        ("binding_id", None),
        ("binding_version", "x" * 513),
        ("context_window", 0),
        ("context_window", True),
        ("max_output_tokens", -1),
        ("generation_options", []),
        ("generation_options", (("temperature", float("nan")),)),
        ("generation_options", (("temperature", float("inf")),)),
        ("generation_options", (("thinking", {"type": "enabled"}),)),
        ("generation_options", (("thinking", ["enabled"]),)),
        ("generation_options", (("api_key", "synthetic-secret"),)),
        (
            "generation_options",
            (("additional_model_request_fields", (("api_key", "synthetic-secret"),)),),
        ),
        (
            "generation_options",
            (("additional_model_request_fields", ("api_key", "synthetic-secret")),),
        ),
        ("generation_options", (("temperature", 1.0), ("temperature", 0.5))),
        ("generation_options", (("temperature", object()),)),
    ],
)
def test_invalid_context_is_refused(field, value):
    with pytest.raises(InferenceRefused) as caught:
        context(**{field: value})
    assert caught.value.category == "context"


@pytest.mark.parametrize(
    "provider,model", [("", None), ("other", None), (None, ""), (None, "other")]
)
def test_explicit_identity_conflict_refuses(provider, model):
    factory = Factory()
    with pytest.raises(InferenceRefused):
        create_llm(
            "agent",
            {},
            provider_override=provider,
            model_override=model,
            inference_context=context(),
            model_factory=factory,
        )
    assert factory.contexts == []


@pytest.mark.parametrize(
    "params",
    [
        {"api_key": "synthetic-secret"},
        {"additional_model_request_fields": {"api_key": "synthetic-secret"}},
        {"additional_model_request_fields": {"client": object()}},
        {"model_id": "other"},
        {"model_id": ""},
        {"headers": {}},
    ],
)
def test_legacy_key_transport_or_model_injection_refuses(params):
    factory = Factory()
    with pytest.raises(InferenceRefused):
        create_llm("agent", params, inference_context=context(), model_factory=factory)
    assert factory.contexts == []


@pytest.mark.parametrize(
    "pair", ["context_only", "factory_only", "bad_context", "wrong_version"]
)
def test_partial_or_unsupported_binding_refuses(pair):
    ctx, factory = context(), Factory()
    if pair == "context_only":
        factory = None
    elif pair == "factory_only":
        ctx = None
    elif pair == "bad_context":
        ctx = {}
    else:
        factory.capability_version = "unsupported"
    with pytest.raises(InferenceRefused):
        create_llm("agent", {}, inference_context=ctx, model_factory=factory)


def test_absent_pair_keeps_local_mode():
    assert validate_inference_binding(None, None) is False


@pytest.mark.parametrize(
    "bad_model",
    [
        None,
        object(),
        ScopedModel(model_id="model-a", cache=None),
        ScopedModel(model_id="model-a", cache=True),
    ],
)
def test_factory_cannot_return_nonmodel_or_response_cached_model(bad_model):
    factory = SimpleNamespace(
        capability_version=INFERENCE_BINDING_VERSION, create=lambda **kwargs: bad_model
    )
    with pytest.raises(InferenceRefused) as caught:
        create_llm("agent", {}, inference_context=context(), model_factory=factory)
    assert caught.value.category == "context"


def test_factory_refusal_survives_unchanged():
    refusal = InferenceRefused("budget")

    def create(**kwargs):
        raise refusal

    factory = SimpleNamespace(
        capability_version=INFERENCE_BINDING_VERSION, create=create
    )
    with pytest.raises(InferenceRefused) as caught:
        create_llm("agent", {}, inference_context=context(), model_factory=factory)
    assert caught.value is refusal
