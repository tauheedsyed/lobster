"""Typed inference refusals survive scientific fallbacks and graph creation."""

import socket
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip(
    "langchain_aws", reason="Native refusal tests require the Bedrock provider extra"
)

from lobster.config.providers.bedrock_builder import build_bedrock_converse
from lobster.config.providers.gateway_bedrock_stub import StubBedrockControlClient
from lobster.core.exceptions import InferenceRefused

CATEGORIES = ("context", "auth", "policy", "budget", "provider", "accounting")
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


class RefusingRuntime:
    def __init__(self, refusal):
        self.refusal = refusal
        self.calls = 0

    def converse(self, **request):
        self.calls += 1
        raise self.refusal


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Unexpected network access")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def refusing_model(refusal):
    runtime = RefusingRuntime(refusal)
    model = build_bedrock_converse(
        model_id=MODEL_ID,
        client=runtime,
        bedrock_client=StubBedrockControlClient(),
        cache=False,
    )
    return model, runtime


@pytest.mark.parametrize("category", CATEGORIES)
@pytest.mark.parametrize("operation", ("detect_modality", "extract_strategy_config"))
def test_data_helper_refusal_does_not_return_none(category, operation):
    from lobster.agents.data_expert.assistant import DataExpertAssistant

    refusal = InferenceRefused(category)
    model, runtime = refusing_model(refusal)
    helper = DataExpertAssistant()
    helper._llm = model
    with pytest.raises(InferenceRefused) as caught:
        getattr(helper, operation)({}, "GSE1")
    assert caught.value is refusal
    assert runtime.calls == 1


@pytest.mark.parametrize("category", CATEGORIES)
def test_metadata_refusal_does_not_return_manual_check(category):
    from lobster.services.metadata.metadata_validation_service import (
        MetadataValidationService,
    )

    refusal = InferenceRefused(category)
    model, runtime = refusing_model(refusal)
    helper = MetadataValidationService()
    helper._llm = model
    with pytest.raises(InferenceRefused) as caught:
        helper.validate_dataset_metadata(
            {"samples": {"GSM1": {"characteristics_ch1": ["disease: healthy"]}}},
            "GSE1",
            ["disease"],
        )
    assert caught.value is refusal
    assert runtime.calls == 1


@pytest.mark.parametrize("category", CATEGORIES)
def test_factory_refusal_does_not_prune_specialist(category, monkeypatch):
    from lobster.agents import graph

    refusal = InferenceRefused(category)

    def factory(data_manager, callback_handler, agent_name):
        raise refusal

    monkeypatch.setattr(graph, "import_agent_factory", lambda _: factory)
    config = SimpleNamespace(
        name="example",
        display_name="Example",
        child_agents=[],
        factory_function="example.factory",
    )
    with pytest.raises(InferenceRefused) as caught:
        graph._create_agents_single_pass(
            {"example": config}, {}, None, None, None, "free", None, None, None
        )
    assert caught.value is refusal


def test_optional_factory_failure_still_prunes(monkeypatch):
    from lobster.agents import graph

    def factory(data_manager, callback_handler, agent_name):
        raise RuntimeError("Unavailable optional capability")

    monkeypatch.setattr(graph, "import_agent_factory", lambda _: factory)
    config = SimpleNamespace(
        name="example",
        display_name="Example",
        child_agents=[],
        factory_function="example.factory",
    )
    assert (
        graph._create_agents_single_pass(
            {"example": config}, {}, None, None, None, "free", None, None, None
        )
        == {}
    )


def test_refusal_exposes_only_fixed_safe_message():
    refusal = InferenceRefused("auth", retryable=True)
    assert str(refusal) == "Inference refused (auth)."
    assert refusal.category == "auth"
    assert refusal.retryable is True
    with pytest.raises(ValueError, match="Unknown inference refusal category"):
        InferenceRefused("unknown-secret-value")


@pytest.mark.parametrize("category", CATEGORIES)
def test_queue_strategy_refusal_does_not_use_heuristics(category, tmp_path):
    from lobster.agents.data_expert.assistant import DataExpertAssistant
    from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer

    refusal = InferenceRefused(category)
    model, runtime = refusing_model(refusal)
    manager = SimpleNamespace(workspace_path=tmp_path, metadata_store={})
    helper = DataExpertAssistant(data_manager=manager)
    helper._llm = model
    preparer = GEOQueuePreparer(manager)
    preparer._data_expert = helper
    with pytest.raises(InferenceRefused) as caught:
        preparer.recommend_strategy({}, None, "GSE1")
    assert caught.value is refusal
    assert runtime.calls == 1


@pytest.mark.parametrize("category", CATEGORIES)
def test_content_validation_refusal_is_not_an_error_string(
    category, monkeypatch, tmp_path
):
    from lobster.services.data_access.content_access_service import ContentAccessService
    from lobster.services.metadata.metadata_validation_service import (
        MetadataValidationService,
    )

    refusal = InferenceRefused(category)
    model, runtime = refusing_model(refusal)
    monkeypatch.setattr(MetadataValidationService, "llm", property(lambda self: model))
    manager = SimpleNamespace(
        workspace_path=tmp_path, _get_geo_metadata=lambda _: {"n_samples": 1}
    )
    service = ContentAccessService.__new__(ContentAccessService)
    service.data_manager = manager
    with pytest.raises(InferenceRefused) as caught:
        service.validate_metadata("GSE1", required_fields=["disease"])
    assert caught.value is refusal
    assert runtime.calls == 1


@pytest.mark.parametrize("operation", ("multiple", "single"))
@pytest.mark.parametrize("category", CATEGORIES)
def test_sample_storage_refusal_does_not_guess_a_modality(
    category, operation, monkeypatch, tmp_path
):
    from lobster.agents.data_expert.assistant import DataExpertAssistant
    from lobster.services.data_access.geo.concatenation import SampleConcatenator

    refusal = InferenceRefused(category)
    model, runtime = refusing_model(refusal)
    monkeypatch.setattr(DataExpertAssistant, "llm", property(lambda self: model))
    manager = SimpleNamespace(workspace_path=tmp_path, metadata_store={})
    concatenator = SampleConcatenator(SimpleNamespace(data_manager=manager))
    matrix = pd.DataFrame([[1, 2]])
    with pytest.raises(InferenceRefused) as caught:
        if operation == "multiple":
            concatenator._store_samples_as_anndata({"GSM1": matrix}, "GSE1", {})
        else:
            concatenator._store_single_sample_as_modality(
                "GSM1", matrix, SimpleNamespace(metadata={})
            )
    assert caught.value is refusal
    assert runtime.calls == 1


@pytest.mark.parametrize("category", CATEGORIES)
def test_disease_extraction_refusal_does_not_become_unknown(category):
    from lobster.agents.metadata_assistant.config import extract_disease_with_llm

    refusal = InferenceRefused(category)
    model, runtime = refusing_model(refusal)
    with pytest.raises(InferenceRefused) as caught:
        extract_disease_with_llm(model, "Study of healthy samples", None, "Example", 1)
    assert caught.value is refusal
    assert runtime.calls == 1


@pytest.mark.parametrize("category", CATEGORIES)
def test_structured_ui_helper_refusal_does_not_become_text_input(category):
    from lobster.services.interaction.component_mapper import map_question

    refusal = InferenceRefused(category)
    model, runtime = refusing_model(refusal)
    with pytest.raises(InferenceRefused) as caught:
        map_question("Choose a visualization", llm=model)
    assert caught.value is refusal
    assert runtime.calls == 1


def test_data_helper_uses_owning_workspace(monkeypatch, tmp_path):
    from lobster.agents.data_expert import assistant
    from lobster.core import workspace

    model, _ = refusing_model(InferenceRefused("policy"))
    requests = []

    def create(agent_name, params, *, workspace_path):
        requests.append((agent_name, workspace_path))
        return model

    def no_ambient_workspace(**kwargs):
        raise AssertionError("Unexpected ambient workspace lookup")

    monkeypatch.setattr(assistant, "create_llm", create)
    monkeypatch.setattr(workspace, "resolve_workspace", no_ambient_workspace)
    helper = assistant.DataExpertAssistant(
        data_manager=SimpleNamespace(workspace_path=tmp_path)
    )
    assert helper.llm is model
    assert helper.llm is model
    assert requests == [("data_expert_assistant", tmp_path)]
