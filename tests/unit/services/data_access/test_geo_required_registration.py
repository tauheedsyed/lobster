"""Synthetic opt-in staging refuses before any registered count-data changes."""

import copy
import json
from unittest.mock import MagicMock, patch

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from lobster.core.runtime.data_manager import DataManagerV2
from lobster.core.schemas.download_queue import DownloadQueueEntry, DownloadStatus
from lobster.services.data_access.geo.import_requirements import ImportRequirementNotMet
from lobster.services.data_access.geo.parser import ParseResult
from lobster.services.data_access.geo_download_service import GEODownloadService
from lobster.services.data_access.geo_service import GEOService
from lobster.tools.download_orchestrator import DownloadOrchestrator

GSE = "GSE900000000001"
A, B = "GSM900000000001", "GSM900000000002"
FILE = "Synthetic_Gene_Counts%20v1%2Fpart.tsv"
URL = "https://example.test/" + FILE


@pytest.fixture
def setup_import(tmp_path):
    manager = DataManagerV2(workspace_path=tmp_path)
    metadata = {
        "title": "Synthetic series",
        "samples": {
            B: {"title": "Synthetic B", "characteristics_ch1": ["condition: second"]},
            A: {"title": "Synthetic A", "characteristics_ch1": ["condition: first"]},
        },
        "supplementary_file": [URL],
    }
    manager.metadata_store[GSE] = {"metadata": metadata, "strategy_config": {}}
    service = GEOService(manager)
    service._check_platform_compatibility = MagicMock(return_value=(True, "synthetic"))
    matrix = pd.DataFrame(
        [[1, 3], [2, 4], [5, 6]],
        index=["synthetic_a", "synthetic_b", "synthetic_c"],
        columns=[A, B],
    )
    service._download_and_parse_result = MagicMock(
        side_effect=lambda *_: ParseResult(matrix.copy(), original_tokens_valid=True)
    )
    manager.on_modality_loaded = MagicMock()
    manager.log_tool_usage = MagicMock(wraps=manager.log_tool_usage)
    manager.provenance.create_entity = MagicMock(wraps=manager.provenance.create_entity)
    return service, manager, metadata, matrix


def _run(service, **kwargs):
    kwargs.setdefault("adapter", "transcriptomics_bulk")
    return service.download_dataset(GSE, data_requirement="gene_counts", **kwargs)


def _assert_unregistered(manager):
    assert manager.list_modalities() == []
    manager.on_modality_loaded.assert_not_called()
    manager.log_tool_usage.assert_not_called()
    manager.provenance.create_entity.assert_not_called()


def test_staged_values_exact_basename_and_evidence_survive_actual_h5ad(setup_import):
    service, manager, metadata, matrix = setup_import
    message = _run(service)
    assert "whole-number counts (numeric check)" in message
    assert len(manager.list_modalities()) == 1
    adata = manager.get_modality(manager.list_modalities()[0])
    np.testing.assert_array_equal(adata.X, matrix.T.to_numpy())
    assert adata.X.dtype == np.float32
    assert list(adata.obs["source_sample_id"]) == [A, B]
    assert all(type(value) is str for value in adata.obs["source_sample_id"])
    assert list(adata.obs["condition"]) == ["first", "second"]
    evidence = adata.uns["geo_import"]
    assert evidence["accepted_file"] == FILE  # Never decode %20/%2F or use cache hash.
    assert evidence["data_requirement"] == "gene_counts"
    facts = json.loads(evidence["candidates_json"])
    assert facts[0]["file_name"] == FILE and facts[0]["complete_parse"] is True
    path = manager.data_dir / (manager.list_modalities()[0] + "_raw.h5ad")
    restored = anndata.read_h5ad(path)
    np.testing.assert_array_equal(restored.X, matrix.T.to_numpy())
    assert restored.uns["geo_import"]["accepted_file"] == FILE
    assert json.loads(restored.uns["geo_import"]["candidates_json"]) == facts
    assert list(restored.obs["source_sample_id"]) == [A, B]
    assert all(type(value) is str for value in restored.obs["source_sample_id"])
    manager.on_modality_loaded.assert_called_once()


@pytest.mark.parametrize("largest", [2**24 + 1, 2**24 + 2, 2**53 + 1, 2**63 - 1])
def test_high_count_refuses_without_registration_or_notification(setup_import, largest):
    service, manager, metadata, matrix = setup_import
    matrix.iloc[0, 0] = largest
    before = copy.deepcopy(metadata)
    with pytest.raises(ImportRequirementNotMet) as caught:
        _run(service)
    assert caught.value.problem == "value_fidelity"
    assert str(largest) in str(caught.value)
    assert caught.value.candidates[0].verdict == "rejected"
    _assert_unregistered(manager)
    assert metadata == before


@pytest.mark.parametrize("features", [1, 2, 3])
def test_small_feature_counts_bypass_only_active_legacy_column_heuristic(
    setup_import, features
):
    service, manager, metadata, matrix = setup_import
    matrix.drop(matrix.index[features:], inplace=True)
    _run(service)
    adata = manager.get_modality(manager.list_modalities()[0])
    assert adata.shape == (2, features)
    np.testing.assert_array_equal(adata.X, matrix.T.to_numpy())


def test_float32_exact_limit_is_allowed(setup_import):
    service, manager, metadata, matrix = setup_import
    matrix.iloc[0, 0] = 2**24
    _run(service)
    adata = manager.get_modality(manager.list_modalities()[0])
    np.testing.assert_array_equal(adata.X, matrix.T.to_numpy())


def test_titles_keep_plain_gsm_identity_and_shuffled_clinical_mapping(setup_import):
    service, manager, metadata, matrix = setup_import
    matrix.columns = ["Synthetic A", "Synthetic B"]
    _run(service)
    adata = manager.get_modality(manager.list_modalities()[0])
    assert list(adata.obs_names) == ["Synthetic A", "Synthetic B"]
    assert list(adata.obs["source_sample_id"]) == [A, B]
    assert list(adata.obs["condition"]) == ["first", "second"]
    assert adata.uns["geo_import"]["sample_rule"] == "title_exact"


@pytest.mark.parametrize(
    "key", ["sample id", "source sample id", "sample_id", "source_sample_id"]
)
def test_characteristics_never_overwrite_reserved_identity(setup_import, key):
    service, manager, metadata, matrix = setup_import
    metadata["samples"][A]["characteristics_ch1"].append(key + ": synthetic-other")
    _run(service)
    adata = manager.get_modality(manager.list_modalities()[0])
    assert list(adata.obs["source_sample_id"]) == [A, B]
    assert list(adata.obs["sample_id"]) == [A, B]


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "condition: first",
        [None],
        [3],
        ["malformed"],
        ["condition: first", "condition: conflicting"],
    ],
)
def test_malformed_provided_clinical_metadata_refuses_before_registration(
    setup_import, bad
):
    service, manager, metadata, matrix = setup_import
    metadata["samples"][A]["characteristics_ch1"] = bad
    with pytest.raises(ImportRequirementNotMet) as caught:
        _run(service)
    assert caught.value.problem == "identity"
    _assert_unregistered(manager)


def test_absent_characteristics_are_optional_and_missing_values_save_safely(
    setup_import,
):
    service, manager, metadata, matrix = setup_import
    metadata["samples"][A].pop("characteristics_ch1")
    _run(service)
    adata = manager.get_modality(manager.list_modalities()[0])
    assert list(adata.obs["condition"]) == ["", "second"]


@pytest.mark.parametrize("proven", [False, True])
def test_all_active_modality_caches_refuse_without_download_or_reimport(
    setup_import, proven
):
    service, manager, metadata, matrix = setup_import
    name = "geo_" + GSE.lower() + "_transcriptomics_bulk"
    cached = anndata.AnnData(np.ones((2, 3)))
    if proven:
        cached.uns["geo_import"] = {"data_requirement": "gene_counts"}
    manager.modalities[name] = cached
    with pytest.raises(ImportRequirementNotMet):
        _run(service)
    assert manager.modalities[name] is cached
    assert manager.list_modalities() == [name]
    service._download_and_parse_result.assert_not_called()
    manager.on_modality_loaded.assert_not_called()


@pytest.mark.parametrize("mutation", ["value", "shape", "obs", "var"])
def test_real_adapter_preprocessing_changes_refuse_before_registration(
    setup_import, mutation, monkeypatch
):
    service, manager, metadata, matrix = setup_import
    adapter = manager.adapters["transcriptomics_bulk"]
    original = adapter.preprocess_data

    def mutate(adata, **kwargs):
        result = original(adata, **kwargs)
        if mutation == "value":
            result.X[0, 0] += 1
        elif mutation == "shape":
            result = result[:, :2].copy()
        elif mutation == "obs":
            result.obs_names = ["synthetic_wrong", B]
        else:
            result.var_names = ["synthetic_wrong", "synthetic_b", "synthetic_c"]
        return result

    monkeypatch.setattr(adapter, "preprocess_data", mutate)
    with pytest.raises(ImportRequirementNotMet) as caught:
        _run(service)
    assert caught.value.problem == "value_fidelity"
    _assert_unregistered(manager)


def test_sparse_real_adapter_values_and_h5ad_remain_exact(setup_import):
    service, manager, metadata, _ = setup_import
    values = np.zeros((6000, 2), dtype=np.int64)
    values[:2] = [[1, 2], [3, 7]]
    matrix = pd.DataFrame(
        values, index=[f"synthetic_gene_{i}" for i in range(6000)], columns=[A, B]
    )
    service._download_and_parse_result.side_effect = lambda *_: ParseResult(
        matrix.copy(), original_tokens_valid=True
    )
    _run(service)
    name = manager.list_modalities()[0]
    adata = manager.get_modality(name)
    assert issparse(adata.X)
    np.testing.assert_array_equal(adata.X.toarray(), matrix.T.to_numpy())
    restored = anndata.read_h5ad(manager.data_dir / (name + "_raw.h5ad"))
    assert issparse(restored.X)
    np.testing.assert_array_equal(restored.X.toarray(), matrix.T.to_numpy())


def test_default_active_adapter_does_not_call_modality_inference(setup_import):
    service, manager, metadata, matrix = setup_import
    with patch(
        "lobster.agents.data_expert.assistant.DataExpertAssistant.detect_modality",
        side_effect=AssertionError("Unexpected inference"),
    ) as detect:
        _run(service, adapter=None)
    detect.assert_not_called()
    assert len(manager.list_modalities()) == 1


def test_all_per_file_facts_with_nulls_survive_h5ad(setup_import):
    service, manager, metadata, matrix = setup_import
    metadata["supplementary_file"].extend(
        [
            "https://example.test/synthetic_transcript.tsv",
            "https://example.test/synthetic_metadata.csv",
        ]
    )
    _run(service)
    name = manager.list_modalities()[0]
    restored = anndata.read_h5ad(manager.data_dir / (name + "_raw.h5ad"))
    facts = json.loads(restored.uns["geo_import"]["candidates_json"])
    assert len(facts) == 3
    skipped = [fact for fact in facts if fact["verdict"] == "rejected"]
    assert len(skipped) == 2
    assert all(
        fact["complete_parse"] is None and fact["whole_numbers"] is None
        for fact in skipped
    )
    service._download_and_parse_result.assert_called_once_with(URL, GSE)


@pytest.mark.parametrize(
    "initial_status", [DownloadStatus.PENDING, DownloadStatus.FAILED]
)
def test_real_queue_wrapper_value_fidelity_refusal_is_failed_and_reclaimable(
    setup_import, initial_status
):
    service, manager, metadata, matrix = setup_import
    matrix.iloc[0, 0] = 2**24 + 1
    entry = DownloadQueueEntry(
        entry_id="synthetic_required_entry",
        dataset_id=GSE,
        database="geo",
        status=initial_status,
    )
    manager.download_queue.add_entry(entry)
    wrapper = GEODownloadService(manager)
    wrapper.geo_service = service
    with patch.object(DownloadOrchestrator, "_register_default_services"):
        orchestrator = DownloadOrchestrator(manager)
    orchestrator.register_service(wrapper)
    strategy = {"strategy_params": {"data_requirement": "gene_counts"}}
    for attempt in range(2):
        with pytest.raises(ImportRequirementNotMet) as caught:
            orchestrator.execute_download(entry.entry_id, strategy_override=strategy)
        assert caught.value.problem == "value_fidelity"
        current = manager.download_queue.get_entry(entry.entry_id)
        assert current.status == DownloadStatus.FAILED
        assert len(current.error_log) == attempt + 1
        _assert_unregistered(manager)


def test_incompatible_explicit_adapter_refuses_before_acquisition(setup_import):
    service, manager, metadata, matrix = setup_import
    with pytest.raises(ImportRequirementNotMet):
        _run(service, adapter="transcriptomics_single_cell")
    service._download_and_parse_result.assert_not_called()
    _assert_unregistered(manager)
