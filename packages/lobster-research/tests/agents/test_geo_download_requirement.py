"""Opt-in GEO requirement contract at the real model-visible queue tool."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pytest

# GEO download imports geo/loaders/tenx.py -> scanpy (optional extra). Without it
# the package-only CI env cannot even collect this module; the full env runs it.
pytest.importorskip("scanpy")

from lobster.agents.data_expert.data_expert import data_expert
from lobster.core.queues.download_queue import DownloadQueue
from lobster.core.schemas.download_queue import (
    DownloadQueueEntry,
    DownloadStatus,
    ValidationStatus,
)
from lobster.services.data_access.geo.import_requirements import (
    CandidateEvidence,
    ImportRequirement,
    ImportRequirementNotMet,
)
from lobster.services.data_access.geo_download_service import GEODownloadService
from lobster.testing.provenance_replay import RecordingDataManager, factory_tools
from lobster.tools.download_orchestrator import DownloadOrchestrator

MODULE = "lobster.agents.data_expert.data_expert"
ENTRY_ID = "queue_synthetic"
DATASET_ID = "GSE_SYNTHETIC"
MODALITY = "geo_gse_synthetic_bulk"


@pytest.fixture
def queue_tool(tmp_path):
    dm = RecordingDataManager(workspace_path=tmp_path)
    entry = SimpleNamespace(
        entry_id=ENTRY_ID,
        dataset_id=DATASET_ID,
        database="geo",
        status=DownloadStatus.PENDING,
        recommended_strategy=None,
        validation_status=None,
        validation_result=None,
        modality_name=MODALITY,
    )
    dm.download_queue = MagicMock()
    dm.download_queue.list_entries.return_value = [entry]
    dm.download_queue.get_entry.return_value = entry
    dm.modalities[MODALITY] = ad.AnnData(X=np.array([[1, 2], [3, 4]]))
    ir = object()
    orchestrator = MagicMock()
    orchestrator.execute_download.return_value = (MODALITY, {}, ir)
    tool = factory_tools(data_expert, MODULE, dm)["execute_download_from_queue"]
    with patch(f"{MODULE}.DownloadOrchestrator", return_value=orchestrator):
        yield tool, dm, entry, orchestrator, ir


def test_model_schema_and_docstring(queue_tool):
    tool, *_ = queue_tool
    schema = tool.args_schema.model_json_schema()["properties"]
    assert schema["data_requirement"]["default"] == ""
    assert schema["data_requirement"]["type"] == "string"
    description = " ".join(tool.description.split())
    assert "gene_counts" in description
    assert "numeric check, not verification of raw-count provenance" in description
    assert "do not retry the same entry unchanged" in description


@pytest.mark.parametrize("concatenation", ["auto", "union", "intersection"])
def test_active_requirement_handoff_preserves_other_overrides(
    queue_tool, concatenation
):
    tool, dm, _, orchestrator, ir = queue_tool
    tool.invoke(
        {
            "entry_id": ENTRY_ID,
            "data_requirement": "gene_counts",
            "concatenation_strategy": concatenation,
            "strategy_override": "MATRIX_FIRST",
        }
    )
    expected_params = {"data_requirement": "gene_counts"}
    if concatenation != "auto":
        expected_params["use_intersecting_genes_only"] = concatenation == "intersection"
    orchestrator.execute_download.assert_called_once_with(
        ENTRY_ID,
        {"strategy_name": "MATRIX_FIRST", "strategy_params": expected_params},
        return_ir=True,
    )
    calls = dm.calls_for("execute_download_from_queue")
    assert len(calls) == 1 and calls[0].ir is ir
    assert (
        calls[0].parameters["strategy_override"]["strategy_params"] == expected_params
    )


@pytest.mark.parametrize("explicit_empty", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_empty_requirement_literal_legacy_report_and_log(
    queue_tool, explicit_empty, failure
):
    tool, dm, _, orchestrator, ir = queue_tool
    arguments = {"entry_id": ENTRY_ID}
    if explicit_empty:
        arguments["data_requirement"] = ""
    if failure:
        orchestrator.execute_download.side_effect = RuntimeError("synthetic failure")
        expected = (
            "## Download Failed: GSE_SYNTHETIC\n\n"
            "❌ **Status**: Download failed\n"
            "- **Error**: synthetic failure\n"
            "- **Database**: geo\n"
            "- **Queue entry**: `queue_synthetic` (FAILED)\n"
            "\n**Troubleshooting**:\n"
            "1. Check dataset availability: GSE_SYNTHETIC\n"
            "2. Verify URLs are accessible\n"
            "3. Try a different download strategy:\n"
            '   - strategy_override="MATRIX_FIRST"\n'
            '   - strategy_override="SUPPLEMENTARY_FIRST"\n'
            '   - strategy_override="H5_FIRST"\n'
            "4. Try different concatenation mode:\n"
            '   - concatenation_strategy="union" (preserves all genes)\n'
            '   - concatenation_strategy="intersection" (only common genes)\n'
            "5. Review error log: `get_queue_status(status_filter='FAILED')`\n"
        )
    else:
        expected = (
            "\n✅ **Download completed successfully**\n\n"
            "Dataset ID: GSE_SYNTHETIC\n"
            "Entry ID: queue_synthetic\n"
            "Database: geo\n"
            "Modality Name: geo_gse_synthetic_bulk\n"
            "Strategy Used: auto-detected\n\n"
            "Samples: 2\nFeatures: 2\nConcatenation: auto\n\n"
            "You can now analyze this dataset using the appropriate analysis tools.\n"
        )
    assert tool.invoke(arguments).encode() == expected.encode()
    orchestrator.execute_download.assert_called_once_with(
        ENTRY_ID, None, return_ir=True
    )
    calls = dm.calls_for("execute_download_from_queue")
    if failure:
        assert calls == []
    else:
        assert len(calls) == 1 and calls[0].ir is ir
        assert calls[0].parameters == {
            "entry_id": ENTRY_ID,
            "database": "geo",
            "dataset_id": DATASET_ID,
            "strategy_override": {},
        }


def test_unknown_requirement_rejected_before_queue_access(queue_tool):
    tool, dm, _, orchestrator, _ = queue_tool
    result = tool.invoke(
        {"entry_id": ENTRY_ID, "data_requirement": "synthetic_unknown"}
    )
    assert "Unknown data_requirement" in result
    dm.download_queue.list_entries.assert_not_called()
    dm.download_queue.get_entry.assert_not_called()
    dm.download_queue.update_status.assert_not_called()
    orchestrator.execute_download.assert_not_called()
    assert dm.calls == []


@pytest.mark.parametrize("database", ["pride", "sra", "massive"])
def test_requirement_not_silently_ignored_for_other_databases(queue_tool, database):
    tool, dm, entry, orchestrator, _ = queue_tool
    entry.database = database
    result = tool.invoke({"entry_id": ENTRY_ID, "data_requirement": "gene_counts"})
    assert "supported only for GEO" in result
    orchestrator.execute_download.assert_not_called()
    dm.download_queue.update_status.assert_not_called()
    assert dm.calls == []


def test_structured_refusal_reaches_tool_without_generic_retry_advice(queue_tool):
    tool, dm, _, orchestrator, _ = queue_tool
    error = ImportRequirementNotMet(
        DATASET_ID,
        ImportRequirement.parse("gene_counts"),
        [
            CandidateEvidence(
                "synthetic_gene_counts.tsv",
                level="gene",
                verdict="rejected",
                reason="fractional values",
            )
        ],
        problem="no_candidate",
    )
    orchestrator.execute_download.side_effect = error
    result = tool.invoke({"entry_id": ENTRY_ID, "data_requirement": "gene_counts"})
    assert error.to_message() in result
    assert "synthetic_gene_counts.tsv" in result and "fractional values" in result
    assert "Do not retry this entry unchanged" in result
    assert 'strategy_override="' not in result
    assert "Try different concatenation" not in result
    assert dm.calls == []


@pytest.mark.parametrize("status", [DownloadStatus.PENDING, DownloadStatus.FAILED])
def test_real_queue_wrapper_tool_refusal_is_failed_and_rerunnable(tmp_path, status):
    dm = RecordingDataManager(workspace_path=tmp_path)
    dm.download_queue = DownloadQueue(queue_file=tmp_path / "download_queue.jsonl")
    entry = DownloadQueueEntry(
        entry_id=ENTRY_ID, dataset_id=DATASET_ID, database="geo", status=status
    )
    dm.download_queue.add_entry(entry)
    error = ImportRequirementNotMet(
        DATASET_ID,
        ImportRequirement.parse("gene_counts"),
        [
            CandidateEvidence(
                "synthetic_gene_counts.tsv",
                level="gene",
                verdict="rejected",
                reason="fractional values",
            )
        ],
        problem="no_candidate",
    )
    geo = MagicMock()
    geo.download_dataset.side_effect = error
    with patch(
        "lobster.services.data_access.geo_download_service.GEOService", return_value=geo
    ):
        wrapper = GEODownloadService(dm)
    with patch.object(DownloadOrchestrator, "_register_default_services"):
        orchestrator = DownloadOrchestrator(dm)
    orchestrator.register_service(wrapper)
    tool = factory_tools(data_expert, MODULE, dm)["execute_download_from_queue"]
    with patch(f"{MODULE}.DownloadOrchestrator", return_value=orchestrator):
        for attempt in range(2):
            result = tool.invoke(
                {"entry_id": ENTRY_ID, "data_requirement": "gene_counts"}
            )
            assert error.to_message() in result
            assert "Import Refused" in result and 'strategy_override="' not in result
            failed = dm.download_queue.get_entry(ENTRY_ID)
            assert failed.status == DownloadStatus.FAILED
            assert len(failed.error_log) == attempt + 1
            assert "ImportRequirementNotMet" in failed.error_log[-1]
            assert dm.modalities == {} and dm.calls == []
    assert geo.download_dataset.call_count == 2


def test_active_success_returns_stored_per_file_evidence(queue_tool):
    tool, dm, _, _, _ = queue_tool
    dm.modalities[MODALITY].uns["geo_import"] = {
        "data_requirement": "gene_counts",
        "source_file": "synthetic_gene_counts.tsv",
        "numeric_check": "whole-number counts (numeric check)",
    }
    result = tool.invoke({"entry_id": ENTRY_ID, "data_requirement": "gene_counts"})
    assert "synthetic_gene_counts.tsv" in result
    assert "whole-number counts (numeric check)" in result


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        {},
        {"data_requirement": ""},
        {"data_requirement": "synthetic_other"},
        "invalid",
        {"data_requirement": "gene_counts", "source_file": "synthetic_gene_counts.tsv"},
    ],
)
def test_active_completed_requires_same_stored_requirement(queue_tool, evidence):
    tool, dm, entry, orchestrator, _ = queue_tool
    entry.status = DownloadStatus.COMPLETED
    if evidence is not None:
        dm.modalities[MODALITY].uns["geo_import"] = evidence
    result = tool.invoke({"entry_id": ENTRY_ID, "data_requirement": "gene_counts"})
    if isinstance(evidence, dict) and evidence.get("data_requirement") == "gene_counts":
        assert "already imported" in result
        assert "synthetic_gene_counts.tsv" in result
    else:
        assert (
            "already imported without the gene_counts requirement; not re-imported"
            in result
        )
        assert "Report it" in result and "queue a new entry" in result
    orchestrator.execute_download.assert_not_called()
    dm.download_queue.update_status.assert_not_called()
    assert entry.status == DownloadStatus.COMPLETED and dm.calls == []


def test_active_completed_missing_modality_refuses_without_status_reset(queue_tool):
    tool, dm, entry, orchestrator, _ = queue_tool
    entry.status = DownloadStatus.COMPLETED
    del dm.modalities[MODALITY]
    result = tool.invoke({"entry_id": ENTRY_ID, "data_requirement": "gene_counts"})
    assert "not re-imported" in result
    orchestrator.execute_download.assert_not_called()
    dm.download_queue.update_status.assert_not_called()


@pytest.mark.parametrize("explicit_empty", [False, True])
def test_legacy_completed_literal_message_unchanged(queue_tool, explicit_empty):
    tool, dm, entry, orchestrator, _ = queue_tool
    entry.status = DownloadStatus.COMPLETED
    arguments = {"entry_id": ENTRY_ID}
    if explicit_empty:
        arguments["data_requirement"] = ""
    assert (
        tool.invoke(arguments)
        == "Entry 'queue_synthetic' already downloaded as 'geo_gse_synthetic_bulk'"
    )
    orchestrator.execute_download.assert_not_called()
    dm.download_queue.update_status.assert_not_called()


def test_validation_warning_preserves_active_requirement_in_followup(queue_tool):
    tool, _, entry, orchestrator, _ = queue_tool
    entry.validation_status = ValidationStatus.VALIDATED_WITH_WARNINGS
    entry.validation_result = {"warnings": ["synthetic warning"]}
    result = tool.invoke({"entry_id": ENTRY_ID, "data_requirement": "gene_counts"})
    assert 'force_download=True, data_requirement="gene_counts"' in result
    orchestrator.execute_download.assert_not_called()
    result = tool.invoke(
        {
            "entry_id": ENTRY_ID,
            "force_download": True,
            "data_requirement": "gene_counts",
        }
    )
    assert "Download completed successfully" in result
    assert (
        orchestrator.execute_download.call_args.args[1]["strategy_params"][
            "data_requirement"
        ]
        == "gene_counts"
    )
