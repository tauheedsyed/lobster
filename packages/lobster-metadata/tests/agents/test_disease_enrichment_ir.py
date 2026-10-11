"""Disease enrichment logs IR that re-applies the labels it wrote."""

import copy
import json

import pytest

from lobster.agents.metadata_assistant.metadata_assistant import metadata_assistant
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.testing.provenance_replay import (
    RecordingDataManager,
    factory_tools,
    replay_ir,
)

MODULE = "lobster.agents.metadata_assistant.metadata_assistant"


def _samples():
    return [
        {"run_accession": "SRR1", "publication_entry_id": "pub_a"},
        {"run_accession": "SRR2", "publication_entry_id": "pub_b"},
        {"run_accession": "SRR3", "publication_entry_id": "pub_a", "disease": "uc"},
    ]


@pytest.fixture
def setup(tmp_path):
    dm = RecordingDataManager(workspace_path=tmp_path)
    dm.metadata_store["agg"] = {"samples": _samples()}
    return dm, factory_tools(metadata_assistant, MODULE, dm)


def test_enrichment_replays_recorded_labels(setup):
    dm, tools = setup
    before = copy.deepcopy(dm.metadata_store["agg"]["samples"])
    tools["enrich_samples_with_disease"].invoke(
        {
            "workspace_key": "agg",
            "enrichment_mode": "manual",
            "manual_mappings": json.dumps({"pub_a": "crc"}),
        }
    )
    enriched = dm.metadata_store["agg"]["samples"]
    assert enriched[0]["disease"] == "crc"
    assert "disease" not in enriched[1]

    ir = dm.last_ir("enrich_samples_with_disease")
    assert isinstance(ir, AnalysisStep) and ir.replayable
    ir.validate_rendered_code()
    replayed = replay_ir(ir, samples=before)["samples"]
    assert replayed == enriched


def test_dry_run_is_not_replayable(setup):
    dm, tools = setup
    tools["enrich_samples_with_disease"].invoke(
        {
            "workspace_key": "agg",
            "enrichment_mode": "manual",
            "manual_mappings": json.dumps({"pub_a": "crc"}),
            "dry_run": True,
        }
    )
    assert "disease" not in dm.metadata_store["agg"]["samples"][0]
    ir = dm.last_ir("enrich_samples_with_disease")
    assert isinstance(ir, AnalysisStep) and ir.replayable is False


def test_sample_mapping_is_read_only_utility(setup):
    _, tools = setup
    assert tools["map_samples_by_id"].metadata == {
        "categories": ["UTILITY"],
        "provenance": False,
    }
