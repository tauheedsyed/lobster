"""Peptide library tools log IR that rebuilds the same library."""

from types import SimpleNamespace

import pandas as pd
import pytest

from lobster.agents.proteomics.peptide_tools import create_peptide_tools
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.services.peptidomics.peptide_library_service import (
    PeptideLibraryService,
)
from lobster.testing.provenance_replay import RecordingDataManager, replay_ir

FASTA = ">p1 first\nGIGKFLHSAKKFGKAFVGEIMNS\n>p2\nKWKLFKKIGAVLKVL\n>bad\nXXZZ\n"


@pytest.fixture
def library(tmp_path):
    path = tmp_path / "peptides.fasta"
    path.write_text(FASTA)
    dm = RecordingDataManager(workspace_path=tmp_path)
    tools = {t.name: t for t in create_peptide_tools(dm, workspace_path=tmp_path)}
    tools["import_peptide_sequences"].invoke({"file_path": str(path)})
    return dm, tools, path


def _ir(dm, tool_name):
    ir = dm.last_ir(tool_name)
    assert isinstance(ir, AnalysisStep) and ir.replayable
    ir.validate_rendered_code()
    return ir


def test_import_replays(library):
    dm, _, _ = library
    produced = dm.modalities["peptides"]
    assert list(produced.obs_names) == ["p1", "p2"]  # invalid sequence dropped
    replayed = replay_ir(_ir(dm, "import_peptide_sequences"))["adata"]
    pd.testing.assert_frame_equal(replayed.obs, produced.obs)


def test_filter_replays(library):
    dm, tools, _ = library
    tools["filter_peptides"].invoke({"modality_name": "peptides", "max_length": 20})
    produced = dm.modalities["peptides_filtered"]
    assert list(produced.obs_names) == ["p2"]
    source = dm.modalities["peptides"].copy()
    replayed = replay_ir(_ir(dm, "filter_peptides"), adata=source)["adata"]
    pd.testing.assert_frame_equal(replayed.obs, produced.obs)


@pytest.mark.parametrize(
    "variant_type", ["alanine_scan", "point_mutations", "scrambled"]
)
def test_variants_replay(library, variant_type):
    dm, tools, _ = library
    tools["generate_peptide_variants"].invoke(
        {
            "modality_name": "peptides",
            "variant_type": variant_type,
            "target_peptide": "p2",
        }
    )
    produced = dm.modalities[f"peptides_{variant_type}"]
    source = dm.modalities["peptides"].copy()
    ir = _ir(dm, "generate_peptide_variants")
    if variant_type == "scrambled":
        # The drawn seed is recorded, so the scrambled set can be regenerated.
        assert isinstance(ir.parameters["random_seed"], int)
    replayed = replay_ir(ir, adata=source)["adata"]
    pd.testing.assert_frame_equal(replayed.obs, produced.obs)


def test_activity_annotation_reapplies_recorded_hits(library):
    dm, _, _ = library
    seq = dm.modalities["peptides"].obs.loc["p1", "sequence"]

    class FakeProvider:
        def search_sequences(self, query, max_results=3):
            return [SimpleNamespace(sequence=seq, activities=["antibacterial"])]

    adata_out, stats, ir = PeptideLibraryService().annotate_known_activity(
        dm.modalities["peptides"], providers=[FakeProvider], database="dbaasp"
    )
    assert stats["n_annotated"] == 1
    assert ir.replayable
    # Replay needs no database: it re-applies the recorded hits.
    replayed = replay_ir(ir, adata=dm.modalities["peptides"].copy())["adata"]
    pd.testing.assert_series_equal(
        replayed.obs["known_activities"], adata_out.obs["known_activities"]
    )
    assert list(adata_out.obs["known_activities"]) == ["antibacterial", "unknown"]


def test_report_export_is_utility(library):
    _, tools, _ = library
    tool = tools["export_peptide_report"]
    assert tool.metadata == {"categories": ["UTILITY"], "provenance": False}
