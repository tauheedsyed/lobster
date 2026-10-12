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


def test_duplicate_sequences_get_one_consistent_outcome(library):
    """Duplicates are looked up once, so the sequence-keyed IR replays every row."""
    import anndata as ad

    calls = []

    class FlakyProvider:
        def search_sequences(self, query, max_results=3):
            calls.append(query)
            if len(calls) > 1:
                raise RuntimeError("transient provider failure")
            return [SimpleNamespace(sequence="ACDE", activities=["antibacterial"])]

    peptides = ad.AnnData(
        obs=pd.DataFrame({"sequence": ["ACDE", "ACDE"]}, index=["a", "b"])
    )
    adata_out, _, ir = PeptideLibraryService().annotate_known_activity(
        peptides, providers=[FlakyProvider], database="dbaasp"
    )
    assert len(calls) == 1
    assert list(adata_out.obs["known_activities"]) == ["antibacterial", "antibacterial"]
    replayed = replay_ir(ir, adata=peptides.copy())["adata"]
    pd.testing.assert_series_equal(
        replayed.obs["known_activities"], adata_out.obs["known_activities"]
    )


def test_unbounded_filter_replays(library):
    """inf bounds must render as runnable Python, not the undefined name `inf`."""
    dm, tools, _ = library
    tools["filter_peptides"].invoke(
        {
            "modality_name": "peptides",
            "min_charge": float("-inf"),
            "max_charge": float("inf"),
        }
    )
    produced = dm.modalities["peptides_filtered"]
    source = dm.modalities["peptides"].copy()
    replayed = replay_ir(_ir(dm, "filter_peptides"), adata=source)["adata"]
    pd.testing.assert_frame_equal(replayed.obs, produced.obs)


def test_import_only_notebook_runs_without_an_h5ad_input(tmp_path, monkeypatch):
    """The exported notebook must build adata from the FASTA, not load dataset.h5ad."""
    import nbformat

    monkeypatch.chdir(tmp_path)  # the notebook's save cell writes to the cwd

    from lobster.core.runtime.data_manager import DataManagerV2

    fasta = tmp_path / "lib.fasta"
    fasta.write_text(FASTA)
    dm = DataManagerV2(workspace_path=tmp_path / "ws", auto_scan=False)
    tools = {t.name: t for t in create_peptide_tools(dm, workspace_path=tmp_path)}
    tools["import_peptide_sequences"].invoke({"file_path": str(fasta)})
    tools["filter_peptides"].invoke({"modality_name": "peptides", "max_length": 20})

    nb = nbformat.read(
        dm.export_notebook(name="peptides", filter_strategy="replay"), as_version=4
    )
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "read_h5ad(input_data)" not in code
    scope = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, "<cell>", "exec"), scope)  # nosec B102
    assert list(scope["adata"].obs_names) == ["p2"]


def test_later_peptide_import_keeps_the_input_load(tmp_path):
    """Only a first-step import replaces the notebook's input load."""
    from types import SimpleNamespace as NS

    from lobster.core.notebooks.exporter import NotebookExporter

    def pair(op):
        return ({}, NS(operation=op))

    exporter = NotebookExporter.__new__(NotebookExporter)
    first = [pair("peptidomics.import_sequences"), pair("peptidomics.filter_peptides")]
    later = [pair("lobster.ml.select_ml_features"), pair("peptidomics.import_sequences")]
    assert exporter._has_download_activity(first) is True
    assert exporter._has_download_activity(later) is False


def test_report_export_is_utility(library):
    _, tools, _ = library
    tool = tools["export_peptide_report"]
    assert tool.metadata == {"categories": ["UTILITY"], "provenance": False}
