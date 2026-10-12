"""Provenance IR replay for the annotation and formula-DE tools.

Each tool that changes data must log an AnalysisStep whose template, executed on
a copy of the tool's input, reproduces the tool's own output. Read-only checks
log IR that is exported but not replayable.
"""

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.annotation_expert import annotation_expert
from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.testing.provenance_replay import (
    RecordingDataManager,
    factory_tools,
    replay_ir,
)

ANNOTATION_MODULE = "lobster.agents.transcriptomics.annotation_expert"
DE_MODULE = "lobster.agents.transcriptomics.de_analysis_expert"


def _clustered_cells(n_cells: int = 60) -> ad.AnnData:
    rng = np.random.default_rng(0)
    clusters = np.array([str(i % 4) for i in range(n_cells)])
    obs = pd.DataFrame(
        {
            "seurat_clusters": pd.Categorical(clusters),
            "n_genes_by_counts": np.where(clusters == "3", 50, 800),
            "pct_counts_mt": np.where(clusters == "2", 70.0, 5.0),
            "total_counts": np.full(n_cells, 3000),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    return ad.AnnData(
        X=rng.poisson(2, size=(n_cells, 20)).astype(float),
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(20)]),
    )


def _assert_real_ir(ir, replayable=True):
    assert isinstance(ir, AnalysisStep), f"expected AnalysisStep, got {ir!r}"
    assert ir.replayable is replayable
    ir.validate_rendered_code()


@pytest.fixture
def annotation_setup():
    adata = _clustered_cells()
    dm = RecordingDataManager({"cells": adata})
    tools = factory_tools(annotation_expert, ANNOTATION_MODULE, dm)
    return adata, dm, tools


class TestAnnotationToolIR:
    def test_manual_annotation_replays(self, annotation_setup):
        adata, dm, tools = annotation_setup
        tools["manually_annotate_clusters"].invoke(
            {
                "modality_name": "cells",
                "annotations": {"0": "T cells", "1": "B cells"},
                "cluster_key": "seurat_clusters",
                "save_result": False,
            }
        )
        ir = dm.last_ir("manually_annotate_clusters")
        _assert_real_ir(ir)
        produced = dm.modalities["cells_manually_annotated"]
        replayed = replay_ir(ir, adata=adata.copy())["adata"]
        pd.testing.assert_series_equal(
            replayed.obs["cell_type_manual"], produced.obs["cell_type_manual"]
        )
        assert set(produced.obs["cell_type_manual"]) == {
            "T cells",
            "B cells",
            "Unassigned",
        }

    def test_collapse_replays(self, annotation_setup):
        adata, dm, tools = annotation_setup
        tools["collapse_clusters_to_celltype"].invoke(
            {
                "modality_name": "cells",
                "cluster_list": ["0", "2"],
                "cell_type_name": "Myeloid",
                "cluster_key": "seurat_clusters",
                "save_result": False,
            }
        )
        ir = dm.last_ir("collapse_clusters_to_celltype")
        _assert_real_ir(ir)
        produced = dm.modalities["cells_collapsed"]
        replayed = replay_ir(ir, adata=adata.copy())["adata"]
        pd.testing.assert_series_equal(
            replayed.obs["cell_type_manual"], produced.obs["cell_type_manual"]
        )
        assert (produced.obs["cell_type_manual"] == "Myeloid").sum() == 30

    @pytest.mark.parametrize("remove_debris", [False, True])
    def test_mark_debris_replays(self, annotation_setup, remove_debris):
        adata, dm, tools = annotation_setup
        tools["mark_clusters_as_debris"].invoke(
            {
                "modality_name": "cells",
                "debris_clusters": ["3"],
                "remove_debris": remove_debris,
                "cluster_key": "seurat_clusters",
                "save_result": False,
            }
        )
        ir = dm.last_ir("mark_clusters_as_debris")
        _assert_real_ir(ir)
        suffix = "debris_removed" if remove_debris else "debris_marked"
        produced = dm.modalities[f"cells_{suffix}"]
        replayed = replay_ir(ir, adata=adata.copy())["adata"]
        assert list(replayed.obs_names) == list(produced.obs_names)
        pd.testing.assert_frame_equal(
            replayed.obs[["cell_type_manual", "is_debris"]],
            produced.obs[["cell_type_manual", "is_debris"]],
        )
        assert produced.n_obs == (45 if remove_debris else 60)

    def test_debris_suggestion_uses_cluster_key_and_is_not_replayable(
        self, annotation_setup
    ):
        adata, dm, tools = annotation_setup
        result = tools["suggest_debris_clusters"].invoke(
            {"modality_name": "cells", "cluster_key": "seurat_clusters"}
        )
        # Cluster 3 has 50 genes/cell, cluster 2 has 70% mito: both flagged.
        assert "Cluster 2" in result and "Cluster 3" in result
        ir = dm.last_ir("suggest_debris_clusters")
        _assert_real_ir(ir, replayable=False)
        assert ir.exportable
        suggested = replay_ir(ir, adata=adata.copy())["suggested_debris"]
        assert sorted(suggested) == ["2", "3"]
        assert sorted(ir.execution_context["suggested_debris_clusters"]) == ["2", "3"]

    def test_debris_suggestion_is_stateless_across_datasets(self, annotation_setup):
        """A second dataset must not reuse the first dataset's cluster summary."""
        adata, dm, tools = annotation_setup
        tools["suggest_debris_clusters"].invoke(
            {"modality_name": "cells", "cluster_key": "seurat_clusters"}
        )
        healthy = _clustered_cells()
        healthy.obs["n_genes_by_counts"] = 800
        healthy.obs["pct_counts_mt"] = 5.0
        dm.modalities["healthy"] = healthy
        result = tools["suggest_debris_clusters"].invoke(
            {"modality_name": "healthy", "cluster_key": "seurat_clusters"}
        )
        assert result.startswith("No debris clusters suggested")

    def test_template_records_resolved_mapping(self, annotation_setup, monkeypatch):
        from lobster.services.templates.annotation_templates import (
            AnnotationTemplateService,
        )

        mapping = {"0": "T cells", "1": "B cells", "2": "NK cells"}
        monkeypatch.setattr(
            AnnotationTemplateService,
            "apply_template_to_clusters",
            lambda self, **kwargs: dict(mapping),
        )
        adata, dm, tools = annotation_setup
        tools["apply_annotation_template"].invoke(
            {
                "modality_name": "cells",
                "tissue_type": "pbmc",
                "cluster_key": "seurat_clusters",
                "save_result": False,
            }
        )
        ir = dm.last_ir("apply_annotation_template")
        _assert_real_ir(ir)
        produced = dm.modalities["cells_template_pbmc"]
        replayed = replay_ir(ir, adata=adata.copy())["adata"]
        pd.testing.assert_series_equal(
            replayed.obs["cell_type_template"], produced.obs["cell_type_template"]
        )
        assert (produced.obs["cell_type_template"] == "Unknown").sum() == 15

    def test_import_mapping_replays(self, annotation_setup, tmp_path):
        adata, dm, tools = annotation_setup
        mapping_file = tmp_path / "mapping.json"
        mapping_file.write_text(
            json.dumps(
                {
                    "cluster_column": "seurat_clusters",
                    "cluster_to_celltype": {"0": "T cells", "3": "Debris"},
                }
            )
        )
        tools["import_annotation_mapping"].invoke(
            {
                "modality_name": "cells",
                "mapping_file": str(mapping_file),
                "save_result": False,
            }
        )
        ir = dm.last_ir("import_annotation_mapping")
        _assert_real_ir(ir)
        produced = dm.modalities["cells_imported_annotations"]
        replayed = replay_ir(ir, adata=adata.copy())["adata"]
        pd.testing.assert_series_equal(
            replayed.obs["cell_type_imported"], produced.obs["cell_type_imported"]
        )


    def test_imported_null_labels_stay_unassigned(self, annotation_setup, tmp_path):
        """JSON null must not become a cell type literally named 'None'."""
        adata, dm, tools = annotation_setup
        mapping_file = tmp_path / "mapping.json"
        mapping_file.write_text(
            json.dumps(
                {
                    "cluster_column": "seurat_clusters",
                    "cluster_to_celltype": {"0": None, "1": "B cells"},
                }
            )
        )
        tools["import_annotation_mapping"].invoke(
            {"modality_name": "cells", "mapping_file": str(mapping_file), "save_result": False}
        )
        labels = dm.modalities["cells_imported_annotations"].obs["cell_type_imported"]
        assert "None" not in set(labels)
        assert set(labels) == {"B cells", "Unassigned"}
        replayed = replay_ir(dm.last_ir("import_annotation_mapping"), adata=adata.copy())
        pd.testing.assert_series_equal(replayed["adata"].obs["cell_type_imported"], labels)


# ---------------------------------------------------------------------------
# Formula-based differential expression
# ---------------------------------------------------------------------------


def _pseudobulk(seed: int = 0) -> ad.AnnData:
    """8 pseudobulk samples x 60 genes; the first 10 genes are up in 'treated'."""
    rng = np.random.default_rng(seed)
    condition = ["control"] * 4 + ["treated"] * 4
    counts = rng.negative_binomial(20, 0.3, size=(8, 60)).astype(float)
    counts[4:, :10] *= 4
    obs = pd.DataFrame(
        {"condition": pd.Categorical(condition), "batch": ["a", "b"] * 4},
        index=[f"sample_{i}" for i in range(8)],
    )
    return ad.AnnData(
        X=counts, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(60)])
    )


@pytest.fixture
def de_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the DE IR's BulkRNASeqService writes ./bulk_results
    adata = _pseudobulk()
    original = adata.copy()
    dm = RecordingDataManager({"pb": adata})
    tools = factory_tools(de_analysis_expert, DE_MODULE, dm)
    return original, dm, tools


CONTRAST = ["condition", "treated", "control"]
RESULTS_KEY = "de_results_formula_condition_treated_vs_control"


class TestFormulaDEToolIR:
    def test_prepare_design_replays(self, de_setup):
        original, dm, tools = de_setup
        tools["prepare_de_design"].invoke(
            {"modality_name": "pb", "formula": "~condition", "contrast": CONTRAST}
        )
        ir = dm.last_ir("prepare_de_design")
        _assert_real_ir(ir)
        replayed = replay_ir(ir, adata=original.copy())["adata"]
        stored = dm.modalities["pb"].uns["formula_design"]
        assert replayed.uns["formula_design"]["formula"] == stored["formula"]
        assert replayed.uns["formula_design"]["contrast"] == stored["contrast"]
        assert (
            replayed.uns["formula_design"]["validation_results"]["valid"]
            == stored["validation_results"]["valid"]
        )

    def test_design_validation_is_exported_not_replayed(self, de_setup):
        original, dm, tools = de_setup
        tools["validate_experimental_design"].invoke(
            {"modality_name": "pb", "formula": "~condition"}
        )
        ir = dm.last_ir("validate_experimental_design")
        _assert_real_ir(ir, replayable=False)
        check = replay_ir(ir, adata=original.copy())["design_check"]
        assert check["valid"] is True

    def test_formula_de_replays_identical_results(self, de_setup):
        original, dm, tools = de_setup
        result = tools["run_de_with_formula"].invoke(
            {
                "pseudobulk_modality": "pb",
                "formula": "~condition",
                "contrast": CONTRAST,
                "save_results": False,
            }
        )
        assert "Differential Expression Analysis Complete" in result
        ir = dm.last_ir("run_de_with_formula")
        _assert_real_ir(ir)
        produced = dm.modalities["pb"].uns[RESULTS_KEY]["results_df"]
        replayed = replay_ir(ir, adata=original.copy())["adata"]
        pd.testing.assert_frame_equal(replayed.uns[RESULTS_KEY]["results_df"], produced)
        assert replayed.uns["de_formula_design"]["formula"] == "~condition"

    def test_iteration_and_comparison_replay(self, de_setup):
        original, dm, tools = de_setup
        for name, formula in (("base", "~condition"), ("batch", "~batch + condition")):
            result = tools["iterate_de_analysis"].invoke(
                {
                    "pseudobulk_modality": "pb",
                    "new_formula": formula,
                    "new_contrast": CONTRAST,
                    "iteration_name": name,
                }
            )
            # Regression: the tool used to call run_de_with_formula directly,
            # which fails on a LangChain tool object, so every iteration errored.
            assert "Iteration" in result and "Error" not in result, result

        tracker = dm.modalities["pb"].uns["de_iterations"]
        assert [i["name"] for i in tracker["iterations"]] == ["base", "batch"]

        # Replaying every logged step in order rebuilds the iteration state.
        replayed = original.copy()
        for call in dm.calls:
            if call.tool_name in ("run_de_with_formula", "iterate_de_analysis"):
                _assert_real_ir(call.ir)
                replayed = replay_ir(call.ir, adata=replayed)["adata"]
        assert [i["name"] for i in replayed.uns["de_iterations"]["iterations"]] == [
            "base",
            "batch",
        ]
        assert replayed.uns["de_iterations"]["current_iteration"] == 2

        tools["compare_de_iterations"].invoke({"pseudobulk_modality": "pb"})
        ir = dm.last_ir("compare_de_iterations")
        _assert_real_ir(ir)
        replayed = replay_ir(ir, adata=replayed)["adata"]
        key = "iteration_comparison_batch_vs_base"
        produced = dm.modalities["pb"].uns["iteration_comparisons"][key]
        replayed_cmp = replayed.uns["iteration_comparisons"][key]
        assert replayed_cmp["summary_stats"] == produced["summary_stats"]
        assert sorted(replayed_cmp["overlap_genes"]) == sorted(
            produced["overlap_genes"]
        )

    def test_same_contrast_iterations_keep_their_own_results(self, de_setup):
        """Two formulas for one contrast must not be compared as one result.

        The formula result key depends only on the contrast, so the second run
        overwrote the first and the comparison reported 100% overlap.
        """
        original, dm, tools = de_setup
        for name, formula in (("base", "~condition"), ("batch", "~batch + condition")):
            tools["iterate_de_analysis"].invoke(
                {
                    "pseudobulk_modality": "pb",
                    "new_formula": formula,
                    "new_contrast": CONTRAST,
                    "iteration_name": name,
                }
            )
        uns = dm.modalities["pb"].uns
        base = uns["de_results_iteration_1"]["results_df"]
        batch = uns["de_results_iteration_2"]["results_df"]
        lfc_delta = (base["log2FoldChange"] - batch["log2FoldChange"]).abs().max()
        assert lfc_delta > 1e-3, "the two formulas must give different results"

        tools["compare_de_iterations"].invoke({"pseudobulk_modality": "pb"})
        cmp = uns["iteration_comparisons"]["iteration_comparison_batch_vs_base"]
        expected_corr = batch.loc[cmp["overlap_genes"], "log2FoldChange"].corr(
            base.loc[cmp["overlap_genes"], "log2FoldChange"]
        )
        assert cmp["correlation"] == pytest.approx(expected_corr)
        assert cmp["correlation"] < 1.0 - 1e-9

        # Replaying every step reproduces the same, non-trivial comparison.
        replayed = original.copy()
        for call in dm.calls:
            replayed = replay_ir(call.ir, adata=replayed)["adata"]
        replayed_cmp = replayed.uns["iteration_comparisons"][
            "iteration_comparison_batch_vs_base"
        ]
        assert replayed_cmp["summary_stats"] == cmp["summary_stats"]
        assert replayed_cmp["correlation"] == pytest.approx(cmp["correlation"])

    def test_comparison_without_saving_is_not_replayable(self, de_setup):
        _, dm, tools = de_setup
        for name in ("one", "two"):
            tools["iterate_de_analysis"].invoke(
                {
                    "pseudobulk_modality": "pb",
                    "new_formula": "~condition",
                    "new_contrast": CONTRAST,
                    "iteration_name": name,
                }
            )
        tools["compare_de_iterations"].invoke(
            {"pseudobulk_modality": "pb", "save_comparison": False}
        )
        _assert_real_ir(dm.last_ir("compare_de_iterations"), replayable=False)
