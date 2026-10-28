"""pyDESeq2 DE runs on raw integer counts, never on truncated normalized values."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.core.sparse_utils import SparseConversionError
from lobster.services.analysis.bulk_rnaseq_service import (
    BulkRNASeqService,
    PyDESeq2Error,
)
from lobster.testing.provenance_replay import factory_tools, replay_ir

DE_MODULE = "lobster.agents.transcriptomics.de_analysis_expert"
SERVICE_MODULE = "lobster.services.analysis.bulk_rnaseq_service"
COUNTS = np.array(
    [[10, 0, 5], [12, 1, 4], [9, 2, 6], [30, 0, 1], [28, 3, 2], [33, 1, 0]],
    dtype=float,
)
LOG_CPM = np.log1p(COUNTS / COUNTS.sum(axis=1, keepdims=True) * 1e6)


def matrix(kind):
    if kind == "counts":
        return COUNTS.copy()
    if kind == "integer_counts":
        return COUNTS.astype(int)
    if kind == "sparse_counts":
        return sparse.csr_matrix(COUNTS)
    if kind == "normalized":
        return COUNTS / COUNTS.sum(axis=1, keepdims=True)
    return LOG_CPM.copy()


def pseudobulk(x):
    adata = ad.AnnData(
        X=x,
        obs=pd.DataFrame(
            {"condition": ["sample_a"] * 3 + ["sample_b"] * 3},
            index=[f"sample_{i}" for i in range(6)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(x.shape[1])]),
    )
    adata.uns["is_pseudobulk"] = True
    return adata


def results():
    return (
        pd.DataFrame(
            {"log2FoldChange": [1.0, -1.0, 0.1], "padj": [0.01, 0.02, 0.9]},
            index=pd.Index(["gene_0", "gene_1", "gene_2"], name="gene"),
        ),
        {
            "n_genes_tested": 3,
            "n_significant_genes": 2,
            "n_upregulated": 1,
            "n_downregulated": 1,
            "top_upregulated": ["gene_0"],
            "top_downregulated": ["gene_1"],
        },
    )


def run(tmp_path, monkeypatch, *, formula, x, counts_layer=None):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    adata = pseudobulk(matrix(x) if isinstance(x, str) else x)
    if counts_layer is not None:
        adata.layers["counts"] = (
            matrix(counts_layer) if isinstance(counts_layer, str) else counts_layer
        )
    dm.store_modality("pseudobulk", adata)
    log = Mock(wraps=dm.log_tool_usage)
    monkeypatch.setattr(dm, "log_tool_usage", log)

    science = Mock()
    science.validate_experimental_design.return_value = {"valid": True}
    science.create_formula_design.return_value = {}
    science.run_pydeseq2_from_pseudobulk.return_value = results()
    monkeypatch.setattr(f"{DE_MODULE}.BulkRNASeqService", lambda **_: science)
    monkeypatch.setattr(f"{SERVICE_MODULE}.BulkRNASeqService", lambda **_: science)
    tools = factory_tools(de_analysis_expert, DE_MODULE, dm)
    if formula:
        response = tools["run_de_with_formula"].invoke(
            {
                "pseudobulk_modality": "pseudobulk",
                "formula": "~condition",
                "contrast": ["condition", "sample_b", "sample_a"],
                "save_results": False,
            }
        )
    else:
        response = tools["run_differential_expression"].invoke(
            {
                "modality_name": "pseudobulk",
                "groupby": "condition",
                "group1": "sample_a",
                "group2": "sample_b",
                "save_result": False,
            }
        )
    return response, science.run_pydeseq2_from_pseudobulk, log


PATHS = pytest.mark.parametrize("formula", [False, True], ids=["ordinary", "formula"])


def test_tools_import_worktree_source():
    package = Path(__file__).resolve().parents[2]
    expected = package / "lobster/agents/transcriptomics/de_analysis_expert.py"
    assert Path(import_module(DE_MODULE).__file__).resolve() == expected
    core_source = (
        package.parents[1] / "lobster/services/analysis/bulk_rnaseq_service.py"
    )
    assert Path(import_module(SERVICE_MODULE).__file__).resolve() == core_source


@PATHS
@pytest.mark.parametrize("counts_layer", ["counts", "sparse_counts"])
def test_counts_layer_is_used_when_x_is_normalized(
    tmp_path, monkeypatch, formula, counts_layer
):
    response, service, _ = run(
        tmp_path,
        monkeypatch,
        formula=formula,
        x="normalized",
        counts_layer=counts_layer,
    )
    service.assert_called_once()
    assert service.call_args.kwargs["count_layer"] == "counts"
    assert "Differential Expression Analysis Complete" in response


@PATHS
def test_log_cpm_without_counts_is_refused_before_deseq2(
    tmp_path, monkeypatch, formula
):
    response, service, _ = run(tmp_path, monkeypatch, formula=formula, x="log_cpm")
    service.assert_not_called()
    assert "Raw counts required" in response
    assert "DESeq2 needs raw integer counts" in response
    assert "aggregated from normalized values" in response
    assert "Rebuild it from raw counts" in response


@PATHS
@pytest.mark.parametrize("x", ["counts", "integer_counts", "sparse_counts"])
def test_real_counts_in_x_run_unchanged(tmp_path, monkeypatch, formula, x):
    response, service, _ = run(tmp_path, monkeypatch, formula=formula, x=x)
    service.assert_called_once()
    assert service.call_args.kwargs.get("count_layer") is None
    assert "Differential Expression Analysis Complete" in response


@PATHS
def test_non_integer_counts_layer_is_refused(tmp_path, monkeypatch, formula):
    response, service, _ = run(
        tmp_path, monkeypatch, formula=formula, x="counts", counts_layer="normalized"
    )
    service.assert_not_called()
    assert "Raw counts required" in response
    assert "layer 'counts' contains" in response
    assert "non-integer values" in response


@PATHS
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -1.0])
@pytest.mark.parametrize("as_sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("in_layer", [False, True], ids=["X", "counts-layer"])
def test_invalid_values_are_refused(
    tmp_path, monkeypatch, formula, bad_value, as_sparse, in_layer
):
    invalid = COUNTS.copy()
    invalid[-1, -1] = bad_value
    if as_sparse:
        invalid = sparse.csr_matrix(invalid)
    response, service, _ = run(
        tmp_path,
        monkeypatch,
        formula=formula,
        x="counts" if in_layer else invalid,
        counts_layer=invalid if in_layer else None,
    )
    service.assert_not_called()
    assert "Raw counts required" in response


@PATHS
@pytest.mark.parametrize("count_layer", [None, "counts"])
def test_count_source_is_reported_stored_logged_and_replayed(
    tmp_path, monkeypatch, formula, count_layer
):
    response, service, log = run(
        tmp_path,
        monkeypatch,
        formula=formula,
        x="normalized" if count_layer else "counts",
        counts_layer=count_layer,
    )
    assert "Differential Expression Analysis Complete" in response
    source = "layers['counts']" if count_layer else "X"
    assert source in response
    assert "whole-number counts" in response
    assert "Used raw counts: Yes" not in response
    log.assert_called_once()
    logged = log.call_args.kwargs
    assert "whole-number counts" in logged["description"]
    assert logged["parameters"]["count_layer"] == count_layer
    assert logged["parameters"]["count_source"] == source
    ir = logged["ir"]
    assert ir.parameters["count_layer"] == count_layer
    assert ir.parameters["count_source"] == source
    assert "whole-number counts" in ir.description
    analyzed = service.call_args.kwargs["pseudobulk_adata"]
    key = analyzed.uns["de_current_results_key"]
    assert analyzed.uns[key]["parameters"]["count_layer"] == count_layer
    assert analyzed.uns[key]["parameters"]["count_source"] == source
    service.reset_mock()
    replay_ir(ir, adata=analyzed.copy())
    service.assert_called_once()
    assert service.call_args.kwargs["count_layer"] == count_layer


@PATHS
def test_dense_validation_uses_bounded_blocks(tmp_path, monkeypatch, formula):
    # More than one validation block, with an invalid value in the last row.
    x = np.ones((6, 2**20 // 5 + 1))
    x[-1, -1] = 0.5
    original_mod = np.mod
    block_sizes = []

    def checked_mod(values, divisor):
        block_sizes.append(values.size)
        return original_mod(values, divisor)

    monkeypatch.setattr(np, "mod", checked_mod)
    response, service, _ = run(tmp_path, monkeypatch, formula=formula, x=x)
    service.assert_not_called()
    assert "Raw counts required" in response
    assert len(block_sizes) > 1
    assert max(block_sizes) <= 2**20


def test_real_sparse_service_boundary_preserves_counts(tmp_path, monkeypatch):
    # Real layer -> DataFrame boundary; only downstream fitting is mocked.
    service = BulkRNASeqService(results_dir=tmp_path / "results")
    downstream = Mock(return_value=(*results(), None))
    monkeypatch.setattr(service, "run_pydeseq2_analysis", downstream)
    adata = pseudobulk(matrix("normalized"))
    adata.layers["counts"] = sparse.csr_matrix(COUNTS)
    service.run_pydeseq2_from_pseudobulk(
        pseudobulk_adata=adata,
        formula="~condition",
        contrast=["condition", "sample_b", "sample_a"],
        count_layer="counts",
    )
    downstream.assert_called_once()
    actual = downstream.call_args.args[0]
    expected = pd.DataFrame(COUNTS.T, index=adata.var_names, columns=adata.obs_names)
    pd.testing.assert_frame_equal(actual, expected)


REAL_CASES = pytest.mark.parametrize(
    "case",
    [
        "dense_x_csr_counts",
        "normalized_x_csr_float_whole_counts",
        "normalized_x_csc_counts",
        "anndata_view",
        "sparse_x_dense_counts",
        "sparse_x_fallback",
        "dense_x_fallback",
        "dense_x_dense_counts",
    ],
)


def real_source_case(case):
    counts = COUNTS.astype(np.int64)
    if case == "normalized_x_csr_float_whole_counts":
        return matrix("normalized"), sparse.csr_matrix(COUNTS), COUNTS
    if case == "normalized_x_csc_counts":
        return matrix("normalized"), sparse.csc_matrix(counts), counts
    if case == "anndata_view":
        return counts, sparse.csr_matrix(counts), counts[::-1]
    if case == "sparse_x_dense_counts":
        return sparse.csr_matrix(LOG_CPM), counts, counts
    if case == "sparse_x_fallback":
        return sparse.csr_matrix(counts), None, counts
    if case == "dense_x_fallback":
        return counts, None, counts
    if case == "dense_x_dense_counts":
        return counts, counts.copy(), counts
    return counts, sparse.csr_matrix(counts), counts


@PATHS
@REAL_CASES
def test_real_tools_preserve_selected_count_matrix(
    tmp_path, monkeypatch, formula, case
):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    x, layer, expected_values = real_source_case(case)
    adata = pseudobulk(x)
    if layer is not None:
        adata.layers["counts"] = layer
    if case == "anndata_view":
        adata = adata[::-1, :]
        assert adata.is_view
    dm.store_modality("pseudobulk", adata)
    log = Mock(wraps=dm.log_tool_usage)
    monkeypatch.setattr(dm, "log_tool_usage", log)
    # Do not mock the agent service, formula design or sparse adapter.
    fitter = Mock(return_value=(*results(), None))
    monkeypatch.setattr(BulkRNASeqService, "run_pydeseq2_analysis", fitter)
    tools = factory_tools(de_analysis_expert, DE_MODULE, dm)
    if formula:
        response = tools["run_de_with_formula"].invoke(
            {
                "pseudobulk_modality": "pseudobulk",
                "formula": "~condition",
                "contrast": ["condition", "sample_b", "sample_a"],
                "save_results": False,
            }
        )
    else:
        response = tools["run_differential_expression"].invoke(
            {
                "modality_name": "pseudobulk",
                "groupby": "condition",
                "group1": "sample_a",
                "group2": "sample_b",
                "save_result": False,
            }
        )
    assert "Differential Expression Analysis Complete" in response
    fitter.assert_called_once()
    expected = pd.DataFrame(
        expected_values.T, index=adata.var_names, columns=adata.obs_names
    )
    pd.testing.assert_frame_equal(fitter.call_args.args[0], expected)
    pd.testing.assert_frame_equal(fitter.call_args.args[1], adata.obs)
    assert fitter.call_args.args[2:] == (
        "~condition",
        ["condition", "sample_b", "sample_a"],
    )
    source = "layers['counts']" if layer is not None else "X"
    assert source in response
    log.assert_called_once()
    logged = log.call_args.kwargs
    assert logged["parameters"]["count_source"] == source
    assert logged["parameters"]["count_layer"] == (
        "counts" if layer is not None else None
    )
    analyzed = dm.get_modality("pseudobulk")
    key = analyzed.uns["de_current_results_key"]
    assert analyzed.uns[key]["parameters"]["count_source"] == source
    assert logged["ir"].parameters["count_source"] == source
    fitter.reset_mock()
    replay_ir(logged["ir"], adata=analyzed.copy())
    fitter.assert_called_once()
    pd.testing.assert_frame_equal(fitter.call_args.args[0], expected)


@PATHS
@pytest.mark.parametrize("in_layer", [False, True], ids=["X", "counts-layer"])
def test_real_tools_keep_masked_dense_cell_missing_and_refuse_before_fitting(
    tmp_path, monkeypatch, formula, in_layer
):
    # Keep real agent, selected-source conversion, design and int-cast validation.
    # Only the actual pyDESeq2 fitter construction is replaced; never fit biology.
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    masked = np.ma.array(COUNTS.copy(), mask=np.zeros(COUNTS.shape, dtype=bool))
    masked.mask[0, 0] = True
    adata = pseudobulk(matrix("normalized") if in_layer else masked)
    if in_layer:
        adata.layers["counts"] = masked
    dm.store_modality("pseudobulk", adata)
    selected = adata.layers["counts"] if in_layer else adata.X
    assert np.ma.isMaskedArray(selected) and bool(selected.mask[0, 0])
    expected = pd.DataFrame(selected.T, index=adata.var_names, columns=adata.obs_names)
    assert np.isnan(expected.iloc[0, 0])
    fitter = Mock(side_effect=AssertionError("Hidden masked count reached fitting"))
    monkeypatch.setattr("pydeseq2.dds.DeseqDataSet", fitter)
    actual_analysis = BulkRNASeqService.run_pydeseq2_analysis
    captured = []

    def checked_analysis(service, count_matrix, metadata, *args, **kwargs):
        captured.append(count_matrix)
        return actual_analysis(service, count_matrix, metadata, *args, **kwargs)

    monkeypatch.setattr(BulkRNASeqService, "run_pydeseq2_analysis", checked_analysis)
    tools = factory_tools(de_analysis_expert, DE_MODULE, dm)
    if formula:
        response = tools["run_de_with_formula"].invoke(
            {
                "pseudobulk_modality": "pseudobulk",
                "formula": "~condition",
                "contrast": ["condition", "sample_b", "sample_a"],
                "save_results": False,
            }
        )
    else:
        response = tools["run_differential_expression"].invoke(
            {
                "modality_name": "pseudobulk",
                "groupby": "condition",
                "group1": "sample_a",
                "group2": "sample_b",
                "save_result": False,
            }
        )
    assert len(captured) == 1
    pd.testing.assert_frame_equal(captured[0], expected)
    fitter.assert_not_called()
    assert "Cannot convert non-finite values (NA or inf) to integer" in response
    assert "Differential Expression Analysis Complete" not in response


@pytest.mark.parametrize("in_layer", [False, True], ids=["X", "counts-layer"])
def test_real_sparse_adapter_refuses_insufficient_memory_before_fit(
    tmp_path, monkeypatch, in_layer
):
    service = BulkRNASeqService(results_dir=tmp_path / "results")
    fitter = Mock(return_value=(*results(), None))
    monkeypatch.setattr(service, "run_pydeseq2_analysis", fitter)
    adata = pseudobulk(COUNTS.astype(np.int64))
    counts = sparse.csr_matrix(COUNTS.astype(np.int64))
    if in_layer:
        adata.layers["counts"] = counts
    else:
        adata.X = counts
    monkeypatch.setattr(
        "lobster.core.sparse_utils.psutil.virtual_memory",
        lambda: SimpleNamespace(available=0),
    )
    with pytest.raises(PyDESeq2Error) as error:
        service.run_pydeseq2_from_pseudobulk(
            adata,
            "~condition",
            ["condition", "sample_b", "sample_a"],
            count_layer="counts" if in_layer else None,
        )
    assert isinstance(error.value.__context__, SparseConversionError)
    assert error.value.__context__.shape == (3, 6)
    assert error.value.__context__.available_gb == 0
    assert str(error.value.__context__).startswith(
        "Insufficient memory for dense conversion of 3 x 6 sparse matrix."
    )
    fitter.assert_not_called()
