"""Plot tools log exportable IR that redraws the exact same figure.

Drawing consumes stored QC outputs and never changes scientific data. The
controlled real PlotManager stages and discards the drawing without publishing.
"""

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from langchain_core.messages import ToolMessage

from lobster.agents.visualization_expert import visualization_expert
from lobster.core.figures.store import LocalFigureStore
from lobster.core.plot_manager import PlotManager
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.testing.provenance_replay import (
    RecordingDataManager,
    factory_tools,
    replay_ir,
)
from tests.qc_fixtures import prepare_stored_qc

MODULE = "lobster.agents.visualization_expert"
GENES = ["gene_0", "gene_1", "gene_2"]


@pytest.fixture
def cells() -> ad.AnnData:
    """Full plot-ready cells with stored QC outputs; numpy only, no scanpy."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 80, 1000
    counts = rng.poisson(3, size=(n_cells, n_genes)).astype(np.float32)
    detected = (counts > 0).sum(axis=1)
    totals = counts.sum(axis=1)
    mito = rng.uniform(0, 10, n_cells)
    obs = pd.DataFrame(
        {
            "cluster": pd.Categorical([str(i % 3) for i in range(n_cells)]),
            "sample": pd.Categorical(["s1", "s2"] * (n_cells // 2)),
            "n_genes_by_counts": detected,
            "total_counts": totals,
            "pct_counts_mt": mito,
            # Upstream assessment percentages, supplied before drawing.
            "mt_pct": mito,
            "ribo_pct": np.zeros(n_cells),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    adata = ad.AnnData(
        X=counts.copy(),
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )
    prepare_stored_qc(adata)
    adata.raw = adata.copy()
    adata.X = np.log1p(counts)
    adata.obsm["X_pca"] = rng.normal(size=(n_cells, 10))
    adata.obsm["X_umap"] = rng.normal(size=(n_cells, 2))
    ratios = np.sort(rng.uniform(0.01, 0.2, 10))[::-1]
    adata.uns["pca"] = {"variance_ratio": ratios, "variance": ratios * 5}
    return adata


CASES = [
    ("create_umap_plot", {"color_by": "cluster", "point_size": 4}),
    ("create_qc_plots", {}),
    ("create_violin_plot", {"genes": GENES, "groupby": "cluster"}),
    ("create_feature_plot", {"genes": GENES[:2], "use_raw": False}),
    ("create_dot_plot", {"genes": GENES, "groupby": "cluster", "use_raw": False}),
    (
        "create_heatmap",
        {"groupby": "cluster", "genes": GENES, "use_raw": False},
    ),
    ("create_elbow_plot", {"n_pcs": 10}),
    (
        "create_cluster_composition_plot",
        {"cluster_col": "cluster", "sample_col": "sample"},
    ),
]


def assert_same_data(actual, expected):
    assert actual.shape == expected.shape == (80, 1000)
    np.testing.assert_array_equal(actual.X, expected.X)
    pd.testing.assert_frame_equal(actual.obs, expected.obs)
    pd.testing.assert_frame_equal(actual.var, expected.var)
    assert set(actual.obsm) == set(expected.obsm)
    for key in expected.obsm:
        np.testing.assert_array_equal(actual.obsm[key], expected.obsm[key])
    np.testing.assert_equal(actual.uns, expected.uns)
    assert actual.raw.shape == expected.raw.shape
    np.testing.assert_array_equal(actual.raw.X, expected.raw.X)
    pd.testing.assert_index_equal(actual.raw.obs_names, expected.raw.obs_names)
    pd.testing.assert_frame_equal(actual.raw.var, expected.raw.var)


@pytest.mark.parametrize("tool_name,args", CASES, ids=[c[0] for c in CASES])
def test_plot_ir_redraws_the_same_figure(cells, monkeypatch, tmp_path, tool_name, args):
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)
    original = cells.copy()
    dm = RecordingDataManager({"cells": cells}, workspace_path=tmp_path)
    store = LocalFigureStore(tmp_path, session_id="ir-redraw")
    manager = PlotManager(tmp_path)
    manager.configure_figure_lifecycle(
        store_provider=lambda: store, session_id_provider=lambda: "ir-redraw"
    )
    dm.plot_manager = manager
    produced = []
    real_add_plot = manager.add_plot

    def capture_drawing(**kwargs):
        produced.append(kwargs["plot"])
        return real_add_plot(**kwargs)

    monkeypatch.setattr(manager, "add_plot", capture_drawing)
    tools = factory_tools(visualization_expert, MODULE, dm)
    call_id = f"ir-{tool_name}"
    result = tools[tool_name].invoke(
        {
            "type": "tool_call",
            "name": tool_name,
            "id": call_id,
            "args": {"modality_name": "cells", "save_plot": False, **args},
        }
    )
    assert isinstance(result, ToolMessage)
    reply = result.content
    assert isinstance(reply, str)
    assert "STATUS: FAILED" not in reply
    assert "not saved" in reply
    assert "figure_ref=" not in reply
    assert len(produced) == 1
    assert dm.calls_for(tool_name)[-1].parameters["tool_call_id"] == call_id

    ir = dm.last_ir(tool_name)
    assert isinstance(ir, AnalysisStep)
    assert ir.exportable is True
    assert ir.replayable is False
    ir.validate_rendered_code()

    scope = replay_ir(ir, adata=original.copy())
    assert scope["fig"].to_json() == produced[0].to_json()
    assert_same_data(cells, original)
    assert_same_data(scope["adata"], original)
    assert store.list().items == []
    assert manager._staged == {}
    assert manager.visualization_state["history"] == []
    assert manager.visualization_state["plot_registry"] == {}
