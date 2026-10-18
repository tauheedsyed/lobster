"""Plot tools log IR that redraws the same figure.

Plots change no data, so their IR is exported (the notebook redraws the figure)
but not replayable. The replayed figure must equal the tool's figure.
"""

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from lobster.agents.visualization_expert import visualization_expert
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.testing.provenance_replay import (
    RecordingDataManager,
    factory_tools,
    replay_ir,
)

MODULE = "lobster.agents.visualization_expert"
GENES = ["gene_0", "gene_1", "gene_2"]


@pytest.fixture
def cells() -> ad.AnnData:
    """Plot-ready cells built with numpy only (the package CI has no scanpy)."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 80, 1000  # create_qc_plots refuses subset-sized matrices
    counts = rng.poisson(3, size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "cluster": pd.Categorical([str(i % 3) for i in range(n_cells)]),
            "sample": pd.Categorical(["s1", "s2"] * (n_cells // 2)),
            "n_genes_by_counts": (counts > 0).sum(axis=1),
            "total_counts": counts.sum(axis=1),
            "pct_counts_mt": rng.uniform(0, 10, n_cells),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    adata = ad.AnnData(
        X=np.log1p(counts),
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )
    adata.raw = ad.AnnData(X=counts, obs=obs, var=adata.var.copy())
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


@pytest.mark.parametrize("tool_name,args", CASES, ids=[c[0] for c in CASES])
def test_plot_ir_redraws_the_same_figure(cells, monkeypatch, tool_name, args):
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)
    original = cells.copy()  # replay input: the data before the tool ran
    dm = RecordingDataManager({"cells": cells})
    tools = factory_tools(visualization_expert, MODULE, dm)

    result = tools[tool_name].invoke(
        {"modality_name": "cells", "save_plot": False, **args}
    )
    assert "❌" not in result, result

    ir = dm.last_ir(tool_name)
    assert isinstance(ir, AnalysisStep)
    assert ir.exportable is True
    ir.validate_rendered_code()

    produced = dm.plot_manager.add_plot.call_args.kwargs["plot"]
    scope = replay_ir(ir, adata=original.copy())
    assert scope["fig"].to_json() == produced.to_json()

    # A plot may only be non-replayable if it left the data unchanged.
    changed_obs = sorted(set(cells.obs.columns) - set(original.obs.columns))
    if changed_obs:
        assert ir.replayable is True, f"{tool_name} adds {changed_obs} but is not replayable"
        pd.testing.assert_frame_equal(scope["adata"].obs, cells.obs)
    else:
        assert ir.replayable is False
    if tool_name == "create_qc_plots":
        assert "percent_mito" in changed_obs  # the QC service writes QC metrics
