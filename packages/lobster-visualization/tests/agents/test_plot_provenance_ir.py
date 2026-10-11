"""Plot tools log IR that redraws the same figure.

Plots change no data, so their IR is exported (the notebook redraws the figure)
but not replayable. The replayed figure must equal the tool's figure.
"""

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import scanpy as sc

from lobster.agents.visualization_expert import visualization_expert
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.testing.provenance_replay import (
    RecordingDataManager,
    factory_tools,
    replay_ir,
)

MODULE = "lobster.agents.visualization_expert"
GENES = ["gene_0", "gene_1", "gene_2"]


@pytest.fixture(scope="module")
def cells() -> ad.AnnData:
    rng = np.random.default_rng(0)
    adata = ad.AnnData(
        # 1000 genes: create_qc_plots refuses smaller (subset-like) matrices.
        X=rng.poisson(3, size=(80, 1000)).astype(np.float32),
        obs=pd.DataFrame(
            {
                "cluster": pd.Categorical([str(i % 3) for i in range(80)]),
                "sample": pd.Categorical(["s1", "s2"] * 40),
            },
            index=[f"cell_{i}" for i in range(80)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(1000)]),
    )
    adata.var["mt"] = False
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)
    adata.raw = adata.copy()
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=10)
    sc.pp.neighbors(adata, n_neighbors=10)
    sc.tl.umap(adata, random_state=0)
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
    dm = RecordingDataManager({"cells": cells})
    tools = factory_tools(visualization_expert, MODULE, dm)

    result = tools[tool_name].invoke(
        {"modality_name": "cells", "save_plot": False, **args}
    )
    assert "❌" not in result, result

    ir = dm.last_ir(tool_name)
    assert isinstance(ir, AnalysisStep)
    assert ir.replayable is False and ir.exportable is True
    ir.validate_rendered_code()

    produced = dm.plot_manager.add_plot.call_args.kwargs["plot"]
    replayed = replay_ir(ir, adata=cells.copy())["fig"]
    assert replayed.to_json() == produced.to_json()
