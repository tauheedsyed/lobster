"""The exported marker cell compares the same groups as the session.

Found in a manual CLI test: ``find_marker_genes(groups=[...])`` passed
``groups`` to scanpy, but the exported cell did not, so it tested every group
(in default order). With three or more groups the results differ.
"""

import anndata
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from lobster.services.analysis.enhanced_singlecell_service import (
    EnhancedSingleCellService,
)


def _adata():
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, size=(90, 30)).astype(np.float32)
    counts[:30, :5] += 6  # group "a" markers
    adata = anndata.AnnData(counts)
    adata.var_names = [f"g{i}" for i in range(30)]
    adata.obs["grp"] = pd.Categorical(np.repeat(["a", "b", "c"], 30))
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


@pytest.mark.parametrize("groups", [None, ["c", "a"]])
def test_exported_marker_cell_matches_the_session(groups):
    adata = _adata()
    result, _, ir = EnhancedSingleCellService().find_marker_genes(
        adata.copy(), groupby="grp", groups=groups, n_genes=10
    )
    replay = adata.copy()
    exec(compile(ir.render(), "marker_cell", "exec"), {"adata": replay})  # nosec B102

    session = result.uns["rank_genes_groups"]
    exported = replay.uns["rank_genes_groups"]
    assert session["names"].dtype.names == exported["names"].dtype.names
    for group in session["names"].dtype.names:
        assert list(session["names"][group]) == list(exported["names"][group])
        assert np.allclose(session["scores"][group], exported["scores"][group])
