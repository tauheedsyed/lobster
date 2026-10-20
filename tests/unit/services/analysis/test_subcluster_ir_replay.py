"""Exported sub-clustering code reproduces the session's labels.

Found in a manual CLI test: the exported cell wrote "<parent>.<sub>" labels
into the categorical Leiden column and pandas raised. It also skipped the PCA
the service runs on the subset, so labels could differ even without the error.
"""

import anndata
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from lobster.services.analysis.clustering_service import ClusteringService

pytest.importorskip("leidenalg")

HOSTILE_KEY = "cl'\"\n{k}\\"
PARENTS = ["a'b", 'c"\n{d}\\']


def _adata():
    rng = np.random.default_rng(0)
    adata = anndata.AnnData(rng.poisson(1.0, size=(150, 60)).astype(np.float32))
    adata.var_names = [f"g{i}" for i in range(60)]
    adata.obs_names = [f"c{i}" for i in range(150)]
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=20)
    labels = np.array(PARENTS + ["other"])[np.arange(150) % 3]
    adata.obs[HOSTILE_KEY] = pd.Categorical(labels)
    return adata


@pytest.mark.parametrize("resolutions", [None, [0.3, 0.8]])
def test_exported_subclustering_matches_the_session(resolutions):
    adata = _adata()
    result, _, ir = ClusteringService().subcluster_cells(
        adata.copy(),
        cluster_key=HOSTILE_KEY,
        clusters_to_refine=PARENTS,
        resolution=0.5,
        resolutions=resolutions,
        n_pcs=10,
        n_neighbors=10,
    )
    replay = adata.copy()
    namespace = {"adata": replay}
    exec(compile(ir.render(), "subcluster_cell", "exec"), namespace)  # nosec B102

    keys = [c for c in result.obs.columns if c.startswith("leiden_sub")]
    assert keys
    for key in keys:
        assert list(replay.obs[key]) == list(result.obs[key].astype(str)), key
