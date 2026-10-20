"""Saved and reloaded log-transformed data still works with scanpy.

Found in a manual CLI test: the uns sanitizer wrote ``uns['log1p']['base']``
(None for natural log) as ``""``. After reloading, ``rank_genes_groups``
called ``np.log("")`` and failed.
"""

import anndata
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from lobster.core.backends.h5ad_backend import H5ADBackend


def _adata(base=None):
    rng = np.random.default_rng(0)
    adata = anndata.AnnData(rng.poisson(2.0, size=(40, 20)).astype(np.float32))
    adata.var_names = [f"g{i}" for i in range(20)]
    adata.obs["group"] = pd.Categorical(["a", "b"] * 20)
    sc.pp.log1p(adata, base=base)
    return adata


@pytest.mark.parametrize("base", [None, 2])
def test_log1p_base_survives_save_and_load(tmp_path, base):
    backend = H5ADBackend(base_path=tmp_path)
    backend.save(_adata(base), tmp_path / "d.h5ad")

    loaded = backend.load(tmp_path / "d.h5ad")

    assert loaded.uns["log1p"].get("base") == base
    sc.tl.rank_genes_groups(loaded, "group", method="wilcoxon")


def test_file_saved_with_empty_base_is_repaired_on_load(tmp_path):
    adata = _adata()
    adata.uns["log1p"] = {"base": ""}  # what earlier versions wrote
    adata.write_h5ad(tmp_path / "old.h5ad")

    loaded = H5ADBackend(base_path=tmp_path).load(tmp_path / "old.h5ad")

    assert loaded.uns["log1p"].get("base") is None
    sc.tl.rank_genes_groups(loaded, "group", method="wilcoxon")
