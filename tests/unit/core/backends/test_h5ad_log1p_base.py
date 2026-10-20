"""Saved and reloaded log-transformed data keeps a log1p base scanpy can use.

Found in a manual CLI test: the uns sanitizer wrote ``uns['log1p']['base']``
(None for natural log) as ``""``. scanpy reads ``uns['log1p'].get('base')``
and calls ``np.log`` on anything that is not None, so after reloading
``sc.tl.rank_genes_groups`` failed with a TypeError.
"""

import anndata
import numpy as np
import pytest

from lobster.core.backends.h5ad_backend import H5ADBackend


def _adata(log1p):
    adata = anndata.AnnData(np.ones((4, 3), dtype=np.float32))
    adata.uns["log1p"] = log1p
    return adata


def _base_as_scanpy_reads_it(adata):
    base = adata.uns.get("log1p", {}).get("base")
    if base is not None:
        np.log(base)  # what scanpy does; raises TypeError for strings
    return base


@pytest.mark.parametrize("base", [None, 2])
def test_log1p_base_survives_save_and_load(tmp_path, base):
    backend = H5ADBackend(base_path=tmp_path)
    backend.save(_adata({"base": base}), tmp_path / "d.h5ad")

    loaded = backend.load(tmp_path / "d.h5ad")

    assert _base_as_scanpy_reads_it(loaded) == base


def test_file_saved_with_empty_base_is_repaired_on_load(tmp_path):
    _adata({"base": ""}).write_h5ad(tmp_path / "old.h5ad")  # earlier versions

    loaded = H5ADBackend(base_path=tmp_path).load(tmp_path / "old.h5ad")

    assert _base_as_scanpy_reads_it(loaded) is None
