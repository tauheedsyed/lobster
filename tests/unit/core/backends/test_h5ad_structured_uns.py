"""Marker-gene tables survive Lobster's save and load.

Found in a manual CLI test: ``sc.tl.rank_genes_groups`` stores ``names``,
``scores``, ``pvals`` ... as structured arrays with one field per group. The
uns sanitizer stringified each row, so the saved session lost the groups while
an exported notebook (plain anndata write) kept them.
"""

import anndata
import numpy as np

from lobster.core.backends.h5ad_backend import H5ADBackend

GROUPS = ["a'b", 'c"\n{d}\\']


def _rank_genes_groups():
    names = np.rec.fromarrays(
        [np.array(["g1", "g2"], dtype=object), np.array(["g3", "g4"], dtype=object)],
        names=GROUPS,
    )
    scores = np.rec.fromarrays(
        [
            np.array([2.5, 1.0], dtype=np.float32),
            np.array([3.0, 0.5], dtype=np.float32),
        ],
        names=GROUPS,
    )
    return {"params": {"groupby": "cluster"}, "names": names, "scores": scores}


def test_rank_genes_groups_fields_survive_save_and_load(tmp_path):
    adata = anndata.AnnData(np.ones((4, 3), dtype=np.float32))
    adata.uns["rank_genes_groups"] = _rank_genes_groups()
    backend = H5ADBackend(base_path=tmp_path)

    backend.save(adata, tmp_path / "d.h5ad")
    loaded = backend.load(tmp_path / "d.h5ad").uns["rank_genes_groups"]

    for key in ("names", "scores"):
        assert list(loaded[key].dtype.names) == GROUPS, key
    assert list(loaded["names"][GROUPS[1]]) == ["g3", "g4"]
    assert np.allclose(loaded["scores"][GROUPS[0]], [2.5, 1.0])
