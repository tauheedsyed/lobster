"""Synthetic stored assessment outputs shared by visualization tests only."""

import numpy as np


def prepare_stored_qc(adata):
    """Prepare fixture values with the assessment's percent/log1p conventions.

    This is test preparation, not a product QC API or a drawing-side analysis.
    Fixture matrices are nonnegative counts. Stored mitochondrial/ribosomal
    percentages are supplied by the fixture, or zero for unannotated genes.
    """
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    detected = np.asarray((adata.X != 0).sum(axis=1)).ravel()
    adata.obs["n_counts"] = totals
    adata.obs["n_genes"] = detected
    adata.obs["gene_detection_rate"] = detected / max(adata.n_vars, 1) * 100
    adata.obs["log10_counts"] = np.log10(totals + 1)
    adata.obs["log10_genes"] = np.log10(detected + 1)
    for source, target in (("mt_pct", "percent_mito"), ("ribo_pct", "percent_ribo")):
        adata.obs[target] = adata.obs[source].to_numpy() if source in adata.obs else 0.0
    hb_mask = adata.var_names.str.startswith(("HBA", "HBB", "hba", "hbb"))
    hb_counts = np.asarray(adata[:, hb_mask].X.sum(axis=1)).ravel()
    adata.obs["percent_hb"] = np.divide(
        hb_counts * 100,
        totals,
        out=np.zeros(adata.n_obs, dtype=float),
        where=totals != 0,
    )
    return adata
