"""Count/abundance/genotype checks on dense matrices.

A dense ndarray has a ``.data`` attribute (a memoryview), so ``hasattr(X,
"data")`` sent dense matrices down the sparse branch: ``memoryview % 1`` and
``(memoryview == 0).sum()`` raised, and the validator only reported
"Custom rule '...' failed". That hit every dense float bulk RNA-seq matrix.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from lobster.core.schemas.genomics import _validate_genotype_data
from lobster.core.schemas.metagenomics import _validate_gene_abundance
from lobster.core.schemas.transcriptomics import (
    TranscriptomicsSchema,
    _validate_count_data,
)


def _bulk(X) -> ad.AnnData:
    n_obs, n_vars = X.shape
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"s{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )


def _tpm(n_obs=20, n_vars=500):
    raw = np.random.default_rng(0).gamma(0.5, 2.0, size=(n_obs, n_vars))
    return (raw / raw.sum(axis=1, keepdims=True) * 1e6).astype(np.float32)


@pytest.mark.parametrize("as_sparse", [False, True], ids=["dense", "sparse"])
def test_float_tpm_reports_non_integer(as_sparse):
    X = _tpm()
    expected = int(np.count_nonzero(X % 1))
    result = _validate_count_data(_bulk(sparse.csr_matrix(X) if as_sparse else X))
    assert result.info == [
        f"Found {expected} non-integer values (may be normalized data)"
    ]


def test_dense_float_integer_counts_report_nothing():
    X = np.random.default_rng(1).poisson(5, size=(20, 300)).astype(np.float64)
    assert _validate_count_data(_bulk(X)).info == []


def test_dense_blocks_cover_every_row():
    X = np.zeros((9, 2**21 + 1), dtype=np.float16)  # blocks of 7 rows: 0-6, 7-8
    X[8, -1] = 0.5
    assert _validate_count_data(_bulk(X)).info == [
        "Found 1 non-integer values (may be normalized data)"
    ]


def test_bulk_validator_runs_count_rule_on_dense_tpm():
    validator = TranscriptomicsSchema.create_validator("bulk")
    result = validator.validate(_bulk(_tpm()))
    messages = result.errors + result.warnings + result.info
    assert not any("check_count_data" in m for m in messages), messages
    assert any("non-integer values" in m for m in result.info), messages


def test_dense_abundance_sparsity_check():
    X = np.zeros((10, 10), dtype=np.float32)
    X[0, :5] = 1.0  # 95% zeros
    result = _validate_gene_abundance(_bulk(X))
    assert any("95.0% of gene abundance values are zero" in w for w in result.warnings)


def test_dense_genotype_missing_rate():
    adata = _bulk(np.zeros((4, 5), dtype=np.float32))
    gt = np.zeros((4, 5), dtype=np.int8)
    gt[0, :2] = -1  # 2 of 20 missing
    adata.layers["GT"] = gt
    result = _validate_genotype_data(adata)
    assert "Missing genotype rate: 10.00%" in result.info
