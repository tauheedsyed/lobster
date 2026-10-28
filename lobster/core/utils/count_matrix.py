"""Numeric count checks. A pass is evidence about values, not raw provenance."""

from __future__ import annotations

import numpy as np
from scipy import sparse

_BLOCK = 2**20


def _blocks(matrix):
    """Yield views or row blocks to bound validation temporaries."""
    if sparse.issparse(matrix):
        values = matrix.data
        for start in range(0, values.size, _BLOCK):
            yield values[start : start + _BLOCK]
    else:
        step = max(1, _BLOCK // max(1, matrix.shape[1]))
        for start in range(0, matrix.shape[0], step):
            yield np.asarray(matrix[start : start + step])


def is_whole_number_matrix(matrix) -> bool:
    """Every value is finite, non-negative and whole; never convert counts to float."""
    return all(
        block.dtype.kind in "biuf"
        and np.all(np.isfinite(block))
        and np.all(block >= 0)
        and np.all(np.mod(block, 1) == 0)
        for block in _blocks(matrix)
    )


def has_equal_sample_totals(totals, rel_tol: float = 1e-6) -> bool:
    """Flag a constant-total numeric signature, not proof of normalization."""
    totals = np.asarray(totals, dtype=float).ravel()
    return bool(
        totals.size > 1
        and np.all(np.isfinite(totals))
        and np.allclose(totals, totals[0], rtol=rel_tol, atol=0)
    )
