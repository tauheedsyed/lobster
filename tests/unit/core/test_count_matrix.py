"""Shared count checks provide numeric evidence, not raw-count provenance."""

import numpy as np
import pytest
from scipy import sparse


@pytest.mark.parametrize("as_sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize(
    "values,expected",
    [
        ([[0, 1, 2], [3, 4, 5]], True),
        ([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], True),
        ([[0, 1, 2], [3, 4, 5.5]], False),
        ([[0, 1, 2], [3, 4, -1]], False),
        ([[0, 1, 2], [3, 4, np.nan]], False),
        ([[0, 1, 2], [3, 4, np.inf]], False),
        ([[0, 1, 2], [3, 4, -np.inf]], False),
    ],
    ids=["integers", "whole-floats", "fraction", "negative", "nan", "inf", "neg-inf"],
)
def test_whole_number_matrix(values, expected, as_sparse):
    from lobster.core.utils.count_matrix import is_whole_number_matrix

    matrix = np.asarray(values)
    if as_sparse:
        matrix = sparse.csr_matrix(matrix)
    assert is_whole_number_matrix(matrix) is expected


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_large_integers_remain_valid_without_float_conversion(dtype):
    from lobster.core.utils.count_matrix import is_whole_number_matrix

    matrix = np.array([[2**53 + 1, 2**53 + 3]], dtype=dtype)
    before = matrix.copy()
    assert is_whole_number_matrix(matrix)
    np.testing.assert_array_equal(matrix, before)
    assert matrix.dtype == dtype


@pytest.mark.parametrize("values", [[["a", "b"]], [[1 + 2j, 2 + 0j]]])
def test_non_real_numeric_values_are_not_counts(values):
    from lobster.core.utils.count_matrix import is_whole_number_matrix

    assert is_whole_number_matrix(np.asarray(values)) is False


@pytest.mark.parametrize("as_sparse", [False, True], ids=["dense", "sparse"])
def test_validation_is_block_bounded_and_reaches_the_last_value(monkeypatch, as_sparse):
    from lobster.core.utils import count_matrix

    # Exercise multiple blocks without allocating a large fixture on test machines.
    monkeypatch.setattr(count_matrix, "_BLOCK", 8)
    matrix = np.ones((7, 3))
    matrix[-1, -1] = 0.5
    if as_sparse:
        matrix = sparse.csr_matrix(matrix)
    original_mod = np.mod
    sizes = []

    def checked_mod(values, divisor):
        sizes.append(values.size)
        return original_mod(values, divisor)

    monkeypatch.setattr(np, "mod", checked_mod)
    assert count_matrix.is_whole_number_matrix(matrix) is False
    assert len(sizes) > 1
    assert max(sizes) <= 8


@pytest.mark.parametrize(
    "totals,expected",
    [
        ([1_000_000, 1_000_000, 1_000_000], True),
        ([30, 45, 60], False),
        ([1_000_000, 1_000_000.5], True),
        ([1_000_000, 1_000_010], False),
        ([], False),
        ([30], False),
        ([np.nan, np.nan], False),
        ([np.inf, np.inf], False),
    ],
    ids=[
        "constant",
        "unequal",
        "rounding",
        "different",
        "empty",
        "single",
        "nan",
        "inf",
    ],
)
def test_equal_sample_totals_are_a_numeric_signature(totals, expected):
    from lobster.core.utils.count_matrix import has_equal_sample_totals

    assert has_equal_sample_totals(totals) is expected
