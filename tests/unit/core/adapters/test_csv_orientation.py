"""CSVs that mix text columns and counts fail with a reason, not an empty matrix.

Regression for "No variables in dataset" on a cells x genes CSV with cluster
and donor columns: the CSV default transposes (genes as rows), after which
every column held a text value and the numeric filter dropped the whole
matrix. The loader does not guess the orientation; it says what happened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter

N_CELLS, N_GENES = 30, 12


@pytest.fixture
def counts():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.poisson(2, (N_CELLS, N_GENES)),
        index=pd.Index([f"cell{i}" for i in range(N_CELLS)], name="cell_id"),
        columns=[f"G{i}" for i in range(N_GENES)],
    )


@pytest.fixture
def mixed_csv(tmp_path, counts):
    df = counts.copy()
    df.insert(0, "cluster", ["T" if i % 2 else "B" for i in range(N_CELLS)])
    df.insert(1, "donor", ["d1" if i < N_CELLS // 2 else "d2" for i in range(N_CELLS)])
    path = tmp_path / "mixed.csv"
    df.to_csv(path)
    return path


def _load(path, **kwargs):
    return TranscriptomicsAdapter("single_cell").from_source(str(path), **kwargs)


def test_mixed_text_and_counts_fail_naming_the_text_columns(mixed_csv):
    with pytest.raises(ValueError) as info:
        _load(mixed_csv)
    message = str(info.value)
    assert "2 text column(s) (cluster, donor)" in message
    assert f"{N_GENES} numeric columns" in message
    assert "rows as observations" in message
    assert "No variables" not in message


def test_explicit_rows_as_observations_still_loads_the_counts(mixed_csv, counts):
    adata = _load(mixed_csv, transpose=False)
    assert adata.shape == (N_CELLS, N_GENES)
    assert list(adata.obs_names) == list(counts.index)
    assert list(adata.var_names) == list(counts.columns)
    np.testing.assert_array_equal(np.asarray(adata.X), counts.to_numpy())


def test_all_numeric_csv_keeps_the_genes_as_rows_default(tmp_path, counts):
    path = tmp_path / "pure.csv"
    counts.to_csv(path)
    adata = _load(path)
    assert adata.shape == (N_GENES, N_CELLS)
    assert list(adata.obs_names) == list(counts.columns)


def test_numeric_categorical_columns_still_transpose():
    from lobster.core.adapters.base import BaseAdapter

    df = pd.DataFrame(
        {"s1": pd.Categorical([1, 2]), "s2": [3, 4]},
        index=pd.Index(["g1", "g2"]),
    )
    adapter = TranscriptomicsAdapter("single_cell")
    adata = BaseAdapter._create_anndata_from_dataframe(adapter, df, transpose=True)
    assert adata.shape == (2, 2)
    assert list(adata.obs_names) == ["s1", "s2"]
