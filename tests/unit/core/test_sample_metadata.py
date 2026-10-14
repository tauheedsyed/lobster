"""Exact sample identity and annotation alignment without dataset persistence."""

import numpy as np
import pandas as pd
import pytest

from lobster.core.utils.sample_metadata import merge_obs_metadata, read_sample_metadata


def _table_path(tmp_path, table, suffix):
    path = tmp_path / f"annotations.{suffix}"
    if suffix == "xlsx":
        pytest.importorskip("openpyxl")
        table.to_excel(path, index=False)
    else:
        table.to_csv(path, index=False, sep="\t" if suffix == "tsv" else ",")
    return path


@pytest.mark.parametrize("sample_id_column", ["auto", "id"])
@pytest.mark.parametrize("suffix", ["csv", "tsv", "xlsx"])
@pytest.mark.parametrize("ids", [["001"], ["NA", "nan", "null", "N/A"]])
def test_exact_sample_ids(tmp_path, sample_id_column, suffix, ids):
    names = pd.Index(["001", "1", "NA", "nan", "null", "N/A", "unmatched"])
    obs = pd.DataFrame(
        {"condition": [f"original-{name}" for name in names]}, index=names
    )
    labels = [f"changed-{name}" for name in ids]
    path = _table_path(tmp_path, pd.DataFrame({"id": ids, "condition": labels}), suffix)
    metadata, column = read_sample_metadata(path, names, sample_id_column)
    assert column == "id"
    assert list(metadata.index) == ids
    before = obs.copy(deep=True)
    result, written = merge_obs_metadata(obs, metadata, overwrite_existing=True)
    expected = obs["condition"].copy()
    expected.loc[ids] = labels
    assert written == ["condition"]
    assert list(result["condition"]) == list(expected)
    assert result.loc["1", "condition"] == "original-1"
    pd.testing.assert_frame_equal(obs, before)


@pytest.mark.parametrize("identifier", ["001", "1.0", " 1 "])
def test_nonexact_identity_remains_unmatched(tmp_path, identifier):
    obs = pd.DataFrame({"condition": ["original"]}, index=["1"])
    path = _table_path(
        tmp_path, pd.DataFrame({"id": [identifier], "condition": ["wrong"]}), "csv"
    )
    metadata, _ = read_sample_metadata(path, obs.index, "id")
    result, written = merge_obs_metadata(obs, metadata, overwrite_existing=True)
    pd.testing.assert_frame_equal(result, obs)
    assert written == []


@pytest.mark.parametrize("bool_dtype", ["bool", "category"])
def test_boolean_and_numeric_annotations(tmp_path, bool_dtype):
    names = pd.Index(["s1", "s2", "s3"])
    obs = pd.DataFrame(
        {
            "use_sample": pd.Series(
                [False, True, False], index=names, dtype=bool_dtype
            ),
            "nullable": pd.array([False, None, False], dtype="boolean"),
            "age": [31, 32, 33],
        },
        index=names,
    )
    path = _table_path(
        tmp_path,
        pd.DataFrame(
            {
                "id": ["s1"],
                "use_sample": [True],
                "nullable": [True],
                "new_flag": [False],
                "age": [64],
                "score": [1.5],
            }
        ),
        "csv",
    )
    metadata, _ = read_sample_metadata(path, names)
    result, _ = merge_obs_metadata(obs, metadata, overwrite_existing=True)
    assert pd.api.types.is_bool_dtype(result["use_sample"])
    assert list(result["use_sample"]) == [True, True, False]
    assert str(result["nullable"].dtype) == "boolean"
    assert pd.isna(result.loc["s2", "nullable"])
    assert str(result["new_flag"].dtype) == "boolean"
    assert list(result["new_flag"].isna()) == [False, True, True]
    assert pd.api.types.is_integer_dtype(result["age"])
    assert list(result["age"]) == [64, 32, 33]
    assert pd.api.types.is_float_dtype(result["score"])
    assert list(result["score"].isna()) == [False, True, True]


@pytest.mark.parametrize(
    "table, column, message",
    [
        ({"id": [""], "condition": ["changed"]}, "id", "nonempty"),
        (
            {"id_a": ["s1"], "id_b": ["s2"], "condition": ["changed"]},
            "auto",
            "Ambiguous",
        ),
        ({"id": ["s1", "s1"], "condition": ["A", "B"]}, "id", "unique"),
    ],
)
def test_invalid_identity_is_rejected(tmp_path, table, column, message):
    names = pd.Index(["s1", "s2"])
    path = _table_path(tmp_path, pd.DataFrame(table), "csv")
    with pytest.raises(ValueError, match=message):
        read_sample_metadata(path, names, column)


def test_duplicate_obs_ids_are_rejected(tmp_path):
    path = _table_path(tmp_path, pd.DataFrame({"id": ["s1"], "v": [1]}), "csv")
    with pytest.raises(ValueError, match="unique"):
        read_sample_metadata(path, pd.Index(["s1", "s1"]), "id")
    with pytest.raises(ValueError, match="unique"):
        merge_obs_metadata(
            pd.DataFrame(index=["s1", "s1"]), pd.DataFrame({"v": [1]}, index=["s1"])
        )


def test_existing_columns_and_missing_values_can_be_preserved():
    obs = pd.DataFrame({"age": [31, 32, 33]}, index=["s1", "s2", "s3"])
    metadata = pd.DataFrame({"age": [64, np.nan]}, index=["s1", "s2"])
    unchanged, written = merge_obs_metadata(obs, metadata)
    pd.testing.assert_frame_equal(unchanged, obs)
    assert written == []
    result, written = merge_obs_metadata(
        obs, metadata, overwrite_existing=True, preserve_missing=True
    )
    assert list(result["age"]) == [64, 32, 33]
    assert written == ["age"]
