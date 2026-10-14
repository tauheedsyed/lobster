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


@pytest.mark.parametrize(
    "existing, update", [([2**60 + 1, 2, 3], 0.5), ([1.0, 2.0, 3.0], 2**60 + 1)]
)
def test_numeric_widening_cannot_change_integer_values(existing, update):
    obs = pd.DataFrame({"large_value": existing}, index=["s1", "s2", "s3"])
    metadata = pd.DataFrame({"large_value": [update]}, index=["s2"])
    before = obs.copy(deep=True)
    result, written = merge_obs_metadata(obs, metadata, overwrite_existing=True)
    expected = [str(value) for value in existing]
    expected[1] = str(update)
    assert list(result["large_value"]) == expected
    assert written == ["large_value"]
    assert any(
        "stored exact values as text" in warning
        for warning in result.attrs["metadata_writeback_warnings"]
    )
    pd.testing.assert_frame_equal(obs, before)


def test_numeric_category_expansion_preserves_unmatched_values(tmp_path):
    big = 2**60 + 1
    obs = pd.DataFrame({"v": pd.Categorical([big, 2, 3])}, index=["s1", "s2", "s3"])
    path = _table_path(tmp_path, pd.DataFrame({"id": ["s2"], "v": ["0.5"]}), "csv")
    metadata, _ = read_sample_metadata(path, obs.index)
    result, _ = merge_obs_metadata(obs, metadata, overwrite_existing=True)
    assert list(result["v"]) == [str(big), "0.5", "3"]
    assert int(result.loc["s1", "v"]) == big
    assert any(
        "stored exact values as text" in warning
        for warning in result.attrs["metadata_writeback_warnings"]
    )


@pytest.mark.parametrize("existing", ["integer", "float", "new"])
@pytest.mark.parametrize("second", ["", "0.5"])
@pytest.mark.parametrize("suffix", ["csv", "tsv", "xlsx"])
def test_raw_numeric_precision_survives_parse_and_alignment(
    tmp_path, existing, second, suffix
):
    names = pd.Index(["s1", "s2", "s3"])
    obs = pd.DataFrame(index=names)
    if existing == "integer":
        obs["v"] = [1, 2, 3]
    elif existing == "float":
        obs["v"] = [1.0, 2.0, 3.0]
    big = str(2**60 + 1)
    path = _table_path(
        tmp_path, pd.DataFrame({"id": ["s1", "s2"], "v": [big, second]}), suffix
    )
    metadata, _ = read_sample_metadata(path, names, "id")
    result, _ = merge_obs_metadata(obs, metadata, overwrite_existing=True)
    assert result.loc["s1", "v"] == big
    assert any(
        "retained original numeric tokens as text" in warning
        for warning in result.attrs["metadata_writeback_warnings"]
    )
    if second:
        assert result.loc["s2", "v"] == second
    else:
        assert pd.isna(result.loc["s2", "v"])
    if existing == "new":
        assert pd.isna(result.loc["s3", "v"])
    else:
        assert result.loc["s3", "v"] == ("3" if existing == "integer" else "3.0")


@pytest.mark.parametrize(
    "tokens",
    [
        ["0.1", "0.2", "0.3", "0.1", "0.2", "0.3"],
        ["0.5", "12.75", "0.125", "0.5", "12.75", "0.125"],
        ["1e-1", "2e-1", "3e-1", "1e-1", "2e-1", "3e-1"],
        ["1", "2", "3", "1", "2", "3"],
        ["0.30000000000000004", "1e-3", "0.10", "0.30000000000000004", "1e-3", "0.10"],
        ["-0", "0.1", "0.2", "-0", "0.1", "0.2"],
    ],
)
@pytest.mark.parametrize("partial_overwrite", [False, True])
@pytest.mark.parametrize("suffix", ["csv", "tsv", "xlsx"])
def test_continuous_covariates_keep_numeric_design(
    tmp_path, tokens, partial_overwrite, suffix
):
    from lobster.services.analysis.differential_formula_service import (
        DifferentialFormulaService,
    )

    names = pd.Index([f"s{index}" for index in range(1, 7)])
    integers = all(token.isdigit() for token in tokens)
    expected = np.array([int(token) if integers else float(token) for token in tokens])
    obs = pd.DataFrame(index=names)
    if partial_overwrite:
        obs["dose"] = expected
    path = _table_path(
        tmp_path,
        pd.DataFrame(
            {
                "id": names[:1] if partial_overwrite else names,
                "dose": tokens[:1] if partial_overwrite else tokens,
            }
        ),
        suffix,
    )
    metadata, _ = read_sample_metadata(path, names)
    result, _ = merge_obs_metadata(obs, metadata, overwrite_existing=partial_overwrite)
    assert not result.attrs["metadata_writeback_warnings"]
    assert pd.api.types.is_numeric_dtype(result["dose"])
    assert (
        pd.api.types.is_integer_dtype(result["dose"])
        if integers
        else pd.api.types.is_float_dtype(result["dose"])
    )
    np.testing.assert_array_equal(result["dose"].to_numpy(), expected)
    service = DifferentialFormulaService()
    reference = pd.DataFrame({"dose": expected}, index=names)
    reference_design = service.construct_design_matrix(
        service.parse_formula("~dose", reference), reference
    )
    components = service.parse_formula("~dose", result)
    assert components["variable_info"]["dose"]["type"] == "continuous"
    design = service.construct_design_matrix(components, result)
    assert design["design_matrix"].shape == (6, 2)
    assert design["coefficient_names"] == reference_design["coefficient_names"]
    np.testing.assert_array_equal(
        design["design_matrix"], reference_design["design_matrix"]
    )


def test_exact_numeric_text_can_update_string_categories(tmp_path):
    obs = pd.DataFrame(
        {"v": pd.Categorical(["001", "002", "003"])}, index=["s1", "s2", "s3"]
    )
    path = _table_path(tmp_path, pd.DataFrame({"id": ["s1"], "v": ["004"]}), "csv")
    metadata, _ = read_sample_metadata(path, obs.index)
    result, _ = merge_obs_metadata(obs, metadata, overwrite_existing=True)
    assert list(result["v"]) == ["004", "002", "003"]


def test_shortest_float_repr_property(tmp_path):
    rng = np.random.default_rng(2026)
    bits = rng.integers(0, np.iinfo(np.uint64).max, size=512, dtype=np.uint64)
    generated = bits.view(np.float64)
    generated = generated[np.isfinite(generated)]
    tokens = [repr(float(value)) for value in generated]
    tokens += ["0.30000000000000004", "1e-3", "0.10", "-0", "+0.5"]
    expected = np.array([float(token) for token in tokens])
    names = pd.Index([f"s{index}" for index in range(len(tokens))])
    path = _table_path(tmp_path, pd.DataFrame({"id": names, "v": tokens}), "csv")
    metadata, _ = read_sample_metadata(path, names)
    result, _ = merge_obs_metadata(pd.DataFrame(index=names), metadata)
    assert not result.attrs["metadata_writeback_warnings"]
    assert pd.api.types.is_float_dtype(result["v"])
    actual = result["v"].to_numpy()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


def test_annotation_nan_is_missing_and_infinity_is_reported_text(tmp_path):
    names = pd.Index(["s1", "s2", "s3"])
    path = _table_path(
        tmp_path, pd.DataFrame({"id": names, "v": ["NaN", "inf", "-inf"]}), "csv"
    )
    metadata, _ = read_sample_metadata(path, names)
    result, _ = merge_obs_metadata(pd.DataFrame(index=names), metadata)
    assert pd.isna(result.loc["s1", "v"])
    assert result.loc["s2", "v"] == "inf"
    assert result.loc["s3", "v"] == "-inf"
    assert any(
        "retained original numeric tokens as text" in warning
        for warning in result.attrs["metadata_writeback_warnings"]
    )


def test_matched_missing_boolean_and_numeric_values(tmp_path):
    obs = pd.DataFrame(
        {"use_sample": [True, True, False], "age": [31, 32, 33]},
        index=["s1", "s2", "s3"],
    )
    path = _table_path(
        tmp_path,
        pd.DataFrame({"id": ["s1"], "use_sample": [None], "age": [None]}),
        "csv",
    )
    metadata, _ = read_sample_metadata(path, obs.index)
    result, _ = merge_obs_metadata(obs, metadata, overwrite_existing=True)
    assert str(result["use_sample"].dtype) == "boolean"
    assert pd.isna(result.loc["s1", "use_sample"])
    assert not result.loc["s3", "use_sample"]
    assert pd.isna(result.loc["s1", "age"])
    assert list(result["age"].iloc[1:]) == [32, 33]


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
