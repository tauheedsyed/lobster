"""GEO sample orientation requires an exact full-set identity, never position."""

import importlib

import numpy as np
import pandas as pd
import pytest

# Deliberately synthetic identifiers; no live accession or download is used.
A = "GSM900000000001"
B = "GSM900000000002"
EXTRA = "GSM900000000003"
SAMPLES = {A: {"title": "Synthetic sample A"}, B: {"title": "Synthetic sample B"}}
FEATURES = ["synthetic_gene_a", "synthetic_gene_b", "synthetic_gene_c"]


def _api():
    module = importlib.import_module(
        "lobster.services.data_access.geo.import_requirements"
    )
    return module.orient_by_sample_identity, module.SampleIdentityError


def _matrix(labels, samples_in_rows=False, dtype=np.int64):
    frame = pd.DataFrame(
        np.array([[1, 3], [2, 4], [5, 6]], dtype=dtype), index=FEATURES, columns=labels
    )
    return frame.T.copy() if samples_in_rows else frame


@pytest.mark.parametrize("samples_in_rows", [False, True])
@pytest.mark.parametrize("title_labels", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_exact_full_set_in_either_axis_retains_order_values_and_ids(
    samples_in_rows, title_labels, reverse
):
    orient, _ = _api()
    ids = list(SAMPLES)
    labels = [SAMPLES[g]["title"] for g in ids] if title_labels else ids.copy()
    if reverse:
        ids.reverse()
        labels.reverse()
    frame = _matrix(labels, samples_in_rows)
    frame.attrs["synthetic_parser_note"] = "kept"
    original = frame.copy(deep=True)
    result, rule = orient(frame, SAMPLES)
    expected = frame.copy() if samples_in_rows else frame.T.copy()
    pd.testing.assert_frame_equal(result, expected)
    assert result is not frame and result.shape == (2, 3)
    assert list(result.index) == labels
    assert rule == ("title_exact" if title_labels else "gsm_id")
    assert result.attrs["source_sample_id"] == dict(zip(labels, ids))
    assert all(
        isinstance(gsm, str) for gsm in result.attrs["source_sample_id"].values()
    )
    assert result.attrs["synthetic_parser_note"] == "kept"
    pd.testing.assert_frame_equal(frame, original)
    assert frame.attrs == original.attrs and "source_sample_id" not in frame.attrs


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
def test_orientation_does_not_explicitly_cast_numeric_values(dtype):
    orient, _ = _api()
    frame = _matrix(list(SAMPLES), dtype=dtype)
    frame.iloc[0, 0] = dtype(2**53 + 1) if dtype != np.float64 else dtype(11)
    result, _ = orient(frame, SAMPLES)
    pd.testing.assert_frame_equal(result.T, frame)
    assert result.iloc[0, 0] == frame.iloc[0, 0]
    assert all(column == np.dtype(dtype) for column in result.dtypes)


@pytest.mark.parametrize(
    "labels",
    [
        ["Synthetic sample a", "Synthetic sample B"],
        ["Synthetic sample A ", "Synthetic sample B"],
        ["synthetic_unknown_a", "synthetic_unknown_b"],
        [A, "Synthetic sample B"],
        [A, A],
        ["Synthetic sample A", "Synthetic sample A"],
        [1, 2],
        [None, B],
        [np.nan, B],
        ["", B],
        ["   ", B],
    ],
)
@pytest.mark.parametrize("samples_in_rows", [False, True])
def test_no_case_whitespace_coercion_mixed_or_positional_identity(
    labels, samples_in_rows
):
    orient, error = _api()
    frame = _matrix(labels, samples_in_rows)
    with pytest.raises(error):
        orient(frame, SAMPLES)
    assert "source_sample_id" not in frame.attrs


@pytest.mark.parametrize(
    "samples",
    [
        None,
        {},
        [],
        {1: {}},
        {None: {}},
        {"": {}},
        {"   ": {}},
        {"synthetic_not_gsm": {}},
    ],
)
def test_missing_or_invalid_metadata_sample_ids_refuse(samples):
    orient, error = _api()
    with pytest.raises(error):
        orient(_matrix(list(SAMPLES)), samples)


@pytest.mark.parametrize(
    "samples",
    [
        {A: {}, B: {}},
        {A: {"title": None}, B: {"title": "Synthetic sample B"}},
        {A: {"title": ["Synthetic sample A"]}, B: {"title": "Synthetic sample B"}},
        {A: {"title": "synthetic_duplicate"}, B: {"title": "synthetic_duplicate"}},
        {A: None, B: None},
    ],
)
def test_unusable_titles_refuse_title_rule_but_do_not_override_explicit_gsm_ids(
    samples,
):
    orient, error = _api()
    with pytest.raises(error):
        orient(_matrix(["Synthetic sample A", "Synthetic sample B"]), samples)
    result, rule = orient(_matrix(list(SAMPLES)), samples)
    assert rule == "gsm_id"
    assert result.attrs["source_sample_id"] == {g: g for g in SAMPLES}


def test_subset_is_not_full_series_identity():
    orient, error = _api()
    frame = _matrix(list(SAMPLES)).iloc[:, :1]
    with pytest.raises(error):
        orient(frame, SAMPLES)


def test_extra_samples_do_not_match_full_series_identity():
    orient, error = _api()
    frame = _matrix(list(SAMPLES))
    frame[EXTRA] = 1
    with pytest.raises(error):
        orient(frame, SAMPLES)


def test_both_axes_matching_is_ambiguous_not_a_transpose_guess():
    orient, error = _api()
    frame = pd.DataFrame([[1, 2], [3, 4]], index=list(SAMPLES), columns=list(SAMPLES))
    with pytest.raises(error, match="ambiguous"):
        orient(frame, SAMPLES)


@pytest.mark.parametrize("samples_in_rows", [False, True])
def test_empty_feature_axis_is_not_a_matrix(samples_in_rows):
    orient, error = _api()
    frame = _matrix(list(SAMPLES), samples_in_rows)
    frame = frame.iloc[:, :0] if samples_in_rows else frame.iloc[:0, :]
    with pytest.raises(error):
        orient(frame, SAMPLES)


def test_actual_metadata_whitespace_is_not_trimmed_for_a_guess():
    orient, error = _api()
    samples = {g: {"title": m["title"] + " "} for g, m in SAMPLES.items()}
    with pytest.raises(error):
        orient(_matrix([m["title"] for m in SAMPLES.values()]), samples)
    result, rule = orient(_matrix([m["title"] for m in samples.values()]), samples)
    assert rule == "title_exact"
    assert list(result.attrs["source_sample_id"].values()) == list(SAMPLES)


@pytest.mark.parametrize(
    "labels", [[A, "Synthetic sample B"], [EXTRA, "Synthetic sample B"]]
)
def test_title_cannot_reassign_or_mix_an_accession_column(labels):
    orient, error = _api()
    samples = {A: {"title": labels[0]}, B: {"title": labels[1]}}
    with pytest.raises(error):
        orient(_matrix(labels), samples)


def test_exact_gsm_ids_have_authority_over_conflicting_titles():
    orient, _ = _api()
    samples = {A: {"title": B}, B: {"title": A}}
    result, rule = orient(_matrix([A, B]), samples)
    assert rule == "gsm_id"
    assert result.attrs["source_sample_id"] == {A: A, B: B}


def test_safe_mixed_dtypes_preserve_values_when_oriented():
    orient, _ = _api()
    frame = pd.DataFrame(
        {
            A: np.array([1, 2, 3], dtype=np.int64),
            B: np.array([3, 4, 6], dtype=np.float64),
        },
        index=FEATURES,
    )
    result, _ = orient(frame, SAMPLES)
    pd.testing.assert_frame_equal(result.T.astype(object), frame.astype(object))


def test_orientation_precision_loss_is_refused_not_silently_accepted():
    orient, error = _api()
    frame = pd.DataFrame(
        {
            A: np.array([2**53 + 1, 2, 3], dtype=np.int64),
            B: np.array([3, 4, 6], dtype=np.float64),
        },
        index=FEATURES,
    )
    with pytest.raises(error, match="values"):
        orient(frame, SAMPLES)
