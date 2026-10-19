"""Tests for the notebook replay contract and comparator.

The comparator decides whether a replayed notebook reproduced its reference.
A false MATCH is the failure that matters, so most tests change exactly one
thing and check that the change is detected, plus controls that show the
comparator still matches when nothing relevant changed.
"""

import json
import os
import subprocess  # nosec B404 - runs a fixed interpreter check only
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from lobster.testing.replay_compare import compare, output_files
from lobster.testing.replay_contract import (
    Claim,
    ComparisonPolicy,
    ContractError,
    Exclusion,
    Execution,
    Preflight,
    Refusal,
    RefusalOrigin,
    ReplayReport,
    SlotResult,
    SlotStatus,
    Tolerance,
    Verdict,
    VerificationStatus,
    aggregate_verdict,
    binding_differences,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SLOTS = ("X", "layers", "obs", "var", "obsm", "obsp", "varm", "varp", "uns", "raw")


def reference_adata():
    """A small AnnData with every slot populated."""
    rng = np.random.default_rng(7)
    n_obs, n_vars = 6, 4
    obs = pd.DataFrame(
        {
            "n_genes": np.arange(n_obs, dtype=np.int64) + 10,
            "pct_mito": rng.random(n_obs),
            "leiden": pd.Categorical(["0", "1", "0", "2", "1", "0"]),
            "keep": np.array([True, False, True, True, False, True]),
        },
        index=[f"cell{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"highly_variable": [True, False, True, False], "mean": rng.random(n_vars)},
        index=[f"gene{i}" for i in range(n_vars)],
    )
    adata = anndata.AnnData(X=rng.random((n_obs, n_vars)), obs=obs, var=var)
    adata.layers["counts"] = sp.csr_matrix(
        rng.integers(0, 5, size=(n_obs, n_vars)).astype(np.int64)
    )
    adata.obsm["X_pca"] = rng.random((n_obs, 2))
    adata.obsp["connectivities"] = sp.csr_matrix(
        np.triu(rng.random((n_obs, n_obs)) > 0.6).astype(np.float64)
    )
    adata.varm["PCs"] = rng.random((n_vars, 2))
    adata.varp["corr"] = sp.csr_matrix(np.eye(n_vars))
    adata.uns["params"] = {
        "target_sum": 1e4,
        "method": "seurat",
        "n_top": 2,
        "flavors": ["a", "b"],
        "nested": {"seed": 0, "weights": np.array([0.5, 0.25])},
    }
    adata.uns["log1p"] = {"base": None}
    raw = adata.copy()
    raw.varm["raw_loadings"] = rng.random((n_vars, 3))
    adata.raw = raw
    return adata


def all_slots_policy(**kwargs):
    return ComparisonPolicy(required=tuple(f"ad/{slot}" for slot in SLOTS), **kwargs)


def run(expected_adata, actual_adata, policy=None):
    policy = policy or all_slots_policy()
    results = compare({"ad": expected_adata}, {"ad": actual_adata}, policy)
    return {r.path: r for r in results}, aggregate_verdict(tuple(results))


# --------------------------------------------------------------------------
# Comparator: controls
# --------------------------------------------------------------------------


def test_identical_copy_matches_every_slot_with_evidence():
    reference = reference_adata()
    slots, verdict = run(reference, reference.copy())
    assert verdict is Verdict.MATCH
    for path, result in slots.items():
        assert result.status is SlotStatus.MATCH, (path, result.diffs)
        assert result.substantive, path


# --------------------------------------------------------------------------
# Comparator: one change at a time
# --------------------------------------------------------------------------


def _set_column(axis, column, change):
    """Replace one obs/var column with ``change(old_column)``."""

    def mutate(adata):
        frame = getattr(adata, axis)
        frame[column] = change(frame[column]) if column in frame else change(None)

    return mutate


def _set_obs(column, values):
    return _set_column("obs", column, lambda _old: values)


def _first_value(old, value):
    values = old.to_numpy().copy()
    values[0] = value
    return values


def _uns_nested(key, value):
    def mutate(adata):
        adata.uns["params"]["nested"][key] = value

    return mutate


def _raw_with(change):
    def mutate(adata):
        raw = adata.raw.to_adata()
        change(raw)
        adata.raw = raw

    return mutate


def _bump(array, index=(0, 0), by=1e-3):
    array = array.copy()
    array[index] += by
    return array


MUTATIONS = {
    "X value": ("X", lambda a: setattr(a, "X", _bump(a.X))),
    "X dtype": ("X", lambda a: setattr(a, "X", a.X.astype(np.float32))),
    "X NaN": ("X", lambda a: a.X.__setitem__((1, 1), np.nan)),
    "X infinity": ("X", lambda a: a.X.__setitem__((0, 0), -np.inf)),
    "X dense to sparse": ("X", lambda a: setattr(a, "X", sp.csr_matrix(a.X))),
    "layer value": (
        "layers",
        lambda a: a.layers.__setitem__(
            "counts",
            sp.csr_matrix(a.layers["counts"].toarray() + np.eye(6, 4, dtype=np.int64)),
        ),
    ),
    "layer missing": ("layers", lambda a: a.layers.pop("counts")),
    "layer extra": ("layers", lambda a: a.layers.__setitem__("extra", a.X.copy())),
    "obs float": (
        "obs",
        _set_column("obs", "pct_mito", lambda c: _bump(c.to_numpy(), (0,))),
    ),
    "obs int": ("obs", _set_column("obs", "n_genes", lambda c: _first_value(c, 99))),
    "obs int dtype": ("obs", _set_obs("n_genes", np.arange(6, dtype=np.int32) + 10)),
    "obs bool": (
        "obs",
        _set_obs("keep", np.array([False, False, True, True, False, True])),
    ),
    "obs label": (
        "obs",
        _set_obs("leiden", pd.Categorical(["0", "1", "0", "2", "1", "1"])),
    ),
    "obs categories order": (
        "obs",
        _set_column(
            "obs", "leiden", lambda c: c.cat.reorder_categories(["2", "1", "0"])
        ),
    ),
    "obs categories ordered": (
        "obs",
        _set_column("obs", "leiden", lambda c: c.cat.as_ordered()),
    ),
    "obs category unused added": (
        "obs",
        _set_column("obs", "leiden", lambda c: c.cat.add_categories("9")),
    ),
    "obs column missing": ("obs", lambda a: a.obs.drop(columns="keep", inplace=True)),
    "obs column extra": ("obs", _set_obs("extra", 1.0)),
    "obs column order": (
        "obs",
        lambda a: setattr(a, "obs", a.obs[["pct_mito", "n_genes", "leiden", "keep"]]),
    ),
    "obs name": (
        "obs",
        lambda a: setattr(a, "obs_names", ["cellX"] + list(a.obs_names[1:])),
    ),
    "var name": (
        "var",
        lambda a: setattr(a, "var_names", ["geneX"] + list(a.var_names[1:])),
    ),
    "var column": (
        "var",
        _set_column("var", "mean", lambda c: _bump(c.to_numpy(), (0,))),
    ),
    "obsm value": (
        "obsm",
        lambda a: a.obsm.__setitem__("X_pca", _bump(a.obsm["X_pca"])),
    ),
    "obsm missing": ("obsm", lambda a: a.obsm.pop("X_pca")),
    "obsp structure": (
        "obsp",
        lambda a: a.obsp.__setitem__(
            "connectivities", a.obsp["connectivities"] + sp.csr_matrix(np.eye(6))
        ),
    ),
    "obsp sparse to dense": (
        "obsp",
        lambda a: a.obsp.__setitem__(
            "connectivities", a.obsp["connectivities"].toarray()
        ),
    ),
    "varm value": ("varm", lambda a: a.varm.__setitem__("PCs", _bump(a.varm["PCs"]))),
    "varp value": (
        "varp",
        lambda a: a.varp.__setitem__("corr", sp.csr_matrix(np.eye(4) * 2)),
    ),
    "uns nested float": ("uns", _uns_nested("weights", np.array([0.5, 0.26]))),
    "uns nested int": ("uns", _uns_nested("seed", 1)),
    "uns int to float": ("uns", _uns_nested("seed", 0.0)),
    "uns int to bool": ("uns", _uns_nested("seed", False)),
    "uns nested missing": ("uns", lambda a: a.uns["params"]["nested"].pop("seed")),
    "uns nested extra": ("uns", _uns_nested("added", 1)),
    "uns array dtype": (
        "uns",
        _uns_nested("weights", np.array([0.5, 0.25], np.float32)),
    ),
    "uns list to array": (
        "uns",
        lambda a: a.uns["params"].__setitem__("flavors", np.array(["a", "b"])),
    ),
    "uns list order": (
        "uns",
        lambda a: a.uns["params"].__setitem__("flavors", ["b", "a"]),
    ),
    "uns string": (
        "uns",
        lambda a: a.uns["params"].__setitem__("method", "cell_ranger"),
    ),
    "uns float": ("uns", lambda a: a.uns["params"].__setitem__("target_sum", 1e6)),
    "uns None to value": ("uns", lambda a: a.uns["log1p"].__setitem__("base", 2)),
    "uns key missing": ("uns", lambda a: a.uns.pop("log1p")),
    "raw X": ("raw", _raw_with(lambda r: setattr(r, "X", _bump(r.X)))),
    "raw var column": (
        "raw",
        _raw_with(_set_column("var", "mean", lambda c: _bump(c.to_numpy(), (1,)))),
    ),
    "raw varm": (
        "raw",
        _raw_with(
            lambda r: r.varm.__setitem__("raw_loadings", _bump(r.varm["raw_loadings"]))
        ),
    ),
    "raw removed": ("raw", lambda a: delattr(a, "raw")),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_single_change_is_a_mismatch(name):
    slot, mutate = MUTATIONS[name]
    reference = reference_adata()
    actual = reference.copy()
    mutate(actual)
    slots, verdict = run(reference, actual)
    assert slots[f"ad/{slot}"].status is SlotStatus.MISMATCH, slots[f"ad/{slot}"]
    assert slots[f"ad/{slot}"].diffs
    assert verdict is Verdict.MISMATCH


# Renaming an axis changes every slot aligned to it, so it has no "unrelated" slots.
AXIS_RENAMES = {"obs name", "var name"}


@pytest.mark.parametrize("name", sorted(set(MUTATIONS) - AXIS_RENAMES))
def test_single_change_leaves_unrelated_slots_matching(name):
    slot, mutate = MUTATIONS[name]
    reference = reference_adata()
    actual = reference.copy()
    mutate(actual)
    slots, _ = run(reference, actual)
    others = {p: r.status for p, r in slots.items() if p != f"ad/{slot}"}
    assert set(others.values()) == {SlotStatus.MATCH}, others


def test_permuted_cells_are_a_mismatch_not_sorted_away():
    reference = reference_adata()
    actual = reference[list(reversed(reference.obs_names))].copy()
    slots, verdict = run(reference, actual)
    assert verdict is Verdict.MISMATCH
    assert slots["ad/obs"].status is SlotStatus.MISMATCH
    assert slots["ad/X"].status is SlotStatus.MISMATCH


# --------------------------------------------------------------------------
# Comparator: tolerance controls
# --------------------------------------------------------------------------


def _x_only(**tolerance):
    rules = (Tolerance("ad/X", **tolerance),) if tolerance else ()
    return ComparisonPolicy(required=("ad/X",), tolerances=rules)


def _with_x(adata, x):
    adata = adata.copy()
    adata.X = x
    return adata


def test_tolerance_below_matches_and_above_mismatches():
    reference = reference_adata()
    x = reference.X
    policy = _x_only(rtol=1e-6)
    below = _with_x(reference, x * (1 + 1e-7))
    above = _with_x(reference, x * (1 + 1e-5))
    assert run(reference, below, policy)[1] is Verdict.MATCH
    assert run(reference, above, policy)[1] is Verdict.MISMATCH


def test_default_is_exact():
    reference = reference_adata()
    nudged = _with_x(reference, np.nextafter(reference.X, 2.0))
    assert run(reference, nudged, _x_only())[1] is Verdict.MISMATCH


def test_relative_tolerance_does_not_cover_values_near_zero():
    reference = reference_adata()
    zero = reference.X.copy()
    zero[0, 0] = 0.0
    reference = _with_x(reference, zero)
    near_zero = zero.copy()
    near_zero[0, 0] = 1e-12
    actual = _with_x(reference, near_zero)
    assert run(reference, actual, _x_only(rtol=1e-6))[1] is Verdict.MISMATCH
    assert run(reference, actual, _x_only(rtol=1e-6, atol=1e-9))[1] is Verdict.MATCH


def test_most_specific_tolerance_applies():
    reference = reference_adata()
    actual = reference.copy()
    actual.obsm["X_pca"] = actual.obsm["X_pca"] * (1 + 1e-5)
    loose = ComparisonPolicy(
        required=("ad/obsm",),
        tolerances=(
            Tolerance("ad/obsm", rtol=1e-3),
            Tolerance("ad/obsm/X_pca", rtol=1e-9),
        ),
    )
    assert run(reference, actual, loose)[1] is Verdict.MISMATCH


def test_same_nan_positions_match_and_different_positions_mismatch():
    reference = reference_adata()
    x = reference.X.copy()
    x[2, 3] = np.nan
    reference = _with_x(reference, x)
    assert run(reference, reference.copy(), _x_only())[1] is Verdict.MATCH
    moved = x.copy()
    moved[2, 3], moved[2, 2] = 0.5, np.nan
    assert run(reference, _with_x(reference, moved), _x_only(atol=1.0))[1] is (
        Verdict.MISMATCH
    )


def test_infinity_must_be_identical_even_with_a_tolerance():
    reference = reference_adata()
    x = reference.X.copy()
    x[0, 0] = np.inf
    reference = _with_x(reference, x)
    finite = x.copy()
    finite[0, 0] = 1e308
    assert run(reference, _with_x(reference, finite), _x_only(rtol=1.0))[1] is (
        Verdict.MISMATCH
    )


# --------------------------------------------------------------------------
# Comparator: scope, missing values, unsupported types
# --------------------------------------------------------------------------


def test_exclusion_skips_only_the_excluded_path():
    reference = reference_adata()
    reference.uns["params"]["timestamp"] = "2026-01-01"
    actual = reference.copy()
    actual.uns["params"]["timestamp"] = "2026-01-02"
    plain = ComparisonPolicy(required=("ad/uns",))
    excluded = ComparisonPolicy(
        required=("ad/uns",),
        excluded=(Exclusion("ad/uns/params/timestamp", "wall-clock time"),),
    )
    assert run(reference, actual, plain)[1] is Verdict.MISMATCH
    assert run(reference, actual, excluded)[1] is Verdict.MATCH
    actual.uns["params"]["method"] = "other"
    assert run(reference, actual, excluded)[1] is Verdict.MISMATCH


def test_excluded_obs_column_is_ignored_on_both_sides():
    reference = reference_adata()
    actual = reference.copy()
    actual.obs.drop(columns="pct_mito", inplace=True)
    policy = ComparisonPolicy(
        required=("ad/obs",), excluded=(Exclusion("ad/obs/pct_mito", "random"),)
    )
    assert run(reference, actual, policy)[1] is Verdict.MATCH


def test_missing_reference_is_not_compared_and_missing_output_is_mismatch():
    reference = reference_adata()
    policy = ComparisonPolicy(required=("ad/obsm/X_umap",))
    slots, verdict = run(reference, reference.copy(), policy)
    assert slots["ad/obsm/X_umap"].status is SlotStatus.NOT_COMPARED
    assert verdict is Verdict.NOT_COMPARED

    reference.obsm["X_umap"] = np.zeros((6, 2))
    actual = reference.copy()
    del actual.obsm["X_umap"]
    slots, verdict = run(reference, actual, policy)
    assert slots["ad/obsm/X_umap"].status is SlotStatus.MISMATCH
    assert verdict is Verdict.MISMATCH


def test_missing_output_object_is_a_mismatch():
    results = compare({"ad": reference_adata()}, {}, all_slots_policy())
    assert {r.status for r in results} == {SlotStatus.MISMATCH}


def test_empty_slot_matches_but_is_not_evidence():
    reference = anndata.AnnData(X=np.zeros((2, 2)))
    policy = ComparisonPolicy(required=("ad/uns",))
    slots, verdict = run(reference, reference.copy(), policy)
    assert slots["ad/uns"].status is SlotStatus.MATCH
    assert not slots["ad/uns"].substantive
    assert verdict is Verdict.NOT_COMPARED


class _Opaque:
    def __str__(self):
        return "same text"

    __repr__ = __str__


def test_unsupported_type_is_not_compared_never_stringified():
    reference = reference_adata()
    reference.uns["opaque"] = _Opaque()
    actual = reference.copy()
    actual.uns["opaque"] = _Opaque()
    policy = ComparisonPolicy(required=("ad/uns",))
    slots, verdict = run(reference, actual, policy)
    assert slots["ad/uns"].status is SlotStatus.NOT_COMPARED
    assert "unsupported type" in slots["ad/uns"].not_compared[0]
    assert verdict is Verdict.NOT_COMPARED


def test_a_mismatch_outranks_a_not_compared_value():
    reference = reference_adata()
    reference.uns["opaque"] = _Opaque()
    actual = reference.copy()
    actual.uns["params"]["method"] = "other"
    slots, _ = run(reference, actual, ComparisonPolicy(required=("ad/uns",)))
    assert slots["ad/uns"].status is SlotStatus.MISMATCH


def test_nullable_and_missing_values_compare_by_position():
    frame = pd.DataFrame({"n": pd.array([1, None, 3], dtype="Int64")})
    same = frame.copy()
    moved = pd.DataFrame({"n": pd.array([None, 2, 3], dtype="Int64")})
    policy = ComparisonPolicy(required=("t",))
    assert (
        aggregate_verdict(compare({"t": frame}, {"t": same}, policy)) is Verdict.MATCH
    )
    assert aggregate_verdict(compare({"t": frame}, {"t": moved}, policy)) is (
        Verdict.MISMATCH
    )


# --------------------------------------------------------------------------
# Comparator: sparse policy
# --------------------------------------------------------------------------


def _sparse_case(policy_name, expected, actual):
    policy = ComparisonPolicy(required=("m",), sparse=policy_name)
    return aggregate_verdict(compare({"m": expected}, {"m": actual}, policy))


def test_semantic_sparse_ignores_format_and_explicit_zeros():
    dense = np.array([[0.0, 1.0], [2.0, 0.0]])
    csr = sp.csr_matrix(dense)
    csc = sp.csc_matrix(dense)
    explicit = sp.csr_matrix(
        (np.array([1.0, 0.0, 2.0]), np.array([1, 0, 0]), np.array([0, 2, 3])),
        shape=(2, 2),
    )
    assert _sparse_case("semantic", csr, csc) is Verdict.MATCH
    assert _sparse_case("semantic", csr, explicit) is Verdict.MATCH
    assert _sparse_case("representation", csr, csc) is Verdict.MISMATCH
    assert _sparse_case("representation", csr, explicit) is Verdict.MISMATCH
    changed = sp.csr_matrix(np.array([[0.0, 1.0], [2.5, 0.0]]))
    assert _sparse_case("semantic", csr, changed) is Verdict.MISMATCH


# --------------------------------------------------------------------------
# Comparator: output files
# --------------------------------------------------------------------------


def _write_outputs(directory, table=None, extra=None):
    directory.mkdir()
    table = table if table is not None else pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]})
    table.to_csv(directory / "train.csv", index=False)
    (directory / "summary.json").write_text(json.dumps({"n": 2, "auc": 0.75}))
    (directory / "model.bin").write_bytes(b"\x00\x01")
    for name, content in (extra or {}).items():
        (directory / name).write_bytes(content)
    return directory


FILE_POLICY = ComparisonPolicy(required=("files",))


def _files_verdict(expected_dir, actual_dir, policy=FILE_POLICY):
    results = compare(
        {"files": output_files(expected_dir)},
        {"files": output_files(actual_dir)},
        policy,
    )
    return results[0], aggregate_verdict(results)


def test_identical_output_files_match(tmp_path):
    result, verdict = _files_verdict(
        _write_outputs(tmp_path / "ref"), _write_outputs(tmp_path / "out")
    )
    assert verdict is Verdict.MATCH, result


@pytest.mark.parametrize(
    "change",
    ["table value", "extra file", "missing file", "json value", "binary", "renamed"],
)
def test_output_file_changes_are_mismatches(tmp_path, change):
    reference = _write_outputs(tmp_path / "ref")
    table = None
    if change == "table value":
        table = pd.DataFrame({"a": [1, 2], "b": [0.5, 1.6]})
    output = _write_outputs(tmp_path / "out", table=table)
    if change == "extra file":
        (output / "stale.csv").write_text("a\n1\n")
    elif change == "missing file":
        (output / "model.bin").unlink()
    elif change == "json value":
        (output / "summary.json").write_text(json.dumps({"n": 2, "auc": 0.76}))
    elif change == "binary":
        (output / "model.bin").write_bytes(b"\x00\x02")
    elif change == "renamed":
        (output / "train.csv").rename(output / "train.tsv")
    result, verdict = _files_verdict(reference, output)
    assert verdict is Verdict.MISMATCH, result


def test_symlinked_output_file_is_not_followed(tmp_path):
    reference = _write_outputs(tmp_path / "ref")
    output = _write_outputs(tmp_path / "out")
    (output / "train.csv").unlink()
    os.symlink(reference / "train.csv", output / "train.csv")
    result, verdict = _files_verdict(reference, output)
    assert result.status is SlotStatus.NOT_COMPARED
    assert verdict is Verdict.NOT_COMPARED


def test_h5ad_output_files_compare_their_contents(tmp_path):
    reference = reference_adata()
    del reference.uns["log1p"]  # None values do not survive H5AD
    for name in ("ref", "out"):
        (tmp_path / name).mkdir()
    reference.write_h5ad(tmp_path / "ref" / "result.h5ad")
    reference.write_h5ad(tmp_path / "out" / "result.h5ad")
    assert _files_verdict(tmp_path / "ref", tmp_path / "out")[1] is Verdict.MATCH
    changed = reference.copy()
    changed.X = _bump(changed.X)
    changed.write_h5ad(tmp_path / "out" / "result.h5ad")
    assert _files_verdict(tmp_path / "ref", tmp_path / "out")[1] is Verdict.MISMATCH


# --------------------------------------------------------------------------
# Policy validation
# --------------------------------------------------------------------------

VALID_POLICY = {
    "schema_version": 1,
    "required": ["ad/X", "ad/uns"],
    "excluded": [{"path": "ad/uns/time", "rationale": "wall clock"}],
    "tolerances": [{"path": "ad/X", "rtol": 1e-6, "atol": 1e-9}],
    "sparse": "semantic",
}


def _policy_with(**changes):
    data = json.loads(json.dumps(VALID_POLICY))
    for key, value in changes.items():
        if value is None:
            data.pop(key)
        else:
            data[key] = value
    return data


def test_valid_policy_round_trips_with_a_stable_digest():
    policy = ComparisonPolicy.from_dict(VALID_POLICY)
    assert ComparisonPolicy.from_dict(policy.to_dict()) == policy
    assert ComparisonPolicy.from_dict(policy.to_dict()).digest() == policy.digest()
    looser = _policy_with(tolerances=[{"path": "ad/X", "rtol": 1e-3}])
    assert ComparisonPolicy.from_dict(looser).digest() != policy.digest()


@pytest.mark.parametrize(
    "bad",
    [
        _policy_with(extra_field=True),
        _policy_with(required=None),
        _policy_with(required=[]),
        _policy_with(required=["ad/X", "ad/X"]),
        _policy_with(required=["ad", "ad/X"]),
        _policy_with(required=["ad/*"]),
        _policy_with(required=["ad/../X"]),
        _policy_with(required=["ad//X"]),
        _policy_with(required="ad/X"),
        _policy_with(excluded=[{"path": "ad/uns", "rationale": "all of it"}]),
        _policy_with(excluded=[{"path": "other/x", "rationale": "outside"}]),
        _policy_with(excluded=[{"path": "ad/uns/time", "rationale": " "}]),
        _policy_with(excluded=[{"path": "ad/uns/time"}]),
        _policy_with(
            excluded=[{"path": "ad/uns/time", "rationale": "x", "broad": True}]
        ),
        _policy_with(
            excluded=[
                {"path": "ad/uns/a", "rationale": "x"},
                {"path": "ad/uns/a/b", "rationale": "y"},
            ]
        ),
        _policy_with(tolerances=[{"path": "ad/X", "rtol": -1}]),
        _policy_with(tolerances=[{"path": "ad/X", "rtol": True}]),
        _policy_with(tolerances=[{"path": "ad/X", "rtol": "1e-6"}]),
        _policy_with(tolerances=[{"path": "ad/X", "atol": float("inf")}]),
        _policy_with(tolerances=[{"path": "ad/X"}]),
        _policy_with(tolerances=[{"path": "ad/X", "rtol": 0.1, "scale": 2}]),
        _policy_with(tolerances=[{"path": "ad/obs", "rtol": 0.1}]),
        _policy_with(tolerances=[{"path": "ad/uns/time", "rtol": 0.1}]),
        _policy_with(
            tolerances=[{"path": "ad/X", "rtol": 0.1}, {"path": "ad/X", "rtol": 0.2}]
        ),
        _policy_with(sparse="loose"),
        _policy_with(schema_version=2),
        _policy_with(schema_version=True),
    ],
)
def test_invalid_policies_are_rejected(bad):
    with pytest.raises(ContractError):
        ComparisonPolicy.from_dict(bad)


def test_policy_json_rejects_duplicate_keys_and_nan():
    text = json.dumps(VALID_POLICY)
    assert ComparisonPolicy.from_json(text).required == ("ad/X", "ad/uns")
    with pytest.raises(ContractError, match="duplicate"):
        ComparisonPolicy.from_json(text[:-1] + ', "sparse": "semantic"}')
    with pytest.raises(ContractError, match="non-standard"):
        ComparisonPolicy.from_json(
            '{"required": ["x"], "tolerances": [{"path": "x", "rtol": NaN}]}'
        )


def test_tolerance_lookup_is_segment_aware():
    policy = ComparisonPolicy(
        required=("ad",), tolerances=(Tolerance("ad/X", rtol=0.5),)
    )
    assert policy.tolerance_for("ad/X") == (0.5, 0.0)
    assert policy.tolerance_for("ad/Xb") == (0.0, 0.0)


# --------------------------------------------------------------------------
# Verdict aggregation and report state rules
# --------------------------------------------------------------------------

MATCH_SLOT = SlotResult("ad/X", SlotStatus.MATCH, compared=4)
MISMATCH_SLOT = SlotResult("ad/obs", SlotStatus.MISMATCH, compared=4, diffs=("x",))
SKIPPED_SLOT = SlotResult("ad/uns", SlotStatus.NOT_COMPARED, not_compared=("x",))
EMPTY_SLOT = SlotResult("ad/uns", SlotStatus.MATCH, compared=0)


@pytest.mark.parametrize(
    "slots,verdict",
    [
        ((), Verdict.NOT_COMPARED),
        ((EMPTY_SLOT,), Verdict.NOT_COMPARED),
        ((MATCH_SLOT, EMPTY_SLOT), Verdict.MATCH),
        ((MATCH_SLOT, SKIPPED_SLOT), Verdict.NOT_COMPARED),
        ((MATCH_SLOT, MISMATCH_SLOT, SKIPPED_SLOT), Verdict.MISMATCH),
    ],
)
def test_aggregate_verdict(slots, verdict):
    assert aggregate_verdict(slots) is verdict


def test_slot_results_must_explain_themselves():
    with pytest.raises(ContractError):
        SlotResult("ad/X", SlotStatus.MISMATCH, compared=1)
    with pytest.raises(ContractError):
        SlotResult("ad/X", SlotStatus.NOT_COMPARED)
    with pytest.raises(ContractError):
        SlotResult("ad/X", SlotStatus.MATCH, compared=1, diffs=("x",))
    with pytest.raises(ContractError):
        SlotResult("ad/X", SlotStatus.MATCH, compared=-1)


BINDINGS = {
    "notebook_sha256": "1" * 64,
    "recipe_sha256": "2" * 64,
    "root_sha256": {"pbmc.h5ad": "3" * 64},
    "policy_sha256": "4" * 64,
    "params_sha256": "5" * 64,
    "source_revision": "a" * 40,
    "source_dirty": False,
    "env_receipt_sha256": "6" * 64,
}

REFUSAL = Refusal("ambiguous input", "step 2", RefusalOrigin.PRODUCT)


def report(**changes):
    fields = dict(
        backend="template",
        scenario_id="S1",
        requested_mode="replay",
        claim_requested=Claim.HIST,
        claim_achieved=Claim.HIST,
        verification_status=VerificationStatus.CHECKED,
        bindings=BINDINGS,
        preflight=Preflight.ADMITTED,
        execution=Execution.OK,
        verdict=Verdict.MATCH,
        slots=(MATCH_SLOT,),
    )
    fields.update(changes)
    return ReplayReport(**fields)


VALID_STATES = {
    "match": {},
    "mismatch": dict(
        slots=(MATCH_SLOT, MISMATCH_SLOT),
        verdict=Verdict.MISMATCH,
        claim_achieved=Claim.EXEC,
    ),
    "error": dict(
        execution=Execution.ERROR,
        slots=(),
        verdict=Verdict.NOT_COMPARED,
        claim_achieved=None,
        verification_status=VerificationStatus.NOT_RUN,
    ),
    "timeout": dict(
        execution=Execution.TIMEOUT,
        slots=(),
        verdict=Verdict.NOT_COMPARED,
        claim_achieved=None,
        verification_status=VerificationStatus.NOT_RUN,
    ),
    "preflight refused": dict(
        preflight=Preflight.REFUSED,
        execution=Execution.NOT_RUN,
        slots=(),
        verdict=Verdict.REFUSED,
        refusal=Refusal("input not declared", None, RefusalOrigin.HARNESS),
        claim_achieved=None,
        verification_status=VerificationStatus.NOT_RUN,
    ),
    "runtime product refusal": dict(
        execution=Execution.REFUSED_AT_RUNTIME,
        slots=(),
        verdict=Verdict.REFUSED,
        refusal=REFUSAL,
        claim_achieved=None,
        verification_status=VerificationStatus.NOT_RUN,
    ),
    "exec only": dict(
        claim_requested=Claim.EXEC,
        claim_achieved=Claim.EXEC,
        slots=(),
        verdict=Verdict.NOT_COMPARED,
        verification_status=VerificationStatus.NOT_RUN,
    ),
}


@pytest.mark.parametrize("name", sorted(VALID_STATES))
def test_valid_states_are_accepted_and_round_trip(name):
    original = report(**VALID_STATES[name])
    assert ReplayReport.from_json(original.to_json()) == original


INVALID_STATES = {
    "verdict not from slots": dict(slots=(MATCH_SLOT, MISMATCH_SLOT)),
    "match from empty comparison": dict(slots=(EMPTY_SLOT,)),
    "match without slots": dict(slots=()),
    "error claims match": dict(execution=Execution.ERROR),
    "timeout mismatch": dict(
        execution=Execution.TIMEOUT, verdict=Verdict.MISMATCH, claim_achieved=None
    ),
    "runtime refusal from harness": dict(
        execution=Execution.REFUSED_AT_RUNTIME,
        slots=(),
        verdict=Verdict.REFUSED,
        refusal=Refusal("crash", "step 2", RefusalOrigin.HARNESS),
        claim_achieved=None,
        verification_status=VerificationStatus.NOT_RUN,
    ),
    "refused without refusal": dict(
        preflight=Preflight.REFUSED,
        execution=Execution.NOT_RUN,
        slots=(),
        verdict=Verdict.REFUSED,
        claim_achieved=None,
        verification_status=VerificationStatus.NOT_RUN,
    ),
    "refused preflight that ran": dict(
        preflight=Preflight.REFUSED, verdict=Verdict.REFUSED, refusal=REFUSAL
    ),
    "refused preflight not refused": dict(
        preflight=Preflight.REFUSED,
        execution=Execution.NOT_RUN,
        slots=(),
        verdict=Verdict.NOT_COMPARED,
        claim_achieved=None,
        verification_status=VerificationStatus.NOT_RUN,
    ),
    "admitted but not run": dict(
        execution=Execution.NOT_RUN,
        slots=(),
        verdict=Verdict.NOT_COMPARED,
        claim_achieved=None,
        verification_status=VerificationStatus.NOT_RUN,
    ),
    "refusal with a match": dict(refusal=REFUSAL),
    "historical claim without match": dict(
        slots=(MATCH_SLOT, SKIPPED_SLOT),
        verdict=Verdict.NOT_COMPARED,
        verification_status=VerificationStatus.NOT_RUN,
    ),
    "claim not requested": dict(claim_requested=Claim.RERUN),
    "exec claim after error": dict(
        execution=Execution.ERROR,
        slots=(),
        verdict=Verdict.NOT_COMPARED,
        claim_achieved=Claim.EXEC,
        verification_status=VerificationStatus.NOT_RUN,
    ),
    "checked without comparison": dict(
        slots=(MATCH_SLOT, SKIPPED_SLOT),
        verdict=Verdict.NOT_COMPARED,
        claim_achieved=None,
    ),
    "not run but compared": dict(verification_status=VerificationStatus.NOT_RUN),
    "unknown mode": dict(requested_mode="best"),
    "blank scenario": dict(scenario_id=" "),
    "repeated slot": dict(slots=(MATCH_SLOT, MATCH_SLOT)),
    "bad digest": dict(bindings={**BINDINGS, "notebook_sha256": "nope"}),
    "short revision": dict(bindings={**BINDINGS, "source_revision": "abc123"}),
    "dirty as text": dict(bindings={**BINDINGS, "source_dirty": "no"}),
    "no roots": dict(bindings={**BINDINGS, "root_sha256": {}}),
    "missing binding": dict(
        bindings={k: v for k, v in BINDINGS.items() if k != "env_receipt_sha256"}
    ),
    "extra binding": dict(bindings={**BINDINGS, "verdict": "MATCH"}),
    "unknown verdict": dict(verdict="PASS"),
}


@pytest.mark.parametrize("name", sorted(INVALID_STATES))
def test_invalid_states_are_rejected(name):
    with pytest.raises(ContractError):
        report(**INVALID_STATES[name])


def test_report_json_rejects_unknown_and_missing_fields():
    data = report().to_dict()
    with pytest.raises(ContractError, match="unknown"):
        ReplayReport.from_dict({**data, "coverage_complete": True})
    data.pop("refusal")
    with pytest.raises(ContractError, match="missing"):
        ReplayReport.from_dict(data)


def test_coverage_complete_is_derived():
    assert report().coverage_complete
    partial = report(
        **INVALID_STATES["checked without comparison"]
        | {"verification_status": VerificationStatus.NOT_RUN}
    )
    assert not partial.coverage_complete


def test_validate_against_policy_checks_scope_and_binding():
    policy = ComparisonPolicy(required=("ad/X",))
    bound = report(bindings={**BINDINGS, "policy_sha256": policy.digest()})
    bound.validate(policy)
    with pytest.raises(ContractError, match="not bound"):
        report().validate(policy)
    wider = ComparisonPolicy(required=("ad/X", "ad/obs"))
    with pytest.raises(ContractError, match="required paths"):
        report(bindings={**BINDINGS, "policy_sha256": wider.digest()}).validate(wider)


def test_binding_differences_lists_every_changed_binding():
    changed = {**BINDINGS, "notebook_sha256": "7" * 64, "source_dirty": True}
    assert binding_differences(BINDINGS, BINDINGS) == []
    assert binding_differences(BINDINGS, changed) == ["notebook_sha256", "source_dirty"]


def test_end_to_end_report_from_a_comparison():
    reference = reference_adata()
    policy = all_slots_policy()
    slots = compare({"ad": reference}, {"ad": reference.copy()}, policy)
    built = report(
        slots=slots,
        verdict=aggregate_verdict(slots),
        bindings={**BINDINGS, "policy_sha256": policy.digest()},
    )
    built.validate(policy)
    assert built.coverage_complete
    restored = ReplayReport.from_json(built.to_json())
    restored.validate(policy)


# --------------------------------------------------------------------------
# Import weight
# --------------------------------------------------------------------------


def test_contract_and_comparator_import_without_the_scientific_stack():
    code = (
        "import sys\n"
        "import lobster.testing.replay_contract, lobster.testing.replay_compare\n"
        "heavy = [m for m in ('numpy', 'pandas', 'anndata', 'scipy') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    completed = subprocess.run(  # nosec B603
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
