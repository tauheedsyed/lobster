"""ML preparation tools log IR that reproduces the feature matrix and the splits."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.machine_learning.machine_learning_expert import (
    machine_learning_expert,
)
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.testing.provenance_replay import (
    RecordingDataManager,
    factory_tools,
    replay_ir,
)

MODULE = "lobster.agents.machine_learning.machine_learning_expert"


@pytest.fixture
def samples() -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(4, size=(60, 200)).astype(np.float32)
    X[:, 150:] = 0  # sparse tail exercises zero handling
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"condition": pd.Categorical(["a", "b", "c"] * 20)},
            index=[f"s{i}" for i in range(60)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(200)]),
    )


def _tools(adata):
    dm = RecordingDataManager({"data": adata})
    return dm, factory_tools(machine_learning_expert, MODULE, dm)


@pytest.mark.parametrize(
    "feature_selection,handle_zeros",
    [
        ("highly_variable", "keep"),
        ("pca", "keep"),
        ("all", "remove"),
        ("all", "impute"),
    ],
)
def test_prepare_ml_features_replays(samples, feature_selection, handle_zeros):
    original = samples.copy()
    dm, tools = _tools(samples)
    result = tools["prepare_ml_features"].invoke(
        {
            "modality_name": "data",
            "feature_selection": feature_selection,
            "n_features": 40,
            "handle_zeros": handle_zeros,
            "save_result": False,
        }
    )
    assert "Successfully prepared" in result, result
    ir = dm.last_ir("prepare_ml_features")
    assert isinstance(ir, AnalysisStep) and ir.replayable
    produced = dm.modalities["data_ml_features"]
    replayed = replay_ir(ir, adata=original.copy())["adata"]
    assert list(replayed.var_names) == list(produced.var_names)
    np.testing.assert_allclose(np.asarray(replayed.X), np.asarray(produced.X))
    assert replayed.uns["ml_preprocessing"] == produced.uns["ml_preprocessing"]


@pytest.mark.parametrize(
    "validation_size,stratify_by", [(0.2, "condition"), (0.0, None)]
)
def test_create_ml_splits_replays(samples, validation_size, stratify_by):
    original = samples.copy()
    dm, tools = _tools(samples)
    result = tools["create_ml_splits"].invoke(
        {
            "modality_name": "data",
            "test_size": 0.25,
            "validation_size": validation_size,
            "stratify_by": stratify_by,
            "random_state": 7,
            "save_result": False,
        }
    )
    assert "Successfully created" in result, result
    ir = dm.last_ir("create_ml_splits")
    assert isinstance(ir, AnalysisStep) and ir.replayable
    scope = replay_ir(ir, adata=original.copy())
    pd.testing.assert_series_equal(
        scope["adata"].obs["ml_split"], dm.modalities["data"].obs["ml_split"]
    )
    assert list(scope["adata_train"].obs_names) == list(
        dm.modalities["data_train"].obs_names
    )
    assert list(scope["adata_test"].obs_names) == list(
        dm.modalities["data_test"].obs_names
    )
    if validation_size:
        assert list(scope["adata_validation"].obs_names) == list(
            dm.modalities["data_validation"].obs_names
        )
    else:
        assert scope["adata_validation"] is None


def _split(tools, validation_size=0.2, random_state=7):
    tools["create_ml_splits"].invoke(
        {"modality_name": "data", "test_size": 0.25, "validation_size": validation_size,
         "random_state": random_state, "save_result": False}
    )


def _prepare(tools, name, **kwargs):
    args = {"modality_name": name, "feature_selection": "all", "scale": False,
            "save_result": False}
    args.update(kwargs)
    result = tools["prepare_ml_features"].invoke(args)
    assert "Successfully prepared" in result, result


def _replay_all(dm, original):
    """Replay every logged step in order, like an exported notebook."""
    scope = {"adata": original.copy()}
    for call in dm.calls:
        scope = replay_ir(call.ir, **scope)
    return scope


def _assert_same(replayed, produced):
    assert list(replayed.obs_names) == list(produced.obs_names)
    assert list(replayed.var_names) == list(produced.var_names)
    np.testing.assert_allclose(np.asarray(replayed.X), np.asarray(produced.X))


@pytest.mark.parametrize(
    "steps",
    [
        ["data_train"],
        ["data_train", "data_test"],
        ["data_test", "data_train"],
        ["data_train", "data_validation"],
        ["data_train", "data_train_ml_features"],
        ["data_train", "data_test", "data_train_ml_features"],
    ],
    ids="->".join,
)
def test_sequential_feature_steps_replay_on_their_own_input(samples, steps):
    """Each feature step must replay on the modality it used, never on held-out
    samples or on the previous step's output."""
    original = samples.copy()
    dm, tools = _tools(samples)
    _split(tools)
    for name in steps:
        _prepare(tools, name)
    scope = _replay_all(dm, original)
    for name in steps:
        _assert_same(
            scope["lobster_modalities"][f"{name}_ml_features"],
            dm.modalities[f"{name}_ml_features"],
        )
    _assert_same(scope["adata"], dm.modalities[f"{steps[-1]}_ml_features"])


def test_same_split_prepared_twice_uses_the_split_both_times(samples):
    original = samples.copy()
    dm, tools = _tools(samples)
    _split(tools)
    _prepare(tools, "data_train", feature_selection="pca", n_features=10)
    _prepare(tools, "data_train")  # again from the split, not from the PCA output
    scope = _replay_all(dm, original)
    _assert_same(scope["adata"], dm.modalities["data_train_ml_features"])
    assert scope["adata"].n_vars == 200


def test_split_retained_from_an_earlier_split_replays_on_that_split(samples):
    """Re-splitting without validation keeps the old validation modality; its
    feature step must replay on those samples, not on the whole dataset."""
    original = samples.copy()
    dm, tools = _tools(samples)
    _split(tools, validation_size=0.2, random_state=7)
    _split(tools, validation_size=0.0, random_state=42)
    _prepare(tools, "data_validation")
    produced = dm.modalities["data_validation_ml_features"]
    assert produced.n_obs == 9
    _assert_same(_replay_all(dm, original)["adata"], produced)


def test_resplit_after_a_feature_step_splits_the_parent(samples):
    """The second split cell must split the parent, not the previous step's features."""
    original = samples.copy()
    dm, tools = _tools(samples)
    _split(tools, random_state=7)
    _prepare(tools, "data_train", feature_selection="pca", n_features=10)
    _split(tools, random_state=42)
    _prepare(tools, "data_train")
    _assert_same(_replay_all(dm, original)["adata"], dm.modalities["data_train_ml_features"])


@pytest.mark.parametrize("second_step", ["prepare_original", "split_original"])
def test_original_is_still_available_after_its_features(samples, second_step):
    """Preparing features must not lose the original: a later step on it
    replays on the original, not on the previous step's output."""
    original = samples.copy()
    dm, tools = _tools(samples)
    _prepare(tools, "data", feature_selection="pca", n_features=10)
    if second_step == "prepare_original":
        _prepare(tools, "data")
        expected = dm.modalities["data_ml_features"]
    else:
        _split(tools)
        _prepare(tools, "data_train")
        expected = dm.modalities["data_train_ml_features"]
    replayed = _replay_all(dm, original)["adata"]
    _assert_same(replayed, expected)
    assert replayed.n_vars == 200


def test_replay_refuses_a_name_a_non_ml_step_may_have_replaced(samples):
    """After a non-ML cell rebinds adata, a registered name is ambiguous."""
    original = samples.copy()
    dm, tools = _tools(samples)
    _split(tools)
    scope = _replay_all(dm, original)
    scope["adata"] = original[:12, :3].copy()  # e.g. a re-import under the same name
    _prepare(tools, "data")
    with pytest.raises(ValueError, match="Ambiguous input 'data'"):
        replay_ir(dm.last_ir("prepare_ml_features"), **scope)


def test_ambiguity_persists_across_later_ml_steps(samples):
    """An unrelated ML step after a rebind must not make old names trusted again."""
    original = samples.copy()
    dm, tools = _tools(samples)
    _split(tools)
    scope = _replay_all(dm, original)
    scope["adata"] = original[:12, :3].copy()  # a non-ML cell rebinds adata
    _prepare(tools, "data_train")
    _prepare(tools, "data")
    with pytest.raises(ValueError, match="Ambiguous input"):
        for call in dm.calls[1:]:
            scope = replay_ir(call.ir, **scope)


@pytest.mark.parametrize("first", [{"feature_selection": "highly_variable", "n_features": 10},
                                   {"feature_selection": "all", "scale": True}])
def test_loaded_split_stays_the_input_for_later_steps(samples, first):
    """A split loaded as notebook input is registered, so a second step on it does
    not pick up the first step's output (which no longer carries the split tag)."""
    dm, tools = _tools(samples)
    _split(tools)
    loaded = dm.modalities["data_train"].copy()
    dm.calls.clear()
    _prepare(tools, "data_train", **first)
    assert "ml_split_source" not in dm.modalities["data_train_ml_features"].uns
    _prepare(tools, "data_train")
    scope = {"adata": loaded}
    for call in dm.calls:
        scope = replay_ir(call.ir, **scope)
    _assert_same(scope["adata"], dm.modalities["data_train_ml_features"])


def test_split_loaded_as_input_replays_without_its_split_step(samples):
    """A saved split loaded in a new session carries its tag and replays alone."""
    dm, tools = _tools(samples)
    _split(tools)
    _prepare(tools, "data_train")
    loaded = dm.modalities["data_train"].copy()
    replayed = replay_ir(dm.last_ir("prepare_ml_features"), adata=loaded)["adata"]
    _assert_same(replayed, dm.modalities["data_train_ml_features"])


def test_split_feature_step_fails_closed_without_its_split(samples):
    original = samples.copy()
    dm, tools = _tools(samples)
    _split(tools)
    _prepare(tools, "data_train")
    with pytest.raises(ValueError, match="create_ml_splits"):
        replay_ir(dm.last_ir("prepare_ml_features"), adata=original.copy())


@pytest.mark.parametrize("with_parent", [False, True])
def test_unrelated_train_named_modality_gets_no_split_selection(samples, with_parent):
    """A modality merely named *_train (even with ml_split == 'train') is not a
    split unless create_ml_splits tagged it."""
    dm, tools = _tools(samples)
    if with_parent:
        _split(tools)
    cohort = samples[:8].copy()
    cohort.obs["ml_split"] = "train"
    cohort.uns.pop("ml_split_source", None)
    dm.modalities["data_train"] = cohort  # replaces any real split
    _prepare(tools, "data_train")
    code = dm.last_ir("prepare_ml_features").render()
    assert "split_id" not in code
