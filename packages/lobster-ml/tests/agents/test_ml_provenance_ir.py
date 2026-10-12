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


@pytest.mark.parametrize("split", ["train", "test"])
def test_features_on_a_split_replay_on_that_split_only(samples, split):
    """split -> prepare(<split>) must not replay feature preparation on all samples.

    The notebook carries a single `adata`, so the feature step has to select the
    split itself; otherwise held-out samples leak into the training features.
    """
    original = samples.copy()
    dm, tools = _tools(samples)
    tools["create_ml_splits"].invoke(
        {
            "modality_name": "data",
            "test_size": 0.25,
            "validation_size": 0.0,
            "random_state": 7,
            "save_result": False,
        }
    )
    tools["prepare_ml_features"].invoke(
        {
            "modality_name": f"data_{split}",
            "feature_selection": "all",
            "scale": False,
            "save_result": False,
        }
    )
    produced = dm.modalities[f"data_{split}_ml_features"]

    # With the split's variables available (sequential notebook replay) ...
    scope = replay_ir(dm.last_ir("create_ml_splits"), adata=original.copy())
    scope = replay_ir(dm.last_ir("prepare_ml_features"), **scope)
    assert list(scope["adata"].obs_names) == list(produced.obs_names)
    np.testing.assert_allclose(np.asarray(scope["adata"].X), np.asarray(produced.X))

    # ... and with only the split-annotated adata (the split cell's variables gone).
    annotated = replay_ir(dm.last_ir("create_ml_splits"), adata=original.copy())["adata"]
    alone = replay_ir(dm.last_ir("prepare_ml_features"), adata=annotated)["adata"]
    assert sorted(alone.obs_names) == sorted(produced.obs_names)
    assert set(alone.obs["ml_split"]) == {split}


@pytest.mark.parametrize("with_parent", [False, True])
def test_unrelated_train_named_modality_gets_no_split_selection(samples, with_parent):
    """A modality that merely ends in _train, with ml_split == 'train', is not a split
    unless its samples are exactly the parent's train samples."""
    dm, tools = _tools(samples)
    if with_parent:
        tools["create_ml_splits"].invoke(
            {"modality_name": "data", "test_size": 0.25, "validation_size": 0.0,
             "random_state": 7, "save_result": False}
        )
    cohort = samples[:8].copy()
    cohort.obs["ml_split"] = "train"
    dm.modalities["data_train"] = cohort  # replaces any real split
    tools["prepare_ml_features"].invoke(
        {"modality_name": "data_train", "feature_selection": "all", "scale": False,
         "save_result": False}
    )
    code = dm.last_ir("prepare_ml_features").render()
    assert "ml_split" not in code


def test_split_selection_fails_closed_and_ignores_mismatched_views(samples):
    unsplit = samples.copy()  # create_ml_splits annotates the stored modality
    dm, tools = _tools(samples)
    tools["create_ml_splits"].invoke(
        {"modality_name": "data", "test_size": 0.25, "validation_size": 0.0,
         "random_state": 7, "save_result": False}
    )
    tools["prepare_ml_features"].invoke(
        {"modality_name": "data_train", "feature_selection": "all", "scale": False,
         "save_result": False}
    )
    ir = dm.last_ir("prepare_ml_features")
    expected = sorted(dm.modalities["data_train"].obs_names)

    # No ml_split column: refuse instead of preparing features on everything.
    with pytest.raises(ValueError, match="ml_split"):
        replay_ir(ir, adata=unsplit)

    # A stale adata_train with other samples is ignored; the mask is used.
    annotated = dm.modalities["data"].copy()
    stale = annotated[:5].copy()
    replayed = replay_ir(ir, adata=annotated, adata_train=stale)["adata"]
    assert sorted(replayed.obs_names) == expected
