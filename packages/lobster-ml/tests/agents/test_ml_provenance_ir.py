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
