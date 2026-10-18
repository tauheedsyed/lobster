"""Peptide IR must reproduce all live output slots, not a partial calculation."""

import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.services.peptidomics.peptide_activity_service import PeptideActivityService
from lobster.services.peptidomics.peptide_digestion_service import (
    PeptideDigestionService,
)
from lobster.services.peptidomics.peptide_property_service import PeptidePropertyService
from tests.unit.core.test_ir_template_fidelity import _replay


@pytest.fixture(params=["builtin", "peptides"])
def property_backend(request, monkeypatch):
    if request.param == "builtin":
        monkeypatch.setitem(sys.modules, "peptides", None)
    else:
        pytest.importorskip("peptides", reason="optional peptide calculation backend")

    return request.param


def _source(existing_x=False):
    obs = pd.DataFrame(
        {"sequence": ["KLAKLAKKLAKLAK", "rrr-rrr", "AADE", "", "KLAKLAKKLAKLAK", "a"]},
        index=["p0", "p1", "p2", "p3", "p4", "p5"],
    )
    source = ad.AnnData(obs=obs, X=np.ones((6, 2)) if existing_x else None)
    source.uns["sentinel"] = {"unchanged": 1}
    source.obsm["positions"] = np.arange(12).reshape(6, 2)
    return source


def _assert_same_output(replay, live):
    pd.testing.assert_frame_equal(replay.obs, live.obs)
    pd.testing.assert_frame_equal(replay.var, live.var)
    if live.X is None:
        assert replay.X is None
    else:
        np.testing.assert_allclose(replay.X, live.X, equal_nan=True)
    assert replay.uns == live.uns
    assert set(replay.obsm) == set(live.obsm)
    for key in live.obsm:
        np.testing.assert_array_equal(replay.obsm[key], live.obsm[key])
    assert set(replay.layers) == set(live.layers)
    for key in live.layers:
        np.testing.assert_array_equal(replay.layers[key], live.layers[key])


@pytest.mark.parametrize("existing_x", [False, True])
@pytest.mark.parametrize(
    "properties", [None, ["length", "charge", "hydrophobic_ratio"]]
)
def test_property_template_replays_all_requested_properties(
    property_backend, existing_x, properties
):
    source = _source(existing_x)
    live, _, ir = PeptidePropertyService().calculate_properties(
        source, properties=properties
    )
    replay = _replay(ir, source)
    _assert_same_output(replay, live)


@pytest.mark.parametrize(
    "activity_type", ["antimicrobial", "cell_penetrating", "toxic"]
)
def test_activity_template_replays_scores_and_labels(property_backend, activity_type):
    source = _source(existing_x=True)
    live, _, ir = PeptideActivityService().predict_activity(
        source, activity_type=activity_type
    )
    replay = _replay(ir, source)
    _assert_same_output(replay, live)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enzyme": "trypsin", "missed_cleavages": 1, "min_length": 2, "max_length": 20},
        {"enzyme": "asp_n", "min_length": 1, "max_length": 20},
        {
            "custom_pattern": r"(?<=K)|['\"\\]",
            "missed_cleavages": 2,
            "min_length": 1,
            "max_length": 12,
        },
        {"enzyme": "trypsin", "min_length": 100, "max_length": 200},
    ],
)
def test_digestion_template_replays_fragments_including_empty_output(kwargs):
    source = _source()
    # Exercise the non-default sequence-column parameter as well.
    source.obs.rename(columns={"sequence": "parent'\\sequence"}, inplace=True)
    live, _, ir = PeptideDigestionService().digest(
        source, sequence_col="parent'\\sequence", **kwargs
    )
    replay = _replay(ir, source)
    _assert_same_output(replay, live)
