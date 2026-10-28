"""The existing ML CSV export is a session-path control."""

from importlib.util import find_spec
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools

# ML is optional: skip an absent package, not broken imports in a present one.
if find_spec("lobster.agents.machine_learning") is None:
    pytest.skip("Optional ML agent package is not installed", allow_module_level=True)

from lobster.agents.machine_learning.machine_learning_expert import (  # noqa: E402
    machine_learning_expert,
)


def test_ml_csv_export_already_uses_session_exports(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(index=["sample_a", "sample_b"]),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )
    dm.store_modality("features", adata)
    tools = factory_tools(
        machine_learning_expert,
        "lobster.agents.machine_learning.machine_learning_expert",
        dm,
    )
    # Use the exports root so DataManager's non-recursive listing can see it.
    response = tools["export_for_ml_framework"].invoke(
        {
            "modality_name": "features",
            "format": "csv",
            "include_labels": False,
            "output_dir": ".",
        }
    )
    path = dm.exports_dir / "features_ml_data.csv"
    assert path.is_file(), response
    assert path.read_bytes()
    assert list(cwd.iterdir()) == []
    reported_dir = Path(response.split("**Export directory**: ", 1)[1].splitlines()[0])
    reported_name = response.split("- features_ml_data.csv", 1)
    assert len(reported_name) == 2
    assert (reported_dir / "features_ml_data.csv").resolve() == path.resolve()
    assert any(
        Path(item["path"]).resolve() == path.resolve()
        for item in dm.list_workspace_files()["exports"]
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(path, index_col=0),
        pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names),
    )
