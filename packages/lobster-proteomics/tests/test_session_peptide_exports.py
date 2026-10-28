"""Peptide reports use the data manager's session export directory."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.proteomics.peptide_tools import create_peptide_tools
from lobster.core.runtime.data_manager import DataManagerV2


@pytest.mark.parametrize("pass_workspace", [False, True])
def test_peptide_csv_is_a_visible_session_export(tmp_path, monkeypatch, pass_workspace):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    adata = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(
            {"sequence": ["ACDE", "FGHI"]}, index=["peptide_a", "peptide_b"]
        ),
    )
    dm.store_modality("peptides", adata)
    tools = {
        t.name: t
        for t in create_peptide_tools(
            dm, workspace_path=dm.workspace_path if pass_workspace else None
        )
    }
    response = tools["export_peptide_report"].invoke(
        {"modality_name": "peptides", "output_format": "csv"}
    )
    path = dm.exports_dir / "peptides_report.csv"
    assert path.is_file(), response
    assert path.read_bytes()
    assert list(cwd.iterdir()) == []
    assert not (dm.workspace_path / path.name).exists()
    reported = Path(response.split("file=", 1)[1].splitlines()[0])
    if not reported.is_absolute():
        reported = dm.workspace_path / reported
    assert reported.resolve() == path.resolve()
    assert any(
        Path(item["path"]).resolve() == path.resolve()
        for item in dm.list_workspace_files()["exports"]
    )
    pd.testing.assert_frame_equal(pd.read_csv(path, index_col=0), adata.obs)
