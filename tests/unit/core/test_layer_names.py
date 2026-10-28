"""The .X alias in AnnData layers (anndata>=0.13 lists it under the key None).

Every place that lists layer names must skip that key: joins over it raise
TypeError, and counts over it report one layer too many.
"""

import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.cli_internal.commands.output_adapter import JsonOutputAdapter
from lobster.core.runtime.data_manager import DataManagerV2


def _adata(with_layer: bool) -> ad.AnnData:
    rng = np.random.default_rng(0)
    adata = ad.AnnData(
        X=rng.random((6, 8), dtype=np.float32),
        obs=pd.DataFrame(index=[f"s{i}" for i in range(6)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(8)]),
    )
    if with_layer:
        adata.layers["counts"] = np.ones((6, 8), dtype=np.float32)
    return adata


@pytest.fixture(params=[True, False], ids=["counts_layer", "no_layer"])
def case(request, tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path, auto_scan=False)
    dm.store_modality(name="m", adata=_adata(request.param), step_summary="t")
    return dm, (["counts"] if request.param else [])


def _text(output: JsonOutputAdapter) -> str:
    return json.dumps(output.to_dict(), default=str)


def test_layer_names_helper():
    from lobster.core.utils.layer_keys import layer_names

    assert layer_names(_adata(False)) == []
    assert layer_names(_adata(True)) == ["counts"]
    assert layer_names(SimpleNamespace()) == []
    assert layer_names(SimpleNamespace(layers=None)) == []


def test_modality_management_info(case):
    from lobster.services.data_management.modality_management_service import (
        ModalityManagementService,
    )

    dm, expected = case
    info, _, _ = ModalityManagementService(dm).get_modality_info("m")
    assert info["layers"] == expected


def test_data_manager_status_and_summaries(case):
    dm, expected = case
    assert dm.get_workspace_status()["modality_details"]["m"]["layers"] == expected

    dm.current_dataset = "m"
    assert dm.get_data_summary()["layers"] == expected
    dm.current_dataset = None
    summary = dm.get_data_summary()
    assert summary["modalities"]["m"]["layers"] == len(expected)

    technical = dm.get_technical_summary()
    if expected:
        assert "- Data layers: counts\n" in technical
    else:
        assert "- Data layers" not in technical


def test_mudata_backend_info(case):
    mudata = pytest.importorskip("mudata")
    from lobster.core.backends.mudata_backend import MuDataBackend

    dm, expected = case
    mdata = mudata.MuData({"rna": dm.get_modality("m")})
    info = MuDataBackend.__new__(MuDataBackend).get_modality_info(mdata)
    assert info["rna"]["layers"] == expected


def test_summarize_modality_tool(case):
    from lobster.tools.knowledgebase_tools import create_summarize_modality_tool

    dm, expected = case
    tool = create_summarize_modality_tool(dm)
    listing = tool.invoke({})
    detail = tool.invoke({"modality_name": "m"})
    assert "None" not in listing.split("**m**", 1)[1].splitlines()[0]
    assert f"**Layers:** {expected or 'None'}" in detail


def test_cli_layer_rows(case):
    from lobster.cli_internal.commands.heavy.data_commands import data_summary
    from lobster.cli_internal.commands.heavy.modality_commands import (
        modalities_list,
        modality_describe,
    )
    from lobster.cli_internal.commands.light.workspace_commands import (
        workspace_status,
    )

    dm, expected = case
    dm.current_dataset = "m"
    client = SimpleNamespace(data_manager=dm, workspace_path=dm.workspace_path)
    for command in (data_summary, modalities_list, workspace_status):
        output = JsonOutputAdapter()
        command(client, output)
        text = _text(output)
        if expected:
            assert '"counts"' in text and "None, counts" not in text, command
        assert '"None"' not in text, command

    output = JsonOutputAdapter()
    modality_describe(client, output, "m")
    text = _text(output)
    assert "None:" not in text
    if expected:
        assert "Layers (1)" in text
    else:
        assert "Layers (" not in text
