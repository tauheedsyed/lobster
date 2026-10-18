"""Annotation replay must address the exact live obs key, including escapes."""

import importlib
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from tests.unit.core.test_ir_template_fidelity import _replay


@pytest.fixture
def annotation_tools(monkeypatch):
    """Construct real tool closures without creating an LLM or agent loop."""
    module = importlib.import_module("lobster.agents.transcriptomics.annotation_expert")
    monkeypatch.setattr(module, "get_settings", lambda: Mock())
    monkeypatch.setattr(module, "create_llm", lambda *args, **kwargs: Mock())
    monkeypatch.setattr(module, "create_react_agent", lambda **kwargs: kwargs["tools"])
    monkeypatch.setattr(module.component_registry, "get_service", lambda name: None)
    manager = Mock()
    manager.list_modalities.return_value = ["synthetic"]
    tools = module.annotation_expert(manager)
    return {tool.name: tool for tool in tools}, manager


@pytest.mark.parametrize(
    "name", ["leiden", r"cluster\nkey", r"cluster\u0041", "cluster'\"{key}\n"]
)
def test_auto_annotation_replays_exact_cluster_key(name, annotation_tools):
    tools, manager = annotation_tools
    source = ad.AnnData(
        X=np.array(
            [
                [9, 8, 1, 2],
                [8, 9, 2, 1],
                [7, 8, 1, 2],
                [1, 2, 9, 8],
                [2, 1, 8, 9],
                [1, 2, 7, 8],
            ],
            dtype=float,
        ),
        var=pd.DataFrame(index=["CD3D", "CD3E", "MS4A1", "CD79A"]),
    )
    source.obs[name] = pd.Categorical(["0", "0", "0", "1", "1", "1"])
    decoded = name.encode().decode("unicode_escape") if "\\" in name else name
    if decoded != name:
        source.obs[decoded] = pd.Categorical(["0", "1", "0", "1", "0", "1"])
    manager.get_modality.return_value = source.copy()
    response = tools["annotate_cell_types_auto"].invoke(
        {
            "modality_name": "synthetic",
            "cluster_key": name,
            "reference_markers": {"T": ["CD3D", "CD3E"], "B": ["MS4A1", "CD79A"]},
        }
    )
    assert "Successfully" in response, response
    live = manager.store_modality.call_args.kwargs["adata"]
    ir = manager.log_tool_usage.call_args.kwargs["ir"]
    replay = _replay(ir, source)
    pd.testing.assert_frame_equal(replay.obs, live.obs)
    np.testing.assert_array_equal(replay.X, live.X)
    if name == "leiden":
        assert "cluster_key='leiden'" in ir.render()


@pytest.mark.parametrize(
    "name", ["gene_set_score", r"signal\nscore", r"signal\u0041", "signal'\"{score}\n"]
)
def test_gene_set_score_replays_exact_column(name, annotation_tools):
    tools, manager = annotation_tools
    source = ad.AnnData(
        X=np.random.default_rng(9).poisson(5, size=(8, 80)).astype(float),
        var=pd.DataFrame(index=[f"G{i}" for i in range(80)]),
    )
    source.obs[name] = -99.0
    decoded = name.encode().decode("unicode_escape") if "\\" in name else name
    if decoded != name:
        source.obs[decoded] = -77.0
    manager.get_modality.return_value = source.copy()
    response = tools["score_gene_set"].invoke(
        {
            "modality_name": "synthetic",
            "gene_list": ["G0", "G1"],
            "score_name": name,
            "ctrl_size": 2,
        }
    )
    assert "Successfully" in response, response
    live = manager.store_modality.call_args.kwargs["adata"]
    ir = manager.log_tool_usage.call_args.kwargs["ir"]
    replay = _replay(ir, source)
    pd.testing.assert_frame_equal(replay.obs, live.obs)
    np.testing.assert_array_equal(replay.X, live.X)
    if name == "gene_set_score":
        assert 'score_name="gene_set_score"' in ir.render()
