"""Fingerprint analysis computes a matrix, not a figure."""

import json
from unittest.mock import MagicMock

import numpy as np

from lobster.agents.drug_discovery.cheminformatics_tools import (
    create_cheminformatics_tools,
)
from lobster.figure_templates import get_template
from lobster.services.execution.custom_code_execution_service import (
    CustomCodeExecutionService,
)


def test_fingerprint_similarity_keeps_results_without_drawing(tmp_path):
    manager = MagicMock()
    manager.workspace_path = tmp_path
    service = MagicMock()
    matrix = [[1.0, 0.5], [0.5, 1.0]]
    stats = {
        "n_compounds": 2,
        "similarity_matrix": matrix,
        "compound_ids": ["CC", "CCC"],
        "mean_similarity": 0.5,
        "min_similarity": 0.5,
        "max_similarity": 0.5,
        "n_bits": 2048,
        "smiles_list": ["CC", "CCC"],
    }
    activity_ir = object()
    service.fingerprint_similarity.return_value = (None, stats, activity_ir)
    tools = create_cheminformatics_tools(
        manager, service, MagicMock(), MagicMock(), MagicMock()
    )
    tool = next(t for t in tools if t.name == "fingerprint_similarity")
    reply = tool.invoke({"smiles_list": "CC, CCC"})

    assert "chem.similarity_heatmap" in reply and "0.5000" in reply
    service.fingerprint_similarity.assert_called_once_with(
        ["CC", "CCC"], fingerprint="morgan", radius=2
    )
    manager.plot_manager.add_plot.assert_not_called()
    manager.plot_manager.save_plots_to_workspace.assert_not_called()
    assert manager.log_tool_usage.call_args.args[2]["similarity_matrix"] == matrix
    assert manager.log_tool_usage.call_args.kwargs["ir"] is activity_ir
    result_ref = manager.log_tool_usage.call_args.args[1]["result_ref"]
    assert result_ref in reply
    path = tmp_path / result_ref
    assert path.is_file() and path.parent == tmp_path / "metadata"
    # Use the actual custom-code workspace-key resolver, then draw the exact
    # retained payload; chemistry is never invoked again for this figure.
    resolved = CustomCodeExecutionService(manager)._resolve_workspace_key_path(
        path.stem
    )
    assert resolved == path
    saved = json.loads(resolved.read_text())["data"]
    assert saved["coverage"] == "all 2x2, unsampled"
    assert saved["compound_mapping"] == [
        {"id": "CC", "smiles": "CC"},
        {"id": "CCC", "smiles": "CCC"},
    ]
    assert saved["parameters"] == {"fingerprint": "morgan", "radius": 2, "n_bits": 2048}
    figure = get_template("chem.similarity_heatmap")(
        saved["analysis"]["similarity_matrix"],
        compound_ids=saved["analysis"]["compound_ids"],
    )
    np.testing.assert_array_equal(figure.data[0].z, matrix)
    assert "similarity_matrix" not in figure.layout.meta["lobster_template"]["args"]
    assert service.fingerprint_similarity.call_count == 1
    manager.plot_manager.add_plot.assert_not_called()
