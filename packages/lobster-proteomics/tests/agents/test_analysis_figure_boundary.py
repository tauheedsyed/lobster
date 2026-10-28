"""Analysis wrappers retain data/provenance but never produce figures."""

import importlib
from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "tool_name,service_module,service_class,service_method,template",
    [
        (
            "find_differential_proteins",
            "lobster.services.analysis.proteomics_differential_service",
            "ProteomicsDifferentialService",
            "perform_differential_expression",
            "proteomics.volcano",
        ),
        (
            "run_pathway_enrichment",
            "lobster.services.analysis.proteomics_pathway_service",
            "ProteomicsPathwayService",
            "run_enrichment",
            "proteomics.pathway_bubble",
        ),
    ],
)
def test_analysis_preserves_outputs_without_figure(
    tool_name, service_module, service_class, service_method, template, tmp_path
):
    module = importlib.import_module("lobster.agents.proteomics.de_analysis_expert")
    data = ad.AnnData(
        np.ones((6, 2)),
        obs=pd.DataFrame(
            {"group": ["a"] * 3 + ["b"] * 3}, index=[f"s{i}" for i in range(6)]
        ),
        var=pd.DataFrame(index=["P1", "P2"]),
    )
    data.uns["differential_expression"] = {
        "all_results": [
            {
                "protein": "P1",
                "comparison": "a_vs_b",
                "log2_fold_change": 2.0,
                "p_adjusted": 0.01,
            }
        ]
    }
    data.uns["pathway_enrichment"] = {
        "results": [{"Term": "Pathway", "Adjusted P-value": 0.01, "Overlap": "1/5"}]
    }
    expected = data.copy()
    activity_ir = object()
    service = MagicMock()
    getattr(service, service_method).return_value = (expected, {}, activity_ir)
    manager = MagicMock()
    manager.workspace_path = tmp_path
    manager.list_modalities.return_value = ["proteins"]
    manager.get_modality.return_value = data

    with (
        patch.object(module, "create_llm", return_value=MagicMock()),
        patch.object(module, "create_react_agent", side_effect=lambda **kwargs: kwargs),
        patch(f"{service_module}.{service_class}", return_value=service),
        patch.object(module, service_class, return_value=service, create=True),
    ):
        agent = module.de_analysis_expert(manager, workspace_path=tmp_path)
        tools = {tool.name: tool for tool in agent["tools"]}
        args = {"modality_name": "proteins"}
        if tool_name == "find_differential_proteins":
            args.update(group_column="group", platform_type="mass_spec")
        reply = tools[tool_name].invoke(args)

    assert "Complete" in reply and template in reply, reply
    assert "generated" not in reply.lower()
    manager.plot_manager.add_plot.assert_not_called()
    manager.plot_manager.save_plots_to_workspace.assert_not_called()
    manager.store_modality.assert_called_once()
    assert manager.store_modality.call_args.kwargs["adata"] is expected
    manager.log_tool_usage.assert_called_once()
    assert manager.log_tool_usage.call_args.kwargs["ir"] is activity_ir
    assert (
        data.uns["differential_expression"] == expected.uns["differential_expression"]
    )
