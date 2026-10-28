"""Eight legacy producers: real trusted disk store, mocked drawing/LLM.

Local correctness evidence, not independent validation or release acceptance.
The reader assertions were first strict-xfailed; all 24 assertions
passed normally before the markers were removed.
"""

import copy
import importlib
import json
import re
from types import SimpleNamespace
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.sparse import csr_matrix

from lobster.core.figures.errors import FigureError
from lobster.core.figures.records import STATUS_PENDING, CommitResult
from lobster.core.figures.saver import HEADER_PENDING
from lobster.core.figures.store import LocalFigureStore
from lobster.core.plot_manager import PlotManager
from tests.qc_fixtures import prepare_stored_qc

WRAPPERS = {
    "create_umap_plot": {"color_by": "cluster"},
    "create_qc_plots": {"metrics": ["n_counts"]},
    "create_violin_plot": {"genes": ["g0", "g1"], "groupby": "cluster"},
    "create_feature_plot": {"genes": ["g0", "g1"]},
    "create_dot_plot": {"genes": ["g0", "g1"], "groupby": "cluster"},
    "create_heatmap": {"groupby": "cluster", "genes": ["g0", "g1"]},
    "create_elbow_plot": {},
    "create_cluster_composition_plot": {
        "cluster_col": "cluster",
        "sample_col": "sample",
    },
}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    module = importlib.import_module("lobster.agents.visualization_expert")
    store = LocalFigureStore(tmp_path, session_id="wrapper-test")
    manager = PlotManager(tmp_path)
    manager.configure_figure_lifecycle(
        store_provider=lambda: store, session_id_provider=lambda: "wrapper-test"
    )
    data = ad.AnnData(
        np.arange(8000, dtype=float).reshape(8, 1000) % 7,
        obs=pd.DataFrame(
            {
                "cluster": ["a", "b"] * 4,
                "sample": ["s1", "s1", "s2", "s2"] * 2,
            },
            index=[f"cell-{i}" for i in range(8)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(1000)]),
    )
    prepare_stored_qc(data)
    data.obsm["X_umap"] = np.arange(16, dtype=float).reshape(8, 2)
    data.uns["pca"] = {"variance_ratio": np.array([0.5, 0.3, 0.2])}
    events = []
    activities = []
    stages = []
    draws = {}
    state = {"failure": None, "provenance": True}
    original_add = manager.add_plot
    original_commit = manager.commit_staged

    def stage(**kwargs):
        events.append("stage")
        if state["failure"] == "stage_raise":
            raise RuntimeError("injected staging failure")
        if state["failure"] == "stage_return":
            return None, {"error": "injected staging failure"}, None
        result = original_add(**kwargs)
        stages.append((result[0], kwargs))
        return result

    def log(**kwargs):
        events.append("log")
        if state["failure"] == "log":
            raise RuntimeError("injected activity failure")
        activities.append(kwargs)
        return {"id": f"activity-{len(activities)}"} if state["provenance"] else None

    def commit(activity, **kwargs):
        events.append("commit")
        assert events[-2] == "log"
        assert kwargs["tool_call_id"]
        assert kwargs["staged_ids"] == [stages[-1][0]]
        if state["failure"] == "commit":
            raise RuntimeError("injected trusted-save failure")
        return original_commit(activity, **kwargs)

    manager.add_plot = stage
    manager.commit_staged = commit
    manager.add_visualization_record = Mock(
        side_effect=AssertionError("No phantom legacy history writes")
    )
    manager.save_plots_to_workspace = Mock(
        side_effect=AssertionError("No legacy bulk save")
    )
    dm = SimpleNamespace(
        plot_manager=manager,
        modalities={"cells": data},
        current_dataset="cells",
        get_modality=lambda name: data,
        list_modalities=lambda: ["cells"],
        workspace_path=tmp_path,
        log_tool_usage=log,
    )
    service = SimpleNamespace()
    for name in WRAPPERS:

        def draw(_name=name, **kwargs):
            events.append("draw")
            if state["failure"] == "draw":
                raise RuntimeError("injected drawing failure")
            draws[_name] = kwargs
            figure = go.Figure(go.Scatter(x=[1, 2], y=[2, 1], uid="stable-trace"))
            figure.update_layout(title="Caller service title")
            return figure

        setattr(service, name, draw)
    monkeypatch.setattr(module, "SingleCellVisualizationService", lambda: service)
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: SimpleNamespace(get_agent_llm_params=lambda name: {}),
    )
    monkeypatch.setattr(module, "create_llm", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "create_react_agent", lambda **kwargs: kwargs)
    execution = importlib.import_module(
        "lobster.services.execution.custom_code_execution_service"
    )
    code_tool = importlib.import_module("lobster.tools.custom_code_tool")
    monkeypatch.setattr(execution, "CustomCodeExecutionService", lambda dm: object())
    monkeypatch.setattr(
        code_tool,
        "create_execute_custom_code_tool",
        lambda *args: SimpleNamespace(name="execute_custom_code"),
    )
    graph = module.visualization_expert(dm)
    return SimpleNamespace(
        tools={tool.name: tool for tool in graph["tools"]},
        prompt=graph["prompt"],
        store=store,
        manager=manager,
        data=data,
        events=events,
        state=state,
        activities=activities,
        stages=stages,
        draws=draws,
        service=service,
        dm=dm,
    )


def invoke(harness, name, *, call_id=None, save=True):
    return (
        harness.tools[name]
        .invoke(
            {
                "type": "tool_call",
                "name": name,
                "id": call_id or f"call-{name}",
                "args": {"modality_name": "cells", **WRAPPERS[name], "save_plot": save},
            }
        )
        .content
    )


def snapshot(data):
    return (
        data.X.copy(),
        data.obs.copy(deep=True),
        data.var.copy(deep=True),
        copy.deepcopy(dict(data.obsm)),
        copy.deepcopy(data.uns),
    )


def assert_unchanged(data, before):
    np.testing.assert_array_equal(data.X, before[0])
    pd.testing.assert_frame_equal(data.obs, before[1])
    pd.testing.assert_frame_equal(data.var, before[2])
    assert set(data.obsm) == set(before[3])
    for key in data.obsm:
        np.testing.assert_array_equal(data.obsm[key], before[3][key])
    assert set(data.uns) == set(before[4])
    for key in data.uns["pca"]:
        np.testing.assert_array_equal(data.uns["pca"][key], before[4]["pca"][key])


def assert_no_history(harness):
    assert harness.manager.visualization_state["history"] == []
    assert harness.manager.visualization_state["plot_registry"] == {}
    harness.manager.add_visualization_record.assert_not_called()
    harness.manager.save_plots_to_workspace.assert_not_called()


def test_qc_fixture_values_match_the_assessed_matrix(harness):
    totals = np.sum(harness.data.X, axis=1)
    detected = np.count_nonzero(harness.data.X, axis=1)
    np.testing.assert_array_equal(harness.data.obs["n_counts"], totals)
    np.testing.assert_array_equal(harness.data.obs["n_genes"], detected)
    assert totals[0] == 2997
    assert detected[6] == 858
    np.testing.assert_allclose(
        harness.data.obs["gene_detection_rate"],
        detected / max(harness.data.n_vars, 1) * 100,
    )
    np.testing.assert_allclose(harness.data.obs["log10_counts"], np.log10(totals + 1))
    np.testing.assert_allclose(harness.data.obs["log10_genes"], np.log10(detected + 1))
    for metric in ("percent_mito", "percent_ribo", "percent_hb"):
        np.testing.assert_array_equal(harness.data.obs[metric], np.zeros(8))


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("n_vars", [0, 3])
def test_qc_fixture_preparation_is_zero_safe(sparse, n_vars):
    matrix = np.zeros((2, n_vars), dtype=float)
    data = ad.AnnData(csr_matrix(matrix) if sparse else matrix.copy())
    if n_vars:
        data.var_names = ["HBA1", "HBB1", "g2"]
    before = data.X.copy()

    prepare_stored_qc(data)

    for metric in (
        "n_counts",
        "n_genes",
        "gene_detection_rate",
        "log10_counts",
        "log10_genes",
        "percent_mito",
        "percent_ribo",
        "percent_hb",
    ):
        np.testing.assert_array_equal(data.obs[metric], np.zeros(2))
        assert np.isfinite(data.obs[metric]).all()
    if sparse:
        assert (data.X != before).nnz == 0
    else:
        np.testing.assert_array_equal(data.X, before)


@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("numeric_columns", ["legacy_qc", "other_numeric"])
def test_qc_requires_assessed_metrics_before_drawing(harness, save, numeric_columns):
    """Numeric obs alone is not QC readiness; no drawing or scientific mutation."""
    harness.data.obs = harness.data.obs.drop(
        columns=[
            "n_genes",
            "gene_detection_rate",
            "log10_counts",
            "log10_genes",
            "percent_mito",
            "percent_ribo",
            "percent_hb",
        ]
    )
    if numeric_columns == "legacy_qc":
        harness.data.obs["n_genes_by_counts"] = 1000
        harness.data.obs["total_counts"] = 3000.0
        harness.data.obs["pct_counts_mt"] = 0.0
    else:
        harness.data.obs["other_numeric"] = np.arange(harness.data.n_obs)
    harness.data.raw = harness.data.copy()
    before = snapshot(harness.data)
    raw_before = harness.data.raw.X.copy()

    reply = invoke(harness, "create_qc_plots", save=save)

    assert "STATUS: FAILED" in reply
    assert "modality_name=cells" in reply
    assert "assess_data_quality" in reply
    assert "QC-assessed" in reply
    assert "unexpected_error" not in reply
    assert harness.events == []
    assert harness.stages == harness.activities == []
    assert harness.store.list().items == []
    assert_no_history(harness)
    assert_unchanged(harness.data, before)
    np.testing.assert_array_equal(harness.data.raw.X, raw_before)


@pytest.mark.parametrize(
    "missing_metric",
    [
        "n_genes",
        "n_counts",
        "gene_detection_rate",
        "log10_counts",
        "log10_genes",
        "percent_mito",
        "percent_ribo",
        "percent_hb",
    ],
)
def test_qc_requires_every_stored_assessment_metric(harness, missing_metric):
    harness.data.obs = harness.data.obs.drop(columns=[missing_metric])
    before = snapshot(harness.data)

    reply = invoke(harness, "create_qc_plots")

    assert "STATUS: FAILED" in reply
    assert "modality_name=cells" in reply
    assert "assess_data_quality" in reply
    assert harness.events == []
    assert harness.store.list().items == []
    assert_no_history(harness)
    assert_unchanged(harness.data, before)


@pytest.mark.parametrize("name", WRAPPERS)
def test_real_drawing_service_preserves_dataset(harness, name):
    from lobster.services.visualization.visualization_service import (
        SingleCellVisualizationService,
    )

    # Stored QC outputs are prepared by the fixture, not by a drawing tool or
    # an optional transcriptomics import. Preserve the original full dataset.
    harness.data.raw = harness.data.copy()
    before = snapshot(harness.data)
    raw_before = harness.data.raw.X.copy()
    real_service = SingleCellVisualizationService()
    setattr(harness.service, name, getattr(real_service, name))
    reply = invoke(harness, name)
    assert "STATUS: SUCCESS" in reply
    assert len(harness.store.list().items) == 1
    assert_unchanged(harness.data, before)
    np.testing.assert_array_equal(harness.data.raw.X, raw_before)
    assert_no_history(harness)


@pytest.mark.parametrize("name", WRAPPERS)
def test_success_uses_injected_id_and_committed_reference(harness, name):
    before = snapshot(harness.data)
    tool = harness.tools[name]
    assert "tool_call_id" not in tool.tool_call_schema.model_json_schema()["properties"]
    assert tool.metadata == {"categories": ["ANALYZE"], "provenance": True}
    reply = invoke(harness, name)
    assert harness.events == ["draw", "stage", "log", "commit"]
    assert "STATUS: SUCCESS" in reply
    records = harness.store.list().items
    assert len(records) == 1
    record = records[0]
    reference = f"{record['figure_id']}@{record['revision_id']}"
    assert f"figure_ref={reference}" in reply
    path = re.search(r"path=(\S+)", reply).group(1)
    assert (harness.store.workspace / path).is_file()
    execution = harness.store.read_execution(record["execution"]["execution_id"])
    assert execution["tool"] == name
    assert execution["activity_id"] == "activity-1"
    assert record["execution"]["attempt_key"].endswith(f":call-{name}:0")
    assert "plot_id" not in harness.stages[0][1]["dataset_info"]
    assert "plot_id" not in harness.activities[0]["parameters"]
    assert harness.activities[0]["parameters"]["tool_call_id"] == f"call-{name}"
    assert harness.activities[0]["ir"].replayable is False
    assert harness.activities[0]["ir"].output_entities == ["fig"]
    _, figure, _ = harness.store.read(reference)
    assert json.loads(figure)["layout"]["title"]["text"] == "Caller service title"
    assert not harness.manager._staged
    assert_no_history(harness)
    assert_unchanged(harness.data, before)


@pytest.mark.parametrize("name", WRAPPERS)
@pytest.mark.parametrize(
    "failure", ["draw", "stage_raise", "stage_return", "log", "commit"]
)
def test_failures_publish_nothing_and_create_no_phantom_history(harness, name, failure):
    before = snapshot(harness.data)
    harness.state["failure"] = failure
    reply = invoke(harness, name)
    assert "STATUS: FAILED" in reply
    assert "figure_ref=" not in reply
    assert harness.store.list().items == []
    assert not harness.manager._staged
    if failure in {"draw", "stage_raise", "stage_return"}:
        assert "log" not in harness.events
    assert_no_history(harness)
    assert_unchanged(harness.data, before)


@pytest.mark.parametrize("name", WRAPPERS)
def test_dry_run_discards_without_committing(harness, name):
    reply = invoke(harness, name, save=False)
    assert harness.events == ["draw", "stage", "log"]
    assert "STATUS: DRY_RUN" in reply and "not saved" in reply
    assert "Drawing discarded" in reply
    assert "no view delivered" in reply
    assert "PREVIEW" not in reply
    assert "figure_ref=" not in reply
    assert harness.store.list().items == []
    assert not harness.manager._staged
    assert_no_history(harness)


@pytest.mark.parametrize("name", WRAPPERS)
def test_identical_retry_reuses_reference(harness, name):
    first = invoke(harness, name, call_id="same-call")
    second = invoke(harness, name, call_id="same-call")
    assert re.search(r"figure_ref=(\S+)", first).group(1) == re.search(
        r"figure_ref=(\S+)", second
    ).group(1)
    assert len(harness.store.list().items) == 1
    assert_no_history(harness)


@pytest.mark.parametrize("name", WRAPPERS)
def test_empty_id_fails_without_revision(harness, name):
    reply = harness.tools[name].func(
        modality_name="cells", tool_call_id="", **WRAPPERS[name]
    )
    assert "STATUS: FAILED" in reply
    assert harness.store.list().items == []
    assert not harness.manager._staged
    assert_no_history(harness)


@pytest.mark.parametrize("name", WRAPPERS)
def test_model_cannot_spoof_injected_call_id(harness, name):
    reply = (
        harness.tools[name]
        .invoke(
            {
                "type": "tool_call",
                "name": name,
                "id": "actual-call",
                "args": {
                    "modality_name": "cells",
                    **WRAPPERS[name],
                    "tool_call_id": "forged-call",
                },
            }
        )
        .content
    )
    assert "STATUS: SUCCESS" in reply
    assert harness.activities[0]["parameters"]["tool_call_id"] == "actual-call"
    assert (
        harness.store.list()
        .items[0]["execution"]["attempt_key"]
        .endswith(":actual-call:0")
    )


@pytest.mark.parametrize("name", WRAPPERS)
@pytest.mark.parametrize("save", [True, False])
def test_completion_affects_only_own_selected_staging_token(harness, name, save):
    sibling_id, _, _ = harness.manager.add_plot(
        plot=go.Figure(go.Scatter(x=[0], y=[1], uid="sibling")),
        title="Unrelated pending figure",
        source="test_fixture",
    )
    reply = invoke(harness, name, save=save)
    assert f"STATUS: {'SUCCESS' if save else 'DRY_RUN'}" in reply
    assert set(harness.manager._staged) == {sibling_id}
    assert len(harness.store.list().items) == int(save)
    assert sibling_id not in reply
    assert_no_history(harness)


@pytest.mark.parametrize("name", WRAPPERS)
def test_dry_run_does_not_remove_existing_revision(harness, name):
    first = invoke(harness, name)
    reference = re.search(r"figure_ref=(\S+)", first).group(1)
    before = harness.store.read(reference)
    dry_run = invoke(harness, name, call_id="dry-run-call", save=False)
    assert "not saved" in dry_run
    assert "no view delivered" in dry_run
    assert harness.store.read(reference) == before
    assert len(harness.store.list().items) == 1
    assert_no_history(harness)


@pytest.mark.parametrize("name", WRAPPERS)
@pytest.mark.parametrize("point", ["draw", "log", "commit"])
def test_failure_replies_and_logs_never_echo_exception_values(
    harness, name, point, caplog
):
    sentinel = "synthetic-sensitive-marker-not-a-real-secret"

    def fail(*args, **kwargs):
        raise RuntimeError(sentinel)

    if point == "draw":
        setattr(harness.service, name, fail)
    elif point == "log":
        # The actual function closure holds the mutable data-manager object.
        harness.dm.log_tool_usage = fail
    else:
        harness.manager.commit_staged = fail
    reply = invoke(harness, name)
    assert "STATUS: FAILED" in reply and "unexpected_error" in reply
    assert sentinel not in reply and sentinel not in caplog.text
    assert harness.store.list().items == []
    assert not harness.manager._staged


@pytest.mark.parametrize("name", WRAPPERS)
@pytest.mark.parametrize(
    "field", ["instance_code", "subclass_code", "detail_reason", "result_reason"]
)
def test_figure_error_allowed_charset_canaries_never_echo(harness, name, field, caplog):
    sentinel = "synthetic_sensitive_marker_123"
    error = FigureError("synthetic message", details={})
    if field == "instance_code":
        error.code = sentinel
    elif field == "subclass_code":

        class UntrustedFigureError(FigureError):
            code = sentinel

        error = UntrustedFigureError("synthetic message")
    elif field == "detail_reason":
        error.details["reason"] = sentinel
    else:
        error.details["results"] = [{"status": "failed", "reason": sentinel}]

    def fail(*args, **kwargs):
        raise error

    harness.manager.commit_staged = fail
    reply = invoke(harness, name)
    assert "STATUS: FAILED" in reply
    assert sentinel not in reply and sentinel not in caplog.text
    assert harness.store.list().items == []
    assert not harness.manager._staged
    assert_no_history(harness)


@pytest.mark.parametrize("name", WRAPPERS)
@pytest.mark.parametrize(
    "results",
    [
        None,
        123,
        "synthetic_sensitive_marker_123",
        {"reason": "synthetic_sensitive_marker_123"},
        [None],
        [123],
        ["synthetic_sensitive_marker_123"],
        [
            {
                "status": ["synthetic_sensitive_marker_123"],
                "reason": "synthetic_sensitive_marker_123",
            }
        ],
    ],
)
def test_malformed_figure_error_results_never_raise_or_echo(
    harness, name, results, caplog
):
    sentinel = "synthetic_sensitive_marker_123"

    def fail(*args, **kwargs):
        raise FigureError("synthetic message", details={"results": results})

    harness.manager.commit_staged = fail
    reply = invoke(harness, name)
    assert "STATUS: FAILED" in reply
    assert sentinel not in reply and sentinel not in caplog.text
    assert harness.store.list().items == []
    assert not harness.manager._staged
    assert_no_history(harness)


@pytest.mark.parametrize("name", WRAPPERS)
@pytest.mark.parametrize("mixed", [False, True])
def test_pending_publication_details_preserve_truthful_candidate_status(
    harness, name, mixed, caplog
):
    sentinel = "synthetic_sensitive_marker_123"
    results = [{"status": "pending", "reason": sentinel}]
    if mixed:
        results.append({"status": "failed", "reason": sentinel})

    def pending(*args, **kwargs):
        raise FigureError("synthetic pending message", details={"results": results})

    harness.manager.commit_staged = pending
    reply = invoke(harness, name)
    assert f"STATUS: {'PARTIAL' if mixed else 'PENDING'}" in reply
    assert "pending:" in reply
    if mixed:
        assert "failed:" in reply
    assert HEADER_PENDING in reply
    assert "publication pending" in caplog.text
    assert f"Figure producer {name} failed:" not in caplog.text
    assert "No figure revision created" not in reply
    assert "No available figure revision was returned" not in reply
    assert "figure_ref=" not in reply
    assert sentinel not in reply and sentinel not in caplog.text
    assert harness.store.list().items == []
    assert not harness.manager._staged
    assert_no_history(harness)


@pytest.mark.parametrize("name", WRAPPERS)
def test_actual_plot_manager_returns_pending_from_store_without_reader_leak(
    harness, name, monkeypatch
):
    pending_calls = []

    def store_pending(candidate, **kwargs):
        pending_calls.append((candidate.candidate_index, kwargs))
        return CommitResult(
            candidate.candidate_index, STATUS_PENDING, label=candidate.label
        )

    # Only the store's final commit is replaced. The actual PlotManager,
    # TrustedSaver, execution-record write, activity and wrapper run normally.
    monkeypatch.setattr(harness.store, "commit", store_pending)
    reply = invoke(harness, name)
    assert harness.events == ["draw", "stage", "log", "commit"]
    assert len(pending_calls) == 1
    assert "STATUS: PENDING" in reply and HEADER_PENDING in reply
    assert "figure_ref=" not in reply
    assert "No figure saved" not in reply and "No figure revision created" not in reply
    assert harness.store.list().items == []
    assert not harness.manager._staged
    latest, _, _ = harness.manager.get_latest_plots()
    history, _, _ = harness.manager.get_plot_history()
    figure, stats, _ = harness.manager.get_plot_by_id(harness.stages[0][0])
    assert latest == [] and history == []
    assert figure is None and stats["found"] is False
    assert_no_history(harness)


def test_provenance_disabled_none_is_supported_by_core_contract(harness):
    harness.state["provenance"] = False
    reply = invoke(harness, "create_umap_plot")
    assert "STATUS: SUCCESS" in reply
    revision = harness.store.list().items[0]
    execution = harness.store.read_execution(revision["execution"]["execution_id"])
    assert execution["activity_id"] is None
    assert execution["activity_unavailable"] == "provenance_disabled"


def test_legacy_history_is_not_available_catalog(harness):
    invoke(harness, "create_umap_plot")
    reply = harness.tools["get_visualization_history"].invoke({})
    assert "LEGACY" in reply and "does not mean no managed figures" in reply
    harness.manager.visualization_state["history"].append(
        {"plot_id": "old-id", "metadata": {"type": "umap", "modality": "cells"}}
    )
    reply = harness.tools["get_visualization_history"].invoke({})
    assert "Legacy plot ID: old-id" in reply and "not managed revisions" in reply
    assert "Do not create UUIDs or write visualization history" in harness.prompt
    assert "canonical" in harness.prompt


@pytest.mark.parametrize("name", WRAPPERS)
@pytest.mark.parametrize("outcome", ["log", "commit", "preview"])
def test_failed_preview_candidates_are_not_exposed_by_any_reader(
    harness, name, outcome
):
    if outcome != "preview":
        harness.state["failure"] = outcome
    invoke(harness, name, save=outcome != "preview")
    staged_id = harness.stages[0][0]
    latest, _, _ = harness.manager.get_latest_plots()
    history, _, _ = harness.manager.get_plot_history()
    figure, stats, _ = harness.manager.get_plot_by_id(staged_id)
    assert latest == []
    assert history == []
    assert figure is None and stats["found"] is False
