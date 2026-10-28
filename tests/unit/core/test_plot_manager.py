"""
Unit tests for PlotManager.

Tests cover:
- Plot addition with metadata (3-tuple return pattern)
- FIFO buffer behavior (max_plots_history enforcement)
- Thread safety (counter lock)
- Managed figure lifecycle: add_plot stages, commit_staged publishes (trusted save),
  save_plots_to_workspace writes nothing
- Visualization state management
- Provenance IR generation
- Backward compatibility (property access)
- Error handling (invalid plots, missing workspace)
- Fresh plot managers on one workspace never overwrite an earlier saved figure

All tests except the fresh-worker tests at the end run WITHOUT instantiating DataManagerV2
(isolated unit tests); those use one real DataManagerV2 per fresh worker.
"""

import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lobster.core.plot_manager import PlotManager
from lobster.core.provenance.analysis_ir import AnalysisStep

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def workspace_path(tmp_path):
    """Create temporary workspace directory."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def plot_manager(workspace_path):
    """Create PlotManager instance."""
    return PlotManager(
        workspace_path=workspace_path,
        max_plots_history=5,
        min_save_interval=0.1,  # Short interval for testing
        enable_ir=True,
    )


@pytest.fixture
def mock_plotly_figure():
    """Real Plotly Figure (staging serializes byte-exact JSON, so mocks no longer suffice)."""
    import plotly.graph_objects as go
    import plotly.io as pio

    figure = go.Figure(go.Scatter(x=[1, 2, 3], y=[3, 1, 2], uid="trace-a"))
    figure.update_layout(title="Service title")
    yield figure, go, pio


_CALLS = iter(range(10**6))


def _publish(pm, figure, **kw):
    """add_plot + commit_staged: the only way a figure becomes visible to legacy readers."""
    plot_id, stats, ir = pm.add_plot(plot=figure, **kw)
    pm.commit_staged(
        {"id": "act"}, tool_call_id=f"call_{next(_CALLS)}", staged_ids=[plot_id]
    )
    return plot_id, stats, ir


@pytest.fixture
def sample_modalities():
    """Create sample modalities dict for context."""
    mock_adata = Mock()
    mock_adata.shape = (1000, 2000)
    mock_adata.n_obs = 1000
    mock_adata.n_vars = 2000
    return {"test_modality": mock_adata}


# =============================================================================
# Test: Plot Addition (Core Functionality)
# =============================================================================


def test_add_plot_basic(plot_manager, mock_plotly_figure):
    """Test basic plot addition with 3-tuple return."""
    figure, mock_go, _ = mock_plotly_figure

    plot_id, stats, ir = plot_manager.add_plot(
        plot=figure,
        title="Test Plot",
        source="test_service",
    )

    # Verify 3-tuple return
    assert isinstance(plot_id, str)
    assert plot_id.startswith("plot_")
    assert isinstance(stats, dict)
    assert isinstance(ir, AnalysisStep)

    # Verify stats content
    assert stats["plot_id"] == plot_id
    assert stats["title"].startswith("Test Plot")
    assert stats["source"] == "test_service"
    assert stats["status"] == "staged"
    assert stats["plots_in_history"] == 0  # staged: not visible until commit_staged

    # Verify IR generation
    assert ir.operation == "plot_manager.add_plot"
    assert ir.library == "plotly"


def test_add_plot_with_modality_context(
    plot_manager, mock_plotly_figure, sample_modalities
):
    """Test plot addition with modality context."""
    figure, _, _ = mock_plotly_figure

    plot_id, stats, ir = plot_manager.add_plot(
        plot=figure,
        title="Cluster Plot",
        modalities=sample_modalities,
        current_dataset="test_modality",
    )

    assert plot_id
    assert "test_modality" in stats["title"]  # Modality name added to title


def test_add_plot_invalid_figure(plot_manager):
    """Test error handling for invalid figure."""
    with patch("lobster.core.plot_manager._ensure_plotly") as mock_ensure:
        mock_go = Mock()
        mock_ensure.return_value = (mock_go, None)
        mock_go.Figure = type(Mock())

        with pytest.raises(ValueError, match="Plot must be a plotly Figure"):
            plot_manager.add_plot(plot="not a figure")


def test_add_plot_increments_counter(plot_manager, mock_plotly_figure):
    """Test that plot_counter increments correctly."""
    figure, _, _ = mock_plotly_figure

    plot_id_1, _, _ = plot_manager.add_plot(plot=figure, title="Plot 1")
    plot_id_2, _, _ = plot_manager.add_plot(plot=figure, title="Plot 2")
    plot_id_3, _, _ = plot_manager.add_plot(plot=figure, title="Plot 3")

    assert plot_id_1 == "plot_1"
    assert plot_id_2 == "plot_2"
    assert plot_id_3 == "plot_3"
    assert plot_manager.plot_counter == 3


# =============================================================================
# Test: FIFO Buffer Behavior
# =============================================================================


def test_fifo_buffer_enforcement(workspace_path, mock_plotly_figure):
    """Test that max_plots_history is enforced (FIFO)."""
    figure, _, _ = mock_plotly_figure
    pm = PlotManager(workspace_path=workspace_path, max_plots_history=3)

    # Add 5 plots (exceeds max of 3)
    for i in range(5):
        _publish(pm, figure, title=f"Plot {i + 1}")

    # Should only keep last 3 plots
    assert len(pm.latest_plots) == 3
    plot_ids = [p["id"] for p in pm.latest_plots]
    assert plot_ids == ["plot_3", "plot_4", "plot_5"]


def test_fifo_buffer_preserves_latest(workspace_path, mock_plotly_figure):
    """Test that FIFO buffer preserves most recent plots."""
    figure, _, _ = mock_plotly_figure
    pm = PlotManager(workspace_path=workspace_path, max_plots_history=2)

    _publish(pm, figure, title="Old Plot")
    time.sleep(0.01)
    _publish(pm, figure, title="Middle Plot")
    time.sleep(0.01)
    _publish(pm, figure, title="New Plot")

    assert len(pm.latest_plots) == 2
    titles = [p["title"] for p in pm.latest_plots]
    assert any("Middle Plot" in t for t in titles)
    assert any("New Plot" in t for t in titles)


# =============================================================================
# Test: Thread Safety
# =============================================================================


def test_counter_thread_safety(workspace_path, mock_plotly_figure):
    """Test that plot_counter is thread-safe."""
    figure, _, _ = mock_plotly_figure
    pm = PlotManager(workspace_path=workspace_path, max_plots_history=100)

    plot_ids = []
    errors = []

    def add_plots(n=10):
        try:
            for _ in range(n):
                plot_id, _, _ = pm.add_plot(plot=figure, title="Concurrent Plot")
                plot_ids.append(plot_id)
        except Exception as e:
            errors.append(e)

    # Spawn 5 threads, each adding 10 plots
    threads = [threading.Thread(target=add_plots) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # No errors should occur
    assert len(errors) == 0

    # All 50 plots should be added with unique IDs
    assert len(plot_ids) == 50
    assert len(set(plot_ids)) == 50  # All unique
    assert pm.plot_counter == 50


# =============================================================================
# Test: Rate Limiting
# =============================================================================


# =============================================================================
# Test: Workspace Export
# =============================================================================


# =============================================================================
# Test: PNG Skip Threshold
# =============================================================================


# =============================================================================
# Test: Plot Retrieval
# =============================================================================


def test_get_plot_by_id_found(plot_manager, mock_plotly_figure):
    """Test retrieving plot by ID."""
    figure, _, _ = mock_plotly_figure
    plot_id, _, _ = _publish(plot_manager, figure, title="Find Me")

    retrieved_figure, stats, none_ir = plot_manager.get_plot_by_id(plot_id)

    assert retrieved_figure == figure
    assert stats["found"] is True
    assert stats["plot_id"] == plot_id
    assert none_ir is None


def test_get_plot_by_id_not_found(plot_manager):
    """Test retrieving non-existent plot."""
    retrieved_figure, stats, _ = plot_manager.get_plot_by_id("plot_999")

    assert retrieved_figure is None
    assert stats["found"] is False
    assert stats["plot_id"] == "plot_999"


def test_get_latest_plots_all(plot_manager, mock_plotly_figure):
    """Test getting all plots."""
    figure, _, _ = mock_plotly_figure
    for i in range(3):
        _publish(plot_manager, figure, title=f"Plot {i + 1}")

    plots, stats, _ = plot_manager.get_latest_plots(n=None)

    assert len(plots) == 3
    assert stats["total_plots"] == 3
    assert stats["returned_plots"] == 3


def test_get_latest_plots_limited(plot_manager, mock_plotly_figure):
    """Test getting limited number of plots."""
    figure, _, _ = mock_plotly_figure
    for i in range(5):
        _publish(plot_manager, figure, title=f"Plot {i + 1}")

    plots, stats, _ = plot_manager.get_latest_plots(n=2)

    assert len(plots) == 2
    assert stats["total_plots"] == 5
    assert stats["returned_plots"] == 2
    # Should return last 2 plots
    assert plots[0]["id"] == "plot_4"
    assert plots[1]["id"] == "plot_5"


def test_get_plot_history(plot_manager, mock_plotly_figure):
    """Test getting plot history (minimal metadata)."""
    figure, _, _ = mock_plotly_figure
    _publish(plot_manager, figure, title="Plot 1", source="service_a")
    _publish(plot_manager, figure, title="Plot 2", source="service_b")

    history, stats, _ = plot_manager.get_plot_history()

    assert len(history) == 2
    assert stats["total_plots"] == 2

    # History should NOT include figure objects
    for entry in history:
        assert "figure" not in entry
        assert "id" in entry
        assert "title" in entry
        assert "timestamp" in entry
        assert "source" in entry


# =============================================================================
# Test: Clear Operations
# =============================================================================


def test_clear_plots(plot_manager, mock_plotly_figure):
    """Test clearing all plots."""
    figure, _, _ = mock_plotly_figure
    _publish(plot_manager, figure, title="Plot 1")
    _publish(plot_manager, figure, title="Plot 2")

    assert len(plot_manager.latest_plots) == 2

    _, stats, ir = plot_manager.clear_plots()

    assert len(plot_manager.latest_plots) == 0
    assert stats["cleared_count"] == 2
    assert isinstance(ir, AnalysisStep)


def test_clear_plots_empty(plot_manager):
    """Test clearing when no plots exist."""
    _, stats, ir = plot_manager.clear_plots()

    assert stats["cleared_count"] == 0
    assert isinstance(ir, AnalysisStep)


# =============================================================================
# Test: Visualization State Management
# =============================================================================


def test_add_visualization_record(plot_manager):
    """Test adding visualization record."""
    metadata = {"type": "scatter", "params": {"color": "red"}}

    _, stats, _ = plot_manager.add_visualization_record("plot_123", metadata)

    assert stats["plot_id"] == "plot_123"
    assert stats["history_size"] == 1
    assert stats["registry_size"] == 1
    assert "plot_123" in plot_manager.visualization_state["plot_registry"]


def test_get_visualization_history(plot_manager):
    """Test retrieving visualization history."""
    plot_manager.add_visualization_record("plot_1", {"type": "scatter"})
    plot_manager.add_visualization_record("plot_2", {"type": "heatmap"})
    plot_manager.add_visualization_record("plot_3", {"type": "volcano"})

    history, stats, _ = plot_manager.get_visualization_history(limit=2)

    assert len(history) == 2
    assert stats["total_history"] == 3
    assert stats["returned"] == 2
    # Should return last 2
    assert history[0]["plot_id"] == "plot_2"
    assert history[1]["plot_id"] == "plot_3"


def test_get_visualization_settings(plot_manager):
    """Test getting visualization settings."""
    settings, stats, _ = plot_manager.get_visualization_settings()

    assert isinstance(settings, dict)
    assert "default_width" in settings
    assert settings["default_width"] == 800
    assert stats["settings_count"] == len(settings)


def test_update_visualization_settings(plot_manager):
    """Test updating visualization settings."""
    new_settings = {"default_width": 1200, "color_scheme": "viridis"}

    _, stats, _ = plot_manager.update_visualization_settings(new_settings)

    assert stats["updated_keys"] == ["default_width", "color_scheme"]
    assert plot_manager.visualization_state["settings"]["default_width"] == 1200
    assert plot_manager.visualization_state["settings"]["color_scheme"] == "viridis"


def test_get_plot_by_uuid(plot_manager):
    """Test retrieving plot by UUID."""
    metadata = {"type": "bar", "data": "test"}
    plot_manager.add_visualization_record("uuid_123", metadata)

    retrieved_metadata, stats, _ = plot_manager.get_plot_by_uuid("uuid_123")

    assert retrieved_metadata == metadata
    assert stats["found"] is True
    assert stats["plot_id"] == "uuid_123"


def test_get_plot_by_uuid_not_found(plot_manager):
    """Test retrieving non-existent UUID."""
    retrieved_metadata, stats, _ = plot_manager.get_plot_by_uuid("nonexistent")

    assert retrieved_metadata is None
    assert stats["found"] is False


def test_clear_visualization_history(plot_manager):
    """Test clearing visualization history."""
    plot_manager.add_visualization_record("plot_1", {"type": "scatter"})
    plot_manager.add_visualization_record("plot_2", {"type": "bar"})

    _, stats, _ = plot_manager.clear_visualization_history()

    assert stats["cleared_history"] == 2
    assert stats["cleared_registry"] == 2
    assert len(plot_manager.visualization_state["history"]) == 0
    assert len(plot_manager.visualization_state["plot_registry"]) == 0


# =============================================================================
# Test: Provenance IR Generation
# =============================================================================


def test_ir_generation_enabled(workspace_path, mock_plotly_figure):
    """Test that IR is generated when enable_ir=True."""
    figure, _, _ = mock_plotly_figure
    pm = PlotManager(workspace_path=workspace_path, enable_ir=True)

    _, _, ir = pm.add_plot(plot=figure, title="Test")

    assert isinstance(ir, AnalysisStep)
    assert ir.operation == "plot_manager.add_plot"
    assert "lobster.core.plot_manager" in ir.imports[0]


def test_ir_generation_disabled(workspace_path, mock_plotly_figure):
    """Test that IR is NOT generated when enable_ir=False."""
    figure, _, _ = mock_plotly_figure
    pm = PlotManager(workspace_path=workspace_path, enable_ir=False)

    _, _, ir = pm.add_plot(plot=figure, title="Test")

    assert ir is None


def test_ir_for_clear_operation(plot_manager, mock_plotly_figure):
    """Test IR generation for clear_plots."""
    figure, _, _ = mock_plotly_figure
    plot_manager.add_plot(plot=figure, title="Test")

    _, _, ir = plot_manager.clear_plots()

    assert isinstance(ir, AnalysisStep)
    assert ir.operation == "plot_manager.clear_plots"


# =============================================================================
# Test: Configuration
# =============================================================================


def test_custom_max_plots_history(workspace_path, mock_plotly_figure):
    """Test custom max_plots_history configuration."""
    figure, _, _ = mock_plotly_figure
    pm = PlotManager(workspace_path=workspace_path, max_plots_history=10)

    for i in range(15):
        _publish(pm, figure, title=f"Plot {i + 1}")

    assert len(pm.latest_plots) == 10


# =============================================================================
# Test: Error Handling
# =============================================================================


def test_add_plot_exception_handling(plot_manager):
    """Test error handling in add_plot for invalid figure."""
    # Pass invalid plot (not a Figure) - should raise ValueError
    with patch("lobster.core.plot_manager._ensure_plotly") as mock_ensure:
        mock_go = Mock()
        mock_ensure.return_value = (mock_go, None)
        mock_go.Figure = type(Mock())

        with pytest.raises(ValueError, match="Plot must be a plotly Figure"):
            plot_manager.add_plot(plot=None, title="Invalid")


# =============================================================================
# Managed figure lifecycle (contract §6.5)
# =============================================================================


def test_add_plot_stages_and_does_not_mutate_title(
    plot_manager, mock_plotly_figure, workspace_path
):
    figure, _, _ = mock_plotly_figure
    plot_id, stats, _ = plot_manager.add_plot(
        plot=figure, title="UMAP - leiden", source="visualization_expert"
    )
    assert stats["status"] == "staged"
    assert figure.layout.title.text == "Service title"  # no enhanced-title mutation
    assert not (workspace_path / "plots").exists()  # nothing written before commit


def test_commit_staged_publishes_after_activity(
    plot_manager, mock_plotly_figure, workspace_path
):
    from lobster.core.figures.store import LocalFigureStore

    figure, _, _ = mock_plotly_figure
    plot_id, _, _ = plot_manager.add_plot(
        plot=figure, title="UMAP", source="visualization_expert"
    )
    results = plot_manager.commit_staged(
        {"id": "act-7", "tool_name": "create_umap_plot"},
        tool_call_id="call_1",
        staged_ids=[plot_id],
        tool_name="create_umap_plot",
    )
    assert [r.status for r in results] == ["available"]
    store = LocalFigureStore(workspace_path)
    record = store.read_record(results[0].ref)
    assert record["change"]["actor"] == "legacy_wrapper:visualization_expert"
    assert (
        store.read_execution(record["execution"]["execution_id"])["activity_id"]
        == "act-7"
    )
    assert plot_manager.latest_plots[-1]["figure_ref"] == results[0].ref
    paths, stats, ir = plot_manager.save_plots_to_workspace()
    assert paths == [results[0].path] and ir is None
    assert (
        plot_manager.save_plots_to_workspace()[0] == []
    )  # writes nothing, returns only new commits


def test_commit_staged_requires_tool_call_id(plot_manager, mock_plotly_figure):
    from lobster.core.figures.errors import AttemptContextError

    figure, _, _ = mock_plotly_figure
    plot_id, _, _ = plot_manager.add_plot(plot=figure, title="X")
    with pytest.raises(AttemptContextError):
        plot_manager.commit_staged({"id": "a"}, tool_call_id=None, staged_ids=[plot_id])


def test_uncommitted_staged_figure_never_becomes_visible(
    plot_manager, mock_plotly_figure, workspace_path
):
    figure, _, _ = mock_plotly_figure
    plot_id, _, _ = plot_manager.add_plot(
        plot=figure, title="preview only"
    )  # save_plot=False path
    assert plot_manager.discard_staged([plot_id]) == 1
    assert plot_manager.save_plots_to_workspace()[0] == []
    assert not (workspace_path / "plots").exists()


def test_commit_staged_retry_is_idempotent(plot_manager, mock_plotly_figure):
    figure, _, _ = mock_plotly_figure
    first_id, _, _ = plot_manager.add_plot(plot=figure, title="A")
    [r1] = plot_manager.commit_staged(
        {"id": "a"}, tool_call_id="call_9", staged_ids=[first_id]
    )
    again_id, _, _ = plot_manager.add_plot(plot=figure, title="A")
    [r2] = plot_manager.commit_staged(
        {"id": "a"}, tool_call_id="call_9", staged_ids=[again_id]
    )
    assert r2.ref == r1.ref


# --- review finding: no public legacy reader sees a staged/failed/preview figure ---------


def _public_view(pm, plot_id):
    """Every existing public PlotManager reader (and the DataManagerV2/client wrappers over them)."""
    return {
        "latest_plots": [p["id"] for p in pm.latest_plots],
        "by_id": pm.get_plot_by_id(plot_id)[0] is not None,
        "latest": [p["id"] for p in pm.get_latest_plots()[0]],
        "history": [p["id"] for p in pm.get_plot_history()[0]],
    }


_HIDDEN = {"latest_plots": [], "by_id": False, "latest": [], "history": []}


def test_staged_preview_is_invisible_to_every_reader(plot_manager, mock_plotly_figure):
    figure, _, _ = mock_plotly_figure
    plot_id, _, _ = plot_manager.add_plot(plot=figure, title="save_plot=False preview")
    assert _public_view(plot_manager, plot_id) == _HIDDEN
    plot_manager.discard_staged([plot_id])
    assert _public_view(plot_manager, plot_id) == _HIDDEN


def test_failed_commit_stays_invisible_to_every_reader(
    plot_manager, mock_plotly_figure, monkeypatch
):
    from lobster.core.figures.errors import FigureError
    from lobster.core.figures.store import LocalFigureStore

    def boom(*_a, **_k):
        raise OSError("CANARY disk detail")

    monkeypatch.setattr(LocalFigureStore, "commit", boom)
    figure, _, _ = mock_plotly_figure
    plot_id, _, _ = plot_manager.add_plot(plot=figure, title="will fail")
    with pytest.raises(FigureError) as exc:
        plot_manager.commit_staged(
            {"id": "a"}, tool_call_id="call_f", staged_ids=[plot_id]
        )
    assert "CANARY" not in str(exc.value)
    assert _public_view(plot_manager, plot_id) == _HIDDEN
    assert plot_manager.save_plots_to_workspace()[0] == []


def test_successful_commit_is_visible_with_committed_bytes(
    plot_manager, mock_plotly_figure
):
    import json as _json

    from lobster.core.figures.store import LocalFigureStore

    figure, _, _ = mock_plotly_figure
    plot_id, _, _ = plot_manager.add_plot(plot=figure, title="kept")
    figure.update_layout(
        title="mutated after staging"
    )  # caller mutation is not the truth
    [res] = plot_manager.commit_staged(
        {"id": "a"}, tool_call_id="call_ok", staged_ids=[plot_id]
    )
    view = _public_view(plot_manager, plot_id)
    assert view == {
        "latest_plots": [plot_id],
        "by_id": True,
        "latest": [plot_id],
        "history": [plot_id],
    }
    entry = plot_manager.latest_plots[-1]
    assert entry["figure_ref"] == res.ref
    _, fig_bytes, _ = LocalFigureStore(plot_manager.workspace_path).read(res.ref)
    assert entry["figure"].layout.title.text == "Service title"
    assert _json.loads(fig_bytes)["layout"]["title"]["text"] == "Service title"


# --- fresh plot managers on one workspace never overwrite an earlier saved figure -----
# A fresh worker starts its in-memory plot counter again, so every worker stages "plot_1".
# That id is staging-only: each saved figure gets its own random figure/revision ids with
# exclusive create, and no flat ``plot_1*.html`` is written.

_FRESH_WORKER = r"""
import json, sys
import plotly.graph_objects as go
from lobster.core.runtime.data_manager import DataManagerV2

workspace, call_id, y = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
pm = DataManagerV2(workspace_path=workspace).plot_manager
figure = go.Figure(go.Scatter(x=[0, 1, 2], y=y, mode="markers"), layout={"title": "UMAP"})
plot_id, _, _ = pm.add_plot(figure, title="UMAP", source="fresh_worker")
assert plot_id == "plot_1", plot_id  # precondition, before the save
[res] = pm.commit_staged(None, tool_call_id=call_id, staged_ids=[plot_id])
print("RESULT " + json.dumps(
    {"staged_id": plot_id, "status": res.status, "ref": res.ref, "path": res.path}
))
"""


def _stage_and_commit_umap(data_manager, call_id, y):
    import plotly.graph_objects as go

    figure = go.Figure(
        go.Scatter(x=[0, 1, 2], y=y, mode="markers"), layout={"title": "UMAP"}
    )
    pm = data_manager.plot_manager
    plot_id, _, _ = pm.add_plot(figure, title="UMAP", source="fresh_worker")
    assert plot_id == "plot_1"  # precondition, before the save
    [res] = pm.commit_staged(None, tool_call_id=call_id, staged_ids=[plot_id])
    return {
        "staged_id": plot_id,
        "status": res.status,
        "ref": res.ref,
        "path": res.path,
    }


def _fresh_worker_process(workspace, call_id, y):
    """Stage and commit one UMAP figure in its own fresh Python process."""
    import json
    import subprocess
    import sys

    from lobster.utils.subprocess_env import minimal_child_env

    proc = subprocess.run(
        [sys.executable, "-c", _FRESH_WORKER, str(workspace), call_id, json.dumps(y)],
        env=minimal_child_env(),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert len(lines) == 1
    return json.loads(lines[0][len("RESULT ") :])


def _plots_tree(workspace):
    """{relative path: sha256} of every regular file under ``plots/``."""
    import hashlib

    root = Path(workspace) / "plots"
    return {
        str(p.relative_to(workspace)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.is_symlink()
    }


def _check_two_workers(workspace, commit):
    from lobster.core.figures.ids import FigureRef
    from lobster.core.figures.store import LocalFigureStore

    first = commit("call_worker_a", [1, 2, 3])
    assert first["status"] == "available"
    first_tree = _plots_tree(workspace)
    first_read = LocalFigureStore(workspace).read(first["ref"])

    second = commit("call_worker_b", [3, 2, 1])
    assert second["status"] == "available"

    # both fresh workers staged the same staging id (each also checked before its save)
    assert first["staged_id"] == second["staged_id"] == "plot_1"
    # two distinct saved figures, not one figure overwritten or revised
    assert first["ref"] != second["ref"]
    a, b = FigureRef.parse(first["ref"]), FigureRef.parse(second["ref"])
    assert a.figure_id != b.figure_id
    assert first["path"] != second["path"]
    # every file the first save wrote is byte-identical after the second save
    after = _plots_tree(workspace)
    assert {path: after.get(path) for path in first_tree} == first_tree
    assert set(after) > set(first_tree)
    # both records and payloads read back, verified, from a fresh store
    store = LocalFigureStore(workspace)
    assert store.read(first["ref"]) == first_read
    store.read_record(first["ref"])
    store.read_record(second["ref"])
    assert store.read(second["ref"])[1] != first_read[1]
    # no flat legacy HTML named after the staging id
    assert not list(Path(workspace).rglob("plot_1*.html"))


def test_fresh_plot_managers_never_overwrite_an_earlier_saved_figure(tmp_path):
    from lobster.core.runtime.data_manager import DataManagerV2

    def commit(call_id, y):  # a fresh data manager (and plot manager) per save
        return _stage_and_commit_umap(
            DataManagerV2(workspace_path=tmp_path), call_id, y
        )

    _check_two_workers(tmp_path, commit)


def test_fresh_worker_processes_never_overwrite_an_earlier_saved_figure(tmp_path):
    _check_two_workers(
        tmp_path, lambda call_id, y: _fresh_worker_process(tmp_path, call_id, y)
    )
