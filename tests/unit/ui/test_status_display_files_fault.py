"""The recent-files panel shows a listing fault, never "No files found".

Real DataManagerV2 on tmp_path; the managed-figure inventory fault is the typed one
``list_workspace_files`` raises.
"""

from __future__ import annotations

from types import SimpleNamespace

import plotly.graph_objects as go
from rich.console import Console

from lobster.core.runtime.data_manager import DataManagerV2
from lobster.ui.components.status_display import EnhancedStatusDisplay


def _render(panel) -> str:
    console = Console(width=120, record=True, color_system=None)
    console.print(panel)
    return console.export_text()


def _client(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path)
    fig = go.Figure(go.Bar(x=[1], y=[2]))
    plot_id, _, _ = dm.plot_manager.add_plot(plot=fig, title="Bars", source="t")
    dm.plot_manager.commit_staged({"id": "a1"}, tool_call_id="c1", staged_ids=[plot_id])
    return SimpleNamespace(data_manager=dm)


def test_an_inventory_fault_renders_unable_to_list_with_a_safe_reason(
    tmp_path, monkeypatch
):
    from lobster.core.figures.errors import StorageError

    client = _client(tmp_path)

    def boom(*_a, **_k):
        raise StorageError(f"CANARY {tmp_path}/secret")

    monkeypatch.setattr(client.data_manager.figure_store, "list", boom)
    text = _render(EnhancedStatusDisplay()._create_recent_files_panel(client))
    assert "No files found" not in text
    assert "Unable to list files (storage_failed)" in text
    assert "CANARY" not in text and "secret" not in text


def test_an_untyped_fault_is_also_unable_to_list(tmp_path, monkeypatch):
    client = _client(tmp_path)
    monkeypatch.setattr(
        client.data_manager,
        "list_workspace_files",
        lambda: (_ for _ in ()).throw(RuntimeError("CANARY detail")),
    )
    text = _render(EnhancedStatusDisplay()._create_recent_files_panel(client))
    assert "No files found" not in text
    assert "Unable to list files (unexpected_error)" in text
    assert "CANARY" not in text


def test_a_healthy_workspace_still_lists_its_files(tmp_path):
    text = _render(
        EnhancedStatusDisplay()._create_recent_files_panel(_client(tmp_path))
    )
    assert "Unable to list files" not in text and "No files found" not in text
    assert "figure.json" in text or "f_" in text
