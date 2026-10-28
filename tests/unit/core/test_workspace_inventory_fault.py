"""The workspace inventory is complete or an explicit typed fault, never silently partial.

``list_workspace_files`` feeds /files, the status panel and the artifact manifest. A
managed-figure catalog fault after some records were read must not come back as an
ordinary (smaller) inventory. Real store on tmp_path.
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
import pytest

from lobster.agents.artifact_manifest import take_snapshot
from lobster.core.figures.errors import FigureError, StorageError, safe_reason
from lobster.core.runtime.data_manager import DataManagerV2


@pytest.fixture()
def dm(tmp_path):
    return DataManagerV2(workspace_path=tmp_path)


def _commit(dm, title, call):
    fig = go.Figure(go.Bar(x=[1], y=[2]))
    plot_id, _, _ = dm.plot_manager.add_plot(plot=fig, title=title, source="t")
    [res] = dm.plot_manager.commit_staged(
        {"id": f"a-{call}"}, tool_call_id=call, staged_ids=[plot_id]
    )
    return res.ref


def _two_pages_second_fails(dm, monkeypatch, error):
    """Page 1 = one real record with a cursor; page 2 = the fault."""
    store = dm.figure_store
    real = store.list
    first = real(limit=1)
    assert first.items and first.next_cursor  # fixture sanity: two revisions

    def paged(cursor=None, limit=500):
        if cursor is None:
            return first
        raise error

    monkeypatch.setattr(store, "list", paged)


def test_a_second_page_fault_is_a_typed_fault_not_a_smaller_inventory(dm, monkeypatch):
    _commit(dm, "A", "c1")
    _commit(dm, "B", "c2")
    (dm.workspace_path / "plots" / "plot_1_Old.html").write_text("<html></html>")
    _two_pages_second_fails(dm, monkeypatch, StorageError("catalog page unreadable"))

    with pytest.raises(FigureError) as exc:
        dm.list_workspace_files()
    assert safe_reason(exc.value) == "storage_failed"


def test_an_untyped_page_fault_is_raised_typed_without_its_text(dm, monkeypatch):
    _commit(dm, "A", "c1")
    _commit(dm, "B", "c2")
    _two_pages_second_fails(dm, monkeypatch, RuntimeError("driver detail /secret"))

    with pytest.raises(StorageError) as exc:
        dm.list_workspace_files()
    assert safe_reason(exc.value) == "storage_failed"
    assert "secret" not in str(exc.value)


def test_a_stat_fault_after_one_record_is_a_typed_fault(dm, monkeypatch):
    _commit(dm, "A", "c1")
    second = _commit(dm, "B", "c2")
    fid, rid = second.split("@")
    target = str(dm.workspace_path / "plots" / fid / rid / "figure.json")
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        if os.fspath(path) == target:
            raise PermissionError(13, "denied")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)
    with pytest.raises(StorageError) as exc:
        dm.list_workspace_files()
    assert safe_reason(exc.value) == "storage_failed"


def test_the_artifact_manifest_snapshot_is_marked_failed(dm, monkeypatch):
    _commit(dm, "A", "c1")
    _commit(dm, "B", "c2")
    _two_pages_second_fails(dm, monkeypatch, StorageError("catalog page unreadable"))
    assert take_snapshot(dm).failed is True


def test_a_healthy_multi_page_inventory_is_complete(dm, monkeypatch):
    refs = {_commit(dm, "A", "c1"), _commit(dm, "B", "c2")}
    (dm.workspace_path / "plots" / "plot_1_Old.html").write_text("<html></html>")
    store = dm.figure_store
    real = store.list
    monkeypatch.setattr(
        store, "list", lambda cursor=None, limit=500: real(cursor=cursor, limit=1)
    )
    plots = dm.list_workspace_files()["plots"]
    assert {p["figure_ref"] for p in plots if not p["legacy"]} == refs
    assert [p["name"] for p in plots if p["legacy"]] == ["plot_1_Old.html"]


def test_a_missing_revision_file_keeps_the_recorded_size(dm):
    ref = _commit(dm, "A", "c1")
    fid, rid = ref.split("@")
    path = dm.workspace_path / "plots" / fid / rid / "figure.json"
    recorded = dm.figure_store.read_record(ref)["figure"]["bytes"]
    os.chmod(path, 0o644)
    path.unlink()
    [entry] = [p for p in dm.list_workspace_files()["plots"] if not p["legacy"]]
    assert entry["size"] == recorded and entry["modified"] is None


def test_no_plots_folder_is_an_empty_inventory_not_a_fault(dm):
    assert dm.list_workspace_files()["plots"] == []
