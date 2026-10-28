"""Saved-figure listing for the workspace overview (contract §8). Real disk."""

from __future__ import annotations

import asyncio
import json

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.figures.listing import (
    MAX_LABEL_CHARS,
    describe_saved_figures,
    figure_line,
    safe_label,
)
from lobster.services.execution.custom_code_execution_service import (
    CustomCodeExecutionService,
)
from lobster.tools.custom_code_tool import create_execute_custom_code_tool
from lobster.tools.workspace_tool import create_get_content_from_workspace_tool

FIG = "import plotly.graph_objects as go\n"


def _run(tool, code, call_id):
    async def go():
        msg = await tool.ainvoke(
            {
                "args": {"python_code": code, "load_workspace_files": False},
                "id": call_id,
                "name": tool.name,
                "type": "tool_call",
            }
        )
        return json.loads(msg.content)

    return asyncio.run(go())


def _overview(dm):
    return create_get_content_from_workspace_tool(dm).invoke({})


@pytest.fixture()
def saved(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path)
    tool = create_execute_custom_code_tool(
        dm, CustomCodeExecutionService(dm), agent_name="proteomics_expert"
    )
    [r1] = _run(
        tool,
        FIG
        + "submit_figure(go.Figure(go.Bar(y=[1])), source=[{'y': 1}], label='Totals')\n",
        "c1",
    )["figure_refs"]
    [r2] = _run(
        tool,
        FIG
        + f"b = load_figure({r1!r})\nb.update_layout(title='t')\n"
        + f"submit_figure(b, parent={r1!r}, label='Totals (titled)')\n",
        "c2",
    )["figure_refs"]
    return tmp_path, r1, r2


def test_lines_name_label_ref_kind_source_and_parent(saved):
    root, r1, r2 = saved
    dm = DataManagerV2(workspace_path=root)
    items = dm.figure_store.list().items
    lines = [figure_line(i) for i in items]
    assert lines == [
        f"Totals · {r1} · created · source yes",
        f"Totals (titled) · {r2} · edited · source no · edit of {r1}",
    ]


def test_the_overview_lists_saved_refs_after_a_restart(saved):
    root, r1, r2 = saved
    (root / "plots" / "plot_1_umap.html").write_text(
        "<html></html>"
    )  # legacy flat file
    text = _overview(DataManagerV2(workspace_path=root))  # fresh DM: nothing in memory
    assert "## Saved figures (2)" in text
    assert f"Totals · {r1} · created · source yes" in text
    assert f"edit of {r1}" in text and r2 in text
    assert "legacy:plot_1_umap.html" in text
    assert "the original stays unchanged" in text
    assert "Do NOT recreate" not in text  # facts only in tool output


def test_an_unreadable_catalog_is_stated_never_shown_empty(saved):
    root, r1, _ = saved
    (root / "plots" / r1.replace("@", "/") / "revision.json").write_bytes(b"{not json")
    text = _overview(DataManagerV2(workspace_path=root))
    assert "- Not listed: fingerprint_mismatch:invalid_json" in text


def test_nothing_saved_adds_no_section(tmp_path):
    assert describe_saved_figures(None) == ""
    assert "Saved figures" not in _overview(DataManagerV2(workspace_path=tmp_path))


def test_older_revisions_beyond_the_limit_are_counted_not_hidden(saved):
    root, r1, r2 = saved
    text = describe_saved_figures(
        DataManagerV2(workspace_path=root).figure_store, limit=1
    )
    assert "1 older revision(s) not listed" in text and f"· {r2} ·" in text
    assert (
        f"Totals · {r1} ·" not in text
    )  # r1 itself is not listed (only as the parent)


def test_pending_is_stated_on_the_line():
    item = {
        "figure_id": "f_a",
        "revision_id": "r_b",
        "label": "X",
        "change": {"kind": "created"},
        "availability": "pending",
    }
    assert figure_line(item).endswith("· save confirmation pending")


def test_labels_are_one_bounded_line_without_angle_brackets():
    assert safe_label("a\n<b>\tc") == "a b c"
    assert len(safe_label("x" * 500)) == MAX_LABEL_CHARS
    assert safe_label(None) == "" and safe_label(5) == ""


# --- an unreadable inventory is stated, never silently empty ----------------------


def _eio_for(monkeypatch, target, fn_name="stat"):
    """Make os.<fn_name> raise EIO for exactly ``target`` (all other paths unchanged)."""
    import errno
    import os

    real = getattr(os, fn_name)

    def faulty(path, *a, **k):
        if os.fspath(path) == os.fspath(target):
            raise OSError(errno.EIO, "injected")
        return real(path, *a, **k)

    monkeypatch.setattr(os, fn_name, faulty)


def test_a_plots_folder_stat_fault_is_stated_in_the_overview(saved, monkeypatch):
    root, _, _ = saved
    dm = DataManagerV2(workspace_path=root)
    _eio_for(monkeypatch, root / "plots")
    text = _overview(dm)
    assert "## Saved figures" in text and "- Not listed: storage_failed" in text


def test_a_plots_folder_enumeration_fault_is_stated(saved, monkeypatch):
    root, _, _ = saved
    dm = DataManagerV2(workspace_path=root)
    _eio_for(monkeypatch, root / "plots", "listdir")  # pathlib.iterdir (store catalog)
    _eio_for(monkeypatch, root / "plots", "scandir")  # legacy flat-file listing
    text = _overview(dm)
    assert "Not listed: storage_failed" in text


def test_a_figure_store_access_fault_is_stated(saved, monkeypatch):
    from lobster.core.figures.errors import StorageError

    root, _, _ = saved
    dm = DataManagerV2(workspace_path=root)

    def broken(self):
        raise StorageError("store unavailable")

    monkeypatch.setattr(DataManagerV2, "figure_store", property(broken))
    assert "- Not listed: storage_failed" in _overview(dm)


def test_legacy_file_names_are_one_line_and_tag_safe(tmp_path):
    plots = tmp_path / "plots"
    plots.mkdir()
    (plots / "a\nb<script>.html").write_text("x")
    text = _overview(DataManagerV2(workspace_path=tmp_path))
    assert "legacy:a b script .html" in text
    assert "<script>" not in text


def test_the_manifest_is_withheld_when_the_figure_inventory_is_unreadable(saved):
    from lobster.agents.artifact_manifest import append_manifest, take_snapshot

    root, r1, _ = saved
    dm = DataManagerV2(workspace_path=root)
    before = take_snapshot(dm)
    assert not before.failed and r1 in before.figures
    (root / "plots" / r1.replace("@", "/") / "revision.json").write_bytes(b"{bad")
    after = take_snapshot(DataManagerV2(workspace_path=root))
    assert after.failed  # existing contract: incomplete snapshot -> no manifest
    assert (
        append_manifest("prose", before, after) == "prose"
    )  # never "No new artifacts"


def test_the_manifest_is_withheld_on_a_catalog_read_fault(saved, monkeypatch):
    from lobster.agents.artifact_manifest import take_snapshot
    from lobster.core.figures.errors import StorageError
    from lobster.core.figures.store import LocalFigureStore

    root, _, _ = saved

    def fault(self, *a, **k):
        raise StorageError("read fault")

    monkeypatch.setattr(LocalFigureStore, "list", fault)
    assert take_snapshot(DataManagerV2(workspace_path=root)).failed


def test_a_legacy_listing_fault_is_stated_and_saved_refs_stay_listed(
    saved, monkeypatch
):
    root, r1, _ = saved
    dm = DataManagerV2(workspace_path=root)
    _eio_for(monkeypatch, root / "plots", "scandir")  # the store reads via listdir
    text = _overview(dm)
    assert r1 in text
    assert "- Legacy plot files: not listed (storage_failed)" in text


def test_a_plots_file_is_not_an_empty_catalog(tmp_path):
    from lobster.core.figures.errors import StorageError

    (tmp_path / "plots").write_text("not a folder")
    dm = DataManagerV2(workspace_path=tmp_path)
    assert "- Not listed: storage_failed" in _overview(dm)
    with pytest.raises(StorageError):
        dm._saved_figure_records()


@pytest.mark.parametrize(
    "raw, shown",
    [
        ("a\x1b[31mb", "a [31mb"),
        ("x\x00y", "x y"),
        ("a\u202eb\u200bc", "a b c"),
        ("a\rb\tc\u2028d\x85e", "a b c d e"),
        ("<img src=x>", "img src=x"),
    ],
)
def test_labels_and_names_are_rendered_without_control_or_markup_characters(raw, shown):
    from lobster.core.figures.listing import safe_label

    assert safe_label(raw) == shown
