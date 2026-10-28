"""A file with no recorded mtime (a saved revision whose file is missing) is "unknown".

Never the current time (``gmtime(None)``), never a TypeError against float mtimes, and
the listing's ``None`` is not rewritten.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from rich.console import Console

from lobster.cli_internal.commands.heavy.slash_commands import _build_files_blocks
from lobster.ui.components.status_display import EnhancedStatusDisplay

T_OLD, T_NEW = 1_700_000_000.0, 1_750_000_000.0


def _entry(name, modified):
    return {
        "name": name,
        "path": f"/w/plots/{name}",
        "size": 2048,
        "modified": modified,
    }


def _rows(files):
    [block] = _build_files_blocks({"plots": files})
    return block.data["rows"]


def _fmt(ts):
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts))


def test_a_single_unknown_mtime_is_shown_as_unknown_not_now():
    entry = _entry("f_a/r_a/figure.json", None)
    [row] = _rows([entry])
    assert row[2] == "unknown"
    assert entry["modified"] is None  # the listing's value is not rewritten


def test_mixed_unknown_and_known_mtimes_sort_newest_first_unknown_last():
    files = [
        _entry("unknown.json", None),
        _entry("old.html", T_OLD),
        _entry("new.html", T_NEW),
    ]
    rows = _rows(files)
    assert [r[0] for r in rows] == ["new.html", "old.html", "unknown.json"]
    assert [r[2] for r in rows] == [_fmt(T_NEW), _fmt(T_OLD), "unknown"]
    assert files[0]["modified"] is None


def test_known_mtimes_are_unchanged():
    rows = _rows([_entry("old.html", T_OLD), _entry("new.html", T_NEW)])
    assert [(r[0], r[2]) for r in rows] == [
        ("new.html", _fmt(T_NEW)),
        ("old.html", _fmt(T_OLD)),
    ]


def _panel_text(workspace_files):
    client = SimpleNamespace(
        data_manager=SimpleNamespace(list_workspace_files=lambda: workspace_files)
    )
    panel = EnhancedStatusDisplay()._create_recent_files_panel(client)
    console = Console(width=120, record=True, color_system=None)
    console.print(panel)
    return console.export_text()


def test_the_status_panel_sorts_mixed_mtimes_with_unknown_last():
    files = {
        "plots": [_entry("unknown.json", None), _entry("new.html", T_NEW)],
        "data": [_entry("old.csv", T_OLD)],
    }
    text = _panel_text(files)
    assert "Unable to list files" not in text
    assert text.index("new.html") < text.index("old.csv") < text.index("unknown.json")
    assert files["plots"][0]["modified"] is None
