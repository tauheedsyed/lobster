"""Saved plots are visible to the agent, and the plot tool tells the truth.

Regression for the Cloud report "the agent couldn't confirm the plot file":
the HTML was written to ``<workspace>/plots`` but the workspace listing (and
therefore the handoff artifact manifest) never looked there, and the tool
returned no path. PNG export failures logged an apparently empty reason
because the exporter starts its message with a newline.

Deterministic; real PlotManager / DataManagerV2 / plotly, no model calls.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import plotly.graph_objects as go
import pytest

from lobster.agents.artifact_manifest import diff_snapshots, take_snapshot
from lobster.core.plot_manager import PlotManager, _one_line_reason
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.tools.plot_tools import create_plot_tools

KALEIDO_MESSAGE = (
    "\nImage export requires the Kaleido package, v1.0.0 or greater,\n"
    '    $ pip install --upgrade "kaleido>=1"\n'
)


@pytest.fixture
def no_png(monkeypatch):
    """Make PNG export fail the way an old kaleido does in the Cloud worker."""
    import plotly.io as pio

    def _fail(*_args, **_kwargs):
        raise ValueError(KALEIDO_MESSAGE)

    monkeypatch.setattr(pio, "write_image", _fail)


@pytest.fixture
def png_ok(monkeypatch):
    import plotly.io as pio

    def _write(_fig, path, *_args, **_kwargs):
        open(path, "wb").write(b"\x89PNG")

    monkeypatch.setattr(pio, "write_image", _write)


def _bar() -> go.Figure:
    return go.Figure(go.Bar(x=["A", "B", "C"], y=[3, 5, 2]))


def _tool(workspace, plot_manager):
    dm = SimpleNamespace(plot_manager=plot_manager)
    (tool,) = create_plot_tools(dm, workspace)
    return tool


def _invoke(tool, title="Simple Bar Chart"):
    return tool.invoke(
        {
            "plot_type": "bar",
            "data_json": json.dumps(
                [{"label": k, "value": v} for k, v in (("A", 3), ("B", 5), ("C", 2))]
            ),
            "x": "label",
            "y": "value",
            "title": title,
        }
    )


# --- one-line reason -------------------------------------------------------


def test_one_line_reason_skips_leading_newline_and_trailing_comma():
    assert (
        _one_line_reason(ValueError(KALEIDO_MESSAGE))
        == "Image export requires the Kaleido package, v1.0.0 or greater"
    )


def test_one_line_reason_falls_back_to_type_name():
    assert _one_line_reason(RuntimeError("  \n ")) == "RuntimeError"


@pytest.mark.parametrize(
    "exc, expected",
    [
        (
            PermissionError(13, "Permission denied", "/abs/ws/p.html"),
            "Permission denied: p.html",
        ),
        (PermissionError(13, None, "/abs/ws/p.html"), "PermissionError: p.html"),
        (OSError(28, "No space left", b"/abs/ws/p.png"), "No space left: p.png"),
        (OSError(9, "Bad file descriptor", 7), "Bad file descriptor"),
        (OSError("\ndisk full"), "disk full"),
    ],
)
def test_one_line_reason_for_os_errors_never_shows_a_path(exc, expected):
    reason = _one_line_reason(exc)
    assert reason == expected
    assert "/abs/" not in reason


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            "Failed to write /Users/someone/ws/plots/plot_1.html: disk full",
            "Failed to write plot_1.html: disk full",
        ),
        ("C:\\Users\\someone\\ws\\p.png not writable", "p.png not writable"),
        ("path '/var/folders/ab/T/tmpx/file.json' bad", "path 'file.json' bad"),
        (
            "\nChrome is required at /opt/google/chrome/chrome",
            "Chrome is required at chrome",
        ),
        ("path:/home/someone/ws/plot.png bad", "path:plot.png bad"),
        ("\\\\server\\share\\dir\\p.png gone", "p.png gone"),
        ("/abs/dir/ trailing", "dir trailing"),
    ],
)
def test_one_line_reason_strips_directories_from_any_message(message, expected):
    assert _one_line_reason(ValueError(message)) == expected


@pytest.mark.parametrize(
    "message",
    [
        "see https://plotly.com/python/static-image-export/ for help",
        "saved under plots/plot_1.html",
        "and / or 1 / 2",
        "1 /2",
        "~/plots/a.html",
        "see https://example.org/a/b?next=/c/d for help",
        "/srv",
    ],
)
def test_one_line_reason_keeps_urls_relative_paths_and_slashes(message):
    assert _one_line_reason(RuntimeError(message)) == message


def test_one_line_reason_os_error_without_filename_strips_directories():
    assert _one_line_reason(OSError("cannot open /tmp/x/y.png")) == "cannot open y.png"


def test_one_line_reason_is_bounded():
    assert len(_one_line_reason(ValueError("x" * 1000))) == 200


# --- PlotManager per-plot outcome -----------------------------------------


def test_save_reports_html_path_and_png_reason(tmp_path, no_png):
    pm = PlotManager(workspace_path=tmp_path, min_save_interval=0)
    plot_id, _, _ = pm.add_plot(plot=_bar(), title="Bars", source="test")

    saved, stats, _ = pm.save_plots_to_workspace()

    outcome = stats["per_plot"][plot_id]
    assert outcome["html"] in saved
    assert (tmp_path / "plots").is_dir()
    assert outcome["html"].endswith(".html")
    assert outcome["png"] is None
    assert outcome["png_skipped"] == (
        "Image export requires the Kaleido package, v1.0.0 or greater"
    )
    assert outcome["error"] is None


def test_save_reports_png_path_when_export_works(tmp_path, png_ok):
    pm = PlotManager(workspace_path=tmp_path, min_save_interval=0)
    plot_id, _, _ = pm.add_plot(plot=_bar(), title="Bars", source="test")

    saved, stats, _ = pm.save_plots_to_workspace()

    outcome = stats["per_plot"][plot_id]
    assert outcome["png"] in saved
    assert outcome["png_skipped"] is None


# --- tool output ------------------------------------------------------------


def test_tool_names_saved_path_and_png_skip(tmp_path, no_png):
    pm = PlotManager(workspace_path=tmp_path, min_save_interval=0)
    out = _invoke(_tool(tmp_path, pm))

    assert out.startswith("Plot registered: plot_1 | Simple Bar Chart | 3 rows")
    assert "registered for display" in out
    assert "saved: plots/plot_1_Simple_Bar_Chart" in out
    saved_rel = out.split("saved: ")[1].split(" |")[0]
    assert (tmp_path / saved_rel).is_file()
    assert (
        "png: not saved (Image export requires the Kaleido package, v1.0.0 or greater)"
        in out
    )


def test_tool_names_png_path_when_written(tmp_path, png_ok):
    pm = PlotManager(workspace_path=tmp_path, min_save_interval=0)
    out = _invoke(_tool(tmp_path, pm))

    png_rel = out.split("png: ")[1].strip()
    assert png_rel.startswith("plots/") and png_rel.endswith(".png")
    assert (tmp_path / png_rel).is_file()


def test_tool_reports_rate_limited_save_without_a_path(tmp_path, no_png):
    pm = PlotManager(workspace_path=tmp_path, min_save_interval=3600)
    tool = _tool(tmp_path, pm)
    _invoke(tool, title="First")

    out = _invoke(tool, title="Second")

    assert "plot_2" in out
    assert "file not saved: rate_limited" in out
    assert "| saved:" not in out and ".html" not in out


def test_tool_reports_failed_html_without_a_path(tmp_path, monkeypatch):
    import plotly.io as pio

    def _boom(*_a, **_k):
        raise OSError("\ndisk full")

    monkeypatch.setattr(pio, "write_html", _boom)
    pm = PlotManager(workspace_path=tmp_path, min_save_interval=0)

    out = _invoke(_tool(tmp_path, pm))

    assert "file not saved: disk full" in out
    assert "| saved:" not in out and ".html" not in out


def test_reason_for_filesystem_error_hides_absolute_path(tmp_path, monkeypatch):
    import plotly.io as pio

    def _denied(_fig, path, *_a, **_k):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(pio, "write_html", _denied)
    pm = PlotManager(workspace_path=tmp_path, min_save_interval=0)

    out = _invoke(_tool(tmp_path, pm))

    assert "file not saved: Permission denied: plot_1_Simple_Bar_Chart.html" in out
    assert str(tmp_path) not in out


def test_tool_reports_the_current_plot_not_an_earlier_one(tmp_path, monkeypatch):
    """An earlier plot saved fine; the current one fails: no path is named."""
    import plotly.io as pio

    real_write_html = pio.write_html

    def _fail_second(fig, path, *a, **k):
        if "Second" in str(path):
            raise OSError("\nno space left")
        return real_write_html(fig, path, *a, **k)

    monkeypatch.setattr(pio, "write_html", _fail_second)
    monkeypatch.setattr(pio, "write_image", lambda *_a, **_k: None)
    pm = PlotManager(workspace_path=tmp_path, min_save_interval=0)
    tool = _tool(tmp_path, pm)
    first = _invoke(tool, title="First")

    out = _invoke(tool, title="Second")

    assert "saved: plots/plot_1_First" in first
    assert out.startswith("Plot registered: plot_2 | Second")
    assert "file not saved: no space left" in out
    assert "plot_1" not in out


def test_tool_reports_the_current_plot_after_history_eviction(tmp_path, no_png):
    pm = PlotManager(workspace_path=tmp_path, max_plots_history=2, min_save_interval=0)
    tool = _tool(tmp_path, pm)
    for title in ("One", "Two"):
        _invoke(tool, title=title)

    out = _invoke(tool, title="Three")

    assert out.startswith("Plot registered: plot_3 | Three")
    assert "saved: plots/plot_3_Three.html" in out
    assert (tmp_path / "plots" / "plot_3_Three.html").is_file()


# --- workspace listing + artifact manifest ----------------------------------


def test_list_workspace_files_has_empty_plots_before_first_save(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path)
    files = dm.list_workspace_files()
    assert files["plots"] == []
    assert {"data", "exports", "cache"} <= set(files)


def test_list_workspace_files_lists_saved_plots(tmp_path, no_png):
    dm = DataManagerV2(workspace_path=tmp_path)
    dm.plot_manager._min_save_interval = 0
    dm.plot_manager.add_plot(plot=_bar(), title="Bars", source="test")
    dm.plot_manager.save_plots_to_workspace()

    plots = dm.list_workspace_files()["plots"]

    assert [p["name"] for p in plots] == ["plot_1_Bars.html"]
    assert plots[0]["size"] > 0 and "modified" in plots[0]


def test_artifact_manifest_reports_a_saved_plot(tmp_path, no_png):
    dm = DataManagerV2(workspace_path=tmp_path)
    dm.plot_manager._min_save_interval = 0
    before = take_snapshot(dm)

    dm.plot_manager.add_plot(plot=_bar(), title="Bars", source="test")
    dm.plot_manager.save_plots_to_workspace()
    diff = diff_snapshots(before, take_snapshot(dm))

    assert not before.failed
    assert diff.files_written == ["plots/plot_1_Bars.html"]
