"""Local HTML renditions and the three explicit view/save/export paths.

Real store on tmp_path. No Chrome/Kaleido; plotly.js is inlined from the installed package.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

import plotly.graph_objects as go
import pytest

from lobster.core.figures import renditions as R
from lobster.core.figures.errors import IntegrityError
from lobster.core.figures.ids import FigureRef
from lobster.core.runtime.data_manager import DataManagerV2

CANARY = "CANARY_render_detail /Users/someone/x"


def _commit(dm, title="Volcano", call="call_1", color="red"):
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4], marker={"color": color}))
    plot_id, _, _ = dm.plot_manager.add_plot(plot=fig, title=title, source="test")
    [res] = dm.plot_manager.commit_staged(
        {"id": f"act-{call}"}, tool_call_id=call, staged_ids=[plot_id]
    )
    return plot_id, res


@pytest.fixture()
def dm(tmp_path):
    return DataManagerV2(workspace_path=tmp_path)


def test_html_view_is_rendered_from_exact_bytes_once_and_append_only(dm):
    _, res = _commit(dm)
    store = dm.figure_store
    ref = FigureRef.parse(res.ref)
    rev_before = (store._rev_dir(ref) / "revision.json").read_bytes()

    first = R.ensure_html_rendition(store, res.ref)
    second = R.ensure_html_rendition(store, res.ref)

    assert first.status == "available" and not first.reused
    assert second.reused and second.rendition_id == first.rendition_id
    html = (dm.workspace_path / first.path).read_text(encoding="utf-8")
    assert not re.findall(
        r"<script[^>]*\bsrc=", html
    )  # plotly.js inline, no CDN script
    assert f'id="lobster-{ref.revision_id}"' in html
    [record] = store.list_renditions(res.ref)
    revision = store.read_record(res.ref)
    assert record["inputs"] == {
        "figure_sha256": revision["figure"]["sha256"],
        "config_sha256": revision["config"]["sha256"],
    }
    assert record["renderer"]["name"] == "plotly.io.to_html"
    # the revision itself is never rewritten by a rendition
    assert (store._rev_dir(ref) / "revision.json").read_bytes() == rev_before
    assert store.read(res.ref)[0] == rev_before


def test_damaged_view_is_not_reused_and_a_new_one_is_appended(dm):
    _, res = _commit(dm)
    store = dm.figure_store
    first = R.ensure_html_rendition(store, res.ref)
    path = dm.workspace_path / first.path
    path.chmod(0o644)
    path.write_text("<html>changed</html>")
    with pytest.raises(IntegrityError):
        store.read_rendition(res.ref, first.rendition_id)
    again = R.ensure_html_rendition(store, res.ref)
    assert again.status == "available" and again.rendition_id != first.rendition_id
    assert len(store.list_renditions(res.ref)) == 2  # append-only: the old record stays


def test_rendition_inputs_must_match_the_revision(dm):
    _, res = _commit(dm)
    with pytest.raises(IntegrityError):
        dm.figure_store.add_rendition(
            res.ref,
            b"<html></html>",
            R.HTML_SETTINGS,
            {"renderer": {}, "inputs": {"figure_sha256": "0" * 64}},
        )


def test_view_failure_is_separate_from_the_saved_revision(dm, monkeypatch):
    _, res = _commit(dm)

    def boom(*_a, **_k):
        raise RuntimeError(CANARY)

    monkeypatch.setattr(R, "render_html", boom)
    view = R.ensure_html_rendition(dm.figure_store, res.ref)
    assert (view.status, view.reason) == ("failed", "html_rendition:unexpected_error")
    assert dm.figure_store.read(res.ref)  # revision intact and readable
    text = R.describe_renditions([view])
    assert "revisions unaffected" in text and "CANARY" not in text


# --- explicit paths: CLI /plot, dashboard /save + preview, data package -----------------


def _cli(dm, monkeypatch):
    from lobster.cli_internal.commands.heavy import visualization_commands as V
    from lobster.cli_internal.commands.output_adapter import JsonOutputAdapter

    opened = []
    monkeypatch.setattr(
        V, "open_path", lambda p: (opened.append(Path(p)), (True, "ok"))[1]
    )
    return V, JsonOutputAdapter(), SimpleNamespace(data_manager=dm), opened


def test_cli_plot_opens_the_html_view_by_plot_id_and_after_restart_by_ref(
    dm, monkeypatch, tmp_path
):
    plot_id, res = _commit(dm, title="Volcano treated")
    V, out, client, opened = _cli(dm, monkeypatch)

    reply = V.plot_show(client, out, plot_id)

    assert reply == f"Opened plot: Volcano treated ({res.ref})"
    assert opened[-1].suffix == ".html" and "renditions" in opened[-1].parts
    # restart: a fresh DataManagerV2 has empty latest_plots; the store still resolves
    fresh = DataManagerV2(workspace_path=tmp_path)
    assert fresh.latest_plots == []
    V2, out2, client2, opened2 = _cli(fresh, monkeypatch)
    assert (
        V2.plot_show(client2, out2, res.ref)
        == f"Opened plot: Volcano treated ({res.ref})"
    )
    assert V2.plot_show(client2, out2, "treated").startswith("Opened plot")
    assert opened2[-1] == opened[-1]  # same rendition reused, no duplicate


def test_cli_plot_without_identifier_writes_missing_views_then_opens_dir(
    dm, monkeypatch
):
    _commit(dm, call="c1")
    _commit(dm, call="c2", color="blue")
    V, out, client, opened = _cli(dm, monkeypatch)
    assert V.plot_show(client, out, None).startswith("Opened plots directory")
    assert opened == [dm.workspace_path / "plots"]
    assert "HTML views: 2 ready (2 new)" in json.dumps(out.blocks)


def test_cli_plot_reports_saved_revision_when_view_fails(dm, monkeypatch):
    plot_id, res = _commit(dm)
    V, out, client, opened = _cli(dm, monkeypatch)
    monkeypatch.setattr(R, "render_html", lambda *a, **k: 1 / 0)
    reply = V.plot_show(client, out, plot_id)
    assert (
        reply
        == f"Figure {res.ref} saved; HTML view failed: html_rendition:unexpected_error"
    )
    assert opened == []


def test_cli_plot_unknown_identifier(dm, monkeypatch):
    V, out, client, _ = _cli(dm, monkeypatch)
    assert V.plot_show(client, out, "nope") == "Plot not found: nope"


def test_dashboard_save_writes_views_and_preview_opens_the_ref(dm, monkeypatch):
    pytest.importorskip("textual")
    from lobster.ui.screens.analysis_screen import AnalysisScreen
    from lobster.ui.widgets.plot_preview import PlotPreview

    _, res = _commit(dm)
    messages, refreshed = [], []
    results = SimpleNamespace(append_system_message=messages.append)
    preview = SimpleNamespace(refresh_plots=lambda: refreshed.append(True))
    fake = SimpleNamespace(
        client=SimpleNamespace(data_manager=dm),
        query_one=lambda kind: preview if kind is PlotPreview else results,
        notify=lambda *a, **k: None,
    )
    AnalysisScreen._handle_slash_command(fake, "/save")
    assert messages == ["1 saved figure revision(s); HTML views: 1 ready (1 new)"]
    assert refreshed == [True]

    import webbrowser

    urls = []
    monkeypatch.setattr(webbrowser, "open", urls.append)
    widget = SimpleNamespace(
        _plots=list(dm.latest_plots),
        client=SimpleNamespace(data_manager=dm),
        notify=lambda *a, **k: None,
    )
    PlotPreview._open_plot(widget, 0)
    view = R.ensure_html_rendition(dm.figure_store, res.ref)
    assert urls == [f"file://{dm.workspace_path / view.path}"]


def test_data_package_includes_exact_revision_bytes_and_html_view_after_restart(
    dm, tmp_path
):
    _, res = _commit(dm)
    fid, rid = res.ref.split("@")
    exact = dm.figure_store.read(res.ref)
    fresh = DataManagerV2(workspace_path=tmp_path)  # restart: latest_plots empty
    zip_path = fresh.create_data_package(output_dir=str(tmp_path / "out"))
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        index = json.loads(zf.read("plots/index.json"))
        assert zf.read(f"plots/{fid}/{rid}/figure.json") == exact[1]
        assert zf.read(f"plots/{fid}/{rid}/config.json") == exact[2]
        assert zf.read(f"plots/{fid}/{rid}/revision.json") == exact[0]
        assert f"plots/{fid}/{rid}/figure.html" in names
    assert index[0]["figure_ref"] == res.ref
    assert index[0]["png"] == "not_rendered_locally"
    assert not any(n.endswith(".png") for n in names)


@pytest.mark.parametrize("include_png", [True, False])
def test_data_package_readme_never_claims_png_or_standards_compliance(
    dm, tmp_path, include_png
):
    _commit(dm)  # fixture with NO recorded PNG rendition
    zip_path = dm.create_data_package(
        output_dir=str(tmp_path / f"out_{include_png}"), include_png=include_png
    )
    with zipfile.ZipFile(zip_path) as zf:
        readme = zf.read("README.md").decode("utf-8")
        assert not any(n.endswith(".png") for n in zf.namelist())
    assert "no new PNG was rendered" in readme
    assert "no PNG rendition is recorded for these revisions" in readme
    if not include_png:
        assert "(--no-png: generation disabled)" in readme
    assert "## Not copied" not in readme
    assert dm.last_package_report["recorded_png_not_copied"] == 0
    assert "(included)" not in readme
    assert "W3C" not in readme and "compliant" not in readme


def _record_png(dm, ref, data=b"\x89PNG\r\n\x1a\nrecorded-elsewhere"):
    rec = dm.figure_store.read_record(ref)
    receipt = {
        "renderer": {"name": "browser", "version": "test"},
        "inputs": {
            "figure_sha256": rec["figure"]["sha256"],
            "config_sha256": rec["config"]["sha256"],
        },
    }
    return dm.figure_store.add_rendition(ref, data, {"format": "png"}, receipt), data


def test_data_package_preserves_already_recorded_png_renditions(dm, tmp_path):
    """NOT generated locally != NOT preserved: recorded PNG bytes travel exactly."""
    _, res = _commit(dm)
    added, data = _record_png(dm, res.ref)
    fid, rid = res.ref.split("@")
    fresh = DataManagerV2(workspace_path=tmp_path)  # restart-safe
    zip_path = fresh.create_data_package(output_dir=str(tmp_path / "out"))
    member = f"plots/{fid}/{rid}/{added['rendition_id']}.png"
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.read(member) == data
        index = json.loads(zf.read("plots/index.json"))
        readme = zf.read("README.md").decode("utf-8")
        plots_readme = zf.read("plots/README.md").decode("utf-8")
    [png] = index[0]["png_renditions"]
    assert png["file"] == member[len("plots/") :]
    assert png["sha256"] == added["sha256"]
    assert index[0]["png"] == "not_rendered_locally"  # still: nothing NEW rendered here
    assert "1 already recorded PNG rendition(s) included" in readme
    assert "no new PNG was rendered locally" in readme
    assert png["file"] in plots_readme
    assert fresh.last_package_report["recorded_png_included"] == 1
    # --no-png disables GENERATION only: recorded PNGs are retained
    zip2 = fresh.create_data_package(
        output_dir=str(tmp_path / "out2"), include_png=False
    )
    with zipfile.ZipFile(zip2) as zf:
        assert zf.read(member) == data
        readme2 = zf.read("README.md").decode("utf-8")
    assert "1 already recorded PNG rendition(s) included" in readme2
    assert "(--no-png: generation disabled)" in readme2


def test_data_package_reports_a_changed_recorded_png_instead_of_copying_it(
    dm, tmp_path
):
    _, res = _commit(dm)
    added, _ = _record_png(dm, res.ref)
    (dm.workspace_path / added["path"]).write_bytes(b"tampered")
    zip_path = dm.create_data_package(output_dir=str(tmp_path / "out"))
    with zipfile.ZipFile(zip_path) as zf:
        index = json.loads(zf.read("plots/index.json"))
        assert not any(n.endswith(".png") for n in zf.namelist())
        readme = zf.read("README.md").decode("utf-8")
        plots_readme = zf.read("plots/README.md").decode("utf-8")
    assert index[0]["png_errors"] == [
        {
            "rendition_id": added["rendition_id"],
            "reason": "fingerprint_mismatch:rendition_changed",
        }
    ]
    # visible beyond the index: top README, plots README and the package report
    line = (
        f"- {res.ref}: recorded PNG {added['rendition_id']} not copied "
        "(fingerprint_mismatch:rendition_changed)"
    )
    assert "## Not copied" in readme and line in readme
    assert "1 recorded PNG rendition(s) were NOT copied" in readme
    assert f"NOT copied: {added['rendition_id']}" in plots_readme
    assert dm.last_package_report["recorded_png_not_copied"] == 1


def test_export_command_prints_what_the_package_did_not_copy(dm, tmp_path):
    import anndata
    import numpy as np

    from lobster.cli_internal.commands.heavy.visualization_commands import export_data

    _, res = _commit(dm)
    added, _ = _record_png(dm, res.ref)
    (dm.workspace_path / added["path"]).write_bytes(b"tampered")
    dm.modalities["m"] = anndata.AnnData(np.ones((2, 2)))
    printed = []
    output = SimpleNamespace(print=lambda msg, style=None: printed.append(msg))
    export_data(SimpleNamespace(data_manager=dm), output, include_png=False)
    assert any(
        "Not copied: 0 figure revision(s), 0 HTML view(s), 1 recorded PNG(s)" in m
        for m in printed
    )


def test_data_package_keeps_the_revision_when_its_view_fails(dm, tmp_path, monkeypatch):
    _, res = _commit(dm)
    monkeypatch.setattr(R, "render_html", lambda *a, **k: 1 / 0)
    zip_path = dm.create_data_package(output_dir=str(tmp_path / "out"))
    fid, rid = res.ref.split("@")
    with zipfile.ZipFile(zip_path) as zf:
        index = json.loads(zf.read("plots/index.json"))
        assert f"plots/{fid}/{rid}/figure.json" in zf.namelist()
    assert index[0]["html"] is None
    assert index[0]["html_error"] == "html_rendition:unexpected_error"
