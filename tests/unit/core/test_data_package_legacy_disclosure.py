"""Legacy flat plot files: listed by the workspace, not supported by export or /plot here.

Disclosure only: legacy files stay read-only and unchanged, no history is invented for
them, and their presence is never reported as "no plots". Real store on tmp_path.
"""

from __future__ import annotations

import hashlib
import zipfile
from types import SimpleNamespace

import anndata
import numpy as np
import plotly.graph_objects as go
import pytest

from lobster.cli_internal.commands.heavy import visualization_commands as V
from lobster.core.runtime.data_manager import DataManagerV2

LEGACY = "plot_1_Old.html"


@pytest.fixture()
def dm(tmp_path):
    return DataManagerV2(workspace_path=tmp_path)


def _commit(dm, title="Bars"):
    fig = go.Figure(go.Bar(x=[1], y=[2]))
    plot_id, _, _ = dm.plot_manager.add_plot(plot=fig, title=title, source="t")
    [res] = dm.plot_manager.commit_staged(
        {"id": "a1"}, tool_call_id="c1", staged_ids=[plot_id]
    )
    return res.ref


def _legacy(dm, name=LEGACY, data=b"<html>old</html>"):
    path = dm.workspace_path / "plots" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(dm):
    root = dm.workspace_path / "plots"
    return {
        str(p.relative_to(root)): _sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _output():
    printed = []
    return printed, SimpleNamespace(
        print=lambda msg, style=None: printed.append(str(msg)),
        print_table=lambda table: printed.append(repr(table)),
    )


# --- package (DataManagerV2) ---------------------------------------------------------


def test_mixed_package_reports_the_legacy_files_it_does_not_copy(dm, tmp_path):
    _commit(dm)
    _legacy(dm)
    dm.modalities["m"] = anndata.AnnData(np.ones((2, 2)))
    before = _tree(dm)

    zip_path = dm.create_data_package(output_dir=str(tmp_path / "out"))
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        readme = zf.read("README.md").decode("utf-8")

    assert not any(n.endswith(LEGACY) for n in names)
    assert "## Not copied" in readme
    assert (
        f"- plots/{LEGACY}: legacy plot file, not supported by this export "
        "(unchanged in the workspace)" in readme
    )
    assert dm.last_package_report["legacy_plot_files_not_copied"] == 1
    # read-only: legacy bytes unchanged, nothing new written for them (no invented history)
    after = _tree(dm)
    assert after[LEGACY] == before[LEGACY]
    assert {k: v for k, v in after.items() if k in before} == before


def test_package_without_legacy_files_reports_zero(dm, tmp_path):
    _commit(dm)
    zip_path = dm.create_data_package(output_dir=str(tmp_path / "out"))
    with zipfile.ZipFile(zip_path) as zf:
        readme = zf.read("README.md").decode("utf-8")
    assert dm.last_package_report["legacy_plot_files_not_copied"] == 0
    assert "legacy plot file" not in readme


def test_legacy_only_workspace_is_not_reported_as_having_no_plots(dm):
    legacy = _legacy(dm)
    before = _sha(legacy)
    with pytest.raises(ValueError) as exc:
        dm.create_data_package()
    message = str(exc.value)
    assert "no data or plots" not in message.lower()
    assert "1 legacy plot file(s) in plots/ are not supported by export" in message
    assert _sha(legacy) == before


def test_no_data_and_no_plots_message_is_unchanged(dm):
    with pytest.raises(ValueError, match="No data or plots to export"):
        dm.create_data_package()


def test_legacy_names_are_display_safe_in_the_readme(dm, tmp_path):
    _commit(dm)
    _legacy(dm, name="evil<script>\x1b[31m.html")
    zip_path = dm.create_data_package(output_dir=str(tmp_path / "out"))
    with zipfile.ZipFile(zip_path) as zf:
        readme = zf.read("README.md").decode("utf-8")
    assert "<script>" not in readme and "\x1b" not in readme
    assert "legacy plot file, not supported by this export" in readme


# --- CLI (visualization commands) ------------------------------------------------------


def test_export_command_names_legacy_files_left_out(dm):
    _commit(dm)
    _legacy(dm)
    dm.modalities["m"] = anndata.AnnData(np.ones((2, 2)))
    printed, output = _output()
    V.export_data(SimpleNamespace(data_manager=dm), output, include_png=False)
    assert any("1 legacy plot file(s)" in m for m in printed)


def test_export_without_data_still_says_legacy_files_are_not_included(dm, tmp_path):
    _legacy(dm)
    printed, output = _output()
    client = SimpleNamespace(
        data_manager=dm, export_session=lambda: tmp_path / "session.json"
    )
    V.export_data(client, output)
    assert any(
        "1 legacy plot file(s) in plots/ are not supported by export" in m
        for m in printed
    )


def test_plots_after_restart_is_not_a_false_no_plots(dm, tmp_path):
    _commit(dm)
    _legacy(dm)
    fresh = DataManagerV2(workspace_path=tmp_path)  # restart: no session plots
    printed, output = _output()
    V.plots_list(SimpleNamespace(data_manager=fresh), output)
    text = "\n".join(printed)
    assert "No plots generated yet" not in text
    assert "1 saved figure(s), 1 revision(s) in this workspace" in text
    assert "1 legacy plot file(s) in plots/" in text


def test_plots_with_nothing_saved_keeps_the_plain_message(dm):
    printed, output = _output()
    V.plots_list(SimpleNamespace(data_manager=dm), output)
    assert any("No plots generated yet" in m for m in printed)


def test_plot_show_names_a_legacy_file_instead_of_not_found(dm, monkeypatch):
    legacy = _legacy(dm)
    before = _sha(legacy)
    opened = []
    monkeypatch.setattr(V, "open_path", lambda p: opened.append(p) or (True, "ok"))
    printed, output = _output()
    summary = V.plot_show(SimpleNamespace(data_manager=dm), output, LEGACY)
    text = "\n".join(printed)
    assert "Plot not found" not in text
    assert "legacy plot file" in text and "not supported" in text
    assert "legacy" in summary.lower()
    assert opened == []  # no legacy open fallback in this release
    assert _sha(legacy) == before


def test_plot_show_unknown_identifier_is_still_not_found(dm, monkeypatch):
    _legacy(dm)
    monkeypatch.setattr(V, "open_path", lambda p: (True, "ok"))
    printed, output = _output()
    summary = V.plot_show(SimpleNamespace(data_manager=dm), output, "nothing-like-it")
    assert summary == "Plot not found: nothing-like-it"


# --- an unlistable plots/ folder is "could not be listed", never 0 or "not found" ---------


def _legacy_listing_fails(monkeypatch):
    from lobster.core.figures import listing
    from lobster.core.figures.errors import StorageError

    def boom(_workspace):
        raise StorageError("plots folder unreadable")

    monkeypatch.setattr(listing, "legacy_plot_files", boom)


def test_export_says_legacy_files_could_not_be_listed_never_zero(dm, monkeypatch):
    _commit(dm)
    dm.modalities["m"] = anndata.AnnData(np.ones((2, 2)))
    _legacy_listing_fails(monkeypatch)
    printed, output = _output()
    V.export_data(SimpleNamespace(data_manager=dm), output, include_png=False)
    assert dm.last_package_report["legacy_plot_files_not_copied"] is None
    text = "\n".join(printed)
    assert "legacy plot files could not be listed" in text
    assert "0 legacy plot file(s)" not in text


def test_plot_show_with_unlistable_legacy_files_is_not_a_plain_not_found(
    dm, monkeypatch
):
    _legacy_listing_fails(monkeypatch)
    monkeypatch.setattr(V, "open_path", lambda p: (True, "ok"))
    printed, output = _output()
    summary = V.plot_show(SimpleNamespace(data_manager=dm), output, LEGACY)
    text = "\n".join(printed)
    assert "not among saved figures" in text
    assert "legacy plot files could not be listed" in text
    assert "could not be listed" in summary
