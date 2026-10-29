"""/plots without session plots counts figures and revisions separately."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lobster.cli_internal.commands.heavy import visualization_commands as V
from lobster.core.runtime.data_manager import DataManagerV2


@pytest.fixture()
def dm(tmp_path):
    return DataManagerV2(workspace_path=tmp_path)


def _output():
    printed = []
    return printed, SimpleNamespace(
        print=lambda msg, style=None: printed.append(str(msg)),
        print_table=lambda table: printed.append(repr(table)),
    )


def _records(dm, monkeypatch, records):
    monkeypatch.setattr(dm, "_saved_figure_records", lambda: records)
    monkeypatch.setattr(V, "_legacy_plot_names", lambda _dm: [])


def test_one_figure_with_two_revisions_is_one_figure(dm, monkeypatch):
    _records(
        dm,
        monkeypatch,
        [
            {"figure_id": "f_a", "revision_id": "r_1"},
            {"figure_id": "f_a", "revision_id": "r_2"},
        ],
    )
    printed, output = _output()
    result = V.plots_list(SimpleNamespace(data_manager=dm), output)
    text = "\n".join(printed)
    assert "1 saved figure(s), 2 revision(s) in this workspace" in text
    assert (
        result == "No session plots; 1 saved figure(s), 2 revision(s) in the workspace"
    )


def test_two_figures_count_each_figure_once(dm, monkeypatch):
    _records(
        dm,
        monkeypatch,
        [
            {"figure_id": "f_a", "revision_id": "r_1"},
            {"figure_id": "f_b", "revision_id": "r_2"},
            {"figure_id": "f_b", "revision_id": "r_3"},
        ],
    )
    printed, output = _output()
    result = V.plots_list(SimpleNamespace(data_manager=dm), output)
    assert "2 saved figure(s), 3 revision(s) in this workspace" in "\n".join(printed)
    assert result.endswith("2 saved figure(s), 3 revision(s) in the workspace")


def test_no_records_and_no_legacy_keeps_the_plain_message(dm, monkeypatch):
    _records(dm, monkeypatch, [])
    printed, output = _output()
    assert (
        V.plots_list(SimpleNamespace(data_manager=dm), output) == "No plots available"
    )
    assert any("No plots generated yet" in m for m in printed)
