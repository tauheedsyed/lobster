"""Callers of auto-save never report a refused modality as saved.

``auto_save_state`` returns a ``Not saved: <n> modalities refused (managed figure
folder)`` entry when a modality's target is inside the managed figure folder (that
modality stays dirty or spilled). The CLI ``/save``, the TUI save command and the
cache loader each show that entry instead of an all-saved message.

Real ``DataManagerV2`` and a revision committed through the real store; only the UI
boundary (output adapter, ``app.notify``) and the cache loader's archive reading are
recorded or faked.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest

from lobster.core.runtime.data_manager import DataManagerV2, estimate_anndata_bytes
from tests.unit.export_guard_fixtures import alias, saved_revision, workspace_tree

NOT_SAVED = "Not saved: 1 modalities refused (managed figure folder)"
SHAPE = (50, 40)


def _adata(seed=0, shape=(6, 5)):
    rng = np.random.default_rng(seed)
    adata = ad.AnnData(X=rng.random(shape))
    adata.obs_names = [f"cell{i}" for i in range(shape[0])]
    adata.var_names = [f"gene{i}" for i in range(shape[1])]
    return adata


@pytest.fixture
def session(tmp_path, monkeypatch):
    def make(budget_of=None):
        cwd = tmp_path / "cwd"
        cwd.mkdir(exist_ok=True)
        monkeypatch.chdir(cwd)
        if budget_of is None:
            monkeypatch.delenv("LOBSTER_MODALITY_MEMORY_BUDGET_GB", raising=False)
        else:
            size = estimate_anndata_bytes(_adata(0, SHAPE))
            monkeypatch.setenv(
                "LOBSTER_MODALITY_MEMORY_BUDGET_GB", str(budget_of * size / 1024**3)
            )
        dm = DataManagerV2(workspace_path=tmp_path / "ws")
        rev = saved_revision(dm.workspace_path)
        return SimpleNamespace(dm=dm, ws=Path(dm.workspace_path), rev=rev)

    return make


def _rev_file(s):
    return s.rev.directory / sorted(s.rev.files)[0]


def _refused(s, name):
    """``data/<name>_autosave.h5ad`` points into the revision."""
    alias(s.dm.data_dir / f"{name}_autosave.h5ad", _rev_file(s))


class _Output:
    """The CLI output adapter, recording the rendered blocks."""

    def __init__(self):
        self.bodies = []

    def render_blocks(self, blocks):
        self.bodies.extend(b.data.get("body") for b in blocks)


def _command_save(s):
    from lobster.cli_internal.commands.heavy.slash_commands import _command_save

    output = _Output()
    returned = _command_save(SimpleNamespace(data_manager=s.dm), output)
    return returned, output.bodies


# ---------------------------------------------------------------- CLI /save


def test_cli_save_reports_a_refused_modality_as_not_saved(session):
    s = session()
    s.dm.store_modality(name="pbmc", adata=_adata(0))
    s.dm.store_modality(name="ordinary", adata=_adata(1))
    _refused(s, "pbmc")
    returned, bodies = _command_save(s)
    assert returned == f"Saved 2 items, skipped 0 unchanged; {NOT_SAVED}"
    assert bodies == [
        "Saved: Modality 'ordinary': ordinary_autosave.h5ad",
        "Saved: Processing log",
        NOT_SAVED,
    ]
    assert "pbmc" in s.dm._modality_dirty and s.rev.unchanged()


def test_cli_save_reaches_a_spilled_only_session(session):
    s = session(budget_of=1.5)
    s.dm.store_modality(name="pbmc", adata=_adata(0, SHAPE))
    s.dm.store_modality(name="other", adata=_adata(1, SHAPE))  # spills pbmc
    s.dm.remove_modality("other")
    assert s.dm.modalities == {} and "pbmc" in s.dm._modality_spilled
    _refused(s, "pbmc")
    spilled = Path(s.dm._modality_spilled["pbmc"])
    state = (
        dict(s.dm._modality_spilled),
        set(s.dm._modality_dirty),
        workspace_tree(spilled),
    )
    returned, bodies = _command_save(s)
    assert returned == f"Saved 1 items, skipped 0 unchanged; {NOT_SAVED}"
    assert bodies == ["Saved: Processing log", NOT_SAVED]
    assert (
        dict(s.dm._modality_spilled),
        set(s.dm._modality_dirty),
        workspace_tree(spilled),
    ) == state
    assert s.rev.unchanged()


def test_cli_save_without_data_is_unchanged(session):
    s = session()
    assert _command_save(s) == (
        "No data to save",
        ["Nothing to save (no data loaded)"],
    )


def test_cli_ordinary_save_is_unchanged(session):
    s = session()
    s.dm.store_modality(name="ordinary", adata=_adata(1))
    returned, bodies = _command_save(s)
    assert returned == "Saved 2 items, skipped 0 unchanged"
    assert bodies == [
        "Saved: Modality 'ordinary': ordinary_autosave.h5ad",
        "Saved: Processing log",
    ]


# ---------------------------------------------------------------- TUI save


def _tui_save(s):
    from lobster.ui.commands import LobsterCommands

    notes = []
    app = SimpleNamespace(notify=lambda message, **kw: notes.append((message, kw)))
    provider = SimpleNamespace(client=SimpleNamespace(data_manager=s.dm), app=app)
    asyncio.run(LobsterCommands._cmd_save(provider))
    return notes


def test_tui_save_reports_a_partial_save(session):
    pytest.importorskip("textual", reason="the TUI needs the classic-tui extra")
    s = session()
    s.dm.store_modality(name="pbmc", adata=_adata(0))
    _refused(s, "pbmc")
    assert _tui_save(s) == [
        (f"Session partly saved: {NOT_SAVED}", {"severity": "warning"})
    ]
    assert s.rev.unchanged()


def test_tui_ordinary_save_is_unchanged(session):
    pytest.importorskip("textual", reason="the TUI needs the classic-tui extra")
    s = session()
    s.dm.store_modality(name="ordinary", adata=_adata(1))
    assert _tui_save(s) == [("Session saved (2 items)", {"timeout": 2})]


# ------------------------------------------------- cache loader (auto-save after)


@pytest.fixture
def cached_sample(monkeypatch, tmp_path):
    """One cached sample, read by fakes at the archive boundary; it is stored for
    real and the auto-save that follows is real."""
    from lobster.core import client as client_module

    class _Cache:
        def __init__(self, workspace):
            pass

        def load_from_cache(self, cache_id, pattern, limit):
            return [tmp_path / "pbmc.tar.gz"]

    class _Extractor:
        def extract_to_temp(self, archive, prefix):
            return tmp_path

        def cleanup(self):
            pass

    class _Detector:
        def detect_content_type(self, directory):
            return client_module.ArchiveContentType.TEN_X_MTX

    monkeypatch.setattr(client_module, "_get_extraction_cache_manager", lambda: _Cache)
    monkeypatch.setattr(client_module, "ArchiveExtractor", _Extractor)
    monkeypatch.setattr(client_module, "ContentDetector", _Detector)

    def load(s):
        def load_10x(directory, name):
            s.dm.store_modality(name=name, adata=_adata(0))
            return {"success": True, "modality_name": name}

        owner = SimpleNamespace(
            workspace_path=s.ws,
            data_manager=s.dm,
            _clean_archive_name=lambda archive: "pbmc",
            _load_10x_from_directory=load_10x,
        )
        return client_module.AgentClient.load_from_cache(owner, "c1", "pbmc")

    return load


def _client_records(caplog):
    return [
        (r.levelname, r.getMessage())
        for r in caplog.records
        if r.name == "lobster.core.client" and "uto-save" in r.getMessage()
    ]


def test_cache_load_reports_an_incomplete_auto_save(session, cached_sample, caplog):
    s = session()
    _refused(s, "pbmc")
    with caplog.at_level(logging.INFO):
        assert cached_sample(s)["success"] is True
    assert _client_records(caplog) == [
        ("WARNING", f"Auto-save incomplete: {NOT_SAVED}")
    ]
    assert "pbmc" in s.dm._modality_dirty and s.rev.unchanged()


def test_cache_load_ordinary_auto_save_is_unchanged(session, cached_sample, caplog):
    s = session()
    with caplog.at_level(logging.INFO):
        assert cached_sample(s)["success"] is True
    assert _client_records(caplog) == [
        ("INFO", "Auto-saved 1 loaded modalities to workspace")
    ]
