"""Prospective typed-refusal seams; these tests do NOT detect unsafe HDF bytes."""

import hashlib
from pathlib import Path
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import lobster.core.runtime.data_manager as module
from lobster.core.exceptions import HdfReadRefusal
from lobster.core.runtime.data_manager import DataManagerV2


def write_valid(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = ad.AnnData(
        np.array([[1, 2], [3, 4]], dtype=np.float32),
        obs=pd.DataFrame(index=["s_a", "s_b"]),
        var=pd.DataFrame(index=["g_a", "g_b"]),
    )
    value.write_h5ad(path)
    return value


def identity(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def state(dm):
    return (
        list((name, id(value)) for name, value in dm.modalities.items()),
        dict(dm._modality_sources),
        set(dm._modality_dirty),
        dict(dm._modality_spilled),
        id(dm._available_datasets),
        dict(dm._available_datasets),
        id(dm._available_datasets_cache),
        dm._scan_timestamp,
    )


def initialized(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path, auto_scan=False)
    dm.modalities["unrelated"] = ad.AnnData(np.ones((2, 2), dtype=np.float32))
    dm._modality_sources["unrelated"] = tmp_path / "original.h5ad"
    dm._modality_dirty.add("unrelated")
    dm._modality_spilled["unrelated_spill"] = tmp_path / "spill.h5ad"
    return dm


class SeamVisits(list):
    """Reader visits and direct Path mutations, not syscall-complete tracing."""

    def __init__(self):
        super().__init__()
        self.mutations = []


def reader_refusal(monkeypatch, seam, path):
    original = seam[0].__dict__[seam[1]]
    error = HdfReadRefusal()
    reached = SeamVisits()
    for method in ("unlink", "rename", "replace", "write_bytes", "write_text"):
        operation = getattr(Path, method)

        def mutation(candidate, *args, _method=method, _operation=operation, **kwargs):
            if candidate == path:
                reached.mutations.append(_method)
            return _operation(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, method, mutation)

    def read(candidate, *args, **kwargs):
        if Path(candidate) == path:
            reached.append(True)
            raise error
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(seam[0], seam[1], read)
    return error, reached


def assert_refused(operation, error, reached):
    with pytest.raises(HdfReadRefusal) as caught:
        operation()
    assert caught.value is error
    assert reached == [True], "reader_seam_not_reached"
    assert reached.mutations == [], "denied_source_mutated"


def test_constructor_denied_autosave_survives_then_real_reload(monkeypatch, tmp_path):
    path = tmp_path / "data" / "denied_autosave.h5ad"
    write_valid(path)
    before = identity(path)
    with monkeypatch.context() as scoped:
        error, reached = reader_refusal(scoped, (module, "_read_h5ad_smart"), path)
        assert_refused(
            lambda: DataManagerV2(workspace_path=tmp_path, auto_scan=True),
            error,
            reached,
        )
    assert identity(path) == before
    dm = DataManagerV2(workspace_path=tmp_path, auto_scan=True)
    assert "denied_autosave" in dm.modalities
    assert identity(path) == before
    # Successful metadata scans deliberately omit autosaves from the listing.
    assert "denied_autosave" not in dm.get_available_datasets(force_refresh=True)
    assert identity(path) == before


def test_repeated_autoload_denial_preserves_unrelated_state(monkeypatch, tmp_path):
    dm = initialized(tmp_path)
    path = dm.data_dir / "denied_autosave.h5ad"
    write_valid(path)
    before_file, before_state = identity(path), state(dm)
    for _ in range(2):
        with monkeypatch.context() as scoped:
            error, reached = reader_refusal(scoped, (module, "_read_h5ad_smart"), path)
            assert_refused(dm._auto_load_modalities, error, reached)
        assert identity(path) == before_file
        assert state(dm) == before_state
    dm._auto_load_modalities()
    assert "denied_autosave" in dm.modalities
    dm._scan_workspace()
    assert identity(path) == before_file


def test_autoload_resident_skip_is_not_refusal_credit(monkeypatch, tmp_path):
    dm = initialized(tmp_path)
    path = dm.data_dir / "unrelated.h5ad"
    write_valid(path)
    with monkeypatch.context() as scoped:
        _, reached = reader_refusal(scoped, (module, "_read_h5ad_smart"), path)
        dm._auto_load_modalities()
        assert reached == []


@pytest.mark.parametrize("trigger", ["forced", "getter"])
def test_scan_denial_preserves_autosave_and_cache(monkeypatch, tmp_path, trigger):
    import h5py

    dm = initialized(tmp_path)
    path = dm.data_dir / "denied_autosave.h5ad"
    write_valid(path)
    if trigger == "forced":
        dm._available_datasets = {"old": {"shape": (2, 2)}}
        dm._available_datasets_cache = {"cached": {"shape": (2, 2)}}
        dm._scan_timestamp = 123.0

        def operation():
            return dm.get_available_datasets(force_refresh=True)

    else:
        dm._auto_scan_on_access = True
        dm._available_datasets = {}
        dm._available_datasets_cache = None

        def operation():
            return dm.available_datasets

    before_file, before_state = identity(path), state(dm)
    for _ in range(2):
        with monkeypatch.context() as scoped:
            error, reached = reader_refusal(scoped, (h5py, "File"), path)
            assert_refused(operation, error, reached)
        assert identity(path) == before_file
        assert state(dm) == before_state
    assert "denied_autosave" not in dm.get_available_datasets(force_refresh=True)
    dm._auto_load_modalities()
    assert "denied_autosave" in dm.modalities
    assert identity(path) == before_file


@pytest.mark.parametrize("operation", ["workspace", "forced_dataset"])
def test_same_name_backed_resident_not_replaced_or_closed(
    monkeypatch, tmp_path, operation
):
    dm = initialized(tmp_path)
    old_path = tmp_path / "resident.h5ad"
    write_valid(old_path)
    old = ad.read_h5ad(old_path, backed="r")
    dm.modalities["denied"] = old
    dm._modality_sources["denied"] = old_path
    dm._modality_dirty.add("denied")
    directory = tmp_path / "saved" if operation == "workspace" else dm.data_dir
    path = directory / "denied.h5ad"
    write_valid(path)
    dm._available_datasets = {"denied": {"path": str(path), "size_mb": 1}}
    before_file, before_state = identity(path), state(dm)
    with monkeypatch.context() as scoped:
        error, reached = reader_refusal(scoped, (module, "_read_h5ad_smart"), path)

        def call():
            return (
                dm.load_workspace("saved")
                if operation == "workspace"
                else dm.load_dataset("denied", force_reload=True)
            )

        assert_refused(call, error, reached)
    assert identity(path) == before_file
    assert state(dm) == before_state
    assert dm.modalities["denied"] is old
    assert old.file.is_open
    np.testing.assert_array_equal(old.X[:], [[1, 2], [3, 4]])
    old.file.close()


@pytest.mark.parametrize("operation", ["autoload", "workspace"])
def test_mixed_file_refusal_is_terminal_but_not_rollback(
    monkeypatch, tmp_path, operation
):
    dm = initialized(tmp_path)
    directory = dm.data_dir if operation == "autoload" else tmp_path / "saved"
    first = directory / "first.h5ad"
    denied = directory / "second.h5ad"
    write_valid(first)
    write_valid(denied)
    before_file = identity(denied)
    old = dm.modalities["unrelated"]
    with monkeypatch.context() as scoped:
        # Pin only enumeration order; actual readers and handlers run unchanged.
        original_glob = Path.glob

        def glob(path, pattern):
            if path == directory and pattern == "*.h5ad":
                return iter([first, denied])
            return original_glob(path, pattern)

        scoped.setattr(Path, "glob", glob)
        error, reached = reader_refusal(scoped, (module, "_read_h5ad_smart"), denied)
        assert_refused(
            (
                dm._auto_load_modalities
                if operation == "autoload"
                else lambda: dm.load_workspace("saved")
            ),
            error,
            reached,
        )
    assert "first" in dm.modalities  # Explicit per-artifact, not staging/rollback.
    assert "second" not in dm.modalities
    assert dm.modalities["unrelated"] is old
    assert identity(denied) == before_file


def test_tabular_autoload_closed_refusal_is_not_swallowed(monkeypatch, tmp_path):
    dm = initialized(tmp_path)
    path = dm.data_dir / "denied.csv"
    path.write_text("gene,sample\ng,1\n")
    error = HdfReadRefusal()
    read = Mock(side_effect=error)
    adapter = Mock()
    adapter.from_source = read
    monkeypatch.setattr(
        dm, "_resolve_adapter_for_file", lambda candidate: ("custom", adapter)
    )
    before_file, before_state = identity(path), state(dm)
    with pytest.raises(HdfReadRefusal) as caught:
        dm._auto_load_modalities()
    assert caught.value is error
    read.assert_called_once_with(path)
    assert identity(path) == before_file
    assert state(dm) == before_state


def test_valid_restore_scan_and_ordinary_corruption_controls(tmp_path):
    dm = initialized(tmp_path)
    saved = tmp_path / "saved" / "valid.h5ad"
    original = write_valid(saved)
    result = dm.load_workspace("saved")
    assert result["loaded"] == ["valid"]
    assert result["errors"] == []
    np.testing.assert_array_equal(dm.modalities["valid"].X, original.X)
    autosave = dm.data_dir / "valid_autosave.h5ad"
    write_valid(autosave)
    dm._auto_load_modalities()
    assert "valid_autosave" in dm.modalities
    assert "valid_autosave" not in dm._scan_workspace()
    assert autosave.exists()
    malformed = dm.data_dir / "malformed_autosave.h5ad"
    malformed.write_bytes(b"not hdf")
    dm._auto_load_modalities()
    assert not malformed.exists()  # Legacy ordinary-corruption policy is retained.
    assert dm.load_dataset("missing") is False
