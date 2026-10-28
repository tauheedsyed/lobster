"""Reads of files the code child leaves behind are bounded and never block.

Parent-side only: no child process is launched. ``subprocess.run`` (or the whole child
step) is replaced by a fake that leaves the hostile entries an exited child could leave:
a FIFO, a symlink, a device, an over-size file. Every call runs under an outer watchdog,
so a read that blocks fails the test instead of hanging it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading

import anndata
import numpy as np
import pytest

from lobster.core.figures.records import read_submitted_figures
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.execution import custom_code_execution_service as svc

WATCHDOG_SECONDS = 5


def _release_fifos(directory):
    """Open every FIFO below ``directory`` for writing, so a blocked reader gets EOF."""
    for root, _, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            try:
                os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))
            except OSError:
                pass


def _within(directory, fn, *args, **kwargs):
    box = {}

    def run():
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(WATCHDOG_SECONDS)
    if worker.is_alive():
        _release_fifos(directory)
        worker.join(WATCHDOG_SECONDS)
        pytest.fail("a post-child read blocked")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _envelope(**over):
    env = {"schema": svc.RESULT_SCHEMA, "has_result": False, "result": None}
    env.update({"type": None}, **over)
    return json.dumps(env).encode()


# --------------------------------------------------------------- result envelope
def test_a_fifo_result_is_unreadable_without_blocking(tmp_path):
    path = tmp_path / "result.json"
    os.mkfifo(path)
    assert _within(tmp_path, svc._read_result_envelope, path) == (
        None,
        svc.COMPLETION_RESULT_UNREADABLE,
    )


@pytest.mark.parametrize("target", ["regular", "device"])
def test_a_linked_result_is_unreadable(tmp_path, target):
    real = tmp_path / "elsewhere.json"
    real.write_bytes(_envelope())
    path = tmp_path / "result.json"
    os.symlink(real if target == "regular" else os.devnull, path)
    assert _within(tmp_path, svc._read_result_envelope, path) == (
        None,
        svc.COMPLETION_RESULT_UNREADABLE,
    )


def test_a_result_over_the_cap_is_unreadable_and_at_the_cap_is_read(tmp_path):
    body = _envelope(has_result=True, result="r", type="str")
    at_cap = body + b" " * (svc.MAX_RESULT_BYTES - len(body))
    path = tmp_path / "result.json"
    path.write_bytes(at_cap)
    assert svc._read_result_envelope(path) == ("r", svc.COMPLETION_COMPLETE)
    path.write_bytes(at_cap + b" ")
    assert _within(tmp_path, svc._read_result_envelope, path) == (
        None,
        svc.COMPLETION_RESULT_UNREADABLE,
    )


def test_a_deeply_nested_result_is_unreadable_and_never_raises(tmp_path):
    path = tmp_path / "result.json"
    path.write_bytes(b"[" * 200_000 + b"]" * 200_000)
    assert svc._read_result_envelope(path) == (None, svc.COMPLETION_RESULT_UNREADABLE)


def test_a_missing_result_is_still_missing(tmp_path):
    assert svc._read_result_envelope(tmp_path / "absent.json") == (
        None,
        svc.COMPLETION_RESULT_MISSING,
    )


# ------------------------------------------------- the child step, with a fake run
_NONCE = re.compile(r"\.execution_complete'\)\.write_text\('([0-9a-f]{32})'\)")


def _run_with(tmp_path, monkeypatch, leave, returncode=0, outcome=None):
    """Call ``_execute_in_namespace`` with a fake child (exit ``returncode``) that runs
    ``leave``; ``outcome`` (a list) receives the call's return value."""
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    service = svc.CustomCodeExecutionService(dm)
    ipc = tmp_path / "ws" / "cache" / "execution" / "x1"

    def fake_run(argv, **kwargs):
        script = open(argv[1], encoding="utf-8").read()
        leave(ipc, _NONCE.search(script).group(1))
        return subprocess.CompletedProcess(argv, returncode, stdout="", stderr="")

    monkeypatch.setattr(svc.subprocess, "run", fake_run)
    context = {
        "workspace_path": tmp_path / "ws",
        "exec_cache_path": ipc,
        "load_workspace_files": False,
    }
    out = _within(tmp_path, service._execute_in_namespace, "x = 1", context)
    if outcome is not None:
        outcome.append(out)
    return context["completion"]


def _good_result(ipc):
    (ipc / ".execution_result.json").write_bytes(_envelope())


def _good_marker(ipc, nonce):
    (ipc / ".execution_complete").write_text(nonce)


def test_a_normal_exit_is_complete(tmp_path, monkeypatch):
    def leave(ipc, nonce):
        _good_result(ipc)
        _good_marker(ipc, nonce)

    assert _run_with(tmp_path, monkeypatch, leave) == svc.COMPLETION_COMPLETE


def test_a_fifo_marker_is_unconfirmed_without_blocking(tmp_path, monkeypatch):
    def leave(ipc, nonce):
        _good_result(ipc)
        os.mkfifo(ipc / ".execution_complete")

    assert _run_with(tmp_path, monkeypatch, leave) == svc.COMPLETION_UNCONFIRMED


@pytest.mark.parametrize("target", ["regular", "device"])
def test_a_linked_marker_is_unconfirmed(tmp_path, monkeypatch, target):
    def leave(ipc, nonce):
        _good_result(ipc)
        real = tmp_path / "marker_elsewhere"
        real.write_text(nonce)
        os.symlink(
            real if target == "regular" else os.devnull, ipc / ".execution_complete"
        )

    assert _run_with(tmp_path, monkeypatch, leave) == svc.COMPLETION_UNCONFIRMED


def test_a_marker_with_trailing_bytes_is_unconfirmed(tmp_path, monkeypatch):
    def leave(ipc, nonce):
        _good_result(ipc)
        (ipc / ".execution_complete").write_text(nonce + "x" * 100)

    assert _run_with(tmp_path, monkeypatch, leave) == svc.COMPLETION_UNCONFIRMED


def test_a_fifo_result_after_exit_zero_is_unreadable(tmp_path, monkeypatch):
    def leave(ipc, nonce):
        os.mkfifo(ipc / ".execution_result.json")
        _good_marker(ipc, nonce)

    assert _run_with(tmp_path, monkeypatch, leave) == svc.COMPLETION_RESULT_UNREADABLE


# ------------------------------------------------------------ unsubmitted marker
def _candidates(tmp_path):
    root = tmp_path / "candidates"
    root.mkdir()
    return root


def test_a_fifo_unsubmitted_marker_is_invalid_without_blocking(tmp_path):
    os.mkfifo(_candidates(tmp_path) / "unsubmitted.json")
    assert _within(tmp_path, read_submitted_figures, tmp_path) == (
        [],
        [],
        "figure_submission_invalid",
    )


@pytest.mark.parametrize("target", ["regular", "device", "dangling"])
def test_a_linked_unsubmitted_marker_is_invalid(tmp_path, target):
    root = _candidates(tmp_path)
    real = tmp_path / "names.json"
    if target == "regular":
        real.write_bytes(b'["fig"]')
    os.symlink(os.devnull if target == "device" else real, root / "unsubmitted.json")
    assert _within(tmp_path, read_submitted_figures, tmp_path) == (
        [],
        [],
        "figure_submission_invalid",
    )


def test_a_regular_unsubmitted_marker_is_still_read(tmp_path):
    (_candidates(tmp_path) / "unsubmitted.json").write_bytes(b'["fig", "fig2"]')
    assert read_submitted_figures(tmp_path) == ([], ["fig", "fig2"], None)


@pytest.mark.parametrize("name", ["figure.json", "candidate.json"])
def test_a_fifo_candidate_file_is_invalid_without_blocking(tmp_path, name):
    cdir = _candidates(tmp_path) / "0"
    cdir.mkdir()
    for other in ("figure.json", "candidate.json"):
        if other != name:
            (cdir / other).write_bytes(b"{}")
    os.mkfifo(cdir / name)
    assert _within(tmp_path, read_submitted_figures, tmp_path) == (
        [],
        [],
        "figure_submission_invalid",
    )


CANDIDATE_FILES = ["figure.json", "candidate.json", "source.json", "config.json"]


def _candidate(tmp_path):
    """One complete submitted candidate (figure, metadata, source and config)."""
    cdir = _candidates(tmp_path) / "0"
    cdir.mkdir()
    meta = {"index": 0, "label": "L", "has_source": True, "has_config": True}
    (cdir / "figure.json").write_bytes(b'{"data": [], "layout": {}}')
    (cdir / "candidate.json").write_bytes(json.dumps(meta).encode())
    (cdir / "source.json").write_bytes(b'{"rows": [], "declared": {}}')
    (cdir / "config.json").write_bytes(b"{}")
    return cdir


def test_a_complete_candidate_is_still_read(tmp_path):
    _candidate(tmp_path)
    [figure], unsubmitted, error = read_submitted_figures(tmp_path)
    assert error is None and unsubmitted == []
    assert figure.figure_bytes == b'{"data": [], "layout": {}}'
    assert figure.config_bytes == b"{}" and figure.source is not None


@pytest.mark.parametrize("kind", ["fifo", "link_regular", "link_device"])
@pytest.mark.parametrize("name", CANDIDATE_FILES)
def test_a_non_regular_candidate_file_is_invalid_without_blocking(tmp_path, name, kind):
    path = _candidate(tmp_path) / name
    if kind == "fifo":
        path.unlink()
        os.mkfifo(path)
    else:
        copy = tmp_path / f"copy_{name}"
        copy.write_bytes(path.read_bytes())
        path.unlink()
        os.symlink(copy if kind == "link_regular" else os.devnull, path)
    assert _within(tmp_path, read_submitted_figures, tmp_path) == (
        [],
        [],
        "figure_submission_invalid",
    )


def test_every_candidate_file_goes_through_the_bounded_reader(tmp_path, monkeypatch):
    from lobster.utils import ipc_read

    seen = []
    real = ipc_read.read_regular

    def spy(path, limit):
        seen.append(os.path.basename(path))
        return real(path, limit)

    monkeypatch.setattr(ipc_read, "read_regular", spy)
    _candidate(tmp_path)
    figures, _, error = read_submitted_figures(tmp_path)
    assert error is None and len(figures) == 1
    assert set(CANDIDATE_FILES) <= set(seen)


# ------------------------------------------------------ modified data readback
def _dm_with_modality(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality(name="m", adata=anndata.AnnData(np.ones((2, 2))), step="t")
    return dm


def _execute_leaving(tmp_path, monkeypatch, leave):
    dm = _dm_with_modality(tmp_path)
    service = svc.CustomCodeExecutionService(dm)

    def fake_child(code, context):
        leave(context["exec_cache_path"] / ".modified_adata.h5ad")
        context["completion"] = svc.COMPLETION_COMPLETE
        return None, "", "", None

    monkeypatch.setattr(service, "_execute_in_namespace", fake_child)
    _, stats, _ = _within(
        tmp_path,
        service.execute,
        "x = 1",
        modality_name="m",
        load_workspace_files=False,
    )
    return dm, stats


def _h5ad(path):
    anndata.AnnData(np.zeros((2, 2))).write_h5ad(path)


def test_a_regular_modified_file_is_still_stored(tmp_path, monkeypatch):
    dm, stats = _execute_leaving(tmp_path, monkeypatch, _h5ad)
    assert "m_custom" in dm.list_modalities()
    assert stats["write_back_error"] is None


def test_a_fifo_modified_file_is_refused_without_opening_it(tmp_path, monkeypatch):
    # The HDF5 reader opens its file without releasing the interpreter lock, so a blocked
    # open would freeze the whole test process (no watchdog could run). A writer is held
    # open here so that an open of the FIFO cannot block; the refusal must not depend on it.
    writers = []

    def leave(path):
        os.mkfifo(path)
        writers.append(os.open(path, os.O_RDWR | os.O_NONBLOCK))

    try:
        dm, stats = _execute_leaving(tmp_path, monkeypatch, leave)
    finally:
        for fd in writers:
            os.close(fd)
    assert "m_custom" not in dm.list_modalities()
    assert stats["write_back_error"] == svc.MODIFIED_DATA_NOT_REGULAR


@pytest.mark.parametrize("target", ["regular", "device"])
def test_a_linked_modified_file_is_refused(tmp_path, monkeypatch, target):
    real = tmp_path / "elsewhere.h5ad"
    _h5ad(real)

    def leave(path):
        os.symlink(real if target == "regular" else os.devnull, path)

    dm, stats = _execute_leaving(tmp_path, monkeypatch, leave)
    assert "m_custom" not in dm.list_modalities()
    assert stats["write_back_error"] == svc.MODIFIED_DATA_NOT_REGULAR


# ------------------------------------------- a refusal survives the cleanup after it
@pytest.mark.parametrize(
    "name,expected",
    [
        (".execution_result.json", svc.COMPLETION_RESULT_UNREADABLE),
        (".execution_complete", svc.COMPLETION_UNCONFIRMED),
    ],
)
def test_a_directory_left_at_an_ipc_path_keeps_the_refusal(
    tmp_path, monkeypatch, name, expected
):
    def leave(ipc, nonce):
        _good_result(ipc)
        _good_marker(ipc, nonce)
        target = ipc / name
        target.unlink()
        target.mkdir()

    assert _run_with(tmp_path, monkeypatch, leave) == expected


# --------------------------------------------------- the actual open sites refuse
def _swap_to_fifo_at_open(monkeypatch, name, keep):
    """Turn ``name`` into a FIFO between the reader's check and its open."""
    real_open = os.open

    def swapping_open(path, flags, *args):
        if os.path.basename(os.fspath(path)) == name and not os.path.exists(keep):
            os.rename(path, keep)
            os.mkfifo(path)
        return real_open(path, flags, *args)

    monkeypatch.setattr(os, "open", swapping_open)


@pytest.mark.parametrize("name", CANDIDATE_FILES)
def test_a_candidate_file_turned_into_a_fifo_at_open_is_refused(
    tmp_path, monkeypatch, name
):
    _candidate(tmp_path)
    _swap_to_fifo_at_open(monkeypatch, name, tmp_path / "kept")
    assert _within(tmp_path, read_submitted_figures, tmp_path) == (
        [],
        [],
        "figure_submission_invalid",
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        (".execution_result.json", svc.COMPLETION_RESULT_UNREADABLE),
        (".execution_complete", svc.COMPLETION_UNCONFIRMED),
    ],
)
def test_an_ipc_file_turned_into_a_fifo_at_open_is_refused(
    tmp_path, monkeypatch, name, expected
):
    def leave(ipc, nonce):
        _good_result(ipc)
        _good_marker(ipc, nonce)
        _swap_to_fifo_at_open(monkeypatch, name, tmp_path / "kept")

    assert _run_with(tmp_path, monkeypatch, leave) == expected


# ------------------------------------------------------- closed codes, never text
@pytest.mark.parametrize("declared", [[1, 2], [["a", 1]], "abc", 5, False, None])
def test_a_non_mapping_source_declaration_is_invalid(tmp_path, declared):
    cdir = _candidate(tmp_path)
    payload = {"rows": [], "declared": declared}
    (cdir / "source.json").write_bytes(json.dumps(payload).encode())
    assert read_submitted_figures(tmp_path) == ([], [], "figure_submission_invalid")


@pytest.mark.parametrize("payload", [b"[]", b'"rows"', b"7", b"null"])
def test_a_non_object_source_is_invalid(tmp_path, payload):
    (_candidate(tmp_path) / "source.json").write_bytes(payload)
    assert read_submitted_figures(tmp_path) == ([], [], "figure_submission_invalid")


def test_a_source_without_a_declaration_is_still_read(tmp_path):
    (_candidate(tmp_path) / "source.json").write_bytes(b'{"rows": []}')
    [figure], _, error = read_submitted_figures(tmp_path)
    assert error is None and figure.source.declared == {}


def test_an_unreadable_modified_file_is_a_closed_code(tmp_path, monkeypatch):
    def leave(path):
        path.write_bytes(b"not an hdf5 file " + str(tmp_path).encode())

    dm, stats = _execute_leaving(tmp_path, monkeypatch, leave)
    assert "m_custom" not in dm.list_modalities()
    assert stats["write_back_error"] == svc.MODIFIED_DATA_UNREADABLE


# ------------------------------------------------------ failed-script archival
def _failed_run(tmp_path, monkeypatch):
    outcome = []
    _run_with(tmp_path, monkeypatch, lambda ipc, nonce: None, 1, outcome)
    [(_, _, _, error)] = outcome
    assert error is not None  # the failed run is still reported, never replaced
    return tmp_path / "ws" / ".failed_scripts"


def test_a_failed_script_is_still_archived(tmp_path, monkeypatch):
    archive = _failed_run(tmp_path, monkeypatch)
    [archived] = list(archive.iterdir())
    assert archived.name.startswith(".script_") and archived.is_file()


def test_a_failed_script_is_never_copied(tmp_path, monkeypatch):
    import errno
    import shutil

    real_rename = os.rename

    def cross_device(src, dst, *a, **k):
        if os.path.basename(os.fspath(src)).startswith(".script_"):
            raise OSError(errno.EXDEV, "cross-device link")
        return real_rename(src, dst, *a, **k)

    def no_copy(*a, **k):
        raise AssertionError("the failed script was copied")

    monkeypatch.setattr(os, "rename", cross_device)
    monkeypatch.setattr(shutil, "copyfile", no_copy)
    archive = _failed_run(tmp_path, monkeypatch)
    assert not archive.exists() or not any(archive.iterdir())
    ipc = tmp_path / "ws" / "cache" / "execution" / "x1"
    assert not any(p.name.startswith(".script_") for p in ipc.iterdir())


def test_a_linked_archive_directory_is_not_used(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "ws").mkdir()
    os.symlink(outside, tmp_path / "ws" / ".failed_scripts")
    _failed_run(tmp_path, monkeypatch)
    assert list(outside.iterdir()) == []
