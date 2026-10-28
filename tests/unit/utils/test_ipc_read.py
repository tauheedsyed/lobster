"""Bounded reader for files another process left behind."""

from __future__ import annotations

import os
import threading

import pytest

from lobster.utils.ipc_read import (
    NOT_REGULAR,
    TOO_LARGE,
    RefusedRead,
    read_regular,
    regular_identity,
)


def _within(seconds, fn, *args):
    """Run ``fn`` under an outer watchdog; a blocked call fails instead of hanging."""
    box = {}

    def run():
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # reported to the test thread
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        for path in args:
            if isinstance(path, os.PathLike) and os.path.exists(path):
                try:  # release a reader blocked on a FIFO
                    os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))
                except OSError:
                    pass
        worker.join(5)
        pytest.fail("the read blocked")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _refused(path, limit=1024):
    with pytest.raises(RefusedRead) as caught:
        _within(5, read_regular, path, limit)
    return caught.value.reason


def test_a_regular_file_within_the_cap_is_read_exactly(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"x" * 1024)
    assert _within(5, read_regular, path, 1024) == b"x" * 1024
    assert regular_identity(path) == (path.stat().st_dev, path.stat().st_ino)


def test_an_empty_file_is_read(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"")
    assert read_regular(path, 0) == b""


def test_an_absent_file_is_none(tmp_path):
    assert read_regular(tmp_path / "missing", 10) is None
    assert regular_identity(tmp_path / "missing") is None


def test_one_byte_over_the_cap_is_refused(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"x" * 1025)
    assert _refused(path) == TOO_LARGE


def test_a_fifo_is_refused_without_blocking(tmp_path):
    path = tmp_path / "fifo"
    os.mkfifo(path)
    assert _refused(path) == NOT_REGULAR
    with pytest.raises(RefusedRead):
        regular_identity(path)


@pytest.mark.parametrize("target", ["regular", "fifo", "device", "dangling"])
def test_a_final_symlink_is_refused(tmp_path, target):
    real = tmp_path / "real"
    if target == "regular":
        real.write_bytes(b"{}")
    elif target == "fifo":
        os.mkfifo(real)
    elif target == "device":
        real = os.devnull
    link = tmp_path / "link"
    os.symlink(real, link)
    assert _refused(link) == NOT_REGULAR
    with pytest.raises(RefusedRead):
        regular_identity(link)


def test_a_device_and_a_directory_are_refused(tmp_path):
    assert _refused(os.devnull) == NOT_REGULAR
    assert _refused(tmp_path) == NOT_REGULAR


def test_a_file_replaced_after_the_check_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "f"
    path.write_bytes(b"one")
    real_open = os.open

    def swap_then_open(p, flags, *a):
        if os.fspath(p) == os.fspath(path):
            os.rename(path, tmp_path / "kept")  # keeps the first inode allocated
            path.write_bytes(b"two")  # a different file at the same name
        return real_open(p, flags, *a)

    monkeypatch.setattr(os, "open", swap_then_open)
    with pytest.raises(RefusedRead) as caught:
        read_regular(path, 10)
    assert caught.value.reason == NOT_REGULAR


def test_the_reason_never_carries_the_path(tmp_path):
    path = tmp_path / "secret_name"
    os.mkfifo(path)
    with pytest.raises(RefusedRead) as caught:
        read_regular(path, 10)
    assert "secret_name" not in str(caught.value)
    assert str(caught.value) == caught.value.reason
