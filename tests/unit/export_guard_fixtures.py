"""Shared fixtures for tests of the export guard (figure contract §1).

A revision saved through the real store (not a hand-made folder), a whole-tree
snapshot that follows nothing, and a link helper. Owned once, used by every export
lane: tests read it, they do not change it.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from lobster.core.figures.records import STATUS_AVAILABLE
from lobster.core.figures.saver import TrustedSaver
from lobster.core.figures.store import LocalFigureStore
from tests.unit.core.figures.test_store_and_saver import _cand, _info


@dataclass(frozen=True)
class SavedRevision:
    """A committed figure revision: its directory and every file's bytes."""

    ref: str
    directory: Path
    files: Dict[str, bytes]

    def unchanged(self) -> bool:
        """True when the revision still holds exactly the committed files and bytes."""
        current = {
            p.name: p.read_bytes()
            for p in self.directory.iterdir()
            if p.is_file() and not p.is_symlink()
        }
        return current == self.files


def saved_revision(workspace: Path, session_id: str = "s1") -> SavedRevision:
    """Commit one figure through ``LocalFigureStore`` + ``TrustedSaver`` in ``workspace``."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    store = LocalFigureStore(workspace, session_id=session_id)
    [res] = TrustedSaver(store).commit_attempt(_info(), [_cand()], succeeded=True)
    assert res.status == STATUS_AVAILABLE, res.reason
    path = Path(res.path)
    directory = path if path.is_dir() else path.parent
    if not directory.is_absolute():
        directory = workspace / directory
    files = {
        p.name: p.read_bytes()
        for p in directory.iterdir()
        if p.is_file() and not p.is_symlink()
    }
    assert files, "the committed revision holds no file"
    return SavedRevision(ref=str(res.ref), directory=directory, files=files)


def _entry(path: Path) -> Tuple:
    """One entry by lstat (never followed), with its modification time.

    ``("link", target, mtime_ns)``, ``("file", sha256, size, mtime_ns)``,
    ``("dir", mtime_ns)`` or ``("other", mode, mtime_ns)``. The access time is not
    recorded (reading a file changes it).
    """
    st = os.lstat(path)
    mode = st.st_mode
    if stat.S_ISLNK(mode):
        return ("link", os.readlink(path), st.st_mtime_ns)
    if stat.S_ISREG(mode):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return ("file", digest, st.st_size, st.st_mtime_ns)
    if stat.S_ISDIR(mode):
        return ("dir", st.st_mtime_ns)
    return ("other", f"{stat.S_IFMT(mode):o}", st.st_mtime_ns)


def _raise(error: OSError) -> None:
    raise error


def workspace_tree(*roots: Path) -> Dict[str, Tuple]:
    """Every entry under each root (lstat, nothing followed), see :func:`_entry`.

    Keys start with the root's full absolute spelling (not resolved), so two roots with
    one basename never share a key. Each root is itself recorded first, by lstat: a
    missing root is ``("absent",)``, a root that is a link is recorded with its target and
    not entered, and only a real directory is walked. A walk error is raised, never
    skipped. A same-byte rewrite shows as a new modification time.
    Ancestors of a root are looked up by the system as usual.
    """
    out: Dict[str, Tuple] = {}
    for root in roots:
        root = Path(root).absolute()
        name = str(root)
        try:
            out[name] = _entry(root)
        except FileNotFoundError:
            out[name] = ("absent",)
            continue
        if out[name][0] != "dir":
            continue
        for dirpath, dirnames, filenames in os.walk(
            root, followlinks=False, onerror=_raise
        ):
            for child in dirnames + filenames:
                path = Path(dirpath) / child
                out[f"{name}/{path.relative_to(root).as_posix()}"] = _entry(path)
    return out


def alias(link: Path, target: Path) -> Path:
    """Create ``link`` as a symbolic link to ``target`` (parents created)."""
    link = Path(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, link)
    return link
