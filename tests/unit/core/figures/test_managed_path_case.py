"""A case-variant spelling of ``plots`` is the managed figure folder when the filesystem
says it is the same directory (canonical filesystem identity at the managed-root boundary).

Generic writers (the ``write_file`` tool, the client's workspace write) classify through
:func:`paths.assert_not_managed`; on a case-insensitive filesystem ``PLOTS/<f>/<r>/...``
addresses the real managed files and must be refused like ``plots/<f>/<r>/...``. Normal
workspace files and legacy flat files stay allowed. This is a path rule for generic writers,
not filesystem confinement.

Coverage honesty: the ``real_ci`` tests need a real case-insensitive filesystem under
``tmp_path`` (macOS APFS default). They SKIP elsewhere with a ``NOT COVERED`` reason; that
skip is not a pass. The classifier tests below them run on any filesystem (case-sensitive CI
included) by simulating the identity answers, and one test pins the real probe result.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from lobster.core.figures import paths as P
from lobster.core.figures.errors import ManagedPathError
from lobster.core.figures.ids import FigureRef
from lobster.core.figures.records import STATUS_AVAILABLE
from lobster.core.figures.saver import TrustedSaver
from lobster.core.figures.store import LocalFigureStore
from tests.unit.core.figures.test_store_and_saver import _cand, _info


def _real_ci(tmp_path: Path) -> bool:
    d = tmp_path / "probe-root"
    d.mkdir()
    (d / "plots").mkdir()
    try:
        a, b = os.lstat(d / "plots"), os.lstat(d / "PLOTS")
    except FileNotFoundError:
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


@pytest.fixture()
def real_ci(tmp_path):
    if not _real_ci(tmp_path):
        pytest.skip(
            "NOT COVERED: no case-insensitive filesystem under tmp_path "
            "(the classifier tests below still run)"
        )
    return True


def _hashes(root: Path) -> dict:
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            out[rel] = (
                hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir"
            )
    return out


def _saved(ws: Path):
    store = LocalFigureStore(ws, session_id="s1")
    [res] = TrustedSaver(store).commit_attempt(_info(), [_cand()], succeeded=True)
    assert res.status == STATUS_AVAILABLE
    return store, FigureRef.parse(res.ref)


def _write_file(ws: Path):
    from lobster.tools.filesystem_tools import create_filesystem_tools

    tools = {t.name: t for t in create_filesystem_tools(workspace_path=ws)}
    return lambda path, content="x": tools["write_file"].invoke(
        {"path": path, "content": content}
    )


def _refused(ws, target) -> bool:
    try:
        P.assert_not_managed(ws, target)
    except ManagedPathError:
        return True
    return False


# ------------------------------------------- real case-insensitive filesystem (APFS)


def test_real_ci_write_file_refuses_case_variants_of_the_managed_folder(
    tmp_path, real_ci
):
    ws = tmp_path / "ws"
    ws.mkdir()
    store, ref = _saved(ws)
    # first establish that the spellings name ONE directory here
    a, b = os.lstat(ws / "plots"), os.lstat(ws / "PLOTS")
    assert (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)
    before = _hashes(ws / "plots")
    write = _write_file(ws)
    fig, rev = ref.figure_id, ref.revision_id
    for spelling in ("PLOTS", "Plots", "pLoTs", "plotS"):
        for rel in (
            f"{spelling}/{fig}/{rev}/{P.FIGURE_FILE}",
            f"{spelling}/{fig}/{rev}/{P.REVISION_FILE}",
            f"{spelling}/{fig}/{rev}/new.json",
            f"{spelling}/{fig}/new_rev/{P.FIGURE_FILE}",
            f"{spelling}/f_new/r_new/{P.REVISION_FILE}",
            f"{spelling}/{P.SOURCES_DIR}/x.json",
            f"{spelling}/_SOURCES/x.json",
            f"{spelling}/{fig}",
            spelling,
        ):
            out = write(rel, "forged")
            assert out.startswith("Error"), rel
    assert _hashes(ws / "plots") == before
    assert store.read_record(ref)["figure"]["sha256"]  # still reads


def test_real_ci_ancestor_and_reserved_spellings_refuse_at_the_classifier(
    tmp_path, real_ci
):
    """The client's workspace write passes ``workspace / filename`` straight in."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _, ref = _saved(ws)
    rev = Path(ref.figure_id) / ref.revision_id / P.FIGURE_FILE
    variant_ws = tmp_path / "WS"  # the same workspace, spelled differently
    assert os.path.samefile(variant_ws, ws)
    for target in (
        ws / "PLOTS" / rev,
        variant_ws / "plots" / rev,
        variant_ws / "PLOTS" / rev,
        ws / "data" / ".." / "PLOTS" / rev,
        ws / "plots" / "_ATTEMPTS" / "x.json",
        ws / "PLOTS",
        variant_ws,
    ):
        assert _refused(ws, target), target


def test_real_ci_a_missing_plots_folder_cannot_be_created_under_another_spelling(
    tmp_path, real_ci
):
    ws = tmp_path / "ws"
    ws.mkdir()
    write = _write_file(ws)
    out = write(f"PLOTS/f_1/r_1/{P.REVISION_FILE}", "{}")
    assert out.startswith("Error")
    assert not (ws / "plots").exists()  # neither spelling was created
    assert _refused(ws, ws / "Plots" / "f_1" / "r_1" / P.FIGURE_FILE)


def test_real_ci_ordinary_and_legacy_writes_still_work(tmp_path, real_ci):
    ws = tmp_path / "ws"
    ws.mkdir()
    _saved(ws)
    write = _write_file(ws)
    assert write("data/x.txt").startswith("Wrote")
    assert write("PLOTS2/x.txt").startswith("Wrote")  # another name, not the folder
    assert write("plots/notes.html").startswith("Wrote")  # a legacy flat file
    assert write("PLOTS/notes.html", "y").startswith("Wrote")  # the same legacy file
    assert (ws / "plots" / "notes.html").read_text() == "y"


# ------------------------------------------------- classifier (any filesystem, CI too)


def test_the_probe_reports_this_filesystem_truthfully(tmp_path):
    """A witness INSIDE the directory answers; a sibling ``probe-root`` proves nothing
    about ``ws``, so an empty ``ws`` is unknown (None), never inferred."""
    ws = tmp_path / "ws"
    ws.mkdir()
    observed = _real_ci(tmp_path)
    assert P._case_sensitive(str(ws)) is None  # empty: no witness inside ws
    (ws / "data").mkdir()  # a local witness
    assert P._case_sensitive(str(ws)) is (not observed)


def _per_directory_modes(monkeypatch, modes):
    """Simulate per-directory case modes (Linux casefold / ext4 +F style), which cannot
    be built on macOS APFS: a swapped-case lookup of an existing name in ``directory``
    finds a different entry or none (case-sensitive -> True) or the same entry
    (case-insensitive -> False). Only the lookup results are simulated; the probe logic
    is real."""

    def swap(directory, name):
        if not name.isascii() or name.swapcase() == name:
            return None
        return modes[os.path.normpath(directory)]

    monkeypatch.setattr(P, "_swap_answer", swap)


def test_a_case_sensitive_parent_does_not_prove_an_empty_workspace_case_sensitive(
    tmp_path, monkeypatch
):
    """ws is case-INSENSITIVE and empty; its same-device parent is case-sensitive.
    The parent's answer must not be taken for ws: unknown -> the variant is refused."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _per_directory_modes(monkeypatch, {str(tmp_path): True, str(ws): False})
    assert P._case_sensitive(str(ws)) is None
    target = ws / "PLOTS" / "f_1" / "r_1" / P.FIGURE_FILE
    assert P.classify(ws, target, follow_final=True).lexical == P.INSIDE
    assert _refused(ws, target)


def test_a_case_sensitive_workspace_does_not_prove_an_empty_plots_folder_sensitive(
    tmp_path, monkeypatch
):
    """plots/ exists, is empty and case-INSENSITIVE inside a case-sensitive ws: an absent
    reserved name under another spelling stays reserved (no flat file at ``_sources``).
    """
    ws = tmp_path / "ws"
    (ws / "plots").mkdir(parents=True)
    _per_directory_modes(
        monkeypatch, {str(tmp_path): True, str(ws): True, str(ws / "plots"): False}
    )
    assert P._case_sensitive(str(ws / "plots")) is None
    assert P.classify(ws, ws / "plots" / "_SOURCES", follow_final=True).lexical == (
        P.RESERVED
    )
    assert _refused(ws, ws / "plots" / "_SOURCES")


@pytest.mark.parametrize("mode", [True, False])
def test_a_witness_inside_the_directory_is_still_taken(tmp_path, monkeypatch, mode):
    ws = tmp_path / "ws"
    (ws / "data").mkdir(parents=True)
    _per_directory_modes(monkeypatch, {str(tmp_path): not mode, str(ws): mode})
    assert P._case_sensitive(str(ws)) is mode


def test_an_identical_entry_under_another_spelling_is_the_managed_root(
    tmp_path, monkeypatch
):
    """Simulates the filesystem's identity answers, so a case-sensitive CI runs it."""
    ws = tmp_path / "ws"
    (ws / "plots" / "f_1" / "r_1").mkdir(parents=True)
    real_id = P._lstat_id

    def aliasing(path):  # what a case-insensitive lookup would answer
        head, name = os.path.split(path)
        if name.casefold() == "plots" and os.path.isdir(os.path.join(head, "plots")):
            return real_id(os.path.join(head, "plots"))
        return real_id(path)

    monkeypatch.setattr(P, "_lstat_id", aliasing)
    target = ws / "PLOTS" / "f_1" / "r_1" / P.FIGURE_FILE
    assert P.classify(ws, target, follow_final=True).lexical == P.INSIDE
    assert _refused(ws, target)
    assert P.classify(ws, ws / "PLOTS", follow_final=True).lexical == P.ROOT
    assert not _refused(ws, ws / "data" / "x")


def test_a_distinct_directory_under_another_spelling_stays_a_workspace_path(tmp_path):
    """Case-sensitive answer: ``PLOTS`` exists and is NOT ``plots``."""
    ws = tmp_path / "ws"
    (ws / "plots").mkdir(parents=True)
    if _real_ci(tmp_path):
        pytest.skip("NOT COVERED here: needs a case-sensitive filesystem (CI)")
    (ws / "PLOTS").mkdir()
    assert P.classify(ws, ws / "PLOTS" / "f" / "r", follow_final=True) == (
        P.WORKSPACE,
        P.WORKSPACE,
    )
    assert not _refused(ws, ws / "PLOTS" / "f" / "r" / P.FIGURE_FILE)


def test_real_cs_write_file_keeps_a_distinct_spelling_writable(tmp_path):
    """Case-sensitive answer through the real tool, with and without ``plots/`` present."""
    if _real_ci(tmp_path):
        pytest.skip("NOT COVERED here: needs a case-sensitive filesystem (CI)")
    ws = tmp_path / "ws"
    (ws / "data").mkdir(
        parents=True
    )  # a witness inside ws: the proof is about ws itself
    write = _write_file(ws)
    assert write("PLOTS/f_1/r_1/notes.txt").startswith("Wrote")  # no plots/ yet: proven
    assert not (ws / "plots").exists()
    _saved(ws)
    assert write("Plots/f_2/notes.txt").startswith("Wrote")
    assert write(f"plots/f_9/r_9/{P.REVISION_FILE}").startswith(
        "Error"
    )  # exact: managed


@pytest.mark.parametrize(
    "sensitivity, refused", [(True, False), (False, True), (None, True)]
)
def test_with_no_plots_folder_yet_only_a_proven_case_sensitive_fs_allows_a_variant(
    tmp_path, monkeypatch, sensitivity, refused
):
    """Neither spelling exists: the write would create the folder the store later uses.
    Allowed only when this filesystem is PROVEN case-sensitive (unknown refuses)."""
    ws = tmp_path / "ws"
    ws.mkdir()
    real_id = P._lstat_id

    def absent_both(path):
        if os.path.basename(path).casefold() == "plots":
            return None
        return real_id(path)

    monkeypatch.setattr(P, "_lstat_id", absent_both)
    monkeypatch.setattr(P, "_case_sensitive", lambda _path: sensitivity)
    target = ws / "Plots" / "f_1" / "r_1" / P.REVISION_FILE
    assert _refused(ws, target) is refused
    assert _refused(ws, ws / "plots" / "f_1" / "r_1" / P.REVISION_FILE)  # exact: always
    assert not _refused(ws, ws / "data" / "x")


def test_a_variant_workspace_spelling_is_rebased_by_identity(tmp_path, monkeypatch):
    """An ancestor spelled differently (same directory by identity) is the workspace."""
    ws = tmp_path / "ws"
    (ws / "plots" / "f_1" / "r_1").mkdir(parents=True)
    twin = tmp_path / "twin"  # stands in for ``WS``: same identity as ``ws``
    twin.mkdir()
    real_id = P._lstat_id

    def same_ws(path):
        if os.path.normpath(path) == os.path.realpath(twin):
            return real_id(os.path.realpath(ws))
        return real_id(path)

    monkeypatch.setattr(P, "_lstat_id", same_ws)
    (twin / "plots" / "f_1" / "r_1").mkdir(parents=True)  # what the alias would show
    assert _refused(ws, twin / "plots" / "f_1" / "r_1" / P.FIGURE_FILE)
    assert not _refused(ws, tmp_path / "elsewhere.txt")  # outside: unchanged
