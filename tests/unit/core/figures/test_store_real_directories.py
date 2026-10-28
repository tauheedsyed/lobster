"""The local store reaches its artifacts only through real directories it can see by lstat.

Every store access to an existing or new artifact goes through one choke point
(``LocalFigureStore._guard`` and the methods built on it). A directory component that
already exists as a symbolic link (or as any other non-directory entry) at or below the
store root is refused, typed, BEFORE anything is created, written, linked, read or reused
through it: reserved directories, figure and revision directories, renditions and the root
itself. Final files are read without following a link (O_NOFOLLOW + regular-file check), so
a committed revision, a parent, a claim or a rendition behind a link is never accepted.

Scope: links that exist before the store looks. A link created concurrently between the
check and the write is out of scope (no descriptor-relative traversal here), and nothing in
this module confines what custom code can do on the filesystem. The workspace path and its
ancestors are trusted as given (a symlinked workspace path keeps working).

Real filesystem (tmp_path), real store; POSIX only (the store refuses other platforms).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import os
from pathlib import Path

import pytest

from lobster.core.figures import paths as P
from lobster.core.figures import store as store_module
from lobster.core.figures.errors import StorageError
from lobster.core.figures.ids import FigureRef
from lobster.core.figures.payload import source_to_bytes
from lobster.core.figures.records import STATUS_AVAILABLE, InputEdge, SourcePayload
from lobster.core.figures.renditions import ensure_html_rendition
from lobster.core.figures.saver import TrustedSaver
from lobster.core.figures.store import LocalFigureStore
from tests.unit.core.figures.test_store_and_saver import _cand, _fig, _info


def _tree(root: Path) -> dict:
    """Every entry under ``root`` (lstat, nothing followed) with its bytes' sha256."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                out[rel] = "link:" + os.readlink(path)
            elif path.is_file():
                out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                out[rel] = "dir"
    return out


@pytest.fixture()
def outside(tmp_path):
    """A directory the store must never write into, holding one sentinel file."""
    d = tmp_path / "outside"
    d.mkdir()
    (d / "sentinel.txt").write_bytes(b"do not touch\n")
    return d


@pytest.fixture()
def ws(tmp_path):
    d = tmp_path / "ws"
    d.mkdir()
    return d


def _src(tag="a"):
    return SourcePayload(source_to_bytes([{"k": tag}]), {"comparison": tag})


def _commit(store, base="s1:call_1", source=None, code=b"fig = 1\n", fence=False, **kw):
    inputs = (InputEdge("prot_de", "inst-1", "store_event", True),)
    extra = {"fence": store.attempt_fence()} if fence else {}
    [res] = TrustedSaver(store).commit_attempt(
        _info(code=code, base=base),
        [_cand(base=base, inputs=inputs, source=source, **kw)],
        succeeded=True,
        **extra,
    )
    return res


def _available(store):
    return [r for r in store.list().items]


def _assert_refused_and_untouched(store, res, outside, before):
    assert res.status != STATUS_AVAILABLE
    assert (res.ref, res.path) == (None, None)
    # typed (a store refusal, possibly stage-prefixed), never an unexpected error
    assert res.reason in ("storage_failed", "execution_record:storage_failed")
    assert _tree(outside) == before  # no new, changed or removed entry outside


# ------------------------------------------------------------- reserved directories


@pytest.mark.parametrize(
    "reserved", [P.SOURCES_DIR, P.CODE_DIR, P.EXECUTIONS_DIR, P.ATTEMPTS_DIR]
)
@pytest.mark.parametrize("fenced", [False, True])
def test_a_symlinked_reserved_directory_is_refused_before_any_write(
    ws, outside, reserved, fenced
):
    root = ws / P.PLOTS_DIR
    root.mkdir()
    (root / reserved).symlink_to(outside, target_is_directory=True)
    store = LocalFigureStore(ws, session_id="s1")
    before = _tree(outside)
    res = _commit(store, source=_src(), fence=fenced)
    _assert_refused_and_untouched(store, res, outside, before)


def test_a_symlinked_sources_directory_is_refused_on_an_otherwise_good_store(
    ws, outside
):
    store = LocalFigureStore(ws, session_id="s1")
    first = _commit(store, base="s1:call_a")  # no source: _sources not created yet
    assert first.status == STATUS_AVAILABLE
    (store.root / P.SOURCES_DIR).symlink_to(outside, target_is_directory=True)
    before = _tree(outside)
    res = _commit(store, base="s1:call_b", source=_src("b"))
    _assert_refused_and_untouched(store, res, outside, before)
    assert [f"{r['figure_id']}@{r['revision_id']}" for r in _available(store)] == [
        first.ref
    ]


# ------------------------------------------------------------------- the store root


@pytest.mark.parametrize("fenced", [False, True])
def test_a_symlinked_plots_root_is_refused_with_or_without_an_attempt_fence(
    ws, outside, fenced
):
    (ws / P.PLOTS_DIR).symlink_to(outside, target_is_directory=True)
    store = LocalFigureStore(ws, session_id="s1")
    before = _tree(outside)
    res = _commit(store, source=_src(), fence=fenced)
    _assert_refused_and_untouched(store, res, outside, before)


def test_a_symlinked_workspace_path_still_saves(tmp_path, ws):
    """Control: the workspace path itself (and its ancestors) is trusted as given."""
    link = tmp_path / "ws-link"
    link.symlink_to(ws, target_is_directory=True)
    store = LocalFigureStore(link, session_id="s1")
    res = _commit(store, source=_src())
    assert res.status == STATUS_AVAILABLE
    assert (ws / res.path).is_file()


# ------------------------------------------------------- figure and revision directories


def test_a_symlinked_existing_figure_directory_is_refused_for_an_edit(ws, outside):
    store = LocalFigureStore(ws, session_id="s1")
    parent = _commit(store, base="s1:call_a")
    assert parent.status == STATUS_AVAILABLE
    fig_dir = store.root / FigureRef.parse(parent.ref).figure_id
    # the figure directory now resolves elsewhere (a copy of it, so the parent still reads)
    copy = outside / "fig-copy"
    os.rename(fig_dir, copy)
    fig_dir.symlink_to(copy, target_is_directory=True)
    before = _tree(outside)
    res = _commit(
        store, base="s1:call_b", parent_ref=parent.ref, fig=_fig(title="edited")
    )
    _assert_refused_and_untouched(store, res, outside, before)


@pytest.mark.parametrize("level", ["figure", "revision"])
def test_an_interrupted_commit_through_a_symlinked_directory_is_not_completed(
    ws, outside, monkeypatch, level
):
    store = LocalFigureStore(ws, session_id="s1")
    real = store_module._link_no_replace

    def no_marker(path, data):
        if Path(path).name == P.REVISION_FILE:
            raise OSError("simulated interruption before the marker link")
        return real(path, data)

    monkeypatch.setattr(store_module, "_link_no_replace", no_marker)
    first = _commit(store, base="s1:call_a")
    monkeypatch.undo()
    assert first.status != STATUS_AVAILABLE
    [fig_dir] = [
        p for p in store.root.iterdir() if p.name not in P.RESERVED_DIRS
    ]  # the claimed, interrupted figure
    [rev_dir] = list(fig_dir.iterdir())
    linked = fig_dir if level == "figure" else rev_dir
    copy = outside / f"{level}-copy"
    os.rename(linked, copy)
    linked.symlink_to(copy, target_is_directory=True)
    before = _tree(outside)
    retry = _commit(store, base="s1:call_a")  # same attempt: completes the claim
    _assert_refused_and_untouched(store, retry, outside, before)
    assert _available(store) == []
    assert not any(name.endswith(P.REVISION_FILE) for name in _tree(outside))


def test_an_existing_figure_file_that_is_a_link_is_never_reused(
    ws, outside, monkeypatch
):
    """Identical bytes behind a link are not "the entry the store wrote"."""
    store = LocalFigureStore(ws, session_id="s1")
    real = store_module._link_no_replace

    def no_marker(path, data):
        if Path(path).name == P.REVISION_FILE:
            raise OSError("simulated interruption before the marker link")
        return real(path, data)

    monkeypatch.setattr(store_module, "_link_no_replace", no_marker)
    first = _commit(store, base="s1:call_a")
    monkeypatch.undo()
    assert first.status != STATUS_AVAILABLE
    [fig_dir] = [p for p in store.root.iterdir() if p.name not in P.RESERVED_DIRS]
    [rev_dir] = list(fig_dir.iterdir())
    figure_file = rev_dir / P.FIGURE_FILE
    moved = outside / "figure-copy.json"
    os.rename(figure_file, moved)
    figure_file.symlink_to(moved)  # same bytes, now behind a link
    before = _tree(outside)
    retry = _commit(store, base="s1:call_a")
    _assert_refused_and_untouched(store, retry, outside, before)
    assert not (rev_dir / P.REVISION_FILE).exists()


# --------------------------------------------------------------------------- renditions


def test_a_symlinked_renditions_directory_is_refused(ws, outside):
    store = LocalFigureStore(ws, session_id="s1")
    res = _commit(store)
    assert res.status == STATUS_AVAILABLE
    rev_dir = store._rev_dir(FigureRef.parse(res.ref))
    (rev_dir / P.RENDITIONS_DIR).symlink_to(outside, target_is_directory=True)
    before = _tree(outside)
    with pytest.raises(StorageError):
        store.add_rendition(
            res.ref,
            b"<html></html>",
            {"format": "html"},
            {
                "renderer": "test",
                "inputs": {
                    "figure_sha256": store.read_record(res.ref)["figure"]["sha256"],
                    "config_sha256": store.read_record(res.ref)["config"]["sha256"],
                },
            },
        )
    assert _tree(outside) == before


def test_a_symlinked_renditions_directory_never_yields_an_available_view(ws, outside):
    store = LocalFigureStore(ws, session_id="s1")
    res = _commit(store)
    rev_dir = store._rev_dir(FigureRef.parse(res.ref))
    (rev_dir / P.RENDITIONS_DIR).symlink_to(outside, target_is_directory=True)
    before = _tree(outside)
    view = ensure_html_rendition(store, res.ref)
    assert view.status != STATUS_AVAILABLE
    assert _tree(outside) == before


# ------------------------------------------------------------------------------ controls


def test_ordinary_saves_edits_sources_and_renditions_still_work(ws):
    store = LocalFigureStore(ws, session_id="s1")
    a = _commit(store, base="s1:call_a", source=_src("a"))
    assert a.status == STATUS_AVAILABLE
    b = _commit(store, base="s1:call_b", parent_ref=a.ref, fig=_fig(title="edited"))
    assert b.status == STATUS_AVAILABLE
    again = _commit(store, base="s1:call_b", parent_ref=a.ref, fig=_fig(title="edited"))
    assert (again.status, again.ref) == (STATUS_AVAILABLE, b.ref)  # idempotent retry
    assert ensure_html_rendition(store, b.ref).status == STATUS_AVAILABLE
    for reserved in (P.SOURCES_DIR, P.CODE_DIR, P.EXECUTIONS_DIR, P.ATTEMPTS_DIR):
        assert not (store.root / reserved).is_symlink()


# ------------------------------------- reuse of COMMITTED artifacts (marker written)


def _move_out_and_link(path: Path, outside: Path, name: str) -> Path:
    """Move a store entry to ``outside`` (identical bytes) and link it back in place."""
    copy = outside / name
    os.rename(path, copy)
    path.symlink_to(copy, target_is_directory=copy.is_dir())
    return copy


_LINKED = ["figure_dir", "revision_dir", P.REVISION_FILE, P.FIGURE_FILE, P.CONFIG_FILE]


def _linked_entry(store, ref, which):
    rev_dir = store._rev_dir(FigureRef.parse(ref))
    return {"figure_dir": rev_dir.parent, "revision_dir": rev_dir}.get(
        which, rev_dir / which
    )


@pytest.mark.parametrize("which", _LINKED)
@pytest.mark.parametrize("fresh", [False, True])
def test_a_committed_retry_through_a_link_is_refused(ws, outside, which, fresh):
    store = LocalFigureStore(ws, session_id="s1")
    first = _commit(store)
    assert first.status == STATUS_AVAILABLE
    assert store.read(first.ref)  # the warm store has confirmed the marker
    _move_out_and_link(_linked_entry(store, first.ref, which), outside, f"{which}-x")
    if fresh:
        store = LocalFigureStore(ws, session_id="s1")
    before = _tree(outside)
    retry = _commit(store)  # the identical attempt: would reuse the committed ref
    _assert_refused_and_untouched(store, retry, outside, before)
    with pytest.raises(StorageError):
        store.read(first.ref)  # payload read, sha-verified: never through a link
    if which not in (P.FIGURE_FILE, P.CONFIG_FILE):  # read_record is metadata only
        with pytest.raises(StorageError):
            store.read_record(first.ref)


@pytest.mark.parametrize("which", ["revision_dir", P.REVISION_FILE])
def test_an_edit_of_a_parent_behind_a_link_is_refused(ws, outside, which):
    """Only the PARENT's revision directory (or its record) is linked; the figure
    directory stays real, so the new sibling revision would be writable. (An edit's
    commit reads the parent's record; its payload is read, guarded, by ``read``.)"""
    store = LocalFigureStore(ws, session_id="s1")
    parent = _commit(store, base="s1:call_a")
    assert parent.status == STATUS_AVAILABLE
    _move_out_and_link(_linked_entry(store, parent.ref, which), outside, f"{which}-p")
    before = _tree(outside)
    edit = _commit(
        store, base="s1:call_b", parent_ref=parent.ref, fig=_fig(title="edited")
    )
    _assert_refused_and_untouched(store, edit, outside, before)
    fig_dir = store._rev_dir(FigureRef.parse(parent.ref)).parent
    assert [p.name for p in fig_dir.iterdir()] == [
        FigureRef.parse(parent.ref).revision_id
    ]  # no new sibling revision directory


def test_a_linked_committed_figure_is_never_listed_as_available(ws, outside):
    store = LocalFigureStore(ws, session_id="s1")
    kept = _commit(store, base="s1:call_a")
    moved = _commit(store, base="s1:call_b", fig=_fig(title="other"))
    _move_out_and_link(_linked_entry(store, moved.ref, "figure_dir"), outside, "f")
    for s in (store, LocalFigureStore(ws, session_id="s1")):
        assert [f"{r['figure_id']}@{r['revision_id']}" for r in _available(s)] == [
            kept.ref
        ]


# --------------------------------------------- reuse of an EXISTING rendition


def _rendered(ws):
    store = LocalFigureStore(ws, session_id="s1")
    res = _commit(store)
    view = ensure_html_rendition(store, res.ref)
    assert view.status == STATUS_AVAILABLE and not getattr(view, "reused", False)
    [rec] = store.list_renditions(res.ref)
    return store, res.ref, rec


@pytest.mark.parametrize("which", ["renditions_dir", "marker", "data"])
@pytest.mark.parametrize("fresh", [False, True])
def test_an_existing_rendition_behind_a_link_is_never_reused(ws, outside, which, fresh):
    store, ref, rec = _rendered(ws)
    rdir = store._rev_dir(FigureRef.parse(ref)) / P.RENDITIONS_DIR
    entry = {
        "renditions_dir": rdir,
        "marker": rdir / f"{rec['rendition_id']}.json",
        "data": rdir / rec["file"],
    }[which]
    _move_out_and_link(entry, outside, f"{which}-r")
    if fresh:
        store = LocalFigureStore(ws, session_id="s1")
    before = _tree(outside)
    view = ensure_html_rendition(store, ref)
    assert view.status != STATUS_AVAILABLE
    assert getattr(view, "path", None) is None
    assert _tree(outside) == before
    with pytest.raises(StorageError):
        store.read_rendition(ref, rec["rendition_id"])
    if which != "data":
        with pytest.raises(StorageError):
            store.list_renditions(ref)


# ------------------------------- the durability memo is bookkeeping only (after publish)


def _eio_once_after_link(monkeypatch, is_marker):
    """After the marker ``is_marker`` was linked, the NEXT lstat fails once with EIO (the
    memo's chain check runs then). Everything before the link is untouched."""
    real_link, real_lstat = store_module._link_no_replace, os.lstat
    state = {"armed": False, "fired": 0}

    def link(target, data):
        out = real_link(target, data)
        if is_marker(Path(target)):
            state["armed"] = True
        return out

    def lstat(path, *a, **k):
        if state["armed"]:
            state["armed"] = False
            state["fired"] += 1
            raise OSError(5, "EIO")
        return real_lstat(path, *a, **k)

    monkeypatch.setattr(store_module, "_link_no_replace", link)
    monkeypatch.setattr(os, "lstat", lstat)
    return state


def test_an_io_fault_in_the_memo_after_publish_keeps_the_save_available(
    ws, monkeypatch
):
    store = LocalFigureStore(ws, session_id="s1")
    state = _eio_once_after_link(monkeypatch, lambda p: p.name == P.REVISION_FILE)
    res = _commit(store)
    assert state["fired"] == 1  # the fault happened, after the marker was published
    monkeypatch.undo()
    assert (res.status, res.reason) == (STATUS_AVAILABLE, None)
    assert res.ref and res.path
    assert store.read(res.ref)  # re-confirmed on read: the memo was only skipped


def test_an_io_fault_in_the_memo_after_a_rendition_publish_keeps_it_available(
    ws, monkeypatch
):
    store = LocalFigureStore(ws, session_id="s1")
    res = _commit(store)
    record = store.read_record(res.ref)
    receipt = {
        "renderer": "test",
        "inputs": {
            "figure_sha256": record["figure"]["sha256"],
            "config_sha256": record["config"]["sha256"],
        },
    }
    state = _eio_once_after_link(
        monkeypatch, lambda p: p.name.startswith("p_") and p.suffix == ".json"
    )
    added = store.add_rendition(res.ref, b"<html></html>", {"format": "html"}, receipt)
    assert state["fired"] == 1
    monkeypatch.undo()
    assert added["availability"] == STATUS_AVAILABLE
    assert store.read_rendition(res.ref, added["rendition_id"]) == b"<html></html>"


# ---------------------------------------- deletion keeps working on a linked entry


def test_a_linked_revision_can_still_be_hidden_and_is_never_removed_through(
    ws, outside
):
    store = LocalFigureStore(ws, session_id="s1")
    res = _commit(store)
    copy = _move_out_and_link(
        _linked_entry(store, res.ref, "revision_dir"), outside, "d"
    )
    before = _tree(outside)
    assert store.fence_deletion([res.ref]).complete  # hiding never follows the link
    report = store.remove_fenced([res.ref])
    assert (report.removed, report.failed) == (0, 1)
    assert (copy / P.REVISION_FILE).is_file() and _tree(outside) == before


# ------------------------------------------------------------- the choke point (static)
#
# A structural checker over the WHOLE store module (every top-level function, every class
# and method, module-level statements and imports), with negative controls it must reject.
# Limits, stated: dynamic access (``getattr(p, "read_bytes")()``, a callable stored in a
# variable, ``exec``) and filesystem access inside OTHER modules called by the store are not
# seen; the guard's argument is not related to the accessed path. It pins the structure
# the review asked for; it does not prove the absence of every bypass.

#: The ONLY LocalFigureStore methods that may touch the filesystem.
_CHOKE = {
    "_guard",
    "_linked",
    "_require_real_dir",
    "_read",
    "_read_or",
    "_present",
    "_identity",
    "_fsync",
    "_listdir_real",
    "_lstat_or_none",
    "_lstat",
    "_lstat_mode",
    "_unlink",
    "_rmtree",
    "_rmdir_quiet",
    "_mkdir_new",
    "_mkdirs",
    "_write",
    "_link",
}
#: The lstat walks themselves: ``os.lstat`` only, no guard of their own.
_WALKS = {"_guard", "_linked"}
#: Module-level primitives and EXACTLY the raw calls each may make (pinned).
_PRIMITIVE_CALLS = {
    "_root_lock": {"os.path.realpath"},
    "_fsync_dir": {"os.open", "os.fsync", "os.close"},
    "_marker_present": {"os.lstat"},
    "_read_regular": {"os.open", "os.fstat", "os.read", "os.close"},
    "_marker_identity": {"os.lstat"},
    "_ensure_dir": {".is_dir", "_ensure_dir", "os.mkdir", "_fsync_dir"},
    "_write_no_replace": {"_link_no_replace"},
    "_link_no_replace": {
        "_read_regular",
        "_ensure_dir",
        "_fsync_dir",
        "tempfile.mkstemp",
        "os.fdopen",
        "os.fsync",
        "os.link",
        "os.unlink",
    },
}
_PRIMITIVES = set(_PRIMITIVE_CALLS) - {"_root_lock"}
_FS_MODULES = {"os", "shutil", "tempfile", "io", "pathlib", "builtins", "glob"}
_PATH_METHODS = {
    "read_bytes",
    "read_text",
    "write_bytes",
    "write_text",
    "open",
    "is_file",
    "is_dir",
    "exists",
    "stat",
    "lstat",
    "glob",
    "rglob",
    "iterdir",
    "resolve",
    "unlink",
    "mkdir",
    "rmdir",
    "rename",
    "replace",
    "symlink_to",
    "hardlink_to",
    "is_symlink",
    "samefile",
    "readlink",
    "touch",
    "chmod",
    "walk",
}
_PURE_OS = {"fspath", "fsencode", "fsdecode", "getpid", "urandom"}  # no path access
_PURE_OS_PATH = {"join", "basename", "dirname", "split", "normpath", "relpath"}


def _call_name(node):
    """The raw-filesystem name of a call, or None for anything else."""
    f = node.func
    if isinstance(f, ast.Name):
        if f.id in _PRIMITIVES or f.id in ("open", "__import__"):
            return f.id
        return None
    if not isinstance(f, ast.Attribute):
        return None
    v = f.value
    if isinstance(v, ast.Name) and v.id in _FS_MODULES:
        if v.id == "os" and f.attr in _PURE_OS:
            return None
        return f"{v.id}.{f.attr}"
    if (
        isinstance(v, ast.Attribute)
        and isinstance(v.value, ast.Name)
        and v.value.id == "os"
        and v.attr == "path"
    ):
        return None if f.attr in _PURE_OS_PATH else f"os.path.{f.attr}"
    if f.attr in _PATH_METHODS:
        return f".{f.attr}"
    return None


def _raw_calls(node):
    found = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            name = _call_name(n)
            if name is not None:
                found.append(name)
    return found


def _is_guard_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "_guard"
    )


def _unguarded(stmts, guarded=False):
    """Raw calls in ``stmts`` NOT dominated by an evaluated ``self._guard(...)``.

    Structural, not by line number: a guard counts from the statement AFTER the one whose
    unconditionally evaluated expression holds it (an ``Expr``/assignment/``return`` value,
    an ``if``/``while`` test, a ``for`` iterator, ``with`` items); an ``if``/``while``/
    ``for``/``with`` head also covers that statement's own blocks. A guard inside a branch,
    a loop body, a ``try`` body or a handler covers nothing after that statement.
    """
    out = []
    for stmt in stmts:
        if isinstance(stmt, (ast.If, ast.While)):
            heads, blocks = [stmt.test], [stmt.body, stmt.orelse]
        elif isinstance(stmt, ast.For):
            heads, blocks = [stmt.iter], [stmt.body, stmt.orelse]
        elif isinstance(stmt, ast.With):
            heads, blocks = [i.context_expr for i in stmt.items], [stmt.body]
        elif isinstance(stmt, ast.Try):
            heads = []
            blocks = [stmt.body] + [h.body for h in stmt.handlers]
            blocks += [stmt.orelse, stmt.finalbody]
        elif isinstance(stmt, (ast.FunctionDef, ast.ClassDef)):
            heads, blocks = [], [stmt.body]  # a nested def is checked as its own body
        else:
            heads, blocks = [stmt], []
        here = any(_is_guard_call(n) for h in heads for n in ast.walk(h))
        if not guarded:
            for h in heads:
                out += _raw_calls(h)
        compound = isinstance(stmt, (ast.If, ast.While, ast.For, ast.With))
        for block in blocks:
            out += _unguarded(block, guarded or (here and compound))
        guarded = guarded or here
    return out


def _violations(source):
    """Every place in ``source`` (a store module) that breaks the choke-point rule."""
    out = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in _FS_MODULES and a.asname is not None:
                    out.append(("import", f"{a.name} as {a.asname}"))
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module in _FS_MODULES - {"pathlib"}:
                out.append(("import", f"from {node.module}"))
            if module == "pathlib" and any(a.name != "Path" for a in node.names):
                out.append(("import", "from pathlib (not Path)"))
        elif isinstance(node, ast.FunctionDef):
            calls = set(_raw_calls(node))
            allowed = _PRIMITIVE_CALLS.get(node.name, set())
            out += [(node.name, name) for name in sorted(calls - allowed)]
            if node.name in _PRIMITIVE_CALLS and calls != allowed:
                out.append((node.name, "primitive calls changed"))
        elif isinstance(node, ast.ClassDef):
            store = node.name == "LocalFigureStore"
            for item in node.body:
                fn = item if isinstance(item, ast.FunctionDef) else None
                where = f"{node.name}.{fn.name if fn else '<body>'}"
                calls = _raw_calls(item)
                if not store or fn is None or fn.name not in _CHOKE:
                    for name in calls:
                        if (where, name) != (
                            "LocalFigureStore.__init__",
                            "os.path.realpath",
                        ):
                            out.append((where, name))
                elif fn.name in _WALKS:
                    out += [(where, n) for n in calls if n != "os.lstat"]
                else:
                    out += [(where, f"unguarded {n}") for n in _unguarded(fn.body)]
        else:
            out += [("<module>", name) for name in _raw_calls(node)]
    return out


def _mentions_nofollow(expr):
    for n in ast.walk(expr):
        if isinstance(n, ast.Attribute) and n.attr == "O_NOFOLLOW":
            return True
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "getattr"
            and len(n.args) >= 2
            and isinstance(n.args[1], ast.Constant)
            and n.args[1].value == "O_NOFOLLOW"
        ):
            return True
    return False


def _nofollow_ok(source):
    """Structural: every ``os.open`` in ``_read_regular`` passes flags that include
    ``O_NOFOLLOW`` (by AST, never by text), the reader checks ``S_ISREG``, and the marker
    primitives lstat only."""
    fns = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    reader = fns.get("_read_regular")
    if reader is None:
        return False
    flag_names = {
        t.id
        for n in ast.walk(reader)
        if isinstance(n, ast.Assign) and _mentions_nofollow(n.value)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    opens = [
        n
        for n in ast.walk(reader)
        if isinstance(n, ast.Call) and _call_name(n) == "os.open"
    ]
    opens_ok = bool(opens) and all(
        len(c.args) >= 2
        and (
            _mentions_nofollow(c.args[1])
            or (isinstance(c.args[1], ast.Name) and c.args[1].id in flag_names)
        )
        for c in opens
    )
    isreg = any(
        isinstance(n, ast.Attribute) and n.attr == "S_ISREG" for n in ast.walk(reader)
    )
    lstat_only = all(
        name in fns and set(_raw_calls(fns[name])) == {"os.lstat"}
        for name in ("_marker_present", "_marker_identity")
    )
    return opens_ok and isreg and lstat_only


def _store_source():
    return inspect.getsource(store_module)


def test_only_the_choke_point_touches_the_filesystem():
    """The real store module: no violation anywhere; every choke method exists."""
    source = _store_source()
    assert _violations(source) == []
    [cls] = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "LocalFigureStore"
    ]
    assert _CHOKE <= {f.name for f in cls.body if isinstance(f, ast.FunctionDef)}


def test_the_module_primitives_do_not_follow_a_final_link(tmp_path):
    assert _nofollow_ok(_store_source())
    assert not hasattr(store_module, "_read_typed")  # the following reader is gone
    real = tmp_path / "real.json"
    real.write_bytes(b"{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(StorageError):  # behaviour, not only structure
        store_module._read_regular(link)
    with pytest.raises(StorageError):
        store_module._marker_present(link)


_GOOD = (
    "class LocalFigureStore:\n"
    "    def _read(self, path):\n"
    "        if not self._guard(path.parent):\n"
    "            return None\n"
    "        return _read_regular(path)\n"
    "\n"
    "    def _rmtree(self, d):\n"
    "        if self._guard(d):\n"
    "            shutil.rmtree(d)\n"
    "\n"
    "    def _unlink(self, p):\n"
    "        self._guard(p.parent)\n"
    "        os.unlink(p)\n"
)


def _method(body):
    return "class LocalFigureStore:\n    def _unlink(self, p):\n" + "".join(
        f"        {line}\n" for line in body
    )


_BYPASSES = [
    ("hoisted module helper", "def _sneaky(p):\n    return p.read_bytes()\n" + _GOOD),
    (
        "non-choke method",
        _GOOD + "\n    def read_source(self, p):\n        return p.read_text()\n",
    ),
    (
        "primitive from a non-choke method",
        _GOOD + "\n    def read(self, p):\n        return _read_regular(p)\n",
    ),
    ("dead guard", _method(["if False:", "    self._guard(p.parent)", "os.unlink(p)"])),
    ("guard after the access", _method(["os.unlink(p)", "self._guard(p.parent)"])),
    (
        "guard only in a try body",
        _method(
            [
                "try:",
                "    self._guard(p.parent)",
                "except Exception:",
                "    pass",
                "os.unlink(p)",
            ]
        ),
    ),
    (
        "guard in a loop body",
        _method(["for _ in ():", "    self._guard(p.parent)", "os.unlink(p)"]),
    ),
    (
        "guard in the other branch",
        _method(["if p:", "    self._guard(p.parent)", "else:", "    os.unlink(p)"]),
    ),
    ("aliased os import", "import os as _o\n" + _GOOD),
    ("from-os import", "from os import unlink\n" + _GOOD),
    ("module-level access", _GOOD + "\nopen('x')\n"),
    (
        "a walk that reads",
        "class LocalFigureStore:\n    def _guard(self, d):\n        return d.read_bytes()\n",
    ),
    (
        "a changed primitive",
        "def _marker_present(path):\n    return os.stat(path)\n" + _GOOD,
    ),
    (
        "another class",
        _GOOD + "\nclass Helper:\n    def f(self, p):\n        return p.read_bytes()\n",
    ),
]


@pytest.mark.parametrize("label, source", _BYPASSES, ids=[b[0] for b in _BYPASSES])
def test_the_checker_rejects_each_bypass(label, source):
    assert _violations(source), label


def test_the_checker_accepts_the_guarded_forms():
    assert _violations(_GOOD) == []


_MARKERS = (
    "def _marker_present(path):\n    os.lstat(path)\n"
    "def _marker_identity(path):\n    os.lstat(path)\n"
)
_GOOD_READER = (
    "def _read_regular(path):\n"
    '    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)\n'
    "    fd = os.open(str(path), flags)\n"
    "    _stat.S_ISREG(os.fstat(fd).st_mode)\n"
)
_TEXT_ONLY = [
    (
        "comment only",
        "def _read_regular(path):\n    # O_NOFOLLOW\n    flags = os.O_RDONLY\n"
        "    fd = os.open(str(path), flags)\n    _stat.S_ISREG(os.fstat(fd).st_mode)\n",
    ),
    (
        "docstring only",
        'def _read_regular(path):\n    """getattr(os, "O_NOFOLLOW")"""\n'
        "    fd = os.open(str(path), os.O_RDONLY)\n"
        "    _stat.S_ISREG(os.fstat(fd).st_mode)\n",
    ),
    (
        "a second open without it",
        _GOOD_READER + "    os.open(str(path), os.O_RDONLY)\n",
    ),
    (
        "no regular-file check",
        "def _read_regular(path):\n"
        '    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)\n'
        "    fd = os.open(str(path), flags)\n",
    ),
    (
        "a marker primitive that stats",
        _GOOD_READER + "def _marker_present(path):\n    os.stat(path)\n",
    ),
]


@pytest.mark.parametrize("label, reader", _TEXT_ONLY, ids=[t[0] for t in _TEXT_ONLY])
def test_the_nofollow_check_rejects_text_only_forms(label, reader):
    assert not _nofollow_ok(reader + _MARKERS if "marker" not in label else reader)


def test_the_nofollow_check_accepts_the_real_form():
    assert _nofollow_ok(_GOOD_READER + _MARKERS)
