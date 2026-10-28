"""Durability follow-up of the figure store (items 1 and 2).

Real filesystem (tmp_path). Faults are injected at the store's OS seam (``_fsync_dir`` /
``os.link``) and every directory fsync is traced, so the tests assert the promised fsync
contract: an entry a previous call left UNCONFIRMED is confirmed again before ``available``
is returned. They do not claim crash recovery of arbitrary filesystems.
"""

from __future__ import annotations

import json
import os

import pytest

from lobster.core.figures import paths as P
from lobster.core.figures import store as store_mod
from lobster.core.figures.errors import (
    DurabilityError,
    FigureNotFoundError,
    IntegrityError,
)
from lobster.core.figures.ids import FigureRef
from lobster.core.figures.records import SourcePayload
from lobster.core.figures.saver import TrustedSaver
from lobster.core.figures.store import LocalFigureStore

from .test_store_and_saver import _cand, _info


@pytest.fixture()
def store(tmp_path):
    return LocalFigureStore(tmp_path, session_id="s1")


class FsyncTrace:
    """Trace every directory fsync; optionally fail matching ones (once or always)."""

    def __init__(self, monkeypatch, fail=None, once=True):
        self.real = store_mod._fsync_dir
        self.fail = fail
        self.once = once
        self.failed = 0
        self.ok: list = []
        monkeypatch.setattr(store_mod, "_fsync_dir", self)

    def __call__(self, path):
        if (
            self.fail is not None
            and self.fail(path)
            and not (self.once and self.failed)
        ):
            self.failed += 1
            raise DurabilityError("injected", details={"reason": "dir_fsync_failed"})
        self.real(path)
        self.ok.append(path)

    def reset(self):
        self.ok = []


def _commit(store, **kw):
    [res] = TrustedSaver(store).commit_attempt(_info(), [_cand(**kw)], succeeded=True)
    return res


# --- 1. a retry re-confirms entries a previous call left unconfirmed ------------------------


def test_identical_retry_reconfirms_a_code_blob_whose_dir_fsync_failed(
    store, monkeypatch
):
    trace = FsyncTrace(monkeypatch, fail=lambda p: p.name == P.CODE_DIR)
    first = _commit(store)
    assert (first.status, first.reason) == (
        "failed",
        "execution_record:durability_failed:dir_fsync_failed",
    )
    trace.reset()
    again = _commit(store)
    assert again.status == "available"
    assert store.root / P.CODE_DIR in trace.ok  # the linked-but-unconfirmed blob entry


def test_persistent_code_dir_fsync_failure_never_turns_into_success(store, monkeypatch):
    FsyncTrace(monkeypatch, fail=lambda p: p.name == P.CODE_DIR, once=False)
    assert _commit(store).status == "failed"
    again = _commit(store)
    assert (again.status, again.reason) == (
        "failed",
        "execution_record:durability_failed:dir_fsync_failed",
    )
    assert store.list().items == []


def test_identical_retry_reconfirms_an_intermediate_directory(store, monkeypatch):
    """mkdir r_* succeeded but fsync of its parent f_* failed; the retry must fsync f_*."""
    trace = FsyncTrace(monkeypatch, fail=lambda p: p.name.startswith("f_"))
    first = _commit(store)
    assert (first.status, first.reason) == (
        "failed",
        "durability_failed:dir_fsync_failed",
    )
    trace.reset()
    again = _commit(store)
    assert again.status == "available"
    fig_dir = store.root / FigureRef.parse(again.ref).figure_id
    assert fig_dir in trace.ok


def test_identical_retry_reconfirms_a_candidate_claim_whose_dir_fsync_failed(
    store, monkeypatch
):
    seen = {"n": 0}

    def second_attempts_fsync(p):  # 1st = execution claim, 2nd = candidate claim
        if p.name == P.ATTEMPTS_DIR:
            seen["n"] += 1
            return seen["n"] == 2
        return False

    trace = FsyncTrace(monkeypatch, fail=second_attempts_fsync)
    assert _commit(store).status == "failed"
    trace.reset()
    again = _commit(store)
    assert again.status == "available"
    assert store.root / P.ATTEMPTS_DIR in trace.ok


def test_concurrent_identical_writer_branch_still_confirms_the_entry(
    store, monkeypatch
):
    """os.link loses to an identical concurrent writer; the entry must still be fsynced."""
    trace = FsyncTrace(monkeypatch)
    real_link = store_mod.os.link

    def racing_link(src, dst):
        if f"/{P.SOURCES_DIR}/" in str(dst):
            real_link(src, dst)  # the "other writer" wins with identical bytes
            raise FileExistsError(dst)
        return real_link(src, dst)

    monkeypatch.setattr(store_mod.os, "link", racing_link)
    src = SourcePayload(b'{"rows":[]}', {"regeneration": []})
    res = _commit(store, source=src)
    assert res.status == "available"
    assert store.root / P.SOURCES_DIR in trace.ok


# --- 2. a failed final-marker fsync: outcome and fresh reads cohere (PENDING) --------------
#
# Agreed semantics: the marker is never rolled back; an
# uncertain acknowledgement is the existing PENDING status; list() items carry a metadata
# projection availability=available|pending; read()/read_record() of a pending ref raise
# DurabilityError (never "not found"); a retry/reader re-confirms before available; the
# durability memo is per immutable marker identity and caches nothing else.


def _marker_linked(p):
    return p.name.startswith("r_") and (p / P.REVISION_FILE).exists()


def _claimed_ref(store):
    return store._read_claim(
        store.root / P.ATTEMPTS_DIR / f"{store_mod._key_hash(_cand().attempt_key)}.json"
    )["ref"]


def test_failed_marker_fsync_is_pending_and_fresh_reads_cohere(
    tmp_path, store, monkeypatch
):
    FsyncTrace(monkeypatch, fail=_marker_linked, once=False)  # persistent I/O failure
    first = _commit(store)
    assert (first.status, first.reason) == (
        "pending",
        "durability_failed:dir_fsync_failed",
    )
    assert first.ref is None and first.path is None
    claimed = _claimed_ref(store)
    fresh = LocalFigureStore(tmp_path, session_id="s1")
    [item] = fresh.list().items  # listed, never omitted; explicitly pending
    assert (item["availability"], item["availability_reason"]) == (
        "pending",
        "durability_failed:dir_fsync_failed",
    )
    for reader in (fresh.read_record, fresh.read):
        with pytest.raises(DurabilityError):  # explicit, never FigureNotFound
            reader(claimed)
    monkeypatch.undo()  # the disk recovers: the next reader re-confirms
    [item] = LocalFigureStore(tmp_path, session_id="s1").list().items
    assert item["availability"] == "available" and "availability_reason" not in item
    # revision.json itself was never rewritten by the projection
    marker = store._rev_dir(FigureRef.parse(claimed)) / P.REVISION_FILE
    assert "availability" not in json.loads(marker.read_bytes())


def test_failed_marker_fsync_retry_completes_the_same_claimed_ref(store, monkeypatch):
    FsyncTrace(monkeypatch, fail=_marker_linked)
    first = _commit(store)
    assert first.status == "pending"
    claimed = _claimed_ref(store)
    again = _commit(store)
    assert (again.status, again.ref) == ("available", claimed)
    assert [r["revision_id"] for r in store.list().items] == [
        FigureRef.parse(claimed).revision_id
    ]


def test_failed_marker_fsync_is_pending_for_the_plot_manager(tmp_path, monkeypatch):
    import plotly.graph_objects as go

    from lobster.core.figures.saver import HEADER_PENDING, summarize
    from lobster.core.runtime.data_manager import DataManagerV2

    dm = DataManagerV2(workspace_path=tmp_path)
    FsyncTrace(monkeypatch, fail=_marker_linked)
    plot_id, _, _ = dm.plot_manager.add_plot(plot=go.Figure(go.Bar(y=[1])), title="t")
    results = dm.plot_manager.commit_staged(
        {"id": "a"}, tool_call_id="c1", staged_ids=[plot_id]
    )  # all-pending does not raise
    assert [r.status for r in results] == ["pending"]
    assert summarize(results).startswith(HEADER_PENDING)
    assert HEADER_PENDING == "Save confirmation pending."
    assert dm.plot_manager.latest_plots == []  # only confirmed figures are active


def test_reuse_reconfirmation_failure_is_pending_and_never_revokes_a_confirmed_result(
    tmp_path, store, monkeypatch
):
    first = _commit(store)
    assert first.status == "available"
    FsyncTrace(monkeypatch, fail=lambda p: p.name.startswith("r_"), once=False)
    # the instance that confirmed it keeps its receipt (memo = durability only)...
    assert _commit(store).status == "available"
    # ...a fresh instance cannot confirm now: PENDING, explicit, nothing removed
    other = LocalFigureStore(tmp_path, session_id="s1")
    again = _commit(other)
    assert (again.status, again.reason) == (
        "pending",
        "durability_failed:dir_fsync_failed",
    )
    assert [r["revision_id"] for r in other.list().items] == [
        FigureRef.parse(first.ref).revision_id
    ]
    store.read(first.ref)  # full read still verifies bytes on the confirming instance


def test_edit_from_a_pending_parent_is_refused_as_durability_not_deleted(
    tmp_path, store, monkeypatch
):
    from .test_core_review_fixes import _edit

    first = _commit(store)
    other = LocalFigureStore(tmp_path, session_id="s1")
    FsyncTrace(monkeypatch, fail=lambda p: p.name.startswith("r_"), once=False)
    cand = _edit(first.ref, ())
    [res] = TrustedSaver(other).commit_attempt(
        _info(base="s1:omni_1"), [cand], succeeded=True
    )
    assert res.status != "available"
    assert res.reason.endswith("durability_failed:dir_fsync_failed")
    assert "parent" not in res.reason


def test_pending_revision_is_listed_but_never_materialized_for_a_child(
    tmp_path, store, monkeypatch
):
    from lobster.core.figures.snapshot import build_snapshot

    first = _commit(store)
    other = LocalFigureStore(tmp_path, session_id="s1")
    FsyncTrace(monkeypatch, fail=lambda p: p.name.startswith("r_"), once=False)
    index = build_snapshot(other, f"load_figure('{first.ref}')", tmp_path / "snap")
    assert index["status"][first.ref] == {
        "status": "pending",
        "reason": "dir_fsync_failed",
    }
    assert not (tmp_path / "snap" / FigureRef.parse(first.ref).figure_id).exists()


def test_execution_outcomes_report_pending_without_a_public_ref(
    tmp_path, store, monkeypatch
):
    first = _commit(store)
    execution_id = store.read_record(first.ref)["execution"]["execution_id"]
    other = LocalFigureStore(tmp_path, session_id="s1")
    FsyncTrace(monkeypatch, fail=lambda p: p.name.startswith("r_"), once=False)
    assert other.execution_outcomes(execution_id) == [
        {"index": 0, "status": "pending", "ref": None}
    ]


def test_failed_rendition_record_fsync_is_pending(store, monkeypatch):
    from lobster.core.figures.renditions import ensure_html_rendition

    res = _commit(store)
    FsyncTrace(
        monkeypatch,
        fail=lambda p: p.name == "renditions" and any(p.glob("p_*.json")),
    )
    view = ensure_html_rendition(store, res.ref)
    assert view.status == "pending" and view.path is None
    assert view.reason == "html_rendition:durability_failed:dir_fsync_failed"
    monkeypatch.undo()
    again = ensure_html_rendition(store, res.ref)  # reconfirms; no duplicate appended
    assert again.status == "available" and again.reused
    assert len(store.list_renditions(res.ref)) == 1


def test_new_rendition_marker_after_a_warm_memo_is_confirmed_on_its_own(
    store, monkeypatch
):
    """The memo is per marker identity: p1's receipt never covers a later p2."""
    from lobster.core.figures import renditions as R

    res = _commit(store)
    p1 = R.ensure_html_rendition(store, res.ref)
    assert p1.status == "available"
    assert store.list_renditions(res.ref)[0]["availability"] == "available"  # warm memo
    rec = store.read_record(res.ref)
    receipt = {
        "renderer": {"name": "test"},
        "inputs": {
            "figure_sha256": rec["figure"]["sha256"],
            "config_sha256": rec["config"]["sha256"],
        },
    }
    FsyncTrace(  # only once p2's marker exists: p2's data file write still confirms
        monkeypatch,
        fail=lambda p: p.name == "renditions" and len(list(p.glob("p_*.json"))) >= 2,
        once=False,
    )
    p2 = store.add_rendition(
        res.ref, b"<html>2</html>", {"format": "html", "v": 2}, receipt
    )
    assert p2["availability"] == "pending"
    by_id = {
        r["rendition_id"]: r["availability"] for r in store.list_renditions(res.ref)
    }
    assert by_id == {p1.rendition_id: "available", p2["rendition_id"]: "pending"}


# --- 2b. concurrency: an observed or acknowledged marker never disappears -----


class PausedMarkerFsync:
    """Writer A (its own thread) links a marker, then pauses inside the marker's directory
    fsync until released, and then fails it. Every other thread's fsync is real."""

    def __init__(self, monkeypatch, is_marker_dir):
        import threading

        self.real = store_mod._fsync_dir
        self.is_marker_dir = is_marker_dir
        self.writer = None
        self.linked = threading.Event()
        self.release = threading.Event()
        monkeypatch.setattr(store_mod, "_fsync_dir", self)

    def __call__(self, path):
        import threading

        if threading.get_ident() == self.writer and self.is_marker_dir(path):
            self.linked.set()
            assert self.release.wait(10), "test barrier timed out"
            raise DurabilityError("injected", details={"reason": "dir_fsync_failed"})
        return self.real(path)

    def run_writer(self, fn):
        import threading

        box = {}

        def target():
            self.writer = threading.get_ident()
            try:
                box["result"] = fn()
            except Exception as exc:  # recorded, inspected by the test
                box["error"] = exc

        t = threading.Thread(target=target)
        t.start()
        assert self.linked.wait(10), "writer never linked its marker"
        return t, box


def _try_writer_lock(store) -> bool:
    """Non-blocking probe; released at once if it was (unexpectedly) acquired, so a failed
    assertion never leaves the lock held."""
    got = store.writer_lock.acquire(blocking=False)
    if got:
        store.writer_lock.release()
    return got


def test_revision_acknowledged_by_a_retry_never_disappears(tmp_path, monkeypatch):
    """Writer lock: the writers of one root are serialized in-process, so a retry can no longer
    interleave with A's paused marker fsync (only a reader can). The retry AFTER A's
    failed confirmation acknowledges the linked marker, which never disappears.

    Historical record: this was
    ``test_revision_acknowledged_by_a_concurrent_retry_never_disappears``. Its intent was a
    second writer acknowledging WHILE A was paused, before A's fault. Under the lock that schedule
    cannot occur in-process, so it is NOT claimed here: no concurrent pre-fault
    acknowledgment is proven. The same applies to the rendition test below."""
    store_a = LocalFigureStore(tmp_path, session_id="s1")
    store_b = LocalFigureStore(tmp_path, session_id="s1")
    gate = PausedMarkerFsync(monkeypatch, _marker_linked)
    t, box = gate.run_writer(lambda: _commit(store_a))
    # while A is paused after linking: a reader (no writer lock) sees the complete marker...
    [seen] = store_b.list().items
    # ...and no second writer can interleave: A holds the root's one writer lock
    assert not _try_writer_lock(store_b)
    gate.release.set()
    t.join(10)
    assert not t.is_alive() and "error" not in box
    assert box["result"].status == "pending"  # A's confirmation failed: PENDING
    # a retry under the same claim (second store instance) acknowledges it
    acked = _commit(store_b)
    assert acked.status == "available"
    # an acknowledged / observed revision must still exist after A's failure
    assert [r["revision_id"] for r in store_b.list().items] == [seen["revision_id"]]
    store_b.read(acked.ref)  # bytes still verify


def test_rendition_acknowledged_by_a_retry_never_disappears(tmp_path, monkeypatch):
    from lobster.core.figures.renditions import ensure_html_rendition

    store_a = LocalFigureStore(tmp_path, session_id="s1")
    store_b = LocalFigureStore(tmp_path, session_id="s1")
    ref = _commit(store_a).ref
    gate = PausedMarkerFsync(
        monkeypatch, lambda p: p.name == "renditions" and any(p.glob("p_*.json"))
    )
    t, box = gate.run_writer(lambda: ensure_html_rendition(store_a, ref))
    # B's writer cannot interleave with A's paused rendition (one writer lock)
    assert not _try_writer_lock(store_b)
    gate.release.set()
    t.join(10)
    assert not t.is_alive() and "error" not in box
    acked = ensure_html_rendition(store_b, ref)  # the retry after A's failure
    assert acked.status == "available"
    assert [r["rendition_id"] for r in store_b.list_renditions(ref)] == [
        acked.rendition_id
    ]


def test_a_recreated_marker_never_inherits_the_durability_receipt(store, monkeypatch):
    """Memo key = marker file identity (dev/inode/size/mtime), not the ref string."""
    res = _commit(store)
    store.read(res.ref)  # warm receipt
    marker = store._rev_dir(FigureRef.parse(res.ref)) / P.REVISION_FILE
    data = marker.read_bytes()
    marker.unlink()  # simulate replacement by another writer (test-only)
    marker.write_bytes(data)
    trace = FsyncTrace(monkeypatch, fail=lambda p: p.name.startswith("r_"), once=False)
    with pytest.raises(DurabilityError):
        store.read_record(res.ref)
    assert trace.failed  # re-confirmation was attempted, not skipped by the memo


# --- 2c. bounded I/O faults at the exact pending seams ---------------------------


def test_marker_unreadable_after_a_failed_confirmation_is_still_pending(
    store, monkeypatch
):
    """No post-failure re-read decides the outcome: a failing read cannot turn PENDING into
    FAILED / not found."""
    import pathlib

    real_read, real_is_file = pathlib.Path.read_bytes, pathlib.Path.is_file

    def broken(fn):
        def wrapped(self, *a, **k):
            if self.name == P.REVISION_FILE:
                raise OSError(5, "EIO")
            return fn(self, *a, **k)

        return wrapped

    FsyncTrace(monkeypatch, fail=_marker_linked)
    monkeypatch.setattr(pathlib.Path, "read_bytes", broken(real_read))
    monkeypatch.setattr(pathlib.Path, "is_file", broken(real_is_file))
    res = _commit(store)
    assert (res.status, res.reason) == ("pending", "durability_failed:dir_fsync_failed")


def test_rendition_marker_unreadable_after_a_failed_confirmation_is_still_pending(
    store, monkeypatch
):
    import pathlib

    res = _commit(store)
    rec = store.read_record(res.ref)
    receipt = {
        "renderer": {"name": "test"},
        "inputs": {
            "figure_sha256": rec["figure"]["sha256"],
            "config_sha256": rec["config"]["sha256"],
        },
    }
    real_read = pathlib.Path.read_bytes

    def read_bytes(self, *a, **k):
        if self.name.startswith("p_") and self.suffix == ".json":
            raise OSError(5, "EIO")
        return real_read(self, *a, **k)

    FsyncTrace(
        monkeypatch,
        fail=lambda p: p.name == "renditions" and any(p.glob("p_*.json")),
    )
    monkeypatch.setattr(pathlib.Path, "read_bytes", read_bytes)
    added = store.add_rendition(res.ref, b"<html/>", {"format": "html"}, receipt)
    assert added["availability"] == "pending"
    assert added["availability_reason"] == "durability_failed:dir_fsync_failed"


def test_memo_bookkeeping_failure_never_undoes_a_successful_ack(store, monkeypatch):
    def stat_fails(path):
        raise OSError(5, "EIO")

    monkeypatch.setattr(store_mod, "_marker_identity", stat_fails)
    res = _commit(store)
    assert res.status == "available" and res.ref  # the ACK stands
    monkeypatch.undo()
    trace = FsyncTrace(monkeypatch)
    store.read(res.ref)  # no receipt was kept: the read re-confirms, then verifies
    assert trace.ok


def test_marker_stat_failure_on_read_or_list_is_an_explicit_storage_error(
    store, monkeypatch
):
    from lobster.core.figures.errors import FigureNotFoundError, StorageError

    res = _commit(store)
    other = LocalFigureStore(store.workspace, session_id="s1")  # no receipt yet

    def stat_fails(path):
        raise OSError(5, "EIO")

    monkeypatch.setattr(store_mod, "_marker_identity", stat_fails)
    for reader in (other.read_record, other.read):
        with pytest.raises(StorageError) as info:
            reader(res.ref)
        assert not isinstance(info.value, FigureNotFoundError)
    [item] = other.list().items  # listed, never a false absence
    assert (item["availability"], item["availability_reason"]) == (
        "pending",
        "storage_failed",
    )


# --- 2d. real first stat/read I/O and corrupt registered records -----------------
#
# Typed StorageError / IntegrityError at the exact seams: never a raw OSError, never a false
# "not found", never an empty or partial catalog, never a fabricated pending item.


def _eio_for(name, real):
    def wrapped(path, *a, **k):
        if os.fspath(path).endswith(name):
            raise OSError(5, "EIO /secret/path")
        return real(path, *a, **k)

    return wrapped


def test_first_marker_stat_eio_on_read_is_a_typed_storage_error(store, monkeypatch):
    from lobster.core.figures.errors import (
        FigureNotFoundError,
        StorageError,
        safe_reason,
    )

    res = _commit(store)
    other = LocalFigureStore(store.workspace, session_id="s1")
    # the marker is stat'ed without following a link (lstat), so the fault goes there
    monkeypatch.setattr(os, "lstat", _eio_for(P.REVISION_FILE, os.lstat))
    for reader in (other.read_record, other.read):
        with pytest.raises(StorageError) as info:
            reader(res.ref)
        assert not isinstance(info.value, FigureNotFoundError)
        assert safe_reason(info.value) == "storage_failed"
    with pytest.raises(StorageError):
        other.list()  # no empty catalog, no pending item fabricated


def test_marker_read_eio_while_listing_is_a_typed_storage_error(store, monkeypatch):
    from lobster.core.figures.errors import StorageError

    _commit(store)
    real = os.open

    def no_follow_open(path, flags, *a, **k):  # markers are read with O_NOFOLLOW
        if os.fspath(path).endswith(P.REVISION_FILE):
            raise OSError(5, "EIO")
        return real(path, flags, *a, **k)

    monkeypatch.setattr(os, "open", no_follow_open)
    with pytest.raises(StorageError):
        LocalFigureStore(store.workspace, session_id="s1").list()


def test_catalog_enumeration_eio_is_a_typed_storage_error(store, monkeypatch):
    from lobster.core.figures.errors import StorageError

    _commit(store)
    real = os.listdir

    def listdir(path):  # the catalog is enumerated by name, then lstat'ed per entry
        if os.fspath(path) == os.fspath(store.root):
            raise OSError(5, "EIO")
        return real(path)

    monkeypatch.setattr(os, "listdir", listdir)
    with pytest.raises(StorageError):
        store.list()


@pytest.mark.parametrize(
    "corrupt, reason",
    [
        (lambda rec: b"{not json", "fingerprint_mismatch:invalid_json"),
        (lambda rec: json.dumps([rec]).encode(), "fingerprint_mismatch:shape"),
        (
            lambda rec: json.dumps({**rec, "revision_id": "r_elsewhere0"}).encode(),
            "fingerprint_mismatch:shape",
        ),
        (
            lambda rec: json.dumps({**rec, "figure": {"bytes": 1}}).encode(),
            "fingerprint_mismatch:shape",
        ),
    ],
)
def test_a_corrupt_registered_revision_is_an_integrity_error_never_skipped(
    store, corrupt, reason
):
    from lobster.core.figures.errors import FigureNotFoundError, safe_reason

    keep = _commit(store)
    [bad] = TrustedSaver(store).commit_attempt(
        _info(base="s1:call_9"), [_cand(base="s1:call_9")], succeeded=True
    )
    marker = store._rev_dir(FigureRef.parse(bad.ref)) / P.REVISION_FILE
    marker.write_bytes(corrupt(json.loads(marker.read_bytes())))
    fresh = LocalFigureStore(store.workspace, session_id="s1")
    with pytest.raises(IntegrityError) as info:
        fresh.list()  # not a silently shorter catalog
    assert safe_reason(info.value) == reason
    for reader in (fresh.read_record, fresh.read):
        with pytest.raises(IntegrityError) as info:
            reader(bad.ref)
        assert not isinstance(info.value, FigureNotFoundError)
    fresh.read(keep.ref)  # an intact revision still reads


def test_a_missing_payload_after_the_marker_is_an_integrity_error(store):
    from lobster.core.figures.errors import safe_reason

    res = _commit(store)
    (store._rev_dir(FigureRef.parse(res.ref)) / P.FIGURE_FILE).unlink()
    with pytest.raises(IntegrityError) as info:
        store.read(res.ref)
    assert safe_reason(info.value) == "fingerprint_mismatch:figure_changed"


def test_a_corrupt_rendition_marker_is_reported_never_skipped_or_duplicated(store):
    from lobster.core.figures.renditions import ensure_html_rendition

    res = _commit(store)
    first = ensure_html_rendition(store, res.ref)
    assert first.status == "available"
    rdir = store._renditions_dir(FigureRef.parse(res.ref))
    (rdir / f"{first.rendition_id}.json").write_bytes(b"{broken")
    with pytest.raises(IntegrityError):
        store.list_renditions(res.ref)
    view = ensure_html_rendition(store, res.ref)
    assert (view.status, view.reason) == (
        "failed",
        "html_rendition:fingerprint_mismatch:invalid_json",
    )
    assert len(list(rdir.glob("p_*.json"))) == 1  # nothing appended over the damage
