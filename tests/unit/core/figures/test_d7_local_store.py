"""Local figure deletion, store side.

LOCAL only: one process per local workspace directory. Real filesystem (tmp_path), real store.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path

import pytest

from lobster.core.figures import paths as P
from lobster.core.figures import store as store_module
from lobster.core.figures.errors import (
    DurabilityError,
    FigureDeletedError,
    FigureNotFoundError,
    GenerationFencedError,
    IntegrityError,
    ManagedPathError,
    StorageError,
    safe_reason,
)
from lobster.core.figures.ids import PREFIX_FIGURE, PREFIX_REVISION, FigureRef, new_id
from lobster.core.figures.payload import source_to_bytes
from lobster.core.figures.records import STATUS_REFUSED, InputEdge, SourcePayload
from lobster.core.figures.renditions import ensure_html_rendition
from lobster.core.figures.saver import TrustedSaver
from lobster.core.figures.store import LocalFigureStore
from tests.unit.core.figures.test_store_and_saver import _cand, _fig, _info


@pytest.fixture()
def store(tmp_path):
    return LocalFigureStore(tmp_path, session_id="s1")


# ---------------------------------------------------------------------- writer lock


def test_every_instance_of_one_root_shares_one_writer_lock(tmp_path):
    (tmp_path / "ws").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "ws", target_is_directory=True)
    a = LocalFigureStore(tmp_path / "ws")
    b = LocalFigureStore(tmp_path / "ws", session_id="s2")
    via_link = LocalFigureStore(tmp_path / "link")
    other = LocalFigureStore(tmp_path / "other")
    assert a._lock is b._lock is via_link._lock
    assert other._lock is not a._lock


def test_every_store_write_runs_under_the_root_lock(store, monkeypatch):
    held = []
    real_write, real_link = (
        store_module._write_no_replace,
        store_module._link_no_replace,
    )

    def write(target, data):
        held.append(("write", target.name, store._lock._is_owned()))
        return real_write(target, data)

    def link(target, data):
        held.append(("link", target.name, store._lock._is_owned()))
        return real_link(target, data)

    monkeypatch.setattr(store_module, "_write_no_replace", write)
    monkeypatch.setattr(store_module, "_link_no_replace", link)
    [res] = TrustedSaver(store).commit_attempt(_info(), [_cand()], succeeded=True)
    assert res.status == "available"
    view = ensure_html_rendition(store, res.ref)
    assert view.status == "available"
    names = {name for _, name, _ in held}
    assert "revision.json" in names  # the commit marker
    assert any(n.endswith(".html") for n in names)  # the rendition bytes
    assert held and all(owned for _, _, owned in held)
    assert not store._lock._is_owned()  # released after every call


def test_a_second_thread_waits_for_the_writer_lock(tmp_path):
    first = LocalFigureStore(tmp_path, session_id="s1")
    second = LocalFigureStore(tmp_path, session_id="s1")  # a different instance
    out = []

    def save():
        out.extend(
            TrustedSaver(second).commit_attempt(_info(), [_cand()], succeeded=True)
        )

    with first._lock:
        worker = threading.Thread(target=save)
        worker.start()
        worker.join(0.5)
        assert worker.is_alive()  # blocked on the shared lock
        assert out == [] and first.list().items == []
    worker.join(30)
    assert not worker.is_alive()
    assert [r.status for r in out] == ["available"]


# ----------------------------------------------------- fences, removal, no resurrection


def _save(store, base, instance_id="inst-1"):
    inputs = (InputEdge("prot_de", instance_id, "store_event", True),)
    [res] = TrustedSaver(store).commit_attempt(
        _info(base=base), [_cand(base=base, inputs=inputs)], succeeded=True
    )
    return res


def _listed(store):
    return [f"{r['figure_id']}@{r['revision_id']}" for r in store.list().items]


def test_a_dataset_fence_hides_then_removal_deletes_only_the_named_revisions(store):
    a = _save(store, "s1:call_a", "inst-1")
    b = _save(store, "s1:call_b", "inst-2")
    report = store.fence_deletion([a.ref], instance_ids=["inst-1"])
    assert report.complete and (report.confirmed, report.unconfirmed) == (2, 0)
    for read in (store.read, store.read_record, store.list_renditions):
        with pytest.raises(FigureNotFoundError):
            read(a.ref)
    assert _listed(store) == [b.ref]
    assert store.dependents("inst-1") == []
    ref_a = FigureRef.parse(a.ref)
    assert store._rev_dir(ref_a).is_dir()  # hidden, not yet physically removed
    removed = store.remove_fenced([a.ref])
    assert (removed.removed, removed.failed) == (1, 0)
    assert not (store.root / ref_a.figure_id).exists()
    assert store.read(b.ref)[2] is not None  # the other revision is intact
    again = store.fence_deletion([a.ref], instance_ids=["inst-1"])  # idempotent re-run
    assert again.complete and again.confirmed == 2


def test_a_retry_of_a_deleted_revision_is_refused_and_never_republished(store):
    a = _save(store, "s1:call_a")
    assert store.fence_deletion([a.ref]).complete
    # fenced, not yet removed: the claimed-revision check refuses before reuse
    again = _save(store, "s1:call_a")
    assert (again.status, again.reason) == (STATUS_REFUSED, "figure_deleted")
    store.remove_fenced([a.ref])
    # removed (and its execution payload cleaned, fenced first): refused earlier
    again = _save(store, "s1:call_a")
    assert (again.status, again.reason) == (
        STATUS_REFUSED,
        "execution_record:generation_fenced",
    )
    assert _listed(store) == []
    assert not (store.root / FigureRef.parse(a.ref).figure_id).exists()


def test_an_interrupted_commit_that_used_a_deleted_input_is_never_completed(store):
    a = _save(store, "s1:call_a", "inst-1")
    marker = store._rev_dir(FigureRef.parse(a.ref)) / P.REVISION_FILE
    os.unlink(marker)  # claim present, marker absent
    assert store.fence_deletion([], instance_ids=["inst-1"]).complete
    again = _save(store, "s1:call_a", "inst-1")
    assert (again.status, again.reason) == (STATUS_REFUSED, "generation_fenced")
    assert not marker.exists()


def test_a_late_commit_using_a_deleted_input_is_fenced_and_others_are_not(store):
    assert store.fence_deletion([], instance_ids=["inst-1"]).complete
    late = _save(store, "s1:call_late", "inst-1")
    other = _save(store, "s1:call_other", "inst-2")
    assert (late.status, late.reason) == (STATUS_REFUSED, "generation_fenced")
    assert other.status == "available"
    # an UNVERIFIED edge with the same id is not fenced ("may depend" figures are kept)
    unverified = (InputEdge("alias", "inst-1", "declared", False),)
    [kept] = TrustedSaver(store).commit_attempt(
        _info(base="s1:call_u"),
        [_cand(base="s1:call_u", inputs=unverified)],
        succeeded=True,
    )
    assert kept.status == "available"


def test_an_edit_of_a_deleted_parent_saves_nothing(store):
    a = _save(store, "s1:call_a")
    assert store.fence_deletion([a.ref]).complete
    [edit] = TrustedSaver(store).commit_attempt(
        _info(base="s1:call_e"),
        [_cand(base="s1:call_e", parent_ref=a.ref, kind="edited")],
        succeeded=True,
    )
    # the saver reads the parent first (prepare): a refusal, never a failure
    assert (edit.status, edit.reason) == (STATUS_REFUSED, "figure_deleted")
    assert _listed(store) == []


def test_fencing_runs_under_the_writer_lock(store, monkeypatch):
    a = _save(store, "s1:call_a")
    held = []
    real_link = store_module._link_no_replace

    def link(target, data):
        held.append(store._lock._is_owned())
        return real_link(target, data)

    monkeypatch.setattr(store_module, "_link_no_replace", link)
    store.fence_deletion([a.ref], instance_ids=["inst-1"])
    store.fence_purge()
    assert held and all(held)


def test_an_unconfirmed_tombstone_hides_the_ref_and_is_reported_pending(
    store, monkeypatch
):
    a = _save(store, "s1:call_a")
    attempts = store.root / P.ATTEMPTS_DIR
    real_fsync = store_module._fsync_dir

    def fsync(path):
        if Path(path) == attempts:
            raise DurabilityError("simulated", details={"reason": "dir_fsync_failed"})
        return real_fsync(path)

    monkeypatch.setattr(store_module, "_fsync_dir", fsync)
    report = store.fence_deletion([a.ref])
    monkeypatch.undo()
    assert (report.confirmed, report.unconfirmed, report.complete) == (0, 1, False)
    with pytest.raises(FigureNotFoundError):
        store.read(a.ref)  # invisible wherever the tombstone is observed
    assert store._rev_dir(FigureRef.parse(a.ref)).is_dir()  # nothing removed


def test_a_failed_fence_write_stops_the_batch_and_reports_only_what_is_known(
    store, monkeypatch
):
    a = _save(store, "s1:call_a")
    b = _save(store, "s1:call_b")
    real_link = store_module._link_no_replace
    calls = {"n": 0}

    def link(target, data):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated")
        return real_link(target, data)

    monkeypatch.setattr(store_module, "_link_no_replace", link)
    report = store.fence_deletion([a.ref, b.ref])
    monkeypatch.undo()
    assert (report.confirmed, report.unconfirmed, report.stopped) == (1, 0, True)
    assert report.failed_entry_may_exist and report.not_attempted == 0
    assert not report.complete
    assert _listed(store) == [b.ref]  # a is hidden; b's state is reported unknown


def test_removal_needs_a_fence_and_never_follows_a_symlink(store, tmp_path):
    a = _save(store, "s1:call_a")
    with pytest.raises(StorageError):
        store.remove_fenced([a.ref])
    assert store.read(a.ref)
    b = _save(store, "s1:call_b")
    rev = store._rev_dir(FigureRef.parse(b.ref))
    outside = tmp_path / "outside"
    shutil.move(str(rev), str(outside))
    rev.symlink_to(outside, target_is_directory=True)
    assert store.fence_deletion([b.ref]).complete
    report = store.remove_fenced([b.ref])
    assert (report.removed, report.failed) == (0, 1)
    assert (outside / P.REVISION_FILE).is_file()  # the symlink target is untouched


def test_a_purge_fences_every_claim_first_and_keeps_only_the_fences(
    store, tmp_path, monkeypatch
):
    a = _save(store, "s1:call_a")
    crash = {"on": True}
    real_commit = store.commit

    def commit(*args, **kwargs):  # stage (ii): execution recorded, no revision claim
        if crash["on"]:
            raise RuntimeError("simulated crash after the execution record")
        return real_commit(*args, **kwargs)

    monkeypatch.setattr(store, "commit", commit)
    [mid] = TrustedSaver(store).commit_attempt(
        _info(base="s1:call_b"), [_cand(base="s1:call_b")], succeeded=True
    )
    assert mid.status == "failed"
    crash["on"] = False
    (store.root / "legacy.png").write_bytes(b"legacy")

    record, tombs = store.fence_purge()  # fence only: nothing is removed here
    assert record.complete and tombs is not None and tombs.complete
    assert tombs.confirmed == 3  # EXEC a, REV a, EXEC b (the directory ref is the same)
    attempts = store.root / P.ATTEMPTS_DIR
    assert len([p for p in attempts.iterdir() if p.name.startswith("purge-")]) == 1
    assert _listed(store) == []
    assert (store.root / "legacy.png").is_file()
    report = store.purge()  # a new action: fence again, re-verify, remove
    assert report.complete and report.fence_verified
    assert report.tombstones.confirmed == 3 and report.removal.failed == 0
    assert len([p for p in attempts.iterdir() if p.name.startswith("purge-")]) == 2
    assert sorted(p.name for p in store.root.iterdir()) == [P.ATTEMPTS_DIR]

    again_a = _save(store, "s1:call_a")
    again_b = _save(store, "s1:call_b")
    for again in (again_a, again_b):
        assert (again.status, again.reason) == (
            STATUS_REFUSED,
            "execution_record:generation_fenced",
        )
    fresh = _save(store, "s1:call_new")
    assert fresh.status == "available"  # a NEW attempt after the purge is a new figure


def test_a_purge_with_zero_claims_still_writes_its_record(tmp_path):
    empty = LocalFigureStore(tmp_path / "empty")
    record, tombs = empty.fence_purge()
    assert record.complete and tombs is not None and tombs.complete
    assert tombs.planned == 0
    names = [p.name for p in (empty.root / P.ATTEMPTS_DIR).iterdir()]
    assert len(names) == 1 and names[0].startswith("purge-")


def test_an_unreadable_claim_stops_the_purge_before_any_tombstone(store):
    a = _save(store, "s1:call_a")
    (store.root / P.ATTEMPTS_DIR / ("0" * 64 + ".json")).write_bytes(b"{not json")
    record, tombs = store.fence_purge()
    assert record.complete
    assert tombs.inventory_unreadable and not tombs.complete
    assert tombs.not_attempted is None
    assert store.read(a.ref)  # nothing hidden: no tombstone was written


def test_a_purge_refuses_a_symlinked_store_root(tmp_path):
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "plots").symlink_to(
        tmp_path / "elsewhere", target_is_directory=True
    )
    with pytest.raises(StorageError):
        LocalFigureStore(tmp_path / "ws").fence_purge()
    assert list((tmp_path / "elsewhere").iterdir()) == []


# ------------------------------------------------ payload cleanup after the last live ref


def _src(tag):
    return SourcePayload(source_to_bytes([{"k": tag}]), {"comparison": tag})


def _save_n(store, base, code, n=1, source=None, instance_id="inst-1"):
    inputs = (InputEdge("prot_de", instance_id, "store_event", True),)
    cands = [
        _cand(index=i, base=base, inputs=inputs, source=source, fig=_fig(title=f"T{i}"))
        for i in range(n)
    ]
    return TrustedSaver(store).commit_attempt(
        _info(code=code, base=base), cands, succeeded=True
    )


def _execution_id(store, ref):
    return store.read_record(ref)["execution"]["execution_id"]


def _delete(store, refs):
    assert store.fence_deletion(refs).complete
    return store.remove_fenced(refs)


def test_cleanup_removes_payload_only_after_the_last_live_reference(store):
    [a] = _save_n(store, "s1:call_a", b"a = 1\n", source=_src("shared"))
    [b] = _save_n(store, "s1:call_b", b"b = 2\n", source=_src("shared"))
    exec_a, exec_b = _execution_id(store, a.ref), _execution_id(store, b.ref)
    code_a = store.read_execution(exec_a)["code_id"]
    source_id = store.read_record(a.ref)["source"]["source_id"]
    executions = store.root / P.EXECUTIONS_DIR

    report = _delete(store, [a.ref])
    assert (report.removed, report.failed) == (1, 0) and not report.cleanup_pending
    assert (report.cleanup.executions, report.cleanup.code) == (1, 1)
    assert report.cleanup.sources == 0  # b still uses the shared source
    assert not (executions / f"{exec_a}.json").exists()
    assert not (executions / f"{exec_a}.activity.json").exists()
    assert not (store.root / P.CODE_DIR / f"{code_a}.py").exists()
    assert store.read_source(source_id)  # kept
    assert store.read(b.ref)  # the other figure is intact
    assert store.read_execution(exec_b)["code_id"]  # and its execution too

    # minimal fences stay: the claims are retained; a retry never re-creates the payload
    again = _save_n(store, "s1:call_a", b"a = 1\n", source=_src("shared"))[0]
    assert (again.status, again.reason) == (
        STATUS_REFUSED,
        "execution_record:generation_fenced",
    )
    assert not (executions / f"{exec_a}.json").exists()

    report = _delete(store, [b.ref])
    assert report.cleanup.sources == 1
    with pytest.raises(FigureNotFoundError):
        store.read_source(source_id)


def test_an_execution_stays_while_any_of_its_candidates_is_live(store):
    r0, r1 = _save_n(store, "s1:call_m", b"m = 1\n", n=2)
    execution_id = _execution_id(store, r0.ref)
    first = _delete(store, [r0.ref])
    assert first.cleanup.executions == 0
    assert store.read_execution(execution_id)  # r1 still needs it
    second = _delete(store, [r1.ref])
    assert second.cleanup.executions == 1
    with pytest.raises(FigureNotFoundError):
        store.read_execution(execution_id)


def test_shared_code_stays_while_another_execution_uses_it(store):
    [a] = _save_n(store, "s1:call_a", b"same = 1\n")
    [b] = _save_n(store, "s1:call_b", b"same = 1\n")
    code_id = store.read_execution(_execution_id(store, b.ref))["code_id"]
    report = _delete(store, [a.ref])
    assert (report.cleanup.executions, report.cleanup.code) == (1, 0)
    assert (store.root / P.CODE_DIR / f"{code_id}.py").is_file()
    assert store.read(b.ref)


# ----------------------------------------------- the purge inventory is read strictly


@pytest.mark.parametrize(
    "kind", ["symlink", "figure_id_symlink", "unknown_dir", "fifo"]
)
def test_an_unknown_or_symlinked_plots_entry_refuses_the_purge(store, tmp_path, kind):
    a = _save(store, "s1:call_a")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file.txt").write_bytes(b"keep")
    if kind == "symlink":
        (store.root / "link").symlink_to(outside, target_is_directory=True)
    elif kind == "figure_id_symlink":  # readers would follow it: never skipped
        (store.root / new_id(PREFIX_FIGURE)).symlink_to(
            outside, target_is_directory=True
        )
    elif kind == "unknown_dir":  # not an ID (IDs are opaque [a-z0-9_-]{1,64})
        (store.root / "old.runs").mkdir()
    else:
        os.mkfifo(store.root / "pipe")
    record, tombs = store.fence_purge()
    assert record.complete
    assert tombs.inventory_unreadable and not tombs.complete
    assert store.read(a.ref)  # nothing hidden: no tombstone was written
    assert (outside / "file.txt").read_bytes() == b"keep"


@pytest.mark.parametrize("reserved", [P.EXECUTIONS_DIR, P.CODE_DIR, P.SOURCES_DIR])
def test_a_symlinked_reserved_directory_refuses_the_purge(store, tmp_path, reserved):
    [a] = _save_n(store, "s1:call_a", b"a = 1\n", source=_src("x"))
    directory = store.root / reserved
    outside = tmp_path / "outside"
    shutil.move(str(directory), str(outside))
    directory.symlink_to(outside, target_is_directory=True)
    before = sorted(p.name for p in outside.iterdir())
    record, tombs = store.fence_purge()
    assert record.complete and tombs.inventory_unreadable
    assert store.read(a.ref)
    assert sorted(p.name for p in outside.iterdir()) == before


@pytest.mark.parametrize(
    "variant",
    [
        "exact",
        "unbound_name",
        "extra_key",
        "non_canonical",
        "bad_ref",
        "bad_digest",
        "bad_key",
        "huge_index",
    ],
)
def test_a_claim_not_exactly_as_the_store_writes_it_refuses_the_purge(store, variant):
    a = _save(store, "s1:call_a")
    claim = {
        "attempt_key": "s1:call_z:0",
        "attempt_digest": hashlib.sha256(b"control").hexdigest(),  # the emitted format
        "ref": a.ref,
        "execution_id": store.read_record(a.ref)["execution"]["execution_id"],
    }
    name = store_module._key_hash(claim["attempt_key"])
    if variant == "unbound_name":
        name = "1" * 64
    elif variant == "extra_key":
        claim["extra"] = "x"
    elif variant == "bad_ref":
        claim["ref"] = "not-a-ref"
    elif variant == "bad_digest":
        claim["attempt_digest"] = "d"
    elif variant == "huge_index":  # past int()'s digit limit: checked textually
        claim["attempt_key"] = "s1:call_z:" + "9" * 5000
        name = store_module._key_hash(claim["attempt_key"])
    elif variant == "bad_key":  # not <base>:<index>
        claim["attempt_key"] = "s1:call_z:01"
        name = store_module._key_hash(claim["attempt_key"])
    data = store_module._json_bytes(claim)
    if variant == "non_canonical":
        data = json.dumps(claim).encode("utf-8")
    (store.root / P.ATTEMPTS_DIR / f"{name}.json").write_bytes(data)
    record, tombs = store.fence_purge()
    assert record.complete
    if variant in (
        "exact",
        "huge_index",
    ):  # as written (an unbounded int index): accepted,
        # and a huge index never raises past int()'s digit limit
        assert tombs.complete and not tombs.inventory_unreadable
        return
    assert tombs.inventory_unreadable and not tombs.complete
    assert store.read(a.ref)


def test_write_debris_and_legacy_flat_files_do_not_block_the_purge(store):
    _save(store, "s1:call_a")
    (store.root / P.ATTEMPTS_DIR / ".tmp-abc123").write_bytes(b"partial")
    (store.root / "legacy.html").write_bytes(b"<html></html>")
    (
        store.root / new_id(PREFIX_FIGURE)
    ).mkdir()  # an empty figure dir (interrupted commit)
    entries, legacy = store._purge_inventory()
    assert legacy == 1
    assert all(p.parent == store.root / P.ATTEMPTS_DIR for p, _ in entries)
    record, tombs = store.fence_purge()
    assert record.complete and tombs.complete


# ----------------------------------- a destructive step needs the actual, confirmed fence


def test_removal_refuses_a_pending_tombstone_until_it_is_confirmed(store, monkeypatch):
    a = _save(store, "s1:call_a")
    attempts = store.root / P.ATTEMPTS_DIR
    real_fsync = store_module._fsync_dir

    def fsync(path):
        if Path(path) == attempts:
            raise DurabilityError("simulated", details={"reason": "dir_fsync_failed"})
        return real_fsync(path)

    monkeypatch.setattr(store_module, "_fsync_dir", fsync)
    assert store.fence_deletion([a.ref]).unconfirmed == 1
    with pytest.raises(DurabilityError):
        store.remove_fenced([a.ref])  # still unconfirmed: never authority
    monkeypatch.undo()
    rev = store._rev_dir(FigureRef.parse(a.ref))
    assert rev.is_dir()
    report = store.remove_fenced([a.ref])  # confirmed again now
    assert (report.removed, report.failed) == (1, 0) and not rev.exists()


@pytest.mark.parametrize("variant", ["differs", "symlink"])
def test_removal_refuses_a_corrupt_or_symlinked_tombstone(store, tmp_path, variant):
    a = _save(store, "s1:call_a")
    assert store.fence_deletion([a.ref]).complete
    path, data = store._tombstone(store_module._FENCE_REF, "ref", a.ref)
    path.unlink()
    if variant == "differs":
        path.write_bytes(b'{"schema": "something else"}\n')
        expected = IntegrityError
    else:
        real = tmp_path / "tombstone.json"
        real.write_bytes(data)
        path.symlink_to(real)
        expected = StorageError
    with pytest.raises(expected):
        store.remove_fenced([a.ref])
    assert store._rev_dir(FigureRef.parse(a.ref)).is_dir()  # nothing removed


def test_a_purge_removes_nothing_when_its_fence_is_pending(store, monkeypatch):
    a = _save(store, "s1:call_a")
    attempts = store.root / P.ATTEMPTS_DIR
    real_fsync = store_module._fsync_dir

    def fsync(path):
        if Path(path) == attempts:
            raise DurabilityError("simulated", details={"reason": "dir_fsync_failed"})
        return real_fsync(path)

    monkeypatch.setattr(store_module, "_fsync_dir", fsync)
    report = store.purge()
    monkeypatch.undo()
    assert not report.record.complete and report.tombstones is None
    assert report.removal is None and not report.complete
    assert store._rev_dir(FigureRef.parse(a.ref)).is_dir()
    assert (store.root / P.EXECUTIONS_DIR).is_dir()


def test_a_purge_removes_nothing_unless_its_own_fence_re_verifies(store, monkeypatch):
    a = _save(store, "s1:call_a")
    real_inventory = store._purge_inventory
    calls = {"n": 0}

    def inventory():  # the re-verification finds an object this fence did not cover
        entries, legacy = real_inventory()
        calls["n"] += 1
        if calls["n"] == 2:
            born = FigureRef(new_id(PREFIX_FIGURE), new_id(PREFIX_REVISION))
            entries = [
                *entries,
                store._tombstone(store_module._FENCE_REF, "ref", str(born)),
            ]
        return entries, legacy

    monkeypatch.setattr(store, "_purge_inventory", inventory)
    report = store.purge()
    assert report.tombstones.complete and not report.fence_verified
    assert report.removal is None and not report.complete
    assert store._rev_dir(FigureRef.parse(a.ref)).is_dir()  # hidden, never removed


def test_removal_for_a_purge_exists_only_inside_one_purge_call(store):
    assert not hasattr(LocalFigureStore, "remove_purged")  # no token, no separate step
    a = _save(store, "s1:call_a")
    ref = FigureRef.parse(a.ref)
    for destructive in (
        lambda: store._remove_revision_dir(ref),
        lambda: store._cleanup_released([]),
        store._remove_purged_locked,
        lambda: store._write_fences([]),
    ):
        with pytest.raises(StorageError):
            destructive()  # outside a writer-lock hold
    assert store.read(a.ref)


# ------------------------- the revision record is the retry evidence for its payload


@pytest.mark.parametrize("failing", ["code", "activity", "record"])
def test_a_failed_payload_cleanup_keeps_the_revision_as_retry_evidence(
    store, monkeypatch, failing
):
    [a] = _save_n(store, "s1:call_a", b"a = 1\n", source=_src("only-a"))
    eid = _execution_id(store, a.ref)
    executions = store.root / P.EXECUTIONS_DIR
    code_id = store.read_execution(eid)["code_id"]
    target = {
        "code": store.root / P.CODE_DIR / f"{code_id}.py",
        "activity": executions / f"{eid}.activity.json",
        "record": executions / f"{eid}.json",
    }[failing]
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        if Path(path) == target:
            raise OSError("simulated")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    first = _delete(store, [a.ref])
    monkeypatch.undo()
    assert first.cleanup.failed == 1 and first.cleanup_pending
    assert (first.removed, first.failed, first.kept) == (0, 1, (a.ref,))
    rev = store._rev_dir(FigureRef.parse(a.ref))
    assert rev.is_dir() and (executions / f"{eid}.json").is_file()  # evidence kept
    with pytest.raises(FigureNotFoundError):
        store.read(a.ref)  # invisible either way

    second = store.remove_fenced(list(first.kept))  # the honest retry completes
    assert (second.removed, second.failed, second.kept) == (1, 0, ())
    assert not second.cleanup_pending and second.cleanup.executions == 1
    assert not rev.exists() and not target.exists()
    assert not (executions / f"{eid}.json").exists()


def test_an_unreadable_revision_record_is_kept_and_reported_pending(store):
    [a] = _save_n(store, "s1:call_a", b"a = 1\n")
    eid = _execution_id(store, a.ref)
    assert store.fence_deletion([a.ref]).complete
    rev = store._rev_dir(FigureRef.parse(a.ref))
    (rev / P.REVISION_FILE).write_bytes(b"{corrupt")
    report = store.remove_fenced([a.ref])
    assert (report.removed, report.failed, report.kept) == (0, 1, (a.ref,))
    assert report.cleanup_pending  # never reported clean
    assert rev.is_dir()  # its payload is unknown: nothing of it is removed
    assert (store.root / P.EXECUTIONS_DIR / f"{eid}.json").is_file()


def test_a_cleanup_failure_never_frees_code_an_unreleased_execution_still_uses(store):
    r0, r1 = _save_n(store, "s1:call_m", b"same = 1\n", n=2)
    [b] = _save_n(store, "s1:call_b", b"same = 1\n")
    code_id = store.read_execution(_execution_id(store, b.ref))["code_id"]
    # m's execution stays (r1 live); b's goes: the shared code must stay for m
    report = _delete(store, [r0.ref, b.ref])
    assert report.cleanup.executions == 1 and report.cleanup.code == 0
    assert (store.root / P.CODE_DIR / f"{code_id}.py").is_file()
    assert store.read(r1.ref)


# ----------------------------- cleanup lstats parents and final entries, never follows


@pytest.mark.parametrize("reserved", [P.EXECUTIONS_DIR, P.CODE_DIR, P.SOURCES_DIR])
def test_cleanup_never_follows_a_symlinked_payload_directory(store, tmp_path, reserved):
    [a] = _save_n(store, "s1:call_a", b"a = 1\n", source=_src("x"))
    assert store.fence_deletion([a.ref]).complete
    directory = store.root / reserved
    outside = tmp_path / "outside"
    shutil.move(str(directory), str(outside))
    directory.symlink_to(outside, target_is_directory=True)
    before = {p.name: p.read_bytes() for p in outside.iterdir()}
    report = store.remove_fenced([a.ref])
    assert report.cleanup.failed >= 1 and report.cleanup_pending
    assert report.kept == (a.ref,)  # the record stays as the retry evidence
    assert {p.name: p.read_bytes() for p in outside.iterdir()} == before
    assert directory.is_symlink()


def test_cleanup_never_reads_or_unlinks_through_a_symlinked_record(store, tmp_path):
    [a] = _save_n(store, "s1:call_a", b"a = 1\n")
    eid = _execution_id(store, a.ref)
    code_id = store.read_execution(eid)["code_id"]
    assert store.fence_deletion([a.ref]).complete
    record = store.root / P.EXECUTIONS_DIR / f"{eid}.json"
    outside = tmp_path / "record.json"
    outside.write_bytes(record.read_bytes())
    record.unlink()
    record.symlink_to(outside)
    report = store.remove_fenced([a.ref])
    assert report.cleanup.failed == 1 and report.kept == (a.ref,)
    assert record.is_symlink() and outside.is_file()
    assert (store.root / P.CODE_DIR / f"{code_id}.py").is_file()  # nothing freed


# --------------------------------------------------- GC authority is never mere presence


def test_a_fenced_revision_directory_without_its_record_is_kept(store):
    [a] = _save_n(store, "s1:call_a", b"a = 1\n", source=_src("x"))
    eid = _execution_id(store, a.ref)
    record = store.read_record(a.ref)
    code_id = store.read_execution(eid)["code_id"]
    assert store.fence_deletion([a.ref]).complete
    rev = store._rev_dir(FigureRef.parse(a.ref))
    (rev / P.REVISION_FILE).unlink()  # the retry evidence is gone; the payload is not
    report = store.remove_fenced([a.ref])
    assert (report.removed, report.kept) == (0, (a.ref,)) and report.cleanup_pending
    assert (rev / P.FIGURE_FILE).is_file()  # never inferred as cleaned
    assert (store.root / P.EXECUTIONS_DIR / f"{eid}.json").is_file()
    assert (store.root / P.CODE_DIR / f"{code_id}.py").is_file()
    source_id = record["source"]["source_id"]
    assert (store.root / P.SOURCES_DIR / f"{source_id}.json").is_file()


@pytest.mark.parametrize("variant", ["figure_dir", "revision_dir", "marker"])
def test_cleanup_never_follows_a_symlinked_survivor(store, tmp_path, variant):
    [a] = _save_n(store, "s1:call_a", b"a = 1\n", source=_src("shared"))
    [b] = _save_n(store, "s1:call_b", b"b = 2\n", source=_src("shared"))
    source_id = store.read_record(b.ref)["source"]["source_id"]
    eid_a = _execution_id(store, a.ref)
    ref_b = FigureRef.parse(b.ref)
    target = {
        "figure_dir": store.root / ref_b.figure_id,
        "revision_dir": store._rev_dir(ref_b),
        "marker": store._rev_dir(ref_b) / P.REVISION_FILE,
    }[variant]
    outside = tmp_path / "outside"
    shutil.move(str(target), str(outside))
    target.symlink_to(outside, target_is_directory=variant != "marker")
    files = [outside] if outside.is_file() else sorted(outside.rglob("*"))
    before = {str(p): p.read_bytes() for p in files if p.is_file()}
    assert store.fence_deletion([a.ref]).complete
    report = store.remove_fenced([a.ref])
    assert report.cleanup.failed >= 1 and report.kept == (a.ref,)
    assert {str(p): p.read_bytes() for p in files if p.is_file()} == before
    assert target.is_symlink()
    assert (store.root / P.SOURCES_DIR / f"{source_id}.json").is_file()
    assert (store.root / P.EXECUTIONS_DIR / f"{eid_a}.json").is_file()


@pytest.mark.parametrize("variant", ["differs", "symlink"])
def test_a_present_but_unverified_survivor_tombstone_is_not_gc_authority(
    store, tmp_path, variant
):
    [a] = _save_n(store, "s1:call_a", b"a = 1\n", source=_src("shared"))
    [b] = _save_n(store, "s1:call_b", b"b = 2\n", source=_src("shared"))
    source_id = store.read_record(b.ref)["source"]["source_id"]
    eid_b = _execution_id(store, b.ref)
    path, data = store._tombstone(store_module._FENCE_REF, "ref", b.ref)
    if variant == "differs":
        path.write_bytes(b'{"schema": "something else"}\n')
    else:
        real = tmp_path / "tombstone.json"
        real.write_bytes(data)
        path.symlink_to(real)
    assert store.fence_deletion([a.ref]).complete
    report = store.remove_fenced([a.ref])
    assert report.cleanup.failed >= 1 and report.kept == (a.ref,)
    assert (store.root / P.SOURCES_DIR / f"{source_id}.json").is_file()
    assert (store.root / P.EXECUTIONS_DIR / f"{eid_b}.json").is_file()


def _rewrite_shape(store, record, variant):
    """Break the plan shape but keep record, exec claim and plan digest mutually consistent."""
    data = json.loads(record.read_bytes())
    if variant == "count_too_small":
        data["candidate_count"] = 0
    else:
        data["planned_candidates"] = data["planned_candidates"] * 2
    plan = {
        k: v for k, v in data.items() if k not in store_module._EXECUTION_STORE_FIELDS
    }
    digest = store_module.execution_plan_digest(plan)
    data["plan_digest"] = digest
    record.unlink()
    record.write_bytes(store_module._json_bytes(data))
    _rewrite_json(
        _claim_path(store, "exec:" + data["attempt_base"]), plan_digest=digest
    )


def _claim_path(store, key):
    return store.root / P.ATTEMPTS_DIR / f"{store_module._key_hash(key)}.json"


def _rewrite_json(path, **changes):
    data = json.loads(path.read_bytes())
    data.update(changes)
    path.unlink()
    path.write_bytes(store_module._json_bytes(data))


@pytest.mark.parametrize(
    "variant", ["no_source_key", "no_execution_key", "null_execution", "other_schema"]
)
def test_a_survivor_record_not_as_written_keeps_the_shared_source(store, variant):
    [a] = _save_n(store, "s1:call_a", b"a = 1\n", source=_src("shared"))
    [b] = _save_n(store, "s1:call_b", b"b = 2\n", source=_src("shared"))
    source_id = store.read_record(b.ref)["source"]["source_id"]
    marker = store._rev_dir(FigureRef.parse(b.ref)) / P.REVISION_FILE
    data = json.loads(marker.read_bytes())
    if variant == "no_source_key":
        del data["source"]
    elif variant == "no_execution_key":
        del data["execution"]
    elif variant == "null_execution":
        data["execution"] = None
    else:
        data["schema"] = "figure-revision/0"
    marker.unlink()
    marker.write_bytes(store_module._json_bytes(data))
    report = _delete(store, [a.ref])
    assert report.cleanup.failed >= 1 and report.kept == (a.ref,)
    assert (store.root / P.SOURCES_DIR / f"{source_id}.json").is_file()


def test_a_survivor_fence_that_cannot_be_reconfirmed_is_not_gc_authority(
    store, monkeypatch
):
    """B is fenced canonically, but its tombstone's durability cannot be re-confirmed during
    the survivor pass (only B's): B stays live, so the shared source is kept."""
    [a] = _save_n(store, "s1:call_a", b"a = 1\n", source=_src("shared"))
    [b] = _save_n(store, "s1:call_b", b"b = 2\n", source=_src("shared"))
    source_id = store.read_record(b.ref)["source"]["source_id"]
    assert store.fence_deletion([b.ref]).complete  # exact bytes, durable
    assert store.fence_deletion([a.ref]).complete
    b_path, _ = store._tombstone(store_module._FENCE_REF, "ref", b.ref)
    real_fsync, real_verify = store_module._fsync_dir, store._verify_fence
    failing = {"on": False}

    def fsync(path):
        if failing["on"]:
            raise DurabilityError("simulated", details={"reason": "dir_fsync_failed"})
        return real_fsync(path)

    def verify(path, data):
        failing["on"] = path == b_path
        try:
            return real_verify(path, data)
        finally:
            failing["on"] = False

    monkeypatch.setattr(store_module, "_fsync_dir", fsync)
    monkeypatch.setattr(store, "_verify_fence", verify)
    report = store.remove_fenced([a.ref])
    monkeypatch.undo()
    assert report.cleanup.failed >= 1 and report.kept == (a.ref,)
    assert (store.root / P.SOURCES_DIR / f"{source_id}.json").is_file()


@pytest.mark.parametrize("variant", ["empty_object", "no_code_field", "other_id"])
def test_a_malformed_staying_execution_keeps_shared_code(store, variant):
    [a] = _save_n(store, "s1:call_a", b"same = 1\n")
    [b] = _save_n(store, "s1:call_b", b"same = 1\n")
    eid_b = _execution_id(store, b.ref)
    code_id = store.read_execution(eid_b)["code_id"]
    record_b = store.root / P.EXECUTIONS_DIR / f"{eid_b}.json"
    data = json.loads(record_b.read_bytes())
    if variant == "empty_object":
        data = {}
    elif variant == "no_code_field":
        del data["code_id"]
    else:
        data["execution_id"] = new_id(store_module.PREFIX_EXECUTION)
    record_b.unlink()
    record_b.write_bytes(store_module._json_bytes(data))
    report = _delete(store, [a.ref])
    assert report.cleanup.failed >= 1 and report.kept == (a.ref,)
    assert (store.root / P.CODE_DIR / f"{code_id}.py").is_file()  # unknown: kept


@pytest.mark.parametrize("field", ["execution_id", "attempt_digest"])
def test_a_claim_bound_elsewhere_is_never_release_authority(store, field):
    [a] = _save_n(store, "s1:call_a", b"a = 1\n")
    eid = _execution_id(store, a.ref)
    code_id = store.read_execution(eid)["code_id"]
    other = (
        new_id(store_module.PREFIX_EXECUTION)
        if field == "execution_id"
        else hashlib.sha256(b"another plan").hexdigest()
    )
    _rewrite_json(_claim_path(store, "s1:call_a:0"), **{field: other})
    report = _delete(store, [a.ref])
    assert report.cleanup.failed >= 1 and report.kept == (a.ref,)
    assert (store.root / P.EXECUTIONS_DIR / f"{eid}.json").is_file()
    assert (store.root / P.CODE_DIR / f"{code_id}.py").is_file()


@pytest.mark.parametrize(
    "variant",
    [
        "planned_digest",
        "exec_claim",
        "count_too_small",
        "duplicate_index",
        "other_schema",
        "no_schema",
    ],
)
def test_a_corrupted_plan_or_binding_is_never_release_authority(store, variant):
    [a] = _save_n(store, "s1:call_a", b"a = 1\n")
    eid = _execution_id(store, a.ref)
    record = store.root / P.EXECUTIONS_DIR / f"{eid}.json"
    other = hashlib.sha256(b"another plan").hexdigest()
    if (
        variant == "planned_digest"
    ):  # the record no longer recomputes to its plan digest
        planned = json.loads(record.read_bytes())["planned_candidates"]
        _rewrite_json(
            record, planned_candidates=[{**planned[0], "attempt_digest": other}]
        )
    elif variant == "exec_claim":
        _rewrite_json(_claim_path(store, "exec:s1:call_a"), plan_digest=other)
    elif variant in ("other_schema", "no_schema"):  # the plan digest still matches
        data = json.loads(record.read_bytes())
        if variant == "no_schema":
            del data["schema"]
        else:
            data["schema"] = "lobster.figure_execution/0"
        record.unlink()
        record.write_bytes(store_module._json_bytes(data))
    else:  # writer shape violated (a consistent digest alone is not the writer's shape)
        _rewrite_shape(store, record, variant)
    report = _delete(store, [a.ref])
    assert report.cleanup.failed >= 1 and report.kept == (a.ref,)
    assert record.is_file()


@pytest.mark.parametrize("variant", ["absent", "differs"])
def test_removal_checks_every_tombstone_before_removing_any(store, variant):
    a = _save(store, "s1:call_a")
    b = _save(store, "s1:call_b")
    refs = [a.ref] if variant == "absent" else [a.ref, b.ref]
    assert store.fence_deletion(refs).complete
    if variant == "differs":  # good first, bad later
        path, _ = store._tombstone(store_module._FENCE_REF, "ref", b.ref)
        path.unlink()
        path.write_bytes(b'{"schema": "something else"}\n')
    with pytest.raises((StorageError, IntegrityError)):
        store.remove_fenced([a.ref, b.ref])
    assert store._rev_dir(FigureRef.parse(a.ref)).is_dir()  # nothing removed
    assert (store._rev_dir(FigureRef.parse(a.ref)) / P.REVISION_FILE).is_file()


def _after_fencing(store, monkeypatch, action):
    """Run ``action`` once, right after purge's second _write_fences (all fences done)."""
    real = store._write_fences
    calls = {"n": 0}

    def write_fences(entries):
        report = real(entries)
        calls["n"] += 1
        if calls["n"] == 2:
            action()
        return report

    monkeypatch.setattr(store, "_write_fences", write_fences)


def test_a_purge_removes_nothing_when_re_verification_cannot_confirm_durability(
    store, monkeypatch
):
    a = _save(store, "s1:call_a")
    real_fsync = store_module._fsync_dir
    failing = {"on": False}

    def fsync(path):
        if failing["on"]:
            raise DurabilityError("simulated", details={"reason": "dir_fsync_failed"})
        return real_fsync(path)

    monkeypatch.setattr(store_module, "_fsync_dir", fsync)
    _after_fencing(store, monkeypatch, lambda: failing.update(on=True))
    report = store.purge()
    monkeypatch.undo()
    assert report.record.complete and report.tombstones.complete
    assert not report.fence_verified and report.removal is None
    assert store._rev_dir(FigureRef.parse(a.ref)).is_dir()


@pytest.mark.parametrize("variant", ["differs", "vanished"])
def test_a_purge_removes_nothing_when_its_own_record_changes_before_re_verification(
    store, monkeypatch, variant
):
    a = _save(store, "s1:call_a")
    attempts = store.root / P.ATTEMPTS_DIR

    def change():
        [record] = sorted(attempts.glob(f"{store_module._FENCE_PURGE}*.json"))
        record.unlink()
        if variant == "differs":
            record.write_bytes(b'{"schema": "something else"}\n')

    _after_fencing(store, monkeypatch, change)
    report = store.purge()
    assert report.tombstones.complete and not report.fence_verified
    assert report.removal is None and not report.complete
    assert store._rev_dir(FigureRef.parse(a.ref)).is_dir()


# ------------------------------------------------ reasons/3 names a deleted ref


def test_a_deleted_ref_is_distinguishable_from_one_that_never_existed(store):
    a = _save(store, "s1:call_a")
    assert store.fence_deletion([a.ref]).complete
    never = str(FigureRef(new_id(PREFIX_FIGURE), new_id(PREFIX_REVISION)))
    for read in (store.read, store.read_record, store.list_renditions):
        with pytest.raises(FigureDeletedError) as deleted:
            read(a.ref)
        assert safe_reason(deleted.value) == "figure_deleted"
        assert isinstance(deleted.value, FigureNotFoundError)  # old handlers still work
    assert store.list_renditions(never) == []  # unchanged: no revision check there
    for read in (store.read, store.read_record):
        with pytest.raises(FigureNotFoundError) as missing:
            read(never)
        assert not isinstance(missing.value, FigureDeletedError)
        assert safe_reason(missing.value) == "figure_not_found:revision_not_found"


def test_an_execution_tombstone_and_a_deleted_input_stay_generation_fenced(store):
    assert store.fence_deletion([], instance_ids=["inst-1"]).complete
    late = _save(store, "s1:call_late", "inst-1")
    assert (late.status, late.reason) == (STATUS_REFUSED, "generation_fenced")
    a = _save(store, "s1:call_a", "inst-2")
    report = store.purge()  # EXEC claims tombstoned, incl. one with no revision claim
    assert report.complete
    again = _save(store, "s1:call_a", "inst-2")
    assert (again.status, again.reason) == (
        STATUS_REFUSED,
        "execution_record:generation_fenced",
    )
    assert a.status == "available"


# ---------------------------------------------------- lexical AND resolved classification


def _ws(tmp_path):
    ws = tmp_path / "ws"
    (ws / P.PLOTS_DIR / P.SOURCES_DIR).mkdir(parents=True)
    (ws / "data").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file.txt").write_text("outside")
    return ws, outside


def _refused(ws, target):
    try:
        P.assert_not_managed(ws, target)
    except ManagedPathError:
        return True
    return False


def test_the_eight_bounded_cases(tmp_path):
    ws, outside = _ws(tmp_path)
    plots = ws / P.PLOTS_DIR
    # 1. a legacy-location final symlink to an outside file
    (plots / "legacy.html").symlink_to(outside / "file.txt")
    assert _refused(ws, plots / "legacy.html")
    # 2. a reserved-location symlink to outside
    (plots / P.SOURCES_DIR / "x.json").symlink_to(outside / "file.txt")
    assert _refused(ws, plots / P.SOURCES_DIR / "x.json")
    # 3. an outside alias into plots/<f>/<r>/figure.json
    (plots / "f_1" / "r_1").mkdir(parents=True)
    (plots / "f_1" / "r_1" / P.FIGURE_FILE).write_text("{}")
    (outside / "alias.json").symlink_to(plots / "f_1" / "r_1" / P.FIGURE_FILE)
    assert _refused(ws, outside / "alias.json")
    # 4. a workspace parent symlink to plots/
    (ws / "alias").symlink_to(plots, target_is_directory=True)
    assert _refused(ws, ws / "alias" / "x.html")
    assert _refused(ws, ws / "alias" / "f_1" / "r_1" / P.FIGURE_FILE)
    # 6. a dangling final symlink under plots/
    (plots / "dangling.html").symlink_to(tmp_path / "missing.html")
    assert _refused(ws, plots / "dangling.html")
    # 7. a data symlink to outside: unchanged (no new outside policy)
    (ws / "data" / "file.csv").symlink_to(outside / "file.txt")
    assert not _refused(ws, ws / "data" / "file.csv")
    # 8. control: a plain workspace path
    assert not _refused(ws, ws / "data" / "x")


def test_a_symlinked_plots_root_refuses_every_plots_target(tmp_path):
    ws, outside = _ws(tmp_path)  # 5. plots/ itself a symlink
    shutil.rmtree(ws / P.PLOTS_DIR)
    (outside / "plots").mkdir()
    (ws / P.PLOTS_DIR).symlink_to(outside / "plots", target_is_directory=True)
    for rel in ("legacy.html", "f_1/r_1/figure.json", ""):
        assert _refused(ws, ws / P.PLOTS_DIR / rel)
    inner = ws / "inner"  # plots/ -> a workspace directory: still refused
    inner.mkdir()
    (ws / P.PLOTS_DIR).unlink()
    (ws / P.PLOTS_DIR).symlink_to(inner, target_is_directory=True)
    assert _refused(ws, ws / P.PLOTS_DIR / "legacy.html")


def test_root_ancestor_reserved_and_inside_refuse_and_legacy_files_stay_allowed(
    tmp_path,
):
    ws, _ = _ws(tmp_path)
    plots = ws / P.PLOTS_DIR
    (plots / "old.html").write_text("<html></html>")
    (plots / "sub").mkdir()
    for refused in (
        ws,
        ws / "..",
        tmp_path,
        plots,
        plots / P.ATTEMPTS_DIR,
        plots / P.SOURCES_DIR / "new.json",
        plots / "sub",
        plots / "f_9" / "r_9" / P.FIGURE_FILE,
        ws / "data" / ".." / P.PLOTS_DIR / "f_9" / "x",
    ):
        assert _refused(ws, refused), refused
    assert not _refused(ws, plots / "old.html")  # a regular legacy file
    assert not _refused(ws, plots / "new.html")  # a new legacy file
    assert not _refused(ws, tmp_path / "elsewhere.txt")  # outside: unchanged


def test_classify_reports_both_classes(tmp_path):
    ws, outside = _ws(tmp_path)
    (ws / "alias").symlink_to(ws / P.PLOTS_DIR, target_is_directory=True)
    assert P.classify(ws, ws / "alias" / "x", follow_final=True) == (
        P.WORKSPACE,
        P.PLOTS_CHILD,
    )
    link = ws / P.PLOTS_DIR / "l.html"
    link.symlink_to(outside / "file.txt")
    assert P.classify(ws, link, follow_final=True) == (P.PLOTS_CHILD, P.OUTSIDE)
    assert P.classify(ws, link, follow_final=False) == (P.PLOTS_CHILD, P.PLOTS_CHILD)
    assert P.classify(ws, ws / "..", follow_final=False).lexical == P.ANCESTOR


def test_the_resolved_half_resolves_a_symlink_before_its_dot_dot(tmp_path):
    ws, _ = _ws(tmp_path)
    rev = ws / P.PLOTS_DIR / "f_1" / "r_1"
    rev.mkdir(parents=True)
    (ws / "alias").symlink_to(rev, target_is_directory=True)
    target = os.path.join(
        str(ws), "alias", "..", P.FIGURE_FILE
    )  # = plots/f_1/figure.json
    assert P.classify(ws, target, follow_final=True) == (P.WORKSPACE, P.INSIDE)
    assert P.classify(ws, target, follow_final=False) == (P.WORKSPACE, P.INSIDE)
    assert _refused(ws, target)
    plain = os.path.join(str(ws), "data", "..", "data", "x")  # an ordinary .. (control)
    assert P.classify(ws, plain, follow_final=True) == (P.WORKSPACE, P.WORKSPACE)
    assert not _refused(ws, plain)


# ------------------------------------- attempt fence (the born set of one attempt)


def test_a_session_purge_during_an_attempt_fences_its_commit(store):
    fence = store.attempt_fence()
    assert store.fence_purge()[0].complete  # a purge record appears after the start
    inputs = (InputEdge("prot_de", "inst-1", "store_event", True),)
    [res] = TrustedSaver(store).commit_attempt(
        _info(base="s1:call_f"),
        [_cand(base="s1:call_f", inputs=inputs)],
        succeeded=True,
        fence=fence,
    )
    assert (res.status, res.reason) == (
        STATUS_REFUSED,
        "execution_record:generation_fenced",
    )
    assert store.list().items == []
    # control: an attempt that BEGAN after the purge commits normally
    [ok] = TrustedSaver(store).commit_attempt(
        _info(base="s1:call_g"),
        [_cand(base="s1:call_g", inputs=inputs)],
        succeeded=True,
        fence=store.attempt_fence(),
    )
    assert ok.status == "available"


def test_an_attempt_without_a_purge_commits_under_its_fence(store):
    fence = store.attempt_fence()
    assert fence.purges == frozenset()
    [res] = TrustedSaver(store).commit_attempt(
        _info(), [_cand()], succeeded=True, fence=fence
    )
    assert res.status == "available"


def test_any_purge_prefixed_name_added_during_the_attempt_fences_it(store):
    [_] = _save_n(store, "s1:call_a", b"a = 1\n")  # the store exists
    fence = store.attempt_fence()
    (store.root / P.ATTEMPTS_DIR / f"{store_module._FENCE_PURGE}x.json").write_bytes(
        b""
    )
    with pytest.raises(GenerationFencedError):
        store._check_attempt_fence(fence)


def test_an_unreadable_born_set_refuses_every_commit(store, monkeypatch):
    def boom():
        raise StorageError("simulated")

    monkeypatch.setattr(store, "_purge_names", boom)
    fence = store.attempt_fence()  # never raises
    monkeypatch.undo()
    assert fence.purges is None
    [res] = TrustedSaver(store).commit_attempt(
        _info(), [_cand()], succeeded=True, fence=fence
    )
    assert res.status != "available"
    assert res.reason == "execution_record:storage_failed"
    assert store.list().items == []


def test_a_fence_of_another_root_is_never_accepted(store, tmp_path):
    other = LocalFigureStore(tmp_path / "other", session_id="s1")
    [res] = TrustedSaver(store).commit_attempt(
        _info(), [_cand()], succeeded=True, fence=other.attempt_fence()
    )
    assert res.status != "available" and store.list().items == []


@pytest.mark.parametrize("method", ["commit_execution", "commit"])
def test_the_fence_is_checked_before_the_idempotency_lookup(store, monkeypatch, method):
    fence = store.attempt_fence()
    assert store.fence_purge()[0].complete
    reached = []
    monkeypatch.setattr(store, f"_{method}", lambda *a, **k: reached.append(1))
    args = (b"{}", None, None) if method == "commit_execution" else (_cand(),)
    kwargs = (
        {} if method == "commit_execution" else {"execution_id": "x", "digest": "d"}
    )
    with pytest.raises(GenerationFencedError):
        getattr(store, method)(*args, fence=fence, **kwargs)
    assert reached == []
    getattr(store, method)(*args, **kwargs)  # control: no fence, no born-set check
    assert reached == [1]


_REPO = Path(__file__).resolve().parents[4]


def _calls(rel, attr):
    import ast

    tree = ast.parse((_REPO / rel).read_text(encoding="utf-8"))
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", getattr(n.func, "id", None)) == attr
    ]


def test_the_custom_code_writer_site_passes_a_fence_captured_before_execution():
    """Source contract (shape only, not runtime proof): the one production commit site of
    custom code passes fence=, and the fence is captured before execute() runs."""
    rel = "lobster/tools/custom_code_tool.py"
    [commit] = _calls(rel, "commit_attempt")
    assert [k.value.id for k in commit.keywords if k.arg == "fence"] == [
        "attempt_fence"
    ]
    [submit] = _calls(rel, "_commit_submitted_figures")
    assert [k.value.id for k in submit.keywords if k.arg == "attempt_fence"] == [
        "attempt_fence"
    ]
    [capture] = _calls(rel, "_attempt_fence")
    [to_thread] = _calls(rel, "to_thread")
    assert capture.lineno < to_thread.lineno


# ------------------------------------------------------- PlotManager attempt contexts


class _Interrupted(BaseException):
    """A BaseException escaping commit_attempt mid-call (not KeyboardInterrupt)."""


def _pm_dm(tmp_path, instance_id="inst-A"):
    import anndata
    import numpy as np

    from lobster.core.runtime.data_manager import DataManagerV2

    dm = DataManagerV2(workspace_path=tmp_path)
    adata = anndata.AnnData(np.ones((2, 2)))
    adata.uns["lineage"] = {"instance_id": instance_id}
    dm.modalities["m"] = adata
    return dm, dm.plot_manager


def _bar(y=1):
    import plotly.graph_objects as go

    return go.Figure(go.Bar(y=[y]))


def _stage(pm, dm, ctx, title="t", y=1):
    held = pm.read_input(ctx, dm, "m")
    sid, _, _ = pm.add_plot(_bar(y), title=title, context=ctx, inputs=(held,))
    assert sid
    return sid


def test_a_context_commit_uses_the_held_edge_never_the_current_name(tmp_path):
    import anndata
    import numpy as np

    dm, pm = _pm_dm(tmp_path)
    ctx = pm.begin_attempt()
    sid = _stage(pm, dm, ctx)
    replacement = anndata.AnnData(np.zeros((2, 2)))
    replacement.uns["lineage"] = {"instance_id": "inst-B"}
    dm.modalities["m"] = replacement  # same name, another instance
    [res] = pm.commit_staged(
        {"id": "a"}, tool_call_id="c1", staged_ids=[sid], context=ctx
    )
    assert res.status == "available"
    [edge] = dm.figure_store.read_record(res.ref)["inputs"]
    assert edge["instance_id"] == "inst-A"
    assert pm.end_attempt(ctx) == {}


def test_a_purge_after_the_read_fences_the_context_commit(tmp_path):
    from lobster.core.figures.errors import FigureError

    dm, pm = _pm_dm(tmp_path)
    ctx = pm.begin_attempt()
    sid = _stage(pm, dm, ctx)
    assert dm.figure_store.fence_purge()[0].complete
    with pytest.raises(FigureError) as info:
        pm.commit_staged({"id": "a"}, tool_call_id="c1", staged_ids=[sid], context=ctx)
    [r] = info.value.details["results"]
    assert (r["status"], r["reason"]) == (
        "refused",
        "execution_record:generation_fenced",
    )
    assert dm.figure_store.list().items == [] and pm.latest_plots == []
    assert pm.end_attempt(ctx) == {}  # delivered through the error's results


def test_a_closed_context_stays_closed_and_tagged_sids_need_it(tmp_path):
    from lobster.core.figures.errors import AttemptContextError

    dm, pm = _pm_dm(tmp_path)
    ctx = pm.begin_attempt()
    sid = _stage(pm, dm, ctx)
    with pytest.raises(AttemptContextError):  # never the context-free path
        pm.commit_staged({"id": "a"}, tool_call_id="c1", staged_ids=[sid])
    pm.commit_staged({"id": "a"}, tool_call_id="c1", staged_ids=[sid], context=ctx)
    with pytest.raises(AttemptContextError):
        pm.read_input(ctx, dm, "m")
    with pytest.raises(AttemptContextError):
        pm.add_plot(_bar(), title="t", context=ctx)
    with pytest.raises(AttemptContextError):
        pm.commit_staged({"id": "a"}, tool_call_id="c1", staged_ids=[sid], context=ctx)
    assert pm.end_attempt(ctx) == {} and pm.end_attempt(ctx) == {}
    with pytest.raises(AttemptContextError):
        pm.add_plot(_bar(), title="t", context=ctx)


def test_a_context_is_bound_to_its_manager_fence_and_inputs(tmp_path):
    import dataclasses

    from lobster.core.figures.errors import AttemptContextError

    dm, pm = _pm_dm(tmp_path)
    other_dm, other_pm = _pm_dm(tmp_path / "other")
    ctx, foreign = pm.begin_attempt(), other_pm.begin_attempt()
    with pytest.raises(AttemptContextError):
        pm.read_input(foreign, dm, "m")
    with pytest.raises(AttemptContextError):  # a later fence is never swapped in
        pm.read_input(dataclasses.replace(ctx, fence=pm.begin_attempt().fence), dm, "m")
    with pytest.raises(AttemptContextError):  # another manager's data manager
        pm.read_input(ctx, other_dm, "m")
    held_elsewhere = other_pm.read_input(foreign, other_dm, "m")
    with pytest.raises(AttemptContextError):
        pm.add_plot(_bar(), title="t", context=ctx, inputs=(held_elsewhere,))
    with pytest.raises(AttemptContextError):
        pm.add_plot(_bar(), title="t", inputs=(held_elsewhere,))


def test_never_submitted_is_not_committed_and_a_mid_call_raise_is_pending(
    tmp_path, monkeypatch
):
    from lobster.core.figures import saver as saver_module

    dm, pm = _pm_dm(tmp_path)
    ctx = pm.begin_attempt()
    first, second = _stage(pm, dm, ctx, "a", 1), _stage(pm, dm, ctx, "b", 2)

    def interrupted(self, *a, **k):
        raise _Interrupted()

    monkeypatch.setattr(saver_module.TrustedSaver, "commit_attempt", interrupted)
    with pytest.raises(_Interrupted):
        pm.commit_staged(
            {"id": "a"}, tool_call_id="c1", staged_ids=[first], context=ctx
        )
    out = pm.end_attempt(ctx)
    assert (out[first].status, out[first].reason) == ("pending", "unexpected_error")
    assert (out[second].status, out[second].reason, out[second].candidate_index) == (
        "not_committed",
        "attempt_failed",
        -1,
    )


def test_a_pending_batch_is_retried_whole_under_its_original_fence(
    tmp_path, monkeypatch
):
    from lobster.core.figures.errors import AttemptContextError

    from .test_durability_followup import FsyncTrace, _marker_linked

    dm, pm = _pm_dm(tmp_path)
    ctx = pm.begin_attempt()
    sids = [_stage(pm, dm, ctx, "a", 1), _stage(pm, dm, ctx, "b", 2)]
    trace = FsyncTrace(
        monkeypatch, fail=_marker_linked
    )  # first marker fsync fails once
    first = pm.commit_staged(
        {"id": "a"}, tool_call_id="c1", staged_ids=sids, context=ctx
    )
    assert sorted(r.status for r in first) == ["available", "pending"]
    assert trace.failed == 1
    with pytest.raises(AttemptContextError):  # retry_only: no new staging
        pm.add_plot(_bar(), title="t", context=ctx)
    with pytest.raises(AttemptContextError):  # a subset would change the plan
        pm.commit_staged(
            {"id": "a"}, tool_call_id="c1", staged_ids=sids[1:], context=ctx
        )
    retry = pm.commit_staged(
        {"id": "a"}, tool_call_id="c1", staged_ids=sids, context=ctx
    )
    assert [r.status for r in retry] == ["available", "available"]
    available = [r.ref for r in first if r.status == "available"]
    assert available[0] in [r.ref for r in retry]  # re-ACKed, never a second revision
    assert len(dm.figure_store.list().items) == 2 and len(pm.latest_plots) == 2
    assert pm.end_attempt(ctx) == {}


def test_a_purge_before_a_same_context_retry_fences_it(tmp_path, monkeypatch):
    from .test_durability_followup import FsyncTrace, _marker_linked

    dm, pm = _pm_dm(tmp_path)
    ctx = pm.begin_attempt()
    sid = _stage(pm, dm, ctx)
    FsyncTrace(monkeypatch, fail=_marker_linked)
    [r] = pm.commit_staged(
        {"id": "a"}, tool_call_id="c1", staged_ids=[sid], context=ctx
    )
    assert r.status == "pending"
    monkeypatch.undo()
    assert dm.figure_store.fence_purge()[0].complete
    from lobster.core.figures.errors import FigureError

    with pytest.raises(FigureError) as info:
        pm.commit_staged({"id": "a"}, tool_call_id="c1", staged_ids=[sid], context=ctx)
    [again] = info.value.details["results"]
    assert again["reason"] == "execution_record:generation_fenced"


def test_known_outcomes_survive_a_legacy_publish_failure(tmp_path, monkeypatch):
    dm, pm = _pm_dm(tmp_path)
    ctx = pm.begin_attempt()
    sid = _stage(pm, dm, ctx)

    def broken(*a, **k):
        raise RuntimeError("publish failed")

    monkeypatch.setattr(pm, "_publish_entry", broken)
    with pytest.raises(RuntimeError):
        pm.commit_staged({"id": "a"}, tool_call_id="c1", staged_ids=[sid], context=ctx)
    out = pm.end_attempt(ctx)
    assert out[sid].status == "available"  # recorded before the publish
    assert [i["figure_id"] for i in dm.figure_store.list().items] == [
        FigureRef.parse(out[sid].ref).figure_id
    ]


def test_a_claimed_but_unlinked_attempt_is_completed_by_a_same_call_replay(
    tmp_path, monkeypatch
):
    """UNQUALIFIED case (one simulated interruption point, not a general proof): the claim
    is written, the marker link fails; a graph re-execution of the same tool call (NEW
    context, new SIDs, identical held input) completes the ORIGINAL claim: one revision.
    """
    dm, pm = _pm_dm(tmp_path)
    real = store_module._link_no_replace

    def no_marker(path, data):
        if Path(path).name == P.REVISION_FILE:
            raise OSError("simulated interruption before the marker link")
        return real(path, data)

    ctx = pm.begin_attempt()
    sid = _stage(pm, dm, ctx)
    monkeypatch.setattr(store_module, "_link_no_replace", no_marker)
    from lobster.core.figures.errors import FigureError

    with pytest.raises(FigureError):
        pm.commit_staged({"id": "a"}, tool_call_id="c1", staged_ids=[sid], context=ctx)
    pm.end_attempt(ctx)
    monkeypatch.undo()
    replay = pm.begin_attempt()
    [res] = pm.commit_staged(
        {"id": "a"},
        tool_call_id="c1",
        staged_ids=[_stage(pm, dm, replay)],
        context=replay,
    )
    assert res.status == "available"
    assert len(dm.figure_store.list().items) == 1
    pm.end_attempt(replay)


def test_both_plot_manager_commit_sites_pass_fence():
    """Source contract (shape only): the context path passes the recorded fence, and the
    context-free path passes fence=None explicitly (listed as NOT deletion-safe)."""
    calls = _calls("lobster/core/plot_manager.py", "commit_attempt")
    assert len(calls) == 2
    values = sorted(
        ast_unparse(k.value) for c in calls for k in c.keywords if k.arg == "fence"
    )
    assert values == ["None", "rec['fence']"]


def ast_unparse(node):
    import ast

    return ast.unparse(node)


# ------------------------------- verified-flag authority typing


@pytest.mark.parametrize(
    "flag, expected",
    [
        (True, True),
        (False, False),
        (1, False),
        ("true", False),
        ("false", False),
        (1.0, False),
        (None, False),
    ],
)
def test_only_a_stored_true_is_verified_dependent_authority(store, flag, expected):
    edge = InputEdge("d", "inst-X", "store_event", flag)
    [res] = TrustedSaver(store).commit_attempt(
        _info(base="s1:call_v"),
        [_cand(base="s1:call_v", inputs=(edge,))],
        succeeded=True,
    )
    assert res.status == "available"
    assert (
        store.read_record(res.ref)["inputs"][0]["verified"] == flag
    )  # stored as given
    assert store.dependents("inst-X") == [(res.ref, expected)]


def test_any_true_edge_still_wins_over_a_malformed_alias(store):
    edges = (
        InputEdge("d", "inst-X", "store_event", "true"),
        InputEdge("d2", "inst-X", "store_event", True),
    )
    [res] = TrustedSaver(store).commit_attempt(
        _info(), [_cand(inputs=edges)], succeeded=True
    )
    assert store.dependents("inst-X") == [(res.ref, True)]


@pytest.mark.parametrize("flag", [1, "true", "false", "1", 1.0, [True], None])
def test_a_non_bool_stored_flag_is_never_promoted_through_inheritance(flag):
    from lobster.core.figures.records import merge_inherited_inputs

    raw = {"name": "d", "instance_id": "i", "origin": "store_event", "verified": flag}
    assert InputEdge.from_dict(raw).verified is False
    [inherited] = merge_inherited_inputs((), [raw])
    assert inherited.verified is False
    own = InputEdge("d", "i", "declared", False)
    [kept] = merge_inherited_inputs(
        (own,), [raw]
    )  # never upgraded by a malformed parent
    assert kept.verified is False
    good = dict(raw, verified=True)  # control: an actual True parent edge still wins
    [won] = merge_inherited_inputs((own,), [good])
    assert won.verified is True


# ------------------- attempt-context busy policy: one commit serializes the whole attempt


def _blocked_commit(pm, ctx, sids, monkeypatch, where):
    """Run commit_staged(context=ctx) in a thread that blocks inside ``where``
    ('_attempt_candidates': before anything is submitted; '_publish_entry': after the store
    returned), and wait until it is blocked there. Returns (release, thread, box)."""
    entered, release, box = threading.Event(), threading.Event(), {}
    real = getattr(pm, where)

    def blocking(*a, **k):
        entered.set()
        assert release.wait(10)
        return real(*a, **k)

    monkeypatch.setattr(pm, where, blocking)

    def run():
        try:
            box["results"] = pm.commit_staged(
                {"id": "a"}, tool_call_id="c1", staged_ids=sids, context=ctx
            )
        except BaseException as exc:  # recorded for the assertion
            box["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    if not entered.wait(10):  # never leave a writer running into fixture cleanup
        release.set()
        thread.join(10)
        pytest.fail("the commit thread never reached the blocking point")
    return release, thread, box


@pytest.mark.parametrize("where", ["_attempt_candidates", "_publish_entry"])
def test_a_running_commit_refuses_a_concurrent_close_or_retry(
    tmp_path, monkeypatch, where
):
    from lobster.core.figures.errors import AttemptContextError

    dm, pm = _pm_dm(tmp_path)
    ctx = pm.begin_attempt()
    sid = _stage(pm, dm, ctx)
    release, thread, box = _blocked_commit(pm, ctx, [sid], monkeypatch, where)
    try:
        with pytest.raises(AttemptContextError):  # never not_committed while it runs
            pm.end_attempt(ctx)
        with pytest.raises(AttemptContextError):  # no overlapping retry
            pm.commit_staged(
                {"id": "a"}, tool_call_id="c1", staged_ids=[sid], context=ctx
            )
        with pytest.raises(AttemptContextError):
            pm.add_plot(_bar(), title="t", context=ctx)
        with pytest.raises(AttemptContextError):
            pm.read_input(ctx, dm, "m")
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and "error" not in box
    [res] = box["results"]
    assert res.status == "available" and len(pm.latest_plots) == 1
    assert pm.end_attempt(ctx) == {}  # delivered; closed after the commit returned
    with pytest.raises(AttemptContextError):
        pm.add_plot(_bar(), title="t", context=ctx)


def test_end_attempt_checks_the_fence_identity(tmp_path):
    import dataclasses

    from lobster.core.figures.errors import AttemptContextError

    dm, pm = _pm_dm(tmp_path)
    ctx = pm.begin_attempt()
    _stage(pm, dm, ctx)
    swapped = dataclasses.replace(ctx, fence=pm.begin_attempt().fence)
    with pytest.raises(AttemptContextError):
        pm.end_attempt(swapped)
    out = pm.end_attempt(ctx)  # the real context still closes and reports
    assert [r.status for r in out.values()] == ["not_committed"]


# ------------------------------------------------------------------------ deletion seam

from lobster.core.figures import deletion as D  # noqa: E402

_CONFIRM = {"confirmed": True}
_PAUSE = object()


class _Pause(Exception):
    """Stands for GraphInterrupt: the first execution pauses at the prompt."""


class _Saver:
    """A weakref-able stand-in for the executing checkpoint saver."""


def _graph(monkeypatch, saver=None, key=("t", "ns:1", "call_1")):
    saver = saver if saver is not None else _Saver()
    monkeypatch.setattr(D, "_resolve_binding", lambda tool_call_id: (saver, key))
    script, prompts = [], []

    def interrupt(payload):
        prompts.append(payload)
        step = script.pop(0)
        if step is _PAUSE:
            raise _Pause()
        return step

    monkeypatch.setattr(D, "_interrupt", interrupt)
    return saver, script, prompts


def _deletable(store):
    """One verified dependent of inst-X and one unverified ('may depend') one."""
    [dep] = _save_n(store, "s1:call_a", b"a = 1\n", instance_id="inst-X")
    edge = InputEdge("prot_de", "inst-X", "store_event", False)
    [maybe] = TrustedSaver(store).commit_attempt(
        _info(base="s1:call_m"),
        [_cand(base="s1:call_m", inputs=(edge,))],
        succeeded=True,
    )
    return dep.ref, maybe.ref


def _listed_set(store):
    return {f"{i['figure_id']}@{i['revision_id']}" for i in store.list().items}


def _planner(store):
    return lambda: D.plan_dataset_deletion(store, name="prot_de", instance_id="inst-X")


def _run(store, planner, **kw):
    return D.run_confirmed_deletion(store, planner, tool_call_id="call_1", **kw)


def test_a_confirmed_deletion_acts_once_and_answers_from_its_record(store, monkeypatch):
    _, script, prompts = _graph(monkeypatch)
    dep, maybe = _deletable(store)
    removed = []
    script.append(_PAUSE)
    with pytest.raises(_Pause):
        _run(store, _planner(store), remove_dataset=lambda: removed.append(1))
    question = prompts[0]["question"]
    assert dep in question and maybe in question and D.TEXT_NOT_TRACKED in question
    script.append(_CONFIRM)
    text = _run(store, _planner(store), remove_dataset=lambda: removed.append(1))
    assert dep in text and "The dataset was removed." in text and maybe in text
    assert _listed_set(store) == {maybe} and removed == [1]
    again = _run(store, _planner(store), remove_dataset=lambda: removed.append(1))
    assert (
        again == text and removed == [1] and len(prompts) == 2
    )  # no prompt, no effect


@pytest.mark.parametrize(
    "answer",
    [
        {"confirmed": 1},
        {"confirmed": "true"},
        {"confirmed": True, "also": 1},
        type("D2", (dict,), {})(confirmed=True),
        True,
        None,
        "yes",
    ],
)
def test_only_the_exact_answer_approves(store, monkeypatch, answer):
    _, script, _ = _graph(monkeypatch)
    dep, maybe = _deletable(store)
    script.extend([_PAUSE, answer])
    with pytest.raises(_Pause):
        _run(store, _planner(store))
    assert _run(store, _planner(store)) == D.TEXT_DECLINED
    assert _listed_set(store) == {dep, maybe}


def test_a_scope_change_supersedes_and_discards_the_answer(store, monkeypatch):
    _, script, prompts = _graph(monkeypatch)
    dep, maybe = _deletable(store)
    script.extend([_PAUSE, _CONFIRM])
    with pytest.raises(_Pause):
        _run(store, _planner(store))
    [new] = _save_n(store, "s1:call_n", b"n = 1\n", instance_id="inst-X")  # fp changes
    assert _run(store, _planner(store)) == D.TEXT_SUPERSEDED
    assert _run(store, _planner(store)) == D.TEXT_SUPERSEDED and len(prompts) == 2
    assert _listed_set(store) == {dep, maybe, new.ref}


def test_an_answer_without_its_bound_record_is_refused(store, monkeypatch):
    _, script, _ = _graph(monkeypatch)
    dep, maybe = _deletable(store)
    script.append(_CONFIRM)  # a resume value, but this map never bound the warning
    assert _run(store, _planner(store)) == D.TEXT_UNBOUND
    assert _listed_set(store) == {dep, maybe}


def test_no_binding_means_unavailable_and_nothing_deleted(store, monkeypatch):
    dep, maybe = _deletable(store)
    assert D._resolve_binding("call_1") is None  # real: outside any runnable context
    monkeypatch.setattr(D, "_resolve_binding", lambda tool_call_id: None)
    assert _run(store, _planner(store)) == D.TEXT_UNAVAILABLE
    assert _listed_set(store) == {dep, maybe}


def test_a_terminal_record_between_peek_and_get_answers_first(store, monkeypatch):
    _, script, prompts = _graph(monkeypatch)
    _deletable(store)
    script.append(_PAUSE)
    with pytest.raises(_Pause):
        _run(store, _planner(store))
    before, box = _planner(store)(), {}

    def racing():  # another resume of the SAME key wins after this call's peek
        script.append(_CONFIRM)
        box["winner"] = _run(store, _planner(store))
        return before  # this call's plan: the SAME fingerprint as the record

    assert _run(store, racing) == box["winner"]
    assert len(prompts) == 2  # the pause and the winner; this call never prompted


def test_a_terminal_record_wins_over_a_planning_refusal(store, monkeypatch):
    _, script, _ = _graph(monkeypatch)
    _deletable(store)
    script.append(_PAUSE)
    with pytest.raises(_Pause):
        _run(store, _planner(store))
    box = {}

    def racing():
        script.append(_CONFIRM)
        box["winner"] = _run(store, _planner(store))
        return D._plan(
            D.KIND_DATASET, (), ("prot_de", "inst-X"), refusal="inventory_unreadable"
        )

    assert _run(store, racing) == box["winner"]  # never a false "nothing deleted"


def test_a_lost_supersede_returns_the_winners_state(store, monkeypatch):
    saver, script, prompts = _graph(monkeypatch)
    _deletable(store)
    script.append(_PAUSE)
    with pytest.raises(_Pause):
        _run(store, _planner(store))
    real = D._get_or_create

    def then_a_winner(*a):
        rec, created, snap = real(*a)
        with D._CONSENT_LOCK:  # another resume consumes right after this observation
            rec.state, rec.outcome = D.CONSUMED, "the winner's recorded outcome"
        return rec, created, snap

    monkeypatch.setattr(D, "_get_or_create", then_a_winner)
    changed = lambda: D._plan(D.KIND_DATASET, ("other",), ("prot_de", "inst-X"))
    assert _run(store, changed) == "the winner's recorded outcome"
    assert len(prompts) == 1


def test_duplicate_same_root_resumes_act_once(store, monkeypatch):
    _, script, _ = _graph(monkeypatch)
    dep, maybe = _deletable(store)
    script.append(_PAUSE)
    with pytest.raises(_Pause):
        _run(store, _planner(store))
    barrier, removed, texts = threading.Barrier(2, timeout=10), [], []

    def both_answered(payload):
        barrier.wait()  # both resumes are past their prompt before either acts
        return _CONFIRM

    monkeypatch.setattr(D, "_interrupt", both_answered)

    def resume():
        texts.append(
            _run(store, _planner(store), remove_dataset=lambda: removed.append(1))
        )

    threads = [threading.Thread(target=resume) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not any(t.is_alive() for t in threads)
    assert removed == [1] and len(texts) == 2 and texts[0] == texts[1]
    assert _listed_set(store) == {maybe}


def test_an_answer_for_one_root_never_deletes_in_another(store, tmp_path, monkeypatch):
    _, script, prompts = _graph(monkeypatch)
    dep, maybe = _deletable(store)
    other = LocalFigureStore(tmp_path / "b", session_id="s1")
    [b_dep] = _save_n(other, "s1:call_b", b"b = 1\n", instance_id="inst-X")
    script.extend([_PAUSE, _CONFIRM])
    with pytest.raises(_Pause):
        _run(store, _planner(store))
    assert _run(other, _planner(other)) == D.TEXT_SUPERSEDED
    assert _run(store, _planner(store)) == D.TEXT_SUPERSEDED and len(prompts) == 2
    assert _listed_set(store) == {dep, maybe} and _listed_set(other) == {b_dep.ref}


def test_consumed_without_an_outcome_is_unknown_and_never_retried(store, monkeypatch):
    _, script, prompts = _graph(monkeypatch)
    dep, maybe = _deletable(store)
    script.extend([_PAUSE, _CONFIRM])
    with pytest.raises(_Pause):
        _run(store, _planner(store))

    def broken(*a, **k):
        raise StorageError("simulated")

    monkeypatch.setattr(store, "fence_deletion", broken)
    assert _run(store, _planner(store)) == D.TEXT_UNKNOWN
    monkeypatch.setattr(
        store, "fence_deletion", LocalFigureStore.fence_deletion.__get__(store)
    )
    assert _run(store, _planner(store)) == D.TEXT_UNKNOWN and len(prompts) == 2
    assert _listed_set(store) == {dep, maybe}


def test_a_refused_plan_creates_no_record(store, monkeypatch):
    saver, script, prompts = _graph(monkeypatch)
    refused = lambda: D.plan_dataset_deletion(store, name="prot_de", instance_id=None)
    assert _run(store, refused) == D._REFUSAL_TEXT["identity_unknown"]
    assert D._peek(saver, ("t", "ns:1", "call_1")) is None and prompts == []


def test_consents_are_minted_once_and_single_use(store):
    import copy

    _deletable(store)
    with pytest.raises(TypeError):
        D.Consent("fp", ())
    plan = _planner(store)()
    assert D._cas_consume(D._Record("another fp"), plan)[2] is None
    won, _, consent = D._cas_consume(D._Record(plan.fingerprint), plan)
    assert won and consent is not None
    for clone in (copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            clone(consent)
    assert D.delete_planned(store, plan, consent).status == "deleted"
    with pytest.raises(ValueError):  # spent
        D.delete_planned(store, plan, consent)
    other = D.plan_session_purge(store)
    _, _, foreign = D._cas_consume(D._Record(other.fingerprint), other)
    with pytest.raises(ValueError):  # given for another plan
        D.delete_planned(store, _planner(store)(), foreign)
    with pytest.raises(ValueError):  # and spent by that refused use
        D.delete_planned(store, other, foreign)


def test_snapshots_are_immutable_observations():
    saver, key = _Saver(), ("t", "ns", "c")
    rec, created, snap = D._get_or_create(saver, key, "fp")
    assert created and type(snap) is D.Snapshot
    assert all(v is None or type(v) is str for v in snap)
    assert D._cas(rec, D.PENDING, D.CONSUMED)[0]
    D._record_outcome(rec, "done")
    assert snap == ("pending", "fp", None) and D._peek(saver, key) == (
        "consumed",
        "fp",
        "done",
    )
    with pytest.raises(AttributeError):
        snap.state = "x"


def test_a_purge_plan_binds_each_legacy_file_identity(store):
    _save_n(store, "s1:call_a", b"a = 1\n")
    legacy = store.root / "old.html"
    legacy.write_text("x")
    plan = D.plan_session_purge(store)
    st = os.lstat(legacy)
    legacy.unlink()
    legacy.write_text("y")  # same count, same size: another file
    os.utime(legacy, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    replanned = D.plan_session_purge(store)
    assert len(replanned.legacy_files) == len(plan.legacy_files) == 1
    assert replanned.fingerprint != plan.fingerprint
    _, _, consent = D._cas_consume(D._Record(plan.fingerprint), plan)
    assert D.delete_planned(store, plan, consent).status == "superseded"
    _, _, consent = D._cas_consume(D._Record(replanned.fingerprint), replanned)
    out = D.delete_planned(store, replanned, consent)
    assert (
        out.status == "deleted" and "1 legacy plot file(s) were removed" in out.text()
    )
    assert store.list().items == [] and not legacy.exists()
    from lobster.core.figures.listing import legacy_plot_files

    assert legacy_plot_files(store.workspace) == []


# ---------- deletion seam fixes: consent binding, honest wording, known progress

from contextlib import contextmanager  # noqa: E402

from lobster.core.figures.store import (  # noqa: E402
    FenceReport,
    PurgeReport,
    RemoveReport,
)


def _consent_for(plan):
    won, _, consent = D._cas_consume(D._Record(plan.fingerprint), plan)
    assert won and consent is not None
    return consent


def test_a_retargeted_consent_still_authorizes_only_its_minted_plan(store):
    dep, maybe = _deletable(store)
    a_plan, b_plan = _planner(store)(), D.plan_session_purge(store)
    consent = _consent_for(a_plan)
    with pytest.raises(AttributeError):
        consent.fingerprint = b_plan.fingerprint
    with pytest.raises(AttributeError):
        del consent.scope
    object.__setattr__(
        consent, "fingerprint", b_plan.fingerprint
    )  # bypass: display only
    object.__setattr__(consent, "scope", b_plan.scope)
    with pytest.raises(ValueError, match="another plan"):
        D.delete_planned(store, b_plan, consent)
    with pytest.raises(ValueError, match="live"):  # that refused use spent it
        D.delete_planned(store, a_plan, consent)
    assert _listed_set(store) == {dep, maybe}


@pytest.mark.parametrize(
    "report, status, says, never",
    [
        (
            FenceReport(2, 0, 0, stopped=True),
            "partially_fenced",
            "No deletion record exists from this call.",
            "are hidden",
        ),
        (
            FenceReport(0, 0, 0, inventory_unreadable=True),
            "partially_fenced",
            "it is unknown which figures, if any, are hidden",
            "Figures with a deletion record are hidden",
        ),
        (
            FenceReport(2, 0, 0, stopped=True, failed_entry_may_exist=True),
            "partially_fenced",
            "One more may exist",
            "No deletion record exists",
        ),
        (
            FenceReport(2, 1, 1),
            "pending",
            "1 of 2 deletion record(s) confirmed",
            "No deletion record exists",
        ),
    ],
)
def test_an_incomplete_fence_claims_only_entries_known_to_exist(
    store, monkeypatch, report, status, says, never
):
    dep, maybe = _deletable(store)
    plan, called = _planner(store)(), []
    monkeypatch.setattr(store, "fence_deletion", lambda *a, **k: report)
    out = D.delete_planned(
        store, plan, _consent_for(plan), remove_dataset=lambda: called.append(1)
    )
    text = out.text()
    assert out.status == status and says in text and never not in text
    assert (
        "This call removed nothing." in text and "The dataset was not removed." in text
    )
    assert called == [] and out.deleted_refs == () and dep not in text


@pytest.mark.parametrize(
    "record, state",
    [
        (FenceReport(1, 0, 1), "written but unconfirmed"),
        (FenceReport(1, 0, 0, stopped=True, failed_entry_may_exist=True), "unknown"),
        (FenceReport(1, 0, 0, stopped=True), "not written"),
    ],
)
def test_an_incomplete_purge_record_claims_no_figure_was_hidden(
    store, monkeypatch, record, state
):
    _save_n(store, "s1:call_a", b"a = 1\n")
    plan = D.plan_session_purge(store)
    monkeypatch.setattr(store, "purge", lambda: PurgeReport(record, None))
    out = D.delete_planned(store, plan, _consent_for(plan))
    text = out.text()
    assert (
        f"its purge record is {state}" in text and "no figure deletion record" in text
    )
    assert "revisions are hidden" not in text and "removed nothing" in text


def test_a_purge_whose_fence_was_not_reconfirmed_removes_nothing(store, monkeypatch):
    _save_n(store, "s1:call_a", b"a = 1\n")
    plan = D.plan_session_purge(store)
    done = FenceReport(1, 1, 0)
    monkeypatch.setattr(store, "purge", lambda: PurgeReport(done, FenceReport(3, 3, 0)))
    out = D.delete_planned(store, plan, _consent_for(plan))
    assert out.status == "pending" and "could not be re-confirmed" in out.text()


def test_a_legacy_file_that_cannot_be_removed_is_never_called_deleted(
    store, monkeypatch
):
    _save_n(store, "s1:call_a", b"a = 1\n")
    legacy = store.root / "old.html"
    legacy.write_text("x")
    plan = D.plan_session_purge(store)
    real_unlink = os.unlink

    def unlink(path, *a, **k):
        if os.fspath(path).endswith("old.html"):
            raise PermissionError("simulated")
        return real_unlink(path, *a, **k)

    monkeypatch.setattr(store_module.os, "unlink", unlink)
    out = D.delete_planned(store, plan, _consent_for(plan))
    text = out.text()
    assert out.status == "removal_incomplete" and isinstance(out.removal, RemoveReport)
    assert out.removal.failed == 1 and "1 could not be removed" in text
    assert "may still be present" in text and legacy.exists()
    assert store.list().items == []  # the fenced NATIVE revisions stay hidden
    from lobster.core.figures.listing import legacy_plot_files

    assert legacy_plot_files(store.workspace) == ["old.html"]  # still listed: say so
    assert "1 legacy plot file(s) have no deletion record" in text
    assert "may still be present and listed" in text
    assert "Every saved figure" not in text and "were removed" not in text


def test_a_removal_raise_after_the_dataset_step_keeps_known_progress(
    store, monkeypatch
):
    dep, maybe = _deletable(store)
    plan, removed = _planner(store)(), []

    def broken(refs):
        raise StorageError("simulated")

    monkeypatch.setattr(store, "remove_fenced", broken)
    out = D.delete_planned(
        store, plan, _consent_for(plan), remove_dataset=lambda: removed.append(1)
    )
    text = out.text()
    assert out.status == "removal_incomplete" and out.removal_raised
    assert out.dataset_removed is True and "The dataset was removed." in text
    assert "what remains on disk is unknown" in text and dep in text
    assert removed == [1] and _listed_set(store) == {maybe}  # hidden by the fence


def test_a_dataset_step_that_changes_then_raises_is_unknown_never_not_removed(store):
    dep, maybe = _deletable(store)
    plan, changed = _planner(store)(), []

    def half_done():
        changed.append("dataset dropped")
        raise RuntimeError("simulated after a change")

    out = D.delete_planned(store, plan, _consent_for(plan), remove_dataset=half_done)
    text = out.text()
    assert changed and out.dataset_removed is None
    assert "The dataset removal outcome is unknown." in text
    assert "not removed" not in text and out.status == "deleted"
    assert _listed_set(store) == {maybe}


@contextmanager
def _runnable_config(conf):
    from langchain_core.runnables.config import var_child_runnable_config

    token = var_child_runnable_config.set(conf)
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


def _conf(**over):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.constants import CONFIG_KEY_CHECKPOINTER

    configurable = {
        CONFIG_KEY_CHECKPOINTER: InMemorySaver(),
        "thread_id": "thread-1",
        "checkpoint_ns": "",
    }
    for key, value in over.items():
        key = CONFIG_KEY_CHECKPOINTER if key == "saver" else key
        if value is _ABSENT:
            configurable.pop(key)
        else:
            configurable[key] = value
    return {"configurable": configurable}


_ABSENT = object()


def test_the_binding_comes_from_the_executing_saver_and_config():
    from langgraph.constants import CONFIG_KEY_CHECKPOINTER

    conf = _conf()
    with _runnable_config(conf):
        saver, key = D._resolve_binding("call_1")
    assert saver is conf["configurable"][CONFIG_KEY_CHECKPOINTER]
    assert key == ("thread-1", "", "call_1")  # an empty but PRESENT namespace is valid
    with _runnable_config(_conf(checkpoint_ns="tools:abc")):
        assert D._resolve_binding("call_1")[1] == ("thread-1", "tools:abc", "call_1")


@pytest.mark.parametrize(
    "over, tool_call_id",
    [
        ({"checkpoint_ns": _ABSENT}, "call_1"),
        ({"checkpoint_ns": None}, "call_1"),
        ({"thread_id": _ABSENT}, "call_1"),
        ({"thread_id": ""}, "call_1"),
        ({"thread_id": 7}, "call_1"),
        ({"saver": _ABSENT}, "call_1"),
        ({"saver": object()}, "call_1"),
        ({}, ""),
        ({}, None),
        ({}, 5),
    ],
)
def test_an_incomplete_binding_fails_closed(over, tool_call_id):
    with _runnable_config(_conf(**over)):
        assert D._resolve_binding(tool_call_id) is None
    with _runnable_config({"tags": []}):  # a runnable context without configurable
        assert D._resolve_binding("call_1") is None


# ------------------------------------------------------------------- pure figure_path

from lobster.core.figures import paths as P_paths  # noqa: E402


def test_figure_path_is_the_exact_commit_result_formula(store):
    [res] = _save_n(store, "s1:call_a", b"a = 1\n")
    ref = FigureRef.parse(res.ref)
    old = (
        (store.root / ref.figure_id / ref.revision_id / P_paths.FIGURE_FILE)
        .relative_to(store.workspace)
        .as_posix()
    )
    assert res.path == old == store.figure_path(res.ref) == store.figure_path(ref)


def test_figure_path_does_no_io_and_creates_nothing(tmp_path):
    fresh = LocalFigureStore(tmp_path / "ws", session_id="s1")
    ref = FigureRef(new_id(PREFIX_FIGURE), new_id(PREFIX_REVISION))
    path = fresh.figure_path(str(ref))
    assert path == f"plots/{ref.figure_id}/{ref.revision_id}/{P_paths.FIGURE_FILE}"
    assert not (tmp_path / "ws").exists()  # nothing created, nothing read


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "no-at-sign",
        "a@b@c",
        "../x@y",
        "fig@rev\n",
        f"{'f' * 3}/..@{'r' * 3}",
        None,
        5,
        ("fig", "rev"),
    ],
)
def test_figure_path_refuses_anything_that_is_not_a_figure_ref(store, bad):
    with pytest.raises(ValueError):
        store.figure_path(bad)


def test_figure_path_is_pure_and_the_result_delegates_to_it():
    import ast

    tree = ast.parse((_REPO / "lobster/core/figures/store.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LocalFigureStore"
    )
    fns = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    called = {
        ast_unparse(c.func)
        for c in ast.walk(fns["figure_path"])
        if isinstance(c, ast.Call)
    }
    assert called == {"self._rel", "self._rev_dir", "FigureRef.parse"}
    result_calls = {
        ast_unparse(c.func) for c in ast.walk(fns["_result"]) if isinstance(c, ast.Call)
    }
    assert "self.figure_path" in result_calls and "self._rel" not in result_calls


# ----------------- OPTED PlotManager binding, gated publication, forget
# Tests for the opted-in PlotManager binding (not enabled yet).


def _opted(tmp_path, instance_id="inst-A"):
    """An OPTED plot manager: dm's own one, reconfigured with the existing root before
    any managed use (the data manager wiring is not part of this binding)."""
    dm, pm = _pm_dm(tmp_path, instance_id)
    pm.configure_figure_lifecycle(
        store_provider=lambda: dm.figure_store,
        identity_resolver=dm.resolve_input_identity,
        session_id_provider=lambda: dm.session_id,
        d7_root=dm.figure_store.writer_lock,
    )
    return dm, pm


def _commit_ctx(pm, dm, call="c1", y=1):
    ctx = pm.begin_attempt()
    sid = _stage(pm, dm, ctx, y=y)
    [res] = pm.commit_staged(
        {"id": "a"}, tool_call_id=call, staged_ids=[sid], context=ctx
    )
    assert pm.end_attempt(ctx) == {}
    return res


def _refs(pm):
    return [p.get("figure_ref") for p in pm.latest_plots]


class _SpyLock:
    """Lock-shaped; records every acquisition."""

    def __init__(self):
        self.acquired = 0

    def __enter__(self):
        self.acquired += 1
        return self

    def __exit__(self, *exc):
        return False

    def acquire(self, *args, **kwargs):
        self.acquired += 1
        return True

    def release(self):
        pass


def _hold_in(pm, monkeypatch, where, commit):
    """Run ``commit()`` in a thread that blocks inside ``pm.<where>`` (its arguments are
    captured) until released. Returns (release, thread, box, seen)."""
    entered, release, box, seen = threading.Event(), threading.Event(), {}, {}
    real = getattr(pm, where)

    def blocking(*args, **kwargs):
        seen["args"] = args
        entered.set()
        assert release.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(pm, where, blocking)

    def run():
        try:
            box["results"] = commit()
        except BaseException as exc:  # recorded for the assertion
            box["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    if not entered.wait(10):  # never leave a writer running into fixture cleanup
        release.set()
        thread.join(10)
        pytest.fail("the commit thread never reached the blocking point")
    return release, thread, box, seen


def test_k2_first_opted_bind_refuses_a_foreign_lock_before_acquiring_it(tmp_path):
    import types

    from lobster.core.figures.errors import AttemptContextError

    dm, pm = _opted(tmp_path)
    spy, fenced = _SpyLock(), []
    foreign = types.SimpleNamespace(
        writer_lock=spy, attempt_fence=lambda: fenced.append(1)
    )
    pm.configure_figure_lifecycle(  # still unused: reconfiguration is allowed
        store_provider=lambda: foreign, d7_root=dm.figure_store.writer_lock
    )
    with pytest.raises(AttemptContextError):
        pm.begin_attempt()
    sid, _, _ = pm.add_plot(_bar(), title="t")
    with pytest.raises(AttemptContextError):
        pm.commit_staged({"id": "a"}, tool_call_id="c1", staged_ids=[sid])
    with pytest.raises(AttemptContextError):
        pm.forget_refs(foreign, [])
    assert spy.acquired == 0 and fenced == []
    assert pm._d7_store is None and sid in pm._staged  # nothing bound, nothing popped


def test_k2_a_different_store_after_first_use_is_refused_before_any_root(
    tmp_path, monkeypatch
):
    import types

    from lobster.core.figures.errors import AttemptContextError

    dm, pm = _opted(tmp_path)
    store = dm.figure_store
    res = _commit_ctx(pm, dm)
    assert pm._d7_store is store
    fenced = []
    other = types.SimpleNamespace(  # same root: only the object identity differs
        writer_lock=store.writer_lock, attempt_fence=lambda: fenced.append(1)
    )
    monkeypatch.setattr(pm, "_figure_store_provider", lambda: other)
    box = {}

    def run():
        try:
            pm.begin_attempt()
        except BaseException as exc:  # recorded for the assertion
            box["error"] = exc

    with store.writer_lock:  # held here: a thread that needed the root would block
        thread = threading.Thread(target=run)
        thread.start()
        thread.join(10)
        blocked = thread.is_alive()
    thread.join(10)
    assert not blocked and isinstance(box.get("error"), AttemptContextError)
    sid, _, _ = pm.add_plot(_bar(), title="t")
    with pytest.raises(AttemptContextError):
        pm.commit_staged({"id": "a"}, tool_call_id="c2", staged_ids=[sid])
    with pytest.raises(AttemptContextError):  # a supplied foreign store refuses
        pm.forget_refs(other, [])
    # a changed provider does not stop forgetting from the already-bound cache
    assert pm.forget_refs(store, [res.ref]) == frozenset({res.ref})
    assert fenced == [] and pm._d7_store is store and sid in pm._staged


def test_k2_an_opted_pm_refuses_reconfiguration_after_first_use(tmp_path):
    dm, pm = _opted(tmp_path)
    root = dm.figure_store.writer_lock
    _commit_ctx(pm, dm)
    for kwargs in (
        {},
        {"d7_root": root},
        {"store_provider": lambda: dm.figure_store, "d7_root": root},
    ):
        with pytest.raises(ValueError):
            pm.configure_figure_lifecycle(**kwargs)
    assert pm._d7_root is root and pm._d7_store is dm.figure_store
    _commit_ctx(pm, dm, call="c2")  # unchanged and still usable


def test_k2_a_reconfiguration_racing_the_first_bind_refuses_that_bind(tmp_path):
    from lobster.core.figures.errors import AttemptContextError

    dm, pm = _pm_dm(tmp_path)
    root = dm.figure_store.writer_lock

    def provider():  # runs OUTSIDE _attempt_lock, between snapshot and bind
        pm.configure_figure_lifecycle(
            store_provider=lambda: dm.figure_store, d7_root=root
        )
        return dm.figure_store

    pm.configure_figure_lifecycle(store_provider=provider, d7_root=root)
    with pytest.raises(AttemptContextError):
        pm.begin_attempt()
    assert pm._d7_store is None
    ctx = pm.begin_attempt()  # the new configuration binds normally
    assert pm._d7_store is dm.figure_store
    assert pm._attempts[ctx.nonce]["store"] is dm.figure_store
    pm.end_attempt(ctx)


def test_k2_a_hide_before_publication_publishes_into_neither_container(
    tmp_path, monkeypatch
):
    dm, pm = _opted(tmp_path)
    store = dm.figure_store
    ctx = pm.begin_attempt()
    sid = _stage(pm, dm, ctx)
    release, thread, box, seen = _hold_in(
        pm,
        monkeypatch,
        "_prepare_entry",  # unlocked: the store already returned AVAILABLE
        lambda: pm.commit_staged(
            {"id": "a"}, tool_call_id="c1", staged_ids=[sid], context=ctx
        ),
    )
    try:
        ref = seen["args"][1].ref
        store.purge()  # takes the root: the publication does not hold it yet
        with pytest.raises(FigureNotFoundError):  # hidden (or gone)
            store.read_record(ref)
        assert pm.forget_refs(store, [ref]) == frozenset()  # nothing published yet
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and "error" not in box
    [res] = box["results"]
    assert (res.status, res.ref) == ("available", ref)  # the known outcome stays
    assert pm.latest_plots == [] and pm.save_plots_to_workspace()[0] == []
    assert pm.end_attempt(ctx) == {}


class _ContendedRoot:
    """Delegates to the store's REAL root RLock; records every acquisition that found it
    held by another thread (the acquiring thread, then a blocking acquire)."""

    def __init__(self, real):
        self.real = real
        self.contenders = []
        self.contended = threading.Event()

    def acquire(self, blocking=True, timeout=-1):
        if self.real.acquire(blocking=False):  # free, or re-entered by its owner
            return True
        self.contenders.append(threading.current_thread())
        self.contended.set()
        return self.real.acquire(blocking, timeout)

    def release(self):
        self.real.release()

    def _is_owned(self):  # the store's _assert_locked uses it
        return self.real._is_owned()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def test_k2_a_forget_waits_for_the_publication_hold_then_removes_both(
    tmp_path, monkeypatch
):
    dm, pm = _pm_dm(tmp_path)
    store = dm.figure_store
    root = _ContendedRoot(store.writer_lock)
    monkeypatch.setattr(store, "_lock", root)  # store and plot manager share it
    pm.configure_figure_lifecycle(
        store_provider=lambda: dm.figure_store,
        identity_resolver=dm.resolve_input_identity,
        session_id_provider=lambda: dm.session_id,
        d7_root=root,
    )
    ctx = pm.begin_attempt()
    sid = _stage(pm, dm, ctx)
    release, thread, box, seen = _hold_in(
        pm,
        monkeypatch,
        "_append_entry",  # inside the publication's root hold
        lambda: pm.commit_staged(
            {"id": "a"}, tool_call_id="c1", staged_ids=[sid], context=ctx
        ),
    )
    out = {}
    ref = seen["args"][0]["figure_ref"]
    forget = threading.Thread(
        target=lambda: out.update(removed=pm.forget_refs(store, [ref]))
    )
    try:
        forget.start()
        assert root.contended.wait(10)  # the forget reached the HELD real root
        assert root.contenders == [forget] and "removed" not in out
    finally:
        release.set()
        thread.join(10)
        forget.join(10)
    assert not thread.is_alive() and not forget.is_alive() and "error" not in box
    assert out["removed"] == frozenset({ref})
    assert pm.latest_plots == [] and pm.save_plots_to_workspace()[0] == []
    [res] = box["results"]
    assert res.status == "available"


def test_k2_forget_after_publication_removes_both_containers(tmp_path):
    dm, pm = _opted(tmp_path)
    store = dm.figure_store
    res = _commit_ctx(pm, dm)
    assert _refs(pm) == [res.ref] and pm._committed_paths == [(res.ref, res.path)]
    store.purge()
    assert pm.forget_refs(store, [res.ref]) == frozenset({res.ref})
    assert pm.latest_plots == [] and pm.save_plots_to_workspace()[0] == []


def test_k2_a_second_opted_pm_on_the_same_root_never_publishes_a_hidden_ref(
    tmp_path, monkeypatch
):
    from lobster.core.plot_manager import PlotManager

    dm, _ = _opted(tmp_path)
    store = dm.figure_store
    other = PlotManager(workspace_path=tmp_path)
    other.configure_figure_lifecycle(
        store_provider=lambda: dm.figure_store,
        session_id_provider=lambda: dm.session_id,
        d7_root=store.writer_lock,
    )
    sid, _, _ = other.add_plot(_bar(), title="t")
    release, thread, box, seen = _hold_in(
        other,
        monkeypatch,
        "_prepare_entry",
        lambda: other.commit_staged({"id": "a"}, tool_call_id="c2", staged_ids=[sid]),
    )
    try:
        ref = seen["args"][1].ref
        store.purge()  # knows nothing about `other`'s containers
        with pytest.raises(FigureNotFoundError):
            store.read_record(ref)
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and "error" not in box
    assert [r.status for r in box["results"]] == ["available"]
    assert other.latest_plots == [] and other.save_plots_to_workspace()[0] == []


def test_k2_forget_reaches_an_evicted_queue_entry(tmp_path):
    dm, pm = _opted(tmp_path)
    store = dm.figure_store
    pm.max_plots_history = 1
    first = _commit_ctx(pm, dm, call="c1", y=1)
    second = _commit_ctx(pm, dm, call="c2", y=2)
    assert first.ref != second.ref and _refs(pm) == [second.ref]  # first evicted
    assert pm.forget_refs(store, [first.ref]) == frozenset({first.ref})
    assert _refs(pm) == [second.ref]
    assert pm.save_plots_to_workspace()[0] == [second.path]  # never the forgotten one


def test_k2_forget_returns_only_the_unique_refs_it_removed(tmp_path):
    dm, pm = _opted(tmp_path)
    store = dm.figure_store
    res = _commit_ctx(pm, dm)
    with pytest.raises(ValueError):  # malformed: before any container changes
        pm.forget_refs(store, [res.ref, "no-at-sign"])
    assert _refs(pm) == [res.ref] and pm._committed_paths == [(res.ref, res.path)]
    assert pm.forget_refs(store, [res.ref, res.ref]) == frozenset({res.ref})
    assert pm.forget_refs(store, [res.ref]) == frozenset()
    assert pm.forget_refs(store, []) == frozenset()


def test_k2_an_opted_pm_with_no_prior_use_binds_in_forget_without_a_figure(tmp_path):
    from lobster.core.figures.errors import AttemptContextError
    from lobster.core.plot_manager import PlotManager

    dm, pm = _opted(tmp_path)
    store = dm.figure_store
    res = _commit_ctx(pm, dm)
    count = len(store.list().items)
    fresh = PlotManager(workspace_path=tmp_path)
    fresh.configure_figure_lifecycle(
        store_provider=lambda: dm.figure_store, d7_root=store.writer_lock
    )
    assert fresh.clear_plots()[1]["cleared_count"] == 0  # empty and truthful
    assert fresh.save_plots_to_workspace()[0] == []
    assert fresh._d7_store is None  # clear/drain need no binding
    lookalike = LocalFigureStore(tmp_path, session_id=store.session_id)
    with pytest.raises(AttemptContextError):  # same root, not the provider's object
        fresh.forget_refs(lookalike, [res.ref])
    assert fresh._d7_store is None
    assert fresh.forget_refs(store, [res.ref]) == frozenset()
    assert fresh._d7_store is store
    assert len(store.list().items) == count  # no figure was generated
    assert _refs(pm) == [res.ref]  # another manager's cache is not traversed


def test_k2_a_non_opted_pm_keeps_todays_publication_queue_and_clear(
    tmp_path, monkeypatch
):
    from lobster.core.figures.errors import AttemptContextError

    dm, pm = _pm_dm(tmp_path)
    store = dm.figure_store
    assert pm._d7_root is None
    ctx = pm.begin_attempt()
    assert "store" not in pm._attempts[ctx.nonce]  # opted-only state
    sid = _stage(pm, dm, ctx)
    release, thread, box, _ = _hold_in(
        pm,
        monkeypatch,
        "_publish_entry",  # the old helper, called as before
        lambda: pm.commit_staged(
            {"id": "a"}, tool_call_id="c1", staged_ids=[sid], context=ctx
        ),
    )
    try:
        store.purge()  # no gate on a non-opted manager: NOT covered by local deletion
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and "error" not in box
    [res] = box["results"]
    assert _refs(pm) == [res.ref]  # published regardless, as today
    assert pm.save_plots_to_workspace()[0] == [res.path]  # paths only
    assert pm.end_attempt(ctx) == {} and pm._d7_store is None
    with pytest.raises(AttemptContextError):  # the opted-only API
        pm.forget_refs(store, [res.ref])
    pm.configure_figure_lifecycle(store_provider=lambda: dm.figure_store)  # as today
    assert pm.clear_plots()[1]["cleared_count"] == 1 and pm.latest_plots == []


class _ForbiddenLock:
    """Any acquisition fails the test."""

    def __enter__(self):
        raise AssertionError("_attempt_lock acquired")

    def __exit__(self, *exc):
        return False

    def acquire(self, *args, **kwargs):
        raise AssertionError("_attempt_lock acquired")

    def release(self):
        pass


def test_k2_a_bound_forget_inside_an_outer_root_hold_needs_no_provider_or_attempt_lock(
    tmp_path, monkeypatch
):
    """The held-root path of a bound forget. Behavioural guard for what the AST
    control cannot see (helper call-outs)."""
    from lobster.core.figures.errors import AttemptContextError

    dm, pm = _opted(tmp_path)
    store = dm.figure_store
    res = _commit_ctx(pm, dm)  # bound by a managed use, before the outer hold
    lookalike = LocalFigureStore(tmp_path, session_id=store.session_id)

    def provider():
        raise AssertionError("provider called")

    monkeypatch.setattr(pm, "_figure_store_provider", provider)
    monkeypatch.setattr(pm, "_attempt_lock", _ForbiddenLock())
    with store.writer_lock:  # the outer deletion hold
        with pytest.raises(AttemptContextError):  # same root, not the bound object
            pm.forget_refs(lookalike, [res.ref])
        assert pm.forget_refs(store, [res.ref]) == frozenset({res.ref})
        assert pm.forget_refs(store, [res.ref]) == frozenset()
    assert pm.latest_plots == [] and pm.save_plots_to_workspace()[0] == []


def test_k2_an_opted_prepare_failure_appends_nothing_and_keeps_the_outcome(
    tmp_path, monkeypatch
):
    dm, pm = _opted(tmp_path)
    ctx = pm.begin_attempt()
    sid = _stage(pm, dm, ctx)

    def broken(*args, **kwargs):
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(pm, "_prepare_entry", broken)
    with pytest.raises(RuntimeError):
        pm.commit_staged({"id": "a"}, tool_call_id="c1", staged_ids=[sid], context=ctx)
    assert pm.latest_plots == [] and pm._committed_paths == []  # neither container
    out = pm.end_attempt(ctx)
    assert out[sid].status == "available"  # recorded before the publication
    assert [i["figure_id"] for i in dm.figure_store.list().items] == [
        FigureRef.parse(out[sid].ref).figure_id
    ]
    assert pm.save_plots_to_workspace()[0] == []


class _NoReentryLock:
    """A real RLock whose re-entry by its owner fails the test (instead of hanging)."""

    def __init__(self):
        self.real = threading.RLock()

    def acquire(self, blocking=True, timeout=-1):
        if self.real._is_owned():
            raise AssertionError("_attempt_lock re-entered")
        return self.real.acquire(blocking, timeout)

    def release(self):
        self.real.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def test_k2_off_to_off_configuration_keeps_its_original_lock_behaviour(
    tmp_path, monkeypatch
):
    """Non-opted reconfiguration takes no lock, including from a
    session provider called under ``_attempt_lock`` by the context commit."""
    dm, pm = _pm_dm(tmp_path)
    monkeypatch.setattr(pm, "_attempt_lock", _ForbiddenLock())
    pm.configure_figure_lifecycle(  # OFF -> OFF: plain assignments
        store_provider=lambda: dm.figure_store,
        identity_resolver=dm.resolve_input_identity,
        session_id_provider=lambda: dm.session_id,
    )
    monkeypatch.setattr(pm, "_attempt_lock", _NoReentryLock())
    calls = []

    def session_id():  # runs under _attempt_lock (the batch base)
        calls.append(1)
        pm.configure_figure_lifecycle(
            store_provider=lambda: dm.figure_store,
            identity_resolver=dm.resolve_input_identity,
            session_id_provider=session_id,
        )
        return dm.session_id

    pm.configure_figure_lifecycle(
        store_provider=lambda: dm.figure_store,
        identity_resolver=dm.resolve_input_identity,
        session_id_provider=session_id,
    )
    res = _commit_ctx(pm, dm)
    assert calls and res.status == "available"
    assert pm._d7_root is None and pm._d7_store is None and pm._d7_gen == 0


def test_k2_no_root_publication_or_provider_work_under_the_attempt_lock():
    """DIRECT calls only: inside each ``with self._attempt_lock`` body no nested lock and
    no direct call of the listed names. Helper call-outs are NOT followed (not transitive
    proof). It does NOT see ``self._session_id()`` calling its provider under the lock
    (pre-existing), so it makes no transitive no-provider claim. The held-root path is
    guarded behaviourally by
    ``test_k2_a_bound_forget_inside_an_outer_root_hold_needs_no_provider_or_attempt_lock``.
    """
    import ast

    tree = ast.parse((_REPO / "lobster/core/plot_manager.py").read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PlotManager"
    )
    forbidden = {
        "_publish_result",
        "_publish_entry",
        "_prepare_entry",
        "_append_entry",
        "forget_refs",
        "read_record",
        "_bound_store",
        "_figure_store",
        "commit_attempt",
        "attempt_fence",
        "provider",
    }
    held = [
        w
        for w in ast.walk(cls)
        if isinstance(w, ast.With)
        and any(ast_unparse(i.context_expr) == "self._attempt_lock" for i in w.items)
    ]
    assert held
    for w in held:
        for node in ast.walk(w):
            assert node is w or not isinstance(node, ast.With), node.lineno
            if isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else getattr(func, "id", None)
                )
                assert name not in forbidden, (name, node.lineno)
    fns = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    for name in (
        "_publish_result",
        "forget_refs",
        "save_plots_to_workspace",
        "clear_plots",
    ):
        locks = {
            ast_unparse(i.context_expr)
            for w in ast.walk(fns[name])
            if isinstance(w, ast.With)
            for i in w.items
        }
        assert "self._attempt_lock" not in locks, name
