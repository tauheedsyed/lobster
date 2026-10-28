"""Regressions for a source review of the figure core (findings 1–7).

Real filesystem (tmp_path); faults are injected at the OS-call seam only.
"""

from __future__ import annotations

import errno
import json
import logging
import os

import pytest

from lobster.core.figures import paths as P
from lobster.core.figures import store as store_mod
from lobster.core.figures.errors import (
    AttemptConflictError,
    DurabilityError,
    FigureInvalidError,
    IntegrityError,
    StoreUnsupportedError,
)
from lobster.core.figures.ids import FigureRef
from lobster.core.figures.payload import figure_to_bytes
from lobster.core.figures.records import (
    EXECUTION_REQUEST_SCHEMA,
    Candidate,
    InputEdge,
    activity_blob_bytes,
    attempt_digest,
    execution_request,
    json_blob_bytes,
    merge_inherited_inputs,
)
from lobster.core.figures.saver import TrustedSaver, summarize
from lobster.core.figures.store import LocalFigureStore

from .test_store_and_saver import _cand, _fig, _info

CANARY = "CANARY_sk-live-9f8e7d /Users/someone/secret.csv row=42"


@pytest.fixture()
def store(tmp_path):
    return LocalFigureStore(tmp_path, session_id="s1")


def _revisions(store):
    return store.list().items


# --- Finding 1: a retry can never attach a candidate to an incompatible recipe ------------


def test_retry_with_changed_code_and_added_index_attaches_nothing_to_old_execution(
    store,
):
    saver = TrustedSaver(store)
    [first] = saver.commit_attempt(_info(code=b"A\n"), [_cand(0)], succeeded=True)
    old_exec = store.read_record(first.ref)["execution"]["execution_id"]
    results = saver.commit_attempt(
        _info(code=b"B\n"),
        [_cand(0, _fig("blue")), _cand(1, _fig("green"))],
        succeeded=True,
    )
    assert [r.status for r in results] == ["refused", "refused"]
    assert {r.reason for r in results} == {
        "execution_record:attempt_conflict:execution_plan_changed"
    }
    assert [r["execution"]["execution_id"] for r in _revisions(store)] == [old_exec]


@pytest.mark.parametrize(
    "first,second",
    [((0,), (0, 1)), ((0, 1), (0,))],
    ids=["added_index", "removed_index"],
)
def test_retry_with_added_or_removed_index_conflicts(store, first, second):
    saver = TrustedSaver(store)
    colors = ["red", "blue"]
    saver.commit_attempt(
        _info(), [_cand(i, _fig(colors[i])) for i in first], succeeded=True
    )
    before = len(_revisions(store))
    results = saver.commit_attempt(
        _info(), [_cand(i, _fig(colors[i])) for i in second], succeeded=True
    )
    assert all(r.status == "refused" for r in results)
    assert len(_revisions(store)) == before


def test_retry_with_changed_parameters_conflicts(store):
    saver = TrustedSaver(store)
    saver.commit_attempt(_info(), [_cand()], succeeded=True)
    info = _info()
    changed = type(info)(**{**info.__dict__, "parameters": {"modality_name": "other"}})
    [res] = saver.commit_attempt(changed, [_cand()], succeeded=True)
    assert res.reason == "execution_record:attempt_conflict:execution_plan_changed"


def test_store_refuses_a_candidate_outside_the_execution_plan(store):
    [res] = TrustedSaver(store).commit_attempt(_info(), [_cand(0)], succeeded=True)
    execution_id = store.read_record(res.ref)["execution"]["execution_id"]
    stray = _cand(1, _fig("green"))
    with pytest.raises(AttemptConflictError) as exc:
        store.commit(stray, execution_id=execution_id, digest="0" * 64)
    assert exc.value.details["reason"] == "candidate_not_in_plan"
    foreign = _cand(0, base="s1:call_other")
    with pytest.raises(AttemptConflictError) as exc:
        store.commit(foreign, execution_id=execution_id, digest="0" * 64)
    assert exc.value.details["reason"] == "attempt_key_mismatch"


def test_planned_storage_failed_candidate_recovers_under_the_original_execution(
    store, monkeypatch
):
    saver = TrustedSaver(store)
    real = LocalFigureStore._write_revision
    calls = {"n": 0}

    def flaky(self, candidate, ref, **kw):
        if candidate.candidate_index == 1 and calls["n"] == 0:
            calls["n"] += 1
            raise OSError(errno.EIO, CANARY)
        return real(self, candidate, ref, **kw)

    monkeypatch.setattr(LocalFigureStore, "_write_revision", flaky)
    cands = [_cand(0), _cand(1, _fig("blue"))]
    r0, r1 = saver.commit_attempt(_info(), cands, succeeded=True)
    assert (r0.status, r1.status) == ("available", "failed")
    assert r1.reason == "storage:unexpected_error"
    retry_activity = {"id": "act-2", "tool_name": "execute_custom_code"}
    again = saver.commit_attempt(_info(activity=retry_activity), cands, succeeded=True)
    assert [r.status for r in again] == ["available", "available"]
    assert again[0].ref == r0.ref
    executions = {store.read_record(r.ref)["execution"]["execution_id"] for r in again}
    assert len(executions) == 1
    record = store.read_execution(executions.pop())
    assert record["activity_id"] == "act-1"  # the original immutable link is kept


# --- Finding 3: a matching untrusted edge can never downgrade a verified parent edge -------


def _edit(parent_ref, inputs, base="s1:omni_1"):
    return Candidate(
        attempt_key=f"{base}:0",
        candidate_index=0,
        figure_bytes=figure_to_bytes(_fig("orange")),
        config_bytes=b"{}",
        kind="edited",
        actor="omni:editor",
        parent_ref=parent_ref,
        inputs=inputs,
    )


def test_declared_duplicate_cannot_shadow_the_verified_parent_edge(store):
    saver = TrustedSaver(store)
    parent_inputs = (
        InputEdge("prot_de", "inst-1", "store_event", True),
        InputEdge("clinical", None, "declared", False),
    )
    [r1] = saver.commit_attempt(_info(), [_cand(inputs=parent_inputs)], succeeded=True)
    spoof = (InputEdge("prot_de", "inst-1", "declared", False, alias="de"),)
    [r2] = saver.commit_attempt(
        _info(code=None, base="s1:omni_1"), [_edit(r1.ref, spoof)], succeeded=True
    )
    assert r2.status == "available"
    inputs = store.read_record(r2.ref)["inputs"]
    prot = [e for e in inputs if e["name"] == "prot_de"]
    assert prot == [parent_inputs[0].to_dict()]  # authoritative, single, verified
    assert any(e["name"] == "clinical" for e in inputs)  # no dependency dropped
    verified = [ref for ref, ok in store.dependents("inst-1") if ok]
    assert verified == [r1.ref, r2.ref]


def test_record_and_descriptor_agree_on_inputs_and_unnormalized_commit_is_refused(
    store,
):
    saver = TrustedSaver(store)
    [r1] = saver.commit_attempt(_info(), [_cand()], succeeded=True)
    edit = _edit(r1.ref, ())
    [r2] = saver.commit_attempt(
        _info(code=None, base="s1:omni_1"), [edit], succeeded=True
    )
    record = store.read_record(r2.ref)
    parent = store.read_record(r1.ref)
    normalized = Candidate(
        **{
            **edit.__dict__,
            "inputs": tuple(InputEdge.from_dict(e) for e in record["inputs"]),
            "template": record["execution"]["template"],
        }
    )
    execution = store.read_execution(record["execution"]["execution_id"])
    assert record["execution"]["attempt_digest"] == attempt_digest(
        normalized,
        parent_figure_sha256=parent["figure"]["sha256"],
        parent_config_sha256=parent["config"]["sha256"],
        code_fingerprint=execution["code"],
        tool=execution["tool"],
        persist=execution["persist"],
        engine_version=execution["engine_version"],
        plotly_version=execution["plotly_version"],
    )
    # A direct store caller that skips normalization cannot write a disagreeing record.
    raw = _edit(r1.ref, (), base="s1:omni_2")
    with pytest.raises((FigureInvalidError, AttemptConflictError)):
        store.commit(
            raw, execution_id=record["execution"]["execution_id"], digest="0" * 64
        )


def test_merge_is_idempotent_and_keeps_order():
    parent = [
        InputEdge("a", "1", "store_event", True).to_dict(),
        InputEdge("b", None, "declared", False).to_dict(),
    ]
    cand = (
        InputEdge("c", "3", "content_hash", True),
        InputEdge("a", "1", "declared", False),
    )
    merged = merge_inherited_inputs(cand, parent)
    assert [(e.name, e.verified) for e in merged] == [
        ("c", True),
        ("a", True),
        ("b", False),
    ]
    assert merge_inherited_inputs(merged, parent) == merged


# --- Finding 4: byte ABI of commit_execution -------------------------------------------------


def test_execution_request_is_a_byte_abi_without_ids_paths_scope_or_time(store):
    info = _info()
    activity = activity_blob_bytes(info.activity)
    request = execution_request(info, [(0, "d" * 64)], activity_bytes=activity)
    assert request["schema"] == EXECUTION_REQUEST_SCHEMA
    assert not {"execution_id", "scope", "created_at", "path"} & set(request)
    assert request["code_id"].startswith("sha256-")
    assert request["activity_id"] == "act-1"
    record_bytes = json_blob_bytes(request)
    with pytest.raises(IntegrityError):
        store.commit_execution(record_bytes, b"other code", activity)
    with pytest.raises(FigureInvalidError):
        store.commit_execution(record_bytes, info.code_bytes, None)
    execution_id = store.commit_execution(record_bytes, info.code_bytes, activity)
    assert (
        store.commit_execution(record_bytes, info.code_bytes, activity) == execution_id
    )


# --- Finding 5: no raw exception text in outcomes, summaries or logs ------------------------


def test_raw_exception_text_never_surfaces(store, monkeypatch, caplog):
    def boom(*_a, **_k):
        raise RuntimeError(CANARY)

    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(LocalFigureStore, "commit", boom)
    [res] = TrustedSaver(store).commit_attempt(_info(), [_cand()], succeeded=True)
    assert res.reason == "storage:unexpected_error"
    monkeypatch.setattr(LocalFigureStore, "commit_execution", boom)
    [res2] = TrustedSaver(store).commit_attempt(
        _info(base="s1:call_2"), [_cand(base="s1:call_2")], succeeded=True
    )
    assert res2.reason == "execution_record:unexpected_error"
    surfaced = " ".join(
        [
            summarize([res, res2]),
            json.dumps([res.to_dict(), res2.to_dict()]),
            caplog.text,
        ]
    )
    assert "CANARY" not in surfaced and "/Users/someone" not in surfaced


# --- Finding 6: directory durability failures are never swallowed ---------------------------


def test_fsync_dir_raises_on_open_and_fsync_failure(tmp_path, monkeypatch):
    def fail_eio(_fd):
        raise OSError(errno.EIO, CANARY)

    def fail_einval(_fd):
        raise OSError(errno.EINVAL, "unsupported")

    monkeypatch.setattr(store_mod.os, "fsync", fail_eio)
    with pytest.raises(DurabilityError) as exc:
        store_mod._fsync_dir(tmp_path)
    assert exc.value.details["reason"] == "dir_fsync_failed"
    monkeypatch.setattr(store_mod.os, "fsync", fail_einval)
    with pytest.raises(DurabilityError) as exc:
        store_mod._fsync_dir(tmp_path)
    assert exc.value.details["reason"] == "dir_fsync_unsupported"
    with pytest.raises(DurabilityError) as exc:
        store_mod._fsync_dir(tmp_path / "missing")
    assert exc.value.details["reason"] == "dir_open_failed"


def _fail_dir_fsync(monkeypatch, predicate):
    real = store_mod._fsync_dir

    def fake(path):
        if predicate(path):
            raise DurabilityError("injected", details={"reason": "dir_fsync_failed"})
        return real(path)

    monkeypatch.setattr(store_mod, "_fsync_dir", fake)


@pytest.mark.parametrize("target", [P.CODE_DIR, P.EXECUTIONS_DIR, P.ATTEMPTS_DIR])
def test_pre_revision_durability_failure_publishes_nothing(store, monkeypatch, target):
    _fail_dir_fsync(monkeypatch, lambda p: p.name == target)
    [res] = TrustedSaver(store).commit_attempt(_info(), [_cand()], succeeded=True)
    assert res.status == "failed"
    assert res.reason == "execution_record:durability_failed:dir_fsync_failed"
    assert _revisions(store) == []


def test_candidate_claim_durability_failure_publishes_nothing(store, monkeypatch):
    seen = {"n": 0}

    def second_attempts_fsync(p):
        if p.name == P.ATTEMPTS_DIR:
            seen["n"] += 1
            return seen["n"] == 2  # 1st = execution claim, 2nd = candidate claim
        return False

    _fail_dir_fsync(monkeypatch, second_attempts_fsync)
    [res] = TrustedSaver(store).commit_attempt(_info(), [_cand()], succeeded=True)
    assert (res.status, res.reason) == ("failed", "durability_failed:dir_fsync_failed")
    assert _revisions(store) == []


def test_revision_marker_durability_failure_is_not_reported_available(
    store, monkeypatch
):
    _fail_dir_fsync(
        monkeypatch,
        lambda p: p.name.startswith("r_") and (p / P.REVISION_FILE).exists(),
    )
    [res] = TrustedSaver(store).commit_attempt(_info(), [_cand()], succeeded=True)
    # the marker was linked but not confirmed: an uncertain acknowledgement is PENDING
    assert (res.status, res.reason) == ("pending", "durability_failed:dir_fsync_failed")
    assert res.ref is None and res.path is None  # no public ref until confirmed
    monkeypatch.undo()
    [again] = TrustedSaver(store).commit_attempt(_info(), [_cand()], succeeded=True)
    assert again.status == "available"  # retry re-verifies and re-confirms durability


def test_non_posix_platform_is_refused_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod.os, "name", "nt")
    with pytest.raises(StoreUnsupportedError):
        LocalFigureStore(tmp_path)


# --- Finding 7: reuse verifies the committed artifact, not marker presence -----------------


@pytest.mark.parametrize("damaged", [P.FIGURE_FILE, P.CONFIG_FILE])
def test_same_key_retry_on_damaged_payload_is_not_reported_available(store, damaged):
    saver = TrustedSaver(store)
    [res] = saver.commit_attempt(_info(), [_cand()], succeeded=True)
    path = store._rev_dir(FigureRef.parse(res.ref)) / damaged
    os.chmod(path, 0o644)
    path.write_bytes(path.read_bytes() + b" ")
    [again] = saver.commit_attempt(_info(), [_cand()], succeeded=True)
    assert again.status == "failed"
    assert again.reason.startswith("fingerprint_mismatch")


def test_same_key_retry_on_deleted_payload_is_not_reported_available(store):
    saver = TrustedSaver(store)
    [res] = saver.commit_attempt(_info(), [_cand()], succeeded=True)
    os.unlink(store._rev_dir(FigureRef.parse(res.ref)) / P.FIGURE_FILE)
    [again] = saver.commit_attempt(_info(), [_cand()], succeeded=True)
    assert again.status == "failed"


def test_shared_execution_request_vectors_reproduce_and_are_accepted(tmp_path):
    """Generated wire vectors for Cloud: exact record/code/activity bytes + hashes."""
    import hashlib
    from pathlib import Path

    from lobster.core.figures.records import execution_plan_digest

    path = Path(__file__).parent / "fixtures" / "execution_request_vectors.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["request_schema"] == EXECUTION_REQUEST_SCHEMA
    for vec in doc["positives"]:
        record = vec["record_utf8"].encode("utf-8")
        code = (
            vec["code_utf8"].encode("utf-8") if vec["code_utf8"] is not None else None
        )
        activity = (
            vec["activity_utf8"].encode("utf-8")
            if vec["activity_utf8"] is not None
            else None
        )
        request = json.loads(record)
        assert json_blob_bytes(request) == record  # canonical form is a fixed point
        assert hashlib.sha256(record).hexdigest() == vec["record_sha256"]
        assert execution_plan_digest(request) == vec["plan_digest"]
        store = LocalFigureStore(tmp_path / vec["name"], session_id="s1")
        execution_id = store.commit_execution(record, code, activity)
        stored = store.read_execution(execution_id)
        assert stored["request_sha256"] == vec["record_sha256"]
        assert stored["plan_digest"] == vec["plan_digest"]


def test_planned_subset_may_have_holes_and_counts_every_candidate(store):
    bad = Candidate("s1:call_1:1", 1, b"not json", b"{}")
    results = TrustedSaver(store).commit_attempt(
        _info(), [_cand(0), bad, _cand(2, _fig("blue"))], succeeded=True
    )
    assert [r.status for r in results] == ["available", "refused", "available"]
    execution = store.read_execution(
        store.read_record(results[0].ref)["execution"]["execution_id"]
    )
    assert execution["candidate_count"] == 3
    assert [p["index"] for p in execution["planned_candidates"]] == [0, 2]


def test_non_json_native_parameters_are_refused_not_repr_d(store):
    import datetime

    info = _info()
    odd = type(info)(
        **{**info.__dict__, "parameters": {"when": datetime.date(2026, 1, 1)}}
    )
    [res] = TrustedSaver(store).commit_attempt(odd, [_cand()], succeeded=True)
    assert (res.status, res.reason) == (
        "refused",
        "execution_record:figure_invalid:execution_request_not_json",
    )
    assert _revisions(store) == []


def test_shared_negative_vectors_are_all_rejected_with_the_recorded_code(tmp_path):
    from pathlib import Path

    from lobster.core.figures.errors import FigureError

    path = Path(__file__).parent / "fixtures" / "execution_request_vectors.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert len(doc["negatives"]) >= 16
    for i, vec in enumerate(doc["negatives"]):
        enc = lambda t: None if t is None else t.encode("utf-8")  # noqa: E731
        store = LocalFigureStore(tmp_path / f"n{i}", session_id="s1")
        with pytest.raises(FigureError) as exc:
            store.commit_execution(
                enc(vec["record_utf8"]),
                enc(vec["code_utf8"]),
                enc(vec["activity_utf8"]),
            )
        assert f"{exc.value.code}:{exc.value.details.get('reason')}" == vec["expected"]


def test_vector_generator_is_deterministic(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    here = Path(__file__).parent / "fixtures"
    spec = importlib.util.spec_from_file_location(
        "gen_vectors", here / "gen_execution_request_vectors.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OUT", tmp_path / "v.json")
    mod.main()
    assert (tmp_path / "v.json").read_bytes() == (
        here / "execution_request_vectors.json"
    ).read_bytes()


# --- Finding 5 follow-up: fixed vocabulary, not character shape -----------------

MARKER = "synthetic_sensitive_marker_123"


def test_safe_reason_uses_fixed_vocabulary_not_token_shape():
    from lobster.core.figures.errors import FigureError, FigureInvalidError, safe_reason

    assert safe_reason(FigureError("x", details={"reason": MARKER})) == "figure_error"
    assert (
        safe_reason(FigureInvalidError("x", details={"reason": "plotly_schema"}))
        == "figure_invalid:plotly_schema"
    )

    class Sneaky(FigureInvalidError):
        code = MARKER

    assert safe_reason(Sneaky("x")) == "figure_invalid"  # subclass override ignored
    inst = FigureInvalidError("x")
    inst.code = MARKER
    assert safe_reason(inst) == "figure_invalid"  # instance override ignored
    assert safe_reason(ValueError(MARKER)) == "unexpected_error"


def test_safe_result_reason_accepts_only_composed_vocabulary():
    from lobster.core.figures.errors import safe_result_reason

    for ok in (
        "storage:unexpected_error",
        "execution_record:attempt_conflict:execution_plan_changed",
        "durability_failed:dir_fsync_failed",
        "html_rendition:fingerprint_mismatch:figure_changed",
        "attempt_failed",
        "figure_invalid",
    ):
        assert safe_result_reason(ok) == ok
    for bad in (
        MARKER,
        f"storage:{MARKER}",
        f"figure_invalid:{MARKER}",
        f"execution_record:figure_invalid:{MARKER}",
        "figure_invalid:plotly_schema:extra",
        None,
        42,
        {"reason": "x"},
    ):
        assert safe_result_reason(bad) is None


def test_failure_summary_withholds_unknown_nested_reasons_and_malformed_shapes():
    from lobster.core.figures.errors import FigureError
    from lobster.core.figures.saver import failure_summary

    nested = FigureError(
        "x",
        details={
            "reason": "no_figure_available",
            "results": [
                {"status": "failed", "reason": MARKER},
                {"status": MARKER, "reason": "storage:unexpected_error"},
            ],
        },
    )
    text = failure_summary(nested)
    assert MARKER not in text
    assert text == "failed: reason_withheld; failed: storage:unexpected_error"
    for malformed in ("not a list", [MARKER], {"a": 1}, None):
        exc = FigureError(
            "x", details={"reason": "no_figure_available", "results": malformed}
        )
        assert failure_summary(exc) == "figure_error:no_figure_available"
    assert failure_summary(RuntimeError(MARKER)) == "unexpected_error"


def test_summary_and_caller_failure_reason_never_echo_unknown_strings(store):
    from lobster.core.figures.records import CommitResult

    text = summarize([CommitResult(0, "failed", reason=MARKER, label="L")])
    assert MARKER not in text and "reason_withheld" in text
    [res] = TrustedSaver(store).commit_attempt(
        _info(), [_cand()], succeeded=False, failure_reason=MARKER
    )
    assert res.reason == "attempt_failed"


def test_detail_vocabulary_covers_every_reason_literal_in_the_core():
    import re
    from pathlib import Path

    import lobster
    from lobster.core.figures.errors import DETAIL_REASONS

    root = Path(lobster.__path__[0])
    files = list((root / "core" / "figures").glob("*.py")) + [
        root / "core" / "plot_manager.py"
    ]
    pattern = re.compile(r'(?:"reason":\s*|bad\(|reason\s*=\s*)"([a-z0-9_]+)"')
    used = {m for f in files for m in pattern.findall(f.read_text(encoding="utf-8"))}
    used |= {
        "dir_fsync_failed",
        "dir_fsync_unsupported",
        "non_strict_json",
        "invalid_json",
    }
    assert used - DETAIL_REASONS - {"attempt_failed", "unexpected_error"} == set()


def test_all_pending_is_reported_as_pending_not_as_no_save():
    from lobster.core.figures.records import CommitResult
    from lobster.core.figures.saver import HEADER_NONE, HEADER_PENDING, outcome_state

    pending = [
        CommitResult(0, "pending", reason=None),
        CommitResult(1, "failed", reason="storage:unexpected_error"),
    ]
    text = summarize(pending)
    assert text.startswith(HEADER_PENDING) and HEADER_NONE not in text
    assert outcome_state(pending) == "pending"
    assert summarize(
        [CommitResult(0, "refused", reason="attempt_conflict")]
    ).startswith(HEADER_NONE)
    assert (
        outcome_state([CommitResult(0, "available", ref="f_a@r_b", path="p")])
        == "available"
    )
