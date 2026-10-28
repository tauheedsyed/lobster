"""Local deletion seam: plans, the graph confirmation
map and the one executor. LOCAL only, ONE process per workspace directory (store docstring).

What this module guarantees, and only this:

- **Plans are facts.** :func:`plan_dataset_deletion` / :func:`plan_session_purge` read the
  store under its writer lock. Any unreadable inventory or unknown identity is a refusal;
  nothing partial authorizes anything. A plan's ``fingerprint`` binds the store scope (root
  identity) and the actual targets (refs, instance id, purge tombstones, each legacy file's
  lstat identity), never only counts.
- **Graph consent** (:func:`run_confirmed_deletion`): the record is keyed by the EXECUTING
  checkpoint saver plus (thread_id, checkpoint_ns, injected tool_call_id). One
  ``interrupt()`` per tool call. Exactly ``{"confirmed": True}`` (a plain dict, that one
  key, the value ``is True``) approves. A record moves ``pending`` → consumed | declined |
  superseded once; a terminal record answers with its recorded text before any plan,
  prompt or effect. One leaf lock (``_CONSENT_LOCK``) guards the map and every record
  field; nothing is called under it. Lock order: store writer RLock → leaf, never reverse.
- **Consent** is minted ONLY inside the winning ``pending → consumed`` transition and is
  single-use: :func:`delete_planned` refuses any object that is not a live minted consent
  (a copy, a forgery, a spent one). This is a guard against misuse by engine code, not a
  confinement of arbitrary code in this process.

Not here: the session/dataset deletion consumers, the CLI human-confirmation mint, any
Cloud path, known-local invalidation and the dataset-instance check (see delete_planned). A transition landing after the last observation can still race a lock-free
prompt; a later loser returns its atomic snapshot's text and never acts.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from lobster.core.figures.errors import FigureError

PENDING = "pending"
CONSUMED = "consumed"
DECLINED = "declined"
SUPERSEDED = "superseded"
TERMINAL = frozenset({CONSUMED, DECLINED, SUPERSEDED})

KIND_DATASET = "dataset"
KIND_SESSION_PURGE = "session_purge"

TEXT_UNAVAILABLE = "Deletion confirmation is unavailable here; nothing deleted."
TEXT_UNBOUND = "The confirmation could not be bound to this request; nothing deleted."
TEXT_DECLINED = "Deletion declined; nothing deleted."
TEXT_SUPERSEDED = (
    "The deletion scope changed since the warning; nothing deleted. "
    "Call again to see the current warning."
)
TEXT_UNKNOWN = "The outcome of the confirmed deletion is unknown; it is not retried."
TEXT_NOT_TRACKED = (
    "Copies outside this workspace (Cloud editor, exported packages, downloaded files) "
    "are not tracked or deleted."
)
_REFUSAL_TEXT = {
    "identity_unknown": "The dataset's identity is unknown; nothing deleted.",
    "inventory_unreadable": "The figure inventory could not be read; nothing deleted.",
}


# ---------------------------------------------------------------------------- plans
@dataclass(frozen=True)
class DeletionPlan:
    """One deletion's facts. ``fingerprint`` = sha256 of the canonical JSON of every other
    field (computed by :func:`_plan`)."""

    kind: str
    scope: Tuple[Any, ...]
    target: Tuple[Any, ...]  # dataset: (name, instance_id); purge: ()
    delete_refs: Tuple[str, ...]
    kept_refs: Tuple[str, ...]  # may depend (unverified): kept and listed
    purge_entries: Tuple[str, ...]  # strict-inventory tombstone names (purge)
    legacy_files: Tuple[Tuple[Any, ...], ...]  # (name, dev, ino, size, mtime_ns)
    refusal: Optional[str]
    fingerprint: str


def _plan(
    kind, scope, target=(), delete=(), kept=(), purge=(), legacy=(), refusal=None
):
    fields = {
        "kind": kind,
        "scope": list(scope),
        "target": list(target),
        "delete_refs": list(delete),
        "kept_refs": list(kept),
        "purge_entries": list(purge),
        "legacy_files": [list(f) for f in legacy],
        "refusal": refusal,
    }
    blob = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return DeletionPlan(
        kind,
        tuple(scope),
        tuple(target),
        tuple(delete),
        tuple(kept),
        tuple(purge),
        tuple(tuple(f) for f in legacy),
        refusal,
        hashlib.sha256(blob).hexdigest(),
    )


def store_scope(store) -> Tuple[Any, ...]:
    """The trusted local scope: workspace + root realpath, the root's lstat (dev, ino) so a
    deleted-and-recreated root on the same path differs, and the session id. A root that
    does not exist yet has (None, None)."""
    root = str(store.root)
    try:
        st = os.lstat(root)
        dev, ino = st.st_dev, st.st_ino
    except FileNotFoundError:
        dev = ino = None
    return (
        os.path.realpath(str(store.workspace)),
        os.path.realpath(root),
        dev,
        ino,
        str(store.session_id),
    )


def plan_dataset_deletion(
    store, *, name: str, instance_id: Optional[str]
) -> DeletionPlan:
    """Dataset-deletion plan for a dataset held by the caller as (name, instance_id). Verified
    dependents (``is True``) are deleted; every other matching ref is kept and listed. No
    name joins."""
    with store.writer_lock:
        try:
            scope = store_scope(store)
        except OSError:
            return _plan(
                KIND_DATASET, (), (name, instance_id), refusal="inventory_unreadable"
            )
        target = (name, instance_id)
        if not (
            isinstance(name, str)
            and name
            and isinstance(instance_id, str)
            and instance_id
        ):
            return _plan(KIND_DATASET, scope, target, refusal="identity_unknown")
        try:
            found = store.dependents(instance_id)
        except (FigureError, OSError, ValueError, TypeError):
            return _plan(KIND_DATASET, scope, target, refusal="inventory_unreadable")
        delete = sorted(ref for ref, verified in found if verified is True)
        kept = sorted(ref for ref, verified in found if verified is not True)
        return _plan(KIND_DATASET, scope, target, delete, kept)


def plan_session_purge(store) -> DeletionPlan:
    """Session-purge plan: every object of the strict purge inventory plus each legacy flat file."""
    with store.writer_lock:
        try:
            scope = store_scope(store)
            entries, legacy = store.purge_targets()
        except (FigureError, OSError):
            return _plan(KIND_SESSION_PURGE, (), refusal="inventory_unreadable")
        return _plan(KIND_SESSION_PURGE, scope, purge=entries, legacy=legacy)


def _replan(store, plan: DeletionPlan) -> DeletionPlan:
    if plan.kind == KIND_SESSION_PURGE:
        return plan_session_purge(store)
    name, instance_id = plan.target
    return plan_dataset_deletion(store, name=name, instance_id=instance_id)


def render_warning(plan: DeletionPlan) -> Dict[str, Any]:
    """The interrupt payload: facts only (refs are opaque IDs; the dataset name is a label
    and is shown sanitized)."""
    if plan.kind == KIND_SESSION_PURGE:
        lines = [
            f"Delete every saved figure of this session: {len(plan.purge_entries)} store "
            f"object(s) and {len(plan.legacy_files)} legacy plot file(s)."
        ]
    else:
        name = _label(plan.target[0])
        lines = [f"Delete dataset '{name}' (instance {_label(plan.target[1])})."]
        lines.append(
            "Figures deleted with it (verified dependents): "
            + (", ".join(plan.delete_refs) or "none")
        )
        if plan.kept_refs:
            lines.append(
                "Figures kept (may depend; not verified): " + ", ".join(plan.kept_refs)
            )
    lines.append(TEXT_NOT_TRACKED)
    return {
        "type": "confirm_deletion",
        "question": "\n".join(lines),
        "fingerprint": plan.fingerprint,
    }


def _label(value: Any, limit: int = 80) -> str:
    text = "".join(c for c in str(value) if c.isprintable() and c not in "'\\")
    return text[:limit]


# -------------------------------------------------------------------------- outcome
@dataclass(frozen=True)
class DeletionOutcome:
    """Known facts of one executed deletion; ``text()`` is fixed wording from them only.

    ``status``: ``deleted`` (fence complete and physical removal reported no failure; payload
    cleanup may still be pending), ``removal_incomplete`` (fence complete, but physical
    removal reported failures or raised: what remains is unknown or listed by count),
    ``pending`` (every planned fence entry exists but not all are confirmed durable),
    ``partially_fenced`` (fencing stopped or its inventory was unreadable), ``superseded``
    (nothing done). ``fence``/``purge_record``/``removal`` are the store's own reports.
    """

    status: str
    kind: str
    deleted_refs: Tuple[str, ...] = ()
    kept_refs: Tuple[str, ...] = ()
    dataset_removed: Any = "not_attempted"  # True | None (unknown) | "not_attempted"
    fence: Any = None  # FenceReport of the figure/instance fence (or purge tombstones)
    purge_record: Any = None  # FenceReport of the purge-action record (purge only)
    removal: Any = None  # RemoveReport, or None when removal was not reached
    removal_raised: bool = False  # removal raised: its state is unknown
    legacy_files: int = 0  # purge: planned legacy flat files (never covered by a fence)

    def text(self) -> str:
        if self.status == "superseded":
            return TEXT_SUPERSEDED
        parts: List[str] = []
        if self.status in ("partially_fenced", "pending"):
            parts.append(_fence_text(self))
            parts.append("This call removed nothing.")
        else:
            if self.kind == KIND_SESSION_PURGE:
                parts.append(
                    "This session's saved figure revisions are hidden (their deletion "
                    "records are confirmed)."
                )
                parts.append(_legacy_text(self))
            else:
                parts.append(
                    "Figures deleted (hidden): "
                    + (", ".join(self.deleted_refs) or "none")
                    + "."
                )
            parts.append(_removal_text(self))
        if self.kind == KIND_DATASET:
            parts.append(
                {
                    True: "The dataset was removed.",
                    None: "The dataset removal outcome is unknown.",
                }.get(self.dataset_removed, "The dataset was not removed.")
            )
            if self.kept_refs:
                parts.append("Kept (may depend): " + ", ".join(self.kept_refs) + ".")
        return " ".join(p for p in parts if p)


def _fence_text(outcome: "DeletionOutcome") -> str:
    """Honest wording for an incomplete fence, from the reports' counts only. "Hidden" is
    said only for entries known to exist; a possible entry is "may"."""
    record, fence = outcome.purge_record, outcome.fence
    if record is not None and not record.complete:  # tombstones are never written then
        state = (
            "written but unconfirmed"
            if record.unconfirmed
            else "unknown" if record.failed_entry_may_exist else "not written"
        )
        return (
            f"The purge did not complete: its purge record is {state}; this call wrote no "
            "figure deletion record."
        )
    if fence is None or fence.inventory_unreadable:
        return (
            "The deletion did not complete: its inventory could not be read; it is unknown "
            "which figures, if any, are hidden."
        )
    if fence.complete:  # complete but not re-confirmed (purge): never removal authority
        return (
            f"The deletion did not complete: its {fence.planned} deletion record(s) exist "
            "but could not be re-confirmed. Figures with a deletion record are hidden."
        )
    exist = fence.confirmed + fence.unconfirmed
    parts = [
        f"The deletion did not complete: {fence.confirmed} of {fence.planned} deletion "
        f"record(s) confirmed, {fence.unconfirmed} written but unconfirmed."
    ]
    if fence.failed_entry_may_exist:
        parts.append("One more may exist (its write failed).")
    if exist or fence.failed_entry_may_exist:
        parts.append("Figures with a deletion record are hidden.")
    else:
        parts.append("No deletion record exists from this call.")
    return " ".join(parts)


def _legacy_text(outcome: "DeletionOutcome") -> str:
    """Legacy flat files have no deletion record: they are gone only when removal reported
    no failure; otherwise they may still be present AND listed."""
    n = outcome.legacy_files
    if not n:
        return ""
    if outcome.status == "deleted":
        return f"Its {n} legacy plot file(s) were removed."
    return (
        f"Its {n} legacy plot file(s) have no deletion record; their removal is not "
        "confirmed and they may still be present and listed."
    )


def _removal_text(outcome: "DeletionOutcome") -> str:
    removal = outcome.removal
    if outcome.removal_raised:
        return "Physical removal did not complete; what remains on disk is unknown."
    if removal is None:
        return ""
    if removal.failed:
        return (
            f"Physical removal incomplete: {removal.removed} removed, {removal.failed} "
            "could not be removed and may still be present."
        )
    if removal.cleanup_pending:
        return "Removed; cleanup of unused stored files is pending."
    return "Removed."


def _fence_status(report) -> str:
    """An incomplete fence: ``pending`` when every entry exists but durability is not yet
    confirmed; ``partially_fenced`` when a write stopped or the inventory was unreadable.
    """
    if report.stopped or report.inventory_unreadable:
        return "partially_fenced"
    return "pending"


# -------------------------------------------------------------------------- consent
class Consent:
    """A single-use authorization for ONE plan. Only :func:`_cas_consume` creates one. The
    AUTHORITATIVE binding (fingerprint, scope) is held by the live registry, set at mint
    time; the attributes are read-only display copies, so re-targeting the object never
    changes what it authorizes. A guard against misuse by engine code, not a confinement
    of arbitrary code in this process."""

    __slots__ = ("fingerprint", "scope", "__weakref__")

    def __init__(self, fingerprint: str, scope: Tuple[Any, ...], _token: object = None):
        if _token is not _MINT:
            raise TypeError("a Consent is minted only by the confirmation map")
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "scope", scope)

    def __setattr__(self, name, value):
        raise AttributeError("a Consent is immutable")

    def __delattr__(self, name):
        raise AttributeError("a Consent is immutable")

    def __copy__(self):  # a copy is never a live consent
        raise TypeError("a Consent cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("a Consent cannot be copied")

    def __reduce__(self):
        raise TypeError("a Consent cannot be serialized")


_MINT = object()
#: live consent -> its immutable (fingerprint, scope) binding, set once at mint
_MINTED: "weakref.WeakKeyDictionary[Consent, Tuple[str, Tuple[Any, ...]]]" = (
    weakref.WeakKeyDictionary()
)


def _spend(consent: Any, plan: DeletionPlan) -> None:
    with _CONSENT_LOCK:
        binding = _MINTED.pop(consent, None) if isinstance(consent, Consent) else None
    # spent before any check can fail; the attributes are never consulted
    if binding is None:
        raise ValueError("not a live single-use consent")
    if binding != (plan.fingerprint, plan.scope):
        raise ValueError("the consent was given for another plan")


# ----------------------------------------------------------------- confirmation map
class Snapshot(NamedTuple):
    """An immutable observation of a record: every field is a str or None."""

    state: str
    fp: str
    outcome: Optional[str]


class _Record:
    __slots__ = ("state", "fp", "outcome")

    def __init__(self, fp: str):
        self.state, self.fp, self.outcome = PENDING, fp, None

    def snap(self) -> Snapshot:  # caller holds _CONSENT_LOCK
        return Snapshot(self.state, self.fp, self.outcome)


_CONSENTS: "weakref.WeakKeyDictionary[Any, Dict[Tuple[str, str, str], _Record]]" = (
    weakref.WeakKeyDictionary()
)
_CONSENT_LOCK = threading.Lock()  # LEAF: never held while calling anything else


def _peek(saver, key) -> Optional[Snapshot]:
    with _CONSENT_LOCK:
        rec = _CONSENTS.get(saver, {}).get(key)
        return None if rec is None else rec.snap()


def _get_or_create(saver, key, fp: str) -> Tuple[_Record, bool, Snapshot]:
    with _CONSENT_LOCK:
        recs = _CONSENTS.setdefault(saver, {})
        rec = recs.get(key)
        created = rec is None
        if created:
            rec = recs[key] = _Record(fp)
        return rec, created, rec.snap()


def _cas(rec: _Record, frm: str, to: str, *, fp: Optional[str] = None):
    """The only state transition: ``frm`` → ``to`` iff the record is in ``frm`` (and has
    ``fp`` when given). Returns (won, snapshot after)."""
    with _CONSENT_LOCK:
        if rec.state != frm or (fp is not None and rec.fp != fp):
            return False, rec.snap()
        rec.state = to
        return True, rec.snap()


def _cas_consume(rec: _Record, plan: DeletionPlan):
    """``pending`` → ``consumed`` for exactly this plan; the winner alone gets a Consent."""
    with _CONSENT_LOCK:
        if rec.state != PENDING or rec.fp != plan.fingerprint:
            return False, rec.snap(), None
        rec.state = CONSUMED
        consent = Consent(plan.fingerprint, plan.scope, _MINT)
        _MINTED[consent] = (plan.fingerprint, plan.scope)
        return True, rec.snap(), consent


def _record_outcome(rec: _Record, text: str) -> None:
    with _CONSENT_LOCK:
        if rec.state == CONSUMED and rec.outcome is None:
            rec.outcome = text


def _text(snap: Snapshot) -> str:
    if snap.state == CONSUMED:
        return snap.outcome if snap.outcome is not None else TEXT_UNKNOWN
    if snap.state == DECLINED:
        return TEXT_DECLINED
    if snap.state == SUPERSEDED:
        return TEXT_SUPERSEDED
    return TEXT_UNKNOWN  # pending is never rendered as an outcome


def _approved(answer: Any) -> bool:
    return (
        type(answer) is dict
        and set(answer) == {"confirmed"}
        and answer["confirmed"] is True
    )


def _resolve_binding(tool_call_id: Any):
    """(executing saver, (thread_id, checkpoint_ns, tool_call_id)) from the SAME runnable
    config ``interrupt()`` reads, or None (refuse): no runnable context, no saver, a missing
    or non-str key part."""
    try:
        from langgraph.checkpoint.base import BaseCheckpointSaver
        from langgraph.config import get_config
        from langgraph.constants import CONFIG_KEY_CHECKPOINTER

        conf = get_config().get("configurable") or {}
    except Exception:
        return None
    saver = conf.get(CONFIG_KEY_CHECKPOINTER)
    thread_id = conf.get("thread_id")
    if not (
        isinstance(saver, BaseCheckpointSaver)
        and isinstance(thread_id, str)
        and thread_id
        and "checkpoint_ns" in conf
        and isinstance(conf["checkpoint_ns"], str)
        and isinstance(tool_call_id, str)
        and tool_call_id
    ):
        return None
    return saver, (thread_id, conf["checkpoint_ns"], tool_call_id)


def _interrupt(payload: Dict[str, Any]) -> Any:
    from langgraph.types import interrupt

    return interrupt(payload)


# -------------------------------------------------------------------------- execute
def delete_planned(
    store,
    plan: DeletionPlan,
    consent: Any,
    *,
    remove_dataset: Optional[Callable[[], Any]] = None,
) -> DeletionOutcome:
    """Execute ONE consented plan under the store writer lock: spend the consent, re-plan
    and compare the fingerprint before any fence, then fence → (dataset step) → remove.
    ``remove_dataset`` runs only after a COMPLETE fence; if it raises an Exception the
    dataset outcome is UNKNOWN (None), never "not removed". A raise from physical removal
    keeps every known fact (fence, dataset step) and reports removal as unknown. Raises
    ValueError for anything that is not a live consent for this plan.

    OPEN (consumer work, not done here): (a) the re-plan uses the PLANNED instance id, so it
    cannot see the DataManager replacing the dataset under the same name; the consumer's
    ``remove_dataset`` must validate the current instance under this hold. (b) Forgetting
    known local refs (``dm.plot_manager`` entries and committed paths) is not done by this
    store-only executor; where it lives is to be agreed before any consumer is wired. This
    executor alone does not complete the deletion seam."""
    with store.writer_lock:
        _spend(consent, plan)
        if plan.refusal is not None:
            raise ValueError("a refused plan cannot be executed")
        fresh = _replan(store, plan)
        if fresh.refusal is not None or fresh.fingerprint != plan.fingerprint:
            return DeletionOutcome("superseded", plan.kind)
        if plan.kind == KIND_SESSION_PURGE:
            return _purge_outcome(plan, store.purge())
        name, instance_id = plan.target
        fence = store.fence_deletion(plan.delete_refs, instance_ids=(instance_id,))
        if not fence.complete:  # no dataset step, nothing removed
            return DeletionOutcome(
                _fence_status(fence), plan.kind, kept_refs=plan.kept_refs, fence=fence
            )
        known = dict(
            deleted_refs=plan.delete_refs, kept_refs=plan.kept_refs, fence=fence
        )
        dataset_removed: Any = "not_attempted"
        if remove_dataset is not None:
            try:
                remove_dataset()
                dataset_removed = True
            except Exception:  # it may have changed something: unknown, never False
                dataset_removed = None
        removal, raised = None, False
        if plan.delete_refs:
            try:
                removal = store.remove_fenced(plan.delete_refs)
            except Exception:  # known progress is kept; the removal state is unknown
                raised = True
        incomplete = raised or (removal is not None and removal.failed > 0)
        return DeletionOutcome(
            "removal_incomplete" if incomplete else "deleted",
            plan.kind,
            dataset_removed=dataset_removed,
            removal=removal,
            removal_raised=raised,
            **known,
        )


def _purge_outcome(plan: DeletionPlan, report) -> DeletionOutcome:
    """Map the store's PurgeReport to known facts only (no claim beyond its counts)."""
    known = dict(
        purge_record=report.record,
        fence=report.tombstones,
        legacy_files=len(plan.legacy_files),
    )
    if not report.record.complete:
        return DeletionOutcome(_fence_status(report.record), plan.kind, **known)
    tombs = report.tombstones
    if tombs is None or not tombs.complete:
        status = "partially_fenced" if tombs is None else _fence_status(tombs)
        return DeletionOutcome(status, plan.kind, **known)
    if report.removal is None:  # fence complete but not re-confirmed: nothing removed
        return DeletionOutcome("pending", plan.kind, **known)
    status = "removal_incomplete" if report.removal.failed else "deleted"
    return DeletionOutcome(status, plan.kind, removal=report.removal, **known)


def run_confirmed_deletion(
    store,
    planner: Callable[[], DeletionPlan],
    *,
    tool_call_id: Any,
    remove_dataset: Optional[Callable[[], Any]] = None,
) -> str:
    """The agent-tool flow (one interrupt slot per tool call).
    Returns fixed text. ``interrupt()`` raising pauses the graph (propagates)."""
    bound = _resolve_binding(tool_call_id)
    if bound is None:
        return TEXT_UNAVAILABLE
    saver, key = bound
    snap = _peek(saver, key)
    if snap is not None and snap.state in TERMINAL:
        return _text(snap)  # recorded outcome before any plan, prompt or effect
    plan = planner()
    if plan.refusal is not None:  # a terminal record wins over the refusal
        snap = _peek(saver, key)
        if snap is not None and snap.state in TERMINAL:
            return _text(snap)
        return _REFUSAL_TEXT.get(plan.refusal, TEXT_UNAVAILABLE)
    rec, created, snap = _get_or_create(saver, key, plan.fingerprint)
    if snap.state in TERMINAL:  # terminal at get/create, before fp/interrupt
        return _text(snap)
    stale = False
    if not created and snap.fp != plan.fingerprint:
        won, snap = _cas(rec, PENDING, SUPERSEDED)
        if not won:
            return _text(snap)
        stale = True  # this call's own supersede: it still uses its one slot
    answer = _interrupt(render_warning(plan))
    if (
        created
    ):  # an answer to a warning this record never bound (lost map, unstable key)
        won, snap = _cas(rec, PENDING, SUPERSEDED)
        return TEXT_UNBOUND if won else _text(snap)
    if stale:
        return TEXT_SUPERSEDED  # the answer is discarded
    if not _approved(answer):
        won, snap = _cas(rec, PENDING, DECLINED, fp=plan.fingerprint)
        return TEXT_DECLINED if won else _text(snap)
    with store.writer_lock:  # root RLock, then the leaf inside the CAS
        won, snap, consent = _cas_consume(rec, plan)
        if not won:
            return _text(snap)
        try:
            outcome = delete_planned(
                store, plan, consent, remove_dataset=remove_dataset
            )
        except Exception:  # consumed, outcome unknown: never retried, never raw text
            return TEXT_UNKNOWN
        text = outcome.text()
        _record_outcome(rec, text)
    return text
