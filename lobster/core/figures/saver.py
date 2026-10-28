"""Trusted saver: the one save step outside model code (contract §6.3).

Order per attempt:
  1. fence: nothing is committed unless the attempt succeeded;
  2. validate every candidate (strict JSON, caps, Plotly schema) without modifying bytes;
  3. compute semantic digests;
  4. durably write ONE execution record (code + exact activity record) — before any revision;
  5. commit candidates one by one; per-candidate outcomes are preserved.

Surfaced reasons are stable codes only (:func:`safe_reason`): no exception message, repr or
argument ever reaches a CommitResult, the summary or the log.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lobster.core.figures.errors import (
    FigureError,
    FigureInvalidError,
    safe_reason,
    safe_result_reason,
)
from lobster.core.figures.payload import (
    declared_template,
    validate_config_bytes,
    validate_figure_bytes,
)
from lobster.core.figures.records import (
    STATUS_AVAILABLE,
    STATUS_FAILED,
    STATUS_NOT_COMMITTED,
    STATUS_PENDING,
    STATUS_REFUSED,
    STATUSES,
    Candidate,
    CommitResult,
    ExecutionInfo,
    activity_blob_bytes,
    attempt_digest,
    execution_request,
    json_blob_bytes,
    merge_inherited_inputs,
)
from lobster.core.figures.store import FigureStore

logger = logging.getLogger(__name__)

#: Error codes that are a refusal by policy/lifecycle (vs. an unexpected failure).
_REFUSAL_CODES = {
    "figure_invalid",
    "attempt_conflict",
    "parent_deleted",
    "generation_fenced",
    "figure_not_found",
    "figure_deleted",
}


class TrustedSaver:
    def __init__(self, store: FigureStore):
        self.store = store

    def commit_attempt(
        self,
        info: ExecutionInfo,
        candidates: Sequence[Candidate],
        *,
        succeeded: bool,
        failure_reason: Optional[str] = None,
        pre_refused: Optional[Mapping[int, FigureError]] = None,
        fence: Optional[Any] = None,
    ) -> List[CommitResult]:
        """Commit one attempt's candidates (all-or-nothing per candidate, never per batch).

        ``pre_refused`` (trusted tool input, NOT a wire/Candidate field): candidate indices the
        caller already refused with a typed error (e.g. an edit whose parent config could not
        be read). They keep their ordinal and count toward ``candidate_count`` but never enter
        validation, the digest, ``planned_candidates``, storage or public refs.

        ``fence`` (the attempt fence, a store ``AttemptFence``): passed to the store's
        ``commit_execution``/``commit`` only when not None, so the ``FigureStore`` protocol
        is unchanged for stores without one.
        """
        fence_kw: Dict[str, Any] = {} if fence is None else {"fence": fence}
        if not succeeded:
            # caller-supplied text is never surfaced unless it is fixed vocabulary
            reason = safe_result_reason(failure_reason) or "attempt_failed"
            return [
                CommitResult(
                    c.candidate_index,
                    STATUS_NOT_COMMITTED,
                    reason=reason,
                    label=c.label,
                )
                for c in candidates
            ]
        if not candidates:
            if pre_refused:
                raise ValueError("pre_refused names candidates that were not submitted")
            return []
        pre = dict(pre_refused or {})
        if pre:  # an ambiguous mapping is rejected, never silently masked
            indices = [c.candidate_index for c in candidates]
            if len(set(indices)) != len(indices):
                raise ValueError("pre_refused needs unique candidate indices")
            if (
                not all(
                    type(k) is int for k in pre
                )  # no bool/float aliases of an index
                or not set(pre) <= set(indices)
                or not all(isinstance(e, FigureError) for e in pre.values())
            ):
                raise ValueError(
                    "pre_refused must map submitted indices to FigureError"
                )

        prepared: List[tuple] = []  # (candidate, digest)
        results: dict[int, CommitResult] = {}
        for cand in candidates:
            if cand.candidate_index in pre:  # same status/reason policy as any refusal
                results[cand.candidate_index] = _error_result(
                    cand, pre[cand.candidate_index]
                )
                continue
            try:
                parsed = validate_figure_bytes(cand.figure_bytes)
                validate_config_bytes(cand.config_bytes)
                template = (
                    cand.template
                    if cand.template is not None
                    else declared_template(parsed)
                )
                parent_fig_sha = parent_cfg_sha = None
                parent_inputs: list = []
                if cand.parent_ref:
                    parent_record = self.store.read_record(cand.parent_ref)
                    parent_fig_sha = parent_record["figure"]["sha256"]
                    parent_cfg_sha = parent_record["config"]["sha256"]
                    parent_inputs = parent_record.get("inputs", [])
                # Authoritative inputs BEFORE the digest, so descriptor and record agree.
                cand = _replace(
                    cand,
                    template=template,
                    inputs=merge_inherited_inputs(cand.inputs, parent_inputs),
                )
                digest = attempt_digest(
                    cand,
                    parent_figure_sha256=parent_fig_sha,
                    parent_config_sha256=parent_cfg_sha,
                    code_bytes=info.code_bytes,
                    tool=info.tool,
                    persist=info.persist,
                    engine_version=info.engine_version,
                    plotly_version=info.plotly_version,
                )
                prepared.append((cand, digest))
            except FigureError as exc:
                results[cand.candidate_index] = _error_result(cand, exc)
            except Exception as exc:
                results[cand.candidate_index] = _unexpected(cand, "prepare", exc)

        if prepared:
            try:
                activity_bytes = (
                    activity_blob_bytes(info.activity)
                    if info.activity is not None
                    else None
                )
                request = execution_request(
                    info,
                    [(c.candidate_index, d) for c, d in prepared],
                    activity_bytes=activity_bytes,
                    candidate_count=len(candidates),
                )
                try:
                    record_bytes = json_blob_bytes(request)
                except (TypeError, ValueError):
                    raise FigureInvalidError(
                        "execution request is not JSON-native",
                        details={"reason": "execution_request_not_json"},
                    ) from None
                execution_id = self.store.commit_execution(
                    record_bytes, info.code_bytes, activity_bytes, **fence_kw
                )
            except (
                Exception
            ) as exc:  # no revision may become available without its execution record
                reason = "execution_record:" + safe_reason(exc)
                status = (
                    STATUS_REFUSED
                    if isinstance(exc, FigureError) and exc.code in _REFUSAL_CODES
                    else STATUS_FAILED
                )
                logger.warning("figure execution record not written: %s", reason)
                for cand, _ in prepared:
                    results[cand.candidate_index] = CommitResult(
                        cand.candidate_index, status, reason=reason, label=cand.label
                    )
                prepared = []
            for cand, digest in prepared:
                try:
                    results[cand.candidate_index] = self.store.commit(
                        cand, execution_id=execution_id, digest=digest, **fence_kw
                    )
                except FigureError as exc:
                    results[cand.candidate_index] = _error_result(cand, exc)
                except (
                    Exception
                ) as exc:  # one candidate's failure never hides the others
                    results[cand.candidate_index] = _unexpected(cand, "storage", exc)
        return [results[c.candidate_index] for c in candidates]


def _error_result(cand: Candidate, exc: FigureError) -> CommitResult:
    status = STATUS_REFUSED if exc.code in _REFUSAL_CODES else STATUS_FAILED
    reason = safe_reason(exc)
    logger.warning("figure candidate %s not saved: %s", cand.candidate_index, reason)
    return CommitResult(cand.candidate_index, status, reason=reason, label=cand.label)


def _unexpected(cand: Candidate, stage: str, exc: BaseException) -> CommitResult:
    reason = f"{stage}:unexpected_error"
    logger.warning(
        "figure candidate %s not saved: %s (%s)",
        cand.candidate_index,
        reason,
        type(exc).__name__,
    )
    return CommitResult(
        cand.candidate_index, STATUS_FAILED, reason=reason, label=cand.label
    )


def _replace(cand: Candidate, **changes) -> Candidate:
    from dataclasses import replace

    return replace(cand, **changes)


def summarize(results: Sequence[CommitResult]) -> str:
    """Human/model-facing one-line-per-figure summary. Never claims display."""
    if not results:
        return HEADER_NONE
    lines = []
    for res in results:
        label = f' ("{res.label}")' if res.label else ""
        if res.status == STATUS_AVAILABLE:
            lines.append(f"Created {res.ref}{label} → {res.path}")
        else:
            lines.append(
                f"Figure #{res.candidate_index}{label} {_safe_status(res.status)}: "
                f"{safe_result_reason(res.reason) or WITHHELD}"
            )
    state = outcome_state(results)
    if state == "pending":
        lines.insert(0, HEADER_PENDING)
    elif state == "none":
        lines.insert(0, HEADER_NONE)
    return "\n".join(lines)


#: Reply headers shared by every surface (wrappers use the same constants).
HEADER_NONE = "No figure saved."
# Pending = THIS save's confirmation is uncertain (a local fsync or a Cloud ACK). Another
# caller may already have it confirmed, so never say "not available".
HEADER_PENDING = "Save confirmation pending."


def outcome_state(results: Sequence[CommitResult]) -> str:
    """``available`` if any figure is available, else ``pending`` if any publication is pending
    (not a failure, not "no save"), else ``none``."""
    statuses = {getattr(r, "status", None) for r in results}
    if STATUS_AVAILABLE in statuses:
        return "available"
    if STATUS_PENDING in statuses:
        return "pending"
    return "none"


#: Placeholder when a reason is not composed of the fixed vocabulary.
WITHHELD = "reason_withheld"


def _safe_status(value) -> str:
    return value if value in STATUSES else "failed"


def failure_summary(exc: BaseException) -> str:
    """Central, data-free description of a failed commit for tool replies (wrappers use this).

    Reads ``exc.details["results"]`` only if it is a list of dicts; every status/reason is
    checked against the fixed vocabulary (unknown → ``reason_withheld``). Otherwise falls back
    to :func:`safe_reason`. Never includes exception text, labels or refs from details.
    """
    details = getattr(exc, "details", None)
    results = details.get("results") if isinstance(details, dict) else None
    if (
        isinstance(exc, FigureError)
        and isinstance(results, list)
        and results
        and all(isinstance(r, dict) for r in results)
    ):
        return "; ".join(
            f"{_safe_status(r.get('status'))}: "
            f"{safe_result_reason(r.get('reason')) or WITHHELD}"
            for r in results
        )
    return safe_reason(exc)


__all__ = [
    "TrustedSaver",
    "summarize",
    "failure_summary",
    "outcome_state",
    "HEADER_NONE",
    "HEADER_PENDING",
    "WITHHELD",
    "FigureInvalidError",
]
