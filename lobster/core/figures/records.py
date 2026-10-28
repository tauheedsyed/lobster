"""Record types: candidates, execution records, commit results, attempt digest (contract §3, §6).

Trusted (T) fields are computed by the saver/store; declared (D) fields are recorded as
supplied and are never proof.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

REVISION_SCHEMA = "lobster.figure_revision/1"
EXECUTION_SCHEMA = "lobster.figure_execution/1"
ATTEMPT_DIGEST_DOMAIN = "lobster.figure_attempt/1"
EXECUTION_REQUEST_SCHEMA = "lobster.figure_execution_request/2"
EXECUTION_PLAN_DOMAIN = "lobster.figure_execution_plan/2"
#: Request fields that describe ONE delivery's provenance observation; excluded from the plan
#: identity so a re-executed retry reuses the original immutable execution (contract §6.3).
EXECUTION_DELIVERY_FIELDS = ("activity_id", "activity_unavailable", "activity_record")

STATUS_AVAILABLE = "available"
STATUS_PENDING = "pending"
STATUS_REFUSED = "refused"
STATUS_FAILED = "failed"
STATUS_NOT_COMMITTED = "not_committed"
STATUSES = (
    STATUS_AVAILABLE,
    STATUS_PENDING,
    STATUS_REFUSED,
    STATUS_FAILED,
    STATUS_NOT_COMMITTED,
)

KIND_CREATED = "created"
KIND_EDITED = "edited"
KIND_REGENERATED = "regenerated"


@dataclass(frozen=True)
class InputEdge:
    """A dataset the figure was drawn from (contract §5.2)."""

    name: str
    instance_id: Optional[str]
    origin: str  # store_event | provider_object_version | content_hash | declared
    verified: bool
    alias: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "alias": self.alias,
            "instance_id": self.instance_id,
            "origin": self.origin,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputEdge":
        """A stored edge. Only an actual ``True`` is verified: a non-bool flag (``1``,
        ``"false"``, ...) reads as unverified and is never promoted through
        inheritance."""
        return cls(
            name=data.get("name"),
            instance_id=data.get("instance_id"),
            origin=data.get("origin") or "declared",
            verified=data.get("verified") is True,
            alias=data.get("alias"),
        )


def merge_inherited_inputs(
    candidate_inputs: Sequence["InputEdge"], parent_inputs: Sequence[Dict[str, Any]]
) -> Tuple["InputEdge", ...]:
    """Authoritative inputs of an edit: candidate edges + ALL parent edges (contract §5.2).

    Keyed by ``(name, instance_id)``. A verified edge is never downgraded by a matching
    declared/unverified one (whichever side supplies it); the parent's verified edge wins over
    the candidate's. No dependency is dropped. Order: candidate order, then remaining parent
    edges in parent order. Idempotent: merging an already-merged tuple returns it unchanged.
    """
    merged: List[InputEdge] = []
    index: Dict[Tuple[Any, Any], int] = {}
    for edge in candidate_inputs:
        key = (edge.name, edge.instance_id)
        if key in index:
            if edge.verified and not merged[index[key]].verified:
                merged[index[key]] = edge
            continue
        index[key] = len(merged)
        merged.append(edge)
    for raw in parent_inputs or ():
        edge = InputEdge.from_dict(raw)
        key = (edge.name, edge.instance_id)
        if key in index:
            current = merged[index[key]]
            if edge.verified and (not current.verified or current != edge):
                merged[index[key]] = edge  # trusted parent edge is authoritative
            continue
        index[key] = len(merged)
        merged.append(edge)
    return tuple(merged)


@dataclass(frozen=True)
class SourcePayload:
    payload_bytes: bytes
    declared: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    """One figure handed to the trusted saver. Bytes are stored exactly as given."""

    attempt_key: str
    candidate_index: int
    figure_bytes: bytes
    config_bytes: bytes
    kind: str = KIND_CREATED
    actor: str = "unknown"
    label: Optional[str] = None
    request: Optional[str] = None
    resolved_intent: Optional[List[Dict[str, Any]]] = None
    parent_ref: Optional[str] = None
    inputs: tuple = ()
    source: Optional[SourcePayload] = None
    template: Optional[Dict[str, Any]] = None
    attempt_context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionInfo:
    """Once-per-attempt execution facts (contract §6.3)."""

    attempt_base: str
    tool: str
    agent: Optional[str]
    parameters: Dict[str, Any]
    persist: bool
    code_bytes: Optional[bytes]
    activity: Optional[Dict[str, Any]]
    engine_version: str
    plotly_version: str

    @property
    def activity_id(self) -> Optional[str]:
        if not self.activity:
            return None
        value = self.activity.get("id") or self.activity.get("activity_id")
        return str(value) if value is not None else None


@dataclass
class CommitResult:
    candidate_index: int
    status: str
    ref: Optional[str] = None
    path: Optional[str] = None
    reason: Optional[str] = None
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "status": self.status,
            "ref": self.ref,
            "path": self.path,
            "reason": self.reason,
            "label": self.label,
        }


def _sha(data: Optional[bytes]) -> Optional[str]:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def attempt_descriptor(
    candidate: Candidate,
    *,
    parent_figure_sha256: Optional[str],
    parent_config_sha256: Optional[str],
    code_bytes: Optional[bytes] = None,
    code_fingerprint: Optional[Dict[str, Any]] = None,
    tool: str,
    persist: bool,
    engine_version: str,
    plotly_version: str,
) -> Dict[str, Any]:
    """Semantic descriptor (contract §6.2; field names shared with Cloud).

    Excludes timestamps, assigned IDs, transport/worker/request IDs, per-delivery activity IDs
    and lifecycle/catalog generations. Payload bytes are hashed as stored, never re-serialized.
    """
    source = candidate.source
    parents: List[Dict[str, Any]] = []
    if candidate.parent_ref:
        fid, rid = candidate.parent_ref.split("@", 1)
        parents.append(
            {
                "ref": {"figure_id": fid, "revision_id": rid},
                "figure_sha256": parent_figure_sha256,
                "config_sha256": parent_config_sha256,
            }
        )
    declared = dict(source.declared) if source is not None else {}
    return {
        "schema": ATTEMPT_DIGEST_DOMAIN,
        "change_kind": candidate.kind,
        "actor": candidate.actor,
        "tool": tool,
        "parents": parents,
        "figure": {
            "sha256": _sha(candidate.figure_bytes),
            "bytes": len(candidate.figure_bytes),
        },
        "config": {
            "sha256": _sha(candidate.config_bytes),
            "bytes": len(candidate.config_bytes),
        },
        "source": (
            None
            if source is None
            else {
                "payload": {
                    "sha256": _sha(source.payload_bytes),
                    "bytes": len(source.payload_bytes),
                },
                "declared": declared,
            }
        ),
        "code": (
            {"sha256": _sha(code_bytes), "bytes": len(code_bytes)}
            if code_bytes is not None
            else (
                {
                    "sha256": code_fingerprint["sha256"],
                    "bytes": code_fingerprint["bytes"],
                }
                if code_fingerprint
                else None
            )
        ),
        "inputs": [edge.to_dict() for edge in candidate.inputs],
        "label": candidate.label,
        "request": candidate.request,
        "resolved_intent": candidate.resolved_intent,
        "template": candidate.template,
        "persist": persist,
        "engine_version": engine_version,
        "plotly_version": plotly_version,
        "regeneration": list(declared.get("regeneration") or []),
    }


def canonical_descriptor_bytes(descriptor: Dict[str, Any]) -> bytes:
    """Strict canonical JSON: sorted keys, compact separators, UTF-8 (ensure_ascii=False), no NaN."""
    return json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def attempt_digest(candidate: Candidate, **kwargs: Any) -> str:
    """sha256 of the canonical semantic descriptor (see :func:`attempt_descriptor`)."""
    return hashlib.sha256(
        canonical_descriptor_bytes(attempt_descriptor(candidate, **kwargs))
    ).hexdigest()


def json_blob_bytes(obj: Any) -> bytes:
    """Canonical bytes of a JSON-NATIVE value (execution request): sorted keys, compact,
    UTF-8 (ensure_ascii=False), no NaN. Non-JSON values are REFUSED (TypeError), never repr'd:
    producers normalize explicitly before building a request.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def activity_blob_bytes(activity: Dict[str, Any]) -> bytes:
    """Producer-boundary serialization of the exact ``log_tool_usage`` dict.

    The provenance dict may hold datetimes/paths, so ONLY here non-JSON leaves are stringified;
    once produced, these bytes are fixed and travel/verify as exact bytes.
    """
    return json.dumps(
        activity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def execution_request(
    info: "ExecutionInfo",
    planned: Sequence[Tuple[int, str]],
    *,
    activity_bytes: Optional[bytes],
    candidate_count: Optional[int] = None,
) -> Dict[str, Any]:
    """The ``record_bytes`` of ``FigureStore.commit_execution`` (contract §9 byte ABI).

    Carries no execution ID, path, scope, timestamp or worker/request ID: the store allocates
    the ID and derives scope from its own authenticated context. Field names match Cloud's
    ``verify_execution_blobs`` (code_id, activity_id, activity_unavailable, activity_record).
    """
    code = info.code_bytes
    return {
        "schema": EXECUTION_REQUEST_SCHEMA,
        "attempt_base": info.attempt_base,
        "tool": info.tool,
        "agent": info.agent,
        "parameters": info.parameters,
        "persist": info.persist,
        "code_id": f"sha256-{_sha(code)}" if code is not None else None,
        "code": (
            {"sha256": _sha(code), "bytes": len(code)} if code is not None else None
        ),
        "activity_id": info.activity_id if activity_bytes is not None else None,
        "activity_unavailable": (
            None if activity_bytes is not None else "provenance_disabled"
        ),
        "activity_record": (
            {"sha256": _sha(activity_bytes), "bytes": len(activity_bytes)}
            if activity_bytes is not None
            else None
        ),
        # All candidates of the attempt are indexed 0..candidate_count-1; planned_candidates is the
        # eligible subset that passed validation (holes = refused before the execution record).
        "candidate_count": (
            int(candidate_count)
            if candidate_count is not None
            else (max(int(i) for i, _ in planned) + 1 if planned else 0)
        ),
        "planned_candidates": [
            {"index": int(i), "attempt_digest": d} for i, d in sorted(planned)
        ],
        "engine_version": info.engine_version,
        "plotly_version": info.plotly_version,
    }


def execution_plan_digest(request: Dict[str, Any]) -> str:
    """Identity of the immutable recipe: the request minus per-delivery activity fields.

    Same ``attempt_base`` + same plan digest → reuse the original execution; any change of code,
    tool, parameters, persist, versions or the planned (index, digest) set → conflict.
    """
    plan = {k: v for k, v in request.items() if k not in EXECUTION_DELIVERY_FIELDS}
    plan["schema"] = EXECUTION_PLAN_DOMAIN
    return hashlib.sha256(canonical_descriptor_bytes(plan)).hexdigest()


# ---------------------------------------------------------------- child submissions (§6.1)

MAX_SUBMITTED_FIGURES = 100
MAX_CANDIDATE_META_BYTES = 64 * 1024
MAX_UNSUBMITTED_NAMES = 100


@dataclass(frozen=True)
class SubmittedFigure:
    """One figure the child submitted, as read (and bounded) by the trusted parent."""

    index: int
    figure_bytes: bytes
    label: Optional[str] = None
    request: Optional[str] = None
    resolved_intent: Optional[List[Dict[str, Any]]] = None
    parent: Optional[str] = None
    source: Optional[SourcePayload] = None
    #: Exact submitted config bytes; ``None`` = not given (an edit then keeps its parent's
    #: exact config, a new figure gets ``{}``; resolved by the tool, never by ``{}``).
    config_bytes: Optional[bytes] = None


def read_submitted_figures(
    exec_cache,
) -> Tuple[List[SubmittedFigure], List[str], Optional[str]]:
    """Read ``EXEC_CACHE/candidates`` written by the child (contract §6.1).

    Returns ``(figures, unsubmitted_variable_names, error_code)``. All-or-nothing: any
    structural violation (symlink, gap in indices, oversize, bad metadata) returns no figures and
    ``error_code="figure_submission_invalid"``. Bytes are returned exactly as written.
    """
    import os
    from pathlib import Path

    from lobster.core.figures.ids import FigureRef
    from lobster.core.figures.payload import (
        MAX_CONFIG_BYTES,
        MAX_FIGURE_BYTES,
        MAX_SOURCE_BYTES,
        strict_json_loads,
        validate_config_bytes,
    )
    from lobster.utils.ipc_read import RefusedRead, read_regular

    root = Path(exec_cache) / "candidates"
    if not root.exists():
        return [], [], None
    invalid = ([], [], "figure_submission_invalid")
    if root.is_symlink() or not root.is_dir():
        return invalid

    unsubmitted: List[str] = []
    marker = root / "unsubmitted.json"
    try:  # a regular file only: never open a FIFO/device/symlink left by the child
        marker_bytes = read_regular(marker, MAX_CANDIDATE_META_BYTES)
    except RefusedRead:
        return invalid
    if marker_bytes is not None:
        try:
            names = strict_json_loads(marker_bytes, what="unsubmitted")
        except Exception:
            return invalid
        if not isinstance(names, list) or not all(
            isinstance(n, str) and n.isidentifier() and len(n) <= 64 for n in names
        ):
            return invalid
        unsubmitted = names[:MAX_UNSUBMITTED_NAMES]

    dirs = [p for p in root.iterdir() if p.name != "unsubmitted.json"]
    if len(dirs) > MAX_SUBMITTED_FIGURES:
        return invalid
    if any(p.is_symlink() or not p.is_dir() or not p.name.isdigit() for p in dirs):
        return invalid
    indices = sorted(int(p.name) for p in dirs)
    if indices != list(range(len(indices))):
        return invalid

    figures: List[SubmittedFigure] = []
    for index in indices:
        cdir = root / str(index)
        fig_path, meta_path, src_path, cfg_path = (
            cdir / "figure.json",
            cdir / "candidate.json",
            cdir / "source.json",
            cdir / "config.json",
        )
        for path in (fig_path, meta_path, src_path, cfg_path):
            if path.is_symlink():
                return invalid
        try:  # regular files only, capped (a FIFO/device/link left by the child is refused)
            figure_bytes = read_regular(fig_path, MAX_FIGURE_BYTES)
            meta_bytes = read_regular(meta_path, MAX_CANDIDATE_META_BYTES)
        except RefusedRead:
            return invalid
        if figure_bytes is None or meta_bytes is None:
            return invalid
        try:
            meta = strict_json_loads(meta_bytes, what="candidate")
        except Exception:
            return invalid
        if not isinstance(meta, dict) or meta.get("index") != index:
            return invalid
        label, request = meta.get("label"), meta.get("request")
        resolved, parent = meta.get("resolved_intent"), meta.get("parent")
        if not (label is None or (isinstance(label, str) and len(label) <= 500)):
            return invalid
        if not (request is None or (isinstance(request, str) and len(request) <= 4000)):
            return invalid
        if not (
            resolved is None
            or (
                isinstance(resolved, list)
                and all(isinstance(i, dict) for i in resolved)
            )
        ):
            return invalid
        if parent is not None:
            try:
                parent = str(FigureRef.parse(parent))
            except Exception:
                return invalid
        source = None
        if meta.get("has_source") is True:
            try:
                payload = read_regular(src_path, MAX_SOURCE_BYTES)
            except RefusedRead:
                return invalid
            if payload is None:
                return invalid
            try:
                decoded = strict_json_loads(payload, what="source")
            except Exception:
                return invalid
            if not isinstance(decoded, dict):
                return invalid
            declared = decoded.get(
                "declared", {}
            )  # absent = none; null is not a mapping
            if not isinstance(declared, dict):
                return invalid
            source = SourcePayload(payload, dict(declared))
        elif os.path.lexists(src_path):
            return invalid
        config_bytes = None
        if meta.get("has_config") is True:
            try:
                config_bytes = read_regular(cfg_path, MAX_CONFIG_BYTES)
                if config_bytes is None:
                    return invalid
                validate_config_bytes(config_bytes)  # strict JSON object
            except Exception:
                return invalid
        elif os.path.lexists(cfg_path):
            return invalid
        figures.append(
            SubmittedFigure(
                index=index,
                figure_bytes=figure_bytes,
                label=label,
                request=request,
                resolved_intent=resolved,
                parent=parent,
                source=source,
                config_bytes=config_bytes,
            )
        )
    return figures, unsubmitted, None
