"""Typed errors for the managed figure lifecycle.

Each error carries a stable machine ``code`` (contract §6/§9 status vocabulary) so that
callers, Cloud adapters and tool replies can report outcomes without parsing messages.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class FigureError(Exception):
    """Base class for managed-figure errors."""

    code: str = "figure_error"

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details: Dict[str, Any] = dict(details or {})

    def to_dict(self) -> Dict[str, Any]:
        """Internal diagnostics only. Surfaced outcomes use :func:`safe_reason`, never messages."""
        return {"code": self.code, "message": str(self), "details": self.details}


class FigureInvalidError(FigureError):
    """Candidate bytes are not a valid, strict-JSON Plotly figure/config (or exceed caps)."""

    code = "figure_invalid"


class AttemptConflictError(FigureError):
    """Same attempt key, different semantic digest."""

    code = "attempt_conflict"


class FigureNotFoundError(FigureError):
    """Unknown figure/revision/source inside the caller's scope."""

    code = "figure_not_found"


class FigureDeletedError(FigureNotFoundError):
    """A revision in the caller's scope was DELETED: its tombstone exists (contract §6.2).

    Distinguishable from a revision that never existed, and never republished. It subclasses
    :class:`FigureNotFoundError`, so every existing not-found handler keeps working.
    """

    code = "figure_deleted"


class ParentDeletedError(FigureError):
    """Parent or input was deleted, or its lifecycle generation changed."""

    code = "parent_deleted"


class GenerationFencedError(FigureError):
    """Lifecycle generation captured at attempt start no longer matches at commit."""

    code = "generation_fenced"


class IntegrityError(FigureError):
    """Stored bytes do not match their recorded fingerprint (local detection)."""

    code = "fingerprint_mismatch"


class ManagedPathError(FigureError):
    """A generic writer attempted to modify the managed figure namespace."""

    code = "managed_path"


class AttemptContextError(FigureError):
    """A trusted commit inside a graph run was attempted without a tool_call_id."""

    code = "attempt_context_missing"


class StorageError(FigureError):
    """A store write failed (reported by code only; raw OS messages never surface)."""

    code = "storage_failed"


class DurabilityError(StorageError):
    """A required fsync (file or directory) failed: durability of a write is unconfirmed."""

    code = "durability_failed"


class StoreUnsupportedError(StorageError):
    """The local store is qualified for POSIX local filesystems only (contract §1)."""

    code = "storage_unsupported"


#: FIXED vocabulary of detail reasons the core may surface (a test keeps it in sync with the
#: source). Anything else is omitted — safety comes from membership, never from character shape.
DETAIL_REASONS = frozenset(
    {
        # payload / validation
        "not_utf8",
        "non_strict_json",
        "invalid_json",
        "type",
        "shape",
        "plotly_schema",
        "figure_too_large",
        "config_too_large",
        "source_too_large",
        "source_shape",
        "template_meta",
        "template_args_too_large",
        # execution request (byte ABI)
        "execution_request_too_large",
        "execution_request_schema",
        "execution_request_not_json",
        "execution_attempt_base",
        "code_blob_unexpected",
        "code_blob_required",
        "code_too_large",
        "code_blob_mismatch",
        "activity_fields_inconsistent",
        "activity_blob_required",
        "activity_too_large",
        "activity_blob_mismatch",
        "planned_candidates",
        "execution_plan_changed",
        "execution_record_mismatch",
        "code_blob_changed",
        "activity_blob_changed",
        "execution_not_found",
        # commit / claims
        "attempt_key_mismatch",
        "candidate_not_in_plan",
        "parent_unavailable",
        "inputs_not_normalized",
        "digest_mismatch",
        "attempt_digest_changed",
        "claimed_revision_mismatch",
        "existing_file_differs",
        "concurrent_writer_differs",
        # reads / integrity
        "revision_not_found",
        "figure_changed",
        "config_changed",
        "source_not_found",
        "source_changed",
        # durability / platform
        "dir_open_failed",
        "dir_fsync_failed",
        "dir_fsync_unsupported",
        "file_fsync_failed",
        "platform_not_posix",
        # renditions
        "rendition_inputs_mismatch",
        "rendition_format",
        "rendition_changed",
        "rendition_not_found",
        # custom-code submissions
        "figure_submission_invalid",
        # attempt snapshot (child-visible statuses)
        "not_requested",
        "total_cap",
        # plot manager
        "staged_figure_unknown",
        "no_figure_available",
    }
)

#: Stage prefixes the saver/renditions put in front of a reason.
REASON_PREFIXES = frozenset(
    {"execution_record", "storage", "prepare", "html_rendition"}
)
#: Reasons that are not FigureError codes.
STANDALONE_REASONS = frozenset({"unexpected_error", "attempt_failed"})


def _known_classes() -> Dict[type, str]:
    return {
        cls: cls.__dict__["code"]
        for cls in (
            FigureError,
            FigureInvalidError,
            AttemptConflictError,
            FigureNotFoundError,
            FigureDeletedError,
            ParentDeletedError,
            GenerationFencedError,
            IntegrityError,
            ManagedPathError,
            AttemptContextError,
            StorageError,
            DurabilityError,
            StoreUnsupportedError,
        )
    }


def error_codes() -> frozenset:
    """The fixed set of error codes (class-defined in this module only)."""
    return frozenset(_known_classes().values())


def _class_code(exc: BaseException) -> Optional[str]:
    """Code of the nearest class DEFINED HERE in ``exc``'s MRO.

    Instance attributes and subclass overrides of ``code`` are ignored.
    """
    known = _known_classes()
    for cls in type(exc).__mro__:
        if cls in known:
            return known[cls]
    return None


def safe_reason(exc: BaseException) -> str:
    """Stable, data-free reason for model/human-facing outcomes (no message/repr/args).

    ``<code>`` or ``<code>:<detail>`` where ``code`` comes from this module's classes and
    ``detail`` is a member of :data:`DETAIL_REASONS`; an unknown detail is omitted. Non-figure
    exceptions map to ``unexpected_error``.
    """
    code = _class_code(exc)
    if code is None:
        return "unexpected_error"
    details = getattr(exc, "details", None)
    detail = details.get("reason") if isinstance(details, dict) else None
    if isinstance(detail, str) and detail in DETAIL_REASONS:
        return f"{code}:{detail}"
    return code


def safe_result_reason(value: Any) -> Optional[str]:
    """Return ``value`` if it is a reason composed ONLY of fixed vocabulary, else ``None``.

    Accepted forms: ``[<prefix>:]<code>[:<detail>]`` and ``[<prefix>:]unexpected_error`` /
    ``attempt_failed``. Use it for any reason read back from results/details before surfacing.
    """
    if not isinstance(value, str) or len(value) > 200:
        return None
    parts = value.split(":")
    if parts and parts[0] in REASON_PREFIXES:
        parts = parts[1:]
    if len(parts) == 1 and parts[0] in STANDALONE_REASONS:
        return value
    if not parts or parts[0] not in error_codes():
        return None
    if len(parts) == 1 or (len(parts) == 2 and parts[1] in DETAIL_REASONS):
        return value
    return None
