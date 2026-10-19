"""Contract types for notebook replay checks: comparison policy, results, report.

A replay check runs an exported notebook and compares what it produced with a
reference (the "oracle") captured from the live session. This module holds the
parts every backend shares: the comparison policy, per-slot results, the
report, and the rules that decide which verdict a report may carry. It uses
only the standard library, so a checker can validate a report without
importing scientific packages. The comparator lives in ``replay_compare``.

Verdicts are never taken on trust: :func:`aggregate_verdict` recomputes the
verdict from the slot results, and :meth:`ReplayReport.validate` rejects a
report whose stated verdict, claim or state combination does not follow.

Paths address values in a nested tree with ``/`` separators, for example
``"pbmc/X"``, ``"pbmc/obs/n_genes"`` or ``"files/train.csv"``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = 1


class ContractError(ValueError):
    """A policy or report breaks the replay contract."""


class Verdict(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    REFUSED = "REFUSED"
    NOT_COMPARED = "NOT_COMPARED"


class SlotStatus(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    NOT_COMPARED = "NOT_COMPARED"


class Preflight(str, Enum):
    ADMITTED = "ADMITTED"
    REFUSED = "REFUSED"


class Execution(str, Enum):
    OK = "OK"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    NOT_RUN = "NOT_RUN"
    REFUSED_AT_RUNTIME = "REFUSED_AT_RUNTIME"


class Claim(str, Enum):
    """What a replay run set out to show (see the design's claim contract)."""

    EXEC = "C-EXEC"  # the notebook runs end to end; never a MATCH on its own
    HIST = "C-HIST"  # same inputs reproduce the recorded outputs
    RERUN = "C-RERUN"  # an override behaves like the live tool with that override
    CURATED = "C-CURATED"  # a curated export states what it omits


class RefusalOrigin(str, Enum):
    PRODUCT = "product"
    HARNESS = "harness"
    COMPARATOR = "comparator"


class VerificationStatus(str, Enum):
    NOT_RUN = "not_run"
    CHECKED = "checked"


class SparsePolicy(str, Enum):
    # Same matrix values; CSR vs CSC and explicit zeros do not matter.
    SEMANTIC = "semantic"
    # Same storage: format, explicit entries and index arrays must be equal.
    REPRESENTATION = "representation"


REQUESTED_MODES = frozenset({"successful", "all", "manual", "replay"})

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def split_path(path: str) -> Tuple[str, ...]:
    """Split and validate a slot path. Wildcards and relative parts are refused."""
    if not isinstance(path, str) or not path:
        raise ContractError(f"path must be a non-empty string: {path!r}")
    parts = tuple(path.split("/"))
    for part in parts:
        if part == "" or part.strip() != part:
            raise ContractError(f"malformed path: {path!r}")
        if part in (".", "..") or any(ch in part for ch in "*?[]"):
            raise ContractError(f"path parts must be literal names: {path!r}")
    return parts


def is_within(path: str, ancestor: str) -> bool:
    """True if ``path`` equals ``ancestor`` or lies below it (whole segments)."""
    return path == ancestor or path.startswith(ancestor + "/")


def _overlap(a: str, b: str) -> bool:
    return is_within(a, b) or is_within(b, a)


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


def _strict_keys(data: Any, allowed: Iterable[str], what: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ContractError(f"{what} must be an object")
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ContractError(f"{what} has unknown fields: {unknown}")
    return data


def _tolerance_value(value: Any, name: str, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} for {path!r} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ContractError(f"{name} for {path!r} must be finite and >= 0")
    return value


@dataclass(frozen=True)
class Exclusion:
    path: str
    rationale: str

    def to_dict(self) -> Dict[str, str]:
        return {"path": self.path, "rationale": self.rationale}


@dataclass(frozen=True)
class Tolerance:
    path: str
    rtol: float = 0.0
    atol: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "rtol": self.rtol, "atol": self.atol}


@dataclass(frozen=True)
class ComparisonPolicy:
    """What must be compared, what is excluded, and float tolerances.

    Rules, all enforced at construction:

    * ``required`` is non-empty; no path is repeated or nested in another.
    * each exclusion lies strictly below a required path, has a rationale, and
      does not overlap another exclusion;
    * each tolerance lies within a required path and outside every exclusion,
      with finite, non-negative ``rtol``/``atol``; the most specific one
      applies, and values without a tolerance are compared exactly;
    * every inventory is exact: keys, columns and files present on only one
      side are a mismatch.
    """

    required: Tuple[str, ...]
    excluded: Tuple[Exclusion, ...] = ()
    tolerances: Tuple[Tolerance, ...] = ()
    sparse: SparsePolicy = SparsePolicy.SEMANTIC
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"unsupported schema_version: {self.schema_version}")
        object.__setattr__(self, "required", tuple(self.required))
        object.__setattr__(self, "excluded", tuple(self.excluded))
        object.__setattr__(self, "tolerances", tuple(self.tolerances))
        try:
            object.__setattr__(self, "sparse", SparsePolicy(self.sparse))
        except ValueError:
            raise ContractError(f"unknown sparse policy: {self.sparse!r}") from None

        if not self.required:
            raise ContractError("policy must require at least one path")
        for path in self.required:
            split_path(path)
        for i, path in enumerate(self.required):
            for other in self.required[i + 1 :]:
                if _overlap(path, other):
                    raise ContractError(
                        f"required paths repeat or nest: {path!r}, {other!r}"
                    )

        for i, rule in enumerate(self.excluded):
            if not isinstance(rule, Exclusion):
                raise ContractError("exclusions must be Exclusion objects")
            split_path(rule.path)
            if not isinstance(rule.rationale, str) or not rule.rationale.strip():
                raise ContractError(f"exclusion {rule.path!r} needs a rationale")
            if not any(
                is_within(rule.path, req) and rule.path != req for req in self.required
            ):
                raise ContractError(
                    f"exclusion {rule.path!r} must lie strictly below a required path"
                )
            for other in self.excluded[i + 1 :]:
                if _overlap(rule.path, other.path):
                    raise ContractError(
                        f"exclusions repeat or nest: {rule.path!r}, {other.path!r}"
                    )

        seen = set()
        for rule in self.tolerances:
            if not isinstance(rule, Tolerance):
                raise ContractError("tolerances must be Tolerance objects")
            split_path(rule.path)
            if rule.path in seen:
                raise ContractError(f"duplicate tolerance for {rule.path!r}")
            seen.add(rule.path)
            if not any(is_within(rule.path, req) for req in self.required):
                raise ContractError(
                    f"tolerance {rule.path!r} lies outside the required paths"
                )
            if any(is_within(rule.path, ex.path) for ex in self.excluded):
                raise ContractError(f"tolerance {rule.path!r} lies in an exclusion")
            _tolerance_value(rule.rtol, "rtol", rule.path)
            _tolerance_value(rule.atol, "atol", rule.path)

    @classmethod
    def from_dict(cls, data: Any) -> "ComparisonPolicy":
        data = _strict_keys(
            data,
            ("schema_version", "required", "excluded", "tolerances", "sparse"),
            "policy",
        )
        if "required" not in data:
            raise ContractError("policy needs 'required'")
        required = data["required"]
        if not isinstance(required, list) or not all(
            isinstance(p, str) for p in required
        ):
            raise ContractError("'required' must be a list of paths")
        excluded = []
        for item in _list(data.get("excluded", []), "excluded"):
            item = _strict_keys(item, ("path", "rationale"), "exclusion")
            if set(item) != {"path", "rationale"}:
                raise ContractError("exclusion needs 'path' and 'rationale'")
            excluded.append(Exclusion(item["path"], item["rationale"]))
        tolerances = []
        for item in _list(data.get("tolerances", []), "tolerances"):
            item = _strict_keys(item, ("path", "rtol", "atol"), "tolerance")
            if "path" not in item or not ({"rtol", "atol"} & set(item)):
                raise ContractError("tolerance needs 'path' and 'rtol' and/or 'atol'")
            path = item["path"]
            tolerances.append(
                Tolerance(
                    path,
                    _tolerance_value(item.get("rtol", 0.0), "rtol", path),
                    _tolerance_value(item.get("atol", 0.0), "atol", path),
                )
            )
        version = data.get("schema_version", SCHEMA_VERSION)
        if isinstance(version, bool) or not isinstance(version, int):
            raise ContractError("schema_version must be an integer")
        return cls(
            required=tuple(required),
            excluded=tuple(excluded),
            tolerances=tuple(tolerances),
            sparse=data.get("sparse", SparsePolicy.SEMANTIC.value),
            schema_version=version,
        )

    @classmethod
    def from_json(cls, text: str) -> "ComparisonPolicy":
        return cls.from_dict(_loads_strict(text))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "required": list(self.required),
            "excluded": [e.to_dict() for e in self.excluded],
            "tolerances": [t.to_dict() for t in self.tolerances],
            "sparse": self.sparse.value,
        }

    def digest(self) -> str:
        """SHA-256 of the canonical JSON form; a report binds to this."""
        return canonical_sha256(self.to_dict())

    def exclusion_for(self, path: str) -> Optional[Exclusion]:
        for rule in self.excluded:
            if is_within(path, rule.path):
                return rule
        return None

    def tolerance_for(self, path: str) -> Tuple[float, float]:
        """(rtol, atol) of the most specific tolerance covering ``path``."""
        best: Optional[Tolerance] = None
        for rule in self.tolerances:
            if is_within(path, rule.path) and (
                best is None or len(rule.path) > len(best.path)
            ):
                best = rule
        return (best.rtol, best.atol) if best else (0.0, 0.0)


def _list(value: Any, what: str) -> list:
    if not isinstance(value, list):
        raise ContractError(f"'{what}' must be a list")
    return value


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(name):
    raise ContractError(f"non-standard JSON number: {name}")


def _loads_strict(text: str) -> Any:
    """json.loads that refuses duplicate keys and NaN/Infinity."""
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )


def canonical_sha256(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Results and verdicts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotResult:
    """Outcome for one required path.

    ``compared`` counts the values actually compared. A slot that matched
    with ``compared == 0`` (an empty slot) is structural, not evidence.
    """

    path: str
    status: SlotStatus
    compared: int = 0
    rtol: float = 0.0
    atol: float = 0.0
    diffs: Tuple[str, ...] = ()
    not_compared: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        split_path(self.path)
        object.__setattr__(self, "status", SlotStatus(self.status))
        object.__setattr__(self, "diffs", tuple(self.diffs))
        object.__setattr__(self, "not_compared", tuple(self.not_compared))
        if isinstance(self.compared, bool) or not isinstance(self.compared, int):
            raise ContractError("compared must be an integer")
        if self.compared < 0:
            raise ContractError("compared must be >= 0")
        if self.status is SlotStatus.MISMATCH and not self.diffs:
            raise ContractError(f"MISMATCH slot {self.path!r} must list a difference")
        if self.status is SlotStatus.MATCH and (self.diffs or self.not_compared):
            raise ContractError(f"MATCH slot {self.path!r} cannot carry findings")
        if self.status is SlotStatus.NOT_COMPARED and not (
            self.not_compared or self.diffs
        ):
            raise ContractError(f"NOT_COMPARED slot {self.path!r} must say why")

    @property
    def substantive(self) -> bool:
        return self.compared > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "status": self.status.value,
            "compared": self.compared,
            "rtol": self.rtol,
            "atol": self.atol,
            "diffs": list(self.diffs),
            "not_compared": list(self.not_compared),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "SlotResult":
        data = _strict_keys(
            data,
            ("path", "status", "compared", "rtol", "atol", "diffs", "not_compared"),
            "slot",
        )
        if not {"path", "status", "compared"} <= set(data):
            raise ContractError("slot needs 'path', 'status' and 'compared'")
        try:
            status = SlotStatus(data["status"])
        except ValueError:
            raise ContractError(f"unknown slot status: {data['status']!r}") from None
        path = data["path"]
        return cls(
            path=path,
            status=status,
            compared=data["compared"],
            rtol=_tolerance_value(data.get("rtol", 0.0), "rtol", path),
            atol=_tolerance_value(data.get("atol", 0.0), "atol", path),
            diffs=tuple(_str_list(data.get("diffs", []), "diffs")),
            not_compared=tuple(_str_list(data.get("not_compared", []), "not_compared")),
        )


def _str_list(value: Any, what: str) -> list:
    value = _list(value, what)
    if not all(isinstance(v, str) for v in value):
        raise ContractError(f"'{what}' must be a list of strings")
    return value


def aggregate_verdict(slots: Sequence[SlotResult]) -> Verdict:
    """Verdict of a completed comparison.

    MATCH needs at least one slot, every slot MATCH, and at least one slot
    with compared values: an empty comparison is never a MATCH.
    """
    if not slots:
        return Verdict.NOT_COMPARED
    statuses = {slot.status for slot in slots}
    if SlotStatus.MISMATCH in statuses:
        return Verdict.MISMATCH
    if SlotStatus.NOT_COMPARED in statuses:
        return Verdict.NOT_COMPARED
    if not any(slot.substantive for slot in slots):
        return Verdict.NOT_COMPARED
    return Verdict.MATCH


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

BINDING_KEYS = (
    "notebook_sha256",
    "recipe_sha256",
    "root_sha256",
    "policy_sha256",
    "params_sha256",
    "source_revision",
    "source_dirty",
    "env_receipt_sha256",
)


def validate_bindings(bindings: Any) -> Dict[str, Any]:
    """Check the shape of the bindings block (values are re-derived elsewhere)."""
    bindings = _strict_keys(bindings, BINDING_KEYS, "bindings")
    missing = [key for key in BINDING_KEYS if key not in bindings]
    if missing:
        raise ContractError(f"bindings missing: {missing}")
    for key in BINDING_KEYS:
        value = bindings[key]
        if key == "root_sha256":
            if not isinstance(value, Mapping) or not value:
                raise ContractError("root_sha256 must map each root to its SHA-256")
            for name, digest in value.items():
                if not isinstance(name, str) or not name:
                    raise ContractError("root names must be non-empty strings")
                if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                    raise ContractError(f"root_sha256[{name!r}] is not a SHA-256")
        elif key == "source_dirty":
            if not isinstance(value, bool):
                raise ContractError("source_dirty must be true or false")
        elif key == "source_revision":
            if not isinstance(value, str) or not _REVISION.fullmatch(value):
                raise ContractError("source_revision must be a full commit SHA")
        elif not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ContractError(f"{key} is not a SHA-256")
    return dict(bindings)


def binding_differences(stated: Mapping[str, Any], recomputed: Mapping[str, Any]):
    """Binding keys whose stated value differs from the recomputed one.

    A checker recomputes every binding from the files themselves; a report's
    own values are never authoritative.
    """
    stated = validate_bindings(stated)
    recomputed = validate_bindings(recomputed)
    return [key for key in BINDING_KEYS if stated[key] != recomputed[key]]


@dataclass(frozen=True)
class Refusal:
    reason: str
    step: Optional[str]
    origin: RefusalOrigin

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ContractError("refusal needs a reason")
        if self.step is not None and not isinstance(self.step, str):
            raise ContractError("refusal step must be a string or null")
        try:
            object.__setattr__(self, "origin", RefusalOrigin(self.origin))
        except ValueError:
            raise ContractError(f"unknown refusal origin: {self.origin!r}") from None

    def to_dict(self) -> Dict[str, Any]:
        return {"reason": self.reason, "step": self.step, "origin": self.origin.value}

    @classmethod
    def from_dict(cls, data: Any) -> "Refusal":
        data = _strict_keys(data, ("reason", "step", "origin"), "refusal")
        if set(data) != {"reason", "step", "origin"}:
            raise ContractError("refusal needs 'reason', 'step' and 'origin'")
        return cls(data["reason"], data["step"], data["origin"])


def _enum(kind, value, what):
    try:
        return kind(value)
    except ValueError:
        raise ContractError(f"unknown {what}: {value!r}") from None


@dataclass(frozen=True)
class ReplayReport:
    """The versioned report one replay run produces (``report.json``)."""

    backend: str
    scenario_id: str
    requested_mode: str
    claim_requested: Claim
    claim_achieved: Optional[Claim]
    verification_status: VerificationStatus
    bindings: Mapping[str, Any]
    preflight: Preflight
    execution: Execution
    verdict: Verdict
    slots: Tuple[SlotResult, ...] = ()
    excluded: Tuple[Exclusion, ...] = ()
    refusal: Optional[Refusal] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", tuple(self.slots))
        object.__setattr__(self, "excluded", tuple(self.excluded))
        self.validate()

    @property
    def coverage_complete(self) -> bool:
        """Every required slot was compared and matched (derived, never stored)."""
        return bool(self.slots) and all(
            slot.status is SlotStatus.MATCH for slot in self.slots
        )

    def validate(self, policy: Optional[ComparisonPolicy] = None) -> None:
        """Raise ContractError unless the report is internally consistent.

        With ``policy``, also require that the slots are exactly the policy's
        required paths, the exclusions are the policy's, and the report is
        bound to that policy's digest.
        """
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"unsupported schema_version: {self.schema_version}")
        for name in ("backend", "scenario_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"{name} must be a non-empty string")
        if self.requested_mode not in REQUESTED_MODES:
            raise ContractError(f"unknown requested_mode: {self.requested_mode!r}")
        object.__setattr__(
            self, "claim_requested", _enum(Claim, self.claim_requested, "claim")
        )
        if self.claim_achieved is not None:
            object.__setattr__(
                self, "claim_achieved", _enum(Claim, self.claim_achieved, "claim")
            )
        object.__setattr__(
            self,
            "verification_status",
            _enum(VerificationStatus, self.verification_status, "verification"),
        )
        object.__setattr__(
            self, "preflight", _enum(Preflight, self.preflight, "preflight")
        )
        object.__setattr__(
            self, "execution", _enum(Execution, self.execution, "execution")
        )
        object.__setattr__(self, "verdict", _enum(Verdict, self.verdict, "verdict"))
        object.__setattr__(self, "bindings", validate_bindings(self.bindings))
        paths = [slot.path for slot in self.slots]
        if len(paths) != len(set(paths)):
            raise ContractError("slot paths repeat")
        for rule in self.excluded:
            if not isinstance(rule, Exclusion):
                raise ContractError("exclusions must be Exclusion objects")
            split_path(rule.path)
            if not isinstance(rule.rationale, str) or not rule.rationale.strip():
                raise ContractError(f"exclusion {rule.path!r} needs a rationale")
        if self.refusal is not None and not isinstance(self.refusal, Refusal):
            raise ContractError("refusal must be a Refusal")

        self._validate_state()
        self._validate_claim()
        compared = self.verdict in (Verdict.MATCH, Verdict.MISMATCH)
        if compared != (self.verification_status is VerificationStatus.CHECKED):
            raise ContractError(
                "verification_status is 'checked' exactly when outputs were compared"
            )

        if policy is not None:
            if set(paths) != set(policy.required) and self.slots:
                raise ContractError("slots differ from the policy's required paths")
            if set(self.excluded) != set(policy.excluded) and self.slots:
                raise ContractError("exclusions differ from the policy's")
            if self.bindings["policy_sha256"] != policy.digest():
                raise ContractError("report is not bound to this policy")

    def _validate_state(self) -> None:
        pre, run, verdict = self.preflight, self.execution, self.verdict
        if (self.refusal is not None) != (verdict is Verdict.REFUSED):
            raise ContractError("a refusal and a REFUSED verdict go together")
        if pre is Preflight.REFUSED:
            if verdict is not Verdict.REFUSED:
                raise ContractError("a refused preflight has a REFUSED verdict")
            if run is not Execution.NOT_RUN or self.slots:
                raise ContractError("a refused preflight runs and compares nothing")
            return
        if run is Execution.NOT_RUN:
            raise ContractError("an admitted run must have executed")
        if run is Execution.REFUSED_AT_RUNTIME:
            if self.refusal is None or self.refusal.origin is not RefusalOrigin.PRODUCT:
                raise ContractError(
                    "REFUSED_AT_RUNTIME needs a recognized product refusal"
                )
            return
        if verdict is Verdict.REFUSED:
            raise ContractError("only a refused preflight or product refusal REFUSES")
        if run in (Execution.ERROR, Execution.TIMEOUT):
            if verdict is not Verdict.NOT_COMPARED:
                raise ContractError(f"{run.value} execution is NOT_COMPARED")
            return
        expected = aggregate_verdict(self.slots)
        if verdict is not expected:
            raise ContractError(
                f"verdict {verdict.value} does not follow from the slots "
                f"({expected.value})"
            )

    def _validate_claim(self) -> None:
        achieved = self.claim_achieved
        if achieved is None:
            return
        if achieved is not self.claim_requested and achieved is not Claim.EXEC:
            raise ContractError("a run can only achieve the claim it requested")
        if achieved is Claim.EXEC:
            if self.execution is not Execution.OK:
                raise ContractError("C-EXEC needs a completed execution")
        elif achieved in (Claim.HIST, Claim.RERUN):
            if self.verdict is not Verdict.MATCH:
                raise ContractError(f"{achieved.value} needs a MATCH verdict")
        elif achieved is Claim.CURATED and self.execution is not Execution.OK:
            raise ContractError("C-CURATED needs a completed execution")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "scenario_id": self.scenario_id,
            "requested_mode": self.requested_mode,
            "claim_requested": self.claim_requested.value,
            "claim_achieved": (
                self.claim_achieved.value if self.claim_achieved else None
            ),
            "verification_status": self.verification_status.value,
            "bindings": dict(self.bindings),
            "preflight": self.preflight.value,
            "execution": self.execution.value,
            "slots": [slot.to_dict() for slot in self.slots],
            "excluded": [rule.to_dict() for rule in self.excluded],
            "refusal": self.refusal.to_dict() if self.refusal else None,
            "verdict": self.verdict.value,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Any) -> "ReplayReport":
        fields_ = (
            "schema_version",
            "backend",
            "scenario_id",
            "requested_mode",
            "claim_requested",
            "claim_achieved",
            "verification_status",
            "bindings",
            "preflight",
            "execution",
            "slots",
            "excluded",
            "refusal",
            "verdict",
        )
        data = _strict_keys(data, fields_, "report")
        missing = [name for name in fields_ if name not in data]
        if missing:
            raise ContractError(f"report missing: {missing}")
        excluded = []
        for item in _list(data["excluded"], "excluded"):
            item = _strict_keys(item, ("path", "rationale"), "exclusion")
            excluded.append(Exclusion(item.get("path"), item.get("rationale")))
        refusal = data["refusal"]
        return cls(
            backend=data["backend"],
            scenario_id=data["scenario_id"],
            requested_mode=data["requested_mode"],
            claim_requested=data["claim_requested"],
            claim_achieved=data["claim_achieved"],
            verification_status=data["verification_status"],
            bindings=data["bindings"],
            preflight=data["preflight"],
            execution=data["execution"],
            verdict=data["verdict"],
            slots=tuple(SlotResult.from_dict(s) for s in _list(data["slots"], "slots")),
            excluded=tuple(excluded),
            refusal=Refusal.from_dict(refusal) if refusal is not None else None,
            schema_version=data["schema_version"],
        )

    @classmethod
    def from_json(cls, text: str) -> "ReplayReport":
        return cls.from_dict(_loads_strict(text))


__all__ = [
    "SCHEMA_VERSION",
    "BINDING_KEYS",
    "REQUESTED_MODES",
    "Claim",
    "ComparisonPolicy",
    "ContractError",
    "Exclusion",
    "Execution",
    "Preflight",
    "Refusal",
    "RefusalOrigin",
    "ReplayReport",
    "SlotResult",
    "SlotStatus",
    "SparsePolicy",
    "Tolerance",
    "Verdict",
    "VerificationStatus",
    "aggregate_verdict",
    "binding_differences",
    "canonical_sha256",
    "is_within",
    "split_path",
    "validate_bindings",
]
