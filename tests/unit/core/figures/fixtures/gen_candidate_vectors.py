"""Generate candidate-descriptor vectors + outcome examples (contract v0.9 §6.2 / §9).

Run from the repo root:  .venv/bin/python tests/unit/core/figures/fixtures/gen_candidate_vectors.py
Writes ``candidate_vectors.json`` next to this file. The execution-request /2 pair
(``execution_request_vectors.json``) is NOT touched.

Every value is COMPUTED by the engine, never typed: each candidate is committed through the
real :class:`TrustedSaver` into a temporary :class:`LocalFigureStore`, and the vector records
the 14 candidate fields as submitted, the authoritative (normalized) inputs/template, the digest
arguments, the canonical descriptor bytes, its sha256, and the digest the store recorded in
``revision.json``. IDs are drawn from a fixed sequence so the file is reproducible; real IDs are
opaque and random. Each vector is self-contained (an edit carries its parent ref and hashes).

Outcome examples are real ``CommitResult``/``list()``/snapshot/outcome outputs, including the
PENDING projection (an injected directory-fsync failure after the marker is linked).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List

from lobster.core.figures import store as store_mod
from lobster.core.figures.errors import DurabilityError
from lobster.core.figures.records import (
    Candidate,
    ExecutionInfo,
    InputEdge,
    SourcePayload,
    attempt_descriptor,
    canonical_descriptor_bytes,
)
from lobster.core.figures.saver import HEADER_PENDING, TrustedSaver, summarize
from lobster.core.figures.snapshot import build_snapshot
from lobster.core.figures.store import LocalFigureStore

OUT = Path(__file__).with_name("candidate_vectors.json")
FIXTURE_VERSION = "engine-candidate-vectors/1"
ENGINE_VERSION = "1.2.0.dev0"
PLOTLY_VERSION = "7.1.0"

FIG_CREATED = (
    b'{"data":[{"type":"bar","x":["ctrl","treated"],"y":[3,5]}],'
    b'"layout":{"title":{"text":"Counts per group"}}}'
)
FIG_EDITED = (
    b'{"data":[{"type":"bar","x":["ctrl","treated"],"y":[3,5],'
    b'"marker":{"color":"orange"}}],"layout":{"title":{"text":"Counts per group"}}}'
)
FIG_MINIMAL = b'{"data":[{"type":"scatter","y":[1,2,3]}],"layout":{}}'
CONFIG = b'{"displaylogo":false}'
SOURCE = b'{"columns":["group","count"],"rows":[["ctrl",3],["treated",5]]}'
CODE = (
    b"fig = go.Figure(go.Bar(x=['ctrl','treated'], y=[3,5]))\n"
    b"submit_figure(fig, source=src, intent='counts per group', label='Counts')\n"
)


def _u(b):
    return None if b is None else b.decode("utf-8")


def _sha(b):
    return None if b is None else hashlib.sha256(b).hexdigest()


@contextmanager
def _fixed_ids() -> Iterator[None]:
    counter = itertools.count(1)
    real = store_mod.new_id

    def fixed(prefix: str) -> str:
        return f"{prefix}_vec{next(counter):07d}"

    store_mod.new_id = fixed
    try:
        yield
    finally:
        store_mod.new_id = real


def _info(base: str, tool: str, agent, code, persist=False) -> ExecutionInfo:
    return ExecutionInfo(
        base, tool, agent, {}, persist, code, None, ENGINE_VERSION, PLOTLY_VERSION
    )


def _candidate_fields(c: Candidate) -> Dict[str, Any]:
    """All 14 Candidate fields, bytes as exact UTF-8 text + sha256."""
    return {
        "attempt_key": c.attempt_key,
        "candidate_index": c.candidate_index,
        "figure_utf8": _u(c.figure_bytes),
        "figure_sha256": _sha(c.figure_bytes),
        "config_utf8": _u(c.config_bytes),
        "config_sha256": _sha(c.config_bytes),
        "kind": c.kind,
        "actor": c.actor,
        "label": c.label,
        "request": c.request,
        "resolved_intent": c.resolved_intent,
        "parent_ref": c.parent_ref,
        "inputs": [e.to_dict() for e in c.inputs],
        "source": (
            None
            if c.source is None
            else {
                "payload_utf8": _u(c.source.payload_bytes),
                "payload_sha256": _sha(c.source.payload_bytes),
                "declared": c.source.declared,
            }
        ),
        "template": c.template,
        "attempt_context": c.attempt_context,
    }


def _vector(name: str, store: LocalFigureStore, info: ExecutionInfo, cand: Candidate):
    [res] = TrustedSaver(store).commit_attempt(info, [cand], succeeded=True)
    assert res.status == "available", (name, res)
    record = store.read_record(res.ref)
    parent = record["parents"][0] if record.get("parents") else None
    parent_fig = parent_cfg = None
    if cand.parent_ref:
        prec = store.read_record(cand.parent_ref)
        parent_fig, parent_cfg = prec["figure"]["sha256"], prec["config"]["sha256"]
    # authoritative candidate = what the saver digested (inputs merged, template resolved)
    authoritative = Candidate(
        **{
            **cand.__dict__,
            "inputs": tuple(InputEdge.from_dict(e) for e in record["inputs"]),
            "template": record.get("template"),
        }
    )
    kwargs = {
        "parent_figure_sha256": parent_fig,
        "parent_config_sha256": parent_cfg,
        "code_bytes": info.code_bytes,
        "tool": info.tool,
        "persist": info.persist,
        "engine_version": info.engine_version,
        "plotly_version": info.plotly_version,
    }
    descriptor = attempt_descriptor(authoritative, **kwargs)
    canonical = canonical_descriptor_bytes(descriptor)
    digest = hashlib.sha256(canonical).hexdigest()
    recorded = record["execution"]["attempt_digest"]
    assert recorded == digest, (name, recorded, digest)
    return (
        {
            "name": name,
            "candidate": _candidate_fields(cand),
            "authoritative": {
                "inputs": record["inputs"],
                "template": record.get("template"),
            },
            "digest_args": {
                **{k: v for k, v in kwargs.items() if k != "code_bytes"},
                "code_utf8": _u(info.code_bytes),
                "code_sha256": _sha(info.code_bytes),
            },
            "descriptor": descriptor,
            "canonical_descriptor_utf8": _u(canonical),
            "attempt_digest": digest,
            "engine_recorded_attempt_digest": recorded,
            "outcome": res.to_dict(),
            "parent": parent,
        },
        res,
    )


class _FailMarkerFsync:
    """Fail the directory fsync that confirms a NEW revision marker (marker already linked).

    Revisions that existed before the fault keep confirming, so the example shows exactly one
    pending item next to available ones.
    """

    def __init__(self, persistent: bool, root: Path):
        self.real = store_mod._fsync_dir
        self.persistent = persistent
        self.existing = set(Path(root).glob("*/r_*"))
        self.failed = 0

    def __call__(self, path: Path) -> None:
        if (
            path.name.startswith("r_")
            and path not in self.existing
            and (path / "revision.json").exists()
            and (self.persistent or not self.failed)
        ):
            self.failed += 1
            raise DurabilityError("injected", details={"reason": "dir_fsync_failed"})
        self.real(path)


@contextmanager
def _fsync_fault(persistent: bool, root: Path) -> Iterator[None]:
    real = store_mod._fsync_dir
    store_mod._fsync_dir = _FailMarkerFsync(persistent, root)
    try:
        yield
    finally:
        store_mod._fsync_dir = real


def _project_item(item: Dict[str, Any]) -> Dict[str, Any]:
    keep = ("figure_id", "revision_id", "availability", "availability_reason")
    return {k: item[k] for k in keep if k in item}


def build() -> Dict[str, Any]:
    vectors: List[Dict[str, Any]] = []
    outcomes: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp, _fixed_ids():
        store = LocalFigureStore(tmp, session_id="s1")
        created = Candidate(
            attempt_key="s1:call_1:0",
            candidate_index=0,
            figure_bytes=FIG_CREATED,
            config_bytes=CONFIG,
            kind="created",
            actor="expert:proteomics_expert",
            label="Counts",
            request="counts per group",
            resolved_intent=None,
            parent_ref=None,
            inputs=(
                InputEdge("prot_de", "inst_store_event_1", "store_event", True),
                InputEdge("supplementary_table", None, "declared", False),
            ),
            source=SourcePayload(SOURCE, {"comparison": "treated_vs_ctrl"}),
            template=None,
        )
        v1, r1 = _vector(
            "created_with_source_verified_and_declared_inputs",
            store,
            _info("s1:call_1", "execute_custom_code", "proteomics_expert", CODE),
            created,
        )
        vectors.append(v1)
        edited = Candidate(
            attempt_key="s1:omni_1:0",
            candidate_index=0,
            figure_bytes=FIG_EDITED,
            config_bytes=CONFIG,
            kind="edited",
            actor="omni:editor",
            label="Counts (orange)",
            request="make the bars orange",
            resolved_intent=[{"operation": "style", "target": "marker.color"}],
            parent_ref=r1.ref,
            inputs=(),  # the saver adds ALL parent edges (contract §5.2)
        )
        v2, r2 = _vector(
            "edited_inherits_parent_inputs",
            store,
            _info("s1:omni_1", "save_revision", None, None, persist=True),
            edited,
        )
        vectors.append(v2)
        minimal = Candidate(
            attempt_key="s1:call_2:0",
            candidate_index=0,
            figure_bytes=FIG_MINIMAL,
            config_bytes=b"{}",
            kind="created",
            actor="expert:transcriptomics_expert",
        )
        v3, _ = _vector(
            "created_minimal_nulls",
            store,
            _info("s1:call_2", "create_umap_plot", "transcriptomics_expert", None),
            minimal,
        )
        vectors.append(v3)

        # --- outcome examples (real outputs) ---------------------------------------------
        saver = TrustedSaver(store)
        outcomes["attempt_failed"] = [
            r.to_dict()
            for r in saver.commit_attempt(
                _info("s1:call_3", "execute_custom_code", "x", CODE),
                [Candidate("s1:call_3:0", 0, FIG_MINIMAL, b"{}")],
                succeeded=False,
            )
        ]
        outcomes["invalid_figure"] = [
            r.to_dict()
            for r in saver.commit_attempt(
                _info("s1:call_4", "execute_custom_code", "x", CODE),
                [Candidate("s1:call_4:0", 0, b'{"data":', b"{}")],
                succeeded=True,
            )
        ]
        pend_info = _info("s1:call_5", "execute_custom_code", "x", CODE)
        pend_cand = Candidate("s1:call_5:0", 0, FIG_MINIMAL, b'{"responsive":true}')
        with _fsync_fault(persistent=True, root=store.root):
            pending = saver.commit_attempt(pend_info, [pend_cand], succeeded=True)
            fresh = LocalFigureStore(tmp, session_id="s1")
            outcomes["pending_commit"] = [r.to_dict() for r in pending]
            outcomes["pending_summary_header"] = summarize(pending).split("\n")[0]
            outcomes["list_items_while_pending"] = [
                _project_item(i) for i in fresh.list().items
            ]
            pending_item = [
                i for i in fresh.list().items if i.get("availability") == "pending"
            ][0]
            pending_ref = f"{pending_item['figure_id']}@{pending_item['revision_id']}"
            try:
                fresh.read_record(pending_ref)
                outcomes["read_record_pending"] = "no_error"
            except DurabilityError as exc:
                outcomes["read_record_pending"] = {
                    "error": type(exc).__name__,
                    "code": exc.code,
                    "reason": exc.details.get("reason"),
                }
            snap = build_snapshot(
                LocalFigureStore(tmp, session_id="s1"), "", Path(tmp) / "snap"
            )
            outcomes["snapshot_status_pending"] = snap["status"][pending_ref]
            execution_id = json.loads(
                (
                    Path(tmp)
                    / "plots"
                    / pending_ref.replace("@", "/")
                    / "revision.json"
                ).read_bytes()
            )["execution"]["execution_id"]
            outcomes["execution_outcomes_pending"] = LocalFigureStore(
                tmp, session_id="s1"
            ).execution_outcomes(execution_id)
        retry = saver.commit_attempt(pend_info, [pend_cand], succeeded=True)
        outcomes["pending_retry_after_recovery"] = [r.to_dict() for r in retry]
        outcomes["list_items_after_recovery"] = [
            _project_item(i)
            for i in LocalFigureStore(tmp, session_id="s1").list().items
        ]
    return {
        "fixture_version": FIXTURE_VERSION,
        "note": (
            "Generated by gen_candidate_vectors.py from real engine commits; IDs come from a "
            "fixed sequence (real IDs are opaque/random). Independent of the /2 "
            "execution-request vectors. WIP core, NOT accepted."
        ),
        "header_pending": HEADER_PENDING,
        "vectors": vectors,
        "outcome_examples": outcomes,
    }


def render() -> bytes:
    return (
        json.dumps(build(), indent=1, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
        + b"\n"
    )


def main() -> None:
    OUT.write_bytes(render())
    print(f"wrote {OUT} sha256={hashlib.sha256(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
