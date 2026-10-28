"""Generate engine wire vectors for ``FigureStore.commit_execution`` (contract v0.9 §9.1).

Run from the repo root:  .venv/bin/python tests/unit/core/figures/fixtures/gen_execution_request_vectors.py
Writes ``execution_request_vectors.json`` next to this file. Positive vectors carry exact UTF-8
record/code/activity bytes, record sha256 and plan digest. Negative vectors are single mutations
of a positive one; ``expected`` is the code the LOCAL engine store returns (computed, not typed).
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path

from lobster.core.figures.errors import FigureError
from lobster.core.figures.records import (
    EXECUTION_PLAN_DOMAIN,
    EXECUTION_REQUEST_SCHEMA,
    ExecutionInfo,
    activity_blob_bytes,
    execution_plan_digest,
    execution_request,
    json_blob_bytes,
)
from lobster.core.figures.store import LocalFigureStore

OUT = Path(__file__).with_name("execution_request_vectors.json")


def _u(b):
    return None if b is None else b.decode("utf-8")


def _engine_outcome(record: bytes, code, activity) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            LocalFigureStore(tmp, session_id="s1").commit_execution(
                record, code, activity
            )
        except FigureError as exc:
            reason = exc.details.get("reason")
            return f"{exc.code}:{reason}" if reason else exc.code
    return "accepted"


def main() -> None:
    code = (
        "fig = make_volcano(adata)\n"
        "submit_figure(fig, label='Volcano — treated vs control')\n"
    ).encode("utf-8")
    activity = {
        "id": "act-1",
        "tool_name": "execute_custom_code",
        "agent": "proteomics_de_analysis_expert",
        "ir": {"operation": "custom_code"},
    }
    info = ExecutionInfo(
        "s1:call_1",
        "execute_custom_code",
        "proteomics_de_analysis_expert",
        {"modality_name": "prot_de"},
        False,
        code,
        activity,
        "1.2.0.dev0",
        "7.1.0",
    )
    act = activity_blob_bytes(activity)
    req = execution_request(
        info, [(2, "b" * 64), (0, "a" * 64)], activity_bytes=act, candidate_count=3
    )
    wrapper = ExecutionInfo(
        "s1:call_2",
        "create_umap_plot",
        "visualization_expert",
        {},
        True,
        None,
        None,
        "1.2.0.dev0",
        "7.1.0",
    )
    req2 = execution_request(
        wrapper, [(0, "c" * 64)], activity_bytes=None, candidate_count=1
    )

    positives = []
    for name, request, c, a in (
        ("custom_code_with_activity_and_hole", req, code, act),
        ("wrapper_provenance_disabled", req2, None, None),
    ):
        record = json_blob_bytes(request)
        positives.append(
            {
                "name": name,
                "record_utf8": _u(record),
                "code_utf8": _u(c),
                "activity_utf8": _u(a),
                "record_sha256": hashlib.sha256(record).hexdigest(),
                "plan_digest": execution_plan_digest(request),
                "expected": _engine_outcome(record, c, a),
            }
        )

    def mutate(name, fn, *, code_bytes=code, activity_bytes=act, base=req):
        r = copy.deepcopy(base)
        fn(r)
        record = json_blob_bytes(r)
        return {
            "name": name,
            "base": "custom_code_with_activity_and_hole",
            "record_utf8": _u(record),
            "code_utf8": _u(code_bytes),
            "activity_utf8": _u(activity_bytes),
            "expected": _engine_outcome(record, code_bytes, activity_bytes),
        }

    negatives = [
        mutate("code_sha_wrong", lambda r: r["code"].update(sha256="0" * 64)),
        mutate(
            "code_bytes_wrong", lambda r: r["code"].update(bytes=r["code"]["bytes"] + 1)
        ),
        mutate("code_id_wrong", lambda r: r.update(code_id="sha256-" + "0" * 64)),
        mutate("code_blob_missing", lambda r: None, code_bytes=None),
        mutate(
            "activity_sha_wrong", lambda r: r["activity_record"].update(sha256="0" * 64)
        ),
        mutate(
            "activity_bytes_wrong",
            lambda r: r["activity_record"].update(
                bytes=r["activity_record"]["bytes"] + 1
            ),
        ),
        mutate("activity_id_mismatch", lambda r: r.update(activity_id="act-other")),
        mutate("activity_blob_missing", lambda r: None, activity_bytes=None),
        mutate(
            "activity_unavailable_with_activity",
            lambda r: r.update(activity_unavailable="provenance_disabled"),
        ),
        mutate("candidate_count_missing", lambda r: r.pop("candidate_count")),
        mutate("planned_index_out_of_range", lambda r: r.update(candidate_count=2)),
        mutate(
            "planned_unsorted",
            lambda r: r.update(
                planned_candidates=list(reversed(r["planned_candidates"]))
            ),
        ),
        mutate(
            "planned_duplicate_index",
            lambda r: r.update(planned_candidates=[r["planned_candidates"][0]] * 2),
        ),
        mutate("planned_empty", lambda r: r.update(planned_candidates=[])),
        mutate(
            "schema_wrong",
            lambda r: r.update(schema="lobster.figure_execution_request/1"),
        ),
        mutate("attempt_base_malformed", lambda r: r.update(attempt_base="s1")),
    ]
    doc = {
        "schema": "lobster.figure_execution_request_vectors/2",
        "request_schema": EXECUTION_REQUEST_SCHEMA,
        "plan_domain": EXECUTION_PLAN_DOMAIN,
        "canonicalization": (
            "record: json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, "
            "allow_nan=False) of JSON-native values (non-JSON refused); activity: same plus "
            "default=str at the producer boundary only, then fixed bytes"
        ),
        "plan_digest": (
            "sha256(canonical(request minus activity_id/activity_unavailable/activity_record, "
            "with schema=plan_domain))"
        ),
        "notes": (
            "candidate_count = every candidate of the attempt (0..n-1); planned_candidates = "
            "eligible subset, sorted, unique, may have holes. 'expected' is the local engine "
            "store outcome (code:reason); Cloud may map codes but must reject every negative."
        ),
        "positives": positives,
        "negatives": negatives,
    }
    OUT.write_text(
        json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
