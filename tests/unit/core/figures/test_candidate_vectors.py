"""Candidate-descriptor vectors + outcome examples (fixtures/candidate_vectors.json).

The fixture is generated from real engine commits (fixtures/gen_candidate_vectors.py). These
tests (1) recompute every digest from the vector's own fields only, independent of the
generator, and (2) check the committed file is exactly what the current core generates.
Regenerate with:  LOBSTER_REGEN_CANDIDATE_VECTORS=1 <guarded pytest> this file
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from lobster.core.figures.records import (
    Candidate,
    InputEdge,
    SourcePayload,
    attempt_digest,
    canonical_descriptor_bytes,
)

_spec = importlib.util.spec_from_file_location(
    "gen_candidate_vectors",
    Path(__file__).parent / "fixtures" / "gen_candidate_vectors.py",
)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

if os.environ.get("LOBSTER_REGEN_CANDIDATE_VECTORS") == "1":  # explicit, never implicit
    gen.OUT.write_bytes(gen.render())

DATA = json.loads(gen.OUT.read_bytes())


def _candidate(v):
    c = v["candidate"]
    src = c["source"]
    return Candidate(
        attempt_key=c["attempt_key"],
        candidate_index=c["candidate_index"],
        figure_bytes=c["figure_utf8"].encode("utf-8"),
        config_bytes=c["config_utf8"].encode("utf-8"),
        kind=c["kind"],
        actor=c["actor"],
        label=c["label"],
        request=c["request"],
        resolved_intent=c["resolved_intent"],
        parent_ref=c["parent_ref"],
        inputs=tuple(InputEdge.from_dict(e) for e in v["authoritative"]["inputs"]),
        source=(
            None
            if src is None
            else SourcePayload(src["payload_utf8"].encode("utf-8"), src["declared"])
        ),
        template=v["authoritative"]["template"],
        attempt_context=c["attempt_context"],
    )


def test_fixture_covers_created_edited_and_declared_unverified_inputs():
    names = [v["name"] for v in DATA["vectors"]]
    assert names == [
        "created_with_source_verified_and_declared_inputs",
        "edited_inherits_parent_inputs",
        "created_minimal_nulls",
    ]
    declared = DATA["vectors"][0]["authoritative"]["inputs"][1]
    assert declared == {
        "name": "supplementary_table",
        "alias": None,
        "instance_id": None,
        "origin": "declared",
        "verified": False,
    }
    edited = DATA["vectors"][1]
    assert edited["candidate"]["inputs"] == []  # inherited, not submitted
    assert (
        edited["authoritative"]["inputs"]
        == DATA["vectors"][0]["authoritative"]["inputs"]
    )
    # the 14 Candidate fields; figure/config bytes travel as exact UTF-8 text + sha256
    assert set(DATA["vectors"][0]["candidate"]) == {
        "attempt_key",
        "candidate_index",
        "figure_utf8",
        "figure_sha256",
        "config_utf8",
        "config_sha256",
        "kind",
        "actor",
        "label",
        "request",
        "resolved_intent",
        "parent_ref",
        "inputs",
        "source",
        "template",
        "attempt_context",
    }
    parent = edited["descriptor"]["parents"][0]
    first = DATA["vectors"][0]["candidate"]
    assert (parent["figure_sha256"], parent["config_sha256"]) == (
        first["figure_sha256"],
        first["config_sha256"],
    )  # the base figure + config identities of the edit


@pytest.mark.parametrize("v", DATA["vectors"], ids=lambda v: v["name"])
def test_vector_digest_recomputes_from_its_own_fields(v):
    canonical = v["canonical_descriptor_utf8"].encode("utf-8")
    assert canonical_descriptor_bytes(v["descriptor"]) == canonical
    assert hashlib.sha256(canonical).hexdigest() == v["attempt_digest"]
    assert v["engine_recorded_attempt_digest"] == v["attempt_digest"]
    args = dict(v["digest_args"])
    code = args.pop("code_utf8")
    assert args.pop("code_sha256") == (
        hashlib.sha256(code.encode()).hexdigest() if code is not None else None
    )
    assert (
        attempt_digest(
            _candidate(v),
            code_bytes=None if code is None else code.encode("utf-8"),
            **args,
        )
        == v["attempt_digest"]
    )
    c = v["candidate"]
    assert hashlib.sha256(c["figure_utf8"].encode()).hexdigest() == c["figure_sha256"]


def test_outcome_examples_show_the_agreed_pending_projection():
    o = DATA["outcome_examples"]
    assert DATA["header_pending"] == "Save confirmation pending."
    [p] = o["pending_commit"]
    assert (p["status"], p["ref"], p["path"]) == ("pending", None, None)
    assert o["pending_summary_header"].startswith("Save confirmation pending.")
    pending = [
        i for i in o["list_items_while_pending"] if i["availability"] == "pending"
    ]
    assert [i["availability_reason"] for i in pending] == [
        "durability_failed:dir_fsync_failed"
    ]
    assert o["read_record_pending"]["code"] == "durability_failed"
    assert o["snapshot_status_pending"]["status"] == "pending"
    assert o["execution_outcomes_pending"][0]["ref"] is None
    [retry] = o["pending_retry_after_recovery"]
    assert retry["status"] == "available"
    assert all(i["availability"] == "available" for i in o["list_items_after_recovery"])
    available = [
        i for i in o["list_items_while_pending"] if i["availability"] == "available"
    ]
    assert len(available) == 3 and all(
        "availability_reason" not in i for i in available
    )


def test_committed_fixture_is_exactly_what_the_current_core_generates():
    assert gen.render() == gen.OUT.read_bytes()
