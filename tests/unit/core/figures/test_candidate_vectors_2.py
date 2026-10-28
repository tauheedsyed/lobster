"""Multi-input candidate vectors (fixtures/candidate_vectors_2.json, ``engine-candidate-vectors/2``).

A NEW immutable multi-input supplement. The /1 vectors and the execution-request /2 vectors stay
byte-stable (checked below). These tests (1) recompute every digest from the vector's own
fields only, independent of the generator, (2) show each edge mutation changes the digest and
distinct ids never collapse, and (3) check the committed file is exactly what the current core
generates (this runs the real capture child).
Regenerate with:  LOBSTER_REGEN_CANDIDATE_VECTORS_2=1 <guarded pytest> this file
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
    attempt_digest,
    canonical_descriptor_bytes,
)

FIXTURES = Path(__file__).parent / "fixtures"
_spec = importlib.util.spec_from_file_location(
    "gen_candidate_vectors_2", FIXTURES / "gen_candidate_vectors_2.py"
)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

# explicit, never implicit
if os.environ.get("LOBSTER_REGEN_CANDIDATE_VECTORS_2") == "1":
    gen.OUT.write_bytes(gen.render())

DATA = json.loads(gen.OUT.read_bytes())

# Published earlier to Cloud; must never change (nor may the /2 request vectors).
V1_SHA256 = "1d8bdeed14eebc9c0276e0bc08ac3fedc4cbdef951d87a8d072ee4d5e0e07104"
EXEC2_SHA256 = "708bdabe7e972167514dac4ac1ddb1fcfa0715da5931298a2483adb4072cc08a"


def _edge(d):
    return (d["name"], d["instance_id"], d["origin"], d["verified"])


def _candidate(v, inputs=None):
    c = v["candidate"]
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
        inputs=tuple(
            InputEdge.from_dict(e)
            for e in (v["authoritative"]["inputs"] if inputs is None else inputs)
        ),
        source=None,
        template=v["authoritative"]["template"],
        attempt_context=c["attempt_context"],
    )


def _digest(v, inputs=None):
    args = dict(v["digest_args"])
    code = args.pop("code_utf8")
    args.pop("code_sha256")
    return attempt_digest(
        _candidate(v, inputs), code_bytes=code.encode("utf-8"), **args
    )


def test_earlier_fixtures_are_byte_stable():
    assert (
        hashlib.sha256((FIXTURES / "candidate_vectors.json").read_bytes()).hexdigest()
        == V1_SHA256
    )
    assert (
        hashlib.sha256(
            (FIXTURES / "execution_request_vectors.json").read_bytes()
        ).hexdigest()
        == EXEC2_SHA256
    )


def test_capture_gives_two_verified_edges_and_one_unknown_sorted_by_name():
    assert DATA["fixture_version"] == "engine-candidate-vectors/2"
    created, edited = DATA["vectors"]
    first = [_edge(e) for e in created["captured_input_edges"]]
    assert [e[0] for e in first] == ["alpha", "nolineage", "zeta"]
    assert first[1] == ("nolineage", None, "declared", False)
    verified = [e for e in first if e[3]]
    assert len(verified) == 2 and len({e[1] for e in verified}) == 2
    assert all(e[2] == "store_event" for e in verified)
    assert created["candidate"]["inputs"] == created["captured_input_edges"]
    assert created["authoritative"]["inputs"] == created["captured_input_edges"]


def test_an_edit_keeps_captured_edges_first_then_every_parent_edge():
    created, edited = DATA["vectors"]
    captured = [_edge(e) for e in edited["captured_input_edges"]]
    parent = [_edge(e) for e in created["authoritative"]["inputs"]]
    merged = [_edge(e) for e in edited["authoritative"]["inputs"]]
    assert captured[2][0] == parent[2][0] == "zeta" and captured[2][1] != parent[2][1]
    # candidate edges first; then the parent's remaining edges (same-name, distinct id kept)
    assert merged == captured + [parent[2]]
    p = edited["descriptor"]["parents"][0]
    assert p["figure_sha256"] == created["candidate"]["figure_sha256"]
    assert p["config_sha256"] == created["candidate"]["config_sha256"]


@pytest.mark.parametrize("v", DATA["vectors"], ids=lambda v: v["name"])
def test_vector_digest_recomputes_from_its_own_fields(v):
    canonical = v["canonical_descriptor_utf8"].encode("utf-8")
    assert canonical_descriptor_bytes(v["descriptor"]) == canonical
    assert hashlib.sha256(canonical).hexdigest() == v["attempt_digest"]
    assert v["engine_recorded_attempt_digest"] == v["attempt_digest"]
    assert _digest(v) == v["attempt_digest"]


def test_every_edge_mutation_changes_the_digest_and_ids_never_collapse():
    created = DATA["vectors"][0]
    m = DATA["digest_mutations"]
    assert m["base"]["attempt_digest"] == created["attempt_digest"]
    for name, variant in m.items():  # recomputed here, not trusted from the generator
        assert _digest(created, variant["inputs"]) == variant["attempt_digest"], name
    digests = [variant["attempt_digest"] for variant in m.values()]
    assert len(set(digests)) == len(digests)
    extra = [_edge(e) for e in m["same_name_distinct_instance_added"]["inputs"]]
    same_name = [e for e in extra if e[0] == extra[-1][0]]
    assert len(same_name) == 2 and same_name[0][1] != same_name[1][1]


def test_committed_fixture_is_exactly_what_the_current_core_generates():
    assert gen.render() == gen.OUT.read_bytes()
