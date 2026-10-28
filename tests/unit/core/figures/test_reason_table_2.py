"""``engine-reason-table/2`` is a FROZEN, B-consumed table (data only, no runtime change).

reasons/3 (deleted refs) changed ``errors.py``, so /2 can no longer be rebuilt from the live source.
It stays byte-identical, pinned to the reasons/2 source it was generated from, and stays
compatible: every /2 token is still live (nothing removed or renamed). The live binding moved
to ``test_reason_table_3.py``.
"""

import hashlib
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
TABLE = FIXTURES / "reason_table_2.json"
TABLE_2_SHA256 = "ebb14aab17e907a682df3a4246ea2997874bd8b0b70ae38915dcc62ca81c114c"


@pytest.fixture(scope="module")
def table():
    return json.loads(TABLE.read_bytes())


def test_the_table_bytes_are_pinned():
    assert hashlib.sha256(TABLE.read_bytes()).hexdigest() == TABLE_2_SHA256


def test_the_table_records_its_reasons2_source(table):
    assert table["schema"] == "engine-reason-table/2"
    assert table["source"] == {
        "path": "lobster/core/figures/errors.py",
        "sha256": "7ce0341f0aad9627a3a77766e4c2d65f50548acac0c280ad7823ef21affa0642",
        "commit": "bf59b3b0011c1ee7eb82f73eb75b64ea28b58cb5",
    }


def test_every_reasons2_token_is_still_live(table):
    from lobster.core.figures import errors

    live = {cls.__name__: code for cls, code in errors._known_classes().items()}
    assert all(live.get(c["class"]) == c["code"] for c in table["codes"])
    assert set(table["details"]) <= errors.DETAIL_REASONS
    assert set(table["prefixes"]) <= errors.REASON_PREFIXES
    assert set(table["standalone"]) <= errors.STANDALONE_REASONS


def test_counts_and_the_delta_from_the_b_pinned_snapshot(table):
    assert table["counts"] == {
        "codes": 12,
        "details": 58,
        "prefixes": 4,
        "standalone": 2,
    }
    base = table["baseline"]
    assert base["commit"] == "b550992b29950b599833d3b6aa4afb9b76b1c925"
    assert base["details_count"] == 55
    assert base["details_added"] == [
        "figure_submission_invalid",
        "not_requested",
        "total_cap",
    ]
    assert base["details_removed"] == []
    assert set(base["details_added"]) <= set(table["details"])


def test_codes_include_the_base_class_and_resolve_like_safe_reason(table):
    from lobster.core.figures import errors

    codes = {c["class"]: c["code"] for c in table["codes"]}
    assert codes["FigureError"] == "figure_error"
    for name, code in codes.items():
        assert errors.safe_reason(getattr(errors, name)("x")) == code


def test_the_table_holds_tokens_only_never_messages(table):
    tokens = (
        [c["code"] for c in table["codes"]]
        + table["details"]
        + table["prefixes"]
        + table["standalone"]
    )
    assert all(
        t.isascii() and t.replace("_", "").isalnum() and t.islower() for t in tokens
    )


def test_existing_vectors_are_byte_identical():
    shas = {
        name: hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
        for name in (
            "candidate_vectors.json",
            "candidate_vectors_2.json",
            "execution_request_vectors.json",
        )
    }
    assert shas == {
        "candidate_vectors.json": "1d8bdeed14eebc9c0276e0bc08ac3fedc4cbdef951d87a8d072ee4d5e0e07104",
        "candidate_vectors_2.json": "3ba1c6b4c44c630814eb75037bb4b8a6c519b557c04d89c6d457f019337216ff",
        "execution_request_vectors.json": "708bdabe7e972167514dac4ac1ddb1fcfa0715da5931298a2483adb4072cc08a",
    }
