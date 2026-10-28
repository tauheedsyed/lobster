"""``engine-reason-table/3`` equals the live engine vocabulary (data only, no runtime change).

The fixture is bound to the exact ``errors.py`` bytes: any change to the reason source fails
here until the table is regenerated (a new versioned table, synchronized with B). Its baseline
is the frozen reasons/2 table (``test_reason_table_2.py``).
"""

import importlib.util
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
TABLE = FIXTURES / "reason_table_3.json"


def _gen():
    spec = importlib.util.spec_from_file_location(
        "gen_reason_table_3", FIXTURES / "gen_reason_table_3.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # module level only: git runs in main() alone
    return module


@pytest.fixture(scope="module")
def table():
    return json.loads(TABLE.read_bytes())


def test_the_table_is_byte_equal_to_a_rebuild_from_the_live_source(table):
    gen = _gen()
    assert gen.render(gen.build(table["source"]["commit"])) == TABLE.read_bytes()


def test_the_table_is_bound_to_the_exact_reason_source_bytes(table):
    gen = _gen()
    assert table["schema"] == "engine-reason-table/3"
    assert table["source"]["path"] == "lobster/core/figures/errors.py"
    assert table["source"]["sha256"] == gen.source_sha256()


def test_counts_and_the_delta_from_reasons2(table):
    assert table["counts"] == {
        "codes": 13,
        "details": 58,
        "prefixes": 4,
        "standalone": 2,
    }
    base = table["baseline"]
    assert base["schema"] == "engine-reason-table/2"
    assert base["table_sha256"] == (
        "ebb14aab17e907a682df3a4246ea2997874bd8b0b70ae38915dcc62ca81c114c"
    )
    assert base["source_sha256"] == (
        "7ce0341f0aad9627a3a77766e4c2d65f50548acac0c280ad7823ef21affa0642"
    )
    assert base["source_commit"] == "bf59b3b0011c1ee7eb82f73eb75b64ea28b58cb5"
    assert base["codes_added"] == ["figure_deleted"]
    assert base["classes_added"] == ["FigureDeletedError"]
    assert base["codes_removed"] == base["classes_removed"] == []
    assert base["details_added"] == base["details_removed"] == []


def test_figure_deleted_is_a_not_found_subclass_and_resolves_to_its_own_code(table):
    codes = {c["class"]: c for c in table["codes"]}
    assert codes["FigureDeletedError"] == {
        "class": "FigureDeletedError",
        "code": "figure_deleted",
        "base": "FigureNotFoundError",
    }
    assert codes["FigureError"]["base"] is None
    assert table["resolution"]["classes"]["FigureDeletedError"] == "figure_deleted"
    assert table["resolution"]["unknown_exception"] == "unexpected_error"


def test_codes_resolve_like_safe_reason(table):
    from lobster.core.figures import errors

    for entry in table["codes"]:
        cls = getattr(errors, entry["class"])
        assert errors.safe_reason(cls("x")) == entry["code"]
        assert table["resolution"]["classes"][entry["class"]] == entry["code"]


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
