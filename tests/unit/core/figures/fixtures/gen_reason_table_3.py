"""Generate ``reason_table_3.json`` (``engine-reason-table/3``) from the engine reason source.

Data only. It reads the fixed reason vocabulary of ``lobster/core/figures/errors.py``: the class
codes (the ``FigureError`` base plus every derived class), each class's nearest known base, the
detail tokens, the prefixes and the standalone reasons. It never reads exception messages, reprs
or contexts, and it changes no runtime behaviour.

The baseline is the FROZEN ``reason_table_2.json`` in this directory (read as data, pinned by its
sha256), so the code/detail/class delta needs no git. Only ``source.commit`` is git-derived: it
is passed in, so the equality test rebuilds the whole file from the live module plus that pinned
field. Git runs in :func:`main` only (generation time).

Regenerate with (from the worktree root):
    .venv/bin/python tests/unit/core/figures/fixtures/gen_reason_table_3.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

SCHEMA = "engine-reason-table/3"
SOURCE_PATH = "lobster/core/figures/errors.py"
OUT = Path(__file__).with_name("reason_table_3.json")
BASELINE_TABLE = Path(__file__).with_name("reason_table_2.json")
#: The frozen, B-consumed reasons/2 table and the source it was generated from.
BASELINE_TABLE_SHA256 = (
    "ebb14aab17e907a682df3a4246ea2997874bd8b0b70ae38915dcc62ca81c114c"
)
BASELINE_SOURCE_SHA256 = (
    "7ce0341f0aad9627a3a77766e4c2d65f50548acac0c280ad7823ef21affa0642"
)
BASELINE_SOURCE_COMMIT = "bf59b3b0011c1ee7eb82f73eb75b64ea28b58cb5"


def _root() -> Path:
    return Path(__file__).resolve().parents[5]


def source_sha256() -> str:
    return hashlib.sha256((_root() / SOURCE_PATH).read_bytes()).hexdigest()


def vocabulary() -> Dict[str, Any]:
    """The live fixed vocabulary; each class with its code and nearest known base."""
    from lobster.core.figures import errors

    known = errors._known_classes()

    def nearest_base(cls: type) -> Any:
        for base in cls.__mro__[1:]:
            if base in known:
                return base.__name__
        return None

    codes = sorted(
        (
            {"class": cls.__name__, "code": code, "base": nearest_base(cls)}
            for cls, code in known.items()
        ),
        key=lambda c: c["class"],
    )
    return {
        "codes": codes,
        "details": sorted(errors.DETAIL_REASONS),
        "prefixes": sorted(errors.REASON_PREFIXES),
        "standalone": sorted(errors.STANDALONE_REASONS),
    }


def resolution() -> Dict[str, Any]:
    """How ``safe_reason`` resolves: per known class, and for an unknown exception."""
    from lobster.core.figures import errors

    return {
        "classes": {
            cls.__name__: errors.safe_reason(cls("x"))
            for cls in errors._known_classes()
        },
        "unknown_exception": errors.safe_reason(Exception("x")),
    }


def baseline() -> Dict[str, Any]:
    """The reasons/2 delta, from the frozen /2 table (data; its sha256 is checked)."""
    data = BASELINE_TABLE.read_bytes()
    if hashlib.sha256(data).hexdigest() != BASELINE_TABLE_SHA256:
        raise ValueError("reason_table_2.json is not the frozen reasons/2 table")
    old = json.loads(data)
    vocab = vocabulary()
    old_codes = {c["class"]: c["code"] for c in old["codes"]}
    new_codes = {c["class"]: c["code"] for c in vocab["codes"]}
    return {
        "schema": old["schema"],
        "table_sha256": BASELINE_TABLE_SHA256,
        "source_sha256": BASELINE_SOURCE_SHA256,
        "source_commit": BASELINE_SOURCE_COMMIT,
        "codes_added": sorted(set(new_codes.values()) - set(old_codes.values())),
        "codes_removed": sorted(set(old_codes.values()) - set(new_codes.values())),
        "classes_added": sorted(set(new_codes) - set(old_codes)),
        "classes_removed": sorted(set(old_codes) - set(new_codes)),
        "details_added": sorted(set(vocab["details"]) - set(old["details"])),
        "details_removed": sorted(set(old["details"]) - set(vocab["details"])),
    }


def build(source_commit: str) -> Dict[str, Any]:
    vocab = vocabulary()
    return {
        "schema": SCHEMA,
        "source": {
            "path": SOURCE_PATH,
            "sha256": source_sha256(),
            "commit": source_commit,
        },
        "baseline": baseline(),
        "counts": {k: len(v) for k, v in vocab.items()},
        "resolution": resolution(),
        **vocab,
    }


def render(table: Dict[str, Any]) -> bytes:
    return (
        json.dumps(table, indent=1, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )


def main() -> None:
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", SOURCE_PATH], cwd=_root()
    ).decode()
    OUT.write_bytes(render(build(commit.strip())))


if __name__ == "__main__":
    main()
