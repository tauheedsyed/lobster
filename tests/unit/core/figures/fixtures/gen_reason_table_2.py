"""Generate ``reason_table_2.json`` (``engine-reason-table/2``) from the engine reason source.

Data only. It reads the fixed reason vocabulary of ``lobster/core/figures/errors.py``: the class
codes (the ``FigureError`` base plus every derived class), the detail tokens, the prefixes and the
standalone reasons. It never reads exception messages, reprs or contexts, and it changes no
runtime behaviour.

The git-derived fields (``source.commit`` and the ``baseline`` snapshot) are passed in, so the
equality test can rebuild the whole file from the live module plus those pinned fields.

Regenerate with (from the worktree root; requires the baseline commit object locally):
    .venv/bin/python tests/unit/core/figures/fixtures/gen_reason_table_2.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

SCHEMA = "engine-reason-table/2"
SOURCE_PATH = "lobster/core/figures/errors.py"
OUT = Path(__file__).with_name("reason_table_2.json")
#: b550 snapshot that B pinned as its known vocabulary (12 codes, 55 details).
BASELINE_COMMIT = "b550992b29950b599833d3b6aa4afb9b76b1c925"


def _root() -> Path:
    return Path(__file__).resolve().parents[5]


def source_sha256() -> str:
    return hashlib.sha256((_root() / SOURCE_PATH).read_bytes()).hexdigest()


def vocabulary() -> Dict[str, Any]:
    """The live fixed vocabulary (codes include the base class)."""
    from lobster.core.figures import errors

    codes = sorted(
        (
            {"class": cls.__name__, "code": code}
            for cls, code in errors._known_classes().items()
        ),
        key=lambda c: c["class"],
    )
    return {
        "codes": codes,
        "details": sorted(errors.DETAIL_REASONS),
        "prefixes": sorted(errors.REASON_PREFIXES),
        "standalone": sorted(errors.STANDALONE_REASONS),
    }


def build(source_commit: str, baseline: Dict[str, Any]) -> Dict[str, Any]:
    vocab = vocabulary()
    return {
        "schema": SCHEMA,
        "source": {
            "path": SOURCE_PATH,
            "sha256": source_sha256(),
            "commit": source_commit,
        },
        "baseline": baseline,
        "counts": {k: len(v) for k, v in vocab.items()},
        **vocab,
    }


def render(table: Dict[str, Any]) -> bytes:
    return (
        json.dumps(table, indent=1, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=_root()).decode()


def _baseline() -> Dict[str, Any]:
    """Details of the B-pinned snapshot, read from its git object (generation time only)."""
    src = _git("show", f"{BASELINE_COMMIT}:{SOURCE_PATH}")
    details = None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DETAIL_REASONS" for t in node.targets
        ):
            details = set(ast.literal_eval(node.value.args[0]))
    current = set(vocabulary()["details"])
    return {
        "commit": BASELINE_COMMIT,
        "source_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
        "details_count": len(details),
        "details_added": sorted(current - details),
        "details_removed": sorted(details - current),
    }


def main() -> None:
    commit = _git("log", "-1", "--format=%H", "--", SOURCE_PATH).strip()
    OUT.write_bytes(render(build(commit, _baseline())))


if __name__ == "__main__":
    main()
