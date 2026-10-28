"""Attempt-start figure snapshot for custom code (contract §6.4, narrowed freshness contract).

Built by the TRUSTED parent before the child starts, at one catalog generation:
1. metadata of ALL available revisions (historical and heads), up to ``max_records``;
2. verified bytes for (a) literal refs in the code (full opaque grammar, token boundaries),
   then (b) heads, within per-item and total caps;
3. sources of every materialized revision, within the same caps.

The literal scan is a bounded convenience, not a parser, a security control or guaranteed
discovery. A ref computed at run time that is not a head is reported ``not_materialized:
not_requested`` (write it literally to load it at the next attempt), never as missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from lobster.core.figures.errors import safe_reason
from lobster.core.figures.ids import scan_refs

SNAPSHOT_SCHEMA = "lobster.figure_snapshot/1"
MAX_RECORDS = 10_000
MAX_ITEM_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024


def _summary(rec: Dict[str, Any], heads: set) -> Dict[str, Any]:
    ref = f"{rec['figure_id']}@{rec['revision_id']}"
    return {
        "ref": ref,
        "figure_id": rec["figure_id"],
        "revision_id": rec["revision_id"],
        "parents": rec.get("parents", []),
        "label": rec.get("label"),
        "kind": (rec.get("change") or {}).get("kind"),
        "created_at": rec.get("created_at"),
        "head": ref in heads,
        "source_id": (rec.get("source") or {}).get("source_id"),
        "inputs": rec.get("inputs", []),
    }


def build_snapshot(
    store,
    code_text: str,
    dest: "str | Path",
    *,
    max_records: int = MAX_RECORDS,
    max_item_bytes: int = MAX_ITEM_BYTES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> Dict[str, Any]:
    """Write ``dest/index.json`` (+ materialized bytes) and return the index (without bytes)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    total_listed = 0
    cursor = None
    generation = None
    while True:
        page = store.list(cursor=cursor, limit=500)
        generation = page.catalog_generation if generation is None else generation
        for rec in page.items:
            total_listed += 1
            if len(records) < max_records:
                records.append(rec)
        cursor = page.next_cursor
        if not cursor:
            break

    parents_of: Dict[str, set] = {}
    for rec in records:
        parents_of.setdefault(rec["figure_id"], set()).update(rec.get("parents", []))
    heads = {
        f"{r['figure_id']}@{r['revision_id']}"
        for r in records
        if r["revision_id"] not in parents_of.get(r["figure_id"], set())
    }
    listed = {f"{r['figure_id']}@{r['revision_id']}": r for r in records}
    literal = [str(ref) for ref in scan_refs(code_text) if str(ref) in listed]
    head_order = sorted(
        heads, key=lambda ref: listed[ref].get("created_at", ""), reverse=True
    )
    order = list(dict.fromkeys(literal + head_order))

    status: Dict[str, Dict[str, Any]] = {
        ref: {"status": "not_materialized", "reason": "not_requested"} for ref in listed
    }
    sources: Dict[str, Dict[str, Any]] = {}
    used = 0
    for ref in order:
        rec = listed[ref]
        if rec.get("availability") == "pending":  # never materialized for a child
            detail = str(rec.get("availability_reason") or "").split(":")[-1]
            status[ref] = {"status": "pending", "reason": detail or None}
            continue
        size = rec.get("figure", {}).get("bytes", 0)
        if size > max_item_bytes:
            status[ref] = {"status": "too_large", "reason": None}
            continue
        if used + size > max_total_bytes:
            status[ref] = {"status": "not_materialized", "reason": "total_cap"}
            continue
        try:
            rev_bytes, fig_bytes, cfg_bytes = store.read(ref)  # sha-verified
        except Exception as exc:
            status[ref] = {
                "status": "not_materialized",
                "reason": safe_reason(exc).split(":")[-1],
            }
            continue
        if len(fig_bytes) + len(cfg_bytes) > max_item_bytes:
            status[ref] = {"status": "too_large", "reason": None}
            continue
        if used + len(fig_bytes) + len(cfg_bytes) > max_total_bytes:
            status[ref] = {"status": "not_materialized", "reason": "total_cap"}
            continue
        rdir = dest / rec["figure_id"] / rec["revision_id"]
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "figure.json").write_bytes(fig_bytes)
        (rdir / "config.json").write_bytes(cfg_bytes)
        (rdir / "revision.json").write_bytes(rev_bytes)
        used += len(fig_bytes) + len(cfg_bytes)
        status[ref] = {"status": "materialized", "reason": None}
        source_id = (rec.get("source") or {}).get("source_id")
        if source_id and source_id not in sources:
            try:
                data = store.read_source(source_id)
            except Exception as exc:
                sources[source_id] = {
                    "status": "not_materialized",
                    "reason": safe_reason(exc).split(":")[-1],
                }
                continue
            if len(data) > max_item_bytes:
                sources[source_id] = {"status": "too_large", "reason": None}
            elif used + len(data) > max_total_bytes:
                sources[source_id] = {
                    "status": "not_materialized",
                    "reason": "total_cap",
                }
            else:
                (dest / "_sources").mkdir(exist_ok=True)
                (dest / "_sources" / f"{source_id}.json").write_bytes(data)
                used += len(data)
                sources[source_id] = {"status": "materialized", "reason": None}

    index = {
        "schema": SNAPSHOT_SCHEMA,
        "catalog_generation": generation,
        "snapshot_truncated": max(0, total_listed - len(records)),
        "revisions": [_summary(r, heads) for r in records],
        "status": status,
        "sources": sources,
        "materialized_bytes": used,
    }
    (dest / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return index


__all__ = ["build_snapshot", "SNAPSHOT_SCHEMA"]
