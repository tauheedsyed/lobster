"""Local HTML renditions of committed revisions.

A rendition is a *representation* of exact committed bytes, never a new revision:
- rendered with Plotly's own HTML serializer from the sha-verified figure + config bytes;
- plotly.js is inlined from the locally installed plotly package (no CDN, no network);
- no Chrome/Kaleido/PNG, no scientific regeneration, no revision rewrite;
- append-only: ``plots/<fid>/<rid>/renditions/<p_id>.html`` + ``<p_id>.json`` (record written last);
- identical inputs + settings + renderer reuse the existing rendition instead of adding one.

Generated only on explicit save/view/export paths, never as an analysis side effect. A failed
rendition never affects the revision: callers report "revision saved" and "HTML view failed"
separately (stable reason codes only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from lobster.core.figures.errors import safe_reason, safe_result_reason
from lobster.core.figures.ids import FigureRef

RENDITION_SCHEMA = "lobster.figure_rendition/1"
HTML_RENDERER = "plotly.io.to_html"
HTML_SETTINGS: Dict[str, Any] = {
    "format": "html",
    "media_type": "text/html",
    "include_plotlyjs": "inline",
    "full_html": True,
}


@dataclass
class RenditionResult:
    ref: str
    status: str  # available | failed
    path: Optional[str] = None  # workspace-relative
    rendition_id: Optional[str] = None
    reused: bool = False
    reason: Optional[str] = None


def renderer_identity() -> Dict[str, Any]:
    import plotly
    from plotly.offline import get_plotlyjs_version

    try:
        from lobster.version import __version__ as engine_version
    except Exception:  # pragma: no cover
        engine_version = "unknown"
    return {
        "name": HTML_RENDERER,
        "plotly_version": plotly.__version__,
        "plotlyjs_version": get_plotlyjs_version(),
        "engine_version": engine_version,
    }


def render_html(figure_bytes: bytes, config_bytes: bytes, *, div_id: str) -> bytes:
    """Plotly-native HTML from the exact committed bytes (deterministic for fixed inputs)."""
    import plotly.io as pio

    figure = json.loads(figure_bytes)
    config = json.loads(config_bytes) or None
    html = pio.to_html(
        figure,
        config=config,
        include_plotlyjs=True,
        full_html=True,
        div_id=div_id,
        validate=False,  # bytes were validated at commit; render them as stored
    )
    return html.encode("utf-8")


def _pending_view(ref: FigureRef, rec: Dict[str, Any]) -> RenditionResult:
    """Uncertain save confirmation: no path is handed out (private locator only)."""
    reason = safe_result_reason(rec.get("availability_reason")) or "durability_failed"
    return RenditionResult(str(ref), "pending", reason="html_rendition:" + reason)


def ensure_html_rendition(store, ref: "str | FigureRef") -> RenditionResult:
    """Return an HTML rendition of ``ref``, creating it once if missing."""
    ref = FigureRef.parse(ref)
    try:
        record_bytes, figure_bytes, config_bytes = store.read(ref)  # sha-verified
        record = json.loads(record_bytes)
        receipt = {
            "renderer": renderer_identity(),
            "inputs": {
                "figure_sha256": record["figure"]["sha256"],
                "config_sha256": record["config"]["sha256"],
            },
        }
        existing = store.find_rendition(ref, HTML_SETTINGS, receipt)
        if existing is not None:
            if existing.get("availability") == "pending":
                return _pending_view(ref, existing)
            return RenditionResult(
                str(ref),
                "available",
                path=existing["path"],
                rendition_id=existing["rendition_id"],
                reused=True,
            )
        html = render_html(
            figure_bytes, config_bytes, div_id=f"lobster-{ref.revision_id}"
        )
        added = store.add_rendition(ref, html, HTML_SETTINGS, receipt)
        if added.get("availability") == "pending":
            return _pending_view(ref, added)
        return RenditionResult(
            str(ref),
            "available",
            path=added["path"],
            rendition_id=added["rendition_id"],
        )
    except Exception as exc:  # the revision is unaffected; report a code only
        from lobster.core.figures.errors import DurabilityError

        status = "pending" if isinstance(exc, DurabilityError) else "failed"
        return RenditionResult(
            str(ref), status, reason="html_rendition:" + safe_reason(exc)
        )


def ensure_html_renditions(
    store, refs: Optional[Iterable[str]] = None
) -> List[RenditionResult]:
    """Ensure HTML views for ``refs`` (default: every available revision in the store)."""
    if refs is None:
        refs = []
        cursor = None
        while True:
            page = store.list(cursor=cursor, limit=500)
            refs.extend(f"{r['figure_id']}@{r['revision_id']}" for r in page.items)
            cursor = page.next_cursor
            if not cursor:
                break
    return [ensure_html_rendition(store, ref) for ref in refs]


def describe_renditions(results: List[RenditionResult]) -> str:
    """One line for humans: revisions are already saved; this reports the HTML views only."""
    if not results:
        return "No saved figures."
    ok = [r for r in results if r.status == "available"]
    new = [r for r in ok if not r.reused]
    pending = [r for r in results if r.status == "pending"]
    failed = [r for r in results if r.status not in ("available", "pending")]
    parts = [f"{len(results)} saved figure revision(s)"]
    parts.append(f"HTML views: {len(ok)} ready ({len(new)} new)")
    if pending:
        parts.append(
            f"{len(pending)} HTML view(s): save confirmation pending: "
            + ", ".join(r.ref for r in pending[:5])
        )
    if failed:
        parts.append(
            f"{len(failed)} HTML view(s) failed (revisions unaffected): "
            + ", ".join(
                f"{r.ref} {safe_result_reason(r.reason) or 'reason_withheld'}"
                for r in failed[:5]
            )
        )
    return "; ".join(parts)


__all__ = [
    "RENDITION_SCHEMA",
    "HTML_SETTINGS",
    "RenditionResult",
    "render_html",
    "ensure_html_rendition",
    "ensure_html_renditions",
    "describe_renditions",
]
