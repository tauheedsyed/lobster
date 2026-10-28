"""Byte-exact figure/config payloads and fingerprints (contract §4).

- Stored bytes are exactly the submitted bytes; validation parses, never re-serializes.
- Our helpers produce initial figure bytes with ``plotly.io.to_json(..., remove_uids=False,
  engine="json")`` from a deep copy (uids, typed arrays and frames preserved).
- Strict JSON only: ``NaN``/``Infinity`` literals are refused, never repaired.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Optional

from lobster.core.figures.errors import FigureInvalidError

#: Size caps (contract §6.4; not product limits).
MAX_FIGURE_BYTES = 64 * 1024 * 1024
MAX_CONFIG_BYTES = 64 * 1024
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_TEMPLATE_ARGS_BYTES = 16 * 1024


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-strict JSON constant {name}")


def strict_json_loads(data: bytes, *, what: str) -> Any:
    """Parse strict JSON (no NaN/Infinity, UTF-8) or raise FigureInvalidError."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FigureInvalidError(
            f"{what} is not UTF-8", details={"reason": "not_utf8"}
        ) from exc
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except ValueError as exc:
        reason = "non_strict_json" if "non-strict" in str(exc) else "invalid_json"
        raise FigureInvalidError(
            f"{what} is not strict JSON: {exc}", details={"reason": reason}
        ) from exc


def figure_to_bytes(fig: Any) -> bytes:
    """Serialize a Plotly figure (object or dict) from a deep copy, preserving uids.

    Same function the custom-code child uses (:mod:`lobster.core.figures.serialize`).
    """
    from lobster.core.figures.serialize import figure_to_bytes as _impl

    return _impl(fig)


def config_to_bytes(config: Optional[Mapping[str, Any]]) -> bytes:
    """Canonical JSON for engine-created configs only (API/editor bytes stay exact).

    Same bytes as the child's :func:`lobster.core.figures.serialize.config_to_bytes`.
    """
    from lobster.core.figures.serialize import config_to_bytes as _impl

    return _impl(dict(config or {}))


def validate_figure_bytes(data: bytes) -> dict:
    """Validate candidate figure bytes; return the parsed dict. Bytes are not modified."""
    if not isinstance(data, (bytes, bytearray)):
        raise FigureInvalidError(
            "figure bytes must be bytes", details={"reason": "type"}
        )
    if len(data) > MAX_FIGURE_BYTES:
        raise FigureInvalidError(
            "figure too large",
            details={"reason": "figure_too_large", "bytes": len(data)},
        )
    parsed = strict_json_loads(bytes(data), what="figure.json")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("data", []), list):
        raise FigureInvalidError(
            "figure.json must be an object with a 'data' list",
            details={"reason": "shape"},
        )
    import plotly.io as pio

    try:
        pio.from_json(bytes(data).decode("utf-8"), skip_invalid=False)
    except Exception as exc:  # plotly raises ValueError subclasses for schema errors
        raise FigureInvalidError(
            f"not a valid Plotly figure: {exc}", details={"reason": "plotly_schema"}
        ) from exc
    return parsed


def validate_config_bytes(data: bytes) -> dict:
    if not isinstance(data, (bytes, bytearray)):
        raise FigureInvalidError(
            "config bytes must be bytes", details={"reason": "type"}
        )
    if len(data) > MAX_CONFIG_BYTES:
        raise FigureInvalidError(
            "config too large",
            details={"reason": "config_too_large", "bytes": len(data)},
        )
    parsed = strict_json_loads(bytes(data), what="config.json")
    if not isinstance(parsed, dict):
        raise FigureInvalidError(
            "config.json must be an object", details={"reason": "shape"}
        )
    return parsed


def declared_template(parsed_figure: Mapping[str, Any]) -> Optional[dict]:
    """Return ``layout.meta.lobster_template`` (declared, never proof) or None.

    Refuses oversize/non-JSON args rather than truncating.
    """
    layout = parsed_figure.get("layout") or {}
    meta = layout.get("meta") if isinstance(layout, dict) else None
    if not isinstance(meta, dict):
        return None
    tmpl = meta.get("lobster_template")
    if tmpl is None:
        return None
    if not isinstance(tmpl, dict) or not isinstance(tmpl.get("name"), str):
        raise FigureInvalidError(
            "layout.meta.lobster_template must be {name, version, args}",
            details={"reason": "template_meta"},
        )
    args = tmpl.get("args", {})
    encoded = json.dumps(
        args, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > MAX_TEMPLATE_ARGS_BYTES:
        raise FigureInvalidError(
            "template args exceed cap",
            details={"reason": "template_args_too_large", "bytes": len(encoded)},
        )
    return {"name": tmpl["name"], "version": str(tmpl.get("version", "")), "args": args}


def source_to_bytes(
    records: Any, declaration: Optional[Mapping[str, Any]] = None
) -> bytes:
    """Encode a declared source payload (provisional JSON): columns + rows + declaration."""
    from lobster.core.figures.serialize import SerializationError
    from lobster.core.figures.serialize import source_to_bytes as _impl

    try:
        data = _impl(records, declaration)
    except SerializationError:
        raise FigureInvalidError(
            "source must be a DataFrame or a list of records",
            details={"reason": "source_shape"},
        ) from None
    if len(data) > MAX_SOURCE_BYTES:
        raise FigureInvalidError(
            "source too large",
            details={"reason": "source_too_large", "bytes": len(data)},
        )
    return data
