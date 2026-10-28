"""Standalone figure serialization (NO lobster imports; loadable by file path in the child).

The custom-code child loads this file directly (``importlib`` by path) so that it never imports
the ``lobster`` package — importing ``lobster.core``/``lobster.config`` in the child would pull
configuration code (and potentially dotenv loading) into user-code processes. The parent uses
the same functions via :mod:`lobster.core.figures.payload`, so both sides produce identical bytes.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Optional

SOURCE_SCHEMA = "lobster.figure_source/1"


class SerializationError(ValueError):
    """Raised for an unsupported source shape (mapped to FigureInvalidError by the parent)."""


def figure_to_bytes(fig: Any) -> bytes:
    """Serialize a Plotly figure (object or dict) from a deep copy, preserving uids."""
    import plotly.io as pio

    snapshot = copy.deepcopy(fig)
    text = pio.to_json(snapshot, validate=True, remove_uids=False, engine="json")
    return text.encode("utf-8")


def config_to_bytes(config: Mapping[str, Any]) -> bytes:
    """Strict canonical JSON for a Plotly config mapping (JSON-native values only, no NaN).

    Engine-created configs and child-submitted configs use these exact bytes; API/editor
    config bytes are stored as given. Non-JSON values are refused, never coerced.
    """
    if not isinstance(config, Mapping):
        raise SerializationError("config must be a mapping (dict)")
    try:
        return json.dumps(
            dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise SerializationError("config must hold JSON-native values only") from None


def source_to_bytes(
    records: Any, declaration: Optional[Mapping[str, Any]] = None
) -> bytes:
    """Encode a declared source payload (provisional JSON): columns + rows + declaration."""
    columns, rows = _tabulate(records)
    payload = {
        "schema": SOURCE_SCHEMA,
        "columns": columns,
        "rows": rows,
        "declared": dict(declaration or {}),
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            item = value.item()
            if isinstance(item, float) and (
                item != item or item in (float("inf"), float("-inf"))
            ):
                return None
            return item
    except ImportError:  # pragma: no cover
        pass
    return str(value)


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (
        value != value or value in (float("inf"), float("-inf"))
    ):
        return None
    return value


def _tabulate(records: Any) -> "tuple[list[str], list[list[Any]]]":
    if hasattr(records, "to_dict") and hasattr(records, "columns"):  # pandas DataFrame
        columns = [str(c) for c in records.columns]
        rows = [
            [_clean(v) for v in row]
            for row in records.itertuples(index=False, name=None)
        ]
        return columns, rows
    if isinstance(records, list) and (not records or isinstance(records[0], Mapping)):
        columns: list[str] = []
        for rec in records:
            for key in rec.keys():
                if str(key) not in columns:
                    columns.append(str(key))
        rows = [[_clean(rec.get(c)) for c in columns] for rec in records]
        return columns, rows
    raise SerializationError("source must be a DataFrame or a list of records")
