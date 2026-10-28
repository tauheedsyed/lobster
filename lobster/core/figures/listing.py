"""One-line descriptions of saved figures for the overview and the handoff manifest (contract §8).

Facts only, never directives: these lines reach agents as tool output. A label is DECLARED by
caller code (untrusted), so it is flattened to one line, stripped of angle brackets (it can never
close a ``<tool_data>``/``<artifact_manifest>`` block) and capped.
"""

from __future__ import annotations

import os
import stat as _stat
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

MAX_LABEL_CHARS = 80
OVERVIEW_LIMIT = 10
LEGACY_LIMIT = 5


def safe_label(label: Any) -> str:
    """A caller-declared label or file name as one bounded line of plain text ("" if none).

    Display only (the stored record/path is unchanged): angle brackets and every Unicode
    control/format/surrogate/private-use/unassigned character (category ``C*``: ESC, NUL,
    bidi overrides, ...) become spaces; whitespace runs collapse to one space.
    """
    if not isinstance(label, str):
        return ""
    cleaned = "".join(
        " " if ch in "<>" or unicodedata.category(ch).startswith("C") else ch
        for ch in label
    )
    flat = " ".join(cleaned.split())
    if len(flat) > MAX_LABEL_CHARS:
        flat = flat[: MAX_LABEL_CHARS - 1].rstrip() + "…"
    return flat


def figure_ref(item: Mapping[str, Any]) -> str:
    return f"{item['figure_id']}@{item['revision_id']}"


def figure_line(item: Mapping[str, Any]) -> str:
    """``<label> · <ref> · kind · source yes/no · edit of <parent ref>`` (+ pending)."""
    parts = [
        safe_label(item.get("label")) or "untitled",
        figure_ref(item),
        (item.get("change") or {}).get("kind") or "unknown",
        "source yes" if item.get("source") else "source no",
    ]
    parents = [p for p in item.get("parents") or [] if isinstance(p, str)]
    if parents:
        parts.append(
            "edit of " + ", ".join(f"{item['figure_id']}@{p}" for p in parents)
        )
    if item.get("availability") == "pending":
        parts.append("save confirmation pending")
    return " · ".join(parts)


def read_saved_figures(
    workspace_path: Any, store_provider: Callable[[], Any]
) -> List[Mapping[str, Any]]:
    """THE native reader of saved revision records (overview, manifest, export, /plots).

    ``[]`` only when the workspace has no ``plots`` folder (absence). Any other fault is
    TYPED and raised, never a partial or empty list: a folder I/O fault or a non-folder
    ``plots`` -> ``StorageError`` (storage_failed); store access and catalog errors
    propagate as raised by the store (typed).
    """
    from lobster.core.figures.errors import StorageError

    plots = Path(workspace_path) / "plots"
    try:
        st = os.stat(plots)
    except (FileNotFoundError, NotADirectoryError):
        return []
    except OSError:
        raise StorageError("plots folder unreadable") from None
    if not _stat.S_ISDIR(st.st_mode):
        raise StorageError("plots is not a folder")
    return _all_items(store_provider())


def legacy_plot_files(workspace_path: Any) -> List[str]:
    """Names of the flat (legacy) files directly in ``plots/``; ``[]`` if there is none.

    Display only: names are rendered through :func:`safe_label`. A fault is a typed
    ``StorageError``, never an empty list.
    """
    from lobster.core.figures.errors import StorageError

    plots = Path(workspace_path) / "plots"
    try:
        with os.scandir(plots) as entries:
            return [e.name for e in entries if e.is_file(follow_symlinks=False)]
    except (FileNotFoundError, NotADirectoryError):
        return []
    except OSError:
        raise StorageError("plots folder unreadable") from None


def _all_items(store) -> List[Mapping[str, Any]]:
    items: List[Mapping[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        page = store.list(cursor=cursor, limit=500)
        items.extend(page.items)
        cursor = page.next_cursor
        if not cursor:
            return items


def describe_saved_figures(
    store,
    legacy_names: Iterable[str] = (),
    limit: int = OVERVIEW_LIMIT,
) -> str:
    """Markdown section listing saved revisions, newest last ("" when there is nothing).

    Restart-safe: read from the store's records, not from in-memory state. A catalog that
    cannot be read is stated explicitly (typed safe reason), never shown as empty.
    """
    from lobster.core.figures.errors import safe_reason

    try:
        items = _all_items(store) if store is not None else []
    except Exception as exc:  # explicit, never a silent empty catalog
        return _not_listed(safe_reason(exc))
    return _render(items, sorted(legacy_names), None, limit)


def describe_workspace_figures(data_manager, limit: int = OVERVIEW_LIMIT) -> str:
    """The overview section for a data manager's workspace (same reader as everyone else).

    No workspace or no ``plots`` folder -> "" (absence). Any read fault -> an explicit
    "Not listed: <reason>" line (fixed vocabulary), never an empty or raw-error section.
    """
    from lobster.core.figures.errors import safe_reason

    try:
        root = Path(data_manager.workspace_path)
    except TypeError:
        return ""  # no workspace path: nothing to list
    try:
        items = read_saved_figures(root, lambda: data_manager.figure_store)
    except Exception as exc:
        return _not_listed(safe_reason(exc))
    legacy_error = None
    try:
        legacy = sorted(legacy_plot_files(root))
    except Exception as exc:
        legacy, legacy_error = [], safe_reason(exc)
    return _render(items, legacy, legacy_error, limit)


def _not_listed(reason: str) -> str:
    return f"\n## Saved figures\n\n- Not listed: {reason}\n"


def _render(
    items: Sequence[Mapping[str, Any]],
    legacy: Sequence[str],
    legacy_error: Optional[str],
    limit: int,
) -> str:
    if not items and not legacy and legacy_error is None:
        return ""
    lines = [f"\n## Saved figures ({len(items)})\n"]
    shown = items[-limit:] if limit else items
    if len(items) > len(shown):
        lines.append(f"- … {len(items) - len(shown)} older revision(s) not listed")
    lines.extend(f"- {figure_line(item)}" for item in shown)
    if legacy_error is not None:
        lines.append(f"- Legacy plot files: not listed ({legacy_error})")
    elif legacy:
        names = ", ".join(f"legacy:{safe_label(n)}" for n in legacy[:LEGACY_LIMIT])
        more = len(legacy) - LEGACY_LIMIT
        lines.append(
            f"- Legacy plot files (read-only, no saved history): {names}"
            + (f", … (+{more} more)" if more > 0 else "")
        )
    lines.append(
        "\nA ref (f_…@r_…) names one saved revision. An edit is saved as a new revision; "
        "the original stays unchanged.\n"
    )
    return "\n".join(lines)
