"""Child-side figure helpers for ``execute_custom_code`` (contract §6.1, §6.4).

Runs INSIDE the custom-code subprocess. It has no write path to the store and no catalog
credentials: it only writes candidate files into the per-attempt ``EXEC_CACHE/candidates/`` and
reads the attempt snapshot the trusted parent prepared in ``EXEC_CACHE/figures/``. The parent
decides what is saved, and only after the attempt succeeded (successful-attempt fence).

It imports NOTHING from the ``lobster`` package (the parent loads this file and
``serialize.py`` by path), so no configuration/dotenv code ever runs in user-code processes.

Explicit submission only: figures are saved ONLY via ``submit_figure`` (``publish_plot`` is an
alias). There is no automatic capture; the epilogue merely names unsubmitted Figure variables
so the reply can tell the model what was not saved.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

MAX_FIGURE_BYTES = 64 * 1024 * 1024  # same caps as the parent (it re-checks)
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_CONFIG_BYTES = 64 * 1024  # same cap as the parent (it re-checks)


def _load_serialize():
    """Load the sibling ``serialize.py`` by path (no package import)."""
    path = Path(__file__).with_name("serialize.py")
    spec = importlib.util.spec_from_file_location("_lobster_fig_serialize", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CANDIDATES_DIR = "candidates"
SNAPSHOT_DIR = "figures"
UNSUBMITTED_FILE = "unsubmitted.json"
SNAPSHOT_INDEX = "index.json"


class FigureUnavailable(LookupError):
    """A figure/source is not available in this attempt's snapshot (never a filesystem fallback)."""

    def __init__(self, ref: str, status: str, reason: Optional[str] = None):
        self.ref, self.status, self.reason = ref, status, reason
        hint = {
            "not_in_snapshot": "it did not exist when this attempt started; run another attempt",
            "not_requested": "write the ref literally in your code so it is loaded at the next attempt",
            "total_cap": "too many figures were loaded at once; reference fewer refs per attempt",
            "too_large": "it exceeds the per-figure size cap",
            "figure_deleted": "it was deleted",
        }.get(reason or status, "")
        detail = f"{status}" + (f" ({reason})" if reason else "")
        super().__init__(f"{ref}: {detail}" + (f" — {hint}" if hint else ""))


def install(namespace: Dict[str, Any], exec_cache: "str | Path") -> None:
    """Seed the figure helpers into the child namespace."""
    exec_cache = Path(exec_cache)
    candidates = exec_cache / CANDIDATES_DIR
    snapshot = exec_cache / SNAPSHOT_DIR
    submitted_ids: set = set()
    counter = {"n": 0}
    ser = _load_serialize()

    def _index() -> Dict[str, Any]:
        path = snapshot / SNAPSHOT_INDEX
        if not path.is_file():
            return {"revisions": [], "status": {}, "sources": {}}
        return json.loads(path.read_text(encoding="utf-8"))

    def submit_figure(
        fig: Any,
        *,
        source: Any = None,
        intent: Any = None,
        parent: Optional[str] = None,
        label: Optional[str] = None,
        config: Any = None,
    ) -> int:
        """Submit a Plotly figure to be saved after this code finishes successfully.

        Args:
            fig: plotly.graph_objects.Figure (serialized NOW; later changes are not included).
            source: the data the figure was drawn from — a DataFrame or list of records, or
                ``{"data": <records>, "declared": {...}}``. ``None`` = "no regeneration source".
            intent: what the figure should show (str), or resolved intent items (list of dicts).
            parent: ``"<figure_id>@<revision_id>"`` when this is an edit of an existing figure.
            label: short human label (defaults to the figure title).
            config: Plotly config dict. Omitted or ``None`` = not given: an edit keeps its
                parent's exact saved config, a new figure gets ``{}``. An explicit dict
                (``{}`` included, meaning "clear") is saved exactly. JSON values only.

        Returns the candidate index. Nothing is saved if the code fails.
        """
        import plotly.graph_objects as go

        if not isinstance(fig, go.Figure):
            raise TypeError("submit_figure expects a plotly.graph_objects.Figure")
        index = counter["n"]
        figure_bytes = ser.figure_to_bytes(fig)
        if len(figure_bytes) > MAX_FIGURE_BYTES:
            raise ValueError(
                "figure exceeds the 64 MiB cap; reduce points or aggregate"
            )
        source_bytes = None
        if source is not None:
            if isinstance(source, dict) and "data" in source:
                source_bytes = ser.source_to_bytes(
                    source["data"], source.get("declared")
                )
            else:
                source_bytes = ser.source_to_bytes(source)
            if len(source_bytes) > MAX_SOURCE_BYTES:
                raise ValueError(
                    "source exceeds the 64 MiB cap; pass the plotted rows only"
                )
        config_bytes = None
        if config is not None:  # validated BEFORE any file or counter is written
            config_bytes = ser.config_to_bytes(config)
            if len(config_bytes) > MAX_CONFIG_BYTES:
                raise ValueError("config exceeds the 64 KiB cap")
        if label is None:
            title = getattr(getattr(fig.layout, "title", None), "text", None)
            label = str(title) if title else None
        request, resolved = (None, None)
        if isinstance(intent, str):
            request = intent
        elif isinstance(intent, dict):
            resolved = [intent]
        elif isinstance(intent, list):
            resolved = intent
        meta = {
            "index": index,
            "label": label,
            "request": request,
            "resolved_intent": resolved,
            "parent": parent,
            "has_source": source_bytes is not None,
            "has_config": config_bytes is not None,
        }
        json.dumps(meta, allow_nan=False)  # fail here, in the child, if not JSON-native
        cdir = candidates / str(index)
        cdir.mkdir(parents=True, exist_ok=False)
        (cdir / "figure.json").write_bytes(figure_bytes)
        if source_bytes is not None:
            (cdir / "source.json").write_bytes(source_bytes)
        if config_bytes is not None:
            (cdir / "config.json").write_bytes(config_bytes)
        (cdir / "candidate.json").write_text(json.dumps(meta), encoding="utf-8")
        counter["n"] += 1
        submitted_ids.add(id(fig))
        return index

    def publish_plot(fig: Any, title: str = "Plot") -> int:
        """Alias of ``submit_figure(fig, label=title)`` (kept for existing code)."""
        return submit_figure(fig, label=title)

    def list_figures() -> List[Dict[str, Any]]:
        """Saved figure revisions as of the START of this attempt (heads and history)."""
        return list(_index().get("revisions", []))

    def load_figure(ref: str):
        """Load a saved revision from this attempt's snapshot as a plotly Figure."""
        import plotly.io as pio

        state = _index().get("status", {}).get(ref)
        if state is None:
            raise FigureUnavailable(ref, "not_in_snapshot")
        if state.get("status") != "materialized":
            raise FigureUnavailable(ref, state.get("status"), state.get("reason"))
        fid, rid = ref.split("@", 1)
        data = (snapshot / fid / rid / "figure.json").read_text(encoding="utf-8")
        return pio.from_json(data)

    def load_figure_config(ref: str) -> Dict[str, Any]:
        """The saved config of a revision in this attempt's snapshot (verified on export)."""
        state = _index().get("status", {}).get(ref)
        if state is None:
            raise FigureUnavailable(ref, "not_in_snapshot")
        if state.get("status") != "materialized":
            raise FigureUnavailable(ref, state.get("status"), state.get("reason"))
        fid, rid = ref.split("@", 1)
        return json.loads((snapshot / fid / rid / "config.json").read_text("utf-8"))

    def load_source(source_id: str):
        """Load a saved figure's source data (pandas DataFrame; declaration in ``.attrs``)."""
        state = _index().get("sources", {}).get(source_id)
        if state is None or state.get("status") != "materialized":
            raise FigureUnavailable(
                source_id,
                (state or {}).get("status", "not_in_snapshot"),
                (state or {}).get("reason"),
            )
        payload = json.loads(
            (snapshot / "_sources" / f"{source_id}.json").read_text(encoding="utf-8")
        )
        import pandas as pd

        frame = pd.DataFrame(
            payload.get("rows", []), columns=payload.get("columns", [])
        )
        frame.attrs["declared"] = payload.get("declared", {})
        return frame

    def _lobster_figure_epilogue(scope: Dict[str, Any]) -> None:
        """Name every Plotly Figure variable that was NOT submitted (warning only)."""
        try:
            import plotly.graph_objects as go
        except ImportError:  # pragma: no cover
            return
        names = sorted(
            name
            for name, value in scope.items()
            if not name.startswith("_")
            and isinstance(value, go.Figure)
            and id(value) not in submitted_ids
        )
        if names:
            candidates.mkdir(parents=True, exist_ok=True)
            (candidates / UNSUBMITTED_FILE).write_text(
                json.dumps(names), encoding="utf-8"
            )

    namespace.update(
        submit_figure=submit_figure,
        publish_plot=publish_plot,
        list_figures=list_figures,
        load_figure=load_figure,
        load_figure_config=load_figure_config,
        load_source=load_source,
        FigureUnavailable=FigureUnavailable,
        _lobster_figure_epilogue=_lobster_figure_epilogue,
    )
    # Template API + house style, when the templates package is installed. It must stay
    # free of lobster.core/lobster.config imports (it runs in the user-code child).
    try:
        import lobster.figure_templates as _templates  # noqa: F401

        for name in (
            "list_templates",
            "get_template",
            "show_template_source",
            "apply_house_style",
            "initialize_house_style",
        ):
            if hasattr(_templates, name):
                namespace[name] = getattr(_templates, name)
        if "initialize_house_style" in namespace:
            namespace["initialize_house_style"]()  # Plotly default template = omics_os
    except Exception:  # templates are optional in this branch; never block execution
        pass
