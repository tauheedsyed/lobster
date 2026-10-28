"""Provisional Omics-OS house style and stable entity labels for Plotly figures."""

from __future__ import annotations

HOUSE_STYLE_NAME = "omics_os"
HOUSE_COLORS = ["#007F86", "#E07A3F", "#5964A6", "#A65D80", "#7A9552", "#D0A345"]


def _house_template():
    import plotly.graph_objects as go

    return go.layout.Template(
        layout={
            "font": {"family": "Arial, sans-serif", "size": 14, "color": "#243740"},
            "paper_bgcolor": "#FFFFFF",
            "plot_bgcolor": "#FFFFFF",
            "colorway": HOUSE_COLORS,
            "xaxis": {
                "showgrid": True,
                "gridcolor": "#E8EEF0",
                "zerolinecolor": "#C5D2D7",
            },
            "yaxis": {
                "showgrid": True,
                "gridcolor": "#E8EEF0",
                "zerolinecolor": "#C5D2D7",
            },
            "margin": {"l": 65, "r": 30, "t": 70, "b": 65},
            "legend": {"bgcolor": "rgba(255,255,255,0)"},
        }
    )


def initialize_house_style():
    """Set the code namespace's default, including fully custom Plotly figures."""
    import plotly.io as pio

    pio.templates[HOUSE_STYLE_NAME] = _house_template()
    pio.templates.default = HOUSE_STYLE_NAME


def apply_house_style(figure):
    """Embed the style in a normal mutable figure; no I/O or dataset writes."""
    figure.update_layout(template=_house_template())
    return figure


def set_trace_identity(trace, entity_ids, *, entity_kind, comparison=None):
    """Entity labels in column zero; the revision source supplies their namespace.

    No source identity is guessed here. With no frozen source in the envelope,
    these are labels only, not bound scientific selections. Aggregate marks
    must not call this function without an explicit membership mapping.
    """
    trace.customdata = [[str(entity_id)] for entity_id in entity_ids]
    meta = dict(trace.meta or {})
    meta["lobster_ids"] = {
        "entity_kind": entity_kind,
        "comparison": comparison,
        "customdata_index": 0,
    }
    trace.meta = meta
