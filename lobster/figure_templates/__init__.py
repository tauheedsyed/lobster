"""Discoverable, versioned Plotly drawing functions, not model-facing tools.

Templates never save figures. Use execute_custom_code's submit_figure helper to
hand a candidate to the trusted saver. Returned figures remain freely editable.
"""

from lobster.figure_templates._api import (
    figure_template,
    get_template,
    list_templates,
    show_template_source,
)
from lobster.figure_templates._style import (
    apply_house_style,
    initialize_house_style,
    set_trace_identity,
)

__all__ = [
    "apply_house_style",
    "figure_template",
    "get_template",
    "initialize_house_style",
    "list_templates",
    "set_trace_identity",
    "show_template_source",
]
