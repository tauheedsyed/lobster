"""How the lane derives a staged notebook from a captured export (stdlib only).

Shared by the driver, which writes the staged notebook, and the checker, which
derives the expected bytes independently from the captured export. The only
accepted difference is the isolation probe cell, and its source is never
taken from the stage: the stage records the probe's arguments, which are
validated, and the cell is generated here from one approved template.
"""

import json

PROBE_VERSION = 1
PROBE_HEADER = "# Isolation probe (test harness only, not part of any export)\n"
# Paths inside the execute container that must not exist (other phases' mounts).
REQUIRED_FORBIDDEN = ("/capture", "/reports", "/src")
# The forbidden host paths must include the live capture's reference and record.
REQUIRED_FORBIDDEN_SUFFIXES = ("/oracle/final.pkl", "/capture.json")
CONTROL_SUFFIX = "/roots/input.h5ad"

_PROBE_TEMPLATE = PROBE_HEADER + """import json as _json
import os as _os

_forbidden = {forbidden}
_control = {control}
_result = {{
    "forbidden_visible": [p for p in _forbidden if _os.path.lexists(p)],
    "control_visible": all(_os.path.isfile(p) for p in _control),
}}
with open(output_prefix + "_visibility.json", "w") as _handle:
    _json.dump(_result, _handle, sort_keys=True)
if _result["forbidden_visible"]:
    raise RuntimeError(f"isolation breach: {{_result['forbidden_visible']}}")
"""


def validate_probe(probe):
    """Return the probe arguments if they make a meaningful probe, else raise."""
    if not isinstance(probe, dict) or set(probe) != {"version", "forbidden", "control"}:
        raise ValueError(f"malformed probe arguments: {probe!r}")
    if probe["version"] != PROBE_VERSION:
        raise ValueError(f"unknown probe version {probe['version']!r}")
    forbidden, control = probe["forbidden"], probe["control"]
    for paths in (forbidden, control):
        if not isinstance(paths, list) or not all(
            isinstance(p, str) and p.startswith("/") for p in paths
        ):
            raise ValueError("probe paths must be lists of absolute paths")
    if forbidden != sorted(set(forbidden)):
        raise ValueError("forbidden paths must be sorted and distinct")
    missing = [p for p in REQUIRED_FORBIDDEN if p not in forbidden]
    missing += [
        s
        for s in REQUIRED_FORBIDDEN_SUFFIXES
        if not any(p.endswith(s) for p in forbidden)
    ]
    if missing:
        raise ValueError(f"probe does not check {missing}")
    if len(control) != 1 or not control[0].endswith(CONTROL_SUFFIX):
        raise ValueError("probe needs exactly its staged input as the control")
    return probe


def probe_cell(probe):
    """The approved probe cell's source for validated arguments."""
    probe = validate_probe(probe)
    return _PROBE_TEMPLATE.format(
        forbidden=json.dumps(probe["forbidden"]), control=json.dumps(probe["control"])
    )


def staged_notebook(export_text, probe=None):
    """The staged notebook: the export, plus the approved probe cell if any."""
    if probe is None:
        return export_text
    data = json.loads(export_text)
    data["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": probe_cell(probe),
        }
    )
    return json.dumps(data, indent=1)
