"""How the lane derives a staged notebook from a captured export (stdlib only).

Shared by the driver, which writes the staged notebook, and the checker, which
derives the expected bytes independently from the captured export, so the
only accepted difference is the approved harness cell.
"""

import json

PROBE_HEADER = "# Isolation probe (test harness only, not part of any export)\n"


def with_appended_cells(notebook_text, sources):
    """The notebook with plain code cells appended, serialized canonically."""
    data = json.loads(notebook_text)
    for source in sources:
        if not source.startswith(PROBE_HEADER):
            raise ValueError("only the approved probe cell may be appended")
        data["cells"].append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }
        )
    return json.dumps(data, indent=1)
