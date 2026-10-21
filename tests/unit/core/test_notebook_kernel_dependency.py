"""`/pipeline run` needs a Jupyter kernel in the user's environment.

papermill does not install one, so without ipykernel every run fails with
"No such kernel named python3".
"""

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"


def _core_dependencies():
    data = tomllib.loads(PYPROJECT.read_text())
    return {
        re.split(r"[<>=!~\[ ;]", dep, maxsplit=1)[0].lower()
        for dep in data["project"]["dependencies"]
    }


def test_core_install_includes_a_notebook_kernel():
    deps = _core_dependencies()
    assert "papermill" in deps
    assert "ipykernel" in deps
