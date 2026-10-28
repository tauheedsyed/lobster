"""A replay notebook never silently misses a step that changed data but cannot be replayed.

A step whose IR says it changed data and cannot be replayed (``replay_incomplete`` in its
execution context) makes a replay export refuse with a fixed message. Where omitted steps
are listed, a step's fixed exclusion reason is listed with it.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import nbformat
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.notebook_exporter import NotebookExporter
from lobster.core.provenance import ProvenanceTracker
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.tools.custom_code_tool import _format_response

REASON = "uses_attempt_figure_helpers"
REFUSAL = (
    "Cannot export a replay notebook: some steps changed data but cannot be "
    "replayed. Export with the default mode, or re-run the analysis without them."
)
SECRET = "/Users/someone/private/run_1"


def _ir(op, *, replayable=True, exportable=True, context=None, outputs=("adata",)):
    return AnalysisStep(
        operation=op,
        tool_name=op,
        description=op,
        library="scanpy",
        code_template=f"print({op!r})",
        imports=[],
        parameters={},
        parameter_schema={},
        input_entities=["adata"],
        output_entities=list(outputs),
        execution_context=dict(context or {}),
        exportable=exportable,
        replayable=replayable,
    )


NORMAL = ("normalize", _ir("normalize"))
HELPER_CHANGE = (
    "custom_code_execution",
    _ir(
        "custom_code_execution",
        replayable=False,
        exportable=False,
        context={"replay_excluded": REASON, "replay_incomplete": True},
        outputs=("m_custom",),
    ),
)
HELPER_NO_CHANGE = (
    "custom_code_execution",
    _ir(
        "custom_code_execution",
        replayable=False,
        exportable=False,
        context={"replay_excluded": REASON},
        outputs=(),
    ),
)


def _exporter(tmp_path, *steps):
    tracker = ProvenanceTracker()
    for kind, ir in steps:
        tracker.create_activity(activity_type=kind, agent="a", description=kind, ir=ir)
    dm = Mock(spec=DataManagerV2)
    dm.modalities = {}
    dm.workspace_path = tmp_path
    dm.workspace_dir = tmp_path
    return NotebookExporter(tracker, dm)


def _text(path):
    return "\n".join(cell.source for cell in nbformat.read(path, as_version=4).cells)


def test_replay_export_refuses_a_data_change_it_cannot_replay(tmp_path):
    with pytest.raises(ValueError) as caught:
        _exporter(tmp_path, NORMAL, HELPER_CHANGE).export(
            "nb", filter_strategy="replay"
        )
    assert str(caught.value) == REFUSAL
    assert not (tmp_path / "notebooks" / "nb.ipynb").exists()


def test_the_default_export_lists_the_omission_with_its_reason(tmp_path):
    path = _exporter(tmp_path, NORMAL, HELPER_CHANGE).export("nb")
    assert f"`custom_code_execution ({REASON})`" in _text(path)


def test_a_replay_without_a_data_change_lists_the_omitted_step_and_reason(tmp_path):
    path = _exporter(tmp_path, NORMAL, HELPER_NO_CHANGE).export(
        "nb", filter_strategy="replay"
    )
    header = nbformat.read(path, as_version=4).cells[0].source
    assert f"`custom_code_execution ({REASON})`" in header


def test_a_session_without_such_steps_still_exports_a_replay(tmp_path):
    path = _exporter(tmp_path, NORMAL).export("nb", filter_strategy="replay")
    assert path.exists() and REASON not in _text(path)


@pytest.mark.parametrize("reason", [REASON, None])
def test_the_tool_response_carries_the_exclusion_reason(reason):
    stats = {"success": True, "replay_excluded": reason}
    assert json.loads(_format_response(None, stats, False))["replay_excluded"] == reason


@pytest.mark.parametrize("mode", ["successful", "replay"])
def test_no_notebook_text_claims_a_reproduction(tmp_path, mode):
    path = _exporter(tmp_path, NORMAL, HELPER_NO_CHANGE).export(
        "nb", filter_strategy=mode
    )
    text = _text(path).lower()
    for claim in ("reproducib", "full reproduction", "needed to reproduce"):
        assert claim not in text
    # the honest negative stays
    assert "does not show that running this notebook reproduces" in text


def _hostile(**context):
    """A step whose activity type and exclusion reason carry a path."""
    return (
        f"{SECRET}/type",
        _ir(
            "x",
            replayable=False,
            exportable=False,
            context={"replay_excluded": f"{SECRET}/reason", **context},
            outputs=(),
        ),
    )


def test_the_refusal_text_never_carries_step_metadata(tmp_path):
    exporter = _exporter(tmp_path, NORMAL, _hostile(replay_incomplete=True))
    with pytest.raises(ValueError) as caught:
        exporter.export("nb", filter_strategy="replay")
    assert str(caught.value) == REFUSAL


@pytest.mark.parametrize("mode", ["successful", "replay"])
def test_omitted_lists_use_fixed_labels_only(tmp_path, mode):
    path = _exporter(tmp_path, NORMAL, _hostile()).export("nb", filter_strategy=mode)
    text = _text(path)
    assert SECRET not in text
    assert "`custom step (excluded)`" in text


def test_a_known_step_with_an_unknown_reason_gets_the_fixed_fallback(tmp_path):
    step = (
        "execute_custom_code",
        _ir(
            "x",
            replayable=False,
            exportable=False,
            context={"replay_excluded": "something_new"},
            outputs=(),
        ),
    )
    text = _text(_exporter(tmp_path, NORMAL, step).export("nb"))
    assert "`execute_custom_code (excluded)`" in text and "something_new" not in text


FAILED_REFUSAL = (
    "Cannot export a replay notebook: the session has failed steps "
    "(custom step (excluded)) that may have changed data before failing. "
    "Re-run the analysis without them, or export with the default mode."
)


def _hostile_exporter(tmp_path, *, failed, exportable=False):
    kind, ir = _hostile()
    ir.exportable = exportable
    exporter = _exporter(tmp_path, NORMAL, (kind, ir))
    step = exporter.provenance.activities[-1]
    step["description"] = "step"
    if failed:
        step["error"] = "failed"
    return exporter


def test_a_failed_step_refusal_never_carries_step_metadata(tmp_path):
    with pytest.raises(ValueError) as caught:
        _hostile_exporter(tmp_path, failed=True).export("nb", filter_strategy="replay")
    assert str(caught.value) == FAILED_REFUSAL
    assert not (tmp_path / "notebooks" / "nb.ipynb").exists()


def test_a_failed_step_is_listed_with_fixed_labels(tmp_path):
    text = _text(_hostile_exporter(tmp_path, failed=True).export("nb"))
    assert SECRET not in text
    assert "`custom step (excluded)`" in text


def test_a_selected_step_with_an_exclusion_is_documented_with_a_fixed_label(
    tmp_path,
):
    exporter = _hostile_exporter(tmp_path, failed=False, exportable=True)
    text = _text(exporter.export("nb"))
    assert SECRET not in text
    assert ": custom step\n" in text
