"""Unit tests for the notebook replay checker (no container, no notebook run).

Each test builds a small fake run on disk (a capture with its reference copy,
a staged notebook and root, outputs, receipts) and changes one thing, then
checks what ``scripts/check_notebook_replay_results.py`` concludes. The
checker must never conclude MATCH or a claim from a run that does not earn it.
"""

import importlib.util
import json
import pickle  # nosec B403 # Writes and reads this test's own fixtures.
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
HARNESS = REPO / "tests" / "integration" / "notebook_replay"
SCRIPT = REPO / "scripts" / "check_notebook_replay_results.py"

sys.path.insert(0, str(HARNESS))
import receipt as harness_receipt  # noqa: E402
import setup_env  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("check_notebook_replay_results", SCRIPT)
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)

OUTPUT = "results_final.h5ad"
POLICY = {"schema_version": 1, "required": ["files"], "excluded": [], "tolerances": []}


def _adata(shift=0.0):
    return ad.AnnData(
        X=np.arange(12, dtype=np.float32).reshape(4, 3) + shift,
        obs=pd.DataFrame(
            {"group": pd.Categorical(["a", "b", "a", "b"])}, index=list("wxyz")
        ),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )


def _kernel_receipt():
    receipt = harness_receipt.collect_receipt()
    receipt["thread_env"] = dict(harness_receipt.THREAD_ENV)
    receipt["pythonpath"] = None
    receipt["pythonhome"] = None
    receipt["user_site"] = False
    return receipt


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


class FakeRun:
    """A complete, consistent run of one scenario; tests then break one part."""

    def __init__(self, root, sid="t1", lane="container", spec=None):
        self.root = root
        self.sid = sid
        self.harness = root / "harness"
        self.capture = root / "capture"
        self.stage = root / "stage"
        self.outputs = root / "outputs"
        self.receipts = root / "receipt"
        self.run_dir = root / "run"
        self.reports = root / "reports"
        for d in (self.outputs, self.receipts, self.reports):
            d.mkdir(parents=True)
        self.spec = spec or {
            "capture": "t",
            "mode": "replay",
            "claim": "C-HIST",
            "policy": "t",
            "stage_root": "roots",
            "lanes": ["container", "unisolated"],
            "expect": {
                "verdict": "MATCH",
                "execution": "OK",
                "binding_differences": [],
                "outputs": [OUTPUT],
                "claim_achieved": "C-HIST",
            },
        }
        _write(self.harness / "policies" / "t.json", POLICY)
        _write(
            self.harness / "expectations.json",
            {"schema_version": 1, "scenarios": {sid: self.spec}},
        )
        receipt = _kernel_receipt()
        (self.capture / "oracle").mkdir(parents=True)
        with open(self.capture / "oracle" / "final.pkl", "wb") as handle:
            pickle.dump(_adata(), handle)
        (self.capture / "provenance.json").write_text("[]")
        (self.stage / "roots").mkdir(parents=True)
        (self.stage / "roots" / "input.h5ad").write_bytes(b"root")
        (self.stage / "notebook.ipynb").write_text("{}")
        tools = [{"tool": "t", "arguments": {"a": 1}}]
        self.parameters = {
            "input_data": "/stage/roots/input.h5ad",
            "output_prefix": "/outputs/results",
        }
        _write(
            self.capture / "capture.json",
            {
                "roots": {
                    "input.h5ad": checker.sha256_file(
                        self.stage / "roots" / "input.h5ad"
                    )
                },
                "tool_parameters": tools,
                "recipe_sha256": checker.sha256_file(self.capture / "provenance.json"),
                "declared_outputs": {OUTPUT: "oracle/final.pkl"},
                "receipt": receipt,
            },
        )
        _write(
            self.stage / "stage.json",
            {
                "scenario_id": sid,
                "notebook_sha256": checker.sha256_file(self.stage / "notebook.ipynb"),
                "parameters": self.parameters,
                "output_basename": "results",
            },
        )
        _adata().write_h5ad(self.outputs / OUTPUT)
        self.execution = {
            "status": "OK",
            "error": None,
            "pre_run_entries": [],
            "parameters": self.parameters,
        }
        _write(self.receipts / "execution.json", self.execution)
        _write(self.receipts / "kernel_receipt.json", receipt)
        _write(
            self.run_dir / "run.json",
            {
                "lane": lane,
                "source_revision": "a" * 40,
                "source_dirty": False,
                "venv_python": receipt["executable"],
            },
        )
        _write(self.run_dir / "env.json", {"receipt": receipt})

    def update(self, path, **changes):
        data = json.loads(path.read_text())
        data.update(changes)
        _write(path, data)

    def build(self):
        return checker.build_report(
            self.sid,
            self.spec,
            harness=self.harness,
            capture=self.capture,
            stage=self.stage,
            outputs=self.outputs,
            receipt_dir=self.receipts,
            run_dir=self.run_dir,
        )

    def check(self, wrapper=None, lane="container"):
        wrapper = self.build() if wrapper is None else wrapper
        (self.reports / f"{self.sid}.json").write_text(json.dumps(wrapper))
        return dict(checker.run_check(self.harness, self.reports, lane))


@pytest.fixture
def run(tmp_path):
    return FakeRun(tmp_path)


def _report(wrapper):
    return wrapper["replay_report"]


# --------------------------------------------------------------------------
# compare: what the checker concludes from the files
# --------------------------------------------------------------------------


def test_consistent_run_matches_and_earns_its_claim(run):
    wrapper = run.build()
    assert _report(wrapper)["verdict"] == "MATCH"
    assert _report(wrapper)["claim_achieved"] == "C-HIST"
    assert _report(wrapper)["verification_status"] == "checked"
    assert wrapper["binding_differences"] == [] and wrapper["receipt_problems"] == []
    assert run.check() == {"t1": []}


def test_unisolated_lane_earns_no_claim(tmp_path):
    run = FakeRun(tmp_path, lane="unisolated")
    wrapper = run.build()
    assert _report(wrapper)["verdict"] == "MATCH"
    assert _report(wrapper)["claim_achieved"] is None
    assert run.check(lane="unisolated") == {"t1": []}
    assert run.check(wrapper, lane="container")["t1"]  # a lane cannot pass as another


def test_changed_output_value_is_a_mismatch(run):
    _adata(shift=0.5).write_h5ad(run.outputs / OUTPUT)
    wrapper = run.build()
    assert _report(wrapper)["verdict"] == "MISMATCH"
    assert _report(wrapper)["claim_achieved"] == "C-EXEC"
    assert "verdict MISMATCH, expected MATCH" in run.check(wrapper)["t1"]


def test_missing_output_is_a_mismatch(run):
    (run.outputs / OUTPUT).unlink()
    assert _report(run.build())["verdict"] == "MISMATCH"


def test_extra_output_is_a_mismatch(run):
    (run.outputs / "unexpected.csv").write_text("x\n1\n")
    wrapper = run.build()
    assert _report(wrapper)["verdict"] == "MISMATCH"
    assert any("outputs" in f for f in run.check(wrapper)["t1"])


def test_linked_output_is_never_followed(run):
    (run.outputs / OUTPUT).unlink()
    elsewhere = run.root / "elsewhere.h5ad"
    _adata().write_h5ad(elsewhere)
    (run.outputs / OUTPUT).symlink_to(elsewhere)
    assert _report(run.build())["verdict"] == "MISMATCH"


def test_stale_outputs_are_refused(run):
    run.update(run.receipts / "execution.json", pre_run_entries=[OUTPUT])
    with pytest.raises(checker.CheckError, match="not empty before the run"):
        run.build()


def test_unfinished_execution_is_refused(run):
    run.update(run.receipts / "execution.json", status="HARNESS_ERROR")
    with pytest.raises(checker.CheckError, match="did not complete"):
        run.build()


def test_stage_of_another_scenario_is_refused(run):
    run.update(run.stage / "stage.json", scenario_id="other")
    with pytest.raises(checker.CheckError, match="belongs to"):
        run.build()


@pytest.mark.parametrize("status", ["ERROR", "TIMEOUT"])
def test_failed_execution_is_not_compared(run, status):
    run.update(run.receipts / "execution.json", status=status)
    wrapper = run.build()
    assert _report(wrapper)["verdict"] == "NOT_COMPARED"
    assert _report(wrapper)["slots"] == []
    assert _report(wrapper)["claim_achieved"] is None


def test_changed_root_is_a_binding_difference(run):
    (run.stage / "roots" / "input.h5ad").write_bytes(b"other root")
    wrapper = run.build()
    assert wrapper["binding_differences"] == ["root_sha256"]
    assert _report(wrapper)["verdict"] == "MATCH"  # the outputs alone still agree
    assert _report(wrapper)["claim_achieved"] == "C-EXEC"  # but the claim is not earned


def test_changed_notebook_is_a_binding_difference(run):
    (run.stage / "notebook.ipynb").write_text('{"changed": true}')
    assert run.build()["binding_differences"] == ["notebook_sha256"]


def test_changed_recipe_is_a_binding_difference(run):
    (run.capture / "provenance.json").write_text('[{"type": "other"}]')
    assert run.build()["binding_differences"] == ["recipe_sha256"]


def test_changed_parameters_are_a_binding_difference(run):
    params = dict(run.parameters, output_prefix="/elsewhere/results")
    run.update(run.receipts / "execution.json", parameters=params)
    assert run.build()["binding_differences"] == ["params_sha256"]


def test_kernel_from_another_environment_is_a_binding_difference(run):
    kernel = _kernel_receipt()
    kernel["distributions_sha256"] = "0" * 64
    _write(run.receipts / "kernel_receipt.json", kernel)
    wrapper = run.build()
    assert "env_receipt_sha256" in wrapper["binding_differences"]
    assert wrapper["receipt_problems"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("executable", "/elsewhere/bin/python"),
        ("distributions_sha256", "0" * 64),
        ("thread_env", {"OMP_NUM_THREADS": "8"}),
        ("pythonpath", "/src"),
        ("pythonhome", "/opt"),
        ("user_site", True),
    ],
)
def test_kernel_environment_problems_block_the_claim(run, field, value):
    kernel = _kernel_receipt()
    kernel[field] = value
    _write(run.receipts / "kernel_receipt.json", kernel)
    _write(
        run.capture / "capture.json",
        dict(json.loads((run.capture / "capture.json").read_text()), receipt=kernel),
    )
    wrapper = run.build()
    assert wrapper["receipt_problems"]
    assert _report(wrapper)["claim_achieved"] is None
    assert any("environment receipt" in f for f in run.check(wrapper)["t1"])


@pytest.mark.parametrize("value", [object(), {"nested": object()}])
def test_capture_refuses_a_reference_copy_it_cannot_fully_compare(tmp_path, value):
    import capture

    policy = tmp_path / "policy.json"
    _write(policy, POLICY)
    final = _adata()
    assert capture._oracle_problems(final, policy, OUTPUT) == []
    final.uns["opaque"] = value
    assert capture._oracle_problems(final, policy, OUTPUT)


# --------------------------------------------------------------------------
# check: reports against the approved expectations
# --------------------------------------------------------------------------


def test_expected_negative_passes_and_unexpected_verdict_fails(tmp_path):
    spec = {
        "capture": "t",
        "mode": "replay",
        "claim": "C-HIST",
        "policy": "t",
        "stage_root": "roots",
        "lanes": ["container"],
        "expect": {
            "verdict": "MISMATCH",
            "execution": "OK",
            "binding_differences": ["root_sha256"],
            "outputs": [OUTPUT],
            "claim_achieved": "C-EXEC",
        },
    }
    run = FakeRun(tmp_path, spec=spec)
    assert run.check()["t1"]  # a matching run is not the expected negative
    (run.stage / "roots" / "input.h5ad").write_bytes(b"perturbed")
    _adata(shift=1.0).write_h5ad(run.outputs / OUTPUT)
    assert run.check() == {"t1": []}


def test_every_expected_scenario_needs_exactly_one_report(run):
    assert checker.run_check(run.harness, run.reports, "container") == [
        ("t1", ["no report for this scenario"])
    ]
    run.check()
    (run.reports / "extra.json").write_text("{}")
    results = dict(checker.run_check(run.harness, run.reports, "container"))
    assert results["t1"] == [] and results["unexpected-reports"]


def test_report_for_another_scenario_fails(run):
    wrapper = run.build()
    wrapper["replay_report"]["scenario_id"] = "t2"
    assert any("not 't1'" in f for f in run.check(wrapper)["t1"])


def test_report_claiming_a_verdict_its_slots_do_not_support_fails(run):
    _adata(shift=0.5).write_h5ad(run.outputs / OUTPUT)
    wrapper = run.build()
    wrapper["replay_report"]["verdict"] = "MATCH"
    assert any("invalid report" in f for f in run.check(wrapper)["t1"])


def test_report_compared_with_a_weaker_policy_fails(run):
    wrapper = run.build()
    wrapper["replay_report"]["slots"][0]["atol"] = 1.0
    assert any("invalid report" in f for f in run.check(wrapper)["t1"])


def test_report_with_unknown_fields_fails(run):
    wrapper = run.build()
    wrapper["verdict_override"] = "MATCH"
    assert run.check(wrapper)["t1"]


def test_report_with_extra_binding_difference_fails(run):
    wrapper = run.build()
    wrapper["binding_differences"] = ["root_sha256"]
    assert run.check(wrapper)["t1"]


def _probe_spec():
    return {
        "capture": "t",
        "mode": "replay",
        "claim": "C-EXEC",
        "policy": "t",
        "stage_root": "roots",
        "probe": True,
        "lanes": ["container"],
        "expect": {
            "verdict": "NOT_COMPARED",
            "execution": "OK",
            "binding_differences": [],
            "outputs": [OUTPUT, "results_visibility.json"],
            "claim_achieved": "C-EXEC",
        },
    }


@pytest.mark.parametrize(
    "probe,passes",
    [
        ({"forbidden_visible": [], "control_visible": True}, True),
        ({"forbidden_visible": ["/capture"], "control_visible": True}, False),
        ({"forbidden_visible": [], "control_visible": False}, False),
        ({"forbidden_visible": []}, False),
    ],
)
def test_isolation_probe(tmp_path, probe, passes):
    run = FakeRun(tmp_path, spec=_probe_spec())
    _write(run.outputs / "results_visibility.json", probe)
    wrapper = run.build()
    assert _report(wrapper)["slots"] == []
    assert _report(wrapper)["verdict"] == "NOT_COMPARED"
    assert (run.check(wrapper)["t1"] == []) is passes


def test_probe_scenario_without_probe_output_fails(tmp_path):
    run = FakeRun(tmp_path, spec=_probe_spec())
    with pytest.raises(checker.CheckError, match="missing file"):
        run.build()


def test_expectations_reject_unknown_fields(run):
    data = json.loads((run.harness / "expectations.json").read_text())
    data["scenarios"]["t1"]["tolerance_override"] = 1
    _write(run.harness / "expectations.json", data)
    with pytest.raises(checker.CheckError, match="unknown fields"):
        checker.load_expectations(run.harness / "expectations.json")


def test_approved_expectations_cover_the_required_scenarios():
    specs = checker.load_expectations(HARNESS / "expectations.json")
    container = {sid for sid, spec in specs.items() if "container" in spec["lanes"]}
    assert {
        "s1-successful",
        "s1-replay",
        "s1-incorrect-root",
        "s1-isolation-probe",
    } <= container
    assert specs["s1-incorrect-root"]["expect"]["verdict"] == "MISMATCH"
    assert specs["s1-isolation-probe"].get("probe") is True
    for sid, spec in specs.items():
        checker.load_policy(HARNESS, spec["policy"])
        if spec["expect"]["verdict"] == "MATCH":
            assert spec["expect"]["claim_achieved"] == spec["claim"], sid


def test_junit_lists_every_scenario(run, tmp_path):
    run.check()
    junit = tmp_path / "junit.xml"
    checker.run_check(run.harness, run.reports, "container", junit)
    text = junit.read_text()
    assert 'tests="1"' in text and 'failures="0"' in text and 'name="t1"' in text


# --------------------------------------------------------------------------
# setup: the environment must be the lock
# --------------------------------------------------------------------------


def _pins(tmp_path, text):
    path = tmp_path / "constraints.txt"
    path.write_text(text)
    return setup_env.read_pins(path)


def test_environment_matching_the_lock_passes(tmp_path):
    pins = _pins(
        tmp_path,
        "numpy==2.2.6\n    # via x\nscipy==1.15.0 ; python_version >= '3.12'\n",
    )
    found = {"numpy": "2.2.6", "scipy": "1.15.0", "lobster-ai": "1.0.0"}
    assert setup_env.check(found, pins, {"lobster-ai": "1.0.0"}) == []


@pytest.mark.parametrize(
    "found,problem",
    [
        ({"numpy": "2.2.5", "lobster-ai": "1.0.0"}, "differs from the lock"),
        (
            {"numpy": "2.2.6", "requests": "2.0", "lobster-ai": "1.0.0"},
            "not in the lock",
        ),
        ({"numpy": "2.2.6", "lobster-ai": "0.9.0"}, "not the source tree's"),
        ({"numpy": "2.2.6"}, "is not installed"),
    ],
)
def test_environment_differing_from_the_lock_fails(tmp_path, found, problem):
    pins = _pins(tmp_path, "numpy==2.2.6\n")
    problems = setup_env.check(found, pins, {"lobster-ai": "1.0.0"})
    assert any(problem in p for p in problems), problems


def test_receipt_code_matches_the_harness_receipt():
    receipt = harness_receipt.collect_receipt()
    assert set(harness_receipt.ENVIRONMENT_FIELDS) <= set(receipt)
    assert harness_receipt.environment_digest(
        receipt
    ) == harness_receipt.environment_digest(dict(receipt))


def test_harness_scripts_are_not_collected_as_tests():
    names = sorted(p.name for p in HARNESS.glob("*.py"))
    assert not [n for n in names if n.startswith("test_")]
