"""Unit tests for the notebook replay checker (no container, no notebook run).

Each test builds a small fake run on disk (a capture with its reference copy,
a staged notebook and root, outputs, receipts) and changes one thing, then
checks what ``scripts/check_notebook_replay_results.py`` concludes. The
checker must never conclude MATCH or a claim from a run that does not earn it.
"""

import importlib.util
import json
import pickle  # nosec B403 # Writes and reads this test's own fixtures.
import shutil
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
import staging  # noqa: E402

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


def subprocess_timeout():
    import subprocess  # nosec B404 # only the exception type

    return subprocess.TimeoutExpired("docker", 1)


def _probe_args(**changes):
    probe = {
        "version": staging.PROBE_VERSION,
        "forbidden": sorted(
            [
                "/capture",
                "/reports",
                "/src",
                "/w/capture/oracle/final.pkl",
                "/w/capture/capture.json",
            ]
        ),
        "control": ["/stage/roots/input.h5ad"],
    }
    probe.update(changes)
    return probe


def _sha(data):
    import hashlib

    return hashlib.sha256(data).hexdigest()


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
        notebooks = {}
        for mode in ("successful", "replay"):
            path = self.capture / "notebooks" / f"{mode}.ipynb"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({"cells": [], "mode": mode}))
            notebooks[mode] = {
                "file": f"notebooks/{mode}.ipynb",
                "sha256": checker.sha256_file(path),
            }
        (self.stage / "roots").mkdir(parents=True)
        (self.stage / "roots" / "input.h5ad").write_bytes(b"root")
        mode = self.spec["mode"]
        self.probe = _probe_args() if self.spec.get("probe") else None
        (self.stage / "notebook.ipynb").write_text(
            staging.staged_notebook(
                (self.capture / "notebooks" / f"{mode}.ipynb").read_text(), self.probe
            )
        )
        tools = [{"tool": "t", "arguments": {"a": 1}}]
        self.parameters = {
            "input_data": "/stage/roots/input.h5ad",
            "output_prefix": "/outputs/results",
        }
        _write(
            self.capture / "capture.json",
            {
                "scenario": "t",
                "roots": {"input.h5ad": _sha(b"root")},
                "perturbed_roots": {"input.h5ad": _sha(b"perturbed")},
                "tool_parameters": tools,
                "notebooks": notebooks,
                "recipe_sha256": checker.sha256_file(self.capture / "provenance.json"),
                "declared_outputs": {
                    OUTPUT: {
                        "file": "oracle/final.pkl",
                        "sha256": checker.sha256_file(
                            self.capture / "oracle" / "final.pkl"
                        ),
                    }
                },
                "receipt": receipt,
            },
        )
        _write(
            self.stage / "stage.json",
            {
                "scenario_id": sid,
                "capture": "t",
                "mode": mode,
                "stage_root": self.spec["stage_root"],
                "probe": self.probe,
                "parameters": self.parameters,
                "output_basename": "results",
            },
        )
        _adata().write_h5ad(self.outputs / OUTPUT)
        kernel = dict(receipt, pid=4242)
        self.execution = {
            "status": "OK",
            "error": None,
            "pre_run_entries": [],
            "pre_run_receipt_entries": [],
            "unexpected_receipt_entries": [],
            "kernel_pid_after": 4242,
            "parameters": self.parameters,
        }
        _write(self.receipts / "execution.json", self.execution)
        _write(self.receipts / "kernel_receipt.json", kernel)
        _write(
            self.run_dir / "run.json",
            {
                "lane": lane,
                "source_revision": "a" * 40,
                "source_dirty": False,
                "venv_python": receipt["executable"],
                "source_packages": ["lobster-ai"] if lane == "container" else [],
            },
        )
        _write(
            self.run_dir / "env.json",
            {
                "verified_against_lock": lane == "container",
                "receipt": receipt,
                "distributions": [],
                "source_distributions": (
                    {"lobster-ai": "1.0.0"} if lane == "container" else {}
                ),
                "memory_peak_bytes": None,
            },
        )

    def update(self, path, **changes):
        data = json.loads(path.read_text())
        data.update(changes)
        _write(path, data)

    def set_kernel(self, **changes):
        kernel = json.loads((self.receipts / "kernel_receipt.json").read_text())
        kernel.update(changes)
        _write(self.receipts / "kernel_receipt.json", kernel)

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
    with pytest.raises(checker.CheckError, match="existed before the run"):
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


def test_stage_holding_other_roots_than_approved_is_refused(run):
    (run.stage / "roots" / "input.h5ad").write_bytes(b"other root")
    with pytest.raises(checker.CheckError, match="approved 'roots'"):
        run.build()


def _negative_spec():
    return {
        "capture": "t",
        "mode": "replay",
        "claim": "C-HIST",
        "policy": "t",
        "stage_root": "roots_perturbed",
        "lanes": ["container"],
        "expect": {
            "verdict": "MISMATCH",
            "execution": "OK",
            "binding_differences": ["root_sha256"],
            "outputs": [OUTPUT],
            "claim_achieved": "C-EXEC",
        },
    }


def test_perturbed_root_is_a_binding_difference(tmp_path):
    run = FakeRun(tmp_path, spec=_negative_spec())
    (run.stage / "roots" / "input.h5ad").write_bytes(b"perturbed")
    wrapper = run.build()
    assert wrapper["binding_differences"] == ["root_sha256"]
    assert _report(wrapper)["verdict"] == "MATCH"  # the outputs alone still agree
    assert _report(wrapper)["claim_achieved"] == "C-EXEC"  # but the claim is not earned


def test_changed_notebook_is_a_binding_difference(run):
    (run.stage / "notebook.ipynb").write_text('{"changed": true}')
    assert run.build()["binding_differences"] == ["notebook_sha256"]


def test_export_of_another_mode_is_a_binding_difference(run):
    shutil.copyfile(
        run.capture / "notebooks" / "successful.ipynb", run.stage / "notebook.ipynb"
    )
    assert run.build()["binding_differences"] == ["notebook_sha256"]


def test_changed_recipe_is_a_binding_difference(run):
    (run.capture / "provenance.json").write_text('[{"type": "other"}]')
    assert run.build()["binding_differences"] == ["recipe_sha256"]


def test_changed_parameters_are_a_binding_difference(run):
    params = dict(run.parameters, output_prefix="/elsewhere/results")
    run.update(run.receipts / "execution.json", parameters=params)
    assert run.build()["binding_differences"] == ["params_sha256"]


def test_kernel_from_another_environment_is_a_binding_difference(run):
    run.set_kernel(distributions_sha256="0" * 64)
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
    run = FakeRun(tmp_path, spec=_negative_spec())
    (run.stage / "roots" / "input.h5ad").write_bytes(b"perturbed")
    assert run.check()["t1"]  # equal outputs are not the expected negative
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


# --------------------------------------------------------------------------
# Review findings (R7)
# --------------------------------------------------------------------------


def test_replaced_reference_copy_is_refused_before_unpickling(run):
    """The same wrong result in the reference and the output must not MATCH."""
    wrong = _adata(shift=0.5)
    with open(run.capture / "oracle" / "final.pkl", "wb") as handle:
        pickle.dump(wrong, handle)
    wrong.write_h5ad(run.outputs / OUTPUT)
    with pytest.raises(checker.CheckError, match="changed since capture"):
        run.build()


def test_reference_bytes_are_checked_before_pickle_runs(run, monkeypatch):
    (run.capture / "oracle" / "final.pkl").write_bytes(b"not the sealed bytes")
    monkeypatch.setattr(checker.pickle, "loads", lambda data: pytest.fail("unpickled"))
    with pytest.raises(checker.CheckError, match="changed since capture"):
        run.build()


@pytest.mark.parametrize(
    "entry",
    [
        {"file": "../outside.pkl", "sha256": "0" * 64},
        {"file": "/etc/passwd", "sha256": "0" * 64},
        {"file": "oracle/final.pkl"},
        "oracle/final.pkl",
    ],
)
def test_malformed_or_escaping_reference_entries_are_refused(run, entry):
    run.update(run.capture / "capture.json", declared_outputs={OUTPUT: entry})
    with pytest.raises(checker.CheckError):
        run.build()


def test_linked_reference_path_is_refused(run):
    real = run.root / "elsewhere.pkl"
    shutil.move(run.capture / "oracle" / "final.pkl", real)
    (run.capture / "oracle" / "final.pkl").symlink_to(real)
    with pytest.raises(checker.CheckError, match="link"):
        run.build()


def test_capture_detects_an_export_that_changes_the_live_object(tmp_path):
    import capture

    policy = tmp_path / "policy.json"
    _write(policy, POLICY)
    live = _adata()
    frozen = capture.freeze(live)
    assert capture._oracle_problems(live, policy, OUTPUT, frozen) == []
    live.X[0, 0] = -1.0  # what a mutating exporter would do
    assert capture._oracle_problems(live, policy, OUTPUT, frozen)
    sealed = pickle.loads(frozen)  # nosec B301 # bytes this test just produced
    assert sealed.X[0, 0] == 0.0  # the sealed copy is unchanged


class _Lossy:
    """Pickles as a different value, so a frozen copy would lose it."""

    def __reduce__(self):
        return float, (3.5,)


def test_capture_refuses_an_unfaithful_copy_before_exporting(tmp_path):
    import capture

    policy = tmp_path / "policy.json"
    _write(policy, POLICY)
    live = _adata()
    live.uns["opaque"] = _Lossy()
    calls = []

    def export():  # an exporter that would also hide the loss
        calls.append(1)
        live.uns["opaque"] = 3.5

    with pytest.raises(capture.CaptureError, match="not faithful"):
        capture.seal_and_export(live, policy, OUTPUT, export)
    assert calls == []


def test_capture_refuses_an_export_that_changes_the_live_object(tmp_path):
    import capture

    policy = tmp_path / "policy.json"
    _write(policy, POLICY)
    live = _adata()

    def export():
        live.X[0, 0] = -1.0

    with pytest.raises(capture.CaptureError, match="exporting changed"):
        capture.seal_and_export(live, policy, OUTPUT, export)


def test_capture_seals_then_exports_once(tmp_path):
    import capture

    policy = tmp_path / "policy.json"
    _write(policy, POLICY)
    live = _adata()
    calls = []
    frozen, exported = capture.seal_and_export(
        live, policy, OUTPUT, lambda: calls.append(1) or "nb"
    )
    assert calls == [1] and exported == "nb"
    assert capture._oracle_problems(live, policy, OUTPUT, frozen) == []


@pytest.mark.parametrize(
    "field,value",
    [("capture", "other"), ("mode", "successful"), ("stage_root", "roots_perturbed")],
)
def test_stage_must_be_the_approved_capture_mode_and_root(run, field, value):
    run.update(run.stage / "stage.json", **{field: value})
    with pytest.raises(checker.CheckError, match=f"staged {field}"):
        run.build()


def test_capture_of_another_scenario_is_refused(run):
    run.update(run.capture / "capture.json", scenario="other")
    with pytest.raises(checker.CheckError, match="capture is 'other'"):
        run.build()


def test_changed_captured_export_is_refused(run):
    (run.capture / "notebooks" / "replay.ipynb").write_text('{"cells": [1]}')
    with pytest.raises(checker.CheckError, match="captured export changed"):
        run.build()


def test_only_the_probe_scenario_may_carry_a_probe(run):
    run.update(run.stage / "stage.json", probe=_probe_args())
    with pytest.raises(checker.CheckError, match="only the probe"):
        run.build()


@pytest.mark.parametrize(
    "changes",
    [
        {"forbidden": []},
        {"forbidden": ["/capture", "/reports", "/src"]},  # not the reference itself
        {
            "forbidden": sorted(
                ["/reports", "/src", "/w/oracle/final.pkl", "/w/capture.json"]
            )
        },
        {"control": []},
        {"control": ["/etc/hostname"]},
        {"version": 2},
        {"extra": "x"},
    ],
)
def test_probe_arguments_must_make_a_meaningful_probe(tmp_path, changes):
    run = FakeRun(tmp_path, spec=_probe_spec())
    run.update(run.stage / "stage.json", probe=_probe_args(**changes))
    with pytest.raises(checker.CheckError):
        run.build()


def test_probe_control_must_be_the_staged_input(tmp_path):
    run = FakeRun(tmp_path, spec=_probe_spec())
    run.update(
        run.stage / "stage.json",
        probe=_probe_args(control=["/stage/roots/other/roots/input.h5ad"]),
    )
    with pytest.raises(checker.CheckError, match="probe control"):
        run.build()


def test_a_forged_success_probe_cell_is_a_binding_difference(tmp_path):
    """A cell that writes a clean result without checking anything."""
    run = FakeRun(tmp_path, spec=_probe_spec())
    _write(
        run.outputs / "results_visibility.json",
        {"forbidden_visible": [], "control_visible": True},
    )
    forged = staging.PROBE_HEADER + (
        "import json\n"
        "with open(output_prefix + '_visibility.json', 'w') as h:\n"
        "    json.dump({'forbidden_visible': [], 'control_visible': True}, h)\n"
    )
    data = json.loads((run.capture / "notebooks" / "replay.ipynb").read_text())
    data["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": forged,
        }
    )
    (run.stage / "notebook.ipynb").write_text(json.dumps(data, indent=1))
    assert run.build()["binding_differences"] == ["notebook_sha256"]


def test_probe_notebook_must_be_the_export_plus_its_cell(tmp_path):
    run = FakeRun(tmp_path, spec=_probe_spec())
    _write(
        run.outputs / "results_visibility.json",
        {"forbidden_visible": [], "control_visible": True},
    )
    assert run.build()["binding_differences"] == []
    data = json.loads((run.stage / "notebook.ipynb").read_text())
    data["cells"].insert(0, data["cells"][-1])  # an extra cell elsewhere
    (run.stage / "notebook.ipynb").write_text(json.dumps(data, indent=1))
    assert run.build()["binding_differences"] == ["notebook_sha256"]


@pytest.mark.parametrize(
    "before,after",
    [(None, None), (4242, None), (True, True), ("4242", "4242"), (0, 0), (-1, -1)],
)
def test_missing_or_malformed_kernel_witnesses_block_the_claim(run, before, after):
    run.set_kernel(pid=before)
    run.update(run.receipts / "execution.json", kernel_pid_after=after)
    wrapper = run.build()
    assert (
        "the receipted kernel did not run the whole notebook"
        in wrapper["receipt_problems"]
    )
    assert _report(wrapper)["claim_achieved"] is None


def test_receipt_from_another_kernel_blocks_the_claim(run):
    run.update(run.receipts / "execution.json", kernel_pid_after=999)
    wrapper = run.build()
    assert (
        "the receipted kernel did not run the whole notebook"
        in wrapper["receipt_problems"]
    )
    assert _report(wrapper)["claim_achieved"] is None


def test_foreign_files_in_the_receipt_directory_block_the_claim(run):
    run.update(
        run.receipts / "execution.json", unexpected_receipt_entries=["forged.json"]
    )
    assert run.build()["receipt_problems"]


def test_receipts_present_before_the_run_are_refused(run):
    run.update(
        run.receipts / "execution.json", pre_run_receipt_entries=["kernel_receipt.json"]
    )
    with pytest.raises(checker.CheckError, match="existed before the run"):
        run.build()


@pytest.mark.parametrize(
    "changes,problem",
    [
        ({"verified_against_lock": False}, "not verified against the lock"),
        ({"source_distributions": {}}, "source packages"),
        (
            {"source_distributions": {"lobster-ai": "1", "other": "1"}},
            "source packages",
        ),
    ],
)
def test_unverified_environment_earns_no_claim(run, changes, problem):
    run.update(run.run_dir / "env.json", **changes)
    wrapper = run.build()
    assert any(problem in p for p in wrapper["receipt_problems"]), wrapper[
        "receipt_problems"
    ]
    assert _report(wrapper)["claim_achieved"] is None


def test_environment_record_without_its_verification_field_fails(run):
    env = json.loads((run.run_dir / "env.json").read_text())
    del env["verified_against_lock"]
    _write(run.run_dir / "env.json", env)
    assert run.build()["receipt_problems"]


def test_kernel_and_capture_agreeing_against_the_setup_environment_fails(run):
    run.set_kernel(version="3.0.0 (other)")
    kernel = json.loads((run.receipts / "kernel_receipt.json").read_text())
    kernel.pop("pid")
    run.update(run.capture / "capture.json", receipt=kernel)
    wrapper = run.build()
    assert wrapper["binding_differences"] == []
    assert (
        "kernel environment differs from the verified environment"
        in wrapper["receipt_problems"]
    )


def test_unisolated_lane_does_not_need_a_lock_verified_environment(tmp_path):
    run = FakeRun(tmp_path, lane="unisolated")
    assert run.build()["receipt_problems"] == []


# -- executing in one kernel ---------------------------------------------


def test_harness_cells_wrap_the_export_without_changing_it(tmp_path):
    import in_boundary
    import nbformat

    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_code_cell("x = 1"),
        nbformat.v4.new_markdown_cell("m"),
    ]
    path = tmp_path / "n.ipynb"
    nbformat.write(notebook, str(path))
    before = path.read_bytes()
    wrapped = in_boundary.instrumented(path)
    assert [c.source for c in wrapped.cells[1:-1]] == ["x = 1", "m"]
    assert all(
        "replay-harness" in c.metadata["tags"]
        for c in (wrapped.cells[0], wrapped.cells[-1])
    )
    assert path.read_bytes() == before


def test_receipt_cell_reports_its_own_process(capsys):
    import os

    import in_boundary

    exec(in_boundary.RECEIPT_CELL, {})  # nosec B102 # harness constant
    exec(in_boundary.FINAL_CELL, {})  # nosec B102 # harness constant
    printed = capsys.readouterr().out
    cell = {"outputs": [{"output_type": "stream", "name": "stdout", "text": printed}]}
    first = in_boundary.harness_output(cell)
    assert first["pid"] == os.getpid()
    assert set(harness_receipt.ENVIRONMENT_FIELDS) <= set(first)
    assert in_boundary.harness_output({"outputs": []}) is None


# -- teardown on every exit ----------------------------------------------


class _Proc:
    def __init__(self, wait):
        self._wait = wait
        self.pid = 4321
        self.killed = False

    def wait(self, timeout=None):
        if isinstance(self._wait, BaseException):
            raise self._wait
        return self._wait

    def poll(self):
        return None if isinstance(self._wait, BaseException) and not self.killed else 0

    def kill(self):
        self.killed = True
        self._wait = 0


def _lane(monkeypatch, tmp_path, proc, listing=("",), listing_code=0, raises=None):
    import run_lane

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if raises is not None and cmd[:2] == raises[0]:
            raise raises[1]
        if cmd[:2] == ["docker", "ps"]:
            out = listing[
                min(
                    len([c for c in calls if c[:2] == ["docker", "ps"]]) - 1,
                    len(listing) - 1,
                )
            ]
            return type("Done", (), {"returncode": listing_code, "stdout": out})()
        return type("Done", (), {"returncode": 0, "stdout": ""})()

    monkeypatch.setattr(run_lane.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(run_lane.subprocess, "run", fake_run)
    monkeypatch.setattr(run_lane.time, "sleep", lambda s: None)
    (tmp_path / "logs").mkdir()
    return run_lane, run_lane.Runner("container", tmp_path), calls


@pytest.mark.parametrize("outcome", [0, 3, KeyboardInterrupt("cancelled")])
def test_container_is_removed_and_confirmed_gone_on_every_exit(
    monkeypatch, tmp_path, outcome
):
    proc = _Proc(outcome)
    run_lane, runner, calls = _lane(monkeypatch, tmp_path, proc)
    if isinstance(outcome, BaseException):
        with pytest.raises(KeyboardInterrupt):
            runner.run("capture", ["true"], [], check=False)
        assert proc.killed
    else:
        assert runner.run("capture", ["true"], [], check=False) == outcome
    assert ["docker", "rm", "--force"] == calls[0][:3]
    assert calls[1][:2] == ["docker", "ps"] and "name=^/" in calls[1][-1]


def test_timeout_removes_the_container_and_fails(monkeypatch, tmp_path):
    proc = _Proc(subprocess_timeout())
    run_lane, runner, calls = _lane(monkeypatch, tmp_path, proc)
    with pytest.raises(run_lane.LaneError, match="exit code 124"):
        runner.run("capture", ["true"], [])
    assert ["docker", "rm", "--force"] == calls[0][:3]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"listing": ("abc123",)},  # still listed
        {"listing_code": 1},  # daemon unreachable: unknown, not gone
        {"raises": (["docker", "ps"], subprocess_timeout())},  # listing hangs
    ],
)
def test_unconfirmed_removal_fails_the_lane_and_reaps_the_client(
    monkeypatch, tmp_path, kwargs
):
    proc = _Proc(KeyboardInterrupt("cancelled"))
    run_lane, runner, calls = _lane(monkeypatch, tmp_path, proc, **kwargs)
    with pytest.raises(run_lane.LaneError, match="cannot confirm"):
        runner.run("capture", ["true"], [], check=False)
    assert proc.killed


def test_cleanup_error_still_reaps_the_client(monkeypatch, tmp_path):
    proc = _Proc(KeyboardInterrupt("cancelled"))
    run_lane, runner, calls = _lane(
        monkeypatch,
        tmp_path,
        proc,
        raises=(["docker", "rm"], OSError("docker missing")),
    )
    with pytest.raises(OSError):
        runner.run("capture", ["true"], [], check=False)
    assert proc.killed


def test_container_listed_once_then_gone_is_confirmed(monkeypatch, tmp_path):
    proc = _Proc(0)
    run_lane, runner, calls = _lane(monkeypatch, tmp_path, proc, listing=("abc", ""))
    assert runner.run("capture", ["true"], [], check=False) == 0


@pytest.mark.parametrize("outcome", [0, KeyboardInterrupt("cancelled")])
def test_host_phase_kills_its_whole_process_group(monkeypatch, tmp_path, outcome):
    import run_lane

    killed = []
    proc = _Proc(outcome)
    monkeypatch.setattr(run_lane.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(run_lane.os, "killpg", lambda pid, sig: killed.append(pid))
    (tmp_path / "logs").mkdir()
    runner = run_lane.Runner("unisolated", tmp_path, python=sys.executable)
    if isinstance(outcome, BaseException):
        with pytest.raises(KeyboardInterrupt):
            runner.run("capture", ["true"], [], check=False)
    else:
        runner.run("capture", ["true"], [], check=False)
    assert killed == [proc.pid]
