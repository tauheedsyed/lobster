#!/usr/bin/env python3
"""Compare and check notebook replay runs (the replay lane's checker).

Two subcommands:

``compare``
    Builds one scenario's report from the files alone. It recomputes every
    binding (notebook, recipe, roots, policy, parameters, source, environment
    receipt) from what was staged and executed, compares the declared outputs
    with the live reference copy under the approved policy, and writes a report.
    Nothing the notebook wrote is trusted as a verdict; the notebook never sees
    this checker, the reference copy or the report directory.

``check``
    Validates every report of a lane against the approved expectations: each
    expected scenario exactly once, the expected verdict, execution state,
    binding differences, environment receipt and isolation probe. Writes a
    JUnit file and exits non-zero on any failure. There is no "everything
    matched" shortcut: a negative scenario passes only with its own expected,
    non-MATCH outcome.

A report is a JSON object holding a ``replay_report`` (the contract in
``lobster.testing.replay_contract``) plus the checker's own findings.
"""

import argparse
import hashlib
import json
import pickle  # nosec B403 # Reads the harness's own reference copy only.
import sys
import xml.etree.ElementTree as ET  # nosec B405 # Writes JUnit; parses nothing.
from pathlib import Path

WRAPPER_VERSION = 1
BACKEND = "lobster-notebook"
LANES = ("container", "unisolated")
EXECUTION_STATES = {"OK": "OK", "ERROR": "ERROR", "TIMEOUT": "TIMEOUT"}


class CheckError(Exception):
    """The run cannot be judged (harness or input problem)."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    from lobster.testing.replay_contract import strict_json_loads

    try:
        return strict_json_loads(Path(path).read_text())
    except FileNotFoundError:
        raise CheckError(f"missing file: {path}") from None


def load_expectations(path):
    data = read_json(path)
    if set(data) != {"schema_version", "scenarios"} or data["schema_version"] != 1:
        raise CheckError("expectations need exactly 'schema_version' 1 and 'scenarios'")
    allowed = {
        "capture",
        "mode",
        "claim",
        "policy",
        "stage_root",
        "probe",
        "lanes",
        "expect",
    }
    for sid, spec in data["scenarios"].items():
        unknown = set(spec) - allowed
        if unknown:
            raise CheckError(f"{sid}: unknown fields {sorted(unknown)}")
        if not set(spec.get("lanes", [])) <= set(LANES) or not spec.get("lanes"):
            raise CheckError(f"{sid}: lanes must be a non-empty subset of {LANES}")
    return data["scenarios"]


def load_policy(harness, name):
    from lobster.testing.replay_contract import ComparisonPolicy

    return ComparisonPolicy.from_json(
        (Path(harness) / "policies" / f"{name}.json").read_text()
    )


def environment_digest(receipt):
    from receipt import environment_digest as digest  # harness module

    return digest(receipt)


def receipt_problems(kernel, env, run):
    """Ways the kernel's environment differs from the verified environment."""
    from receipt import THREAD_ENV

    problems = []
    if kernel["executable"] != run["venv_python"]:
        problems.append(
            f"kernel interpreter {kernel['executable']} is not {run['venv_python']}"
        )
    if kernel["distributions_sha256"] != env["receipt"]["distributions_sha256"]:
        problems.append("kernel distributions differ from the verified environment")
    if kernel["thread_env"] != THREAD_ENV:
        problems.append(f"kernel thread settings differ: {kernel['thread_env']}")
    for key in ("pythonpath", "pythonhome"):
        if kernel[key] is not None:
            problems.append(f"kernel has {key.upper()} set")
    if kernel["user_site"]:
        problems.append("kernel has user site-packages enabled")
    return problems


def _load_outputs(outputs):
    """Output tree with H5AD files loaded; links and other files stay files."""
    import anndata

    from lobster.testing.replay_compare import OutputFile, output_files

    def load(node):
        if isinstance(node, dict):
            return {key: load(value) for key, value in node.items()}
        if isinstance(node, OutputFile) and not node.path.is_symlink():
            if node.path.suffix == ".h5ad":
                return anndata.read_h5ad(node.path)
        return node

    return load(output_files(outputs, anchor=outputs))


def _inventory(outputs):
    return sorted(str(p.relative_to(outputs)) for p in Path(outputs).rglob("*"))


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------


def build_report(sid, spec, *, harness, capture, stage, outputs, receipt_dir, run_dir):
    """The report for one executed scenario, recomputed from files only."""
    from lobster.testing.replay_compare import compare
    from lobster.testing.replay_contract import (
        BINDING_KEYS,
        Claim,
        Execution,
        Preflight,
        ReplayReport,
        SlotStatus,
        Verdict,
        VerificationStatus,
        aggregate_verdict,
        binding_differences,
        canonical_sha256,
    )

    capture, stage, outputs = Path(capture), Path(stage), Path(outputs)
    receipt_dir, run_dir = Path(receipt_dir), Path(run_dir)
    run = read_json(run_dir / "run.json")
    env = read_json(run_dir / "env.json")
    cap = read_json(capture / "capture.json")
    staged = read_json(stage / "stage.json")
    execution = read_json(receipt_dir / "execution.json")
    policy = load_policy(harness, spec["policy"])
    if staged["scenario_id"] != sid:
        raise CheckError(f"stage belongs to {staged['scenario_id']!r}, not {sid!r}")
    if execution["status"] not in EXECUTION_STATES:
        raise CheckError(f"execution did not complete normally: {execution}")
    if execution["pre_run_entries"]:
        raise CheckError(
            f"outputs were not empty before the run: {execution['pre_run_entries']}"
        )

    kernel = read_json(receipt_dir / "kernel_receipt.json")
    roots = {p.name: sha256_file(p) for p in sorted((stage / "roots").iterdir())}
    tools = cap["tool_parameters"]
    recomputed = {
        "notebook_sha256": sha256_file(stage / "notebook.ipynb"),
        "recipe_sha256": sha256_file(capture / "provenance.json"),
        "root_sha256": roots,
        "policy_sha256": policy.digest(),
        "params_sha256": canonical_sha256(
            {"papermill": execution["parameters"], "tools": tools}
        ),
        "source_revision": run["source_revision"],
        "source_dirty": run["source_dirty"],
        "env_receipt_sha256": environment_digest(kernel),
    }
    declared = {
        "notebook_sha256": staged["notebook_sha256"],
        "recipe_sha256": cap["recipe_sha256"],
        "root_sha256": cap["roots"],
        "policy_sha256": policy.digest(),
        "params_sha256": canonical_sha256(
            {"papermill": staged["parameters"], "tools": tools}
        ),
        "source_revision": run["source_revision"],
        "source_dirty": run["source_dirty"],
        "env_receipt_sha256": environment_digest(cap["receipt"]),
    }
    if set(recomputed) != set(BINDING_KEYS):
        raise CheckError("the checker does not recompute every binding")
    differences = binding_differences(declared, recomputed)
    problems = receipt_problems(kernel, env, run)

    state = Execution(EXECUTION_STATES[execution["status"]])
    claim = Claim(spec["claim"])
    slots = ()
    probe = None
    if state is Execution.OK and claim is not Claim.EXEC:
        expected = {}
        for name, oracle in cap["declared_outputs"].items():
            with open(capture / oracle, "rb") as handle:
                expected[name] = pickle.load(handle)  # nosec B301
        slots = compare({"files": expected}, {"files": _load_outputs(outputs)}, policy)
    if spec.get("probe") and state is Execution.OK:
        probe = read_json(outputs / f"{staged['output_basename']}_visibility.json")

    verdict = (
        Verdict.NOT_COMPARED if state is not Execution.OK else aggregate_verdict(slots)
    )
    if claim is Claim.EXEC:
        verdict = Verdict.NOT_COMPARED
    achieved = None
    if run["lane"] == "container" and state is Execution.OK and not problems:
        if claim is Claim.EXEC:
            achieved = Claim.EXEC
        elif verdict is Verdict.MATCH and not differences:
            achieved = claim
        else:
            achieved = Claim.EXEC
    if verdict in (Verdict.MATCH, Verdict.MISMATCH):
        status = VerificationStatus.CHECKED
    elif slots:
        status = VerificationStatus.INCOMPLETE
    else:
        status = VerificationStatus.NOT_RUN
    report = ReplayReport(
        backend=BACKEND,
        scenario_id=sid,
        requested_mode=spec["mode"],
        claim_requested=claim,
        claim_achieved=achieved,
        verification_status=status,
        bindings=recomputed,
        preflight=Preflight.ADMITTED,
        execution=state,
        verdict=verdict,
        slots=slots,
        excluded=policy.excluded if slots else (),
    )
    report.validate(policy)
    if slots and any(s.status is SlotStatus.NOT_COMPARED for s in slots):
        problems.append("a required slot was not compared")
    return {
        "wrapper_version": WRAPPER_VERSION,
        "lane": run["lane"],
        "replay_report": report.to_dict(),
        "binding_differences": differences,
        "receipt_problems": problems,
        "execution": execution,
        "probe": probe,
        "outputs": _inventory(outputs),
    }


def cmd_compare(args):
    specs = load_expectations(Path(args.harness) / "expectations.json")
    if args.scenario not in specs:
        raise CheckError(f"unknown scenario {args.scenario!r}")
    sys.path.insert(0, str(args.harness))
    wrapper = build_report(
        args.scenario,
        specs[args.scenario],
        harness=args.harness,
        capture=args.capture,
        stage=args.stage,
        outputs=args.outputs,
        receipt_dir=args.receipt_dir,
        run_dir=args.run_dir,
    )
    out = Path(args.out)
    if out.exists():
        raise CheckError(f"report already exists: {out}")
    out.write_text(json.dumps(wrapper, sort_keys=True, indent=1))
    report = wrapper["replay_report"]
    print(f"{args.scenario}: {report['verdict']} (execution {report['execution']})")
    return 0


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------


def check_one(sid, spec, wrapper, lane, harness):
    """Failure messages for one scenario's report (empty list: as expected)."""
    from lobster.testing.replay_contract import ReplayReport

    failures = []
    if set(wrapper) != {
        "wrapper_version",
        "lane",
        "replay_report",
        "binding_differences",
        "receipt_problems",
        "execution",
        "probe",
        "outputs",
    }:
        return [f"unexpected report fields: {sorted(wrapper)}"]
    if wrapper["wrapper_version"] != WRAPPER_VERSION:
        return [f"unsupported report version {wrapper['wrapper_version']!r}"]
    if wrapper["lane"] != lane:
        failures.append(f"report is from lane {wrapper['lane']!r}, not {lane!r}")
    try:
        report = ReplayReport.from_dict(wrapper["replay_report"])
        report.validate(load_policy(harness, spec["policy"]))
    except Exception as exc:  # any contract violation fails the scenario
        return failures + [f"invalid report: {exc}"]
    if report.scenario_id != sid:
        failures.append(f"report is for {report.scenario_id!r}, not {sid!r}")
    if (
        report.requested_mode != spec["mode"]
        or report.claim_requested.value != spec["claim"]
    ):
        failures.append("report mode or claim differs from the scenario")
    expect = spec["expect"]
    if report.verdict.value != expect["verdict"]:
        failures.append(f"verdict {report.verdict.value}, expected {expect['verdict']}")
    if report.execution.value != expect["execution"]:
        failures.append(
            f"execution {report.execution.value}, expected {expect['execution']}"
        )
    if wrapper["binding_differences"] != expect["binding_differences"]:
        failures.append(
            f"binding differences {wrapper['binding_differences']}, "
            f"expected {expect['binding_differences']}"
        )
    if wrapper["receipt_problems"]:
        failures.append(f"environment receipt: {wrapper['receipt_problems']}")
    if wrapper["execution"].get("pre_run_entries"):
        failures.append("outputs existed before the run")
    if wrapper["outputs"] != expect["outputs"]:
        failures.append(f"outputs {wrapper['outputs']}, expected {expect['outputs']}")
    achieved = report.claim_achieved.value if report.claim_achieved else None
    wanted = expect["claim_achieved"] if lane == "container" else None
    if achieved != wanted:
        failures.append(f"claim achieved {achieved}, expected {wanted}")
    if spec.get("probe"):
        probe = wrapper["probe"]
        if not isinstance(probe, dict) or set(probe) != {
            "forbidden_visible",
            "control_visible",
        }:
            failures.append(f"isolation probe missing or malformed: {probe!r}")
        else:
            if probe["forbidden_visible"]:
                failures.append(f"notebook could see {probe['forbidden_visible']}")
            if probe["control_visible"] is not True:
                failures.append(
                    "probe could not see its staged input (probe is broken)"
                )
    return failures


def write_junit(path, results, lane):
    suite = ET.Element(
        "testsuite",
        name=f"notebook-replay-{lane}",
        tests=str(len(results)),
        failures=str(sum(1 for _, f in results if f)),
        errors="0",
        skipped="0",
    )
    for sid, failures in results:
        case = ET.SubElement(suite, "testcase", classname="notebook_replay", name=sid)
        if failures:
            node = ET.SubElement(case, "failure", message=failures[0][:200])
            node.text = "\n".join(failures)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def run_check(harness, reports, lane, junit=None):
    specs = load_expectations(Path(harness) / "expectations.json")
    wanted = sorted(sid for sid, spec in specs.items() if lane in spec["lanes"])
    found = sorted(p.name for p in Path(reports).iterdir())
    results = []
    extra = sorted(set(found) - {f"{sid}.json" for sid in wanted})
    for sid in wanted:
        path = Path(reports) / f"{sid}.json"
        if not path.is_file() or path.is_symlink():
            results.append((sid, ["no report for this scenario"]))
            continue
        try:
            wrapper = read_json(path)
        except Exception as exc:  # unreadable report fails its scenario
            results.append((sid, [f"unreadable report: {exc}"]))
            continue
        results.append((sid, check_one(sid, specs[sid], wrapper, lane, harness)))
    if extra:
        results.append(
            ("unexpected-reports", [f"reports for no expected scenario: {extra}"])
        )
    if not wanted:
        results.append(("no-scenarios", [f"lane {lane!r} has no scenarios"]))
    if junit:
        write_junit(junit, results, lane)
    return results


def cmd_check(args):
    sys.path.insert(0, str(args.harness))
    results = run_check(args.harness, args.reports, args.lane, args.junit)
    if args.lane != "container":
        print("UNISOLATED DEV RUN: no isolation boundary; this run certifies nothing.")
    for sid, failures in results:
        print(f"{'PASS' if not failures else 'FAIL'}  {sid}")
        for failure in failures:
            print(f"      {failure}")
    return 1 if any(f for _, f in results) else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cmp_ = sub.add_parser("compare", help="build one scenario's report")
    for name in (
        "harness",
        "capture",
        "stage",
        "outputs",
        "receipt-dir",
        "run-dir",
        "out",
    ):
        cmp_.add_argument(f"--{name}", required=True)
    cmp_.add_argument("--scenario", required=True)
    chk = sub.add_parser("check", help="check a lane's reports against expectations")
    chk.add_argument("--harness", required=True)
    chk.add_argument("--reports", required=True)
    chk.add_argument("--lane", choices=LANES, required=True)
    chk.add_argument("--junit")
    args = parser.parse_args(argv)
    try:
        return cmd_compare(args) if args.command == "compare" else cmd_check(args)
    except CheckError as exc:
        print(f"replay check error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
