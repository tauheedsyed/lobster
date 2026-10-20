#!/usr/bin/env python3
"""Run the notebook replay lane: capture, export, execute, compare, check.

Standard library only; every scientific step runs in its own process. In the
``container`` lane each phase is a separate ``docker run`` with exactly the
mounts it needs:

========  =========  ==========================================================
phase     network    can see
========  =========  ==========================================================
setup     yes        a copy of the source tree, the new environment (rw)
capture   none       environment (ro), harness (ro), its capture dir (rw)
execute   none       environment (ro), harness (ro), the staged notebook and
                     roots (ro), an empty outputs dir (rw), a receipt dir (rw)
compare   none       everything above read-only, the report dir (rw)
check     none       environment, harness, checker, reports (all ro)
========  =========  ==========================================================

The notebook therefore never sees the live capture, the reference copy, the
checker or any report. Execution runs unprivileged, with a read-only root file
system, no capabilities and a memory cap, from a pinned image whose
interpreter the environment was built for. The environment is built fresh
from ``uv.lock`` (wheels of this source tree plus locked dependencies, never
published Lobster wheels) and verified against the lock.

The ``unisolated`` lane runs the same phases as host processes with a given
interpreter. It has no isolation boundary and certifies nothing; use it to
iterate locally.

Usage:
    python tests/integration/notebook_replay/run_lane.py --lane container --work DIR
    python tests/integration/notebook_replay/run_lane.py --lane unisolated \\
        --python .venv/bin/python --work DIR
"""

import argparse
import json
import os
import resource
import shutil
import signal
import subprocess  # nosec B404 # Runs docker and the harness with fixed argv.
import sys
import time
from pathlib import Path

from staging import PROBE_VERSION, staged_notebook

HARNESS = Path(__file__).resolve().parent
REPO = HARNESS.parents[2]
IMAGE = (
    "ghcr.io/astral-sh/uv:0.10.0-python3.12-trixie-slim"
    "@sha256:18bed9c77b76e08650007104743d029c61bd2dec19fac8019d99b097119b0b40"
)
IMAGE_PYTHON = "/usr/local/bin/python3.12"
VENV = "/opt/replay/venv"
# Source packages installed for the replay environment (built from this tree).
SOURCE_PACKAGES = ("lobster-ai", "lobster-ai-tui", "lobster-transcriptomics")
MEMORY_LIMIT = "1200m"
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}
TIMEOUTS = {"setup": 1200, "capture": 600, "execute": 900, "compare": 600, "check": 300}
OUTPUT_BASENAME = "results"


class LaneError(Exception):
    pass


def isolated_env(home):
    env = {
        "HOME": f"{home}/home",
        "TMPDIR": f"{home}",
        "JUPYTER_DATA_DIR": f"{home}/jupyter/data",
        "JUPYTER_CONFIG_DIR": f"{home}/jupyter/config",
        "JUPYTER_RUNTIME_DIR": f"{home}/jupyter/runtime",
        "IPYTHONDIR": f"{home}/ipython",
        "MPLCONFIGDIR": f"{home}/matplotlib",
        "NUMBA_CACHE_DIR": f"{home}/numba",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.update(THREAD_ENV)
    return env


class Runner:
    """Runs one phase in a container, or as a host process."""

    def __init__(self, lane, work, python=None):
        self.lane = lane
        self.work = work
        self.python = python
        self.budget = {}
        self.counter = 0

    def path(self, host, container):
        return container if self.lane == "container" else str(host)

    @property
    def venv_python(self):
        return f"{VENV}/bin/python" if self.lane == "container" else self.python

    def run(self, phase, argv, mounts, *, network=False, label=None, check=True):
        """Run ``argv``; mounts are (host_path, container_path, writable)."""
        label = label or phase
        self.counter += 1
        log = self.work / "logs" / f"{self.counter:02d}-{label}.log"
        started = time.monotonic()
        if self.lane == "container":
            code, peak = self._docker(phase, label, argv, mounts, network, log)
        else:
            code, peak = self._host(phase, label, argv, log)
        self.budget[label] = {
            "seconds": round(time.monotonic() - started, 1),
            "memory_peak_mib": None if peak is None else round(peak / 2**20, 1),
            "exit_code": code,
        }
        if check and code != 0:
            tail = log.read_text(errors="replace")[-4000:]
            raise LaneError(f"{label} failed with exit code {code}:\n{tail}")
        return code

    def _docker(self, phase, label, argv, mounts, network, log):
        name = f"nbreplay-{os.getpid()}-{self.counter}-{label}"[:63]
        cmd = [
            "docker", "run", "--rm", "--name", name,
            "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=1536m",  # nosec B108
            "--memory", MEMORY_LIMIT, "--memory-swap", MEMORY_LIMIT,
            "--pids-limit", "512", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", f"{os.getuid()}:{os.getgid()}", "--workdir", "/tmp",  # nosec B108
        ]  # fmt: skip
        if not network:
            cmd += ["--network", "none"]
        # The container's own private tmpfs (mounted above), not the host /tmp.
        env = isolated_env("/tmp")  # nosec B108
        if phase == "setup":
            env.update({"UV_CACHE_DIR": "/uv-cache", "UV_PYTHON_DOWNLOADS": "never"})
        for key, value in sorted(env.items()):
            cmd += ["--env", f"{key}={value}"]
        for host, inside, writable in mounts:
            spec = f"type=bind,source={host},target={inside}"
            cmd += ["--mount", spec + ("" if writable else ",readonly")]
        # Report the container's peak memory (whole process tree) on exit.
        wrapper = (
            'mkdir -p "$HOME"; "$@"; rc=$?; '
            'echo "REPLAY_MEMORY_PEAK $(cat /sys/fs/cgroup/memory.peak 2>/dev/null)"; '
            "exit $rc"
        )
        cmd += [IMAGE, "sh", "-c", wrapper, "sh", *argv]
        with open(log, "w") as handle:
            proc = subprocess.Popen(  # nosec B603
                cmd, stdout=handle, stderr=subprocess.STDOUT
            )
            code = 124
            try:
                code = proc.wait(timeout=TIMEOUTS[phase])
            except subprocess.TimeoutExpired:
                pass
            finally:
                # Every exit (success, timeout, interrupt, cancellation) ends
                # with the container verified gone, so nothing keeps running
                # after this phase, or after the caller releases a lock. The
                # Docker client is reaped even if that verification fails.
                try:
                    remove_container(name)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=60)
        peak = None
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("REPLAY_MEMORY_PEAK "):
                value = line.split(" ", 1)[1].strip()
                peak = int(value) if value.isdigit() else None
        return code, peak

    def _host(self, phase, label, argv, log):
        home = self.work / "host-home" / f"{self.counter:02d}-{label}"
        home.mkdir(parents=True)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        env.update(isolated_env(str(home)))
        Path(env["HOME"]).mkdir()
        with open(log, "w") as handle:
            proc = subprocess.Popen(  # nosec B603
                argv, stdout=handle, stderr=subprocess.STDOUT, env=env,
                cwd=str(home), start_new_session=True,
            )  # fmt: skip
            code = 124
            try:
                code = proc.wait(timeout=TIMEOUTS[phase])
            except subprocess.TimeoutExpired:
                pass
            finally:
                # The phase ran in its own session: end every process left in
                # it (including descendants of a successful run).
                kill_group(proc.pid)
                if proc.poll() is None:
                    proc.wait()
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        peak = peak if sys.platform == "darwin" else peak * 1024
        return code, peak


def remove_container(name, attempts=30):
    """Force-remove ``name`` and positively confirm it no longer exists.

    Only a successful listing that does not contain the container counts as
    gone; a failing, hanging or unreachable Docker is an unknown state and
    raises, so a caller never treats it as cleaned up.
    """
    try:
        subprocess.run(  # nosec B603 B607
            ["docker", "rm", "--force", name], capture_output=True, timeout=60
        )
    except subprocess.TimeoutExpired:
        pass  # the listing below decides
    listing = [
        "docker", "ps", "--all", "--quiet", "--no-trunc",
        "--filter", f"name=^/{name}$",
    ]  # fmt: skip
    for _ in range(attempts):
        try:
            found = subprocess.run(  # nosec B603 B607
                listing, capture_output=True, text=True, timeout=30
            )
        except subprocess.TimeoutExpired:
            found = None
        if found is not None and found.returncode == 0 and not found.stdout.strip():
            return
        time.sleep(1)
    raise LaneError(
        f"cannot confirm container {name} is gone; check 'docker ps -a' before "
        "starting another run"
    )


def kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------


def git(*args):
    return subprocess.run(  # nosec B603 B607
        ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
    ).stdout


def copy_source(dest):
    """Copy tracked and untracked (not ignored) files; return revision, dirty."""
    revision = git("rev-parse", "HEAD").strip()
    dirty = bool(git("status", "--porcelain").strip())
    files = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    for name in filter(None, files.split("\0")):
        source = REPO / name
        if source.is_symlink() or not source.is_file():
            continue
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return revision, dirty


def setup(runner, work):
    run_dir = work / "run"
    if runner.lane == "container":
        src, wheels, venv, cache = (
            work / n for n in ("src", "wheels", "venv", "uv-cache")
        )
        for d in (src, wheels, venv, cache):
            d.mkdir()
        revision, dirty = copy_source(src)
        packages = " ".join(SOURCE_PACKAGES)
        script = (
            "set -e; cd /src; uv lock --check --no-config; "
            "uv export --frozen --no-config --no-hashes --no-emit-workspace "
            "--all-extras --all-groups --format requirements.txt "
            "-o /run-info/constraints.txt -q; "
            f"for p in {packages}; do "
            "uv build -q --no-config --wheel --package $p --out-dir /wheels; done; "
            f"uv venv -q --python {IMAGE_PYTHON} {VENV}; "
            f"uv pip install -q --no-config --python {VENV}/bin/python "
            "--constraints /run-info/constraints.txt /wheels/*.whl ipykernel; "
            f"{VENV}/bin/python /src/tests/integration/notebook_replay/setup_env.py "
            "--constraints /run-info/constraints.txt --source-wheels /wheels "
            "--out /run-info/env.json"
        )
        runner.run(
            "setup",
            ["sh", "-c", script],
            [
                (src, "/src", True),
                (wheels, "/wheels", True),
                (venv, VENV, True),
                (cache, "/uv-cache", True),
                (run_dir, "/run-info", True),
            ],
            network=True,
        )
        harness = src / "tests" / "integration" / "notebook_replay"
        checker = src / "scripts" / "check_notebook_replay_results.py"
    else:
        revision = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain").strip())
        harness = HARNESS
        checker = REPO / "scripts" / "check_notebook_replay_results.py"
        runner.run(
            "setup",
            [runner.python, str(HARNESS / "setup_env.py"), "--unverified",
             "--out", str(run_dir / "env.json")],
            [],
        )  # fmt: skip
    run = {
        "lane": runner.lane,
        "source_revision": revision,
        "source_dirty": dirty,
        "image": IMAGE if runner.lane == "container" else None,
        "venv_python": runner.venv_python,
        "source_packages": list(SOURCE_PACKAGES) if runner.lane == "container" else [],
    }
    (run_dir / "run.json").write_text(json.dumps(run, sort_keys=True, indent=1))
    return harness, checker


def capture(runner, work, harness, cap, policy):
    out = work / "capture" / cap
    out.mkdir(parents=True)
    runner.run(
        "capture",
        [
            runner.venv_python,
            runner.path(harness / "capture.py", "/harness/capture.py"),
            cap,
            runner.path(out, "/capture"),
            "--policy",
            runner.path(
                harness / "policies" / f"{policy}.json",
                f"/harness/policies/{policy}.json",
            ),
        ],
        [
            (work / "venv", VENV, False),
            (harness, "/harness", False),
            (out, "/capture", True),
        ],
        label=f"capture-{cap}",
    )
    return out


def stage(runner, work, sid, spec, cap_dir):
    cap = json.loads((cap_dir / "capture.json").read_text())
    stage_dir = work / "stage" / sid
    (stage_dir / "roots").mkdir(parents=True)
    for root in sorted((cap_dir / spec["stage_root"]).iterdir()):
        shutil.copy2(root, stage_dir / "roots" / root.name)
    notebook = stage_dir / "notebook.ipynb"
    shutil.copyfile(cap_dir / cap["notebooks"][spec["mode"]]["file"], notebook)
    probe = None
    outputs = work / "outputs" / sid
    outputs.mkdir(parents=True)
    in_stage = runner.path(stage_dir, "/stage")
    if spec.get("probe"):
        forbidden = sorted(
            {
                str(cap_dir / "oracle" / "final.pkl"),
                str(cap_dir / "capture.json"),
                str(cap_dir / "workspace"),
                str(work / "reports"),
                str(work / "src"),
                str(REPO / "pyproject.toml"),
                "/capture",
                "/reports",
                "/src",
            }
        )
        control = [f"{in_stage}/roots/input.h5ad"]
        probe = {"version": PROBE_VERSION, "forbidden": forbidden, "control": control}
        notebook.write_text(staged_notebook(notebook.read_text(), probe))
    parameters = {
        "input_data": f"{in_stage}/roots/input.h5ad",
        "output_prefix": f"{runner.path(outputs, '/outputs')}/{OUTPUT_BASENAME}",
    }
    record = {
        "scenario_id": sid,
        "capture": spec["capture"],
        "mode": spec["mode"],
        "stage_root": spec["stage_root"],
        "probe": probe,
        "parameters": parameters,
        "output_basename": OUTPUT_BASENAME,
    }
    (stage_dir / "stage.json").write_text(json.dumps(record, sort_keys=True, indent=1))
    for path in [*stage_dir.rglob("*"), stage_dir]:
        path.chmod(path.stat().st_mode & ~0o222)
    return stage_dir, outputs, parameters


def execute(runner, work, sid, harness, stage_dir, outputs, parameters):
    receipt = work / "receipts" / sid
    receipt.mkdir(parents=True)
    runner.run(
        "execute",
        [
            runner.venv_python,
            runner.path(harness / "in_boundary.py", "/harness/in_boundary.py"),
            "--notebook", f"{runner.path(stage_dir, '/stage')}/notebook.ipynb",
            "--outputs", runner.path(outputs, "/outputs"),
            "--receipt-dir", runner.path(receipt, "/receipt"),
            "--parameters", json.dumps(parameters, sort_keys=True),
        ],  # fmt: skip
        [
            (work / "venv", VENV, False),
            (harness, "/harness", False),
            (stage_dir, "/stage", False),
            (outputs, "/outputs", True),
            (receipt, "/receipt", True),
        ],
        label=f"execute-{sid}",
    )
    return receipt


def compare(runner, work, sid, harness, checker, cap_dir, stage_dir, outputs, receipt):
    reports = work / "reports"
    runner.run(
        "compare",
        [
            runner.venv_python,
            runner.path(checker, "/checker/check_notebook_replay_results.py"),
            "compare",
            "--harness", runner.path(harness, "/harness"),
            "--capture", runner.path(cap_dir, "/capture"),
            "--stage", runner.path(stage_dir, "/stage"),
            "--outputs", runner.path(outputs, "/outputs"),
            "--receipt-dir", runner.path(receipt, "/receipt"),
            "--run-dir", runner.path(work / "run", "/run-info"),
            "--scenario", sid,
            "--out", f"{runner.path(reports, '/reports')}/{sid}.json",
        ],  # fmt: skip
        [
            (work / "venv", VENV, False),
            (harness, "/harness", False),
            (checker, "/checker/check_notebook_replay_results.py", False),
            (cap_dir, "/capture", False),
            (stage_dir, "/stage", False),
            (outputs, "/outputs", False),
            (receipt, "/receipt", False),
            (work / "run", "/run-info", False),
            (reports, "/reports", True),
        ],
        label=f"compare-{sid}",
    )


def check(runner, work, harness, checker):
    junit = work / "junit"
    return runner.run(
        "check",
        [
            runner.venv_python,
            runner.path(checker, "/checker/check_notebook_replay_results.py"),
            "check",
            "--harness", runner.path(harness, "/harness"),
            "--reports", runner.path(work / "reports", "/reports"),
            "--lane", runner.lane,
            "--junit", f"{runner.path(junit, '/junit')}/notebook-replay.xml",
        ],  # fmt: skip
        [
            (work / "venv", VENV, False),
            (harness, "/harness", False),
            (checker, "/checker/check_notebook_replay_results.py", False),
            (work / "reports", "/reports", False),
            (junit, "/junit", True),
        ],
        check=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", choices=("container", "unisolated"), required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--python", help="interpreter for the unisolated lane")
    parser.add_argument("--scenario", action="append", help="limit to these IDs")
    args = parser.parse_args(argv)
    if args.lane == "unisolated" and not args.python:
        parser.error("--python is required for the unisolated lane")
    work = args.work.absolute()
    if work.exists() and any(work.iterdir()):
        parser.error(f"work directory must be empty or absent: {work}")
    try:
        work.relative_to(REPO)
        parser.error("work directory must be outside the source tree")
    except ValueError:
        pass
    for name in ("run", "logs", "reports", "junit"):
        (work / name).mkdir(parents=True)

    expectations = json.loads((HARNESS / "expectations.json").read_text())["scenarios"]
    selected = {
        sid: spec
        for sid, spec in sorted(expectations.items())
        if args.lane in spec["lanes"] and (not args.scenario or sid in args.scenario)
    }
    runner = Runner(args.lane, work, args.python)
    # CI cancellation sends SIGTERM: turn it into an interrupt so every
    # phase's teardown runs before the driver exits.
    signal.signal(signal.SIGTERM, _interrupt)
    started = time.monotonic()
    code = 1
    try:
        harness, checker = setup(runner, work)
        captures = {}
        for sid, spec in selected.items():
            if spec["capture"] not in captures:
                captures[spec["capture"]] = capture(
                    runner, work, harness, spec["capture"], spec["policy"]
                )
            cap_dir = captures[spec["capture"]]
            stage_dir, outputs, parameters = stage(runner, work, sid, spec, cap_dir)
            receipt = execute(
                runner, work, sid, harness, stage_dir, outputs, parameters
            )
            compare(
                runner,
                work,
                sid,
                harness,
                checker,
                cap_dir,
                stage_dir,
                outputs,
                receipt,
            )
        if args.scenario:
            print(
                "Partial run (--scenario): the check below reports missing scenarios."
            )
        code = check(runner, work, harness, checker)
        print((work / "logs" / f"{runner.counter:02d}-check.log").read_text())
    except LaneError as exc:
        print(f"replay lane error: {exc}", file=sys.stderr)
        code = 2
    finally:
        runner.budget["total"] = {"seconds": round(time.monotonic() - started, 1)}
        (work / "budget.json").write_text(json.dumps(runner.budget, indent=1))
        width = max(len(k) for k in runner.budget)
        for label, entry in runner.budget.items():
            mem = entry.get("memory_peak_mib")
            print(f"{label:<{width}}  {entry['seconds']:>7.1f} s  "
                  f"{'' if mem is None else f'{mem:>7.1f} MiB'}")  # fmt: skip
    return code


if __name__ == "__main__":
    sys.exit(main())
