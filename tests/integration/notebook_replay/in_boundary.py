"""Execute one notebook in a fresh kernel, inside the replay boundary.

Runs where the notebook runs (the container, or the host for an unisolated
dev run). It:

1. refuses to start unless the outputs and receipt directories are empty,
   so a stale file can never pass as a new output or receipt;
2. installs a private kernelspec (in the isolated Jupyter data directory)
   whose kernel gets single-threaded maths libraries and a fixed hash seed;
3. runs the staged notebook with Papermill on that kernel, with a per-cell
   timeout, in ONE kernel: a harness cell first records the kernel's receipt
   (interpreter, installed distributions, thread settings, process id), the
   notebook's own cells follow unchanged, and a last harness cell records the
   process id again, so the receipt describes the kernel that ran the export;
4. reads both harness results from the executed cells' outputs and writes
   ``execution.json``, ``kernel_receipt.json`` and ``executed.ipynb`` to the
   receipt directory only after the kernel has stopped; any other file found
   there marks the run as tampered.

The staged notebook itself is never modified; only the in-memory copy that
executes carries the two harness cells.

Usage: python in_boundary.py --notebook N --outputs O --receipt-dir R
       --parameters JSON [--cell-timeout S]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from receipt import RECEIPT_CODE, THREAD_ENV, memory_peak_bytes

KERNEL_NAME = "lobster-replay"
MARK = "REPLAY-HARNESS-RECEIPT "
RECEIPT_CELL = (
    RECEIPT_CODE
    + "\nimport json as _replay_json\nimport os as _replay_os\n"
    + "_replay_receipt = collect_receipt()\n"
    + "_replay_receipt['pid'] = _replay_os.getpid()\n"
    + f"print({MARK!r} + _replay_json.dumps(_replay_receipt, sort_keys=True))\n"
    + "del collect_receipt, _replay_receipt, _replay_json, _replay_os\n"
)
FINAL_CELL = (
    "import os as _replay_os\n"
    + f"print({MARK!r} + '{{\"pid\": ' + str(_replay_os.getpid()) + '}}')\n"
)
OWN_FILES = {"execution.json", "kernel_receipt.json", "executed.ipynb"}


def install_kernelspec():
    data_dir = Path(os.environ["JUPYTER_DATA_DIR"])
    spec_dir = data_dir / "kernels" / KERNEL_NAME
    spec_dir.mkdir(parents=True, exist_ok=False)
    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Lobster replay",
        "language": "python",
        "env": dict(THREAD_ENV),
    }
    (spec_dir / "kernel.json").write_text(json.dumps(spec))
    return spec_dir


def instrumented(path):
    """The staged notebook with the two harness cells, in memory only."""
    import nbformat

    notebook = nbformat.read(str(path), as_version=4)
    first = nbformat.v4.new_code_cell(RECEIPT_CELL)
    last = nbformat.v4.new_code_cell(FINAL_CELL)
    for cell in (first, last):
        cell.metadata["tags"] = ["replay-harness"]
    notebook.cells = [first, *notebook.cells, last]
    notebook.metadata["kernelspec"] = {
        "name": KERNEL_NAME,
        "display_name": "Lobster replay",
        "language": "python",
    }
    return notebook


def harness_output(cell):
    """The JSON a harness cell printed, or None."""
    for output in cell.get("outputs", []):
        if output.get("output_type") != "stream" or output.get("name") != "stdout":
            continue
        for line in "".join(output.get("text", "")).splitlines():
            if line.startswith(MARK):
                return json.loads(line[len(MARK) :])
    return None


def _failure(exc):
    name = type(exc).__name__
    ename = getattr(exc, "ename", None) or name
    evalue = str(getattr(exc, "evalue", None) or exc)
    status = "TIMEOUT" if "Timeout" in name or "Timeout" in str(ename) else "ERROR"
    return status, {"type": name, "ename": str(ename), "evalue": evalue[:2000]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--cell-timeout", type=int, default=600)
    args = parser.parse_args(argv)
    parameters = json.loads(args.parameters)
    record = {
        "parameters": parameters,
        "pre_run_entries": sorted(os.listdir(args.outputs)),
        "pre_run_receipt_entries": sorted(os.listdir(args.receipt_dir)),
        "status": None,
        "error": None,
        "kernel_pid_after": None,
        "unexpected_receipt_entries": [],
    }
    started = time.monotonic()
    executed = None
    receipt = None
    try:
        if record["pre_run_entries"] or record["pre_run_receipt_entries"]:
            record["status"] = "HARNESS_ERROR"
            record["error"] = {"type": "StaleFiles", "ename": "", "evalue": ""}
            return 3
        import papermill

        install_kernelspec()
        work = Path(os.environ["TMPDIR"]) / "replay-cwd"  # set by the lane driver
        work.mkdir(parents=True, exist_ok=True)
        source = Path(os.environ["TMPDIR"]) / "instrumented.ipynb"
        import nbformat

        nbformat.write(instrumented(args.notebook), str(source))
        result = Path(os.environ["TMPDIR"]) / "executed.ipynb"
        try:
            executed = papermill.execute_notebook(
                str(source),
                str(result),
                parameters=parameters,
                kernel_name=KERNEL_NAME,
                cwd=str(work),
                execution_timeout=args.cell_timeout,
                progress_bar=False,
            )
            record["status"] = "OK"
        except Exception as exc:  # any notebook failure is recorded, not raised
            record["status"], record["error"] = _failure(exc)
            # Papermill writes the partly executed notebook before raising.
            done = result if result.is_file() else source
            executed = nbformat.read(str(done), as_version=4)
        harness = [
            c for c in executed.cells if "replay-harness" in c.metadata.get("tags", [])
        ]
        receipt = harness_output(harness[0]) if harness else None
        if record["status"] == "OK":
            final = harness_output(harness[-1]) if len(harness) == 2 else None
            record["kernel_pid_after"] = final and final.get("pid")
        if receipt is None and record["status"] == "OK":
            record["status"] = "HARNESS_ERROR"
            record["error"] = {"type": "NoKernelReceipt", "ename": "", "evalue": ""}
        return 0
    finally:
        # The kernel has stopped by now; nothing the notebook ran can write here.
        record["duration_s"] = round(time.monotonic() - started, 3)
        record["memory_peak_bytes"] = memory_peak_bytes()
        record["unexpected_receipt_entries"] = sorted(
            set(os.listdir(args.receipt_dir)) - OWN_FILES
        )
        if executed is not None:
            import nbformat

            nbformat.write(executed, str(args.receipt_dir / "executed.ipynb"))
        if receipt is not None:
            (args.receipt_dir / "kernel_receipt.json").write_text(
                json.dumps(receipt, sort_keys=True, indent=1)
            )
        (args.receipt_dir / "execution.json").write_text(
            json.dumps(record, sort_keys=True, indent=1)
        )


if __name__ == "__main__":
    sys.exit(main())
