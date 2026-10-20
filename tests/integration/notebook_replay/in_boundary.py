"""Execute one notebook in a fresh kernel, inside the replay boundary.

Runs where the notebook runs (the container, or the host for an unisolated
dev run). It:

1. refuses to start unless the outputs directory is empty, so a stale file
   can never pass as a new output;
2. installs a private kernelspec (in the isolated Jupyter data directory)
   whose kernel gets single-threaded maths libraries and a fixed hash seed;
3. runs a receipt notebook on that kernel, recording interpreter,
   installed distributions and thread settings;
4. runs the notebook with Papermill on the same kernel, with a per-cell
   timeout, and records how execution ended.

Writes ``execution.json``, ``kernel_receipt.json`` and ``executed.ipynb`` to
the receipt directory, never to the outputs directory.

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


def receipt_notebook(path):
    import nbformat

    code = RECEIPT_CODE + (
        "\nimport json as _json\n"
        f"with open({str(path)!r}, 'w') as _handle:\n"
        "    _json.dump(collect_receipt(), _handle, sort_keys=True)\n"
    )
    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(nbformat.v4.new_code_cell(code))
    notebook.metadata["kernelspec"] = {
        "name": KERNEL_NAME,
        "display_name": "Lobster replay",
        "language": "python",
    }
    return notebook


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
        "status": None,
        "error": None,
    }
    started = time.monotonic()
    try:
        if record["pre_run_entries"]:
            record["status"] = "HARNESS_ERROR"
            record["error"] = {"type": "StaleOutputs", "ename": "", "evalue": ""}
            return 3
        import papermill

        install_kernelspec()
        work = Path(os.environ["TMPDIR"]) / "replay-cwd"  # set by the lane driver
        work.mkdir(parents=True, exist_ok=True)
        receipt_path = args.receipt_dir / "kernel_receipt.json"
        probe = args.receipt_dir / "receipt.ipynb"
        import nbformat

        nbformat.write(receipt_notebook(receipt_path), str(probe))
        papermill.execute_notebook(
            str(probe),
            str(args.receipt_dir / "receipt.executed.ipynb"),
            kernel_name=KERNEL_NAME,
            cwd=str(work),
            execution_timeout=120,
            progress_bar=False,
        )
        try:
            papermill.execute_notebook(
                str(args.notebook),
                str(args.receipt_dir / "executed.ipynb"),
                parameters=parameters,
                kernel_name=KERNEL_NAME,
                cwd=str(work),
                execution_timeout=args.cell_timeout,
                progress_bar=False,
            )
            record["status"] = "OK"
        except Exception as exc:  # any notebook failure is recorded, not raised
            record["status"], record["error"] = _failure(exc)
        return 0
    finally:
        record["duration_s"] = round(time.monotonic() - started, 3)
        record["memory_peak_bytes"] = memory_peak_bytes()
        (args.receipt_dir / "execution.json").write_text(
            json.dumps(record, sort_keys=True, indent=1)
        )


if __name__ == "__main__":
    sys.exit(main())
