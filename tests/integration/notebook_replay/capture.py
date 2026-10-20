"""Capture a live Lobster session for the notebook replay lane.

Runs in its own process, without network, in the environment the notebook
will later run in. For one scenario it:

1. writes deterministic synthetic input files (the roots);
2. runs real Lobster tools on a real ``DataManagerV2`` (only the agent's LLM is
   replaced, through ``factory_tools``);
3. freezes the session's final object as the reference copy as soon as the
   analysis finishes, and checks that the frozen copy equals it, before the
   exporter runs and before anything lossy (such as an H5AD write) touches it;
4. exports notebooks through the public export path, once per export mode,
   then fails if exporting changed the live object;
5. records hashes, tool parameters and an environment receipt in
   ``capture.json``.

The reference copy is a pickle of the live object. Pickle keeps exact types
(NumPy and pandas dtypes, ``None``, nested ``uns`` values) where H5AD drops or
converts them. Only this harness writes and reads it, in a directory the
notebook's container cannot see. Before accepting it, the capture checks that
the comparator can compare every required value of it (nothing reported as
"not compared"), so an unsupported type stops the capture instead of weakening
the comparison later.

Usage: python capture.py <scenario> <capture_dir>
"""

import argparse
import hashlib
import json
import pickle  # nosec B403 # Reference copy written and read only by this harness.
import shutil
import sys
from pathlib import Path

from receipt import collect_receipt, memory_peak_bytes

OUTPUT_BASENAME = "results"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_counts(seed, path):
    """A small sparse single-cell count matrix with mito and ribo genes."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse

    rng = np.random.default_rng(seed)
    n_cells, n_genes = 240, 300
    genes = (
        [f"MT-{i}" for i in range(6)]
        + [f"RPS{i}" for i in range(6)]
        + [f"G{i}" for i in range(n_genes - 12)]
    )
    rate = rng.gamma(0.6, 2.0, size=(n_cells, n_genes))
    rate[:20, :6] *= 25  # some cells with high mitochondrial content
    counts = sparse.csr_matrix(rng.poisson(rate).astype(np.float32))
    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=genes),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def _tool_ok(name, response):
    text = str(response)
    if text.lstrip().lower().startswith(("error", "failed")):
        raise RuntimeError(f"{name} did not succeed: {text[:500]}")


def capture_s1(out):
    """S1: local H5AD -> QC -> filter/normalize with non-default parameters."""
    from lobster.agents.transcriptomics.transcriptomics_expert import (
        transcriptomics_expert,
    )
    from lobster.core.runtime.data_manager import DataManagerV2
    from lobster.testing.provenance_replay import factory_tools

    root = out / "roots" / "input.h5ad"
    synthetic_counts(7, root)
    # A different input with the same shape, for the incorrect-root check.
    synthetic_counts(8, out / "roots_perturbed" / "input.h5ad")

    dm = DataManagerV2(workspace_path=out / "workspace", auto_scan=False)
    dm.load_modality("s1", root, adapter="transcriptomics_single_cell")
    tools = factory_tools(
        transcriptomics_expert,
        "lobster.agents.transcriptomics.transcriptomics_expert",
        dm,
    )
    calls = [
        (
            "assess_data_quality",
            {"modality_name": "s1", "min_genes": 50, "max_mt_pct": 15.0},
        ),
        (
            "filter_and_normalize",
            {
                "modality_name": "s1_quality_assessed",
                "min_genes_per_cell": 50,
                "max_genes_per_cell": 280,
                "min_cells_per_gene": 5,
                "max_mito_percent": 15.0,
                "target_sum": 5000,
                "save_result": True,
            },
        ),
    ]
    for name, arguments in calls:
        _tool_ok(name, tools[name].invoke(arguments))
    final_name = "s1_quality_assessed_filtered_normalized"
    if dm.list_modalities()[-1] != final_name:
        raise RuntimeError(f"unexpected modalities: {dm.list_modalities()}")
    final = dm.get_modality(final_name)

    def export():
        notebooks = {}
        for mode in ("successful", "replay"):
            exported = dm.export_notebook(
                f"s1_{mode}", "S1 replay", filter_strategy=mode
            )
            target = out / "notebooks" / f"{mode}.ipynb"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(exported, target)
            notebooks[mode] = {
                "file": f"notebooks/{mode}.ipynb",
                "sha256": sha256_file(target),
            }
        return notebooks

    return {
        "final": final,
        "declared": f"{OUTPUT_BASENAME}_{final_name}.h5ad",
        "export": export,
        "tool_parameters": [{"tool": name, "arguments": args} for name, args in calls],
        "roots": {"input.h5ad": sha256_file(root)},
        "perturbed_roots": {
            "input.h5ad": sha256_file(out / "roots_perturbed" / "input.h5ad")
        },
        "provenance": dm.provenance.activities,
    }


SCENARIOS = {"s1": capture_s1}


class CaptureError(Exception):
    pass


def seal_and_export(final, policy_path, declared, export):
    """Freeze the reference, prove it faithful, then export, then re-check.

    1. freeze the live object's bytes;
    2. before anything is exported, require the frozen copy to equal the
       still-untouched live object, so a value the copy cannot carry
       faithfully stops the capture here;
    3. run ``export``;
    4. require the live object to still equal the frozen copy, so an exporter
       that changed the session cannot redefine the reference.
    """
    frozen = freeze(final)
    problems = _oracle_problems(final, policy_path, declared, frozen)
    if problems:
        raise CaptureError(f"reference copy is not faithful: {problems[:10]}")
    exported = export()
    problems = _oracle_problems(final, policy_path, declared, frozen)
    if problems:
        raise CaptureError(f"exporting changed the live object: {problems[:10]}")
    return frozen, exported


def freeze(value):
    """The reference copy's bytes, taken from the live object right now."""
    return pickle.dumps(value, protocol=5)


def _oracle_problems(final, policy_path, declared, frozen=None):
    """Ways the frozen copy fails to equal ``final`` under the policy.

    Anything the comparator cannot compare counts as a problem, so an
    unsupported type stops the capture. See :func:`seal_and_export` for when
    it runs.
    """
    from lobster.testing.replay_compare import compare
    from lobster.testing.replay_contract import ComparisonPolicy, SlotStatus

    policy = ComparisonPolicy.from_json(Path(policy_path).read_text())
    frozen = freeze(final) if frozen is None else frozen
    copy = pickle.loads(frozen)  # nosec B301 # bytes this process just produced
    tree = {"files": {declared: copy}}
    results = compare(tree, {"files": {declared: final}}, policy)
    problems = []
    for slot in results:
        if slot.status is not SlotStatus.MATCH:
            problems.append(f"{slot.path}: {slot.status.value} {slot.diffs[:5]}")
        problems.extend(slot.not_compared)
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args(argv)

    out = args.capture_dir
    if any(out.iterdir()):
        raise SystemExit(f"capture directory is not empty: {out}")
    receipt = collect_receipt()
    result = SCENARIOS[args.scenario](out)

    declared = result["declared"]
    try:
        frozen, notebooks = seal_and_export(
            result["final"], args.policy, declared, result["export"]
        )
    except CaptureError as exc:
        raise SystemExit(str(exc)) from None
    oracle = out / "oracle" / "final.pkl"
    oracle.parent.mkdir()
    oracle.write_bytes(frozen)

    provenance = out / "provenance.json"
    provenance.write_text(
        json.dumps(result["provenance"], sort_keys=True, indent=1, default=repr)
    )
    record = {
        "scenario": args.scenario,
        "roots": result["roots"],
        "perturbed_roots": result["perturbed_roots"],
        "tool_parameters": result["tool_parameters"],
        "notebooks": notebooks,
        "recipe_sha256": sha256_file(provenance),
        "declared_outputs": {
            declared: {"file": "oracle/final.pkl", "sha256": sha256_file(oracle)}
        },
        "receipt": receipt,
        "memory_peak_bytes": memory_peak_bytes(),
    }
    (out / "capture.json").write_text(json.dumps(record, sort_keys=True, indent=1))
    print(
        json.dumps(
            {"captured": args.scenario, "outputs": sorted(record["declared_outputs"])}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
