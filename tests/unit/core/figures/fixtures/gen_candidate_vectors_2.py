"""Generate the multi-input supplement ``candidate_vectors_2.json``.

Run from the repo root:
    .venv/bin/python tests/unit/core/figures/fixtures/gen_candidate_vectors_2.py

A NEW immutable supplement. ``candidate_vectors.json`` (/1) and
``execution_request_vectors.json`` (/2) are NOT touched.

Where the values come from (nothing is typed by hand):
- **Input edges** are the real capture output. ``CustomCodeExecutionService.execute`` runs the
  code in its child and returns ``stats["input_edges"]``. The edges come from a temporary
  ``DataManagerV2`` with three inputs:
  - ``zeta`` (``modality_name``) and ``alpha`` (read by path) carry lineage, so their edges are
    verified store events;
  - ``nolineage`` (read by path) has no identity, so its edge is the existing
    declared/null/unverified edge.
  - The lineage ``instance_id`` values come from a fixed sequence (``lineage.uuid`` is
    patched), so the file is reproducible. Real ids are random uuids.
- **Commits** go through the real ``TrustedSaver`` into a ``LocalFigureStore`` with fixed
  figure/revision/execution ids, via the /1 generator's helpers (loaded by path, unchanged).
  The stored digest must equal the digest recomputed from the vector.
- **Mutations** (an edge removed, an instance_id changed, edges reordered, an extra
  same-name edge with a different id) are digested with the engine's own
  ``attempt_descriptor``/``canonical_descriptor_bytes``. They show that each mutation changes
  the digest and that distinct ids never collapse. The ordering rules are NOT redefined: the
  descriptor hashes the supplied tuple order.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List

OUT = Path(__file__).with_name("candidate_vectors_2.json")
FIXTURE_VERSION = "engine-candidate-vectors/2"

_spec = importlib.util.spec_from_file_location(
    "gen_candidate_vectors", Path(__file__).with_name("gen_candidate_vectors.py")
)
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

CAPTURE_CODE = (
    "import anndata\n"
    "a = anndata.read_h5ad('alpha.h5ad')\n"
    "n = anndata.read_h5ad('nolineage.h5ad')\n"
    "result = float(adata.X.sum() + a.X.sum() + n.X.sum())\n"
)
FIG = b'{"data":[{"type":"bar","x":["alpha","zeta"],"y":[4,12]}],"layout":{}}'
FIG_EDITED = (
    b'{"data":[{"type":"bar","x":["alpha","zeta"],"y":[4,12],'
    b'"marker":{"color":"teal"}}],"layout":{}}'
)
CONFIG = b'{"displaylogo":false}'


@contextmanager
def _fixed_lineage_ids() -> Iterator[None]:
    import types

    from lobster.core.provenance import lineage as lineage_mod

    counter = itertools.count(1)
    real = lineage_mod.uuid

    def uuid4():
        return types.SimpleNamespace(hex=f"inst{next(counter):028d}")

    lineage_mod.uuid = types.SimpleNamespace(uuid4=uuid4)
    try:
        yield
    finally:
        lineage_mod.uuid = real


def _capture(dm) -> List[Dict[str, Any]]:
    from lobster.services.execution.custom_code_execution_service import (
        CustomCodeExecutionService,
    )

    result, stats, _ = CustomCodeExecutionService(dm).execute(
        CAPTURE_CODE, modality_name="zeta", load_workspace_files=False
    )
    assert result == 4.0 + 12.0 + 4.0, result
    return stats["input_edges"]


def _digest(cand, info, parent_fig=None, parent_cfg=None) -> str:
    from lobster.core.figures.records import (
        attempt_descriptor,
        canonical_descriptor_bytes,
    )

    descriptor = attempt_descriptor(
        cand,
        parent_figure_sha256=parent_fig,
        parent_config_sha256=parent_cfg,
        code_bytes=info.code_bytes,
        tool=info.tool,
        persist=info.persist,
        engine_version=info.engine_version,
        plotly_version=info.plotly_version,
    )
    return hashlib.sha256(canonical_descriptor_bytes(descriptor)).hexdigest()


def build() -> Dict[str, Any]:
    import anndata
    import numpy as np

    from lobster.core.data_manager_v2 import DataManagerV2
    from lobster.core.figures.records import Candidate, InputEdge
    from lobster.core.figures.store import LocalFigureStore

    code = CAPTURE_CODE.encode("utf-8")
    with tempfile.TemporaryDirectory() as tmp, _fixed_lineage_ids():
        dm = DataManagerV2(workspace_path=Path(tmp) / "ws", enable_provenance=False)
        dm.store_modality(
            name="zeta", adata=anndata.AnnData(np.full((2, 2), 3.0)), step="t"
        )
        dm.store_modality(
            name="alpha", adata=anndata.AnnData(np.ones((2, 2))), step="t"
        )
        dm.modalities["nolineage"] = anndata.AnnData(
            np.ones((2, 2))
        )  # identity unknown
        captured_1 = _capture(dm)
        # a new store event for zeta (same data): the edit's capture names the NEW event
        dm.store_modality(
            name="zeta", adata=anndata.AnnData(np.full((2, 2), 3.0)), step="t"
        )
        captured_2 = _capture(dm)

    vectors: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp, v1._fixed_ids():
        store = LocalFigureStore(tmp, session_id="s1")
        info_1 = v1._info(
            "s1:call_m1", "execute_custom_code", "proteomics_expert", code
        )
        created = Candidate(
            attempt_key="s1:call_m1:0",
            candidate_index=0,
            figure_bytes=FIG,
            config_bytes=CONFIG,
            kind="created",
            actor="expert:proteomics_expert",
            label="Totals",
            request="totals per input",
            resolved_intent=None,
            parent_ref=None,
            inputs=tuple(InputEdge.from_dict(e) for e in captured_1),
        )
        vec_1, res_1 = v1._vector(
            "created_multi_input_two_verified_one_unknown", store, info_1, created
        )
        vec_1["captured_input_edges"] = captured_1
        vectors.append(vec_1)

        info_2 = v1._info(
            "s1:call_m2", "execute_custom_code", "proteomics_expert", code
        )
        edited = Candidate(
            attempt_key="s1:call_m2:0",
            candidate_index=0,
            figure_bytes=FIG_EDITED,
            config_bytes=CONFIG,
            kind="edited",
            actor="expert:proteomics_expert",
            label="Totals (teal)",
            request="make the bars teal",
            resolved_intent=None,
            parent_ref=res_1.ref,
            inputs=tuple(InputEdge.from_dict(e) for e in captured_2),
        )
        vec_2, _ = v1._vector(
            "edited_multi_input_inherits_parent_edges", store, info_2, edited
        )
        vec_2["captured_input_edges"] = captured_2
        vectors.append(vec_2)

    # Digest sensitivity on the created vector's authoritative inputs (engine functions).
    base_inputs = [InputEdge.from_dict(e) for e in vec_1["authoritative"]["inputs"]]
    verified = [e for e in base_inputs if e.verified]
    changed = [
        (
            InputEdge(e.name, e.instance_id + "_other", e.origin, e.verified)
            if e is verified[0]
            else e
        )
        for e in base_inputs
    ]
    extra = base_inputs + [
        InputEdge(verified[0].name, "inst_distinct_same_name", "store_event", True)
    ]
    variants = {
        "base": base_inputs,
        "edge_removed": [e for e in base_inputs if e is not verified[1]],
        "instance_id_changed": changed,
        "order_reversed": list(reversed(base_inputs)),
        "same_name_distinct_instance_added": extra,
    }
    mutations = {}
    for name, inputs in variants.items():
        cand = Candidate(**{**created.__dict__, "inputs": tuple(inputs)})
        mutations[name] = {
            "inputs": [e.to_dict() for e in inputs],
            "attempt_digest": _digest(cand, info_1),
        }
    assert mutations["base"]["attempt_digest"] == vec_1["attempt_digest"]
    digests = [m["attempt_digest"] for m in mutations.values()]
    assert len(set(digests)) == len(digests), "a mutation did not change the digest"

    return {
        "fixture_version": FIXTURE_VERSION,
        "supplements": "engine-candidate-vectors/1 (unchanged)",
        "note": (
            "Generated by gen_candidate_vectors_2.py. Input edges are the real "
            "CustomCodeExecutionService capture output (stats.input_edges) with lineage ids "
            "from a fixed sequence; commits use the real TrustedSaver with fixed figure ids. "
            "An edge names a store event or stored object, not live content and not proof of "
            "what the code read. WIP core, NOT accepted."
        ),
        "capture_code_utf8": CAPTURE_CODE,
        "vectors": vectors,
        "digest_mutations": mutations,
    }


def render() -> bytes:
    return (
        json.dumps(build(), indent=1, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
        + b"\n"
    )


def main() -> None:
    OUT.write_bytes(render())
    print(f"wrote {OUT} sha256={hashlib.sha256(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
