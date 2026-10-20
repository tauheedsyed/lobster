"""A file loaded through an adapter must load the same way in the exported notebook.

``DataManagerV2.load_modality`` loads files through a modality adapter, which
changes the data on load (metadata, QC columns). The ``load_dataset`` IR it
logs is what the exported notebook's load cell renders, so executing that cell
on the same file must give the object the session loaded.
"""

import ast
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from lobster.core.adapters.factory import create_adapter, load_with_adapter
from lobster.core.notebooks.exporter import NotebookExporter
from lobster.core.provenance.analysis_ir import (
    AnalysisStep,
    create_adapter_loading_ir,
    create_data_loading_ir,
)
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.replay_compare import compare
from lobster.testing.replay_contract import ComparisonPolicy, Exclusion

POLICY = ComparisonPolicy(
    required=("adata",),
    excluded=(
        Exclusion("adata/uns/provenance", "session provenance record, not data"),
    ),
)

# Kwargs as the data-expert load tool passes them; the adapter stores them in uns.
DATASET_KWARGS = {"dataset_id": "GSE00000", "dataset_type": "custom"}

HOSTILE_KWARGS = {
    "dataset_id": 'x")\nimport os\nos.system("echo pwned")  # ',
    "note": "{{ 7 * 7 }} {% if 1 %}jinja{% endif %}",
    "threshold": 0.5,
    "nested": {"a": [1, 2.5, None, True], "b": "it's"},
}


def _synthetic_counts(seed=0, n_obs=60, n_vars=40):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(1.2, size=(n_obs, n_vars)).astype(np.float32)
    counts[rng.random((n_obs, n_vars)) < 0.6] = 0
    obs = pd.DataFrame(
        {"sample": rng.choice(["s1", "s2"], n_obs)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    genes = [f"MT-G{i}" for i in range(3)] + [f"RPS{i}" for i in range(3)]
    genes += [f"GENE{i}" for i in range(n_vars - len(genes))]
    var = pd.DataFrame(index=genes)
    return ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)


@pytest.fixture
def h5ad_path(tmp_path):
    path = tmp_path / "input.h5ad"
    _synthetic_counts().write_h5ad(path)
    return path


def _session_load(tmp_path, source, adapter, validate=True, **kwargs):
    dm = DataManagerV2(workspace_path=tmp_path / "ws", auto_scan=False)
    live = dm.load_modality(
        "cells", source, adapter=adapter, validate=validate, **kwargs
    )
    return dm, live


def _load_ir(dm):
    irs = [a["ir"] for a in dm.provenance.activities if a.get("type") == "load_dataset"]
    assert len(irs) == 1, irs
    ir = irs[0]
    return ir if isinstance(ir, AnalysisStep) else AnalysisStep.from_dict(ir)


def _run_cell(source, **namespace):
    scope = dict(namespace)
    # Test only: executes the load cell the exporter generated for this session.
    exec(compile(source, "<load-cell>", "exec"), scope)  # nosec B102
    return scope


def _assert_match(live, replayed):
    results = compare({"adata": live}, {"adata": replayed}, POLICY)
    failures = [r for r in results if r.status.value != "MATCH"]
    assert not failures, "\n".join(
        f"{r.path}: {r.status.value}\n  " + "\n  ".join(r.diffs + r.not_compared)
        for r in failures
    )
    assert all(r.compared > 0 for r in results)


@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("kwargs", [{}, DATASET_KWARGS], ids=["no-kwargs", "kwargs"])
@pytest.mark.parametrize(
    "adapter", ["transcriptomics_single_cell", "transcriptomics_bulk"]
)
def test_exported_load_cell_reproduces_the_session_load(
    tmp_path, h5ad_path, adapter, kwargs, validate
):
    dm, live = _session_load(tmp_path, h5ad_path, adapter, validate, **kwargs)
    live = dm.get_modality("cells")

    # The cell the exported notebook actually contains.
    cell = NotebookExporter(dm.provenance, dm)._create_data_loading_cell()
    replayed = _run_cell(cell.source, input_data=str(h5ad_path))["adata"]
    _assert_match(live, replayed)

    # The logged IR on its own replays the same object.
    ir = _load_ir(dm)
    assert ir.tool_name == "load_dataset"
    assert ir.exportable and ir.replayable
    assert "input_data" in ir.get_papermill_parameters()
    assert set(ir.get_papermill_parameters()) == {"input_data"}
    _assert_match(live, _run_cell(ir.render(), input_data=str(h5ad_path))["adata"])


def test_adapter_load_actually_changes_the_data(tmp_path, h5ad_path):
    """Guards the regression: a plain read_h5ad is not the session's object."""
    dm, live = _session_load(tmp_path, h5ad_path, "transcriptomics_single_cell")
    plain = ad.read_h5ad(h5ad_path)
    assert "n_features_detected" in live.obs and "n_features_detected" not in plain.obs
    assert "n_samples_detected" in live.var and "n_samples_detected" not in plain.var


def test_shared_factory_builds_the_registered_adapters(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path / "ws", auto_scan=False)
    assert {"transcriptomics_single_cell", "transcriptomics_bulk"} <= set(dm.adapters)
    for name, registered in dm.adapters.items():
        fresh = create_adapter(name)
        assert type(fresh) is type(registered), name
        for attr in ("data_type", "strict_validation"):
            assert getattr(fresh, attr, None) == getattr(registered, attr, None), (
                name,
                attr,
            )
    with pytest.raises(ValueError, match="Unknown adapter"):
        create_adapter("no_such_adapter")


def test_load_with_adapter_equals_adapter_from_source(h5ad_path):
    via_factory = load_with_adapter("transcriptomics_single_cell", h5ad_path)
    direct = create_adapter("transcriptomics_single_cell").from_source(h5ad_path)
    _assert_match(direct, via_factory)


def _load_call(tree):
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_with_adapter"
    ]
    assert len(calls) == 1
    return calls[0]


@pytest.mark.parametrize(
    "kwargs", [{}, DATASET_KWARGS, HOSTILE_KWARGS], ids=["none", "plain", "hostile"]
)
def test_rendered_load_cell_parses_and_carries_kwargs_only_as_literals(kwargs):
    ir = create_adapter_loading_ir("transcriptomics_single_cell", kwargs)
    # Rendering after a provenance JSON round trip must give the same code.
    rehydrated = AnalysisStep.from_dict(json.loads(json.dumps(ir.to_dict())))
    code = ir.render()
    assert rehydrated.render() == code
    tree = ast.parse(code)

    # Only the import, the load and the summary print: nothing injected.
    kinds = [type(stmt).__name__ for stmt in tree.body]
    assert kinds == ["ImportFrom", "Assign", "Expr"], kinds
    call = _load_call(tree)
    name_arg, path_arg = call.args
    assert ast.literal_eval(name_arg) == "transcriptomics_single_cell"
    assert isinstance(path_arg, ast.Name) and path_arg.id == "input_data"
    if not kwargs:
        assert call.keywords == []
        return
    (double_star,) = call.keywords
    assert double_star.arg is None  # a single ``**{...}`` literal
    assert ast.literal_eval(double_star.value) == kwargs
    # No kwarg value reached the code outside that literal.
    outside = code.replace(ast.get_source_segment(code, double_star.value), "")
    assert "pwned" not in outside and "jinja" not in outside
    assert "49" not in code


def test_non_finite_float_kwargs_render_as_runnable_literals():
    ir = create_adapter_loading_ir("transcriptomics_single_cell", {"t": float("nan")})
    code = ir.render()
    ast.parse(code)
    assert "**{'t': float('nan')}" in code


class _HostileRepr:
    def __repr__(self):
        return "__import__('os').system('echo pwned')"


@pytest.mark.parametrize(
    "value",
    [pd.DataFrame({"a": [1]}), _HostileRepr(), ("tuple",), np.float64(1.0)],
    ids=["dataframe", "hostile-repr", "tuple", "numpy-scalar"],
)
def test_unrenderable_kwargs_refuse_instead_of_loading_differently(value):
    ir = create_adapter_loading_ir(
        "transcriptomics_single_cell", {"dataset_id": "ok", "extra": value}
    )
    code = ir.render()
    ast.parse(code)
    assert "pwned" not in code and "load_with_adapter(" not in code
    with pytest.raises(NotImplementedError, match="'extra'"):
        _run_cell(code, input_data="unused.h5ad")


def test_in_memory_anndata_load_keeps_the_plain_load_ir(tmp_path):
    source = _synthetic_counts()
    dm, live = _session_load(tmp_path, source, "transcriptomics_single_cell")
    ir = _load_ir(dm)
    expected = create_data_loading_ir(input_param_name="input_data")
    assert ir.operation == expected.operation == "anndata.read_h5ad"
    assert ir.code_template == expected.code_template
    assert ir.imports == expected.imports
    assert ir.parameters == expected.parameters
    assert ir.description == "Load cells data from AnnData object"
    # The session stores the object it was given, unchanged by the adapter.
    assert "n_features_detected" not in live.obs
