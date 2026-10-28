"""Figure contract: analysis tools never create figures; core harness tests.

The fake agent below is test-only. Its tools exercise every observed seam of the
executed harness: clean analysis, figure files, Plotly JSON, staging, committing, error
strings, raising, and producer-shaped tools whose implementation module is set to the
transitional producer module (the harness keys on ``<module>.<tool name>``).
"""

import json
import sys
import types
from types import SimpleNamespace
from typing import Annotated

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from langchain_core.tools import InjectedToolCallId, tool

from lobster.config import aquadif
from lobster.testing.contract_mixins import (
    FigureContractTestMixin,
    FigureToolCase,
    check_figure_tool_case,
)

FAKE_MODULE = "lobster_d8_fake_agent"
VIZ = "lobster.agents.visualization_expert"
TRUSTED = "lobster.tools.custom_code_tool"


def _adata():
    rng = np.random.default_rng(0)
    a = ad.AnnData(
        X=rng.random((12, 4)).astype("float32"),
        obs=pd.DataFrame(
            {"group": pd.Categorical(["a", "b"] * 6)},
            index=[f"c{i}" for i in range(12)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
    )
    a.layers["counts"] = a.X.copy()
    a.obsm["X_umap"] = rng.random((12, 2))
    a.uns["note"] = {"k": 1}
    return a


def _fig():
    return go.Figure(go.Bar(x=["a", "b"], y=[1, 2]))


def _meta(fn, *categories, module=None):
    fn.metadata = {"categories": list(categories), "provenance": True}
    fn.tags = list(categories)
    if module is not None:
        fn.func.__module__ = module
    return fn


LLM_SEEN = []


def llm_agent(
    data_manager, callback_handler, agent_name, delegation_tools, workspace_path
):
    from lobster.config import llm_factory

    LLM_SEEN.append(llm_factory.create_llm())
    return good_agent(
        data_manager, callback_handler, agent_name, delegation_tools, workspace_path
    )


def _graph(tools):
    return SimpleNamespace(
        nodes={"tools": SimpleNamespace(tools_by_name={t.name: t for t in tools})}
    )


def _log(dm, name):
    return dm.log_tool_usage(tool_name=name, parameters={}, description=name)


def _producer(dm, name, *, order="ok", mutate=False, wrong_id=False):
    @tool(name)
    def produce(
        modality_name: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        save_plot: bool = True,
    ) -> str:
        """Fake transitional producer."""
        adata = dm.get_modality(modality_name)
        if mutate:
            adata.obs["leaked"] = 1
        pid, _, _ = dm.plot_manager.add_plot(plot=_fig(), title=name, source="test")
        if not save_plot:
            return f"preview of {adata.n_obs} cells"
        if order == "commit_first":
            dm.plot_manager.commit_staged(
                {"id": "a"}, tool_call_id=tool_call_id, staged_ids=[pid]
            )
            _log(dm, name)
            return f"plotted {adata.n_obs} cells"
        mode = getattr(dm, "_d8_mode", "ok")  # set by a case setup (repair REDs)
        ids = [pid]
        if mode == "two_figures":
            ids.append(
                dm.plot_manager.add_plot(plot=_fig(), title="2", source="test")[0]
            )
        elif mode == "duplicate_sid":
            ids = [pid, pid]
        elif mode == "leftover_sid":
            ids = [dm._d8_leftover]
        activity = _log(dm, name)
        call_id = "someone_else" if wrong_id else tool_call_id
        if mode == "forged_activity":
            activity = {"id": "forged"}
        elif mode == "no_activity":
            activity = None
        dm.plot_manager.commit_staged(activity, tool_call_id=call_id, staged_ids=ids)
        if mode == "overwrite_old":
            dm._d8_old_path.write_text('{"data": []}')
        elif mode == "extra_legacy":
            (dm.workspace_path / "plots" / "extra_legacy.html").write_text("<html/>")
        return f"plotted {adata.n_obs} cells"

    produce.func.__module__ = VIZ
    produce.metadata = {"categories": ["ANALYZE"], "provenance": True}
    produce.tags = ["ANALYZE"]
    return produce


def _tools(dm):
    @tool
    def score_cells(modality_name: str) -> str:
        """Clean analysis: computes a score and stores a new modality."""
        adata = dm.get_modality(modality_name).copy()
        adata.obs["score"] = np.asarray(adata.X.mean(axis=1)).ravel()
        dm.store_modality(f"{modality_name}_scored", adata)
        _log(dm, "score_cells")
        return f"scored {adata.n_obs} cells"

    @tool
    def png_writer(modality_name: str) -> str:
        """Analysis that writes a PNG."""
        out = dm.workspace_path / "exports"
        out.mkdir(exist_ok=True)
        (out / "umap.png").write_bytes(b"\x89PNG fake")
        return "scored and drew"

    @tool
    def json_figure_writer(modality_name: str) -> str:
        """Analysis that writes Plotly JSON outside the store."""
        out = dm.workspace_path / "results"
        out.mkdir(exist_ok=True)
        (out / "fig.json").write_text(_fig().to_json())
        (out / "table.json").write_text(json.dumps({"rows": [1, 2]}))
        return "scored and drew"

    @tool
    def stager(modality_name: str) -> str:
        """Analysis that stages a figure."""
        dm.plot_manager.add_plot(plot=_fig(), title="hidden", source="test")
        return "scored and drew"

    @tool
    def committer(
        modality_name: str, tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> str:
        """Analysis that commits a figure."""
        pid, _, _ = dm.plot_manager.add_plot(plot=_fig(), title="auto", source="test")
        activity = _log(dm, "committer")
        dm.plot_manager.commit_staged(
            activity, tool_call_id=tool_call_id, staged_ids=[pid]
        )
        return "scored and drew"

    @tool
    def error_string(modality_name: str) -> str:
        """Returns an error as a normal reply."""
        return "Error: modality not found"

    @tool
    def raiser(modality_name: str) -> str:
        """Raises."""
        raise ValueError("boom")

    @tool
    def list_things() -> str:
        """Utility."""
        return ", ".join(dm.list_modalities())

    @tool
    def execute_custom_code(code: str) -> str:
        """Stand-in for the trusted submitter (key only)."""
        return "ran"

    async def _dual_async(modality_name: str) -> str:
        out = dm.workspace_path / "exports"
        out.mkdir(exist_ok=True)
        (out / "dual.png").write_bytes(b"\x89PNG fake")
        return "scored and drew"

    def _dual_sync(modality_name: str) -> str:
        return "scored and drew"

    from langchain_core.tools import StructuredTool

    dual = StructuredTool.from_function(
        func=_dual_sync,
        coroutine=_dual_async,
        name="dual_tool",
        description="Dual interface: harmless sync, async writes a PNG.",
    )

    @tool
    def data_only_json_writer(modality_name: str) -> str:
        """Writes a data-only Plotly figure (no layout) outside the store."""
        out = dm.workspace_path / "results"
        out.mkdir(exist_ok=True)
        figure = {"data": [{"type": "bar", "x": ["a"], "y": [1]}]}
        (out / "bars.json").write_text(json.dumps(figure))
        (out / "numbers.json").write_text(json.dumps({"data": [1, 2, 3]}))
        (out / "empty.json").write_text(json.dumps({"data": []}))
        (out / "picture.svg").write_text("<svg/>")
        return "scored and drew"

    @tool
    def suggest_figures(modality_name: str) -> str:
        """Analysis that suggests templates (setup-dependent suggestions)."""
        return "scored"

    suggestions = getattr(dm, "_d8_suggestions", None)
    for t in (
        score_cells,
        png_writer,
        json_figure_writer,
        stager,
        committer,
        error_string,
        raiser,
        dual,
        data_only_json_writer,
        suggest_figures,
    ):
        _meta(t, "ANALYZE")
    if suggestions is not None:
        suggest_figures.metadata["suggests_templates"] = suggestions
    list_things.metadata = {"categories": ["UTILITY"], "provenance": False}
    execute_custom_code.metadata = {"categories": ["CODE_EXEC"], "provenance": False}
    execute_custom_code.func.__module__ = TRUSTED
    return [
        score_cells,
        png_writer,
        json_figure_writer,
        stager,
        committer,
        error_string,
        raiser,
        list_things,
        execute_custom_code,
        dual,
        data_only_json_writer,
        suggest_figures,
        _producer(dm, "create_umap_plot"),
        _producer(dm, "create_heatmap", order="commit_first"),
        _producer(dm, "create_dot_plot", mutate=True),
        _producer(dm, "create_elbow_plot", wrong_id=True),
    ]


def bad_agent(
    data_manager, callback_handler, agent_name, delegation_tools, workspace_path
):
    return _graph(_tools(data_manager))


GOOD = {"score_cells", "list_things", "execute_custom_code", "create_umap_plot"}


def good_agent(
    data_manager, callback_handler, agent_name, delegation_tools, workspace_path
):
    return _graph([t for t in _tools(data_manager) if t.name in GOOD])


@pytest.fixture(autouse=True)
def _fake_module(monkeypatch):
    mod = types.ModuleType(FAKE_MODULE)
    mod.bad_agent, mod.good_agent, mod.llm_agent = bad_agent, good_agent, llm_agent
    monkeypatch.setitem(sys.modules, FAKE_MODULE, mod)


def _seed(dm, mp):
    dm.store_modality("m", _adata())


def _says(text):
    def check(reply, dm):
        assert text in str(reply), f"expected {text!r}"

    return check


def _run(tmp_path, name, args=None, expect="cells", factory=bad_agent):
    case = FigureToolCase(name, args or {"modality_name": "m"}, _seed, _says(expect))
    tmp_path.mkdir(parents=True, exist_ok=True)
    return check_figure_tool_case(
        case,
        factory=factory,
        factory_module_path=FAKE_MODULE,
        workspace=tmp_path,
        tool_call_id="call_1",
    )


# --- layer 1: the exact set --------------------------------------------------------------


def test_the_transitional_set_is_exactly_the_eight_retained_wrappers():
    assert aquadif.TRANSITIONAL_FIGURE_PRODUCERS == {
        f"{VIZ}.{n}"
        for n in (
            "create_umap_plot",
            "create_qc_plots",
            "create_violin_plot",
            "create_feature_plot",
            "create_dot_plot",
            "create_heatmap",
            "create_elbow_plot",
            "create_cluster_composition_plot",
        )
    }
    assert aquadif.TRUSTED_FIGURE_SUBMITTERS == {f"{TRUSTED}.execute_custom_code"}
    assert aquadif.FIGURE_CONTRACT_COVERED == aquadif.PROVENANCE_REQUIRED
    assert aquadif.FIGURE_CONTRACT_NOT_COVERED == {
        aquadif.AquadifCategory.DELEGATE,
        aquadif.AquadifCategory.UTILITY,
    }


@pytest.mark.parametrize(
    "key",
    [
        "lobster.agents.other_expert.create_umap_plot",
        f"{VIZ}.create_umap_plot_v2",
        f"{VIZ}.create_*",
        "create_umap_plot",
        f"{VIZ}.visualization_expert.<locals>.create_umap_plot",
    ],
)
def test_no_pattern_or_same_name_tool_is_a_producer(key):
    assert not aquadif.is_transitional_figure_producer(key)


def test_the_key_is_the_unwrapped_implementation_module_with_coroutine_first():
    import functools

    async def body(x: int) -> str:
        """Async body."""
        return "x"

    @functools.wraps(body)
    async def wrapped(x: int) -> str:
        return await body(x)

    body.__module__ = VIZ
    t = tool("create_heatmap")(wrapped)
    assert t.func is None or t.coroutine is not None
    assert aquadif.figure_tool_key(t) == f"{VIZ}.create_heatmap"


# --- layer 2: observed behaviour per case -------------------------------------------------


def test_a_clean_analysis_case_holds(tmp_path):
    assert _run(tmp_path, "score_cells") == []


@pytest.mark.parametrize(
    "name, expected",
    [
        ("png_writer", "wrote a figure file: exports/umap.png"),
        ("json_figure_writer", "wrote a Plotly figure JSON: results/fig.json"),
        ("stager", "staged 1 figure(s) (add_plot)"),
        ("committer", "commit_staged called 1 time(s)"),
        ("committer", "store_commit called 1 time(s)"),
        ("committer", "the figure catalog changed"),
        ("committer", "changed the managed figure folder"),
    ],
)
def test_an_analysis_tool_that_creates_a_figure_fails(tmp_path, name, expected):
    problems = _run(tmp_path, name, expect="drew")
    assert any(expected in p for p in problems), problems


def test_plain_json_is_not_a_figure(tmp_path):
    problems = _run(tmp_path, "json_figure_writer", expect="drew")
    assert not any("table.json" in p for p in problems)


def test_an_error_string_reply_is_a_failure_not_a_pass(tmp_path):
    problems = _run(tmp_path, "error_string")
    assert any("assert_success failed" in p for p in problems), problems


def test_a_raising_tool_is_a_failure(tmp_path):
    assert _run(tmp_path, "raiser") == ["raiser: the tool raised ValueError"]


def test_a_producer_saving_through_stage_log_commit_holds(tmp_path):
    assert _run(tmp_path, "create_umap_plot", expect="plotted") == []


def test_a_producer_preview_commits_nothing_and_holds(tmp_path):
    args = {"modality_name": "m", "save_plot": False}
    assert _run(tmp_path, "create_umap_plot", args, expect="preview") == []


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "create_heatmap",
            "log_tool_usage did not run between staging and commit_staged",
        ),
        ("create_dot_plot", "modality 'm' changed: obs"),
        ("create_elbow_plot", "commit_staged did not carry the injected tool_call_id"),
    ],
)
def test_a_producer_breaking_the_contract_fails(tmp_path, name, expected):
    problems = _run(tmp_path, name, expect="plotted")
    assert any(expected in p for p in problems), problems


def test_an_unreadable_catalog_fails_the_case_never_passes(tmp_path):
    (tmp_path / "plots").write_text("not a folder")
    with pytest.raises(Exception) as info:
        _run(tmp_path, "score_cells")
    assert type(info.value).__name__ == "StorageError"


def test_a_case_for_an_unexposed_tool_fails(tmp_path):
    problems = _run(tmp_path, "no_such_tool")
    assert problems == [
        "no_such_tool: tool 'no_such_tool' is not exposed by the factory"
    ]


# --- coverage --------------------------------------------------------------------------------


class _Coverage(FigureContractTestMixin):
    agent_module = FAKE_MODULE
    factory_name = "good_agent"

    def __init__(self, cases, skips=None):
        self._cases = cases
        self.figure_contract_skips = skips or {}

    def figure_tool_cases(self):
        return self._cases


def _case(name, **args):
    return FigureToolCase(name, args or {"modality_name": "m"}, _seed, _says(""))


def test_coverage_holds_and_reports_what_is_not_counted(tmp_path):
    cov = _Coverage([_case("score_cells"), _case("create_umap_plot")])
    problems, report = cov.figure_contract_coverage(tmp_path)
    assert problems == []
    assert "inventory across 2 supplied case setup(s)" in report
    assert (
        "with cases (run by test_figure_tool_cases_hold): ['create_umap_plot', 'score_cells']"
        in report
    )
    assert "not covered (non-provenance primary): ['list_things (UTILITY)']" in report
    assert "excluded trusted submitter, not counted: ['execute_custom_code']" in report


def test_a_covered_tool_without_a_case_or_skip_fails(tmp_path):
    problems, _ = _Coverage([_case("create_umap_plot")]).figure_contract_coverage(
        tmp_path
    )
    assert problems == ["score_cells: no case and no skip (ANALYZE)"]


def test_a_skip_with_a_reason_is_reported_not_hidden(tmp_path):
    cov = _Coverage([_case("create_umap_plot")], {"score_cells": "needs GPU fixture"})
    problems, report = cov.figure_contract_coverage(tmp_path)
    assert problems == []
    assert (
        "skipped (reasons in source, not echoed): ['score_cells (ANALYZE)']" in report
    )
    assert "needs GPU fixture" not in report


def test_a_producer_cannot_be_skipped_and_a_preview_does_not_count(tmp_path):
    cov = _Coverage(
        [
            _case("score_cells"),
            _case("create_umap_plot", modality_name="m", save_plot=False),
        ],
        {"create_umap_plot": "slow"},
    )
    problems, _ = cov.figure_contract_coverage(tmp_path)
    assert "create_umap_plot: a transitional producer cannot be skipped" in problems
    assert "create_umap_plot: a transitional producer needs a saving case" in problems


def test_stale_cases_and_skips_fail(tmp_path):
    cov = _Coverage(
        [_case("score_cells"), _case("create_umap_plot"), _case("gone")], {"old": "x"}
    )
    problems, _ = cov.figure_contract_coverage(tmp_path)
    assert "case gone: tool not exposed in its own setup" in problems
    assert "skip names an unexposed tool: old" in problems


def test_a_package_without_cases_fails_loudly():
    class NoCases(FigureContractTestMixin):
        agent_module = FAKE_MODULE
        factory_name = "good_agent"

    with pytest.raises(NotImplementedError):
        NoCases().figure_tool_cases()


# --- the mixin as a package would use it ------------------------------------------------------


class TestGoodAgentFigureContract(FigureContractTestMixin):
    agent_module = FAKE_MODULE
    factory_name = "good_agent"

    def figure_tool_cases(self):
        return [
            FigureToolCase(
                "score_cells", {"modality_name": "m"}, _seed, _says("scored")
            ),
            FigureToolCase(
                "create_umap_plot", {"modality_name": "m"}, _seed, _says("plotted")
            ),
            FigureToolCase(
                "create_umap_plot",
                {"modality_name": "m", "save_plot": False},
                _seed,
                _says("preview"),
                label="create_umap_plot (preview)",
            ),
        ]


# --- repair REDs -------------------------------------------------------


def _seed_mode(mode, **extra):
    def setup(dm, mp):
        _seed(dm, mp)
        dm._d8_mode = mode
        for key, value in extra.items():
            setattr(dm, key, value)

    return setup


def _run_case(tmp_path, case, factory=bad_agent):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return check_figure_tool_case(
        case,
        factory=factory,
        factory_module_path=FAKE_MODULE,
        workspace=tmp_path,
        tool_call_id="call_1",
    )


def test_a_dual_tool_runs_the_coroutine_the_key_names(tmp_path):
    problems = _run(tmp_path, "dual_tool", expect="drew")
    assert any("wrote a figure file: exports/dual.png" in p for p in problems), problems


def test_a_case_llm_patch_wins_over_the_default_guard(tmp_path):
    from lobster.config import llm_factory

    def setup(dm, mp):
        _seed(dm, mp)
        mp.setattr(llm_factory, "create_llm", lambda *a, **k: "SENTINEL_LLM")

    def llm_was_the_sentinel(reply, dm):
        assert LLM_SEEN[-1] == "SENTINEL_LLM"

    case = FigureToolCase(
        "score_cells", {"modality_name": "m"}, setup, llm_was_the_sentinel
    )
    assert _run_case(tmp_path, case, factory=llm_agent) == []


def _seed_with_saved_figure(mode):
    def setup(dm, mp):
        _seed(dm, mp)
        pid, _, _ = dm.plot_manager.add_plot(plot=_fig(), title="old", source="test")
        [res] = dm.plot_manager.commit_staged(
            {"id": "setup"}, tool_call_id="setup_call", staged_ids=[pid]
        )
        fid, rid = res.ref.split("@")
        dm._d8_old_path = dm.workspace_path / "plots" / fid / rid / "figure.json"
        dm._d8_mode = mode

    return setup


@pytest.mark.parametrize(
    "setup, expected",
    [
        (
            _seed_with_saved_figure("overwrite_old"),
            "changed the managed figure folder: plots/f_",
        ),
        (
            _seed_mode("extra_legacy"),
            "changed the managed figure folder: plots/extra_legacy.html",
        ),
    ],
)
def test_a_producer_cannot_change_the_store_outside_its_own_commit(
    tmp_path, setup, expected
):
    case = FigureToolCase(
        "create_umap_plot", {"modality_name": "m"}, setup, _says("plotted")
    )
    problems = _run_case(tmp_path, case)
    assert any(expected in p for p in problems), problems


def test_a_write_inside_the_commit_outside_the_returned_refs_fails(tmp_path):
    from lobster.core.figures.store import LocalFigureStore

    def setup(dm, mp):
        _seed(dm, mp)
        original = LocalFigureStore.commit

        def sneaky(self, *a, **k):
            out = original(self, *a, **k)
            (dm.workspace_path / "plots" / "sneaky.html").write_text("<html/>")
            return out

        mp.setattr(LocalFigureStore, "commit", sneaky)

    case = FigureToolCase(
        "create_umap_plot", {"modality_name": "m"}, setup, _says("plotted")
    )
    problems = _run_case(tmp_path, case)
    assert any(
        "changed the managed figure folder: plots/sneaky.html" in p for p in problems
    )


def _seed_leftover(dm, mp):
    _seed(dm, mp)
    dm._d8_leftover, _, _ = dm.plot_manager.add_plot(plot=_fig(), title="x", source="t")
    dm._d8_mode = "leftover_sid"


@pytest.mark.parametrize(
    "setup, expected",
    [
        (_seed_leftover, "staged_ids are not the figures staged in this call"),
        (
            _seed_mode("forged_activity"),
            "activity is not the activity logged in this call",
        ),
        (_seed_mode("no_activity"), "activity is not the activity logged in this call"),
    ],
)
def test_a_producer_commit_must_bind_this_calls_figures_and_activity(
    tmp_path, setup, expected
):
    case = FigureToolCase(
        "create_umap_plot", {"modality_name": "m"}, setup, _says("plotted")
    )
    problems = _run_case(tmp_path, case)
    assert any(expected in p for p in problems), problems


def test_availability_before_the_log_boundary_fails(tmp_path):
    problems = _run(tmp_path, "create_heatmap", expect="plotted")
    assert any(
        "the catalog changed before log_tool_usage" in p for p in problems
    ), problems


def test_a_reader_showing_a_preview_as_saved_fails(tmp_path):
    from lobster.core.plot_manager import PlotManager

    def setup(dm, mp):
        _seed(dm, mp)
        mp.setattr(
            PlotManager,
            "get_plot_by_id",
            lambda self, pid: (None, {"found": True, "plot_id": pid}, None),
        )

    case = FigureToolCase(
        "create_umap_plot",
        {"modality_name": "m", "save_plot": False},
        setup,
        _says("preview"),
    )
    problems = _run_case(tmp_path, case)
    assert any("get_plot_by_id shows staged figure" in p for p in problems), problems


def test_data_only_plotly_json_is_a_figure_and_the_bounds_are_stated(tmp_path):
    problems = _run(tmp_path, "data_only_json_writer", expect="drew")
    assert any("wrote a Plotly figure JSON: results/bars.json" in p for p in problems)
    for not_a_figure in ("numbers.json", "empty.json", "picture.svg"):
        assert not any(not_a_figure in p for p in problems), problems


def test_figure_shaped_json_the_strict_parser_refuses_cannot_be_cleared():
    from lobster.testing.contract_mixins import _plotly_json_problem

    text = b'{"data": [{"type": "bar", "x": ["a"], "y": [NaN]}], "layout": {}}'
    assert "cannot be cleared" in (_plotly_json_problem(text) or "")
    assert _plotly_json_problem(b'{"rows": [1, 2]}') is None


class _Dm:
    def __init__(self, adata):
        self.adata = adata

    def list_modalities(self):
        return ["m"]

    def get_modality(self, name):
        return self.adata


def _digest(adata):
    from lobster.testing.contract_mixins import _dataset_digests

    return _dataset_digests(_Dm(adata))["m"]


def _ordered(a):
    a.obs["group"] = a.obs["group"].cat.as_ordered()


def _index_name(a):
    a.obs.index.name = "cell"


def _columns_name(a):
    a.var.columns.name = "fields"


def _raw_varm(a):
    a.raw.varm["loadings"] = np.ones((a.raw.n_vars, 2))


@pytest.mark.parametrize(
    "change, field",
    [
        (_ordered, "obs"),
        (_index_name, "obs"),
        (_columns_name, "var"),
        (_raw_varm, "raw"),
    ],
)
def test_digests_see_ordered_index_column_names_and_raw_varm(change, field):
    a = _adata()
    a.var["score"] = 1.0
    a.raw = a
    before = _digest(a)
    change(a)
    after = _digest(a)
    assert before[field] != after[field]


def _with_suggestions(names):
    def setup(dm, mp):
        _seed(dm, mp)
        dm._d8_suggestions = names

    return setup


def _templates(monkeypatch, names, calls=None):
    mod = types.ModuleType("lobster.figure_templates")

    def list_templates():
        if calls is not None:
            calls.append(1)
        return [{"name": n, "version": 1} for n in names]

    mod.list_templates = list_templates
    monkeypatch.setitem(sys.modules, "lobster.figure_templates", mod)


class _TemplateCoverage(FigureContractTestMixin):
    agent_module = FAKE_MODULE
    factory_name = "bad_agent"
    figure_contract_skips = {
        n: "fixture covered elsewhere"
        for n in (
            "score_cells",
            "png_writer",
            "json_figure_writer",
            "stager",
            "committer",
            "error_string",
            "raiser",
            "dual_tool",
            "data_only_json_writer",
        )
    }

    def __init__(self, setups):
        self._setups = setups

    def figure_tool_cases(self):
        cases = [
            FigureToolCase("suggest_figures", {"modality_name": "m"}, s, _says(""))
            for s in self._setups
        ]
        for name in (
            "create_umap_plot",
            "create_heatmap",
            "create_dot_plot",
            "create_elbow_plot",
        ):
            cases.append(FigureToolCase(name, {"modality_name": "m"}, _seed, _says("")))
        return cases


def _template_problems_of(tmp_path, setups):
    problems, _ = _TemplateCoverage(setups).figure_contract_coverage(tmp_path)
    return [p for p in problems if "template" in p]


def test_suggested_templates_resolve_and_a_typo_fails(tmp_path, monkeypatch):
    _templates(monkeypatch, ["proteomics.volcano"])
    ok = _template_problems_of(
        tmp_path / "a", [_with_suggestions(["proteomics.volcano"])]
    )
    assert ok == []
    typo = _template_problems_of(
        tmp_path / "b", [_with_suggestions(["proteomics.volcanoo"])]
    )
    assert typo == [
        "suggest_figures: suggested template not in list_templates(): proteomics.volcanoo"
    ]


def test_every_observed_suggestion_is_validated_not_the_last_writer(
    tmp_path, monkeypatch
):
    _templates(monkeypatch, ["proteomics.volcano"])
    problems = _template_problems_of(
        tmp_path,
        [
            _with_suggestions(["proteomics.volcanoo"]),
            _with_suggestions(["proteomics.volcano"]),
        ],
    )
    assert problems == [
        "suggest_figures: suggested template not in list_templates(): proteomics.volcanoo"
    ]


def test_an_unavailable_template_catalog_fails_and_no_suggestion_means_no_lookup(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "lobster.figure_templates", None)
    missing = _template_problems_of(
        tmp_path / "a", [_with_suggestions(["proteomics.volcano"])]
    )
    assert missing == [
        "suggest_figures: suggests_templates is set but the template catalog "
        "(lobster.figure_templates.list_templates) is not importable"
    ]
    calls = []
    _templates(monkeypatch, [], calls)
    assert _template_problems_of(tmp_path / "b", [_seed]) == [] and calls == []


def test_coverage_uses_each_cases_own_setup(tmp_path):
    seen = []

    def setup(dm, mp):
        _seed(dm, mp)
        seen.append(str(dm.workspace_path))

    cov = _Coverage(
        [
            FigureToolCase("score_cells", {"modality_name": "m"}, setup, _says("")),
            FigureToolCase(
                "create_umap_plot", {"modality_name": "m"}, setup, _says("")
            ),
        ]
    )
    problems, report = cov.figure_contract_coverage(tmp_path)
    assert problems == [] and len(set(seen)) == 2


def test_no_cases_uses_one_empty_baseline_and_all_skipped_holds(tmp_path):
    cov = _Coverage([], {"score_cells": "no fixture yet"})
    problems, report = cov.figure_contract_coverage(tmp_path)
    assert problems == ["create_umap_plot: a transitional producer needs a saving case"]
    assert "inventory: one empty-DataManagerV2 baseline (no cases)" in report


# --- multiplicity, result accounting, missing activity, skip reasons -----------


def _patched_commit(transform, mode):
    """Setup: wrap the REAL commit_staged (de-duplicated ids) and transform its results."""
    from lobster.core.plot_manager import PlotManager

    def setup(dm, mp):
        _seed(dm, mp)
        dm._d8_mode = mode
        original = PlotManager.commit_staged

        def wrapped(self, activity, *, tool_call_id, staged_ids, **kw):
            unique = list(dict.fromkeys(staged_ids))
            results = original(
                self, activity, tool_call_id=tool_call_id, staged_ids=unique, **kw
            )
            return transform(list(results), list(staged_ids))

        mp.setattr(PlotManager, "commit_staged", wrapped)

    return setup


def _one_per_given_id(results, staged_ids):
    return [results[0] for _ in staged_ids]


def _extra_pending(results, staged_ids):
    from lobster.core.figures.records import CommitResult

    return results + [CommitResult(len(results), "pending")]


def _reversed(results, staged_ids):
    return list(reversed(results))


@pytest.mark.parametrize(
    "setup, expected",
    [
        (
            _patched_commit(_one_per_given_id, "duplicate_sid"),
            "staged_ids are not the figures staged in this call, each once",
        ),
        (
            _patched_commit(_extra_pending, "ok"),
            "commit_staged returned 2 result(s) for 1 id(s)",
        ),
        (
            _patched_commit(_reversed, "two_figures"),
            "commit results are not one per staged id in order",
        ),
    ],
)
def test_staged_ids_once_and_complete_ordered_results(tmp_path, setup, expected):
    case = FigureToolCase(
        "create_umap_plot", {"modality_name": "m"}, setup, _says("plotted")
    )
    problems = _run_case(tmp_path, case)
    assert any(expected in p for p in problems), problems


def test_a_two_figure_producer_holds(tmp_path):
    case = FigureToolCase(
        "create_umap_plot",
        {"modality_name": "m"},
        _seed_mode("two_figures"),
        _says("plotted"),
    )
    assert _run_case(tmp_path, case) == []


def test_log_tool_usage_returning_no_activity_fails(tmp_path):
    from lobster.core.runtime.data_manager import DataManagerV2

    def setup(dm, mp):
        _seed(dm, mp)
        original = DataManagerV2.log_tool_usage

        def returns_none(self, *a, **k):
            original(self, *a, **k)
            return None

        mp.setattr(DataManagerV2, "log_tool_usage", returns_none)

    case = FigureToolCase(
        "create_umap_plot", {"modality_name": "m"}, setup, _says("plotted")
    )
    problems = _run_case(tmp_path, case)
    assert any("log_tool_usage returned no activity" in p for p in problems), problems


@pytest.mark.parametrize("reason", [None, False, 0, "", "  ", ["x"]])
def test_a_skip_reason_must_be_a_non_empty_string(tmp_path, reason):
    cov = _Coverage([_case("create_umap_plot")], {"score_cells": reason})
    problems, _ = cov.figure_contract_coverage(tmp_path)
    assert "score_cells: skip needs a non-empty string reason" in problems
