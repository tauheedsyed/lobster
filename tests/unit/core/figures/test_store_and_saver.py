"""Local figure store + trusted saver (the figure contract).

Real filesystem tests (tmp_path), no mocks of the store.
"""

from __future__ import annotations

import json
import os

import numpy as np
import plotly.graph_objects as go
import pytest

from lobster.core.figures import paths as P
from lobster.core.figures.errors import (
    AttemptConflictError,
    FigureNotFoundError,
    IntegrityError,
)
from lobster.core.figures.ids import FigureRef, is_valid_id, new_id, scan_refs
from lobster.core.figures.payload import (
    config_to_bytes,
    figure_to_bytes,
    sha256_hex,
    source_to_bytes,
)
from lobster.core.figures.records import (
    Candidate,
    ExecutionInfo,
    InputEdge,
    SourcePayload,
)
from lobster.core.figures.saver import TrustedSaver, summarize
from lobster.core.figures.store import LocalFigureStore


def _fig(color: str = "red", title: str = "Volcano") -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([0.1, 0.5, 2.0]),
            uid="class:up",
            customdata=[["P1"], ["P2"], ["P3"]],
            meta={
                "lobster_ids": {
                    "entity_kind": "protein",
                    "comparison": "t_vs_c",
                    "customdata_index": 0,
                }
            },
            marker={"color": color},
        )
    )
    fig.update_layout(
        title=title,
        meta={
            "lobster_template": {
                "name": "proteomics.volcano",
                "version": "1",
                "args": {"fdr": 0.05},
            }
        },
    )
    return fig


def _info(
    code: bytes | None = b"fig = 1\n",
    activity: dict | None = None,
    base: str = "s1:call_1",
) -> ExecutionInfo:
    return ExecutionInfo(
        attempt_base=base,
        tool="execute_custom_code",
        agent="proteomics_de_analysis_expert",
        parameters={"modality_name": "prot_de"},
        persist=False,
        code_bytes=code,
        activity=(
            {
                "id": "act-1",
                "tool_name": "execute_custom_code",
                "ir": {"operation": "custom"},
            }
            if activity is None
            else activity
        ),
        engine_version="1.2.0.dev0",
        plotly_version="7.1.0",
    )


def _cand(
    index: int = 0, fig: go.Figure | None = None, base: str = "s1:call_1", **kw
) -> Candidate:
    return Candidate(
        attempt_key=f"{base}:{index}",
        candidate_index=index,
        figure_bytes=figure_to_bytes(fig or _fig()),
        config_bytes=config_to_bytes({"displaylogo": False}),
        actor="expert:proteomics_de_analysis_expert",
        label=kw.pop("label", "Volcano — treated vs control"),
        inputs=kw.pop("inputs", (InputEdge("prot_de", "inst-1", "store_event", True),)),
        **kw,
    )


@pytest.fixture()
def store(tmp_path):
    return LocalFigureStore(tmp_path, session_id="s1")


def test_ids_are_opaque_and_refs_scan_with_token_boundaries():
    fid = new_id("f")
    assert is_valid_id(fid) and fid.startswith("f_") and len(fid) == 12
    ref = FigureRef.parse("f_abcdefghij@r_klmnopqrst")
    assert str(ref) == "f_abcdefghij@r_klmnopqrst"
    code = 'a = load_figure("f_abcdefghij@r_klmnopqrst"); b = "kevin.yar@omics-os.com"'
    refs = [str(r) for r in scan_refs(code)]
    assert "f_abcdefghij@r_klmnopqrst" in refs  # literal ref found
    assert all(
        "." not in r for r in refs
    )  # email-like tokens are only shape-matched (and ignored at lookup)


@pytest.mark.parametrize("bad", ["f_1\n", "f_1\r\n", "\nf_1", "f_1\n\n"])
def test_an_id_with_a_newline_is_refused_everywhere(bad):
    """is_valid_id uses fullmatch, so no final newline passes, and neither
    FigureRef component accepts one."""
    assert not is_valid_id(bad)
    with pytest.raises(ValueError):
        FigureRef.parse(f"{bad}@r_1")
    with pytest.raises(ValueError):
        FigureRef.parse(f"f_1@{bad}")
    with pytest.raises(ValueError):
        FigureRef("f_1", bad)
    with pytest.raises(ValueError):
        FigureRef(bad, "r_1")


def test_valid_id_boundaries_and_opaque_ids_are_unchanged():
    for good in ("a", "0", "-", "_", "x" * 64, "abc-def_123", "not-prefixed", "r_x"):
        assert is_valid_id(good), good
        assert str(FigureRef.parse(f"{good}@{good}")) == f"{good}@{good}"
    for bad in ("", "x" * 65, "A", "a b", "a.b", "a@b", None, 1, b"abc", ["a"]):
        assert not is_valid_id(bad), bad


def test_bytes_are_stored_exactly_with_uid_typed_arrays_and_template_meta(store):
    cand = _cand()
    [res] = TrustedSaver(store).commit_attempt(_info(), [cand], succeeded=True)
    assert res.status == "available"
    rev_bytes, fig_bytes, cfg_bytes = store.read(res.ref)
    assert fig_bytes == cand.figure_bytes  # no re-serialization, nothing stripped
    parsed = json.loads(fig_bytes)
    assert parsed["data"][0]["uid"] == "class:up"
    assert "bdata" in parsed["data"][0]["x"]
    assert parsed["layout"]["meta"]["lobster_template"]["name"] == "proteomics.volcano"
    record = json.loads(rev_bytes)
    assert record["execution"]["template"] == {
        "name": "proteomics.volcano",
        "version": "1",
        "args": {"fdr": 0.05},
    }
    assert record["figure"]["sha256"] == sha256_hex(cand.figure_bytes)
    assert record["config"]["sha256"] == sha256_hex(cfg_bytes)


def test_non_strict_json_is_refused_not_repaired(store):
    bad = b'{"data":[{"type":"scatter","x":[1,NaN]}],"layout":{}}'
    cand = Candidate("s1:call_1:0", 0, bad, b"{}")
    [res] = TrustedSaver(store).commit_attempt(_info(), [cand], succeeded=True)
    assert res.status == "refused" and "non_strict_json" in res.reason
    assert not (
        store.root / P.EXECUTIONS_DIR
    ).exists()  # nothing prepared -> no execution record


def test_failed_attempt_publishes_nothing(store):
    results = TrustedSaver(store).commit_attempt(
        _info(), [_cand(0), _cand(1)], succeeded=False, failure_reason="timeout"
    )
    assert [r.status for r in results] == ["not_committed", "not_committed"]
    assert store.list().items == []
    assert not store.root.exists() or not any(store.root.iterdir())
    assert summarize(results).startswith("No figure saved.")


def test_retry_same_key_same_content_returns_original_ref(store):
    saver = TrustedSaver(store)
    [first] = saver.commit_attempt(_info(), [_cand()], succeeded=True)
    [again] = saver.commit_attempt(_info(), [_cand()], succeeded=True)
    assert again.status == "available" and again.ref == first.ref
    assert len(store.list().items) == 1
    assert (
        len(list((store.root / P.EXECUTIONS_DIR).glob("x_*.json")))
        - len(list((store.root / P.EXECUTIONS_DIR).glob("*.activity.json")))
        == 1
    )  # execution record reused


def test_retry_same_key_different_content_conflicts(store):
    saver = TrustedSaver(store)
    saver.commit_attempt(_info(), [_cand(fig=_fig("red"))], succeeded=True)
    [res] = saver.commit_attempt(_info(), [_cand(fig=_fig("blue"))], succeeded=True)
    # Changed content changes the planned (index, digest) set: the whole plan conflicts.
    assert res.status == "refused"
    assert res.reason == "execution_record:attempt_conflict:execution_plan_changed"
    assert len(store.list().items) == 1
    with pytest.raises(FigureNotFoundError):
        store.commit(
            _cand(fig=_fig("green")), execution_id="x_unused", digest="different"
        )


def test_two_figures_in_one_attempt_stay_distinct(store):
    results = TrustedSaver(store).commit_attempt(
        _info(), [_cand(0, _fig("red")), _cand(1, _fig("blue"))], succeeded=True
    )
    refs = {r.ref for r in results}
    assert len(refs) == 2 and all(r.status == "available" for r in results)
    figure_ids = {FigureRef.parse(r).figure_id for r in refs}
    assert len(figure_ids) == 2


def test_execution_record_precedes_revisions_and_stores_exact_activity(store):
    [res] = TrustedSaver(store).commit_attempt(_info(), [_cand()], succeeded=True)
    record = store.read_record(res.ref)
    execution = store.read_execution(record["execution"]["execution_id"])
    assert execution["activity_id"] == "act-1"
    activity_path = store.workspace / execution["activity_record"]["path"]
    assert (
        sha256_hex(activity_path.read_bytes()) == execution["activity_record"]["sha256"]
    )
    assert json.loads(activity_path.read_bytes())["ir"] == {"operation": "custom"}
    code_path = store.root / P.CODE_DIR / f"{execution['code_id']}.py"
    assert code_path.read_bytes() == b"fig = 1\n"  # code kept even with persist=False
    assert execution["persist"] is False
    assert os.path.getmtime(
        store.workspace / execution["activity_record"]["path"]
    ) <= os.path.getmtime(store._rev_dir(FigureRef.parse(res.ref)) / P.REVISION_FILE)


def test_provenance_disabled_is_recorded_honestly(store):
    info = ExecutionInfo(
        "s1:call_9",
        "execute_custom_code",
        None,
        {},
        False,
        None,
        None,
        "1.2.0.dev0",
        "7.1.0",
    )
    [res] = TrustedSaver(store).commit_attempt(
        info, [_cand(base="s1:call_9")], succeeded=True
    )
    execution = store.read_execution(
        store.read_record(res.ref)["execution"]["execution_id"]
    )
    assert (
        execution["activity_id"] is None
        and execution["activity_unavailable"] == "provenance_disabled"
    )


def test_per_candidate_statuses_survive_a_bad_sibling(store):
    bad = Candidate("s1:call_1:1", 1, b"not json", b"{}")
    results = TrustedSaver(store).commit_attempt(
        _info(), [_cand(0), bad], succeeded=True
    )
    assert results[0].status == "available"
    assert results[1].status == "refused"
    text = summarize(results)
    assert "Created" in text and not text.startswith("No figure saved.")


def test_tampered_bytes_are_detected_on_read(store):
    [res] = TrustedSaver(store).commit_attempt(_info(), [_cand()], succeeded=True)
    ref = FigureRef.parse(res.ref)
    fig_path = store._rev_dir(ref) / P.FIGURE_FILE
    os.chmod(fig_path, 0o644)
    fig_path.write_bytes(fig_path.read_bytes().replace(b"Volcano", b"Hacked!"))
    with pytest.raises(IntegrityError):
        store.read(ref)


def test_edit_keeps_parent_bytes_and_inherits_verified_inputs(store):
    saver = TrustedSaver(store)
    [r1] = saver.commit_attempt(_info(), [_cand()], succeeded=True)
    r1_bytes = store.read(r1.ref)[1]
    edit = Candidate(
        attempt_key="s1:omni_1:0",
        candidate_index=0,
        figure_bytes=figure_to_bytes(_fig("orange", "Treated vs control")),
        config_bytes=b"{}",
        kind="edited",
        actor="omni:editor",
        parent_ref=r1.ref,
        label="Volcano — treated vs control",
        inputs=(InputEdge("unrelated", None, "declared", False),),
    )
    [r2] = saver.commit_attempt(
        _info(code=None, base="s1:omni_1"), [edit], succeeded=True
    )
    assert FigureRef.parse(r2.ref).figure_id == FigureRef.parse(r1.ref).figure_id
    assert store.read(r1.ref)[1] == r1_bytes  # R1 unchanged
    rec2 = store.read_record(r2.ref)
    assert rec2["parents"] == [FigureRef.parse(r1.ref).revision_id]
    names = {(e["name"], e["verified"]) for e in rec2["inputs"]}
    assert ("prot_de", True) in names and ("unrelated", False) in names
    assert [ref for ref, verified in store.dependents("inst-1")] == [r1.ref, r2.ref]


def test_listing_is_restart_safe_and_skips_debris(store, tmp_path):
    TrustedSaver(store).commit_attempt(_info(), [_cand()], succeeded=True)
    (store.root / "f_debris0000" / "r_debris0000").mkdir(
        parents=True
    )  # no revision.json -> not available
    (store.root / "plot_1_legacy.html").write_text("<html></html>")
    fresh = LocalFigureStore(tmp_path, session_id="s1")
    page = fresh.list()
    assert len(page.items) == 1 and page.next_cursor is None


def test_interrupted_commit_is_completed_by_retry(store):
    saver = TrustedSaver(store)
    [res] = saver.commit_attempt(_info(), [_cand()], succeeded=True)
    ref = FigureRef.parse(res.ref)
    rev_path = store._rev_dir(ref) / P.REVISION_FILE
    os.unlink(rev_path)  # simulate crash before the availability marker
    assert store.list().items == []
    # the execution record is x_<id>.json (exactly one dot); x_<id>.activity.json is not
    [record] = [
        p
        for p in (store.root / P.EXECUTIONS_DIR).glob("x_*.json")
        if p.name.count(".") == 1
    ]
    exec_id = json.loads(record.read_bytes())["execution_id"]
    assert store.execution_outcomes(exec_id)[0]["status"] == "not_committed"
    [again] = saver.commit_attempt(_info(), [_cand()], succeeded=True)
    assert again.ref == res.ref and rev_path.is_file()


def test_source_is_content_addressed_and_verified(store):
    src = SourcePayload(
        source_to_bytes([{"protein": "P1", "log2fc": 1.2, "padj": float("nan")}]),
        {"comparison": "t_vs_c"},
    )
    [res] = TrustedSaver(store).commit_attempt(
        _info(), [_cand(source=src)], succeeded=True
    )
    source_id = store.read_record(res.ref)["source"]["source_id"]
    payload = json.loads(store.read_source(source_id))
    assert payload["rows"] == [
        ["P1", 1.2, None]
    ]  # NaN -> null in the source payload, strict JSON


@pytest.mark.parametrize(
    "rel,managed",
    [
        ("plots/plot_1_umap.html", False),
        ("plots/auto_fig.html", False),
        ("plots/f_k3m9q2xa7d/r_4hd8w1zab2/figure.json", True),
        ("plots/anything/x.json", True),
        ("plots/_code/sha256-ab.py", True),
        ("plots/_sources/sha256-ab.json", True),
        ("plots/_executions/x_1.json", True),
        ("plots/f_1/r_1/renditions/p_1.png", True),
        ("plots/.staging/tmp/a", True),
        ("plots\\f_1\\r_1\\figure.json", True),
        ("data/plots/f_1/r_1/figure.json", False),
        ("plots/../plots/f_1/r_1/config.json", True),
    ],
)
def test_managed_path_classifier(rel, managed):
    assert P.is_managed_figure_path(rel) is managed


def test_attempt_descriptor_matches_shared_cloud_vector():
    """Engine and Cloud compute the same digest for the same descriptor (shared field names)."""
    import hashlib
    from pathlib import Path

    from lobster.core.figures.records import (
        attempt_descriptor,
        canonical_descriptor_bytes,
    )

    vector = json.loads(
        (
            Path(__file__).parent / "fixtures" / "attempt_vector_cloud_draft.json"
        ).read_text()
    )
    assert (
        hashlib.sha256(canonical_descriptor_bytes(vector["descriptor"])).hexdigest()
        == vector["expected_digest"]
    )
    ours = attempt_descriptor(
        _cand(
            source=SourcePayload(b"{}", {"regeneration": ["relabel"]}), parent_ref=None
        ),
        parent_figure_sha256=None,
        parent_config_sha256=None,
        code_bytes=b"x",
        tool="execute_custom_code",
        persist=False,
        engine_version="e",
        plotly_version="p",
    )
    assert set(ours) == set(vector["descriptor"])
    assert set(ours["inputs"][0]) == set(vector["descriptor"]["inputs"][0])
    assert ours["regeneration"] == ["relabel"]


# --- pre_refused: a tool-refused candidate keeps its ordinal, nothing else -------


def _pre(reason="revision_not_found"):
    from lobster.core.figures.errors import FigureNotFoundError

    return FigureNotFoundError("parent unreadable", details={"reason": reason})


def test_pre_refused_mixed_batch_keeps_count_and_ordinal_and_persists_only_the_valid(
    store,
):
    saver = TrustedSaver(store)
    cands = [_cand(0), _cand(1, fig=_fig("blue"))]
    r0, r1 = saver.commit_attempt(
        _info(), cands, succeeded=True, pre_refused={0: _pre()}
    )
    assert (r0.status, r0.reason, r0.ref, r0.path) == (
        "refused",
        "figure_not_found:revision_not_found",
        None,
        None,
    )
    assert r1.status == "available" and r1.candidate_index == 1
    [item] = store.list().items  # only the valid candidate is stored
    assert f"{item['figure_id']}@{item['revision_id']}" == r1.ref
    execution = store.read_execution(item["execution"]["execution_id"])
    assert execution["candidate_count"] == 2
    assert [p["index"] for p in execution["planned_candidates"]] == [1]


def test_all_pre_refused_writes_nothing(store):
    out = TrustedSaver(store).commit_attempt(
        _info(),
        [_cand(0), _cand(1)],
        succeeded=True,
        pre_refused={0: _pre(), 1: _pre()},
    )
    assert [r.status for r in out] == ["refused", "refused"]
    assert store.list().items == []
    assert not (store.root / P.EXECUTIONS_DIR).exists() or not any(
        (store.root / P.EXECUTIONS_DIR).iterdir()
    )


@pytest.mark.parametrize(
    "pre",
    [
        {2: "x"},
        {5: None},
        {0: ValueError("not a FigureError")},
        {True: "x"},  # bool equals 1 in Python; never an index alias
        {1.0: "x"},
        {0.0: "x"},
        {"1": "x"},
    ],
)
def test_an_invalid_pre_refused_mapping_is_rejected_never_masked(store, pre):
    pre = {k: (_pre() if v == "x" else v) for k, v in pre.items()}
    with pytest.raises(ValueError):
        TrustedSaver(store).commit_attempt(
            _info(), [_cand(0), _cand(1)], succeeded=True, pre_refused=pre
        )
    assert store.list().items == []


def test_pre_refused_with_duplicate_candidate_indices_is_rejected(store):
    with pytest.raises(ValueError):
        TrustedSaver(store).commit_attempt(
            _info(), [_cand(0), _cand(0)], succeeded=True, pre_refused={0: _pre()}
        )


def test_the_success_fence_stays_first_with_pre_refused(store):
    out = TrustedSaver(store).commit_attempt(
        _info(), [_cand(0), _cand(1)], succeeded=False, pre_refused={0: _pre()}
    )
    assert [(r.status, r.reason) for r in out] == [
        ("not_committed", "attempt_failed"),
        ("not_committed", "attempt_failed"),
    ]


def test_a_retry_whose_eligibility_changed_is_the_existing_plan_conflict(store):
    saver = TrustedSaver(store)
    cands = [_cand(0), _cand(1, fig=_fig("blue"))]
    first = saver.commit_attempt(
        _info(), cands, succeeded=True, pre_refused={0: _pre()}
    )
    assert [r.status for r in first] == ["refused", "available"]
    retry = saver.commit_attempt(_info(), cands, succeeded=True)  # parent now readable
    assert [r.reason for r in retry] == [
        "execution_record:attempt_conflict:execution_plan_changed",
        "execution_record:attempt_conflict:execution_plan_changed",
    ]
    assert len(store.list().items) == 1  # no renumbering, no reuse loophole


# --- dependents(): ANY verified per ref; unknown IDs refused ---


def _dep_edge(name, verified, instance_id="inst-9"):
    return InputEdge(
        name, instance_id, "store_event" if verified else "declared", verified
    )


def _dep_save(store, *edges, index=0):
    [res] = TrustedSaver(store).commit_attempt(
        _info(base=f"s1:call_{index}"),
        [_cand(base=f"s1:call_{index}", inputs=tuple(edges))],
        succeeded=True,
    )
    assert res.status == "available"
    return res.ref


@pytest.mark.parametrize(
    "edges, expected",
    [
        ((("alias_a", False), ("alias_b", True)), True),  # unverified alias first
        ((("alias_b", True), ("alias_a", False)), True),  # order reversed (control)
        (
            (("alias_a", False), ("alias_b", False)),
            False,
        ),  # unverified only stays False
    ],
)
def test_a_ref_is_verified_if_any_same_id_edge_is(store, edges, expected):
    ref = _dep_save(store, *(_dep_edge(n, v) for n, v in edges))
    assert store.dependents("inst-9") == [(ref, expected)]  # one tuple per ref


def test_order_and_one_tuple_per_ref_are_preserved(store):
    r1 = _dep_save(
        store, _dep_edge("alias_a", False), _dep_edge("alias_b", True), index=1
    )
    r2 = _dep_save(store, _dep_edge("other", True, "inst-other"), index=2)
    r3 = _dep_save(store, _dep_edge("alias_c", False), index=3)
    assert store.dependents("inst-9") == [(r1, True), (r3, False)]
    assert store.dependents("inst-other") == [(r2, True)]


@pytest.mark.parametrize("bad", [None, "", 123, b"inst-9"])
def test_unknown_or_non_string_ids_are_refused_before_enumeration(
    store, monkeypatch, bad
):
    _dep_save(store, _dep_edge("declared_only", False, None))  # a null-ID edge exists

    def enumerated(*a, **k):
        raise AssertionError("catalog enumerated")

    monkeypatch.setattr(LocalFigureStore, "_iter_records", enumerated)
    with pytest.raises((TypeError, ValueError)):
        store.dependents(bad)
