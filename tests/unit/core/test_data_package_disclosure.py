"""The export package says exactly what it is and what it leaves out.

A view package copies each saved revision's exact bytes and views. It does NOT copy the
lifecycle artifacts a revision references (source data, execution record, code, activity),
so it must say so per reference instead of looking like a complete, reloadable store.

Real store on tmp_path, parent-side only (no child process).
"""

from __future__ import annotations

import json
import zipfile

import plotly.graph_objects as go
import pytest

from lobster.core.figures.payload import (
    config_to_bytes,
    figure_to_bytes,
    source_to_bytes,
)
from lobster.core.figures.records import (
    Candidate,
    ExecutionInfo,
    InputEdge,
    SourcePayload,
)
from lobster.core.figures.saver import TrustedSaver
from lobster.core.runtime.data_manager import DataManagerV2

CODE = b"fig = make_volcano(df)\nsubmit_figure(fig)\n"


@pytest.fixture()
def dm(tmp_path):
    return DataManagerV2(workspace_path=tmp_path)


def _custom_code_revision(dm):
    """A source-bearing custom-code revision, saved through the real trusted saver."""
    base = f"{dm.figure_store.session_id}:call_1"
    info = ExecutionInfo(
        attempt_base=base,
        tool="execute_custom_code",
        agent="proteomics_expert",
        parameters={"modality_name": "prot"},
        persist=True,
        code_bytes=CODE,
        activity={"id": "act-77", "tool_name": "execute_custom_code"},
        engine_version="test",
        plotly_version="test",
    )
    cand = Candidate(
        attempt_key=f"{base}:0",
        candidate_index=0,
        figure_bytes=figure_to_bytes(go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))),
        config_bytes=config_to_bytes({}),
        actor="expert:proteomics_expert",
        label="Volcano",
        inputs=(InputEdge("prot", "inst-1", "store_event", True),),
        source=SourcePayload(
            source_to_bytes([{"protein": "P1", "log2fc": 1.2}]), {"comparison": "t_c"}
        ),
    )
    [res] = TrustedSaver(dm.figure_store).commit_attempt(info, [cand], succeeded=True)
    assert res.status == "available"
    record = dm.figure_store.read_record(res.ref)
    execution = dm.figure_store.read_execution(record["execution"]["execution_id"])
    return res.ref, record, execution


def _read_package(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        return (
            set(zf.namelist()),
            zf.read("README.md").decode("utf-8"),
            zf.read("plots/README.md").decode("utf-8"),
            json.loads(zf.read("plots/index.json")),
            {n: zf.read(n) for n in zf.namelist()},
        )


# --- A7: referenced lifecycle artifacts ------------------------------------------


def test_source_bearing_custom_code_export_names_every_referenced_omission(
    dm, tmp_path
):
    ref, record, execution = _custom_code_revision(dm)
    source_id = record["source"]["source_id"]
    execution_id = record["execution"]["execution_id"]
    code_id = execution["code_id"]
    assert source_id and code_id and execution["activity_record"]  # fixture sanity
    exact = dm.figure_store.read(ref)

    zip_path = dm.create_data_package(output_dir=str(tmp_path / "out"))
    names, readme, plots_readme, index, blobs = _read_package(zip_path)

    # nothing referenced is silently implied: the blobs are NOT in the package ...
    assert not any("_sources" in n or "_code" in n or "_executions" in n for n in names)
    # ... and every reference is listed, per revision, in the index ...
    [entry] = index
    assert entry["not_included"] == [
        {"kind": "source", "id": source_id},
        {"kind": "execution", "id": execution_id},
        {"kind": "code", "id": code_id},
        {"kind": "activity", "id": "act-77"},
    ]
    # ... in both READMEs ...
    for kind, rid in (
        ("source", source_id),
        ("execution", execution_id),
        ("code", code_id),
        ("activity", "act-77"),
    ):
        assert f"{kind} {rid}" in readme
        assert f"{kind} {rid}" in plots_readme
    assert "## Not included: referenced lifecycle artifacts" in readme
    # ... with explicit view-only / not-reloadable wording ...
    assert "view package" in readme
    assert "not a complete, reloadable figure store" in readme
    # ... and in the package report.
    report = dm.last_package_report
    assert report["view_only"] is True
    assert report["lifecycle_refs_not_included"] == 4
    # the copied bytes are unchanged (disclosure only)
    fid, rid = ref.split("@")
    for name, data in zip(("revision.json", "figure.json", "config.json"), exact):
        assert blobs[f"plots/{fid}/{rid}/{name}"] == data
    # by-design omissions are not failures
    assert "## Not copied" not in readme


def test_unreadable_execution_record_is_disclosed_not_guessed(
    dm, tmp_path, monkeypatch
):
    from lobster.core.figures.errors import IntegrityError

    ref, record, _ = _custom_code_revision(dm)
    execution_id = record["execution"]["execution_id"]

    def boom(_eid):
        raise IntegrityError("changed")

    monkeypatch.setattr(dm.figure_store, "read_execution", boom)
    zip_path = dm.create_data_package(output_dir=str(tmp_path / "out"))
    _, readme, _, index, _ = _read_package(zip_path)

    [entry] = index
    assert {"kind": "source", "id": record["source"]["source_id"]} in entry[
        "not_included"
    ]
    assert {
        "kind": "execution",
        "id": execution_id,
        "unread": "fingerprint_mismatch",
    } in entry["not_included"]
    assert not any(i["kind"] in ("code", "activity") for i in entry["not_included"])
    assert "code/activity references unknown (fingerprint_mismatch)" in readme
    assert entry.get("error") is None  # the revision itself was still exported


def test_a_figure_only_package_still_states_its_scope(dm, tmp_path):
    fig = go.Figure(go.Bar(x=[1], y=[2]))
    plot_id, _, _ = dm.plot_manager.add_plot(plot=fig, title="Bars", source="t")
    [res] = dm.plot_manager.commit_staged(
        {"id": "a1"}, tool_call_id="c1", staged_ids=[plot_id]
    )
    zip_path = dm.create_data_package(output_dir=str(tmp_path / "out"))
    _, readme, _, index, _ = _read_package(zip_path)
    kinds = [i["kind"] for i in index[0]["not_included"]]
    assert kinds == ["execution", "activity"]  # no source, no code: none invented
    assert "not a complete, reloadable figure store" in readme


def test_the_plots_readme_states_the_view_only_scope_too(dm, tmp_path):
    _custom_code_revision(dm)
    zip_path = dm.create_data_package(output_dir=str(tmp_path / "out"))
    _, readme, plots_readme, _, _ = _read_package(zip_path)
    for text in (readme, plots_readme):
        assert "view package" in text
        assert "not a complete, reloadable figure store" in text
