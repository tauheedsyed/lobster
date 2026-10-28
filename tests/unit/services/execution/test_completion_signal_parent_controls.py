"""Parent controls remain authoritative for a forged cooperative completion.

No code child is launched. The transport double fabricates the files that
same-user code could write; the real parent reader, saver and store still run.
This is not evidence of authenticated completion or execution isolation.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lobster.core.figures import paths as P
from lobster.core.figures.errors import (
    FigureInvalidError,
    FigureNotFoundError,
    IntegrityError,
    StorageError,
)
from lobster.core.figures.payload import figure_to_bytes
from lobster.core.figures.records import (
    activity_blob_bytes,
    attempt_digest,
    execution_request,
    json_blob_bytes,
    read_submitted_figures,
)
from lobster.core.figures.saver import TrustedSaver
from lobster.core.figures.store import LocalFigureStore
from lobster.services.execution import custom_code_execution_service as execution
from tests.unit.core.figures.test_store_and_saver import _cand, _fig, _info

CODE = b"raise RuntimeError('the normal epilogue was never executed')\n"


def execution_info(*, base="s1:call_1"):
    return _info(code=CODE, base=base)


def execution_blobs(candidate):
    info = execution_info()
    activity = activity_blob_bytes(info.activity)
    digest = attempt_digest(
        candidate,
        parent_figure_sha256=None,
        parent_config_sha256=None,
        code_bytes=info.code_bytes,
        tool=info.tool,
        persist=info.persist,
        engine_version=info.engine_version,
        plotly_version=info.plotly_version,
    )
    request = execution_request(
        info,
        [(candidate.candidate_index, digest)],
        activity_bytes=activity,
        candidate_count=1,
    )
    return json_blob_bytes(request), info.code_bytes, activity, digest


@pytest.fixture
def forged_completion(tmp_path, monkeypatch):
    store = LocalFigureStore(tmp_path, session_id="s1")
    cache = tmp_path / "cache" / "forged"
    service = execution.CustomCodeExecutionService(
        SimpleNamespace(workspace_path=tmp_path)
    )
    calls = []

    def submit(candidate=None, *, structural_damage=False):
        candidate = candidate or _cand()

        def fabricate(argv, **kwargs):
            # Transport only: never execute the generated script/user code.
            script = ast.parse(Path(argv[1]).read_text())
            nonce = script.body[-1].value.args[0].value
            assert isinstance(nonce, str) and len(nonce) == 32
            calls.append(argv)
            cache.mkdir(parents=True, exist_ok=True)
            (cache / ".execution_result.json").write_text(
                json.dumps(
                    {
                        "schema": execution.RESULT_SCHEMA,
                        "has_result": True,
                        "result": "fabricated, not executed",
                        "type": "str",
                    }
                )
            )
            (cache / ".execution_complete").write_text(nonce)
            directory = cache / "candidates" / "0"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "figure.json").write_bytes(candidate.figure_bytes)
            (directory / "config.json").write_bytes(candidate.config_bytes)
            (directory / "candidate.json").write_text(
                json.dumps(
                    {
                        "index": 7 if structural_damage else 0,
                        "has_config": True,
                        "parent": candidate.parent_ref,
                        "label": candidate.label,
                    }
                )
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(execution.subprocess, "run", fabricate)
        context = {
            "workspace_path": tmp_path,
            "exec_cache_path": cache,
            "load_workspace_files": False,
        }
        # The fake returns COMPLETE even though this code never runs successfully.
        result, _, _, error = service._execute_in_namespace(
            CODE.decode("utf-8"), context
        )
        assert error is None and result == "fabricated, not executed"
        assert context["completion"] == execution.COMPLETION_COMPLETE
        assert calls
        submitted, _, invalid = read_submitted_figures(cache)
        if invalid:
            return store, None, invalid
        assert len(submitted) == 1
        wire = submitted[0]
        candidate = replace(
            candidate,
            figure_bytes=wire.figure_bytes,
            config_bytes=wire.config_bytes,
            parent_ref=wire.parent,
        )
        return store, candidate, None

    return submit


def test_completion_wording_describes_cooperative_not_authenticated_signal():
    source = Path(execution.__file__).read_text()
    comment = source.split("complete_path = ipc_dir", 1)[0].rsplit(
        "result_path = ipc_dir", 1
    )[1]
    assert "cooperative early-exit detection" in comment
    assert "not authentication" in comment
    assert "same OS user" in comment
    assert "parent" in comment and "fences" in comment


def test_service_header_does_not_claim_security_or_compliance_boundary():
    header = ast.get_docstring(ast.parse(Path(execution.__file__).read_text()))
    assert "same OS user" in header and "not a sandbox" in header
    assert "W3C-PROV compliance" not in header
    assert "Workspace-only file access" not in header
    assert "No network access" not in header


def test_execution_method_does_not_claim_host_isolation():
    doc = execution.CustomCodeExecutionService._execute_in_namespace.__doc__
    assert "same OS user" in doc and "not a sandbox" in doc
    assert "cannot affect Lobster's main process" not in doc


def test_forged_valid_completion_can_save_but_records_execution_first(
    forged_completion, monkeypatch
):
    store, candidate, invalid = forged_completion()
    assert invalid is None
    real_commit = store.commit
    seen = []

    def observe(*args, **kwargs):
        record = store.read_execution(kwargs["execution_id"])
        assert record["code"]["sha256"] == hashlib.sha256(CODE).hexdigest()
        assert (
            record["activity_record"]["sha256"]
            == hashlib.sha256(
                activity_blob_bytes(execution_info().activity)
            ).hexdigest()
        )
        seen.append(record["execution_id"])
        return real_commit(*args, **kwargs)

    monkeypatch.setattr(store, "commit", observe)
    [saved] = TrustedSaver(store).commit_attempt(
        execution_info(), [candidate], succeeded=True, fence=store.attempt_fence()
    )
    assert saved.status == "available" and saved.ref
    assert len(seen) == len(store.list().items) == 1
    assert store.read_record(saved.ref)["execution"]["execution_id"] == seen[0]
    assert store.read_record(saved.ref)["change"]["actor"] == candidate.actor
    assert store.read_record(saved.ref)["inputs"][0]["instance_id"] == "inst-1"


def test_forged_completion_cannot_bypass_submission_structure(forged_completion):
    store, candidate, invalid = forged_completion(structural_damage=True)
    assert candidate is None and invalid == "figure_submission_invalid"
    assert store.list().items == []


@pytest.mark.parametrize("kind", ["figure", "config"])
def test_forged_completion_cannot_bypass_candidate_validation(forged_completion, kind):
    candidate = _cand()
    if kind == "figure":
        candidate = replace(candidate, figure_bytes=b'{"data":[{"type":"bogus"}]}')
    else:
        # The parent reader requires a strict JSON object, not an arbitrary value.
        candidate = replace(candidate, config_bytes=b"[]")
    store, candidate, invalid = forged_completion(candidate)
    if invalid is not None:
        assert kind == "config" and invalid == "figure_submission_invalid"
    else:
        [refused] = TrustedSaver(store).commit_attempt(
            execution_info(), [candidate], succeeded=True, fence=store.attempt_fence()
        )
        assert refused.ref is None
        assert refused.status == "refused" and refused.reason.startswith(
            "figure_invalid"
        )
    assert store.list().items == []


def test_forged_completion_cannot_save_without_execution_record(
    forged_completion, monkeypatch
):
    store, candidate, invalid = forged_completion()
    assert invalid is None

    def failed(*args, **kwargs):
        raise StorageError("offline record failure")

    monkeypatch.setattr(store, "commit_execution", failed)
    [refused] = TrustedSaver(store).commit_attempt(
        execution_info(), [candidate], succeeded=True
    )
    assert refused.status == "failed"
    assert refused.ref is None
    assert refused.reason.startswith("execution_record:")
    assert store.list().items == []


def test_forged_completion_cannot_bypass_changed_generation(forged_completion):
    store, candidate, invalid = forged_completion()
    assert invalid is None
    old_fence = store.attempt_fence()
    assert store.fence_purge()[0].complete
    [refused] = TrustedSaver(store).commit_attempt(
        execution_info(), [candidate], succeeded=True, fence=old_fence
    )
    assert (refused.status, refused.reason) == (
        "refused",
        "execution_record:generation_fenced",
    )
    assert refused.ref is None
    assert store.list().items == []


def test_forged_completion_cannot_bypass_deleted_parent_at_commit(
    forged_completion, monkeypatch
):
    store, _, _ = forged_completion()
    base = "s1:parent"
    [parent] = TrustedSaver(store).commit_attempt(
        _info(base=base), [_cand(base=base)], succeeded=True
    )
    assert parent.status == "available"
    store, candidate, invalid = forged_completion(_cand(parent_ref=parent.ref))
    assert invalid is None
    real_commit = store.commit

    def delete_then_commit(*args, **kwargs):
        assert store.read_execution(kwargs["execution_id"])
        assert store.fence_deletion([parent.ref]).complete
        return real_commit(*args, **kwargs)

    monkeypatch.setattr(store, "commit", delete_then_commit)
    [refused] = TrustedSaver(store).commit_attempt(
        execution_info(), [candidate], succeeded=True, fence=store.attempt_fence()
    )
    assert refused.status == "refused"
    assert refused.reason == "parent_deleted:parent_unavailable"
    assert refused.ref is None
    assert store.list().items == []


def test_forged_completion_cannot_bypass_deleted_input(forged_completion):
    store, candidate, invalid = forged_completion()
    assert invalid is None
    fence = store.attempt_fence()
    assert store.fence_deletion([], instance_ids=["inst-1"]).complete
    [refused] = TrustedSaver(store).commit_attempt(
        execution_info(), [candidate], succeeded=True, fence=fence
    )
    assert (refused.status, refused.reason) == ("refused", "generation_fenced")
    assert refused.ref is None
    assert store.list().items == []


def test_forged_completion_cannot_reuse_a_digest_for_changed_figure(
    forged_completion, monkeypatch
):
    store, candidate, invalid = forged_completion()
    assert invalid is None
    captured = []
    real_commit = store.commit

    def observe(cand, **kwargs):
        captured.append((cand, kwargs))
        return real_commit(cand, **kwargs)

    monkeypatch.setattr(store, "commit", observe)
    [saved] = TrustedSaver(store).commit_attempt(
        execution_info(), [candidate], succeeded=True, fence=store.attempt_fence()
    )
    assert saved.status == "available"
    [(prepared, kwargs)] = captured
    changed = replace(prepared, figure_bytes=figure_to_bytes(_fig("green")))
    with pytest.raises(FigureInvalidError) as refused:
        real_commit(changed, **kwargs)
    assert refused.value.details == {"reason": "digest_mismatch"}
    [item] = store.list().items
    assert f"{item['figure_id']}@{item['revision_id']}" == saved.ref


def test_forged_completion_cannot_rebind_an_attempt_to_changed_code(forged_completion):
    store, candidate, invalid = forged_completion()
    assert invalid is None
    [saved] = TrustedSaver(store).commit_attempt(
        execution_info(), [candidate], succeeded=True, fence=store.attempt_fence()
    )
    assert saved.status == "available"
    [refused] = TrustedSaver(store).commit_attempt(
        replace(execution_info(), code_bytes=b"different parent-supplied code\n"),
        [candidate],
        succeeded=True,
        fence=store.attempt_fence(),
    )
    assert (refused.status, refused.reason) == (
        "refused",
        "execution_record:attempt_conflict:execution_plan_changed",
    )
    assert refused.ref is None
    [item] = store.list().items
    assert f"{item['figure_id']}@{item['revision_id']}" == saved.ref


@pytest.mark.parametrize("blob", ["code", "activity"])
def test_forged_completion_cannot_record_mismatched_execution_blob(
    forged_completion, blob
):
    store, candidate, invalid = forged_completion()
    assert invalid is None
    request, code, activity, _ = execution_blobs(candidate)
    if blob == "code":
        code += b"# not the recorded code\n"
    else:
        activity = b'{"id":"not-the-recorded-activity"}'
    with pytest.raises(IntegrityError) as refused:
        store.commit_execution(request, code, activity, fence=store.attempt_fence())
    assert refused.value.code == "fingerprint_mismatch"
    assert refused.value.details == {"reason": f"{blob}_blob_mismatch"}
    assert store.list().items == []


@pytest.mark.parametrize("blob", ["code", "activity"])
def test_forged_completion_cannot_retry_after_stored_execution_blob_changes(
    forged_completion, blob
):
    store, candidate, invalid = forged_completion()
    assert invalid is None
    request, code, activity, _ = execution_blobs(candidate)
    fence = store.attempt_fence()
    execution_id = store.commit_execution(request, code, activity, fence=fence)
    record = store.read_execution(execution_id)
    assert store.list().items == []  # execution exists, but no revision has committed
    if blob == "code":
        path = store.root / P.CODE_DIR / f"{record['code_id']}.py"
    else:
        path = store.workspace / record["activity_record"]["path"]
    path.write_bytes(b"changed after the execution was recorded\n")
    with pytest.raises(IntegrityError) as refused:
        store.commit_execution(request, code, activity, fence=fence)
    assert refused.value.code == "fingerprint_mismatch"
    assert refused.value.details == {"reason": f"{blob}_blob_changed"}
    assert store.list().items == []


def test_forged_completion_cannot_commit_without_a_real_execution_record(
    forged_completion,
):
    store, candidate, invalid = forged_completion()
    assert invalid is None
    _, _, _, digest = execution_blobs(candidate)
    # Exercise the actual store operation: no method replacement/fault injection.
    with pytest.raises(FigureNotFoundError) as refused:
        store.commit(
            candidate,
            execution_id="x_nonexistent",
            digest=digest,
            fence=store.attempt_fence(),
        )
    assert refused.value.code == "figure_not_found"
    assert refused.value.details == {"reason": "execution_not_found"}
    assert store.list().items == []
