"""scVI never saves a model into the managed figure folder (P1-2 lane B, sink F3).

The real tool and the real service run with ``scvi``/``torch`` replaced at the import
boundary (modules with real ``ModuleSpec`` objects, so availability is detected as it
is in production) and a fake ``SCVI`` class whose ``save`` writes real files. A refused
model folder is caught before any data, device or model work; a folder that becomes a
link into a saved revision during training is refused again before the model or the
embeddings are kept, and before the save. Ordinary saves and the existing save-error
outcome are unchanged.
"""

from __future__ import annotations

import importlib.machinery
import os
import sys
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.figures.errors import ManagedPathError
from lobster.core.figures.export_guard import MANAGED_EXPORT_MESSAGE
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools
from tests.unit.export_guard_fixtures import alias, saved_revision, workspace_tree

ML_MODULE = "lobster.agents.machine_learning.machine_learning_expert"
MODALITY = "cells"


class FakeSCVI:
    """Stands in for ``scvi.model.SCVI``; ``save`` writes a real folder."""

    setup_calls: list = []
    instances: list = []
    on_train = None  # called as on_train(model, kwargs); may raise
    save_error = None

    @classmethod
    def setup_anndata(cls, adata, **kwargs):
        cls.setup_calls.append(kwargs)

    def __init__(self, adata, **kwargs):
        self.adata = adata
        self.train_calls = []
        FakeSCVI.instances.append(self)

    def train(self, **kwargs):
        self.train_calls.append(dict(kwargs))
        if FakeSCVI.on_train is not None:
            FakeSCVI.on_train(self, kwargs)

    def get_latent_representation(self):
        return np.zeros((self.adata.n_obs, 2))

    def save(self, path, overwrite=False):
        if FakeSCVI.save_error is not None:
            raise FakeSCVI.save_error
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=overwrite)
        (folder / "model.pt").write_bytes(b"model")


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return module


@pytest.fixture
def fake_scvi(monkeypatch):
    """``scvi`` and ``torch`` at the import boundary; the hardware says cuda."""
    from lobster.tools.gpu_detector import GPUDetector

    FakeSCVI.setup_calls = []
    FakeSCVI.instances = []
    FakeSCVI.on_train = None
    FakeSCVI.save_error = None
    scvi = _module("scvi")
    scvi.model = types.SimpleNamespace(SCVI=FakeSCVI)
    torch = _module("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "scvi", scvi)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        GPUDetector,
        "get_hardware_recommendation",
        staticmethod(
            lambda: {"profile": "ml-gpu", "device": "cuda", "info": "", "command": ""}
        ),
    )
    return FakeSCVI


def _adata():
    return ad.AnnData(
        X=np.ones((4, 3), dtype=np.float32),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )


@pytest.fixture
def session(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    rev = saved_revision(dm.workspace_path)
    logged = []
    monkeypatch.setattr(dm, "log_tool_usage", lambda *a, **k: logged.append(k))
    return dm, rev, logged


def _retarget_on_train(save_path: Path, target: Path, *, on_call: int = 1):
    """During the n-th ``train`` call, make the model folder a link into ``target``."""

    def hook(model, kwargs):
        if len(model.train_calls) == on_call:
            alias(save_path, target)

    return hook


def _cpu_fallback_then(hook):
    """The first ``train`` fails with a device error; the second runs ``hook``."""

    def first_fails(model, kwargs):
        if len(model.train_calls) == 1:
            raise RuntimeError("CUDA device error")
        hook(model, kwargs)

    return first_fails


# ------------------------------------------------------------------- the real tool


def _tool(dm):
    from lobster.agents.machine_learning.machine_learning_expert import (
        machine_learning_expert,
    )

    return factory_tools(machine_learning_expert, ML_MODULE, dm)["train_scvi_embedding"]


@pytest.mark.parametrize(
    "name",
    ["../plots/{fig}/{rev}/m", "../plots/{fig}/new_rev/m"],
    ids=["into-revision", "into-new-revision"],
)
def test_the_tool_refuses_a_model_folder_named_into_the_managed_folder(
    session, fake_scvi, name
):
    dm, rev, logged = session
    name = name.format(fig=rev.directory.parent.name, rev=rev.directory.name)
    adata = _adata()
    dm.modalities[name] = adata
    assert not (dm.workspace_path / "models").exists()
    before = workspace_tree(dm.workspace_path)
    response = _tool(dm).invoke({"modality_name": name, "max_epochs": 1})
    assert response == MANAGED_EXPORT_MESSAGE
    assert fake_scvi.setup_calls == [] and fake_scvi.instances == []
    assert not (dm.workspace_path / "models").exists()
    assert workspace_tree(dm.workspace_path) == before
    assert rev.unchanged()
    assert dm.modalities[name] is adata and "X_scvi" not in adata.obsm
    assert logged == []


def test_the_tool_refuses_a_models_folder_that_is_a_link_into_a_revision(
    session, fake_scvi
):
    dm, rev, logged = session
    dm.modalities[MODALITY] = _adata()
    alias(dm.workspace_path / "models", rev.directory)
    before = workspace_tree(dm.workspace_path)
    response = _tool(dm).invoke({"modality_name": MODALITY, "max_epochs": 1})
    assert response == MANAGED_EXPORT_MESSAGE
    assert fake_scvi.setup_calls == [] and fake_scvi.instances == []
    assert workspace_tree(dm.workspace_path) == before
    assert rev.unchanged()
    assert "X_scvi" not in dm.modalities[MODALITY].obsm
    assert logged == []


def test_the_tool_still_saves_an_ordinary_model(session, fake_scvi):
    dm, rev, logged = session
    dm.modalities[MODALITY] = _adata()
    response = _tool(dm).invoke({"modality_name": MODALITY, "max_epochs": 1})
    assert "Successfully trained scVI embedding" in response
    assert "Model saved: Yes" in response
    folder = dm.workspace_path / "models" / f"{MODALITY}_scvi_model"
    assert (folder / "model.pt").read_bytes() == b"model"
    assert "X_scvi" in dm.modalities[MODALITY].obsm
    assert [k["tool_name"] for k in logged] == ["train_scvi_embedding"]
    assert rev.unchanged()


# ---------------------------------------------------------------- the real service


def _service():
    from lobster.services.analysis.scvi_embedding_service import ScviEmbeddingService

    return ScviEmbeddingService()


def test_the_service_refuses_a_managed_save_path_before_any_work(session, fake_scvi):
    dm, rev, _ = session
    service = _service()
    adata = _adata()
    before = workspace_tree(dm.workspace_path)
    with pytest.raises(ManagedPathError):
        service.train_scvi_embedding(
            adata,
            max_epochs=1,
            save_path=str(rev.directory / "model"),
            workspace_path=dm.workspace_path,
        )
    assert service.device is None  # no device set-up
    assert fake_scvi.setup_calls == [] and fake_scvi.instances == []
    assert service.model is None and "X_scvi" not in adata.obsm
    assert workspace_tree(dm.workspace_path) == before
    assert rev.unchanged()


@pytest.mark.parametrize("branch", ["primary", "cpu-fallback"])
def test_a_save_path_retargeted_during_training_is_refused(session, fake_scvi, branch):
    dm, rev, _ = session
    service = _service()
    adata = _adata()
    save_path = dm.workspace_path / "models" / "m_scvi_model"
    save_path.parent.mkdir()
    hook = _retarget_on_train(
        save_path, rev.directory, on_call=1 if branch == "primary" else 2
    )
    fake_scvi.on_train = hook if branch == "primary" else _cpu_fallback_then(hook)
    with pytest.raises(ManagedPathError):
        service.train_scvi_embedding(
            adata,
            max_epochs=1,
            save_path=str(save_path),
            workspace_path=dm.workspace_path,
        )
    [model] = fake_scvi.instances
    if branch == "primary":
        assert [c["accelerator"] for c in model.train_calls] == ["cuda"]
    else:
        assert [c["accelerator"] for c in model.train_calls] == ["cuda", "cpu"]
    assert service.model is None and "X_scvi" not in adata.obsm
    assert not (rev.directory / "model.pt").exists()
    assert rev.unchanged()


@pytest.mark.parametrize("branch", ["primary", "cpu-fallback"])
def test_an_ordinary_save_is_unchanged(session, fake_scvi, branch):
    dm, rev, _ = session
    service = _service()
    adata = _adata()
    save_path = dm.workspace_path / "models" / "m_scvi_model"
    if branch == "cpu-fallback":
        fake_scvi.on_train = _cpu_fallback_then(lambda model, kwargs: None)
    out_adata, info, ir = service.train_scvi_embedding(
        adata, max_epochs=1, save_path=str(save_path), workspace_path=dm.workspace_path
    )
    assert info["model_saved"] is True and info["model_path"] == str(save_path)
    assert (save_path / "model.pt").read_bytes() == b"model"
    assert "X_scvi" in out_adata.obsm and service.model is not None
    assert info.get("fallback_to_cpu", False) is (branch == "cpu-fallback")
    assert ir is not None
    assert rev.unchanged()


@pytest.mark.parametrize("branch", ["primary", "cpu-fallback"])
def test_regression_a_save_error_still_reports_model_not_saved(
    session, fake_scvi, branch
):
    dm, rev, _ = session
    service = _service()
    save_path = dm.workspace_path / "models" / "m_scvi_model"
    fake_scvi.save_error = OSError("disk full")
    if branch == "cpu-fallback":
        fake_scvi.on_train = _cpu_fallback_then(lambda model, kwargs: None)
    out_adata, info, ir = service.train_scvi_embedding(
        _adata(),
        max_epochs=1,
        save_path=str(save_path),
        workspace_path=dm.workspace_path,
    )
    assert info["model_saved"] is False
    assert "model_path" not in info
    assert ir is not None


def test_regression_without_a_workspace_the_service_does_not_check(session, fake_scvi):
    dm, rev, _ = session
    service = _service()
    save_path = dm.workspace_path / "elsewhere" / "m_scvi_model"
    out_adata, info, ir = service.train_scvi_embedding(
        _adata(), max_epochs=1, save_path=str(save_path)
    )
    assert info["model_saved"] is True
    assert os.path.isdir(save_path)


# --- B2: every file the native save writes, and its folder creation (F1, SEC-03/06)


def test_b2_the_tool_refuses_a_model_file_alias_in_an_ordinary_folder(
    session, fake_scvi
):
    dm, rev, logged = session
    dm.modalities[MODALITY] = _adata()
    folder = dm.workspace_path / "models" / f"{MODALITY}_scvi_model"
    alias(folder / "model.pt", rev.directory / sorted(rev.files)[0])
    before = workspace_tree(dm.workspace_path)
    response = _tool(dm).invoke({"modality_name": MODALITY, "max_epochs": 1})
    assert response == MANAGED_EXPORT_MESSAGE
    assert fake_scvi.setup_calls == [] and fake_scvi.instances == []
    assert workspace_tree(dm.workspace_path) == before and rev.unchanged()
    assert "X_scvi" not in dm.modalities[MODALITY].obsm and logged == []


def test_b2_the_tool_refuses_a_model_folder_named_directly_under_plots(
    session, fake_scvi
):
    dm, rev, logged = session
    name = "../plots/new"  # folder models/../plots/new_scvi_model
    dm.modalities[name] = _adata()
    before = workspace_tree(dm.workspace_path)
    response = _tool(dm).invoke({"modality_name": name, "max_epochs": 1})
    assert response == MANAGED_EXPORT_MESSAGE
    assert fake_scvi.instances == []
    assert not (dm.workspace_path / "plots" / "new_scvi_model").exists()
    assert workspace_tree(dm.workspace_path) == before and rev.unchanged()


@pytest.mark.parametrize("branch", ["primary", "cpu-fallback"])
def test_b2_a_model_file_alias_before_training_is_refused_before_any_work(
    session, fake_scvi, branch
):
    dm, rev, _ = session
    service = _service()
    adata = _adata()
    save_path = dm.workspace_path / "models" / "m_scvi_model"
    alias(save_path / "model.pt", rev.directory / sorted(rev.files)[0])
    if branch == "cpu-fallback":
        fake_scvi.on_train = _cpu_fallback_then(lambda model, kwargs: None)
    before = workspace_tree(dm.workspace_path)
    with pytest.raises(ManagedPathError):
        service.train_scvi_embedding(
            adata,
            max_epochs=1,
            save_path=str(save_path),
            workspace_path=dm.workspace_path,
        )
    assert service.device is None and fake_scvi.instances == []
    assert workspace_tree(dm.workspace_path) == before and rev.unchanged()


@pytest.mark.parametrize("branch", ["primary", "cpu-fallback"])
def test_b2_a_model_file_retargeted_during_training_is_refused(
    session, fake_scvi, branch
):
    dm, rev, _ = session
    service = _service()
    adata = _adata()
    save_path = dm.workspace_path / "models" / "m_scvi_model"
    save_path.mkdir(parents=True)
    hook = _retarget_on_train(
        save_path / "model.pt",
        rev.directory / sorted(rev.files)[0],
        on_call=1 if branch == "primary" else 2,
    )
    fake_scvi.on_train = hook if branch == "primary" else _cpu_fallback_then(hook)
    with pytest.raises(ManagedPathError):
        service.train_scvi_embedding(
            adata,
            max_epochs=1,
            save_path=str(save_path),
            workspace_path=dm.workspace_path,
        )
    assert os.path.islink(save_path / "model.pt")  # the retarget happened
    assert service.model is None and "X_scvi" not in adata.obsm
    assert rev.unchanged()


def test_b2_a_model_folder_passing_through_a_revision_is_refused(session, fake_scvi):
    dm, rev, _ = session
    service = _service()
    save_path = rev.directory.joinpath("NEW", "..", "..", "..", "..", "models", "m")
    before = workspace_tree(dm.workspace_path)
    with pytest.raises(ManagedPathError):
        service.train_scvi_embedding(
            _adata(),
            max_epochs=1,
            save_path=str(save_path),
            workspace_path=dm.workspace_path,
        )
    assert fake_scvi.instances == []
    assert not (rev.directory / "NEW").exists()
    assert workspace_tree(dm.workspace_path) == before and rev.unchanged()


def test_b2_the_tool_refusal_logs_no_modality_name(session, fake_scvi, caplog):
    import logging

    dm, rev, _ = session
    name = f"../plots/{rev.directory.parent.name}/{rev.directory.name}/m"
    dm.modalities[name] = _adata()
    with caplog.at_level(logging.DEBUG, logger=ML_MODULE):
        response = _tool(dm).invoke({"modality_name": name, "max_epochs": 1})
    assert response == MANAGED_EXPORT_MESSAGE
    assert not [r for r in caplog.records if name in r.getMessage()]
