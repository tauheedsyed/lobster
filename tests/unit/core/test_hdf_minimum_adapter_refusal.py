"""Bounded no-pandas policy; NOT HDF secondary-storage admission tests."""

import hashlib
import importlib
import pickle
from pathlib import Path
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import lobster.core.exceptions as exceptions
from lobster.core.adapters.base import BaseAdapter
from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter
from lobster.core.runtime.data_manager import DataManagerV2


def refusal_type():
    """Delayed lookup permits behavioral RED without an import collection error."""
    result = getattr(exceptions, "HdfReadRefusal", None)
    assert result is not None, "closed_type_missing"
    return result


def pandas_fixture(path, *, objects=False):
    frame = pd.DataFrame(
        {"expression": [{"benign": 1}, {"benign": 2}] if objects else [1, 2]},
        index=["gene_a", "gene_b"],
    )
    if objects:
        # Benign builtin serialization only. Never read/deserialize the fixture.
        with pytest.warns(pd.errors.PerformanceWarning):
            frame.to_hdf(path, key="expression_data", mode="w", format="fixed")
    else:
        frame.to_hdf(path, key="expression_data", mode="w", format="fixed")
    assert path.stat().st_size > 0


def forbid_pandas_read(monkeypatch):
    import tables.atom
    import tables.attributeset

    calls = []

    def forbidden(*args, **kwargs):
        calls.append("prohibited_reader")
        raise AssertionError("prohibited_reader")

    monkeypatch.setattr(pd, "read_hdf", forbidden)
    monkeypatch.setattr(pd, "HDFStore", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    # Explicit installed callsites, not a global assertion covering imports.
    assert tables.atom.pickle.loads is forbidden
    assert tables.attributeset.pickle.loads is forbidden
    return calls


@pytest.mark.parametrize("objects", [False, True])
@pytest.mark.parametrize("extension", ["h5", "H5"])
def test_real_pandas_file_refused_without_reader(
    monkeypatch, tmp_path, record_property, objects, extension
):
    path = tmp_path / f"fixture.{extension}"
    pandas_fixture(path, objects=objects)
    record_property("fixture_sha256", hashlib.sha256(path.read_bytes()).hexdigest())
    record_property("fixture_bytes", path.stat().st_size)
    calls = forbid_pandas_read(monkeypatch)
    # Catch any exception first: on B0 the behavioral failure is the actual
    # prohibited pandas entry, NOT the absent refusal class.
    with pytest.raises(Exception) as caught:
        TranscriptomicsAdapter("bulk").from_source(
            path, key="different_key", transpose=False
        )
    assert calls == [], "pandas_fallback_reached"
    assert isinstance(caught.value, refusal_type())
    assert caught.value.reason == "pandas_hdf_not_admitted"
    assert caught.value.retryable is False
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


def test_closed_no_argument_type():
    cls = refusal_type()
    error = cls()
    assert type(error) is cls
    assert not isinstance(error, (ValueError, OSError, exceptions.InferenceRefused))
    assert error.args == ("Pandas HDF loading is not supported.",)
    assert str(error) == "Pandas HDF loading is not supported."
    assert repr(error) == "HdfReadRefusal('Pandas HDF loading is not supported.')"
    assert error.__dict__ == {"reason": "pandas_hdf_not_admitted", "retryable": False}
    for args, kwargs in [
        (("marker",), {}),
        ((), {"details": "marker"}),
        ((), {"reason": "marker"}),
    ]:
        with pytest.raises(TypeError):
            cls(*args, **kwargs)


@pytest.mark.parametrize("ambient", [False, True])
def test_native_failure_has_no_raw_or_ambient_chain(monkeypatch, tmp_path, ambient):
    module = importlib.import_module("lobster.core.adapters.transcriptomics_adapter")

    def fail(*args, **kwargs):
        raise ValueError("synthetic_native_marker")

    monkeypatch.setattr(module.anndata, "read_h5ad", fail)
    calls = forbid_pandas_read(monkeypatch)
    adapter = TranscriptomicsAdapter("bulk")
    with pytest.raises(Exception) as caught:
        if ambient:
            try:
                raise RuntimeError("synthetic_ambient_marker")
            except RuntimeError:
                adapter._load_h5_transcriptomics_data(tmp_path / "absent.h5")
        else:
            adapter._load_h5_transcriptomics_data(tmp_path / "absent.h5")
    assert calls == [], "pandas_fallback_reached"
    assert type(caught.value) is refusal_type()
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


def test_real_repair_failure_does_not_enter_pandas(monkeypatch, tmp_path):
    path = tmp_path / "invalid_base.h5"
    data = ad.AnnData(np.array([[1, 2], [3, 4]], dtype=np.float32))
    data.uns["log1p"] = {"base": "not_a_number"}
    data.write_h5ad(path)
    calls = forbid_pandas_read(monkeypatch)
    with pytest.raises(Exception) as caught:
        TranscriptomicsAdapter("bulk")._load_h5_transcriptomics_data(path)
    assert calls == [], "repair_entered_pandas"
    assert type(caught.value) is refusal_type()
    assert caught.value.__context__ is None


@pytest.mark.parametrize("loader", ["h5", "h5ad", "source"])
@pytest.mark.parametrize("kind", ["hdf", "inference", "managed"])
def test_unrelated_policy_identity_preserved(monkeypatch, tmp_path, loader, kind):
    from lobster.core.figures.export_guard import ManagedPathError

    if kind == "hdf":
        error = refusal_type()()
    elif kind == "inference":
        error = exceptions.InferenceRefused("policy")
    else:
        error = ManagedPathError("synthetic_policy")
    path = tmp_path / ("identity.h5" if loader == "h5" else "identity.h5ad")
    path.touch()

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(ad, "read_h5ad", fail)
    calls = forbid_pandas_read(monkeypatch)
    adapter = TranscriptomicsAdapter("bulk") if loader != "h5ad" else BaseAdapter()
    target = {
        "h5": adapter._load_h5_transcriptomics_data if loader == "h5" else None,
        "h5ad": adapter._load_h5ad_data,
        "source": adapter.from_source,
    }[loader]
    with pytest.raises(Exception) as caught:
        target(path)
    assert calls == [], "policy_entered_pandas"
    assert caught.value is error, "policy_identity_changed"


@pytest.mark.parametrize("encoding", ["dense", "csr", "csc"])
@pytest.mark.parametrize("extension", ["h5", "h5ad"])
def test_native_positive_leaf_and_adapter(monkeypatch, tmp_path, encoding, extension):
    matrix = np.array([[1, 2, 4], [5, 7, 8]], dtype=np.float32)
    x = {"dense": lambda m: m, "csr": sparse.csr_matrix, "csc": sparse.csc_matrix}[
        encoding
    ](matrix)
    original = ad.AnnData(
        x,
        obs=pd.DataFrame(index=["s_a", "s_b"]),
        var=pd.DataFrame(index=["g_a", "g_b", "g_c"]),
    )
    original.uns["log1p"] = {"base": 2.0}
    path = tmp_path / f"valid.{extension}"
    original.write_h5ad(path)
    calls = forbid_pandas_read(monkeypatch)
    adapter = TranscriptomicsAdapter("bulk")
    leaf = (
        adapter._load_h5_transcriptomics_data(path)
        if extension == "h5"
        else adapter._load_h5ad_data(path)
    )
    np.testing.assert_array_equal(
        leaf.X.toarray() if sparse.issparse(leaf.X) else leaf.X, matrix
    )
    assert leaf.obs_names.tolist() == original.obs_names.tolist()
    assert leaf.var_names.tolist() == original.var_names.tolist()
    assert leaf.X.dtype == np.float32
    result = adapter.from_source(path)
    np.testing.assert_array_equal(
        result.X.toarray() if sparse.issparse(result.X) else result.X, matrix
    )
    assert result.obs_names.tolist() == original.obs_names.tolist()
    assert result.var_names.tolist() == original.var_names.tolist()
    assert calls == []


def test_actual_dm_excluded_read_changes_no_state(monkeypatch, tmp_path):
    path = tmp_path / "excluded.h5"
    pandas_fixture(path, objects=True)
    dm = DataManagerV2(workspace_path=tmp_path / "workspace", auto_scan=False)
    old = ad.AnnData(np.ones((2, 2), dtype=np.float32))
    dm.modalities["existing"] = old
    dm._modality_sources["existing"] = tmp_path / "original.h5ad"
    dm._modality_dirty.add("existing")
    before = (
        dict(dm.modalities),
        dict(dm._modality_sources),
        set(dm._modality_dirty),
        dict(dm._modality_spilled),
    )
    dm.log_tool_usage = Mock()
    dm.on_modality_loaded = Mock()
    dm._enforce_memory_budget = Mock()
    calls = forbid_pandas_read(monkeypatch)
    with pytest.raises(Exception) as caught:
        dm.load_modality("existing", path, "transcriptomics_bulk")
    assert calls == [], "dm_entered_pandas"
    assert type(caught.value) is refusal_type()
    assert (
        dict(dm.modalities),
        dict(dm._modality_sources),
        set(dm._modality_dirty),
        dict(dm._modality_spilled),
    ) == before
    assert dm.modalities["existing"] is old
    dm.log_tool_usage.assert_not_called()
    dm.on_modality_loaded.assert_not_called()
    dm._enforce_memory_budget.assert_not_called()


def test_non_hdf_csv_positive(tmp_path):
    path = tmp_path / "counts.csv"
    pd.DataFrame({"s_a": [1, 3], "s_b": [2, 4]}, index=["g_a", "g_b"]).to_csv(path)
    result = TranscriptomicsAdapter("bulk").from_source(path)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(
        result.X, np.array([[1, 3], [2, 4]], dtype=np.float32)
    )


def test_source_origin():
    root = Path(__file__).resolve().parents[3]
    assert Path(exceptions.__file__).resolve() == root / "lobster/core/exceptions.py"
    module = importlib.import_module("lobster.core.runtime.data_manager")
    assert (
        Path(module.__file__).resolve() == root / "lobster/core/runtime/data_manager.py"
    )
