"""get_modality_details on a bulk RNA-seq TPM modality loaded from GEO.

The tool joined AnnData keys with ``', '.join``. anndata>=0.13 lists ``.X`` in
``.layers`` under the key ``None``, so every modality raised ``TypeError:
sequence item 0: expected str instance, NoneType found``. Shape of a real GEO
supplementary TPM matrix: 20 samples x 50,006 genes, no real layers, three
injected sample columns. Values are synthetic.
"""

from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

N_OBS, N_VARS = 20, 50006


def _m5_adata():
    rng = np.random.default_rng(0)
    raw = rng.gamma(0.5, 2.0, size=(N_OBS, N_VARS)).astype(np.float32)
    tpm = raw / raw.sum(axis=1, keepdims=True) * 1e6  # TPM: rows sum to 1e6
    obs = pd.DataFrame(
        {
            "sample_id": pd.Categorical([f"S{i:02d}" for i in range(N_OBS)]),
            "tissue": pd.Categorical(["tumor", "normal"] * (N_OBS // 2)),
            "tissue_group": pd.Categorical(["A"] * 10 + ["B"] * 10),
        },
        index=[f"GSM{9000000 + i}" for i in range(N_OBS)],
    )
    var = pd.DataFrame(index=[f"ENSG{i:011d}" for i in range(N_VARS)])
    return ad.AnnData(X=tpm, obs=obs, var=var)


def _tool(dm, tmp_path, name="get_modality_details"):
    from lobster.agents.data_expert import data_expert as module

    captured = {}

    def capture_agent(**kwargs):
        captured["tools"] = kwargs.get("tools", [])
        return MagicMock()

    settings = MagicMock()
    settings.get_agent_llm_params.return_value = {}
    llm = MagicMock()
    llm.with_config.return_value = llm
    with (
        patch.object(module, "create_llm", return_value=llm),
        patch.object(module, "get_settings", return_value=settings),
        patch.object(module, "create_react_agent", side_effect=capture_agent),
    ):
        module.data_expert(data_manager=dm, workspace_path=tmp_path)
    return next(t for t in captured["tools"] if t.name == name)


@pytest.fixture
def m5(tmp_path):
    from lobster.core.runtime.data_manager import DataManagerV2

    dm = DataManagerV2(workspace_path=tmp_path, auto_scan=False)
    adata = _m5_adata()
    dm.store_modality(name="geo_gse_tpm", adata=adata, step_summary="bulk TPM")
    return dm, tmp_path


def test_m5_tpm_modality_details_render(m5):
    dm, tmp_path = m5
    layers = list(dm.get_modality("geo_gse_tpm").layers.keys())
    # The trigger: on anndata>=0.13 the only "layer" key is None (.X itself).
    if tuple(int(p) for p in ad.__version__.split(".")[:2]) >= (0, 13):
        assert None in layers
    result = _tool(dm, tmp_path).invoke({"modality_name": "geo_gse_tpm"})
    assert result.startswith("## Modality: geo_gse_tpm"), result
    assert f"**Shape**: {N_OBS} obs × {N_VARS} vars" in result
    assert "sample_id, tissue, tissue_group" in result
    assert "Error" not in result and "None" not in result
    assert "**Layers**" not in result  # .X alias is not a layer


def test_real_layers_listed_without_x_alias(m5):
    dm, tmp_path = m5
    adata = dm.get_modality("geo_gse_tpm")
    adata.layers["tpm"] = adata.X.copy()
    dm.store_modality(name="geo_gse_tpm", adata=adata, step_summary="tpm layer")
    result = _tool(dm, tmp_path).invoke({"modality_name": "geo_gse_tpm"})
    assert "**Layers**: tpm\n" in result, result


def test_non_str_keys_and_columns_render_as_text(m5):
    """Same join, other containers: int column labels, int/None mapping keys."""
    from lobster.services.data_management import modality_management_service

    info = {
        "name": "geo_gse_tpm",
        "shape": {"n_obs": N_OBS, "n_vars": N_VARS},
        "is_sparse": False,
        "obs_columns": ["sample_id", 7],
        "var_columns": [0, 1],
        "layers": [None, "counts", 3],
        "obsm_keys": [None, "X_pca"],
        "varm_keys": [2],
        "uns_keys": ["provenance", 5],
        "quality_metrics": {},
    }
    dm, tmp_path = m5
    with patch.object(
        modality_management_service.ModalityManagementService,
        "get_modality_info",
        return_value=(info, {}, None),
    ):
        result = _tool(dm, tmp_path).invoke({"modality_name": "geo_gse_tpm"})
    assert "sample_id, 7" in result and "  0, 1\n" in result, result
    assert "**Layers**: counts, 3\n" in result
    assert "**Obsm Keys**: None, X_pca\n" in result  # only layers drop the alias
    assert "**Varm Keys**: 2\n" in result and "**Uns Keys**: provenance, 5\n" in result
