"""Narrow unit tests for DownloadExecutor domain module.

Tests DownloadExecutor methods in isolation via mocked service.
Part of Phase 4 Plan 03: GEO Service Decomposition.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lobster.services.data_access.geo.download_execution import DownloadExecutor


@pytest.fixture
def mock_service():
    """Create a mock GEOService with all attributes DownloadExecutor needs."""
    service = MagicMock()
    service.data_manager = MagicMock()
    service.data_manager.metadata_store = {}
    service.cache_dir = Path("/tmp/test_geo_cache")
    service.console = MagicMock()
    service.geo_downloader = MagicMock()
    service.geo_parser = MagicMock()
    service.pipeline_engine = MagicMock()
    service.tenx_loader = MagicMock()
    service.download_strategy = MagicMock()
    return service


@pytest.fixture
def executor(mock_service):
    """Create a DownloadExecutor with mocked service."""
    return DownloadExecutor(mock_service)


class TestDownloadExecutorInit:
    """Test DownloadExecutor initialization."""

    def test_init_stores_service_reference(self, mock_service):
        executor = DownloadExecutor(mock_service)
        assert executor.service is mock_service


class TestDownloadDataset:
    """Test download_dataset routing and validation."""

    def test_rejects_invalid_geo_id(self, executor):
        """Non-GSE IDs should return error message."""
        result = executor.download_dataset("INVALID123")
        assert "Invalid GEO ID format" in result

    def test_rejects_gds_id(self, executor):
        """GDS IDs should be rejected (download requires GSE)."""
        result = executor.download_dataset("GDS5826")
        assert "Invalid GEO ID format" in result


class TestDownloadWithStrategy:
    """Test download_with_strategy orchestration."""

    def test_sets_use_intersecting_genes_only_on_service(self, executor, mock_service):
        """download_with_strategy should store concatenation strategy on service."""
        # Make metadata store return something to avoid early exit
        mock_service.data_manager.metadata_store = {"GSE123": {"metadata": {}}}
        mock_service.data_manager._get_geo_metadata.return_value = {
            "metadata": {},
            "strategy_config": {"pipeline_type": "supplementary_first"},
        }
        mock_service._determine_data_type_from_metadata.return_value = (
            "single_cell_rna_seq"
        )
        mock_service.pipeline_engine.determine_pipeline.return_value = (
            MagicMock(name="SUPPLEMENTARY_FIRST"),
            "test",
        )
        mock_service.pipeline_engine.get_pipeline_functions.return_value = []

        executor.download_with_strategy("GSE123", use_intersecting_genes_only=True)
        assert mock_service._use_intersecting_genes_only is True

    def test_cleans_geo_id(self, executor, mock_service):
        """download_with_strategy should strip and uppercase GEO ID."""
        mock_service.data_manager.metadata_store = {"GSE123": {"metadata": {}}}
        mock_service.data_manager._get_geo_metadata.return_value = {
            "metadata": {},
            "strategy_config": {},
        }
        mock_service._determine_data_type_from_metadata.return_value = "bulk_rna_seq"
        mock_service.pipeline_engine.determine_pipeline.return_value = (
            MagicMock(name="FALLBACK"),
            "default fallback",
        )
        mock_service.pipeline_engine.get_pipeline_functions.return_value = []

        executor.download_with_strategy("  gse123  ")
        # The cleaned ID should be used for metadata store lookup
        assert mock_service._use_intersecting_genes_only is None

    @patch("lobster.services.data_access.geo.download_execution.logger")
    def test_pipeline_step_failures_stay_in_debug_logs(
        self, mock_logger, executor, mock_service
    ):
        """Expected fallback misses should not be emitted as warnings."""
        mock_service.data_manager.metadata_store = {"GSE123": {"metadata": {}}}
        mock_service.data_manager._get_geo_metadata.return_value = {
            "metadata": {},
            "strategy_config": {"pipeline_type": "supplementary_first"},
        }
        mock_service._determine_data_type_from_metadata.return_value = "bulk_rna_seq"
        mock_service.pipeline_engine.determine_pipeline.return_value = (
            MagicMock(name="SUPPLEMENTARY_FIRST"),
            "test description",
        )

        failed_step = MagicMock()
        failed_step.__name__ = "_try_processed_matrix_first"
        failed_step.return_value = MagicMock(
            success=False,
            error_message="No processed matrix information available in strategy config",
        )
        mock_service.pipeline_engine.get_pipeline_functions.return_value = [failed_step]

        result = executor.download_with_strategy("GSE123")

        assert result.success is False
        assert result.error_message == "All pipeline steps failed after enough attempts"
        mock_logger.warning.assert_not_called()
        assert any(
            "Step failed: No processed matrix information available in strategy config"
            in call.args[0]
            for call in mock_logger.debug.call_args_list
        )


class TestGetProcessingPipeline:
    """Test _get_processing_pipeline pipeline selection."""

    def test_returns_callable_pipeline_list(self, executor, mock_service):
        """_get_processing_pipeline should return list of callables."""
        mock_service._determine_data_type_from_metadata.return_value = "bulk_rna_seq"

        mock_func = MagicMock()
        mock_service.pipeline_engine.determine_pipeline.return_value = (
            MagicMock(name="SUPPLEMENTARY_FIRST"),
            "test description",
        )
        mock_service.pipeline_engine.get_pipeline_functions.return_value = [mock_func]

        result = executor._get_processing_pipeline(
            "GSE123",
            {"samples": {}},
            {"pipeline_type": "supplementary_first"},
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is mock_func


# These fixtures exercise the cached producer, parser, adapter and H5AD writer,
# not a mocked GEOResult or a mocked DataManager.
def _cached_processed_matrix(tmp_path, frame, suffix, parser_backend, monkeypatch):
    import builtins

    from lobster.core.runtime.data_manager import DataManagerV2
    from lobster.services.data_access.geo_service import GEOService

    if parser_backend == "pandas":
        original_import = builtins.__import__

        def without_polars(name, *args, **kwargs):
            if name == "polars":
                raise ImportError("Exercise the supported pandas parser fallback")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", without_polars)

    dm = DataManagerV2(workspace_path=tmp_path / "workspace", auto_scan=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    geo_id = "GSE999999999"
    filename = f"processed_matrix{suffix}"
    frame.to_csv(cache / f"{geo_id}_{filename}", sep="," if ".csv" in suffix else "\t")
    metadata = {
        "title": "Offline synthetic expression matrix",
        "summary": "Single cell RNA sequencing",
        "samples": {},
        "platforms": {},
        "supplementary_file": [f"https://example.invalid/{filename}"],
    }
    dm._store_geo_metadata(
        geo_id=geo_id,
        metadata=metadata,
        stored_by="offline test fixture",
        strategy_config={
            "processed_matrix_name": "processed_matrix",
            "processed_matrix_filetype": suffix.lstrip("."),
            "raw_data_available": False,
        },
    )
    # This URL identifies an already-written cache file. Any actual network use
    # must fail, rather than silently substituting a mocked download result.
    service = GEOService(dm, cache_dir=cache)

    def no_download(*args, **kwargs):
        pytest.fail("The cached matrix test must not download data")

    monkeypatch.setattr(service.geo_downloader, "download_file", no_download)
    return dm, service, geo_id


@pytest.mark.parametrize("parser_backend", ["polars", "pandas"])
@pytest.mark.parametrize("suffix", [".tsv", ".csv.gz"])
@pytest.mark.parametrize("genes_in_rows", [True, False])
@pytest.mark.parametrize(
    "dimensions,label_style",
    [
        ((120, 80), "prefixed"),
        ((80, 120), "prefixed"),
        ((80, 80), "prefixed"),
        ((80, 80), "10x"),
        ((80, 80), "sample-qualified"),
    ],
)
def test_cached_single_cell_matrix_uses_axis_labels(
    tmp_path,
    monkeypatch,
    parser_backend,
    suffix,
    genes_in_rows,
    dimensions,
    label_style,
):
    """Labels determine orientation even when shape is square or misleading."""
    import anndata
    import numpy as np
    import pandas as pd

    from lobster.services.data_access.geo.strategy import PipelineType

    n_genes, n_cells = dimensions
    genes = ["MT-CO1", "RPS3", "LYZ", "CD3D"] + [
        f"ENSG{i:011d}" for i in range(n_genes - 4)
    ]
    cells = [f"bc_ctrlI_{i:03d}.001" for i in range(n_cells)]
    if label_style != "prefixed":
        cells = [
            format(i, "016b").translate(str.maketrans("01", "AC")) + "-1"
            for i in range(n_cells)
        ]
        if label_style == "sample-qualified":
            cells = ["donor1_" + cell for cell in cells]
    values = np.arange(n_cells * n_genes).reshape(n_cells, n_genes) % 97
    expected = pd.DataFrame(values, index=cells, columns=genes)
    source = expected.T if genes_in_rows else expected
    dm, service, geo_id = _cached_processed_matrix(
        tmp_path, source, suffix, parser_backend, monkeypatch
    )
    response = service.download_dataset(
        geo_id,
        adapter="transcriptomics_single_cell",
        manual_strategy_override=PipelineType.MATRIX_FIRST,
    )
    assert "Successfully downloaded" in response, response
    name = f"geo_{geo_id.lower()}_transcriptomics_single_cell"
    adata = dm.get_modality(name)
    assert list(adata.obs_names) == cells
    assert list(adata.var_names) == genes
    np.testing.assert_array_equal(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, values
    )
    info = adata.uns.get("transpose_info")
    assert info is not None, "Every orientation decision needs provenance"
    # The existing metadata sanitizer serializes scalar booleans as strings.
    assert info["transpose_applied"] == str(genes_in_rows)
    assert info["transpose_reason"]
    assert list(info["original_shape"]) == list(source.shape)
    assert list(info["final_shape"]) == [n_cells, n_genes]
    assert info["orientation_basis"] == "axis_labels"
    assert info["row_axis"] == ("genes" if genes_in_rows else "cells")
    assert info["column_axis"] == ("cells" if genes_in_rows else "genes")
    saved = anndata.read_h5ad(dm.data_dir / f"{name}_raw.h5ad")
    assert list(saved.obs_names) == cells
    assert list(saved.var_names) == genes
    assert saved.uns["transpose_info"]["transpose_applied"] == str(genes_in_rows)


@pytest.mark.parametrize(
    "row_labels,column_labels",
    [
        ([f"row_{i}" for i in range(80)], [f"column_{i}" for i in range(120)]),
        (
            [f"ENSG{i:011d}" for i in range(80)],
            [f"ENSMUSG{i:011d}" for i in range(120)],
        ),
        (
            [f"bc_ctrlI_{i}.001" for i in range(80)],
            [f"bc_case_{i}.001" for i in range(120)],
        ),
        (
            [f"ENSG{i:011d}" if i % 2 else f"bc_{i}.001" for i in range(80)],
            [f"bc_case_{i}.001" for i in range(120)],
        ),
    ],
    ids=[
        "unrecognized",
        "genes-on-both-axes",
        "cells-on-both-axes",
        "conflicting-row-axis",
    ],
)
def test_cached_ambiguous_single_cell_matrix_refuses_before_loading(
    tmp_path, monkeypatch, row_labels, column_labels
):
    import numpy as np
    import pandas as pd

    from lobster.services.data_access.geo.strategy import PipelineType

    source = pd.DataFrame(
        np.ones((len(row_labels), len(column_labels))),
        index=row_labels,
        columns=column_labels,
    )
    dm, service, geo_id = _cached_processed_matrix(
        tmp_path, source, ".tsv", "polars", monkeypatch
    )
    from lobster.services.data_access.geo.constants import GEOServiceError

    with pytest.raises(GEOServiceError, match="[Aa]mbiguous.*orientation"):
        service.download_dataset(
            geo_id,
            adapter="transcriptomics_single_cell",
            manual_strategy_override=PipelineType.MATRIX_FIRST,
        )
    assert not dm.list_modalities(), "Refused data must not be registered"
    assert not list((tmp_path / "workspace").rglob("*_raw.h5ad"))


def test_cached_bulk_processed_matrix_retains_format_transpose(tmp_path, monkeypatch):
    """The single-cell resolver must not change existing bulk handling."""
    import numpy as np
    import pandas as pd

    from lobster.services.data_access.geo.strategy import PipelineType

    genes = [f"gene_{i}" for i in range(120)]
    samples = [f"sample_{i}" for i in range(8)]
    source = pd.DataFrame(np.arange(960).reshape(120, 8), index=genes, columns=samples)
    dm, service, geo_id = _cached_processed_matrix(
        tmp_path, source, ".tsv", "polars", monkeypatch
    )
    response = service.download_dataset(
        geo_id,
        adapter="transcriptomics_bulk",
        manual_strategy_override=PipelineType.MATRIX_FIRST,
    )
    assert "Successfully downloaded" in response
    adata = dm.get_modality(f"geo_{geo_id.lower()}_transcriptomics_bulk")
    assert list(adata.obs_names) == samples
    assert list(adata.var_names) == genes
    np.testing.assert_array_equal(adata.X, source.T.values)


def test_cached_h5ad_retains_explicit_axes(tmp_path, monkeypatch):
    """H5AD format's explicit axes need no generic label inference."""
    import anndata
    import numpy as np
    import pandas as pd

    from lobster.services.data_access.geo.strategy import PipelineType

    source = pd.DataFrame(
        np.arange(9600).reshape(80, 120),
        index=[f"observation_{i}" for i in range(80)],
        columns=[f"feature_{i}" for i in range(120)],
    )
    dm, service, geo_id = _cached_processed_matrix(
        tmp_path, source, ".tsv", "polars", monkeypatch
    )
    filename = "processed_matrix.h5ad"
    anndata.AnnData(source).write_h5ad(service.cache_dir / f"{geo_id}_h5_{filename}")
    entry = dm.metadata_store[geo_id]
    entry["metadata"]["supplementary_file"] = [f"https://example.invalid/{filename}"]
    entry["strategy_config"]["processed_matrix_filetype"] = "h5ad"
    response = service.download_dataset(
        geo_id,
        adapter="transcriptomics_single_cell",
        manual_strategy_override=PipelineType.H5_FIRST,
    )
    assert "Successfully downloaded" in response
    adata = dm.get_modality(f"geo_{geo_id.lower()}_transcriptomics_single_cell")
    assert list(adata.obs_names) == list(source.index)
    assert list(adata.var_names) == list(source.columns)
    np.testing.assert_array_equal(adata.X, source.values)
