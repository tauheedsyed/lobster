"""Drawing metrics are outputs of the QC operation, including zero-count cells."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from lobster.services.quality.quality_service import QualityService


@pytest.mark.parametrize("sparse", [False, True])
def test_drawing_metrics_on_analysis_copy_and_in_ir(sparse):
    matrix = np.array([[4, 2, 2, 2], [0, 0, 0, 0]], dtype=float)
    data = AnnData(
        X=csr_matrix(matrix) if sparse else matrix,
        obs=pd.DataFrame(index=["cell-a", "cell-zero"]),
        var=pd.DataFrame(index=["HBA1", "MT-ND1", "RPL3", "ACTB"]),
    )
    before = data.obs.copy(deep=True)
    result, _, ir = QualityService().assess_quality(data, min_genes=1, max_genes=10)
    pd.testing.assert_frame_equal(data.obs, before)
    assert result.obs["n_counts"].tolist() == [10, 0]
    assert result.obs["n_genes"].tolist() == [4, 0]
    assert result.obs["gene_detection_rate"].tolist() == [100, 0]
    assert result.obs["percent_hb"].tolist() == [40, 0]
    assert np.isfinite(result.obs.select_dtypes(include="number")).all().all()

    replayed = data.copy()
    namespace = {"adata": replayed, "np": np}
    for helper in ir.helper_code:
        exec(
            helper, namespace
        )  # nosec B102 # Trusted repository IR on synthetic inputs.
    exec(
        ir.render(), namespace
    )  # nosec B102 # Trusted repository IR on synthetic inputs.
    pd.testing.assert_frame_equal(replayed.obs, result.obs)
