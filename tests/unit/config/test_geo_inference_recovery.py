"""Real GEO facade and queue adapter preserve native refusal identity."""

from types import SimpleNamespace

import pandas as pd
import pytest

from tests.unit.config._native_inference_helpers import (
    BEDROCK_MODEL,
    HookProbe,
    bedrock_model,
    require_agents,
    require_bedrock,
    require_optional,
)

# Both cases exercise the GEO facade and the actual research helper/native SDK.
require_bedrock()
require_agents("data_expert_agent")
require_optional("scanpy", extra="transcriptomics")

from lobster.config.inference_context import INFERENCE_BINDING_VERSION, InferenceContext
from lobster.config.inference_runtime import capture_inference_binding
from lobster.core.exceptions import InferenceRefused
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.data_access.geo_download_service import GEODownloadService

pytestmark = pytest.mark.no_auto_config


@pytest.mark.parametrize("wrapped", [False, True])
def test_geo_download_adapter_preserves_refusal_from_actual_helper_request(
    tmp_path, monkeypatch, wrapped
):
    from lobster.services.data_access.geo import metadata_fetch

    refusal = InferenceRefused("accounting")
    error = RuntimeError("Synthetic request wrapper") if wrapped else refusal
    if wrapped:
        error.__cause__ = refusal
    probe = HookProbe()
    probe.error = error
    model = bedrock_model(probe)
    context = InferenceContext(
        "bedrock", BEDROCK_MODEL, "synthetic-binding", "version-a", 200000, 137
    )

    class Factory:
        capability_version = INFERENCE_BINDING_VERSION

        def create(self, *, context, component):
            return model

    manager = DataManagerV2(workspace_path=tmp_path)
    capture_inference_binding(manager, context, Factory())
    # Synthetic remote SOFT input, not a replacement scientific-service branch.
    gse = SimpleNamespace(
        metadata={
            "title": ["Synthetic ambiguous study"],
            "summary": ["Synthetic sample study"],
        },
        gsms={
            "GSM1": SimpleNamespace(
                metadata={"characteristics_ch1": ["disease: healthy"]}
            )
        },
        gpls={"GPL11154": SimpleNamespace(metadata={"title": ["Illumina HiSeq"]})},
    )
    monkeypatch.setattr(metadata_fetch, "pre_download_soft_file", lambda *a, **k: None)
    monkeypatch.setattr(metadata_fetch.GEOparse, "get_GEO", lambda **k: gse)
    service = GEODownloadService(manager)
    entry = SimpleNamespace(
        database="geo", dataset_id="GSE1", recommended_strategy=None
    )
    with pytest.raises(InferenceRefused) as caught:
        service.download_dataset(entry)
    assert caught.value is refusal
    assert len(probe.hooks) == 1
    assert not probe.sent
    model.client.close()


@pytest.mark.parametrize("wrapped", [False, True])
def test_sample_worker_future_does_not_turn_native_refusal_into_missing_matrix(
    tmp_path, monkeypatch, wrapped
):
    from lobster.services.data_access.geo import matrix_parsing
    from lobster.services.data_access.geo_service import GEOService

    refusal = InferenceRefused("budget")
    error = RuntimeError("Synthetic request wrapper") if wrapped else refusal
    if wrapped:
        error.__cause__ = refusal
    probe = HookProbe()
    probe.error = error
    model = bedrock_model(probe)
    context = InferenceContext(
        "bedrock", BEDROCK_MODEL, "synthetic-binding", "version-a", 200000, 137
    )

    class Factory:
        capability_version = INFERENCE_BINDING_VERSION

        def create(self, *, context, component):
            return model

    manager = DataManagerV2(workspace_path=tmp_path)
    manager.metadata_store["GSE1"] = {"data_type": "bulk_rna_seq"}
    capture_inference_binding(manager, context, Factory())

    class PartialRemoteSample:
        """Synthetic SOFT object with a transient metadata-read failure."""

        table = pd.DataFrame([[1, 2], [3, 4]])
        reads = 0

        @property
        def metadata(self):
            self.reads += 1
            if self.reads == 1:
                raise ValueError("Synthetic partial SOFT metadata")
            return {"title": "synthetic sample"}

    monkeypatch.setattr(matrix_parsing, "pre_download_soft_file", lambda *a, **k: None)
    monkeypatch.setattr(
        matrix_parsing.GEOparse, "get_GEO", lambda **k: PartialRemoteSample()
    )
    service = GEOService(manager)
    with pytest.raises(InferenceRefused) as caught:
        service._download_sample_matrices({"GSM1": {}}, "GSE1")
    assert caught.value is refusal
    assert len(probe.hooks) == 1
    assert not probe.sent
    model.client.close()
