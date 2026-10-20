"""Real GEO facade and queue adapter preserve native refusal identity."""

from types import SimpleNamespace

import pytest

from lobster.config.inference_context import INFERENCE_BINDING_VERSION, InferenceContext
from lobster.config.inference_runtime import capture_inference_binding
from lobster.core.exceptions import InferenceRefused
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.data_access.geo_download_service import GEODownloadService
from tests.unit.config.test_native_request_hooks import (
    BEDROCK_MODEL,
    HookProbe,
    bedrock_model,
)

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
