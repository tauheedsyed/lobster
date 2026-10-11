"""execute_download_from_queue logs the download once, with the service's IR."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np

from lobster.agents.data_expert.data_expert import data_expert
from lobster.core.schemas.download_queue import DownloadStatus
from lobster.testing.provenance_replay import RecordingDataManager, factory_tools

MODULE = "lobster.agents.data_expert.data_expert"


def test_download_logged_once_with_service_ir(tmp_path):
    dm = RecordingDataManager(workspace_path=tmp_path)
    entry = SimpleNamespace(
        entry_id="queue_GSE1_abc",
        dataset_id="GSE1",
        database="geo",
        status=DownloadStatus.PENDING,
        recommended_strategy=None,
        validation_status=None,
        modality_name=None,
    )
    dm.download_queue = MagicMock()
    dm.download_queue.list_entries.return_value = [entry]
    dm.download_queue.get_entry.return_value = entry
    dm.modalities["geo_gse1"] = ad.AnnData(X=np.ones((4, 20000)))

    download_ir = object()
    orchestrator = MagicMock()
    orchestrator.execute_download.return_value = ("geo_gse1", {}, download_ir)

    tools = factory_tools(data_expert, MODULE, dm)
    with patch(f"{MODULE}.DownloadOrchestrator", return_value=orchestrator):
        result = tools["execute_download_from_queue"].invoke(
            {"entry_id": "queue_GSE1_abc"}
        )

    assert "Download completed successfully" in result, result
    assert orchestrator.execute_download.call_args.kwargs == {"return_ir": True}
    calls = dm.calls_for("execute_download_from_queue")
    assert len(calls) == 1
    assert calls[0].ir is download_ir
    assert calls[0].parameters["database"] == "geo"
    # The orchestrator was told not to log, so nothing else recorded the download.
    assert [c.tool_name for c in dm.calls] == ["execute_download_from_queue"]
