"""DownloadOrchestrator.execute_download(return_ir=True) hands provenance to the caller.

By default the orchestrator logs the download's provenance itself. A tool that
logs the download under its own name passes ``return_ir=True``; the orchestrator
then must not log, so the download appears exactly once in the provenance.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pytest

from lobster.core.schemas.download_queue import DownloadStatus
from lobster.tools.download_orchestrator import DownloadOrchestrator


class FakeService:
    def __init__(self, ir):
        self.ir = ir

    def supported_databases(self):
        return ["fakedb"]

    def download_dataset(self, entry, strategy_override=None):
        return ad.AnnData(X=np.ones((3, 2))), {"strategy_used": "auto"}, self.ir


@pytest.fixture
def setup():
    dm = MagicMock()
    dm.modalities = {}
    dm.download_queue.get_entry.return_value = SimpleNamespace(
        status=DownloadStatus.PENDING,
        database="fakedb",
        dataset_id="DS1",
        modality_name=None,
    )
    orchestrator = DownloadOrchestrator(dm)
    ir = object()
    orchestrator.register_service(FakeService(ir))
    return dm, orchestrator, ir


def test_default_logs_provenance_and_returns_two_values(setup):
    dm, orchestrator, ir = setup
    modality_name, stats = orchestrator.execute_download("entry_1")
    assert modality_name == "fakedb_DS1"
    dm.log_tool_usage.assert_called_once()
    assert dm.log_tool_usage.call_args.kwargs["ir"] is ir


def test_return_ir_hands_back_ir_without_logging(setup):
    dm, orchestrator, ir = setup
    modality_name, stats, returned_ir = orchestrator.execute_download(
        "entry_1", return_ir=True
    )
    assert modality_name == "fakedb_DS1"
    assert returned_ir is ir
    assert stats == {"strategy_used": "auto"}
    dm.log_tool_usage.assert_not_called()
    dm.download_queue.update_status.assert_called_with(
        "entry_1", DownloadStatus.COMPLETED, modality_name="fakedb_DS1"
    )


def test_return_ir_when_entry_claimed_elsewhere(setup):
    dm, orchestrator, _ = setup
    dm.download_queue.update_status.return_value = None  # CAS claim lost
    assert orchestrator.execute_download("entry_1", return_ir=True) == ("", {}, None)
