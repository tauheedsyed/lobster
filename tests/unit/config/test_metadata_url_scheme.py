"""The three metadata HTTP boundaries reject non-HTTPS before transport."""

import urllib.parse
from types import FunctionType, SimpleNamespace

import pytest

from lobster.services.data_access.geo.metadata_fetch import MetadataFetcher
from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer

pytestmark = pytest.mark.no_auto_config

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


@pytest.mark.parametrize("site", ["gds", "entrez", "queue"])
@pytest.mark.parametrize(
    "endpoint",
    ["file:///local-only", "http://example.invalid", "ftp://example.invalid"],
)
def test_metadata_sites_reject_unapproved_endpoint_scheme(monkeypatch, site, endpoint):
    # Exercise the entire real method and real URL parser with a hypothetical
    # future endpoint change. Replace ONLY its fixed endpoint constant, never
    # the validator or service branch; all transport is forbidden.
    method = {
        "gds": MetadataFetcher._fetch_gds_metadata_and_convert,
        "entrez": MetadataFetcher._fetch_gse_metadata_via_entrez,
        "queue": GEOQueuePreparer._resolve_gds_to_gse,
    }[site]
    assert NCBI_BASE in method.__code__.co_consts
    code = method.__code__.replace(
        co_consts=tuple(
            endpoint if c == NCBI_BASE else c for c in method.__code__.co_consts
        )
    )
    call = FunctionType(code, method.__globals__)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Unapproved URL reached HTTP transport")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    # SSL context construction is not the boundary under test; no actual IO.
    if site != "queue":
        monkeypatch.setattr(
            "lobster.services.data_access.geo.metadata_fetch.create_ssl_context",
            lambda: None,
        )
    owner = SimpleNamespace()
    if site == "gds":
        # This legacy metadata API intentionally reports fetch failure as None.
        assert call(owner, "GDS1") == (None, None)
    else:
        with pytest.raises(ValueError, match="must use HTTPS"):
            call(owner, "GSE1" if site == "entrez" else "GDS1")
    assert not calls


@pytest.mark.parametrize("site", ["gds", "entrez", "queue"])
def test_https_metadata_transport_keeps_url_context_and_timeout(monkeypatch, site):
    method = {
        "gds": MetadataFetcher._fetch_gds_metadata_and_convert,
        "entrez": MetadataFetcher._fetch_gse_metadata_via_entrez,
        "queue": GEOQueuePreparer._resolve_gds_to_gse,
    }[site]
    calls = []

    def terminal_transport(url, **kwargs):
        assert urllib.parse.urlsplit(url).scheme == "https"
        calls.append((url, kwargs))
        raise ValueError("Synthetic transport terminal")

    monkeypatch.setattr("urllib.request.urlopen", terminal_transport)
    owner = SimpleNamespace()
    if site == "gds":
        assert method(owner, "GDS1") == (None, None)
    else:
        with pytest.raises(ValueError, match="Synthetic transport terminal"):
            method(owner, "GSE1" if site == "entrez" else "GDS1")
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 30
    assert "context" in calls[0][1]


@pytest.mark.asyncio
async def test_optional_metrics_failure_logs_only_exception_type(monkeypatch, caplog):
    from lobster.services.execution import sdk_delegation_service as module

    monkeypatch.setattr(
        module, "sdk_tool", lambda *a, **k: lambda fn: fn, raising=False
    )
    error_message = "synthetic-confidential-metrics-detail"

    def failed_metrics(name):
        raise ValueError(error_message)

    adata = SimpleNamespace(
        n_obs=1,
        n_vars=2,
        obs=SimpleNamespace(columns=[]),
        var=SimpleNamespace(columns=[]),
        layers={},
        obsm={},
        uns={},
    )
    service = object.__new__(module.SDKDelegationService)
    service.data_manager = SimpleNamespace(
        list_modalities=lambda: ["synthetic"],
        get_modality=lambda name: adata,
        get_quality_metrics=failed_metrics,
    )
    caplog.set_level("DEBUG", logger=module.logger.name)
    tool = service.create_lobster_sdk_tools()[1]
    result = await tool({"modality_name": "synthetic"})
    assert not result.get("is_error")
    assert "1 observations" in result["content"][0]["text"]
    assert "Optional quality metrics unavailable (ValueError)" in caplog.text
    assert error_message not in caplog.text
    assert error_message not in str(result)
