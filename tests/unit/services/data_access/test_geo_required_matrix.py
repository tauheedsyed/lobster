"""Synthetic active GEO selection: one complete count matrix, no fallback."""

import gzip
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from lobster.services.data_access.geo.archive_processing import ArchiveProcessor
from lobster.services.data_access.geo.constants import GEODataSource, GEOResult
from lobster.services.data_access.geo.download_execution import DownloadExecutor
from lobster.services.data_access.geo.import_requirements import ImportRequirementNotMet
from lobster.services.data_access.geo.parser import GEOParser, ParseResult

GSE = "GSE900000000001"
A = "GSM900000000001"
B = "GSM900000000002"
SAMPLES = {A: {"title": "Synthetic A"}, B: {"title": "Synthetic B"}}
BASE = "https://example.test/"


def _table(features=None, titles=False):
    return pd.DataFrame(
        [[1, 3], [2, 4], [5, 6]],
        index=features or ["synthetic_a", "synthetic_b", "synthetic_c"],
        columns=["Synthetic A", "Synthetic B"] if titles else [A, B],
    )


def _executor(files, results=None):
    metadata = {"samples": SAMPLES, "supplementary_file": files}
    service = MagicMock()
    service.data_manager.metadata_store = {GSE: {"metadata": metadata}}
    service.data_manager._get_geo_metadata.return_value = {
        "metadata": metadata,
        "strategy_config": {},
    }
    service._download_and_parse_result.side_effect = lambda url, _: (
        results[url]
        if results is not None
        else ParseResult(_table(), original_tokens_valid=True)
    )
    executor = DownloadExecutor(service)
    executor._get_processing_pipeline = MagicMock(
        side_effect=AssertionError("Legacy chooser called")
    )
    return executor, service


def _required(executor, **kwargs):
    return executor.download_with_strategy(
        GSE, data_requirement="gene_counts", **kwargs
    )


def test_counts_selected_with_all_other_evidence_and_no_fallback():
    urls = [
        BASE + n
        for n in [
            "synthetic_gene_counts.csv",
            "synthetic_gene_fractional.csv",
            "synthetic_gene_scaled.csv",
            "synthetic_transcript_counts.csv",
            "synthetic_gene_archive.tar.gz",
            "synthetic_metadata.csv",
        ]
    ]
    fractional = _table().astype(float)
    fractional.iloc[0, 0] = 1.5
    scaled = _table()
    scaled[B] = scaled[A]
    results = {
        urls[0]: ParseResult(_table(titles=True), original_tokens_valid=True),
        urls[1]: ParseResult(fractional, original_tokens_valid=True),
        urls[2]: ParseResult(scaled, original_tokens_valid=True),
    }
    executor, service = _executor(urls, results)
    result = _required(
        executor, manual_strategy_override="RAW_FIRST", use_intersecting_genes_only=True
    )
    assert result.success and result.source == GEODataSource.SUPPLEMENTARY
    assert result.processing_info["method"] == "required_counts_matrix"
    assert result.processing_info["data_requirement"] == "gene_counts"
    assert result.processing_info["file"] == "synthetic_gene_counts.csv"
    assert result.processing_info["sample_rule"] == "title_exact"
    assert result.data.attrs["source_sample_id"] == {"Synthetic A": A, "Synthetic B": B}
    assert result.data.shape == (2, 3)
    evidence = {e["file_name"]: e for e in result.processing_info["candidates"]}
    assert len(evidence) == 6
    assert (
        evidence["synthetic_gene_counts.csv"]["reason"]
        == "whole-number counts (numeric check)"
    )
    assert evidence["synthetic_gene_counts.csv"]["verdict"] == "accepted"
    assert evidence["synthetic_gene_scaled.csv"]["equal_sample_totals"] is True
    assert evidence["synthetic_gene_fractional.csv"]["whole_numbers"] is False
    assert {
        call.args[0] for call in service._download_and_parse_result.call_args_list
    } == set(urls[:3])
    executor._get_processing_pipeline.assert_not_called()
    service.pipeline_engine.get_pipeline_functions.assert_not_called()
    service.data_manager.load_modality.assert_not_called()


@pytest.mark.parametrize(
    "features, accepted",
    [
        (["synthetic_a", "synthetic_b", "synthetic_c"], False),
        (["ENSG900000000001.2", "ENSMUSG900000000002", "ENSG900000000003"], True),
        (["ENSG900000000001", "synthetic_b", "ENSG900000000003"], False),
    ],
)
def test_unknown_filename_requires_every_feature_to_establish_gene_level(
    features, accepted
):
    url = BASE + "synthetic_matrix.csv"
    executor, _ = _executor(
        [url], {url: ParseResult(_table(features), original_tokens_valid=True)}
    )
    if accepted:
        result = _required(executor)
        assert result.processing_info["candidates"][0]["level"] == "gene"
    else:
        with pytest.raises(ImportRequirementNotMet) as caught:
            _required(executor)
        assert caught.value.problem == "no_candidate"
        assert caught.value.candidates[0].reason == "gene level not established"


@pytest.mark.parametrize(
    "transcript", ["ENST900000000001.3", "ENSMUST900000000001", "enst900000000001"]
)
def test_any_transcript_id_rejects_even_explicit_gene_filename(transcript):
    url = BASE + "synthetic_gene_counts.csv"
    executor, _ = _executor(
        [url],
        {
            url: ParseResult(
                _table([transcript, "ENSG900000000002", "ENSG900000000003"]),
                original_tokens_valid=True,
            )
        },
    )
    with pytest.raises(ImportRequirementNotMet) as caught:
        _required(executor)
    assert caught.value.candidates[0].level == "transcript"
    assert "transcript" in caught.value.candidates[0].reason


@pytest.mark.parametrize(
    "parsed",
    [
        None,
        ParseResult(),
        ParseResult(
            _table(), is_partial=True, rows_read=2, truncation_reason="synthetic limit"
        ),
        _table(),
    ],
)
def test_only_actual_complete_parse_result_is_admitted(parsed):
    url = BASE + "synthetic_gene_counts.csv"
    executor, _ = _executor([url], {url: parsed})
    with pytest.raises(ImportRequirementNotMet) as caught:
        _required(executor)
    evidence = caught.value.candidates[0]
    assert evidence.complete_parse is False
    assert "completely" in evidence.reason


@pytest.mark.parametrize("value", [-1, 1.5, np.nan, np.inf])
def test_invalid_numeric_values_reject(value):
    url = BASE + "synthetic_gene_counts.csv"
    table = _table().astype(float)
    table.iloc[0, 0] = value
    executor, _ = _executor(
        [url], {url: ParseResult(table, original_tokens_valid=True)}
    )
    with pytest.raises(ImportRequirementNotMet) as caught:
        _required(executor)
    assert caught.value.candidates[0].whole_numbers is False


def test_missing_original_token_proof_refuses_even_complete_numeric_matrix():
    url = BASE + "synthetic_gene_counts.csv"
    executor, _ = _executor([url], {url: ParseResult(_table())})
    with pytest.raises(ImportRequirementNotMet) as caught:
        _required(executor)
    assert caught.value.candidates[0].reason == "file could not be read completely"


def test_wrong_sample_identity_refuses():
    url = BASE + "synthetic_gene_counts.csv"
    table = _table()
    table.columns = ["synthetic_unknown_a", "synthetic_unknown_b"]
    executor, _ = _executor(
        [url], {url: ParseResult(table, original_tokens_valid=True)}
    )
    with pytest.raises(ImportRequirementNotMet) as caught:
        _required(executor)
    assert "sample identity" in caught.value.candidates[0].reason


def test_two_qualifiers_are_ambiguous_and_all_candidates_are_examined():
    urls = [BASE + "synthetic_gene_counts_a.csv", BASE + "synthetic_gene_counts_b.csv"]
    executor, service = _executor(urls)
    with pytest.raises(ImportRequirementNotMet) as caught:
        _required(executor)
    assert caught.value.problem == "ambiguous"
    assert service._download_and_parse_result.call_count == 2
    assert all(e.verdict == "accepted" for e in caught.value.candidates)


def test_cap_applies_after_cheap_filters_and_fails_closed_without_download():
    urls = [BASE + f"synthetic_gene_counts_{i}.csv" for i in range(9)] + [
        BASE + "synthetic_transcript_counts.csv",
        BASE + "synthetic_archive.tar.gz",
    ]
    executor, service = _executor(urls)
    with pytest.raises(ImportRequirementNotMet) as caught:
        _required(executor)
    assert caught.value.problem == "too_many_candidates"
    assert len([e for e in caught.value.candidates if e.verdict == "not_checked"]) == 9
    service._download_and_parse_result.assert_not_called()


def test_eight_with_several_cheap_rejections_is_not_a_false_cap():
    urls = [BASE + f"synthetic_gene_counts_{i}.csv" for i in range(8)] + [
        BASE + "synthetic_transcript_counts.csv",
        BASE + "synthetic_archive.tar.gz",
    ]
    results = {
        u: (
            ParseResult(_table(), original_tokens_valid=True)
            if u == urls[0]
            else ParseResult()
        )
        for u in urls[:8]
    }
    executor, service = _executor(urls, results)
    assert _required(executor).success
    assert service._download_and_parse_result.call_count == 8


def test_single_string_inventory_and_exact_duplicate_url_supported():
    url = BASE + "synthetic_gene_counts.csv"
    for inventory in (url, [url, url]):
        executor, service = _executor(inventory)
        assert _required(executor).success
        service._download_and_parse_result.assert_called_once_with(url, GSE)


def test_empty_inventory_refuses_without_fallback():
    executor, service = _executor([])
    with pytest.raises(ImportRequirementNotMet) as caught:
        _required(executor)
    assert caught.value.problem == "no_candidate" and caught.value.candidates == []
    service._download_and_parse_result.assert_not_called()
    executor._get_processing_pipeline.assert_not_called()


def test_unexpected_boundary_exception_propagates_and_never_falls_back():
    executor, service = _executor([BASE + "synthetic_gene_counts.csv"])
    error = RuntimeError("synthetic hard acquisition refusal")
    service._download_and_parse_result.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        _required(executor)
    assert caught.value is error
    executor._get_processing_pipeline.assert_not_called()


def test_unknown_requirement_fails_before_metadata_or_chooser():
    executor, service = _executor([])
    with pytest.raises(ValueError, match="Unknown data_requirement"):
        executor.download_with_strategy(GSE, data_requirement="synthetic_unknown")
    service.data_manager._get_geo_metadata.assert_not_called()


@pytest.mark.parametrize("kwargs", [{}, {"data_requirement": ""}])
def test_legacy_empty_requirement_keeps_pipeline_and_never_calls_sibling(kwargs):
    executor, service = _executor([BASE + "synthetic_gene_counts.csv"])
    expected = GEOResult(
        success=True, data=_table(), processing_info={"method": "synthetic_legacy"}
    )
    step = MagicMock(return_value=expected)
    step.__name__ = "synthetic_legacy"
    executor._get_processing_pipeline = MagicMock(return_value=[step])
    assert executor.download_with_strategy(GSE, **kwargs) is expected
    service._download_and_parse_result.assert_not_called()


@pytest.mark.parametrize("preexisting", [False, True])
def test_complete_parser_sibling_always_calls_download_boundary(tmp_path, preexisting):
    parse = ParseResult(_table())
    downloader = MagicMock()
    downloader.download_file.return_value = True
    parser = MagicMock()
    parser.parse_required_expression_file.return_value = parse
    service = SimpleNamespace(
        cache_dir=tmp_path, geo_downloader=downloader, geo_parser=parser
    )
    processor = ArchiveProcessor(service)
    url = BASE + "synthetic_gene_counts.csv"
    method = processor._download_and_parse_result
    if preexisting:
        assert method(url, GSE) is parse
        destination = downloader.download_file.call_args.args[1]
        destination.write_text("synthetic cached bytes")
        downloader.reset_mock()
    assert method(url, GSE) is parse
    downloader.download_file.assert_called_once()
    destination = downloader.download_file.call_args.args[1]
    assert destination.parent == tmp_path and destination.name.endswith(".csv")
    assert parser.parse_required_expression_file.call_args.args == (destination,)
    parser.parse_expression_file.assert_not_called()


def test_same_basename_different_urls_have_isolated_cache_paths(tmp_path):
    service = SimpleNamespace(
        cache_dir=tmp_path, geo_downloader=MagicMock(), geo_parser=MagicMock()
    )
    service.geo_downloader.download_file.return_value = True
    service.geo_parser.parse_required_expression_file.return_value = ParseResult(
        _table()
    )
    processor = ArchiveProcessor(service)
    for directory in ("synthetic_a", "synthetic_b"):
        processor._download_and_parse_result(
            BASE + directory + "/synthetic_gene_counts.csv", GSE
        )
    paths = [
        call.args[1] for call in service.geo_downloader.download_file.call_args_list
    ]
    assert paths[0] != paths[1]
    assert all(path.parent == tmp_path for path in paths)


@pytest.mark.parametrize(
    "extension, delimiter", [(".csv", ","), (".tsv", "\t"), (".csv.gz", ",")]
)
def test_real_parser_sibling_and_chooser_keep_synthetic_source_values(
    tmp_path, extension, delimiter
):
    source = _table()
    text = source.to_csv(sep=delimiter)
    payload = (
        gzip.compress(text.encode(), mtime=0)
        if extension.endswith(".gz")
        else text.encode()
    )
    url = BASE + "synthetic_gene_counts" + extension
    executor, service = _executor([url])
    service.cache_dir = tmp_path
    service.geo_parser = GEOParser()
    downloader = MagicMock()

    def acquire(candidate_url, destination):
        assert candidate_url == url
        assert destination.parent == tmp_path
        destination.write_bytes(payload)
        return True

    downloader.download_file.side_effect = acquire
    service.geo_downloader = downloader
    processor = ArchiveProcessor(service)
    service._download_and_parse_result = processor._download_and_parse_result
    result = _required(executor)
    assert result.success
    assert list(result.data.index) == list(source.columns)
    assert list(result.data.columns) == list(source.index)
    # Pandas 3/Polars may store identical labels as StringDtype, not object.
    pd.testing.assert_frame_equal(
        result.data.T,
        source,
        check_dtype=False,
        check_names=False,
        check_index_type=False,
    )
    assert result.data.attrs["source_sample_id"] == {A: A, B: B}
    assert all(
        type(gsm) is str for gsm in result.data.attrs["source_sample_id"].values()
    )
    downloader.download_file.assert_called_once()


@pytest.mark.parametrize(
    "url",
    [
        "file:///synthetic_gene_counts.csv",
        "https://example.test/synthetic_gene_counts.zip",
        "https://synthetic_user@example.test/synthetic_gene_counts.csv",
    ],
)
def test_unsupported_url_never_reaches_download_boundary(tmp_path, url):
    service = SimpleNamespace(
        cache_dir=tmp_path, geo_downloader=MagicMock(), geo_parser=MagicMock()
    )
    processor = ArchiveProcessor(service)
    with pytest.raises(ValueError):
        processor._download_and_parse_result(url, GSE)
    service.geo_downloader.download_file.assert_not_called()
    executor, stub = _executor([url])
    with pytest.raises(ImportRequirementNotMet):
        _required(executor)
    stub._download_and_parse_result.assert_not_called()


def test_series_and_url_names_cannot_inject_cache_paths(tmp_path):
    service = SimpleNamespace(
        cache_dir=tmp_path, geo_downloader=MagicMock(), geo_parser=MagicMock()
    )
    service.geo_downloader.download_file.return_value = True
    service.geo_parser.parse_required_expression_file.return_value = ParseResult(
        _table()
    )
    processor = ArchiveProcessor(service)
    processor._download_and_parse_result(
        BASE + "synthetic_%2F_gene_counts.csv", "GSE../../synthetic"
    )
    destination = service.geo_downloader.download_file.call_args.args[1]
    assert destination.parent == tmp_path and len(destination.name) < 255


def test_sibling_does_not_fabricate_complete_result_or_swallow_errors(tmp_path):
    service = SimpleNamespace(
        cache_dir=tmp_path, geo_downloader=MagicMock(), geo_parser=MagicMock()
    )
    processor = ArchiveProcessor(service)
    url = BASE + "synthetic_gene_counts.csv"
    service.geo_downloader.download_file.return_value = False
    assert processor._download_and_parse_result(url, GSE) is None
    service.geo_parser.parse_required_expression_file.assert_not_called()
    service.geo_downloader.download_file.return_value = True
    service.geo_parser.parse_required_expression_file.return_value = _table()
    assert processor._download_and_parse_result(url, GSE) is None
    error = RuntimeError("synthetic containment refusal")
    service.geo_downloader.download_file.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        processor._download_and_parse_result(url, GSE)
    assert caught.value is error
