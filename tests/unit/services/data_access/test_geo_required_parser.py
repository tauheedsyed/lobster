"""Active-only complete CSV records: eager/chunked readers share strict semantics."""

import gzip
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from lobster.services.data_access.geo.parser import GEOParser

A, B = "GSM900000000001", "GSM900000000002"


@pytest.mark.parametrize("chunked", [False, True], ids=["eager", "chunked"])
@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzip"])
@pytest.mark.parametrize("kind", ["header", "feature", "value"])
def test_nul_any_field_refuses_before_pandas(
    tmp_path, monkeypatch, kind, compressed, chunked
):
    text = f"gene\t{A}\t{B}\nsynthetic_a\t1\t3\nsynthetic_b\t2\t4\n"
    if kind == "header":
        text = text.replace(A, A + "\x00suffix")
    elif kind == "feature":
        text = text.replace("synthetic_a", "synthetic_a\x00suffix")
    else:
        text = text.replace("synthetic_a\t1\t3", "synthetic_a\t1\x00suffix\t3")
    parser = parser_for(monkeypatch, chunked)
    read = MagicMock(wraps=pd.read_csv)
    monkeypatch.setattr(pd, "read_csv", read)
    parsed = parser.parse_required_expression_file(
        write_text(tmp_path, text, compressed)
    )
    assert parsed.data is None and not parsed.is_complete
    read.assert_not_called()


@pytest.mark.parametrize("chunked", [False, True], ids=["eager", "chunked"])
@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzip"])
@pytest.mark.parametrize("layout", ["blank", "named", "r_rows"])
def test_internal_index_alias_never_changes_labels_dtype_or_name(
    tmp_path, monkeypatch, layout, compressed, chunked
):
    header = f"\t0\t{B}\n" if layout == "blank" else f"gene\t0\t{B}\n"
    if layout == "r_rows":
        header = f"0\t{B}\n"
    text = header + "00123\t1\t3\n1e5\t2\t4\nTRUE\t3\t5\nNA\t4\t6\n"
    parsed = parser_for(monkeypatch, chunked).parse_required_expression_file(
        write_text(tmp_path, text, compressed)
    )
    assert parsed.is_complete and parsed.original_tokens_valid is True
    assert list(parsed.data.index) == ["00123", "1e5", "TRUE", "NA"]
    assert all(type(value) is str for value in parsed.data.index)
    assert parsed.data.index.name == ("gene" if layout == "named" else None)
    assert list(parsed.data.columns) == ["0", B]
    assert all(type(value) is str for value in parsed.data.columns)
    assert parsed.data.values.tolist() == [[1, 3], [2, 4], [3, 5], [4, 6]]
    assert all(str(dtype) == "int64" for dtype in parsed.data.dtypes)
    assert parsed.data.attrs == {
        "geo_matrix_contract": {
            "source_format": "tabular",
            "orientation": "unoriented",
            "parser_transpose_applied": False,
        }
    }


def matrix(rows=4):
    return pd.DataFrame(
        np.tile([[1, 3]], (rows, 1)),
        index=[f"synthetic_gene_{i}" for i in range(rows)],
        columns=[A, B],
    )


def write_text(tmp_path, text, compressed):
    path = tmp_path / ("synthetic_gene.tsv" + (".gz" if compressed else ""))
    path.write_bytes(gzip.compress(text.encode()) if compressed else text.encode())
    return path


def parser_for(monkeypatch, chunked):
    parser = GEOParser()
    monkeypatch.setattr(
        parser,
        "check_memory_for_dimensions",
        MagicMock(return_value={"can_load": not chunked}),
    )
    monkeypatch.setattr(parser, "_get_adaptive_chunk_size", lambda *_: 1000)
    monkeypatch.setattr(parser, "_check_memory_availability", lambda *_, **__: True)
    return parser


@pytest.mark.parametrize("chunked", [False, True], ids=["eager", "chunked"])
@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzip"])
@pytest.mark.parametrize(
    "kind",
    [
        "plain",
        "r_row_names",
        "quoted_header",
        "stray_quote",
        "double_quote",
        "multiline_quoted_field",
        "bom",
        "crlf",
        "no_final_newline",
        "trailing_blank",
        "whitespace_blank",
        "leading_blank",
        "hash_feature_not_comment",
    ],
)
def test_strict_reader_preserves_every_label_value_and_integer_dtype(
    tmp_path, monkeypatch, kind, compressed, chunked
):
    expected = matrix()
    if kind == "stray_quote":
        expected.index = ['synthetic_"a', *expected.index[1:]]
    elif kind == "double_quote":
        expected.index = ['synthetic_"a', *expected.index[1:]]
    elif kind == "multiline_quoted_field":
        expected.index = ["synthetic_\na", *expected.index[1:]]
    elif kind == "hash_feature_not_comment":
        expected.index = ["#synthetic_a", *expected.index[1:]]
    text = expected.to_csv(sep="\t")
    if kind == "r_row_names":
        text = text.removeprefix("\t")
    elif kind == "quoted_header":
        text = text.replace(f"\t{A}\t{B}", f'"gene"\t"{A}"\t"{B}"', 1)
    elif kind == "stray_quote":
        text = text.replace('"synthetic_""a"', 'synthetic_"a', 1)
    elif kind == "bom":
        text = "\ufeff" + text
    elif kind == "crlf":
        text = text.replace("\n", "\r\n")
    elif kind == "no_final_newline":
        text = text.rstrip("\n")
    elif kind == "trailing_blank":
        text += "\n\n"
    elif kind == "whitespace_blank":
        text += "   \n"
    elif kind == "leading_blank":
        text = "\n  \n" + text
    path = write_text(tmp_path, text, compressed)
    parser = parser_for(monkeypatch, chunked)
    result = parser.parse_required_expression_file(path)
    assert result.is_complete and not result.is_partial
    assert result.rows_read == len(expected)
    assert list(result.data.index) == list(expected.index)
    assert list(result.data.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(
        result.data, expected, check_names=False, check_index_type=False
    )
    assert all(dtype == np.dtype("int64") for dtype in result.data.dtypes)
    assert result.data.attrs["geo_matrix_contract"] == {
        "source_format": "tabular",
        "orientation": "unoriented",
        "parser_transpose_applied": False,
    }


@pytest.mark.parametrize("chunked", [False, True], ids=["eager", "chunked"])
@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzip"])
@pytest.mark.parametrize(
    "kind",
    [
        "overlong_late",
        "short_late",
        "overlong_first",
        "r_mixed_width",
        "quoted_whitespace_short",
        "unclosed_quote",
        "leading_comment",
    ],
)
def test_invalid_records_past_chunk_boundary_refuse_before_any_matrix_read(
    tmp_path, monkeypatch, kind, compressed, chunked
):
    text = matrix(2001).to_csv(sep="\t")
    if kind == "overlong_late":
        text = text.replace(
            "synthetic_gene_2000\t1\t3", "synthetic_gene_2000\t1\t3\t999"
        )
    elif kind == "short_late":
        text = text.replace("synthetic_gene_2000\t1\t3", "synthetic_gene_2000\t1")
    elif kind == "overlong_first":
        text = text.replace("synthetic_gene_0\t1\t3", "synthetic_gene_0\t1\t3\t999")
    elif kind == "r_mixed_width":
        text = text.removeprefix("\t").replace("synthetic_gene_2000\t1\t3", "1\t3")
    elif kind == "quoted_whitespace_short":
        text += '"   "\n'
    elif kind == "unclosed_quote":
        text = text.replace("synthetic_gene_2000\t1\t3", '"synthetic_gene_2000\t1\t3')
    elif kind == "leading_comment":
        text = "# synthetic comment\n" + text
    path = write_text(tmp_path, text, compressed)
    parser = parser_for(monkeypatch, chunked)
    read = MagicMock(
        side_effect=AssertionError("Matrix read before full structural scan")
    )
    monkeypatch.setattr(pd, "read_csv", read)
    result = parser.parse_required_expression_file(path)
    assert result.data is None and result.is_complete is False
    read.assert_not_called()


def test_corrupt_gzip_eof_refuses_before_matrix_read(tmp_path, monkeypatch):
    path = write_text(tmp_path, matrix().to_csv(sep="\t"), True)
    path.write_bytes(path.read_bytes()[:-8])
    parser = parser_for(monkeypatch, False)
    read = MagicMock(side_effect=AssertionError("Read after corrupt gzip"))
    monkeypatch.setattr(pd, "read_csv", read)
    result = parser.parse_required_expression_file(path)
    assert result.data is None and result.is_complete is False
    read.assert_not_called()


def test_real_chunk_resource_limit_keeps_partial_flag_and_rows_read(
    tmp_path, monkeypatch
):
    expected = matrix(2001)
    path = write_text(tmp_path, expected.to_csv(sep="\t"), False)
    parser = parser_for(monkeypatch, True)
    probe = MagicMock(side_effect=[True, False])
    monkeypatch.setattr(parser, "_check_memory_availability", probe)
    result = parser.parse_required_expression_file(path)
    assert result.is_partial and not result.is_complete
    assert result.rows_read == 1000 and result.data.shape == (1000, 2)
    assert "Memory limit" in result.truncation_reason
    pd.testing.assert_frame_equal(
        result.data, expected.iloc[:1000], check_names=False, check_index_type=False
    )
    assert probe.call_count == 2


def test_eager_memory_error_falls_back_to_same_strict_chunk_reader(
    tmp_path, monkeypatch
):
    expected = matrix(2001)
    path = write_text(tmp_path, expected.to_csv(sep="\t"), False)
    parser = parser_for(monkeypatch, False)
    actual_read = pd.read_csv
    calls = []

    def constrained_read(*args, **kwargs):
        calls.append(kwargs.get("chunksize"))
        if "chunksize" not in kwargs:
            raise MemoryError("synthetic capacity boundary")
        return actual_read(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", constrained_read)
    result = parser.parse_required_expression_file(path)
    assert result.is_complete and result.rows_read == 2001
    assert calls == [None, 1000]
    pd.testing.assert_frame_equal(
        result.data, expected, check_names=False, check_index_type=False
    )


def test_tab_separated_empty_cells_are_not_blank_lines(tmp_path, monkeypatch):
    expected = matrix()
    path = write_text(tmp_path, expected.to_csv(sep="\t") + "\t\t\n", False)
    parser = parser_for(monkeypatch, False)
    result = parser.parse_required_expression_file(path)
    assert result.is_complete and result.rows_read == len(expected) + 1
    assert result.data.iloc[-1].isna().all()
    # Numeric/identity checks remain the chooser's job, not structural parsing.


@pytest.mark.parametrize("chunked", [False, True], ids=["eager", "chunked"])
@pytest.mark.parametrize("r_layout", [False, True], ids=["normal", "r"])
@pytest.mark.parametrize("kind", ["zeros", "exponent", "boolean", "na"])
def test_raw_identifier_tokens_are_exact_strings(
    tmp_path, monkeypatch, kind, r_layout, chunked
):
    ids = {
        "zeros": ["00123", "00124", "00125", "00126"],
        "exponent": ["1e5", "2e5", "3e5", "4e5"],
        "boolean": ["TRUE", "FALSE", "TRUE_gene", "FALSE_gene"],
        "na": ["NA", "NA_gene2", "NA_gene3", "NA_gene4"],
    }[kind]
    frame = matrix()
    frame.index = ids
    text = frame.to_csv(sep="\t")
    if r_layout:
        text = text.removeprefix("\t")
    path = write_text(tmp_path, text, False)
    result = parser_for(monkeypatch, chunked).parse_required_expression_file(path)
    assert result.is_complete
    assert list(result.data.index) == ids
    assert all(type(value) is str for value in result.data.index)
    assert list(result.data.columns) == [A, B]
    np.testing.assert_array_equal(result.data.to_numpy(), frame.to_numpy())


@pytest.mark.parametrize("chunked", [False, True], ids=["eager", "chunked"])
@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzip"])
def test_raw_duplicate_header_refuses_before_pandas_can_mangle(
    tmp_path, monkeypatch, compressed, chunked
):
    text = (
        f"sample\tsynthetic_a\tsynthetic_a\tsynthetic_b\n{A}\t1\t2\t3\n{B}\t4\t5\t6\n"
    )
    path = write_text(tmp_path, text, compressed)
    read = MagicMock(side_effect=AssertionError("Duplicate raw header reached pandas"))
    monkeypatch.setattr(pd, "read_csv", read)
    result = parser_for(monkeypatch, chunked).parse_required_expression_file(path)
    assert not result.is_complete and result.data is None
    read.assert_not_called()


@pytest.mark.parametrize(
    "token, valid",
    [
        ("1.0000000000000000001", False),
        ("0.99999999999999999999", False),
        ("2.0", True),
        ("1e3", True),
        ("NaN", True),
        ("", True),
        ("-1", False),
        ("inf", False),
        ("TRUE", False),
    ],
)
def test_original_token_proof_never_accepts_rounded_fraction(
    tmp_path, monkeypatch, token, valid
):
    text = (
        matrix()
        .to_csv(sep="\t")
        .replace("synthetic_gene_0\t1\t3", f"synthetic_gene_0\t{token}\t3")
    )
    path = write_text(tmp_path, text, False)
    result = parser_for(monkeypatch, False).parse_required_expression_file(path)
    assert result.is_complete
    assert result.original_tokens_valid is valid
    assert "original_tokens_valid" not in result.data.attrs


def test_padded_multi_space_records_refuse_without_column_shifting(
    tmp_path, monkeypatch
):
    text = f"gene {A} {B}\nsynthetic_a  1  3\nsynthetic_b  2  4\n"
    path = write_text(tmp_path, text, False)
    result = parser_for(monkeypatch, False).parse_required_expression_file(path)
    assert not result.is_complete and result.data is None


def test_programming_error_is_not_suppressed(tmp_path, monkeypatch):
    path = write_text(tmp_path, matrix().to_csv(sep="\t"), False)
    parser = parser_for(monkeypatch, False)
    monkeypatch.setattr(
        parser,
        "check_memory_for_dimensions",
        MagicMock(side_effect=RuntimeError("synthetic")),
    )
    with pytest.raises(RuntimeError):
        parser.parse_required_expression_file(path)
