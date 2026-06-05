"""Tests for the user-input preprocessing module (path extraction + guardrails)."""
import pytest
from utils.preprocessing import extract_paths, is_reconciliation_related, POLITE_DECLINE


# ---------- Path extraction ----------

def test_extract_quoted_paths():
    """Quoted file paths should be extracted."""
    text = 'Please reconcile "C:\\data\\source.xlsx" vs "target.csv"'
    paths = extract_paths(text)
    assert len(paths) == 2
    assert "C:\\data\\source.xlsx" in paths
    assert "target.csv" in paths


def test_extract_single_quoted_paths():
    """Single-quoted file paths should be extracted."""
    text = "compare 'source.xlsx' and 'target.json'"
    paths = extract_paths(text)
    assert len(paths) == 2
    assert "source.xlsx" in paths
    assert "target.json" in paths


def test_extract_backtick_paths():
    """Backtick-quoted paths should be extracted."""
    text = "reconcile `data/source.xlsx` with `data/target.csv`"
    paths = extract_paths(text)
    assert len(paths) == 2
    assert "data/source.xlsx" in paths
    assert "data/target.csv" in paths


def test_extract_file_uris():
    """file:// URIs should be extracted."""
    text = "check file:///home/user/data/source.xlsx vs file:///home/user/data/target.csv"
    paths = extract_paths(text)
    assert len(paths) == 2
    assert "file:///home/user/data/source.xlsx" in paths
    assert "file:///home/user/data/target.csv" in paths


def test_extract_bare_paths():
    """Unquoted paths ending in valid extensions should be extracted."""
    text = "compare data/source.xlsx with target.csv and also check file.json"
    paths = extract_paths(text)
    assert len(paths) >= 2
    assert "data/source.xlsx" in paths
    assert "target.csv" in paths


def test_extract_no_paths():
    """No file paths should return an empty list."""
    assert extract_paths("help me reconcile some files") == []
    assert extract_paths("what is the weather?") == []
    assert extract_paths("") == []


def test_extract_one_path():
    """A single file path should be extracted."""
    paths = extract_paths('check "source.xlsx" only')
    assert len(paths) == 1
    assert "source.xlsx" in paths


def test_extract_skips_unsupported_extensions():
    """Paths ending in unsupported extensions should be ignored."""
    text = 'check "readme.pdf" and "data.docx" and "source.xlsx"'
    paths = extract_paths(text)
    assert len(paths) == 1
    assert "source.xlsx" in paths


def test_extract_deduplicates():
    """Duplicate paths should appear only once, preserving first-seen order."""
    text = 'compare "file.xlsx" with "file.xlsx" again'
    paths = extract_paths(text)
    assert len(paths) == 1


def test_extract_parquet():
    """.parquet files should be extracted like any other valid extension."""
    paths = extract_paths('load "data.parquet"')
    assert len(paths) == 1
    assert "data.parquet" in paths


def test_extract_txt_extension():
    """.txt files should be extracted when explicitly referenced."""
    paths = extract_paths('read "data.txt" and "source.csv"')
    assert len(paths) == 2
    assert "data.txt" in paths
    assert "source.csv" in paths


# ---------- Guardrails ----------

def test_is_reconciliation_related_positive():
    """Messages with reconciliation keywords should be classified as related."""
    assert is_reconciliation_related("reconcile these files")
    assert is_reconciliation_related("compare source.xlsx and target.csv")
    assert is_reconciliation_related("find differences between two files")
    assert is_reconciliation_related("match data from csv files")
    assert is_reconciliation_related("I need to analyze some JSON files")
    assert is_reconciliation_related("run a report on my data")


def test_is_reconciliation_related_negative():
    """Off-topic messages should be classified as unrelated."""
    assert not is_reconciliation_related("what is the weather today?")
    assert not is_reconciliation_related("write a poem about data")
    assert not is_reconciliation_related("cook me some pasta")
    assert not is_reconciliation_related("what color is the sky?")
    assert not is_reconciliation_related("write a python script")


def test_is_reconciliation_related_ambiguous():
    """Ambiguous input should pass through (return True)."""
    assert is_reconciliation_related("hello")
    assert is_reconciliation_related("thanks")
    assert is_reconciliation_related("")


def test_is_reconciliation_related_mixed():
    """Mixed reconciliation + off-topic keywords should be treated as related."""
    assert is_reconciliation_related("reconcile weather data files")
    assert is_reconciliation_related("compare poem.csv and song.xlsx")


def test_polite_decline_constant():
    """POLITE_DECLINE should be a non-empty string mentioning 'Data Integrity'."""
    assert isinstance(POLITE_DECLINE, str)
    assert len(POLITE_DECLINE) > 20
    assert "Data Integrity" in POLITE_DECLINE


def test_cli_preprocess_returns_structured_paths():
    """CLI preprocessing should pass paths as structured fields."""
    from main import _preprocess_input

    payload = _preprocess_input('compare "source.csv" with "target.json"')
    assert payload["input"] == 'compare "source.csv" with "target.json"'
    assert payload["source_path"] == "source.csv"
    assert payload["target_path"] == "target.json"
    assert payload["report_format"] == "html"


def test_cli_preprocess_detects_excel_report_request():
    """Excel report requests should be carried as structured report format."""
    from main import _preprocess_input

    payload = _preprocess_input('compare "source.csv" with "target.json" and generate excel')
    assert payload["report_format"] == "excel"
