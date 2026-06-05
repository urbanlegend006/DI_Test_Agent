import pandas as pd
from datetime import datetime
from utils.severity_classifier import classify_mismatch


def test_classify_mismatch_empty():
    assert classify_mismatch("col", "value", "") == "Critical"
    assert classify_mismatch("col", None, "value") == "Critical"


def test_classify_mismatch_text_info():
    assert classify_mismatch("col", "Approved", "APPROVED") == "Info"
    assert classify_mismatch("col", "N-1 Bioreactor", "N_1 Bioreactor") == "Info"
    assert classify_mismatch("col", "hello-world", "helloworld ") == "Info"


def test_classify_mismatch_numeric():
    assert classify_mismatch("price", 1.20, 1.25) == "Critical"
    assert classify_mismatch("value", 100, 100.5) == "Warning"
    custom_tol = {"numeric": 0.10}
    assert classify_mismatch("price", 1.20, 1.25, custom_tol) == "Warning"


def test_classify_mismatch_date():
    assert classify_mismatch("date", "2026-06-01", "2026-06-03") == "Critical"
    assert classify_mismatch("time", "2026-06-01 08:00:00", "2026-06-01 08:05:00") == "Warning"
    assert classify_mismatch("time", "2026-06-01 08:00:00", "2026-06-01 08:00:10") == "Info"
    custom_tol = {"date_seconds": 600}
    assert classify_mismatch("time", "2026-06-01 08:00:00", "2026-06-01 08:05:00", custom_tol) == "Info"


# ---------------------------------------------------------------------------
# Edge case tests (Iteration 2)
# ---------------------------------------------------------------------------

def test_classify_mismatch_nan_values():
    """NaN should not crash — classify as Critical or Info."""
    result = classify_mismatch("col", float("nan"), "text")
    assert result in ("Critical", "Info")
    result_both = classify_mismatch("col", float("nan"), float("nan"))
    assert result_both in ("Critical", "Info")


def test_classify_mismatch_inf_values():
    """Infinity should not cause division errors."""
    result = classify_mismatch("col", float("inf"), 1e308)
    assert isinstance(result, str)


def test_classify_mismatch_bool_values():
    """Booleans (subclass of int) should be handled gracefully."""
    result = classify_mismatch("col", True, 1)
    assert isinstance(result, str)
    result_false = classify_mismatch("col", False, 0)
    assert isinstance(result_false, str)


def test_classify_mismatch_none_vs_empty():
    """Both None or both empty should not crash."""
    result = classify_mismatch("col", None, "")
    assert isinstance(result, str)
    result_both = classify_mismatch("col", "", "")
    assert isinstance(result_both, str)


def test_classify_mismatch_none_tolerance():
    """None tolerance_settings should behave identically to no tolerance."""
    result_none = classify_mismatch("price", 1.20, 1.25, None)
    result_default = classify_mismatch("price", 1.20, 1.25)
    assert result_none == result_default


def test_classify_mismatch_datetime_objects():
    """datetime objects should be comparable to date strings."""
    dt1 = datetime(2026, 6, 1, 8, 0, 0)
    dt2 = datetime(2026, 6, 1, 8, 0, 10)
    assert classify_mismatch("col", dt1, dt2) == "Info"
    dt3 = datetime(2026, 6, 3, 8, 0, 0)
    assert classify_mismatch("col", dt1, dt3) == "Critical"


def test_classify_mismatch_pd_timestamp():
    """pd.Timestamp values should be handled correctly."""
    ts1 = pd.Timestamp("2026-06-01 08:00:00")
    ts2 = pd.Timestamp("2026-06-01 08:00:05")
    assert classify_mismatch("col", ts1, ts2) in ("Info", "Warning")
