from utils.severity_classifier import classify_mismatch

def test_classify_mismatch_empty():
    # If one is empty and the other is not, it must be Critical
    assert classify_mismatch("col", "value", "") == "Critical"
    assert classify_mismatch("col", None, "value") == "Critical"


def test_classify_mismatch_text_info():
    # Case differences only -> Info
    assert classify_mismatch("col", "Approved", "APPROVED") == "Info"
    
    # Format differences like whitespace, dashes, underscores -> Info
    assert classify_mismatch("col", "N-1 Bioreactor", "N_1 Bioreactor") == "Info"
    assert classify_mismatch("col", "hello-world", "helloworld ") == "Info"

def test_classify_mismatch_numeric():
    # Strict tolerances (default is 1% relative)
    # 1.20 vs 1.25 is ~4.16% difference, which is above 1% -> Critical
    assert classify_mismatch("price", 1.20, 1.25) == "Critical"
    
    # 100 vs 100.5 is 0.5% difference, below 1% -> Warning
    assert classify_mismatch("value", 100, 100.5) == "Warning"
    
    # Custom tolerance settings
    custom_tol = {"numeric": 0.10} # 10% relative tolerance
    # 1.20 vs 1.25 is 4.16% which is below 10% -> Warning
    assert classify_mismatch("price", 1.20, 1.25, custom_tol) == "Warning"

def test_classify_mismatch_date():
    # Date differences of more than 1 day -> Critical
    assert classify_mismatch("date", "2026-06-01", "2026-06-03") == "Critical"
    
    # Date differences below 1 day but above date_seconds (default 60s) -> Warning
    # Off by 5 minutes
    assert classify_mismatch("time", "2026-06-01 08:00:00", "2026-06-01 08:05:00") == "Warning"
    
    # Date differences below date_seconds (default 60s) -> Info
    # Off by 10 seconds
    assert classify_mismatch("time", "2026-06-01 08:00:00", "2026-06-01 08:00:10") == "Info"
    
    # Custom date tolerance
    custom_tol = {"date_seconds": 600} # 10 mins
    assert classify_mismatch("time", "2026-06-01 08:00:00", "2026-06-01 08:05:00", custom_tol) == "Info"
