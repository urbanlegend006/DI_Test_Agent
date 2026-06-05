import pytest
import pandas as pd
import json
from pathlib import Path
from tools import SESSION_STATE

@pytest.fixture(autouse=True)
def clean_session_state():
    """Fixture to reset the global session state before and after each test."""
    SESSION_STATE.reset()
    yield
    SESSION_STATE.reset()

@pytest.fixture
def mock_source_df():
    return pd.DataFrame([
        {"id": "001", "name": "Apple", "price": 1.20, "updated": "2026-06-01 00:00:00"},
        {"id": "002", "name": "Banana", "price": 0.80, "updated": "2026-06-01 00:00:00"},
        {"id": "003", "name": "Cherry", "price": 2.50, "updated": "2026-06-02 12:00:00"},
        {"id": "005", "name": "Duplicate", "price": 9.99, "updated": "2026-06-05 08:00:00"},
        {"id": "005", "name": "Duplicate", "price": 9.99, "updated": "2026-06-05 08:00:00"}
    ])

@pytest.fixture
def mock_target_df():
    return pd.DataFrame([
        {"id": "001", "name": "Apple", "price": 1.25, "updated": "2026-06-01 00:00:00"},  # price mismatch
        {"id": "002", "name": "Banana", "price": 0.80, "updated": "2026-06-01 00:00:00"},  # match
        {"id": "004", "name": "Dates", "price": 3.00, "updated": "2026-06-03 00:00:00"},   # missing in source
        {"id": "006", "name": "DuplicateTarget", "price": 4.44, "updated": "2026-06-06 00:00:00"},
        {"id": "006", "name": "DuplicateTarget", "price": 4.44, "updated": "2026-06-06 00:00:00"} # duplicate in target
    ])

@pytest.fixture
def temp_files(tmp_path, mock_source_df, mock_target_df):
    """Fixture to create temporary files for testing ingestion tools."""
    src_csv = tmp_path / "source.csv"
    tgt_json = tmp_path / "target.json"
    
    mock_source_df.to_csv(src_csv, index=False)
    
    # Save target as a JSON list of records
    target_records = mock_target_df.to_dict(orient="records")
    with open(tgt_json, "w", encoding="utf-8") as f:
        json.dump(target_records, f, indent=2)
        
    return {
        "source_csv": src_csv,
        "target_json": tgt_json
    }
