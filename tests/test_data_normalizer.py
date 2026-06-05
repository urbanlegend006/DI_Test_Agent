import pandas as pd
import numpy as np
from utils.data_normalizer import clean_value, try_parse_datetime, normalize_dataframe, align_columns

def test_clean_value():
    assert clean_value(" hello ") == "hello"
    assert clean_value(np.nan) == ""
    assert clean_value(None) == ""
    assert clean_value(123) == 123

def test_try_parse_datetime():
    val = "2026-06-01 08:30:00"
    parsed = try_parse_datetime(val)
    assert isinstance(parsed, pd.Timestamp)
    assert parsed.hour == 8
    
    # Non-date strings should be returned as-is
    assert try_parse_datetime("not a date") == "not a date"
    assert try_parse_datetime(123) == 123

def test_normalize_dataframe():
    df = pd.DataFrame([
        {"id": " 001 ", "name": "Apple ", "value": "1.23", "date": "2026-06-01"},
        {"id": "002", "name": "Banana", "value": None, "date": "invalid-date"}
    ])
    
    norm_df = normalize_dataframe(df)
    
    assert norm_df.loc[0, "id"] == "001"
    assert norm_df.loc[0, "name"] == "Apple"
    assert norm_df.loc[0, "value"] == 1.23
    assert isinstance(norm_df.loc[0, "date"], pd.Timestamp)
    
    assert norm_df.loc[1, "value"] == ""
    assert norm_df.loc[1, "date"] == ""


def test_align_columns():
    src_df = pd.DataFrame(columns=["id", "item_name", "extra_src"])
    tgt_df = pd.DataFrame(columns=["ID", "item name", "extra_tgt"])
    
    comp_src, comp_tgt, meta = align_columns(src_df, tgt_df)
    
    # Columns in comp should be aligned and matching
    assert list(comp_src.columns) == ["id", "item_name"]
    assert list(comp_tgt.columns) == ["id", "item_name"]
    
    assert "extra_src" in meta["extra_in_source"]
    assert "extra_tgt" in meta["extra_in_target"]
    assert "item_name" in meta["aligned_columns"]
