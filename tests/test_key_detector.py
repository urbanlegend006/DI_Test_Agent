import pandas as pd
from utils.key_detector import is_column_unique, detect_primary_key

def test_is_column_unique():
    df = pd.DataFrame([{"id": "01"}, {"id": "02"}, {"id": "01"}])
    assert is_column_unique(df, "id") is False
    
    df_uniq = pd.DataFrame([{"id": "01"}, {"id": "02"}, {"id": "03"}])
    assert is_column_unique(df_uniq, "id") is True

def test_detect_primary_key_single():
    # ID column indicator should be picked
    df = pd.DataFrame([
        {"id": "001", "name": "Apple", "value": 1},
        {"id": "002", "name": "Banana", "value": 2}
    ])
    assert detect_primary_key(df) == ["id"]
    
    # Capitalized PK candidate
    df_cap = pd.DataFrame([
        {"PK": "001", "name": "Apple"},
        {"PK": "002", "name": "Banana"}
    ])
    assert detect_primary_key(df_cap) == ["PK"]

def test_detect_primary_key_composite():
    # Neither col1 nor col2 is unique individually, but combined they are unique
    df = pd.DataFrame([
        {"batch": "A", "param": "timer", "value": 10},
        {"batch": "A", "param": "temp", "value": 20},
        {"batch": "B", "param": "timer", "value": 15},
        {"batch": "B", "param": "temp", "value": 25}
    ])
    
    # Unique composite key containing batch and param indicators should be found
    keys = detect_primary_key(df)
    assert keys is not None
    assert set(keys) == {"batch", "param"}

def test_detect_primary_key_none():
    # Entirely duplicate records or non-unique columns
    df = pd.DataFrame([
        {"name": "Apple", "value": 1},
        {"name": "Apple", "value": 1}
    ])
    assert detect_primary_key(df) is None
