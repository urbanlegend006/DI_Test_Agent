import pytest
from tools import SESSION_STATE
from tools.analyze_files import analyze_files

def test_analyze_files_invalid_paths():
    res = analyze_files.invoke({"source_path": "invalid_path_1.csv", "target_path": "invalid_path_2.csv"})
    assert "Error" in res

def test_analyze_files_success(temp_files):
    src_path = str(temp_files["source_csv"])
    tgt_path = str(temp_files["target_json"])
    
    # Invoke the analyze_files tool through tool calling interface
    res = analyze_files.invoke({"source_path": src_path, "target_path": tgt_path})
    
    assert "Successfully loaded and analyzed both files!" in res
    assert "source.csv" in res
    assert "target.json" in res
    assert "Aligned/Matching Columns" in res
    assert "Warning" in res
    assert "No primary key" in res
    
    # Verify cached DataFrames exist in global session state
    assert 'source_df' in SESSION_STATE
    assert 'target_df' in SESSION_STATE
    assert 'comp_source' in SESSION_STATE
    assert 'comp_target' in SESSION_STATE
    
    # Correct columns aligned
    comp_src = SESSION_STATE['comp_source']
    assert set(comp_src.columns) == {"id", "name", "price", "updated"}
