import json
import pytest
from tools import SESSION_STATE
from tools.run_reconciliation import run_reconciliation, get_char_diff_html
from utils.data_normalizer import align_columns

def test_get_char_diff_html():
    diff1 = get_char_diff_html("Apple", "Abple")
    assert '<span class="diff-del">p</span>' in diff1
    assert '<span class="diff-add">b</span>' in diff1
    
    diff2 = get_char_diff_html("", "Apple")
    assert '<span class="diff-add">Apple</span>' in diff2
    
    diff3 = get_char_diff_html("Apple", "")
    assert '<span class="diff-del">Apple</span>' in diff3

def test_run_reconciliation_missing_state():
    # Calling reconciliation without loaded data
    res = run_reconciliation.func("id")
    assert "Error" in res

def test_run_reconciliation_success(mock_source_df, mock_target_df):
    # Align and cache the mock dataframes in state
    comp_src, comp_tgt, meta = align_columns(mock_source_df, mock_target_df)
    SESSION_STATE['comp_source'] = comp_src
    SESSION_STATE['comp_target'] = comp_tgt
    SESSION_STATE['source_df'] = mock_source_df
    SESSION_STATE['target_df'] = mock_target_df
    SESSION_STATE['align_meta'] = meta
    
    summary_res = run_reconciliation.func("id")
    
    # Check text output details
    assert "Reconciliation Completed Successfully!" in summary_res
    assert "Mismatched Rows" in summary_res
    
    # Check cached results state
    results = SESSION_STATE['reconciliation_results']
    assert results is not None
    assert results['summary']['total_source_rows'] == 5
    assert results['summary']['total_target_rows'] == 5
    assert results['summary']['matched_rows'] == 1 # 002 (Banana) matches exactly
    assert results['summary']['mismatched_rows'] == 1 # 001 (Apple) has price mismatch
    assert results['summary']['missing_in_target'] == 1 # 003 (Cherry) is in source only
    assert results['summary']['missing_in_source'] == 1 # 004 (Dates) is in target only
    assert results['summary']['duplicate_source'] == 2 # 005 duplicate row count
    assert results['summary']['duplicate_target'] == 2 # 006 duplicate row count
    
    # Check mismatches structure
    assert len(results['mismatches']) == 1
    mismatch = results['mismatches'][0]
    assert mismatch['row_key'] == "001"
    assert mismatch['column'] == "price"
    assert mismatch['severity'] == "Critical" # 1.20 vs 1.25 exceeds 1% relative tol
