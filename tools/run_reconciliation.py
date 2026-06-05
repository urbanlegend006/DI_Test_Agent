import logging
import json
import difflib
from datetime import datetime
import pandas as pd
import numpy as np
from langchain_core.tools import tool

from tools import SESSION_STATE
from utils.severity_classifier import classify_mismatch

logger = logging.getLogger("reconciliation_agent.tools.run_reconciliation")

def sanitize_value(val):
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val

def sanitize_dict(d: dict) -> dict:
    if not isinstance(d, dict):
        return d
    return {k: sanitize_value(v) for k, v in d.items()}

def get_char_diff_html(val1, val2) -> str:
    """Generate HTML character-level diff using span tags for del/add."""
    s1 = str(val1)
    s2 = str(val2)
    
    # If one of them is empty, just wrap the whole thing
    if not s1:
        return f'<span class="diff-add">{s2}</span>'
    if not s2:
        return f'<span class="diff-del">{s1}</span>'
        
    diff = difflib.ndiff(s1, s2)
    html = []
    for char in diff:
        if len(char) < 2:
            continue
        code = char[0]
        val = char[2:]
        # Escape HTML special chars in values
        val_esc = val.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        if code == '-':
            html.append(f'<span class="diff-del">{val_esc}</span>')
        elif code == '+':
            html.append(f'<span class="diff-add">{val_esc}</span>')
        elif code == ' ':
            html.append(val_esc)
            
    return "".join(html)

@tool
def run_reconciliation(primary_key: str, tolerance: str = "strict") -> str:
    """Performs row-by-row reconciliation between Source and Target DataFrames.
    
    Args:
        primary_key: The column name or comma-separated list of column names for the primary key.
        tolerance: "strict" for exact match or a JSON string (e.g. '{"numeric": 0.05, "date_seconds": 3600}') for custom thresholds.
        
    Returns:
        A text summary showing total rows, matched, mismatched, and missing rows.
    """
    logger.info("Invoking run_reconciliation with primary_key='%s', tolerance='%s'", primary_key, tolerance)
    from rich.console import Console
    from rich.panel import Panel
    Console().print(Panel(
        f"[yellow]Primary Key(s): {primary_key}\nTolerance Settings: {tolerance}[/yellow]",
        title="[bold yellow]Tool Call: run_reconciliation[/bold yellow]",
        border_style="yellow"
    ))
    
    # Check if files have been loaded first
    if 'comp_source' not in SESSION_STATE or 'comp_target' not in SESSION_STATE:
        return "Error: Source and Target data have not been loaded yet. Please call analyze_files first."
        
    comp_source = SESSION_STATE['comp_source']
    comp_target = SESSION_STATE['comp_target']
    source_df = SESSION_STATE['source_df']
    target_df = SESSION_STATE['target_df']
    align_meta = SESSION_STATE['align_meta']
    
    # Parse primary key columns
    key_cols = [k.strip() for k in primary_key.split(",") if k.strip()]
    for key in key_cols:
        if key not in comp_source.columns:
            return f"Error: Key column '{key}' not found in the aligned columns. Available columns: {list(comp_source.columns)}"
            
    # Parse tolerance settings
    tolerance_dict = None
    if tolerance.strip().lower() != "strict":
        try:
            tolerance_dict = json.loads(tolerance)
            logger.info("Parsed custom tolerance configurations: %s", tolerance_dict)
        except Exception as e:
            logger.warning("Failed to parse tolerance JSON. Defaulting to strict: %s", e)
            
    # Set key columns in session state
    SESSION_STATE['primary_key_cols'] = key_cols
    SESSION_STATE['tolerance_settings'] = tolerance_dict
    
    try:
        # Helper to generate unique index keys for mapping rows
        def normalize_single_key(val) -> str:
            if pd.isna(val) or val is None:
                return ""
            val_str = str(val).strip()
            # Handle float representation of integer (e.g. '1.0')
            if val_str.endswith(".0"):
                val_str = val_str[:-2]
            # Strip leading zeros for purely numeric keys to align format differences
            if val_str.isdigit():
                val_str = str(int(val_str))
            return val_str

        def generate_index_key(df: pd.DataFrame, keys: list[str]) -> pd.Series:
            key_df = df[keys].copy()
            for col in keys:
                key_df[col] = key_df[col].apply(normalize_single_key)
            return key_df.agg('|'.join, axis=1)
            
        src_keys = generate_index_key(comp_source, key_cols)
        tgt_keys = generate_index_key(comp_target, key_cols)
        
        # Build raw keys (without formatting normalization) to map back to original display format
        def generate_raw_index_key(df: pd.DataFrame, keys: list[str]) -> pd.Series:
            return df[keys].astype(str).agg('|'.join, axis=1)
            
        src_raw_keys = generate_raw_index_key(comp_source, key_cols)
        tgt_raw_keys = generate_raw_index_key(comp_target, key_cols)
        
        # Build mappings from normalized keys to original keys
        norm_to_orig_src = {}
        for norm_k, raw_k in zip(src_keys, src_raw_keys):
            norm_to_orig_src[norm_k] = raw_k
            
        norm_to_orig_tgt = {}
        for norm_k, raw_k in zip(tgt_keys, tgt_raw_keys):
            norm_to_orig_tgt[norm_k] = raw_k
            
        def get_display_key(norm_key):
            src_k = norm_to_orig_src.get(norm_key)
            tgt_k = norm_to_orig_tgt.get(norm_key)
            if src_k and tgt_k:
                # Prefer target key if it is longer (has more formatting like leading zeros)
                return tgt_k if len(tgt_k) > len(src_k) else src_k
            return src_k or tgt_k or norm_key
        
        # 1. Duplicate detection
        src_dup_mask = src_keys.duplicated(keep=False)
        tgt_dup_mask = tgt_keys.duplicated(keep=False)
        
        src_duplicates = source_df[src_dup_mask].copy()
        tgt_duplicates = target_df[tgt_dup_mask].copy()
        
        # Add index string for tracking
        src_duplicates['_row_key'] = src_keys[src_dup_mask].map(get_display_key)
        tgt_duplicates['_row_key'] = tgt_keys[tgt_dup_mask].map(get_display_key)
        
        # Clean duplicates from comparison
        comp_src_clean = comp_source[~src_dup_mask].copy()
        comp_tgt_clean = comp_target[~tgt_dup_mask].copy()
        
        src_keys_clean = src_keys[~src_dup_mask]
        tgt_keys_clean = tgt_keys[~tgt_dup_mask]
        
        comp_src_clean['_row_key'] = src_keys_clean
        comp_tgt_clean['_row_key'] = tgt_keys_clean
        
        # Set indices to row_key (normalized key) for comparison
        comp_src_clean.set_index('_row_key', inplace=True)
        comp_tgt_clean.set_index('_row_key', inplace=True)
        
        # 2. Missing Rows
        src_keys_set = set(src_keys_clean)
        tgt_keys_set = set(tgt_keys_clean)
        
        missing_in_target_keys = src_keys_set - tgt_keys_set
        missing_in_source_keys = tgt_keys_set - src_keys_set
        common_keys_set = src_keys_set & tgt_keys_set
        
        # Extract missing row data from original dfs (not aligned ones) for detailed reporting
        orig_src_clean = source_df[~src_dup_mask].copy()
        orig_tgt_clean = target_df[~tgt_dup_mask].copy()
        orig_src_clean['_row_key'] = src_keys_clean
        orig_tgt_clean['_row_key'] = tgt_keys_clean
        orig_src_clean.set_index('_row_key', inplace=True)
        orig_tgt_clean.set_index('_row_key', inplace=True)
        
        missing_in_target = []
        for k in sorted(missing_in_target_keys):
            row_data = sanitize_dict(orig_src_clean.loc[k].to_dict())
            missing_in_target.append({"row_key": get_display_key(k), "row_data": row_data})
            
        missing_in_source = []
        for k in sorted(missing_in_source_keys):
            row_data = sanitize_dict(orig_tgt_clean.loc[k].to_dict())
            missing_in_source.append({"row_key": get_display_key(k), "row_data": row_data})
            
        # 3. Value comparison
        mismatches = []
        matched_count = 0
        mismatched_count = 0
        
        # Track column-level mismatch distribution
        col_mismatch_stats = {col: {"total": 0, "Critical": 0, "Warning": 0, "Info": 0} 
                              for col in comp_src_clean.columns if col not in key_cols}
                              
        for key in sorted(common_keys_set):
            src_row = comp_src_clean.loc[key]
            tgt_row = comp_tgt_clean.loc[key]
            
            row_has_mismatch = False
            
            for col in comp_src_clean.columns:
                if col in key_cols:
                    continue
                    
                val_src = src_row[col]
                val_tgt = tgt_row[col]
                
                # Check match status
                is_match = False
                if val_src == val_tgt:
                    is_match = True
                elif isinstance(val_src, (int, float)) and isinstance(val_tgt, (int, float)):
                    if np.isclose(val_src, val_tgt, equal_nan=True):
                        is_match = True
                        
                if not is_match:
                    row_has_mismatch = True
                    sev = classify_mismatch(col, val_src, val_tgt, tolerance_dict)
                    diff_markup = get_char_diff_html(val_src, val_tgt)
                    
                    mismatches.append({
                        "row_key": get_display_key(key),
                        "column": col,
                        "source_value": str(val_src),
                        "target_value": str(val_tgt),
                        "severity": sev,
                        "diff_html": diff_markup
                    })
                    
                    if col in col_mismatch_stats:
                        col_mismatch_stats[col]["total"] += 1
                        col_mismatch_stats[col][sev] += 1
                        
            if row_has_mismatch:
                mismatched_count += 1
            else:
                matched_count += 1
                
        # 4. Compile Results Data Structure
        results = {
            "summary": {
                "total_source_rows": len(source_df),
                "total_target_rows": len(target_df),
                "clean_source_rows": len(comp_src_clean),
                "clean_target_rows": len(comp_tgt_clean),
                "matched_rows": matched_count,
                "mismatched_rows": mismatched_count,
                "missing_in_target": len(missing_in_target),
                "missing_in_source": len(missing_in_source),
                "duplicate_source": len(src_duplicates),
                "duplicate_target": len(tgt_duplicates),
            },
            "mismatches": mismatches,
            "missing_in_target": missing_in_target,
            "missing_in_source": missing_in_source,
            "duplicates_source": [sanitize_dict(r) for r in src_duplicates.to_dict(orient='records')],
            "duplicates_target": [sanitize_dict(r) for r in tgt_duplicates.to_dict(orient='records')],
            "col_mismatch_stats": col_mismatch_stats,
            "primary_keys": key_cols
        }
        
        # Save results in SESSION_STATE
        SESSION_STATE['reconciliation_results'] = results
        
        # Render a text summary for the terminal response
        terminal_summary = (
            f"### Reconciliation Completed Successfully!\n\n"
            f"| Metric | Count |\n"
            f"| :--- | :---: |\n"
            f"| **Total Rows (Source)** | {results['summary']['total_source_rows']} |\n"
            f"| **Total Rows (Target)** | {results['summary']['total_target_rows']} |\n"
            f"| **Fully Matched Rows** | {results['summary']['matched_rows']} |\n"
            f"| **Mismatched Rows** | {results['summary']['mismatched_rows']} |\n"
            f"| **Missing in Target** | {results['summary']['missing_in_target']} |\n"
            f"| **Missing in Source** | {results['summary']['missing_in_source']} |\n"
        )
        
        if results['summary']['duplicate_source'] > 0:
            terminal_summary += f"| **Duplicate Rows (Source)** | {results['summary']['duplicate_source']} |\n"
        if results['summary']['duplicate_target'] > 0:
            terminal_summary += f"| **Duplicate Rows (Target)** | {results['summary']['duplicate_target']} |\n"
            
        # Add severity summary of mismatches
        sev_counts = {"Critical": 0, "Warning": 0, "Info": 0}
        for m in mismatches:
            sev_counts[m["severity"]] += 1
            
        terminal_summary += (
            f"\n**Mismatches Breakdown by Severity:**\n"
            f"  - 🔴 **Critical:** {sev_counts['Critical']}\n"
            f"  - 🟡 **Warning:** {sev_counts['Warning']}\n"
            f"  - 🔵 **Info:** {sev_counts['Info']}\n"
        )
        
        return terminal_summary
        
    except Exception as e:
        logger.exception("Error during reconciliation processing")
        return f"Error executing reconciliation comparison: {str(e)}"
