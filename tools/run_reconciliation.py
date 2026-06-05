import logging
import json
import difflib
from datetime import datetime
from typing import Any
import pandas as pd
import numpy as np
from langchain_core.tools import tool

from tools import SESSION_STATE
from utils.severity_classifier import classify_mismatch

logger = logging.getLogger("reconciliation_agent.tools.run_reconciliation")


# ---------------------------------------------------------------------------
# Value sanitization helpers
# ---------------------------------------------------------------------------

def sanitize_value(val: Any) -> Any:
    """Normalize a single value for JSON-safe serialisation.

    Converts NumPy/Pandas types to native Python types and handles ``NaN``,
    ``Timestamp``, ``datetime``, and ``ndarray``.

    Args:
        val: Raw value from a DataFrame cell.

    Returns:
        A JSON-serialisable Python scalar or ``None`` for missing values.
    """
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
    """Normalise all values in a dictionary via ``sanitize_value``.

    Args:
        d: Input dictionary whose values may contain Pandas/NumPy types.

    Returns:
        New dictionary with all values converted to JSON-safe types.
    """
    if not isinstance(d, dict):
        return d
    return {k: sanitize_value(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# HTML character-level diff
# ---------------------------------------------------------------------------

def get_char_diff_html(val1: Any, val2: Any) -> str:
    """Generate HTML character-level diff using span tags for del/add."""
    s1 = str(val1)
    s2 = str(val2)

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
        val_esc = val.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        if code == '-':
            html.append(f'<span class="diff-del">{val_esc}</span>')
        elif code == '+':
            html.append(f'<span class="diff-add">{val_esc}</span>')
        elif code == ' ':
            html.append(val_esc)

    return "".join(html)


# ---------------------------------------------------------------------------
# Key indexing helpers
# ---------------------------------------------------------------------------

def _normalize_single_key(val: Any) -> str:
    if pd.isna(val) or val is None:
        return ""
    val_str = str(val).strip()
    if val_str.endswith(".0"):
        val_str = val_str[:-2]
    if val_str.isdigit():
        val_str = str(int(val_str))
    return val_str


def _generate_index_key(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    parts = []
    for col in keys:
        normalized = df[col].map(_normalize_single_key)
        parts.append(normalized.replace("", "\x00"))
    combined = parts[0]
    for p in parts[1:]:
        combined = combined + "|" + p
    return combined.str.replace("\x00", "")


def _generate_raw_index_key(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    combined = df[keys[0]].astype(str)
    for c in keys[1:]:
        combined = combined + "|" + df[c].astype(str)
    return combined


def _build_key_mappings(
    src_keys: pd.Series, src_raw_keys: pd.Series,
    tgt_keys: pd.Series, tgt_raw_keys: pd.Series,
) -> callable:
    """Build normalized-to-original key mappings and a display-key resolver."""
    norm_to_orig_src = dict(zip(src_keys, src_raw_keys, strict=True))
    norm_to_orig_tgt = dict(zip(tgt_keys, tgt_raw_keys, strict=True))

    def get_display_key(norm_key):
        src_k = norm_to_orig_src.get(norm_key)
        tgt_k = norm_to_orig_tgt.get(norm_key)
        if src_k and tgt_k:
            return tgt_k if len(tgt_k) > len(src_k) else src_k
        return src_k or tgt_k or norm_key

    return get_display_key


def _detect_duplicates(
    src_keys: pd.Series, tgt_keys: pd.Series,
    source_df: pd.DataFrame, target_df: pd.DataFrame,
    comp_source: pd.DataFrame, comp_target: pd.DataFrame,
    get_display_key: callable,
) -> dict:
    """Detect and separate duplicate rows from clean data."""
    src_dup_mask = src_keys.duplicated(keep=False)
    tgt_dup_mask = tgt_keys.duplicated(keep=False)

    src_duplicates = source_df[src_dup_mask].copy()
    tgt_duplicates = target_df[tgt_dup_mask].copy()
    src_duplicates['_row_key'] = src_keys[src_dup_mask].map(get_display_key)
    tgt_duplicates['_row_key'] = tgt_keys[tgt_dup_mask].map(get_display_key)

    comp_src_clean = comp_source[~src_dup_mask].copy()
    comp_tgt_clean = comp_target[~tgt_dup_mask].copy()
    src_keys_clean = src_keys[~src_dup_mask]
    tgt_keys_clean = tgt_keys[~tgt_dup_mask]

    comp_src_clean['_row_key'] = src_keys_clean
    comp_tgt_clean['_row_key'] = tgt_keys_clean
    comp_src_clean.set_index('_row_key', inplace=True)
    comp_tgt_clean.set_index('_row_key', inplace=True)

    return {
        "src_duplicates": src_duplicates,
        "tgt_duplicates": tgt_duplicates,
        "comp_src_clean": comp_src_clean,
        "comp_tgt_clean": comp_tgt_clean,
        "src_keys_clean": src_keys_clean,
        "tgt_keys_clean": tgt_keys_clean,
    }


def _find_missing_rows(
    missing_in_target_keys: set, missing_in_source_keys: set,
    src_keys_clean: pd.Series, tgt_keys_clean: pd.Series,
    source_df: pd.DataFrame, target_df: pd.DataFrame,
    src_dup_mask: pd.Series, tgt_dup_mask: pd.Series,
    get_display_key: callable,
) -> tuple[list[dict], list[dict]]:
    """Find rows missing in target and source."""
    orig_src_clean = source_df[~src_dup_mask].copy()
    orig_tgt_clean = target_df[~tgt_dup_mask].copy()
    orig_src_clean['_row_key'] = src_keys_clean
    orig_tgt_clean['_row_key'] = tgt_keys_clean
    orig_src_clean.set_index('_row_key', inplace=True)
    orig_tgt_clean.set_index('_row_key', inplace=True)

    missing_in_target = []
    missing_keys_sorted_tgt = sorted(missing_in_target_keys)
    if missing_keys_sorted_tgt:
        src_batch = orig_src_clean.loc[missing_keys_sorted_tgt]
        for k, row in src_batch.iterrows():
            row_data = sanitize_dict(dict(row))
            missing_in_target.append({"row_key": get_display_key(k), "row_data": row_data})

    missing_in_source = []
    missing_keys_sorted_src = sorted(missing_in_source_keys)
    if missing_keys_sorted_src:
        tgt_batch = orig_tgt_clean.loc[missing_keys_sorted_src]
        for k, row in tgt_batch.iterrows():
            row_data = sanitize_dict(dict(row))
            missing_in_source.append({"row_key": get_display_key(k), "row_data": row_data})

    return missing_in_target, missing_in_source


def _compare_columns(
    comp_src_clean: pd.DataFrame, comp_tgt_clean: pd.DataFrame,
    common_keys_set: set,
    key_cols: list[str], get_display_key: callable,
    tolerance_dict: dict | None,
) -> tuple[list[dict], int, int, dict]:
    """Vectorized per-column comparison returning mismatches and stats."""
    mismatches = []
    col_mismatch_stats = {
        col: {"total": 0, "Critical": 0, "Warning": 0, "Info": 0}
        for col in comp_src_clean.columns if col not in key_cols
    }
    compare_columns = [c for c in comp_src_clean.columns if c not in key_cols]

    common_keys_list = sorted(common_keys_set)
    comp_src_aligned = comp_src_clean.reindex(common_keys_list)
    comp_tgt_aligned = comp_tgt_clean.reindex(common_keys_list)

    for col in compare_columns:
        src_vals = comp_src_aligned[col]
        tgt_vals = comp_tgt_aligned[col]

        match_mask = (src_vals == tgt_vals).fillna(False)

        if src_vals.dtype.kind in "fiuc" and tgt_vals.dtype.kind in "fiuc":
            with np.errstate(invalid="ignore"):
                numeric_close = np.isclose(
                    src_vals.astype(float), tgt_vals.astype(float),
                    equal_nan=True, rtol=1e-05, atol=1e-08,
                )
            match_mask = match_mask | pd.Series(numeric_close, index=src_vals.index).fillna(False)

        mismatched_keys = match_mask[~match_mask].index.tolist()
        if not mismatched_keys:
            continue

        for key in mismatched_keys:
            sev = classify_mismatch(col, src_vals[key], tgt_vals[key], tolerance_dict)
            diff_markup = get_char_diff_html(src_vals[key], tgt_vals[key])

            mismatches.append({
                "row_key": get_display_key(key),
                "column": col,
                "source_value": str(src_vals[key]),
                "target_value": str(tgt_vals[key]),
                "severity": sev,
                "diff_html": diff_markup,
            })

            if col in col_mismatch_stats:
                col_mismatch_stats[col]["total"] += 1
                col_mismatch_stats[col][sev] += 1

    mismatch_keys_by_column = {}
    for m in mismatches:
        mismatch_keys_by_column.setdefault(m["column"], set()).add(m["row_key"])
    rows_with_mismatch = set().union(*mismatch_keys_by_column.values()) if mismatch_keys_by_column else set()
    mismatched_count = len(rows_with_mismatch)
    matched_count = len(common_keys_set) - mismatched_count

    return mismatches, matched_count, mismatched_count, col_mismatch_stats


def _render_summary(results: dict, mismatches: list[dict]) -> str:
    """Render the reconciliation results as a text summary."""
    summary = results["summary"]
    terminal = (
        f"### Reconciliation Completed Successfully!\n\n"
        f"| Metric | Count |\n"
        f"| :--- | :---: |\n"
        f"| **Total Rows (Source)** | {summary['total_source_rows']} |\n"
        f"| **Total Rows (Target)** | {summary['total_target_rows']} |\n"
        f"| **Fully Matched Rows** | {summary['matched_rows']} |\n"
        f"| **Mismatched Rows** | {summary['mismatched_rows']} |\n"
        f"| **Missing in Target** | {summary['missing_in_target']} |\n"
        f"| **Missing in Source** | {summary['missing_in_source']} |\n"
    )

    if summary['duplicate_source'] > 0:
        terminal += f"| **Duplicate Rows (Source)** | {summary['duplicate_source']} |\n"
    if summary['duplicate_target'] > 0:
        terminal += f"| **Duplicate Rows (Target)** | {summary['duplicate_target']} |\n"

    sev_counts = {"Critical": 0, "Warning": 0, "Info": 0}
    for m in mismatches:
        sev_counts[m["severity"]] += 1

    terminal += (
        f"\n**Mismatches Breakdown by Severity:**\n"
        f"  - 🔴 **Critical:** {sev_counts['Critical']}\n"
        f"  - 🟡 **Warning:** {sev_counts['Warning']}\n"
        f"  - 🔵 **Info:** {sev_counts['Info']}\n"
    )
    return terminal


# ---------------------------------------------------------------------------
# Main tool function
# ---------------------------------------------------------------------------

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
        f"[yellow]key: {primary_key}\ntolerance: {tolerance}[/yellow]",
        title="[yellow]\U0001f517 run_reconciliation[/yellow]",
        border_style="yellow",
        padding=(0, 1)
    ))

    if SESSION_STATE.comp_source is None or SESSION_STATE.comp_target is None:
        return "Error: Source and Target data have not been loaded yet. Please call analyze_files first."

    comp_source = SESSION_STATE.comp_source
    comp_target = SESSION_STATE.comp_target
    source_df = SESSION_STATE.source_df
    target_df = SESSION_STATE.target_df

    key_cols = [k.strip() for k in primary_key.split(",") if k.strip()]
    for key in key_cols:
        if key not in comp_source.columns:
            return f"Error: Key column '{key}' not found in the aligned columns. Available columns: {list(comp_source.columns)}"

    tolerance_dict = None
    if tolerance.strip().lower() != "strict":
        try:
            tolerance_dict = json.loads(tolerance)
            logger.info("Parsed custom tolerance configurations: %s", tolerance_dict)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse tolerance JSON. Defaulting to strict: %s", e)

    SESSION_STATE.primary_key_cols = key_cols
    SESSION_STATE.tolerance_settings = tolerance_dict

    try:
        src_keys = _generate_index_key(comp_source, key_cols)
        tgt_keys = _generate_index_key(comp_target, key_cols)
        src_raw_keys = _generate_raw_index_key(comp_source, key_cols)
        tgt_raw_keys = _generate_raw_index_key(comp_target, key_cols)

        get_display_key = _build_key_mappings(src_keys, src_raw_keys, tgt_keys, tgt_raw_keys)

        dup = _detect_duplicates(
            src_keys, tgt_keys, source_df, target_df,
            comp_source, comp_target, get_display_key,
        )

        src_keys_set = set(dup["src_keys_clean"])
        tgt_keys_set = set(dup["tgt_keys_clean"])
        missing_in_target_keys = src_keys_set - tgt_keys_set
        missing_in_source_keys = tgt_keys_set - src_keys_set
        common_keys_set = src_keys_set & tgt_keys_set

        missing_in_target, missing_in_source = _find_missing_rows(
            missing_in_target_keys, missing_in_source_keys,
            dup["src_keys_clean"], dup["tgt_keys_clean"],
            source_df, target_df,
            src_keys.duplicated(keep=False), tgt_keys.duplicated(keep=False),
            get_display_key,
        )

        mismatches, matched_count, mismatched_count, col_mismatch_stats = _compare_columns(
            dup["comp_src_clean"], dup["comp_tgt_clean"], common_keys_set,
            key_cols, get_display_key, tolerance_dict,
        )

        results = {
            "summary": {
                "total_source_rows": len(source_df),
                "total_target_rows": len(target_df),
                "clean_source_rows": len(dup["comp_src_clean"]),
                "clean_target_rows": len(dup["comp_tgt_clean"]),
                "matched_rows": matched_count,
                "mismatched_rows": mismatched_count,
                "missing_in_target": len(missing_in_target),
                "missing_in_source": len(missing_in_source),
                "duplicate_source": len(dup["src_duplicates"]),
                "duplicate_target": len(dup["tgt_duplicates"]),
            },
            "mismatches": mismatches,
            "missing_in_target": missing_in_target,
            "missing_in_source": missing_in_source,
            "duplicates_source": [sanitize_dict(r) for r in dup["src_duplicates"].to_dict(orient='records')],
            "duplicates_target": [sanitize_dict(r) for r in dup["tgt_duplicates"].to_dict(orient='records')],
            "col_mismatch_stats": col_mismatch_stats,
            "primary_keys": key_cols,
        }

        SESSION_STATE.reconciliation_results = results
        SESSION_STATE._recon_table_shown = False

        return _render_summary(results, mismatches)

    except Exception as e:
        logger.exception("Error during reconciliation processing")
        return f"Error executing reconciliation comparison: {str(e)}"
