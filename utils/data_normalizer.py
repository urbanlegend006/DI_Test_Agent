import logging
from typing import Any
import pandas as pd
import numpy as np

logger = logging.getLogger("reconciliation_agent.data_normalizer")

_MONTH_ABBREVIATIONS = frozenset({
    'jan', 'feb', 'mar', 'apr', 'may', 'jun',
    'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
})

def clean_value(val: Any) -> Any:
    """Clean a single cell value for comparison."""
    if pd.isna(val) or val is None:
        return ""

    # If it is a string, trim whitespace
    if isinstance(val, str):
        return val.strip()

    return val

def is_date_like(val) -> bool:
    """Check if a string value looks like a date/timestamp to prevent parsing of numeric IDs."""
    if not isinstance(val, str):
        return False
    val_clean = val.strip()
    if not val_clean:
        return False
    if val_clean.isdigit() and len(val_clean) <= 4:
        return False
    has_sep = any(char in val_clean for char in ['-', '/', ':', ','])
    has_month = any(m in val_clean.lower() for m in _MONTH_ABBREVIATIONS)
    return has_sep or has_month


def try_parse_datetime(val: Any) -> Any:
    """Try to parse a value into a datetime. Returns the parsed datetime or original value."""
    if isinstance(val, (int, float)) or pd.isna(val) or val == "":
        return val

    if isinstance(val, pd.Timestamp):
        return val

    val_str = str(val).strip()
    if not is_date_like(val_str):
        return val

    try:
        # Common date formats
        return pd.to_datetime(val_str)
    except (ValueError, TypeError):
        return val

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe values: strip strings, parse datetimes, unify nulls.

    Creates a copy of the dataframe with normalized data.
    """
    logger.debug("Normalizing DataFrame of shape %s", df.shape)
    norm_df = df.copy()

    # 1. Standardize column names (strip whitespace)
    norm_df.columns = [str(col).strip() for col in norm_df.columns]

    for col in norm_df.columns:
        s = norm_df[col]
        is_string_col = pd.api.types.is_string_dtype(s) or s.dtype == object

        # Vectorized cleanup for string columns: strip whitespace, replace NaN/None with ""
        if is_string_col:
            # Vectorized: replace NaN/None with "" in place, then strip strings
            s = s.where(s.notna(), "")
            if pd.api.types.is_string_dtype(s) or s.dtype == object:
                s = s.str.strip()
        else:
            # For non-string columns, just nullify NaNs (caller code keeps these)
            s = s.where(s.notna(), np.nan)

        # Try to detect and convert datetime columns (vectorized)
        try:
            if is_string_col:
                # Quick string check: any non-empty value
                non_empty = s[s != ""]
                if not non_empty.empty:
                    # Check if all values look like dates (sample first 50 for speed)
                    sample = non_empty.head(50)
                    if sample.map(is_date_like).all():
                        parsed = pd.to_datetime(s, errors='coerce')
                        valid_count = parsed.notna().sum()
                        if valid_count > 0 and (valid_count / len(non_empty)) >= 0.5:
                            norm_df[col] = parsed.where(parsed.notna(), np.nan)
                            logger.debug("Column '%s' parsed as datetime", col)
                            continue
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug("Failed checking datetime format for col '%s': %s", col, e)

        # Try to parse string numbers to actual float/int where possible
        try:
            col_lower = col.lower()
            if col_lower in ('id', 'pk', 'code', 'key', 'uuid') or _series_has_leading_zeros(s):
                norm_df[col] = s
                continue

            if is_string_col:
                non_empty = s[s != ""]
                if not non_empty.empty:
                    parsed = pd.to_numeric(s, errors='coerce')
                    valid_count = parsed.notna().sum()
                    # Original semantics: ratio of valid parses to non-empty count.
                    # This allows columns with many nulls to still be converted to numeric.
                    if valid_count > 0 and (valid_count / len(non_empty)) >= 0.8:
                        norm_df[col] = parsed
                        logger.debug("Column '%s' cast to numeric", col)
                        continue
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug("Failed checking numeric format for col '%s': %s", col, e)

        norm_df[col] = s

    # Final pass: replace any remaining NaN/NaT with empty string to match original semantics.
    # This also forces numeric/datetime columns with NaNs to object dtype (mixed values + "" strings).
    for col in norm_df.columns:
        s = norm_df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            # Convert to object first so "" can coexist with non-null values
            s_object = s.astype(object).where(s.notna(), "")
            norm_df[col] = s_object

    return norm_df


def _series_has_leading_zeros(series: pd.Series) -> bool:
    """Vectorized check: any string value starts with '0' (and is longer than 1 char)."""
    if not (pd.api.types.is_string_dtype(series) or series.dtype == object):
        return False
    str_series = series.astype(str)
    # Match strings that start with '0' and have more than 1 digit
    return str_series.str.match(r'^0\d').any()

def align_columns(source_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Align column names between Source and Target DataFrames.

    Matches columns case-insensitively and ignoring underscores/spaces.
    Returns:
        (aligned_source, aligned_target, alignment_metadata)
    """
    logger.info("Aligning columns between Source and Target")

    src_cols = list(source_df.columns)
    tgt_cols = list(target_df.columns)

    def normalize_name(name):
        return str(name).strip().lower().replace("_", "").replace(" ", "").replace("-", "")

    normalized_src = {normalize_name(c): c for c in src_cols}
    normalized_tgt = {normalize_name(c): c for c in tgt_cols}

    aligned_src_cols = []
    aligned_tgt_cols = []

    # Store alignment map
    alignment_map = {}
    extra_in_source = []
    extra_in_target = []

    # Find matching columns and direct maps
    for norm_name, src_name in normalized_src.items():
        if norm_name in normalized_tgt:
            tgt_name = normalized_tgt[norm_name]
            alignment_map[src_name] = tgt_name
            aligned_src_cols.append(src_name)
            aligned_tgt_cols.append(tgt_name)
        else:
            extra_in_source.append(src_name)

    for norm_name, tgt_name in normalized_tgt.items():
        if norm_name not in normalized_src:
            extra_in_target.append(tgt_name)

    logger.info("Aligned %d columns. Extra in Source: %s, Extra in Target: %s",
                len(alignment_map), extra_in_source, extra_in_target)

    # Reindex and rename dataframes for comparison
    # Rename target columns to match source names for the matching columns
    rename_dict = {tgt: src for src, tgt in alignment_map.items()}

    # Create final comparative dataframes
    comp_source = source_df[aligned_src_cols].copy()
    comp_target = target_df[aligned_tgt_cols].copy().rename(columns=rename_dict)

    metadata = {
        "alignment_map": alignment_map,
        "extra_in_source": extra_in_source,
        "extra_in_target": extra_in_target,
        "aligned_columns": aligned_src_cols
    }

    return comp_source, comp_target, metadata
