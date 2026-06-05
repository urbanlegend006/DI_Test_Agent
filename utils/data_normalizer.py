import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("reconciliation_agent.data_normalizer")

def clean_value(val):
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
    has_month = any(m in val_clean.lower() for m in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    return has_sep or has_month

def should_not_convert_to_numeric(series: pd.Series) -> bool:
    """Check if any value in the series has a leading zero and is longer than 1 character."""
    for x in series:
        if isinstance(x, str):
            s = x.strip()
            if len(s) > 1 and s.startswith('0') and s.isdigit():
                return True
    return False


def try_parse_datetime(val):
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
        # Apply basic value cleaning
        norm_df[col] = norm_df[col].apply(clean_value)
        
        # Try to detect and convert datetime columns
        try:
            # Check if column has string dates
            sample_non_empty = norm_df[col][norm_df[col] != ""]
            if not sample_non_empty.empty and all(isinstance(x, str) for x in sample_non_empty):
                # Ensure strings actually look like date strings
                if all(is_date_like(x) for x in sample_non_empty):
                    # Try parsing to datetime
                    parsed = pd.to_datetime(norm_df[col], errors='coerce')
                    # If at least 50% of non-empty values parse as datetime, keep it
                    valid_count = parsed.notna().sum()
                    if valid_count > 0 and (valid_count / len(sample_non_empty)) >= 0.5:
                        norm_df[col] = parsed
                        logger.debug("Column '%s' parsed as datetime", col)
                        continue
        except Exception as e:
            logger.debug("Failed checking datetime format for col '%s': %s", col, e)
            
        # Try to parse string numbers to actual float/int where possible
        try:
            # Skip numeric conversion for ID-like columns or fields with leading zeros (like "001")
            if col.lower() in ['id', 'pk', 'code', 'key', 'uuid'] or should_not_convert_to_numeric(norm_df[col]):
                continue
                
            sample_non_empty = norm_df[col][norm_df[col] != ""]
            if not sample_non_empty.empty and all(isinstance(x, str) for x in sample_non_empty):
                parsed = pd.to_numeric(norm_df[col], errors='coerce')
                valid_count = parsed.notna().sum()
                if valid_count > 0 and (valid_count / len(sample_non_empty)) >= 0.8:
                    norm_df[col] = parsed.fillna(np.nan)
                    logger.debug("Column '%s' cast to numeric", col)
        except Exception as e:
            logger.debug("Failed checking numeric format for col '%s': %s", col, e)
            
    # Standardize remaining null values to ""
    for col in norm_df.columns:
        # replace NaT and NaN with ""
        norm_df[col] = norm_df[col].apply(lambda x: "" if pd.isna(x) else x)
        
    return norm_df

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
