import logging
import re
from datetime import datetime
import pandas as pd

logger = logging.getLogger("reconciliation_agent.severity_classifier")

def try_float(val):
    """Helper to convert value to float, returns float or None."""
    if val == "" or val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def try_datetime(val):
    """Helper to convert value to datetime, returns datetime or None."""
    if isinstance(val, datetime):
        return val
    if val == "" or val is None:
        return None
    try:
        return pd.to_datetime(val)
    except:
        return None


def classify_mismatch(column_name: str, source_val, target_val, tolerance_settings: dict = None) -> str:
    """Classify the mismatch between a source value and target value.
    
    Returns:
        "Critical", "Warning", or "Info"
    """
    # 1. Null / Empty checks - if one is empty and the other is not, it's Critical
    s_is_empty = (source_val == "" or source_val is None)
    t_is_empty = (target_val == "" or target_val is None)
    
    if s_is_empty != t_is_empty:
        logger.debug("Column %s has missing value mismatch (Source: %s, Target: %s). Severity: Critical", 
                     column_name, source_val, target_val)
        return "Critical"
        
    # Standardize to strings for textual analysis
    s_str = str(source_val).strip()
    t_str = str(target_val).strip()
    
    # 2. Case mismatch check - if only case differs, it is Info
    if s_str.lower() == t_str.lower():
        logger.debug("Column %s has case mismatch. Severity: Info", column_name)
        return "Info"
        
    # 3. Spacing / Underscore / Special Char mismatch check - if only punctuation/spacing differs, it is Info
    def simplify_str(text):
        return re.sub(r'[\s_\-\/\(\)]', '', text).lower()
    if simplify_str(s_str) == simplify_str(t_str):
        logger.debug("Column %s has formatting/spacing mismatch. Severity: Info", column_name)
        return "Info"
        
    # 4. Numeric comparison
    s_num = try_float(source_val)
    t_num = try_float(target_val)
    
    if s_num is not None and t_num is not None:
        diff_abs = abs(s_num - t_num)
        max_val = max(abs(s_num), abs(t_num))
        
        # Avoid division by zero
        diff_pct = (diff_abs / max_val) if max_val != 0 else 0
        
        # Load numeric tolerances (default relative: 1%, absolute: 0.001)
        numeric_tol = 0.01  # 1%
        if tolerance_settings and "numeric" in tolerance_settings:
            numeric_tol = tolerance_settings["numeric"]
            
        if diff_pct > numeric_tol:
            # Significant difference
            logger.debug("Column %s has significant numeric mismatch (%s vs %s, diff: %.4f%%). Severity: Critical", 
                         column_name, source_val, target_val, diff_pct * 100)
            return "Critical"
        else:
            # Minor difference below tolerance
            logger.debug("Column %s has minor numeric mismatch (%s vs %s, diff: %.4f%%). Severity: Warning", 
                         column_name, source_val, target_val, diff_pct * 100)
            return "Warning"
            
    # 5. Date comparison
    # Try parsing both as datetimes if they look like dates
    try:
        s_date = pd.to_datetime(source_val, errors='coerce')
        t_date = pd.to_datetime(target_val, errors='coerce')
        
        if not pd.isna(s_date) and not pd.isna(t_date):
            time_diff = abs((s_date - t_date).total_seconds())
            
            # Load date tolerance (default: 60 seconds)
            date_tol_seconds = 60
            if tolerance_settings and "date_seconds" in tolerance_settings:
                date_tol_seconds = tolerance_settings["date_seconds"]
                
            if time_diff > 86400:  # More than a day
                logger.debug("Column %s has date mismatch of >1 day (%s vs %s). Severity: Critical", 
                             column_name, source_val, target_val)
                return "Critical"
            elif time_diff > date_tol_seconds:  # More than tolerance (e.g. 60 seconds)
                logger.debug("Column %s has date mismatch of > tolerance (%s vs %s). Severity: Warning", 
                             column_name, source_val, target_val)
                return "Warning"
            else:
                logger.debug("Column %s has minor timestamp mismatch of %s seconds. Severity: Info", 
                             column_name, time_diff)
                return "Info"
    except Exception as e:
        logger.debug("Failed datetime conversion in severity classifier for column %s: %s", column_name, e)
        
    # 6. Default to Critical for unexplained text mismatches
    logger.debug("Column %s has textual mismatch. Severity: Critical", column_name)
    return "Critical"
