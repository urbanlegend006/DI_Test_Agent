import logging
import re
from typing import Any
import pandas as pd

logger = logging.getLogger("reconciliation_agent.severity_classifier")

# Cache for datetime parsing results - avoids repeated pd.to_datetime() on same values
_DATETIME_CACHE: dict = {}
_DATETIME_CACHE_MAX = 4096
_PUNCTUATION_PATTERN = re.compile(r"[\s_\-/()]")


def _cached_to_datetime(val) -> pd.Timestamp | None:
    """Internal cached datetime parser."""
    key = val if isinstance(val, (str, int, float, bool)) else None
    if key is not None and key in _DATETIME_CACHE:
        return _DATETIME_CACHE[key]
    try:
        result = pd.to_datetime(val)
    except (ValueError, TypeError):
        result = None
    if key is not None and len(_DATETIME_CACHE) < _DATETIME_CACHE_MAX:
        _DATETIME_CACHE[key] = result
    return result


def safe_float(val: Any) -> float | None:
    """Convert *val* to float, returning ``None`` if conversion is impossible."""
    if val == "" or val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _simplify_str(text: str) -> str:
    """Remove punctuation, spacing, and lower-case for fuzzy text comparison."""
    return _PUNCTUATION_PATTERN.sub("", text).lower()


def _classify_date_mismatch(
    column_name: str, source_val: Any, target_val: Any,
    tolerance_settings: dict | None,
) -> str | None:
    """Compare *source_val* and *target_val* as datetimes.

    Returns a severity string (``"Critical"``, ``"Warning"``, ``"Info"``)
    or ``None`` if the values are not comparable as dates.
    """
    try:
        is_number = isinstance(source_val, (int, float))
        s_date = _cached_to_datetime(source_val) if not (is_number and pd.isna(source_val)) else None
        t_date = _cached_to_datetime(target_val) if not (
            isinstance(target_val, (int, float)) and pd.isna(target_val)
        ) else None

        if s_date is None and is_number and not pd.isna(source_val):
            s_date = pd.to_datetime(source_val, errors="coerce")
        if t_date is None and isinstance(target_val, (int, float)) and not pd.isna(target_val):
            t_date = pd.to_datetime(target_val, errors="coerce")

        if s_date is not None and t_date is not None and not pd.isna(s_date) and not pd.isna(t_date):
            time_diff = abs((s_date - t_date).total_seconds())
            date_tol_seconds = 60
            if tolerance_settings and "date_seconds" in tolerance_settings:
                date_tol_seconds = tolerance_settings["date_seconds"]

            if time_diff > 86400:
                logger.debug("Column %s has date mismatch of >1 day (%s vs %s). Severity: Critical",
                             column_name, source_val, target_val)
                return "Critical"
            elif time_diff > date_tol_seconds:
                logger.debug("Column %s has date mismatch of > tolerance (%s vs %s). Severity: Warning",
                             column_name, source_val, target_val)
                return "Warning"
            else:
                logger.debug("Column %s has minor timestamp mismatch of %s seconds. Severity: Info",
                             column_name, time_diff)
                return "Info"
    except (ValueError, TypeError) as e:
        logger.debug("Failed datetime conversion in severity classifier for column %s: %s", column_name, e)

    return None


def classify_mismatch(
    column_name: str,
    source_val: Any,
    target_val: Any,
    tolerance_settings: dict | None = None,
) -> str:
    """Classify the mismatch between a source value and target value.

    Returns:
        ``"Critical"``, ``"Warning"``, or ``"Info"``
    """
    # 1. Null / Empty checks
    s_is_empty = (source_val == "" or source_val is None)
    t_is_empty = (target_val == "" or target_val is None)

    if s_is_empty != t_is_empty:
        logger.debug("Column %s has missing value mismatch (Source: %s, Target: %s). Severity: Critical",
                     column_name, source_val, target_val)
        return "Critical"

    # Standardize to strings for textual analysis
    s_str = str(source_val).strip()
    t_str = str(target_val).strip()

    # 2. Case mismatch check
    if s_str.lower() == t_str.lower():
        logger.debug("Column %s has case mismatch. Severity: Info", column_name)
        return "Info"

    # 3. Spacing / Underscore / Special Char mismatch check
    if _simplify_str(s_str) == _simplify_str(t_str):
        logger.debug("Column %s has formatting/spacing mismatch. Severity: Info", column_name)
        return "Info"

    # 4. Numeric comparison
    s_num = safe_float(source_val)
    t_num = safe_float(target_val)

    if s_num is not None and t_num is not None:
        diff_abs = abs(s_num - t_num)
        max_val = max(abs(s_num), abs(t_num))
        diff_pct = (diff_abs / max_val) if max_val != 0 else 0

        numeric_tol = 0.01
        if tolerance_settings and "numeric" in tolerance_settings:
            numeric_tol = tolerance_settings["numeric"]

        if diff_pct > numeric_tol:
            logger.debug("Column %s has significant numeric mismatch (%s vs %s, diff: %.4f%%). Severity: Critical",
                         column_name, source_val, target_val, diff_pct * 100)
            return "Critical"
        else:
            logger.debug("Column %s has minor numeric mismatch (%s vs %s, diff: %.4f%%). Severity: Warning",
                         column_name, source_val, target_val, diff_pct * 100)
            return "Warning"

    # 5. Date comparison
    date_sev = _classify_date_mismatch(column_name, source_val, target_val, tolerance_settings)
    if date_sev is not None:
        return date_sev

    # 6. Default to Critical for unexplained text mismatches
    logger.debug("Column %s has textual mismatch. Severity: Critical", column_name)
    return "Critical"
