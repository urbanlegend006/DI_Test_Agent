import logging
import pandas as pd

logger = logging.getLogger("reconciliation_agent.key_detector")

# Common names indicating primary keys
PRIMARY_KEY_INDICATORS = [
    "id", "uuid", "key", "pk", "record_id", "code", "index", "uid",
    "identifier", "batch", "batch_number", "lot", "serial", "sequence", "seq", "ref"
]

# Cap composite key search to avoid O(C^2)/O(C^3) explosion on wide tables
MAX_COMPOSITE_COMBOS = 100


def _has_indicator(name: str) -> bool:
    """Return True if the column name contains a primary-key indicator substring."""
    n = name.lower()
    return any(ind in n for ind in PRIMARY_KEY_INDICATORS)


def is_column_unique(df: pd.DataFrame, col: str) -> bool:
    """Check if all non-empty values in a column are unique and the column is not entirely empty."""
    non_empty = df[col][df[col] != ""]
    if non_empty.empty:
        return False
    return non_empty.nunique() == len(non_empty)


def _vec_combo_uniqueness(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Vectorized combination of columns by string concatenation, then return the non-empty combined Series."""
    if len(cols) == 1:
        combined = df[cols[0]].astype(str)
        return combined[combined != ""]
    combined = df[cols[0]].astype(str)
    for c in cols[1:]:
        combined = combined + "|" + df[c].astype(str)
    empty_token = "|".join([""] * len(cols))
    return combined[combined != empty_token]


def detect_primary_key(df: pd.DataFrame) -> list[str] | None:
    """Detect single or composite primary key columns.

    Returns a list of column names, or None if no unique identifier can be determined.
    """
    logger.info("Detecting primary keys for DataFrame of columns: %s", list(df.columns))

    # Clean up column names for matching
    col_map = {col.lower().strip().replace("_", "").replace(" ", "").replace("-", ""): col for col in df.columns}

    # 1. Look for explicit single unique keys matching patterns
    unique_candidates = []
    for indicator in PRIMARY_KEY_INDICATORS:
        if indicator in col_map:
            actual_col = col_map[indicator]
            if is_column_unique(df, actual_col):
                logger.info("Found unique key candidate: %s", actual_col)
                return [actual_col]
            else:
                unique_candidates.append(actual_col)

    # 2. Check if any other single column is unique
    for col in df.columns:
        if col not in unique_candidates:
            if is_column_unique(df, col):
                col_lower = col.lower()
                if "val" in col_lower or "amount" in col_lower or "date" in col_lower or "time" in col_lower:
                    continue
                logger.info("Found fallback unique key: %s", col)
                return [col]

    # 3. Composite key candidates (2 columns).
    # Restrict to combos where at least one column is an indicator name, and cap total combos.
    cols = list(df.columns)
    indicator_cols = [c for c in cols if _has_indicator(c)]
    other_cols = [c for c in cols if c not in indicator_cols]

    two_combos_checked = 0
    for c1 in indicator_cols:
        for c2 in indicator_cols + other_cols:
            if c1 == c2:
                continue
            if two_combos_checked >= MAX_COMPOSITE_COMBOS:
                break
            two_combos_checked += 1
            c1_lower = c1.lower()
            c2_lower = c2.lower()
            if ("value" in c1_lower or "val" in c1_lower) and ("value" in c2_lower or "val" in c2_lower):
                continue
            combined = _vec_combo_uniqueness(df, [c1, c2])
            if not combined.empty and combined.nunique() == len(combined):
                score = sum(1 for cn in (c1, c2) if _has_indicator(cn))
                if score > 0:
                    logger.info("Found composite key candidate: %s + %s (score: %d)", c1, c2, score)
                    return [c1, c2]
        else:
            continue
        break

    # 4. Composite key candidates (3 columns) - require at least two indicator names.
    three_combos_checked = 0
    for i, c1 in enumerate(indicator_cols):
        for j, c2 in enumerate(indicator_cols):
            if j <= i:
                continue
            for c3 in indicator_cols + other_cols:
                if c3 in (c1, c2):
                    continue
                if three_combos_checked >= MAX_COMPOSITE_COMBOS:
                    break
                three_combos_checked += 1
                indicator_count = sum(1 for c in (c1, c2, c3) if _has_indicator(c))
                if indicator_count < 2:
                    continue
                combined = _vec_combo_uniqueness(df, [c1, c2, c3])
                if not combined.empty and combined.nunique() == len(combined):
                    logger.info("Found 3-column composite key candidate: %s + %s + %s", c1, c2, c3)
                    return [c1, c2, c3]
            else:
                continue
            break
        else:
            continue
        break

    logger.warning("No suitable unique key detected.")
    return None
