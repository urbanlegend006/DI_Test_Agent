import logging
import pandas as pd

logger = logging.getLogger("reconciliation_agent.key_detector")

# Common names indicating primary keys
PRIMARY_KEY_INDICATORS = [
    "id", "uuid", "key", "pk", "record_id", "code", "index", "uid",
    "identifier", "batch", "batch_number", "lot", "serial", "sequence"
]

def is_column_unique(df: pd.DataFrame, col: str) -> bool:
    """Check if all non-empty values in a column are unique and the column is not entirely empty."""
    non_empty = df[col][df[col] != ""]
    if non_empty.empty:
        return False
    return non_empty.nunique() == len(non_empty)

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
                # If it's a string or numerical identifier type, recommend it
                col_lower = col.lower()
                # Skip numeric value columns or dates for single keys unless they are named id-like
                if "val" in col_lower or "amount" in col_lower or "date" in col_lower or "time" in col_lower:
                    continue
                logger.info("Found fallback unique key: %s", col)
                return [col]
                
    # 3. Look for composite key candidates (e.g. combination of two or three columns)
    # Common composite structures in records: e.g. (batch/ID + parameter/date/name)
    # Check combinations of 2 columns
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1, col2 = cols[i], cols[j]
            # Skip checking combinations of two numeric/value columns
            c1_lower = col1.lower()
            c2_lower = col2.lower()
            if ("value" in c1_lower or "val" in c1_lower) and ("value" in c2_lower or "val" in c2_lower):
                continue
                
            # Check uniqueness of composite key
            combined = df[col1].astype(str) + "|" + df[col2].astype(str)
            # Remove entirely empty rows
            non_empty_combined = combined[combined != "|"]
            if not non_empty_combined.empty and non_empty_combined.nunique() == len(non_empty_combined):
                # Prefer combinations containing ID indicators
                score = 0
                for col_name in [col1, col2]:
                    if any(ind in col_name.lower() for ind in PRIMARY_KEY_INDICATORS):
                        score += 1
                if score > 0:
                    logger.info("Found composite key candidate: %s + %s (score: %d)", col1, col2, score)
                    return [col1, col2]
                    
    # Check combinations of 3 columns (only if they include indicator names)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            for k in range(j + 1, len(cols)):
                col1, col2, col3 = cols[i], cols[j], cols[k]
                # Check if at least one column is an indicator name
                if not any(any(ind in c.lower() for ind in PRIMARY_KEY_INDICATORS) for c in [col1, col2, col3]):
                    continue
                combined = df[col1].astype(str) + "|" + df[col2].astype(str) + "|" + df[col3].astype(str)
                non_empty_combined = combined[combined != "||"]
                if not non_empty_combined.empty and non_empty_combined.nunique() == len(non_empty_combined):
                    logger.info("Found 3-column composite key candidate: %s + %s + %s", col1, col2, col3)
                    return [col1, col2, col3]
                    
    logger.warning("No suitable unique key detected.")
    return None
