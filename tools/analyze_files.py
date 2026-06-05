import logging
import os
import json
import pandas as pd
from pathlib import Path
from langchain_core.tools import tool

from config import FILE_SIZE_WARNING_MB
from tools import SESSION_STATE
from utils.data_normalizer import normalize_dataframe, align_columns
from utils.key_detector import detect_primary_key

logger = logging.getLogger("reconciliation_agent.tools.analyze_files")

def _detect_csv_delimiter(file_path: Path, encoding: str) -> str:
    """Sniff the most likely delimiter for a CSV/TXT file from its first line."""
    try:
        import csv as _csv
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            sample = f.read(8192)
        try:
            dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except Exception:
            return ','
    except Exception:
        return ','


def _announce_load(file_path: Path) -> None:
    """Print a brief progress message before reading a file."""
    try:
        from rich.console import Console
        size_mb = file_path.stat().st_size / (1024 * 1024)
        suffix = "MB" if size_mb >= 1 else "KB"
        size_str = f"{size_mb:.2f} {suffix}" if size_mb >= 1 else f"{size_mb * 1024:.1f} {suffix}"
        Console().print(f"[dim]  Reading {file_path.name} ({size_str})...[/dim]")
    except Exception:
        pass


def parse_file_to_df(file_path: Path) -> pd.DataFrame:
    """Read XLSX, JSON, CSV, TXT, or Parquet file and return a DataFrame."""
    ext = file_path.suffix.lower()
    _announce_load(file_path)

    if ext == '.csv':
        # Detect encoding
        import chardet
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'
        logger.info("Reading CSV file %s with encoding %s", file_path.name, encoding)
        return pd.read_csv(file_path, encoding=encoding)

    elif ext == '.txt':
        # Treat .txt as a delimited text file (auto-detect delimiter)
        import chardet
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'
        delimiter = _detect_csv_delimiter(file_path, encoding)
        logger.info("Reading TXT file %s as delimited (encoding=%s, delimiter=%r)",
                    file_path.name, encoding, delimiter)
        return pd.read_csv(file_path, encoding=encoding, sep=delimiter)

    elif ext in ('.xlsx', '.xls'):
        logger.info("Reading Excel file %s", file_path.name)
        return pd.read_excel(file_path, engine='openpyxl')

    elif ext == '.json':
        logger.info("Reading JSON file %s", file_path.name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # If the JSON is a dictionary or contains a list inside it
        if isinstance(data, dict):
            # Try to find a list of records in the dict
            for key, val in data.items():
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                    logger.info("Found list of records in JSON key '%s'", key)
                    return pd.json_normalize(val)
            # If no list found, normalize the dict itself (either single record or key-value)
            return pd.json_normalize(data)
        elif isinstance(data, list):
            return pd.json_normalize(data)
        else:
            raise ValueError("Unsupported JSON structure (must be list of objects or dict containing a list of objects)")

    elif ext == '.parquet':
        logger.info("Reading Parquet file %s", file_path.name)
        try:
            return pd.read_parquet(file_path)
        except ImportError as e:
            raise ValueError(
                "Parquet support requires either 'pyarrow' or 'fastparquet'. "
                "Install with: pip install pyarrow"
            ) from e

    else:
        raise ValueError(
            f"Unsupported file extension: {ext}. "
            f"Supported formats: .xlsx, .xls, .csv, .txt, .json, .parquet"
        )

@tool
def analyze_files(source_path: str, target_path: str) -> str:
    """Ingests source and target files, parses them, normalizes columns, and returns data structure details.
    
    Args:
        source_path: Local path to the source data file.
        target_path: Local path to the target data file.
        
    Returns:
        A text summary of the files, their shapes, aligned columns, and detected key candidates.
    """
    logger.info("Invoking analyze_files on source='%s', target='%s'", source_path, target_path)
    from rich.console import Console
    from rich.panel import Panel
    Console().print(Panel(
        f"[yellow]source: {source_path}\ntarget: {target_path}[/yellow]",
        title="[yellow]\U0001f50d analyze_files[/yellow]",
        border_style="yellow",
        padding=(0, 1)
    ))
    
    # 1. Clean paths and verify existence
    def clean_path(path_str: str) -> Path:
        p_str = path_str.strip().strip("'\"")
        # Handle file:/// URIs
        if p_str.startswith("file:///"):
            p_str = p_str.replace("file:///", "")
        p = Path(p_str)
        # Fallback to absolute workspace path if relative
        if not p.is_absolute():
            # Check s:/AI Learning Projects/Test Agents
            workspace_p = Path("s:/AI Learning Projects/Test Agents") / p
            if workspace_p.exists():
                p = workspace_p
        return p
        
    try:
        src_file = clean_path(source_path)
        tgt_file = clean_path(target_path)
        
        if not src_file.exists():
            return f"Error: Source file does not exist at path: {source_path}"
        if not tgt_file.exists():
            return f"Error: Target file does not exist at path: {target_path}"
            
        # Check file sizes
        for path in (src_file, tgt_file):
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > FILE_SIZE_WARNING_MB:
                logger.warning("File %s is %.2f MB, which exceeds warning threshold of %d MB", 
                               path.name, size_mb, FILE_SIZE_WARNING_MB)
                               
        # Load and normalize files in parallel for large files
        # (only parallelize if both files are > 5MB; small files have negligible load time
        # and the thread overhead would slow them down)
        from concurrent.futures import ThreadPoolExecutor
        PARALLEL_THRESHOLD_MB = 5

        src_size_mb = src_file.stat().st_size / (1024 * 1024)
        tgt_size_mb = tgt_file.stat().st_size / (1024 * 1024)
        use_parallel = src_size_mb >= PARALLEL_THRESHOLD_MB and tgt_size_mb >= PARALLEL_THRESHOLD_MB

        if use_parallel:
            logger.info("Loading and normalizing both files in parallel (parallel threshold: %d MB)",
                        PARALLEL_THRESHOLD_MB)
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_src_raw = executor.submit(parse_file_to_df, src_file)
                future_tgt_raw = executor.submit(parse_file_to_df, tgt_file)
                src_raw_df = future_src_raw.result()
                tgt_raw_df = future_tgt_raw.result()

                # Normalization is also CPU-bound; parallelize it as well
                future_src_norm = executor.submit(normalize_dataframe, src_raw_df)
                future_tgt_norm = executor.submit(normalize_dataframe, tgt_raw_df)
                src_df = future_src_norm.result()
                tgt_df = future_tgt_norm.result()
        else:
            # Sequential load for small files (avoids thread overhead)
            src_raw_df = parse_file_to_df(src_file)
            tgt_raw_df = parse_file_to_df(tgt_file)

            # Normalize DataFrames
            src_df = normalize_dataframe(src_raw_df)
            tgt_df = normalize_dataframe(tgt_raw_df)
        
        # Align columns
        comp_src, comp_tgt, align_meta = align_columns(src_df, tgt_df)
        
        # Detect primary keys
        detected_src_keys = detect_primary_key(comp_src)
        detected_tgt_keys = detect_primary_key(comp_tgt)
        
        # Find common keys if they exist in both
        common_keys = None
        if detected_src_keys and detected_tgt_keys:
            intersection = list(set(detected_src_keys) & set(detected_tgt_keys))
            if intersection:
                common_keys = intersection
            else:
                common_keys = detected_src_keys # Fallback to source's suggested keys
        elif detected_src_keys:
            common_keys = detected_src_keys
        elif detected_tgt_keys:
            common_keys = detected_tgt_keys
            
        # Cache DataFrames and metadata in SESSION_STATE
        SESSION_STATE.source_df = src_df
        SESSION_STATE.target_df = tgt_df
        SESSION_STATE.comp_source = comp_src
        SESSION_STATE.comp_target = comp_tgt
        SESSION_STATE.align_meta = align_meta
        SESSION_STATE.source_filename = src_file.name
        SESSION_STATE.target_filename = tgt_file.name
        SESSION_STATE.source_fullpath = str(src_file.resolve())
        SESSION_STATE.target_fullpath = str(tgt_file.resolve())
        
        # Build success response
        summary = (
            f"Successfully loaded and analyzed both files!\n\n"
            f"**File 1 (Source):** {src_file.name}\n"
            f"  - Format: {src_file.suffix}\n"
            f"  - Size: {src_file.stat().st_size / 1024:.2f} KB\n"
            f"  - Total Rows: {len(src_df)}\n"
            f"  - Total Columns: {len(src_df.columns)}\n\n"
            f"**File 2 (Target):** {tgt_file.name}\n"
            f"  - Format: {tgt_file.suffix}\n"
            f"  - Size: {tgt_file.stat().st_size / 1024:.2f} KB\n"
            f"  - Total Rows: {len(tgt_df)}\n"
            f"  - Total Columns: {len(tgt_df.columns)}\n\n"
            f"**Column Alignment:**\n"
            f"  - Aligned/Matching Columns ({len(align_meta['aligned_columns'])}): {', '.join(align_meta['aligned_columns'])}\n"
        )
        
        if align_meta['extra_in_source']:
            summary += f"  - Extra columns in Source: {', '.join(align_meta['extra_in_source'])}\n"
        if align_meta['extra_in_target']:
            summary += f"  - Extra columns in Target: {', '.join(align_meta['extra_in_target'])}\n"
            
        if common_keys:
            SESSION_STATE.detected_keys = common_keys
            summary += f"\n**Suggested Primary Key(s):** {', '.join(common_keys)}\n"
        else:
            SESSION_STATE.detected_keys = None
            summary += f"\n**Warning:** No primary key could be auto-detected. Please ask the user to supply the primary key(s) before running reconciliation.\n"
            
        return summary
        
    except Exception as e:
        logger.exception("Error in analyze_files")
        return f"I encountered an error parsing the files: {str(e)}. Please check the file formats or paths."
