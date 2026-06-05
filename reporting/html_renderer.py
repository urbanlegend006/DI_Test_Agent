import logging
from datetime import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger("reconciliation_agent.reporting.html_renderer")

def render_html_report(session_state, output_dir: Path) -> Path:
    """Renders a self-contained HTML report with Chart.js and client-side SheetJS download.

    Args:
        session_state: A SessionContext (or any object exposing the same attributes).
        output_dir: Directory in which to write the report.

    Returns:
        The path of the generated HTML file.
    """
    logger.info("Starting HTML report rendering")

    results = session_state.reconciliation_results
    primary_keys = session_state.primary_key_cols
    source_filename = session_state.source_filename
    target_filename = session_state.target_filename
    source_fullpath = session_state.source_fullpath
    target_fullpath = session_state.target_fullpath
    tolerance_settings = session_state.tolerance_settings
    
    # 1. Prepare Chart.js data
    # Count severity counts
    sev_counts = {"Critical": 0, "Warning": 0, "Info": 0}
    for m in results["mismatches"]:
        sev_counts[m["severity"]] += 1
        
    # Count mismatch columns
    col_counts = {}
    for m in results["mismatches"]:
        col = m["column"]
        col_counts[col] = col_counts.get(col, 0) + 1
        
    # 2. Structure full_report_data for client-side SheetJS excel export
    run_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_report_data = {
        "summary": {
            "source_file": source_filename,
            "target_file": target_filename,
            "run_time": run_time_str,
            "primary_keys": ", ".join(primary_keys),
            "total_source_rows": results['summary']['total_source_rows'],
            "total_target_rows": results['summary']['total_target_rows'],
            "matched_rows": results['summary']['matched_rows'],
            "mismatched_rows": results['summary']['mismatched_rows'],
            "missing_in_target": results['summary']['missing_in_target'],
            "missing_in_source": results['summary']['missing_in_source'],
            "duplicate_source": results['summary']['duplicate_source'],
            "duplicate_target": results['summary']['duplicate_target']
        },
        "mismatches": [
            {
                "row_key": m["row_key"],
                "column": m["column"],
                "source_value": m["source_value"],
                "target_value": m["target_value"],
                "severity": m["severity"]
            } for m in results["mismatches"]
        ],
        "missing_in_target": results["missing_in_target"],
        "missing_in_source": results["missing_in_source"],
        "duplicates_source": results["duplicates_source"],
        "duplicates_target": results["duplicates_target"]
    }
    
    # Load Template
    template_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report.html")
    
    # Render HTML content
    html_content = template.render(
        run_time=run_time_str,
        source_filename=source_filename,
        target_filename=target_filename,
        source_fullpath=source_fullpath,
        target_fullpath=target_fullpath,
        primary_key_cols=primary_keys,
        tolerance_settings=tolerance_settings,
        results=results,
        full_report_data=full_report_data,
        column_charts_data=col_counts,
        severity_breakdown=sev_counts
    )
    
    # Output to reports directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = output_dir / f"recon_{timestamp}.html"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    logger.info("Successfully wrote HTML report to %s", output_file)
    return output_file
