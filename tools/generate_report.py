import logging
from pathlib import Path
from langchain_core.tools import tool

from config import REPORTS_DIR
from tools import SESSION_STATE

logger = logging.getLogger("reconciliation_agent.tools.generate_report")

@tool
def generate_report(format: str = "html") -> str:
    """Generates a detailed reconciliation report in HTML or Excel format.
    
    Args:
        format: The file format for the output report. Must be "html" (default) or "excel".
        
    Returns:
        The absolute path to the generated report file.
    """
    logger.info("Invoking generate_report with format='%s'", format)
    from rich.console import Console
    from rich.panel import Panel
    Console().print(Panel(
        f"[yellow]format: {format}[/yellow]",
        title="[yellow]\U0001f4ca generate_report[/yellow]",
        border_style="yellow",
        padding=(0, 1)
    ))

    # Auto-clean old reports (best-effort, non-blocking)
    try:
        from config import cleanup_old_reports
        cleanup_old_reports()
    except Exception as e:
        logger.debug("cleanup_old_reports skipped: %s", e)
    
    if SESSION_STATE.reconciliation_results is None:
        return "Error: Reconciliation has not been run yet. Please run reconciliation comparison before generating reports."

    fmt = format.strip().lower()

    try:
        if fmt == "html":
            from reporting.html_renderer import render_html_report
            report_path = render_html_report(SESSION_STATE, REPORTS_DIR)

        elif fmt == "excel":
            from reporting.excel_renderer import render_excel_report
            report_path = render_excel_report(SESSION_STATE, REPORTS_DIR)
            
        else:
            return f"Error: Unsupported format '{format}'. Only 'html' and 'excel' formats are supported."
            
        # Convert path to absolute path and format click-friendly Windows URI
        abs_path = report_path.resolve()
        file_uri = abs_path.as_uri()
        logger.info("Report successfully generated at: %s", abs_path)
        
        return (
            f"Success! The {fmt.upper()} report has been generated.\n"
            f"  \U0001f4c4 Local path: {abs_path}\n"
            f"  \U0001f517 Click to open: {file_uri}"
        )
        
    except Exception as e:
        logger.exception("Error in generate_report tool")
        return f"Failed to generate report: {str(e)}"
