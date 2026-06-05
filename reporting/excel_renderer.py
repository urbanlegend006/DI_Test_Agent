import logging
import json
from datetime import datetime
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

logger = logging.getLogger("reconciliation_agent.reporting.excel_renderer")

_FONT_HEADER = Font(name="Arial", size=11, bold=True, color="FFFFFF")
_FONT_TITLE = Font(name="Arial", size=14, bold=True, color="312E81")
_FONT_BOLD = Font(name="Arial", size=10, bold=True)
_FONT_REGULAR = Font(name="Arial", size=10)
_FILL_INDIGO = PatternFill(start_color="4F46E5", end_color="4F46E5", fill_type="solid")
_FILL_CRITICAL = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
_FILL_WARNING = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
_FILL_INFO = PatternFill(start_color="E0F2FE", end_color="E0F2FE", fill_type="solid")
_THIN_BORDER = Border(
    left=Side(style="thin", color="DDDDDD"),
    right=Side(style="thin", color="DDDDDD"),
    top=Side(style="thin", color="DDDDDD"),
    bottom=Side(style="thin", color="DDDDDD"),
)

def apply_auto_width(ws):
    """Adjust worksheet column widths to fit contents."""
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            # Check cell value length
            val = str(cell.value or '')
            # If it looks like a long JSON block, cap the width calculation
            if len(val) > 40:
                val = val[:40]
            max_len = max(max_len, len(val))
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

def render_excel_report(session_state, output_dir: Path) -> Path:
    """Generates a styled, multi-sheet Excel reconciliation report.

    Args:
        session_state: A SessionContext (or any object exposing the same attributes).
        output_dir: Directory in which to write the report.

    Returns:
        The path of the generated Excel file.
    """
    logger.info("Starting Excel report generation")

    results = session_state.reconciliation_results
    primary_keys = session_state.primary_key_cols
    source_filename = session_state.source_filename
    target_filename = session_state.target_filename

    wb = Workbook()

    # 1. Summary Sheet
    ws_sum = wb.active
    ws_sum.title = "Summary"
    ws_sum.views.sheetView[0].showGridLines = True

    ws_sum["A1"] = "Data Reconciliation Report"
    ws_sum["A1"].font = _FONT_TITLE

    metadata = [
        ("Source File", source_filename),
        ("Target File", target_filename),
        ("Reconciliation Keys", ", ".join(primary_keys)),
        ("Execution Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ]

    ws_sum.append([]) # spacing
    for label, val in metadata:
        ws_sum.append([label, val])
        row_idx = ws_sum.max_row
        ws_sum[f"A{row_idx}"].font = _FONT_BOLD
        ws_sum[f"B{row_idx}"].font = _FONT_REGULAR

    ws_sum.append([]) # spacing

    # Table Header for metrics
    ws_sum.append(["Reconciliation Metrics", "Count"])
    header_row = ws_sum.max_row
    ws_sum[f"A{header_row}"].font = _FONT_HEADER
    ws_sum[f"A{header_row}"].fill = _FILL_INDIGO
    ws_sum[f"B{header_row}"].font = _FONT_HEADER
    ws_sum[f"B{header_row}"].fill = _FILL_INDIGO

    metrics = [
        ("Total Rows (Source)", results['summary']['total_source_rows']),
        ("Total Rows (Target)", results['summary']['total_target_rows']),
        ("Clean Source Rows", results['summary']['clean_source_rows']),
        ("Clean Target Rows", results['summary']['clean_target_rows']),
        ("Matched Rows", results['summary']['matched_rows']),
        ("Mismatched Rows", results['summary']['mismatched_rows']),
        ("Missing in Target", results['summary']['missing_in_target']),
        ("Missing in Source", results['summary']['missing_in_source']),
        ("Duplicate Rows (Source)", results['summary']['duplicate_source']),
        ("Duplicate Rows (Target)", results['summary']['duplicate_target'])
    ]

    for metric, count in metrics:
        ws_sum.append([metric, count])
        row_idx = ws_sum.max_row
        ws_sum[f"A{row_idx}"].font = _FONT_REGULAR
        ws_sum[f"B{row_idx}"].font = _FONT_BOLD
        ws_sum[f"B{row_idx}"].alignment = Alignment(horizontal="right")
        ws_sum[f"A{row_idx}"].border = _THIN_BORDER
        ws_sum[f"B{row_idx}"].border = _THIN_BORDER

    # 2. Mismatches Sheet
    ws_mis = wb.create_sheet(title="Mismatches")
    ws_mis.views.sheetView[0].showGridLines = True

    headers_mis = ["Row Key", "Column Name", "Source Value", "Target Value", "Severity"]
    ws_mis.append(headers_mis)
    for col_idx in range(1, 6):
        cell = ws_mis.cell(row=1, column=col_idx)
        cell.font = _FONT_HEADER
        cell.fill = _FILL_INDIGO

    for item in results["mismatches"]:
        ws_mis.append([
            item["row_key"],
            item["column"],
            item["source_value"],
            item["target_value"],
            item["severity"]
        ])

        row_idx = ws_mis.max_row
        sev = item["severity"]

        # Determine fill based on severity
        fill = _FILL_INFO
        if sev == "Critical":
            fill = _FILL_CRITICAL
        elif sev == "Warning":
            fill = _FILL_WARNING

        # Apply fonts, fills, borders
        for col_idx in range(1, 6):
            cell = ws_mis.cell(row=row_idx, column=col_idx)
            cell.font = _FONT_REGULAR
            cell.fill = fill
            cell.border = _THIN_BORDER
            if col_idx in (3, 4):
                cell.font = Font(name="Courier New", size=10)

    # 3-6. Missing / Duplicate sheets (via shared helpers)
    def _create_missing_sheet(title: str, data: list) -> None:
        ws = wb.create_sheet(title=title)
        ws.views.sheetView[0].showGridLines = True
        ws.append(["Row Key", "Record Details (JSON)"])
        for col_idx in range(1, 3):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = _FONT_HEADER
            cell.fill = _FILL_INDIGO
        for item in data:
            ws.append([item["row_key"], json.dumps(item["row_data"])])
            row_idx = ws.max_row
            ws.cell(row=row_idx, column=1).font = _FONT_BOLD
            ws.cell(row=row_idx, column=1).border = _THIN_BORDER
            ws.cell(row=row_idx, column=2).font = Font(name="Courier New", size=9)
            ws.cell(row=row_idx, column=2).border = _THIN_BORDER

    def _create_duplicate_sheet(title: str, data: list) -> None:
        if not data:
            return
        ws = wb.create_sheet(title=title)
        ws.views.sheetView[0].showGridLines = True
        ws.append(["Row Key", "Full Record JSON"])
        for col_idx in range(1, 3):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = _FONT_HEADER
            cell.fill = _FILL_INDIGO
        for item in data:
            row_key = item.get("_row_key", "")
            clean_item = {k: v for k, v in item.items() if k != "_row_key"}
            ws.append([row_key, json.dumps(clean_item)])
            row_idx = ws.max_row
            ws.cell(row=row_idx, column=1).font = _FONT_BOLD
            ws.cell(row=row_idx, column=1).border = _THIN_BORDER
            ws.cell(row=row_idx, column=2).font = Font(name="Courier New", size=9)
            ws.cell(row=row_idx, column=2).border = _THIN_BORDER

    _create_missing_sheet("Missing In Target", results["missing_in_target"])
    _create_missing_sheet("Missing In Source", results["missing_in_source"])
    _create_duplicate_sheet("Duplicates Source", results["duplicates_source"])
    _create_duplicate_sheet("Duplicates Target", results["duplicates_target"])

    # Adjust column widths for all sheets
    for sheet in wb.worksheets:
        apply_auto_width(sheet)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = output_dir / f"recon_{timestamp}.xlsx"
    wb.save(output_file)

    logger.info("Successfully generated Excel report at %s", output_file)
    return output_file
