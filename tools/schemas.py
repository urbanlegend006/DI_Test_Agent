from pydantic import BaseModel, Field
from typing import Optional

class AnalyzeFilesInput(BaseModel):
    source_path: str = Field(
        ...,
        description="Local path to the source data file."
    )
    target_path: Optional[str] = Field(
        default=None,
        description="Local path to the target data file. Leave empty if analyzing a single file."
    )

class RunReconciliationInput(BaseModel):
    primary_key: str = Field(
        ...,
        description="The column name or comma-separated list of column names for the primary key (e.g., 'id' or 'first_name, last_name')."
    )
    tolerance: str = Field(
        default="strict",
        description="'strict' for exact match or a JSON string (e.g., '{\"numeric\": 0.05, \"date_seconds\": 3600}') for custom thresholds."
    )

class GenerateReportInput(BaseModel):
    format: str = Field(
        default="html",
        description="The file format for the output report. Must be 'html' or 'excel'."
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Optional directory path for the report output. Defaults to REPORTS_DIR."
    )
