# Data Reconciliation CLI Test Agent

A conversational, CLI-based reconciliation tool powered by LangChain and OpenAI. The application ingests two data files (CSV, XLSX, or JSON) representing **Source** and **Target** records, performs robust row-by-row reconciliation, auto-detects primary/composite keys, normalizes data discrepancies, and outputs premium, interactive HTML dashboard reports and styled Excel spreadsheets.

---

## 🛠️ Technology Stack

- **Core Runtime**: Python 3.14+
- **Agent Framework**: LangChain (v1.x) & LangGraph (used for state machine compilation and thread-level state memory saver)
- **Large Language Model**: OpenAI (`gpt-4o`)
- **Data Engineering**: Pandas, Numpy, Openpyxl (Excel parsing and styled generation), Chardet (CSV encoding detection)
- **Terminal User Interface**: Rich (rich logging, prompts, progress spinners, panel frames, and summary result tables)
- **Web Templating & Assets**: Jinja2 (HTML generation), TailwindCSS via CDN (styling), Chart.js (dashboard visualization), FontAwesome (icons)

---

## 📁 Project Architecture & Components

```
s:/AI Learning Projects/Test Agents/
├── README.md                      # Project documentation and changelog
├── requirements.txt               # Main dependencies
├── .env                           # Configured environment variables (API Key, Model, Log Level)
├── main.py                        # Entry point: Rich-based CLI chat loop
├── config.py                      # Configurations and logging engine (Console vs File)
├── agent.py                       # LangGraph agent setup and backwards-compatibility wrapper
├── tools/
│   ├── __init__.py
│   ├── analyze_files.py           # Tool 1: File parser, size analyzer, and column aligner
│   ├── run_reconciliation.py      # Tool 2: Core reconciliation and character-level string diffing engine
│   └── generate_report.py         # Tool 3: Generates HTML and Excel reports from reconciliation session state
├── utils/
│   ├── __init__.py
│   ├── data_normalizer.py         # Formatter for dates, numeric decimal precisions, and spaces
│   ├── key_detector.py            # Primary/composite key candidate auto-detector
│   └── severity_classifier.py     # Classifies mismatches into Critical/Warning/Info categories
├── reporting/
│   ├── __init__.py
│   ├── html_renderer.py           # Populates Jinja2 templates and compiles final HTML dashboards
│   └── excel_renderer.py          # Builds styled multi-sheet Excel workbooks with conditional highlights
├── templates/
│   └── report.html                # Premium Jinja2 template with Light/Dark modes, search, filter, and tabs
├── test_data/                     # Mock data directory containing CSV, JSON, and XLSX sample files
├── reports/                       # Target directory for generated timestamped HTML and Excel reports
└── tests/                         # Full automated test suite (pytest)
```

---

## 🤖 LangChain Agent & Tools

The agent manages the conversational pipeline, deciding when to load files, how to configure reconciliation keys, and when to render reports.

### 1. `analyze_files`
- **Purpose**: Parses input files, identifies shape (rows/columns), detects character encodings, and identifies candidates for the primary/composite key.
- **Parameters**: `source_path` (str), `target_path` (str)
- **Output**: Structuring summary returned to the agent and cached DataFrames stored in global memory.

### 2. `run_reconciliation`
- **Purpose**: Aligns rows based on selected key(s), identifies duplicate items, tracks missing records in either file, and compares matching rows cell-by-cell.
- **Parameters**: `primary_key` (str, comma-separated for composite), `tolerance` (str, defaults to `"strict"`; supports fuzzy JSON thresholds for numbers or dates)
- **Output**: Textual summary of counts (matched, mismatched, missing, duplicate) for the CLI.

### 3. `generate_report`
- **Purpose**: Compiles final reports and outputs absolute file paths.
- **Parameters**: `format` (str, `"html"` or `"excel"`)
- **Output**: Success status with the absolute path of the generated file.

### Memory & Orchestration (LangGraph)
- Uses `create_agent` from the modern `langchain.agents` package.
- Uses `MemorySaver` (from `langgraph.checkpoint.memory`) to maintain conversational state across turns.
- Instantiates a compatibility wrapper `ReconciliationAgentWrapper` to expose standard `AgentExecutor` behavior, ensuring seamless integration with legacy tools and integration tests.

---

## 🛡️ Guardrails & Data Normalization

- **Windows Emoji/Stream Guardrail**: Detects Windows platforms and wraps standard stream outputs in UTF-8 encoding to prevent `UnicodeEncodeError` crashes when printing symbols to the terminal.
- **File Ingestion Size Limit**: Warns the user if file sizes exceed 100MB before loading.
- **Key Normalization**: Normalizes IDs across different formats (e.g. aligning float string representation `'1.0'` and left-padded strings `'001'` to integer `'1'` for comparative lookup), while preserving the original raw display format in the report.
- **JSON Serialization Guardrail**: Sanitizes Pandas datatypes (e.g., `Timestamp`, `NaT`) and Numpy numeric primitives into Python-standard serializable types, avoiding runtime failures in Jinja2 JavaScript blocks.
- **Noisy Log Suppression**: Console logging handler level defaults to `WARNING` to keep the interactive terminal CLI quiet, while full `INFO`-level operational details are written cleanly to `logs/app.log`.

---

## 🧪 Testing Suite

Automated validation is implemented using `pytest` and `pytest-cov`, covering 25 test cases across unit, tool, and integration levels:
- **Unit Tests**:
  - `tests/test_data_normalizer.py`: Tests date aligning, number rounding, and spacing trims.
  - `tests/test_key_detector.py`: Validates primary/composite key detection.
  - `tests/test_severity.py`: Asserts severity classifications (Critical/Warning/Info) based on difference magnitudes.
- **Tool Tests**:
  - `tests/test_agent_tools.py`: Inspects tool behavior, caching, and state transitions.
  - `tests/test_reconciliation.py`: Tests row-by-row comparisons, duplicates, and missing row calculations.
  - `tests/test_report_generation.py`: Checks HTML and Excel file generation outputs.
- **Integration Tests**:
  - `tests/test_agent_integration.py`: Validates agent initialization, tool registration, system prompts, pipeline integrity, and API key failures.

### Execution
Run the full test suite and coverage reporting using:
```bash
.venv\Scripts\pytest --cov=. --cov-report=term-missing
```

---

## 📝 Change History (Changelog)

### [2026-06-05 09:30:00 AM] - v1.0.0: Initial Release
- Configured foundation project structure and virtual environment.
- Implemented core utilities: normalizer, key detector, and severity classifier.
- Built CLI chatbot engine (`main.py` & `agent.py`) using `AgentExecutor` and `ConversationBufferWindowMemory`.
- Created interactive HTML report template (`report.html`) and multi-sheet openpyxl Excel exporter (`excel_renderer.py`).
- Wrote initial test suite (19 passing test cases).

### [2026-06-05 10:25:00 AM] - Scale Optimization & Metric Clarity
- Generated large dataset (1,000 records) to test UI performance under scale.
- Split Bento grid card 2 (**Total Rows**) in HTML report to show both Source and Target row layouts.
- Added duplicate record count indicators and a footnote clarifying duplicate exclusion math.
- Resolved CSS tab visibility bug and fixed theme toggle infinite recursion.

### [2026-06-05 10:45:00 AM] - CLI Theme Correction & API Key Graceful Failures
- Corrected invalid color symbols (`indigo` and `slate.500`) causing Rich console parser crashes.
- Added override check for environment dotenv files to prevent stale host variable overrides.
- Added panel notification for OpenAI API key 401 exceptions.

### [2026-06-05 10:57:00 AM] - LangChain Modernization Refactoring
- Migrated legacy `ConversationBufferWindowMemory` and `AgentExecutor` dependencies (from `langchain_classic` compatibility shim) to modern `create_agent` and `MemorySaver` graphs.
- Created `ReconciliationAgentWrapper` to bridge interfaces for CLI and pytest suites.
- Eliminated all third-party deprecation warnings from standard error, resulting in 25 passing tests with zero warnings.

### [2026-06-05 11:01:00 AM] - CLI Logging Level Separation
- Separated console logging from background logging handlers in `config.py`.
- Re-routed CLI logger level to `WARNING` to keep the interactive CLI terminal free of background initialization info log prints.
