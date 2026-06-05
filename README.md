# Data Reconciliation CLI Test Agent

A conversational, CLI-based reconciliation tool powered by LangChain and OpenAI. The application ingests two data files (CSV, XLSX, TXT, JSON, or Parquet) representing **Source** and **Target** records, performs robust row-by-row reconciliation, auto-detects primary/composite keys, normalizes data discrepancies, and outputs premium, interactive HTML dashboard reports and styled Excel spreadsheets.

The CLI uses a modernized Rich-based terminal UI (welcome banner, slash commands, aligned prompt/response markers, and tool-call panels) and is powered by a typed session context with vectorized data processing for fast performance on large files.

---

## 🛠️ Technology Stack

- **Core Runtime**: Python 3.10+ (tested on 3.14)
- **Agent Framework**: LangChain (v1.x) & LangGraph (state machine + thread-level memory saver)
- **Large Language Model**: OpenAI (`gpt-4o`)
- **Data Engineering**: Pandas (vectorized ops), Numpy (`np.isclose` for fast numeric comparison), Openpyxl (styled Excel), PyArrow (optional Parquet), Chardet (CSV encoding detection)
- **Terminal User Interface**: Rich (welcome banner, panels, prompt/response markers, slash commands, summary tables)
- **Web Templating & Assets**: Jinja2 (HTML generation), TailwindCSS via CDN (styling), Chart.js (dashboard visualization), FontAwesome (icons)
- **Packaging**: `pyproject.toml` with console-script entry point (`reconagent`)

---

## 📁 Project Architecture & Components

```
s:/AI Learning Projects/Test Agents/
├── README.md                      # Project documentation and changelog
├── pyproject.toml                 # Modern packaging metadata + console-script entry point
├── requirements.txt               # Pinned main dependencies
├── .env                           # Configured environment variables (API Key, Model, Log Level)
├── main.py                        # Entry point: Rich-based CLI chat loop, welcome banner, slash commands
├── config.py                      # Config, dual logging (console + rotating file), report-retention helper
├── agent.py                       # LangGraph agent setup and backwards-compatibility wrapper
├── tools/
│   ├── __init__.py                # SessionContext dataclass + module-level SESSION_STATE instance
│   ├── analyze_files.py           # Tool 1: File parser (CSV/TXT/XLSX/JSON/Parquet), size analyzer, parallel load+normalize
│   ├── run_reconciliation.py      # Tool 2: Vectorized reconciliation and character-level string diffing engine
│   └── generate_report.py         # Tool 3: Generates HTML and Excel reports with clickable file:// URIs
├── utils/
│   ├── __init__.py
│   ├── data_normalizer.py         # Vectorized formatter for dates, numeric decimal precisions, and spaces
│   ├── key_detector.py            # Primary/composite key candidate auto-detector (indicator-aware, combo-capped)
│   └── severity_classifier.py     # Classifies mismatches into Critical/Warning/Info (LRU datetime cache)
├── reporting/
│   ├── __init__.py
│   ├── html_renderer.py           # Populates Jinja2 templates and compiles final HTML dashboards
│   └── excel_renderer.py          # Builds styled multi-sheet Excel workbooks with conditional highlights
├── templates/
│   └── report.html                # Premium Jinja2 template with Light/Dark modes, search, filter, and tabs
├── logs/                          # Rotating log files (10 MB x 5) — gitignored
├── reports/                       # Target directory for generated timestamped HTML and Excel reports
├── test_data/                     # Mock data directory containing CSV, JSON, and XLSX sample files
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

## 🖥️ Terminal UI (Rich)

The CLI is built on Rich with a deliberate, minimal layout:

- **Welcome banner** — full-width `Panel` with an ASCII `R A` mark on the left and a metadata `Panel` on the right, joined with `Columns`. The layout adapts to terminal width.
- **Prompt prefix** — `❯` (bold cyan, two leading spaces) via `Console.input()`.
- **Response prefix** — `🤖` (bold green) with text aligned to the same column as the prompt.
- **Tool-call panels** — compact yellow `Panel` per tool invocation with `padding=(0, 1)`. Shows an icon (`🔍` analyze, `🔗` reconcile, `📊` report) and the tool's key parameters.
- **Turn separator** — single blank line; no decorative rules.

### Slash Commands

| Command  | Behavior                                         |
| -------- | ------------------------------------------------ |
| `/help`  | Show available slash commands                    |
| `/clear` | Reset the conversation and reprint the welcome banner |
| `/status`| Show current session status (loaded files, keys, etc.) |
| `/exit`  | Quit the CLI                                     |

The reconciliation summary table prints exactly once per session (guarded by a `SessionContext` sentinel flag) to avoid repetition across multi-turn conversations.

---

## 🧠 Session Context

Tools share data through a typed `SessionContext` dataclass in `tools/__init__.py`:

- **Typed fields** — DataFrames, file metadata, alignment metadata, detected keys, and reconciliation results are all explicit attributes (no more bare `dict`).
- **Module-level instance** — `SESSION_STATE` is the single source of truth.
- **`reset()`** — restores all fields to their declared defaults; used by the test conftest autouse fixture.
- **Duck-typed consumers** — `html_renderer` and `excel_renderer` accept any object with the same attributes, which makes parallel tests and ad-hoc report generation straightforward.

---

## ⚡ Performance

Several data-pipeline hot paths have been vectorized or parallelized:

- **Composite-key generation** — string concatenation done column-wise instead of `df.agg('|'.join, axis=1)` row loops.
- **Numeric comparison** — replaced Python row loops with column-wise `np.isclose` calls.
- **`normalize_dataframe`** — vectorized cell operations; aware of pandas 3.0+ `str` dtype via `pd.api.types.is_string_dtype`.
- **Datetime parsing** — 4,096-entry LRU cache on `pd.to_datetime` in `severity_classifier`.
- **Composite-key detection** — capped at 100 combos and restricted to indicator-named columns (`id`, `key`, `ref`, `seq`, ...).
- **File load + normalize** — `ThreadPoolExecutor` parallelizes both stages when both files are ≥ 5 MB.

---

## 🛡️ Guardrails & Data Normalization

- **Windows Emoji/Stream Guardrail**: Detects Windows platforms and wraps standard stream outputs in UTF-8 encoding to prevent `UnicodeEncodeError` crashes when printing symbols to the terminal.
- **File Ingestion Size Limit**: Warns the user if file sizes exceed 100MB before loading.
- **Key Normalization**: Normalizes IDs across different formats (e.g. aligning float string representation `'1.0'` and left-padded strings `'001'` to integer `'1'` for comparative lookup), while preserving the original raw display format in the report.
- **JSON Serialization Guardrail**: Sanitizes Pandas datatypes (e.g., `Timestamp`, `NaT`) and Numpy numeric primitives into Python-standard serializable types, avoiding runtime failures in Jinja2 JavaScript blocks.
- **Noisy Log Suppression**: Console logging handler level defaults to `WARNING` to keep the interactive terminal CLI quiet, while full `INFO`-level operational details are written cleanly to `logs/app.log`.
- **Rotating Log Files**: `RotatingFileHandler` keeps the most recent 5 log files at 10 MB each, preventing unbounded log growth.
- **Report Retention**: Old reports in `reports/` are auto-cleaned after `REPORT_RETENTION_DAYS` (default 7) when a new report is generated.
- **Scope Restriction**: The agent's system prompt restricts it to reconciliation topics only and explicitly declines general questions.
- **Clickable Report Links**: Generated report paths are emitted as `file:///` URIs so terminals and editors can open them with one click.

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

### [2026-06-05 12:30:00 PM] - v1.1.0: TUI Modernization & Code-Quality Hardening
- **TUI**: Full-width welcome banner with ASCII art and side-by-side metadata, `❯` / `🤖` aligned prompt/response markers, compact tool-call panels, slash commands (`/help`, `/clear`, `/status`, `/exit`), table-once-per-session via a `SessionContext` sentinel.
- **Architecture**: Replaced the global `SESSION_STATE` dict with a typed `SessionContext` dataclass; `reset()` rebuilds from defaults. Report renderers accept any duck-typed context object.
- **Performance**: Vectorized reconciliation (`np.isclose`), key generation, and `normalize_dataframe` (pandas 3.0+ `str` dtype aware). 4K-entry LRU cache for `pd.to_datetime`. Composite-key detection capped at 100 combos, restricted to indicator names. Parallel source/target load + normalize with `ThreadPoolExecutor` for files ≥ 5 MB.
- **Packaging & Ops**: `pyproject.toml` with `reconagent` console-script entry point. Pinned `requirements.txt` ranges. `RotatingFileHandler` (10 MB × 5). Auto-cleanup of old reports (default 7-day retention).
- **Agent**: System prompt restricts scope to reconciliation, parses both file paths from a single message, and auto-proceeds through the full 8-step workflow on "compare/reconcile".
- **Ingestion**: TXT and Parquet (optional) ingestion paths added.
- **Tests**: 25/25 passing. Added Excel report integration test.

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
