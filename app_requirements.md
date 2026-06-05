**Role & Objective:**
Act as an Expert Python Developer and QA Automation Architect. Your task is to build a local CLI-based Data Reconciliation Test Agent. This application will function as an interactive terminal chatbot that communicates with the user, ingests local file paths, performs data integrity tests (row-by-row reconciliation), and generates detailed local reports.

**Technology Stack:**
*   **Core:** Python 3.10+
*   **AI Framework:** LangChain (using the latest Tool-Calling Agent architecture)
*   **LLM:** OpenAI API (`langchain-openai`)
*   **Data Processing:** `pandas` (crucial for parsing and standardizing Excel and JSON files into DataFrames for comparison), `openpyxl`
*   **Terminal UI/UX:** `rich` (for colored outputs, panels, markdown rendering, and progress spinners)
*   **Reporting:** `jinja2` (for HTML report templating) or standard `pandas` to Excel export.

**Core Requirements & Features:**

1.  **Interactive CLI Chatbot (The Interface):**
    *   Implement a continuous `while True` chat loop in the terminal.
    *   Use the `rich` library to create a visually distinct UI (e.g., different colors for User inputs, Agent thoughts, and System outputs).
    *   The agent must be able to maintain conversation history using LangChain's memory components.

2.  **Intelligent Data Parsing & Tooling:**
    *   Provide the LangChain agent with specific Python tools (decorated with `@tool`). 
    *   **Tool 1: `analyze_files`** - Takes file paths, determines their extensions (.xlsx, .json), and loads them into Pandas DataFrames. It should handle cross-format matching (e.g., Excel vs. JSON) by flattening JSON if necessary and aligning column headers.
    *   **Tool 2: `run_reconciliation`** - Performs a row-by-row comparison of the two DataFrames. It must identify missing rows, mismatched values, and duplicate entries based on a primary key (which the agent can ask the user for if not obvious).
    *   **Tool 3: `generate_report`** - Takes the reconciliation results and exports an in-depth HTML or Excel file to the local directory, returning the absolute file path.

3.  **Agent Behavior & Fallbacks:**
    *   If the user provides file paths, the agent should immediately acknowledge them and ask clarifying questions if needed (e.g., "What is the primary key column for the reconciliation?").
    *   If the agent is stuck, encounters a file parsing error, or lacks information to proceed, it must explicitly state this in the terminal without crashing. (e.g., "I encountered an error parsing the JSON file. It appears malformed. Please check the file or provide a new path.").
    *   Do not hallucinate file contents; strictly read from the provided local paths.

4.  **Output Format:**
    *   After running the test, the terminal output must show a clean, high-level summary using `rich.table` (e.g., Total Rows, Matched Rows, Mismatched Rows, Missing Rows).
    *   The agent must output a final message containing a clickable or easily copyable local link/path to the detailed HTML/Excel report.

**First Iteration Deliverables:**
Please generate the complete, modular Python code for this application. Structure the response ideally in separate files (e.g., `main.py`, `tools.py`, `agent.py`) or as a cohesive, well-commented single script if better suited for quick execution. Include a `requirements.txt` file. Focus heavily on ensuring the LangChain tool binding and the `rich` UI loop work flawlessly together.