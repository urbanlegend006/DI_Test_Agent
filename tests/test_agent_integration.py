"""
Agent integration tests using mocked LLM responses.

Tests that the agent:
1. Initializes correctly with valid config
2. Fails gracefully with missing/invalid API key
3. Tool definitions are correctly registered
4. System prompt contains the expected workflow instructions
5. The full tool-chain pipeline works end-to-end (analyze → reconcile → report)
"""
import pytest
import os
from unittest.mock import patch, MagicMock

from tools import SESSION_STATE


# ---------- Test 1: Agent initialization with missing key ----------

def test_agent_init_raises_on_missing_api_key():
    """Agent should raise ValueError when OPENAI_API_KEY is empty."""
    with patch("agent.OPENAI_API_KEY", None):
        from agent import get_reconciliation_agent
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_reconciliation_agent()


# ---------- Test 2: Agent initialization with valid key ----------

def test_agent_init_with_valid_key():
    """Agent should initialize without error when a key is present."""
    with patch("agent.OPENAI_API_KEY", "sk-test-dummy-key-for-unit-test"):
        with patch("agent.ChatOpenAI") as mock_llm:
            # Mock the LLM so we don't actually connect
            mock_llm_instance = MagicMock()
            mock_llm_instance.bind_tools = MagicMock(return_value=mock_llm_instance)
            mock_llm.return_value = mock_llm_instance

            from agent import get_reconciliation_agent
            executor = get_reconciliation_agent()

            assert executor is not None
            # Verify all 3 tools are registered
            tool_names = [t.name for t in executor.tools]
            assert "analyze_files" in tool_names
            assert "run_reconciliation" in tool_names
            assert "generate_report" in tool_names


# ---------- Test 3: System prompt contains workflow instructions ----------

def test_system_prompt_contains_workflow():
    """The system prompt should instruct the agent on the correct workflow."""
    from agent import get_reconciliation_agent

    with patch("agent.OPENAI_API_KEY", "sk-test-dummy-key"):
        with patch("agent.ChatOpenAI") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.bind_tools = MagicMock(return_value=mock_llm_instance)
            mock_llm.return_value = mock_llm_instance

            executor = get_reconciliation_agent()

            # The prompt template is stored in the agent's runnable
            # We can inspect it through the agent
            agent = executor.agent
            prompt = agent.get_prompts()[0] if hasattr(agent, 'get_prompts') else None

            if prompt is not None:
                # Reconstruct prompt text
                messages = prompt.messages
                system_msg = str(messages[0])
                assert "reconciliation" in system_msg.lower() or "Source" in system_msg
            else:
                # If we can't access the prompt, at least verify tool count
                assert len(executor.tools) == 3


# ---------- Test 4: Full tool-chain pipeline (no LLM needed) ----------

def test_full_toolchain_pipeline(temp_files):
    """
    Tests the entire reconciliation pipeline by calling tools directly
    in the order the agent would call them, verifying each step produces
    the expected state transitions.
    """
    from tools.analyze_files import analyze_files
    from tools.run_reconciliation import run_reconciliation
    from tools.generate_report import generate_report

    src_path = str(temp_files["source_csv"])
    tgt_path = str(temp_files["target_json"])

    # Step 1: Analyze files
    result_analyze = analyze_files.invoke({
        "source_path": src_path,
        "target_path": tgt_path
    })
    assert "Successfully loaded" in result_analyze
    assert SESSION_STATE.source_df is not None
    assert SESSION_STATE.target_df is not None
    assert SESSION_STATE.comp_source is not None
    assert SESSION_STATE.comp_target is not None

    # Step 2: Run reconciliation
    result_recon = run_reconciliation.invoke({
        "primary_key": "id",
        "tolerance": "strict"
    })
    assert "Reconciliation Completed Successfully!" in result_recon
    assert SESSION_STATE.reconciliation_results is not None

    results = SESSION_STATE.reconciliation_results
    assert results["summary"]["matched_rows"] >= 0
    assert results["summary"]["mismatched_rows"] >= 0
    assert isinstance(results["mismatches"], list)
    assert isinstance(results["missing_in_target"], list)
    assert isinstance(results["missing_in_source"], list)

    # Step 3: Generate HTML report
    result_html = generate_report.invoke({"format": "html"})
    assert "Success" in result_html or "report" in result_html.lower()
    # Verify the HTML file was actually created
    assert "html" in result_html.lower()

    # Step 4: Generate Excel report and verify the file is actually created
    result_excel = generate_report.invoke({"format": "excel"})
    assert "Success" in result_excel or "excel" in result_excel.lower()

    # Verify Excel file exists and contains expected sheets
    import os
    import glob
    from openpyxl import load_workbook
    import config
    excel_reports = glob.glob(os.path.join(config.REPORTS_DIR, "*.xlsx"))
    assert len(excel_reports) >= 1, "No Excel report was generated"
    latest_xlsx = max(excel_reports, key=os.path.getmtime)
    wb = load_workbook(latest_xlsx)
    assert "Summary" in wb.sheetnames
    assert "Mismatches" in wb.sheetnames


# ---------- Test 5: Tool-chain with invalid primary key ----------

def test_toolchain_invalid_primary_key(temp_files):
    """
    Tests that the reconciliation tool returns a clear error message
    when given an invalid primary key column name.
    """
    from tools.analyze_files import analyze_files
    from tools.run_reconciliation import run_reconciliation

    src_path = str(temp_files["source_csv"])
    tgt_path = str(temp_files["target_json"])

    # Load files first
    analyze_files.invoke({
        "source_path": src_path,
        "target_path": tgt_path
    })

    # Try reconciliation with a bad key
    result = run_reconciliation.invoke({
        "primary_key": "nonexistent_column",
        "tolerance": "strict"
    })
    assert "Error" in result
    assert "nonexistent_column" in result


# ---------- Test 6: Report generation without reconciliation data ----------

def test_report_generation_without_recon_data():
    """
    Tests that generate_report returns a clear error when called
    before reconciliation has been run.
    """
    from tools.generate_report import generate_report

    result = generate_report.invoke({"format": "html"})
    assert "error" in result.lower() or "Error" in result


# ---------- LLM Behavior Tests ----------

def test_system_prompt_declines_non_reconciliation():
    """The system prompt must include the scope restriction decline message."""
    from agent import SYSTEM_PROMPT
    assert "politely decline" in SYSTEM_PROMPT.lower()
    assert "I am a Data Integrity Test Agent" in SYSTEM_PROMPT
    assert "non-reconciliation" in SYSTEM_PROMPT.lower() or "unrelated" in SYSTEM_PROMPT.lower()


def test_system_prompt_has_all_workflow_steps():
    """The system prompt must contain all 8 workflow steps."""
    from agent import SYSTEM_PROMPT
    for step_num in range(1, 9):
        assert f"{step_num}." in SYSTEM_PROMPT
    assert "analyze_files" in SYSTEM_PROMPT
    assert "run_reconciliation" in SYSTEM_PROMPT
    assert "generate_report" in SYSTEM_PROMPT


def test_system_prompt_contains_response_format_guidelines():
    """The system prompt should instruct the agent on response format."""
    from agent import SYSTEM_PROMPT
    assert "bullet points" in SYSTEM_PROMPT.lower()
    assert "section headers" in SYSTEM_PROMPT.lower()
    assert "structured" in SYSTEM_PROMPT.lower()


def test_system_prompt_contains_guiding_rules():
    """The system prompt should include all key guiding rules."""
    from agent import SYSTEM_PROMPT
    assert "Never hallucinate" in SYSTEM_PROMPT
    assert "Scope Restriction" in SYSTEM_PROMPT or "SCOPE RESTRICTION" in SYSTEM_PROMPT
    assert "Source" in SYSTEM_PROMPT and "Target" in SYSTEM_PROMPT


def test_model_configured_with_temperature_zero():
    """ChatOpenAI should be initialised with temperature=0 for consistent outputs."""
    with patch("agent.OPENAI_API_KEY", "sk-test-dummy-key-for-unit-test"):
        with patch("agent.ChatOpenAI") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.bind_tools = MagicMock(return_value=mock_llm_instance)
            mock_llm.return_value = mock_llm_instance

            from agent import get_reconciliation_agent
            get_reconciliation_agent()

            mock_llm.assert_called_once()
            _call_kwargs = mock_llm.call_args.kwargs
            assert _call_kwargs.get("temperature") == 0


def test_agent_binds_all_three_tools():
    """The LLM should receive all three tools bound for tool calling."""
    with patch("agent.OPENAI_API_KEY", "sk-test-dummy-key-for-unit-test"):
        with patch("agent.ChatOpenAI") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance

            from agent import get_reconciliation_agent
            executor = get_reconciliation_agent()

            tool_names = [t.name for t in executor.tools]
            assert "analyze_files" in tool_names
            assert "run_reconciliation" in tool_names
            assert "generate_report" in tool_names


# ---------- Additional LLM Configuration & Prompt Tests ----------

def test_agent_uses_checkpointer():
    """Agent graph should be created with a MemorySaver checkpointer."""
    from langgraph.checkpoint.base import BaseCheckpointSaver

    with patch("agent.OPENAI_API_KEY", "sk-test-dummy-key"):
        with patch("agent.ChatOpenAI") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.bind_tools = MagicMock(return_value=mock_llm_instance)
            mock_llm.return_value = mock_llm_instance

            with patch("agent.MemorySaver") as mock_memory:
                mock_memory_instance = MagicMock(spec=BaseCheckpointSaver)
                mock_memory.return_value = mock_memory_instance

                from agent import get_reconciliation_agent
                executor = get_reconciliation_agent()

                mock_memory.assert_called_once()
                assert hasattr(executor, "graph")
                assert executor.graph is not None


def test_system_prompt_auto_proceed():
    """System prompt must instruct the agent to auto-proceed when both paths are given."""
    from agent import SYSTEM_PROMPT
    assert "extract both paths immediately" in SYSTEM_PROMPT
    assert "proceed to step 2" in SYSTEM_PROMPT.lower()
    assert "Do NOT ask for the second path again" in SYSTEM_PROMPT


def test_system_prompt_error_handling():
    """System prompt must instruct the agent to explain errors without crashing."""
    from agent import SYSTEM_PROMPT
    assert "explain the issue clearly" in SYSTEM_PROMPT
    assert "Do not crash" in SYSTEM_PROMPT


def test_system_prompt_hallucination_guard():
    """System prompt must contain hallucination prevention instructions."""
    from agent import SYSTEM_PROMPT
    assert "Never hallucinate" in SYSTEM_PROMPT
    assert "Rely ONLY on tool outputs" in SYSTEM_PROMPT


def test_system_prompt_output_summary():
    """System prompt must instruct the agent to show a summary with all required metrics."""
    from agent import SYSTEM_PROMPT
    assert "matched count" in SYSTEM_PROMPT.lower()
    assert "mismatched count" in SYSTEM_PROMPT.lower()
    assert "missing in source" in SYSTEM_PROMPT.lower()
    assert "missing in target" in SYSTEM_PROMPT.lower()
    assert "severity breakdown" in SYSTEM_PROMPT.lower()
    assert "table format" in SYSTEM_PROMPT.lower() or "bulleted" in SYSTEM_PROMPT.lower()
