"""Tests for ReconciliationAgentWrapper core logic.

Tests the wrapper class independently from the agent initialization
to verify backward compatibility, config handling, error resilience,
memory windowing, tool-call observability, and deterministic session metadata.
"""
import uuid
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from agent import ReconciliationAgentWrapper, SYSTEM_PROMPT
from tools import SESSION_STATE


# ---------- Wrapper initialization & backward compatibility ----------

def test_wrapper_agent_property():
    """The ``agent`` property should return ``self`` for backward compatibility."""
    wrapper = ReconciliationAgentWrapper(
        graph=MagicMock(),
        tools=[],
        system_prompt="test"
    )
    assert wrapper.agent is wrapper


def test_wrapper_get_prompts():
    """``get_prompts()`` should return a list with one ChatPromptTemplate."""
    wrapper = ReconciliationAgentWrapper(
        graph=MagicMock(),
        tools=[],
        system_prompt="test prompt content"
    )
    prompts = wrapper.get_prompts()
    assert len(prompts) == 1
    assert isinstance(prompts[0], ChatPromptTemplate)
    # Verify the system message content matches (via template string)
    system_msg = str(prompts[0].messages[0])
    assert "test prompt content" in system_msg


# ---------- Config handling ----------

def test_invoke_default_config():
    """When no config is passed, the thread_id from SESSION_STATE should be used."""
    SESSION_STATE.reset()
    SESSION_STATE.session_thread_id = "test-session-id"

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [MagicMock(content="ok")]}

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )
    result = wrapper.invoke({"input": "hello"})

    assert result["output"] == "ok"
    _call_args = mock_graph.invoke.call_args
    config_arg = _call_args[1]["config"]
    assert config_arg["configurable"]["thread_id"] == "test-session-id-gen0"


def test_invoke_does_not_mutate_caller_config():
    """The caller's config dict should not be modified after invoke."""
    SESSION_STATE.reset()
    SESSION_STATE.session_thread_id = "test-session-id"

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [MagicMock(content="ok")]}

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )
    original_config = {"configurable": {"thread_id": "custom-session"}}
    config_copy = dict(original_config)

    wrapper.invoke({"input": "hello"}, config=original_config)

    assert original_config == config_copy, "Caller's config dict was mutated"


# ---------- Error resilience ----------

def test_invoke_missing_input_key():
    """Invoke with dict missing 'input' key should return a graceful error."""
    wrapper = ReconciliationAgentWrapper(
        graph=MagicMock(),
        tools=[],
        system_prompt=""
    )
    result = wrapper.invoke({"foo": "bar"})
    assert "Error" in result["output"]
    assert "input" in result["output"].lower()


def test_invoke_empty_messages_response():
    """When graph returns no messages, invoke should return a graceful error."""
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": []}

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )
    result = wrapper.invoke({"input": "hello"})
    assert "Error" in result["output"]
    assert "empty" in result["output"].lower()


def test_invoke_graph_raises_key_error():
    """When graph.invoke raises KeyError, invoke should return a graceful error."""
    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = KeyError("missing_field")

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )
    result = wrapper.invoke({"input": "hello"})
    assert "Error" in result["output"]
    assert "missing_field" in result["output"]


def test_invoke_graph_raises_generic_exception():
    """When graph.invoke raises an unexpected exception, invoke should return a graceful error."""
    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = RuntimeError("connection timeout")

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )
    result = wrapper.invoke({"input": "hello"})
    assert "Error" in result["output"]
    assert "connection timeout" in result["output"]


# ---------- SYSTEM_PROMPT constant ----------

def test_system_prompt_is_constant_string():
    """SYSTEM_PROMPT should be a non-empty string."""
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 100


# ---------- Full-turn roundtrip ----------

def test_invoke_full_turn_roundtrip():
    """Wrapper should route input through graph and return the last message content.

    Simulates a complete turn: user message -> tool call -> tool result ->
    final answer. The wrapper should extract only the final AI message.
    """
    mock_graph = MagicMock()
    messages = [
        HumanMessage(content="reconcile a.csv vs b.csv"),
        AIMessage(
            content="",
            tool_calls=[{"name": "analyze_files", "args": {}, "id": "call_1"}],
        ),
        AIMessage(content="Reconciliation complete! All rows matched."),
    ]
    mock_graph.invoke.return_value = {"messages": messages}

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )
    result = wrapper.invoke({"input": "reconcile a.csv vs b.csv"})

    assert result["output"] == "Reconciliation complete! All rows matched."

    # Verify graph was called with the right state structure
    _call_args = mock_graph.invoke.call_args
    state_arg = _call_args[0][0]
    assert "messages" in state_arg
    assert isinstance(state_arg["messages"][0], HumanMessage)
    assert "reconcile" in state_arg["messages"][0].content


def test_invoke_passes_structured_paths_to_model_message():
    """Structured source/target paths should be passed without bracket augmentation."""
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [AIMessage(content="ok")]}

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )
    wrapper.invoke({
        "input": "reconcile these",
        "source_path": "source.csv",
        "target_path": "target.json",
    })

    state_arg = mock_graph.invoke.call_args[0][0]
    content = state_arg["messages"][0].content
    assert "Structured inputs:" in content
    assert "source_path: source.csv" in content
    assert "target_path: target.json" in content
    assert "[Extracted paths:" not in content


# ---------- Response contract ----------

def test_invoke_response_always_has_output_key():
    """Invoke should always return a dict with an ``output`` key, even on errors."""
    scenarios = [
        ("missing input key", {"graph": MagicMock(), "input": {"foo": "bar"}}),
        ("empty messages", {
            "graph": MagicMock(**{"invoke.return_value": {"messages": []}}),
            "input": {"input": "hello"},
        }),
        ("graph raises KeyError", {
            "graph": MagicMock(**{"invoke.side_effect": KeyError("boom")}),
            "input": {"input": "hello"},
        }),
        ("graph raises generic exception", {
            "graph": MagicMock(**{"invoke.side_effect": RuntimeError("fail")}),
            "input": {"input": "hello"},
        }),
    ]

    for name, cfg in scenarios:
        wrapper = ReconciliationAgentWrapper(
            graph=cfg["graph"],
            tools=[],
            system_prompt=""
        )
        result = wrapper.invoke(cfg["input"])
        assert isinstance(result, dict), f"{name}: result is not a dict"
        assert "output" in result, f"{name}: missing 'output' key"
        assert isinstance(result["output"], str), f"{name}: output is not a string"
        assert len(result["output"]) > 0, f"{name}: output is empty"


# ---------- Non-string input handling ----------

def test_invoke_non_string_input():
    """Wrapper should coerce non-string ``input`` values into HumanMessage content."""
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [AIMessage(content="42")]}

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )

    result = wrapper.invoke({"input": 42})
    assert result["output"] == "42"

    # Verify the HumanMessage was created with str(42)
    _call_args = mock_graph.invoke.call_args
    msg = _call_args[0][0]["messages"][0]
    assert isinstance(msg, HumanMessage)
    assert msg.content == "42", "HumanMessage content should be stringified"


# ---------- Memory windowing ----------

def test_memory_window_resets_after_limit():
    """After MEMORY_WINDOW_SIZE turns, a new thread generation is started."""
    SESSION_STATE.reset()
    SESSION_STATE.session_thread_id = "mem-test-id"
    SESSION_STATE._turn_count = 0
    SESSION_STATE._session_generation = 0

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [MagicMock(content="ok")]}

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )

    with patch("agent.MEMORY_WINDOW_SIZE", 3):
        # First 3 turns use gen0
        for i in range(3):
            wrapper.invoke({"input": f"msg {i}"})
            _call_args = mock_graph.invoke.call_args
            tid = _call_args[1]["config"]["configurable"]["thread_id"]
            assert tid == "mem-test-id-gen0", f"Turn {i} should be gen0, got {tid}"

        # 4th turn bumps to gen1
        wrapper.invoke({"input": "msg 4th"})
        _call_args = mock_graph.invoke.call_args
        tid = _call_args[1]["config"]["configurable"]["thread_id"]
        assert tid == "mem-test-id-gen1"


def test_memory_window_turn_count_tracked():
    """Turn count is incremented on each invoke."""
    SESSION_STATE.reset()
    SESSION_STATE.session_thread_id = "count-test"
    SESSION_STATE._turn_count = 0

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [MagicMock(content="ok")]}

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )

    wrapper.invoke({"input": "first"})
    assert SESSION_STATE._turn_count == 1

    wrapper.invoke({"input": "second"})
    assert SESSION_STATE._turn_count == 2


# ---------- Tool-call observability ----------

def test_invoke_returns_tool_calls_list():
    """Invoke returns a ``tool_calls`` list with details of each tool call."""
    SESSION_STATE.reset()
    SESSION_STATE.session_thread_id = "tool-call-test"

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "analyze_files", "args": {"source_path": "a.csv"}, "id": "call_1"},
                    {"name": "run_reconciliation", "args": {"primary_key": "id"}, "id": "call_2"},
                ],
            ),
            AIMessage(content="Done."),
        ]
    }

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )

    result = wrapper.invoke({"input": "reconcile files"})
    assert "tool_calls" in result
    assert len(result["tool_calls"]) == 2
    assert result["tool_calls"][0]["name"] == "analyze_files"
    assert result["tool_calls"][1]["name"] == "run_reconciliation"


def test_invoke_empty_tool_calls_when_no_tools():
    """When no tool calls occur, tool_calls is an empty list."""
    SESSION_STATE.reset()
    SESSION_STATE.session_thread_id = "no-tools"

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "messages": [AIMessage(content="Just a text response.")]
    }

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )

    result = wrapper.invoke({"input": "hello"})
    assert result["tool_calls"] == []


# ---------- Deterministic session metadata ----------

def test_metadata_comes_from_session_state_not_output_block():
    """Metadata should be built from SESSION_STATE, not parsed from prose."""
    SESSION_STATE.reset()
    SESSION_STATE.session_thread_id = "meta-test"
    SESSION_STATE.workflow_state = "reported"
    SESSION_STATE.report_path = "/tmp/state-report.html"
    SESSION_STATE.report_format = "html"
    SESSION_STATE.reconciliation_results = {
        "summary": {
            "matched_rows": 42,
            "mismatched_rows": 3,
            "missing_in_source": 1,
            "missing_in_target": 2,
        }
    }

    output_text = (
        "Reconciliation complete!\n\n"
        "[METADATA]\n"
        "matched: 999\n"
        "report_path: /tmp/wrong.html\n"
        "[/METADATA]"
    )

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [AIMessage(content=output_text)]}

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )

    result = wrapper.invoke({"input": "done"})
    assert "metadata" in result
    assert result["metadata"]["matched"] == "42"
    assert result["metadata"]["mismatched"] == "3"
    assert result["metadata"]["report_path"] == "/tmp/state-report.html"


def test_metadata_returned_without_output_block():
    """Metadata is returned even when the model emits plain text."""
    SESSION_STATE.reset()
    SESSION_STATE.session_thread_id = "no-meta"

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [AIMessage(content="Simple text.")]}

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )

    result = wrapper.invoke({"input": "hi"})
    assert result["metadata"]["workflow_state"] == "awaiting_paths"
    assert result["metadata"]["report_path"] == "none"


def test_default_report_generated_after_reconciliation(tmp_path):
    """Wrapper generates a default report when reconciliation exists but no report is present."""
    SESSION_STATE.reset()
    SESSION_STATE.session_thread_id = "report-meta"
    SESSION_STATE.workflow_state = "reconciled"
    SESSION_STATE.source_filename = "source.csv"
    SESSION_STATE.target_filename = "target.csv"
    SESSION_STATE.source_fullpath = "/tmp/source.csv"
    SESSION_STATE.target_fullpath = "/tmp/target.csv"
    SESSION_STATE.primary_key_cols = ["id"]
    SESSION_STATE.reconciliation_results = {
        "summary": {
            "total_source_rows": 1,
            "total_target_rows": 1,
            "clean_source_rows": 1,
            "clean_target_rows": 1,
            "matched_rows": 1,
            "mismatched_rows": 0,
            "missing_in_source": 0,
            "missing_in_target": 0,
            "duplicate_source": 0,
            "duplicate_target": 0,
        },
        "mismatches": [],
        "missing_in_target": [],
        "missing_in_source": [],
        "duplicates_source": [],
        "duplicates_target": [],
        "col_mismatch_stats": {},
        "primary_keys": ["id"],
    }

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [AIMessage(content="Reconciliation complete.")]}

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )

    result = wrapper.invoke({"input": "go", "output_dir": str(tmp_path)})
    assert "Success!" in result["output"]
    assert result["metadata"]["workflow_state"] == "reported"
    assert result["metadata"]["report_path"].endswith(".html")


# ---------- Thread ID from SESSION_STATE ----------

def test_thread_id_generated_when_session_not_set():
    """When SESSION_STATE has no thread_id, a UUID is generated."""
    SESSION_STATE.reset()
    # Explicitly clear thread_id
    SESSION_STATE.session_thread_id = ""

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [MagicMock(content="ok")]}

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )

    result = wrapper.invoke({"input": "hello"})
    assert result["output"] == "ok"

    _call_args = mock_graph.invoke.call_args
    tid = _call_args[1]["config"]["configurable"]["thread_id"]
    # Should be a UUID hex string + "-gen0"
    assert tid.endswith("-gen0")
    assert len(tid) > 36  # UUID hex (32) + "-gen0" (5) = min 37
