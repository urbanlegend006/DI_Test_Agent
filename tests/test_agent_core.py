"""Tests for ReconciliationAgentWrapper core logic.

Tests the wrapper class independently from the agent initialization
to verify backward compatibility, config handling, and error resilience.
"""
import pytest
from unittest.mock import MagicMock, patch

from langchain_core.prompts import ChatPromptTemplate
from agent import ReconciliationAgentWrapper, SYSTEM_PROMPT


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
    """When no config is passed, a default thread_id should be used."""
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [MagicMock(content="ok")]}

    wrapper = ReconciliationAgentWrapper(
        graph=mock_graph,
        tools=[],
        system_prompt=""
    )
    result = wrapper.invoke({"input": "hello"})

    assert result["output"] == "ok"
    # Verify graph was called with a config containing thread_id
    _call_args = mock_graph.invoke.call_args
    config_arg = _call_args[1]["config"]
    assert config_arg["configurable"]["thread_id"] == "reconciliation-session"


def test_invoke_does_not_mutate_caller_config():
    """The caller's config dict should not be modified after invoke."""
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
