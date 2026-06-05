import logging
import uuid
from typing import Any, Protocol, runtime_checkable
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from config import OPENAI_MODEL, OPENAI_API_KEY, MEMORY_WINDOW_SIZE
from utils.prompts import get_system_prompt, PROMPT_VERSIONS
from tools import SESSION_STATE
from tools.analyze_files import analyze_files
from tools.run_reconciliation import run_reconciliation
from tools.generate_report import generate_report

logger = logging.getLogger("reconciliation_agent.agent")

# Backward-compatible alias for existing tests.
SYSTEM_PROMPT = PROMPT_VERSIONS["v1"]


@runtime_checkable
class AgentGraph(Protocol):
    """Minimal protocol for the compiled agent graph."""
    def invoke(self, state: dict, config: dict | None = None) -> dict: ...


class ReconciliationAgentWrapper:
    """Wrapper to maintain backward compatibility with AgentExecutor's interface."""
    def __init__(self, graph: AgentGraph, tools: list[Any], system_prompt: str):
        self.graph = graph
        self.tools = tools
        self.system_prompt = system_prompt

    @property
    def agent(self):
        return self

    def get_prompts(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt)
        ])
        return [prompt]

    @staticmethod
    def _current_thread_id() -> str:
        """Return the thread ID from ``SESSION_STATE`` or generate a fresh one."""
        tid: str = getattr(SESSION_STATE, "session_thread_id", None) or uuid.uuid4().hex
        return tid

    @staticmethod
    def _session_metadata() -> dict[str, str]:
        """Build deterministic response metadata from the current session state."""
        summary = {}
        if SESSION_STATE.reconciliation_results:
            summary = SESSION_STATE.reconciliation_results.get("summary", {})
        return {
            "workflow_state": SESSION_STATE.workflow_state,
            "matched": str(summary.get("matched_rows", 0)),
            "mismatched": str(summary.get("mismatched_rows", 0)),
            "missing_in_source": str(summary.get("missing_in_source", 0)),
            "missing_in_target": str(summary.get("missing_in_target", 0)),
            "report_path": SESSION_STATE.report_path or "none",
            "report_format": SESSION_STATE.report_format or "none",
        }

    @staticmethod
    def _requested_report_format(input_dict: dict) -> str:
        """Infer requested report format from structured input or user text."""
        explicit = str(input_dict.get("report_format") or "").strip().lower()
        if explicit in {"html", "excel"}:
            return explicit

        user_text = str(input_dict.get("input") or "").lower()
        if "excel" in user_text or ".xlsx" in user_text:
            return "excel"
        return "html"

    @staticmethod
    def _build_user_content(input_dict: dict) -> str:
        """Create the HumanMessage content, preserving structured path hints."""
        user_input = str(input_dict.get("input"))
        source_path = input_dict.get("source_path")
        target_path = input_dict.get("target_path")
        if source_path and target_path:
            return (
                "User request:\n"
                f"{user_input}\n\n"
                "Structured inputs:\n"
                f"source_path: {source_path}\n"
                f"target_path: {target_path}"
            )
        return user_input

    def _generate_default_report_if_needed(self, input_dict: dict) -> str | None:
        """Generate a report after reconciliation when one does not exist yet."""
        if SESSION_STATE.reconciliation_results is None:
            return None
        if SESSION_STATE.report_path:
            return None

        fmt = self._requested_report_format(input_dict)
        output_dir = input_dict.get("output_dir")
        args: dict[str, Any] = {"format": fmt}
        if output_dir:
            args["output_dir"] = str(output_dir)
        return generate_report.invoke(args)

    def invoke(self, input_dict: dict, config: dict[str, Any] | None = None) -> dict:
        config = dict(config) if config is not None else {}

        thread_id = self._current_thread_id()
        turn_count: int = getattr(SESSION_STATE, "_turn_count", 0)
        generation: int = getattr(SESSION_STATE, "_session_generation", 0)

        if turn_count >= MEMORY_WINDOW_SIZE:
            generation += 1
            SESSION_STATE._session_generation = generation
            SESSION_STATE._turn_count = 0
            turn_count = 0

        effective_thread_id = f"{thread_id}-gen{generation}"
        config.setdefault("configurable", {})
        config["configurable"].setdefault("thread_id", effective_thread_id)

        user_input = input_dict.get("input")
        if user_input is None:
            return {
                "output": "Error: 'input' key is missing from the request.",
                "tool_calls": [],
                "metadata": self._session_metadata(),
            }

        user_msg = HumanMessage(content=self._build_user_content(input_dict))
        state = {"messages": [user_msg]}

        try:
            response_state = self.graph.invoke(state, config=config)
            messages = response_state.get("messages", [])
            if not messages:
                return {"output": "Error: The agent returned an empty response.", "tool_calls": []}

            tool_calls_list = []
            for msg in messages:
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        tool_calls_list.append({
                            "name": tc.get("name", "unknown"),
                            "args": tc.get("args", {}),
                            "id": tc.get("id", ""),
                        })

            last_msg = messages[-1]
            output = str(last_msg.content) if last_msg.content else ""

            report_output = self._generate_default_report_if_needed(input_dict)
            if report_output:
                output = f"{output}\n\n{report_output}" if output else report_output

            SESSION_STATE._turn_count = turn_count + 1

            result: dict[str, Any] = {
                "output": output,
                "tool_calls": tool_calls_list,
                "metadata": self._session_metadata(),
            }
            return result

        except KeyError as e:
            logger.error("Missing expected key in graph response: %s", e)
            return {
                "output": f"Error: The agent encountered an internal data error ({e}).",
                "tool_calls": [],
                "metadata": self._session_metadata(),
            }
        except Exception as e:
            logger.exception("Unexpected error during graph invocation")
            return {
                "output": f"Error: An unexpected error occurred: {e}",
                "tool_calls": [],
                "metadata": self._session_metadata(),
            }


def get_reconciliation_agent() -> ReconciliationAgentWrapper:
    """Configures and returns the LangChain tool-calling Agent graph wrapper."""
    logger.info("Initializing LangChain reconciliation agent using model %s", OPENAI_MODEL)

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY
    )

    system_prompt = get_system_prompt()
    tools = [analyze_files, run_reconciliation, generate_report]
    checkpointer = MemorySaver()

    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer
    )

    executor = ReconciliationAgentWrapper(graph, tools, system_prompt)

    SESSION_STATE.session_thread_id = uuid.uuid4().hex
    SESSION_STATE._turn_count = 0
    SESSION_STATE._session_generation = 0

    logger.info("Agent graph successfully initialized.")
    return executor
