import logging
from typing import Any, Protocol, runtime_checkable
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from config import OPENAI_MODEL, OPENAI_API_KEY
from tools.analyze_files import analyze_files
from tools.run_reconciliation import run_reconciliation
from tools.generate_report import generate_report

logger = logging.getLogger("reconciliation_agent.agent")


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
        """Return ``self`` for backward compatibility with ``AgentExecutor``."""
        return self

    def get_prompts(self):
        """Allows testing code that expects a prompt template structure."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt)
        ])
        return [prompt]

    def invoke(self, input_dict: dict, config: dict[str, Any] | None = None) -> dict:
        """Invoke the compiled agent graph with history-preserving memory.

        Accepts the caller's ``config`` dict without mutating it, injects a
        default ``thread_id`` when none is present, and wraps graph errors
        into graceful ``{"output": …}`` responses.

        Args:
            input_dict: Must contain an ``"input"`` key with the user message.
            config: Optional LangGraph runtime configuration dict.

        Returns:
            Dictionary with an ``"output"`` key containing the agent's text
            response or an error message.
        """
        config = dict(config) if config is not None else {}
        config.setdefault("configurable", {})
        config["configurable"].setdefault("thread_id", "reconciliation-session")

        user_input = input_dict.get("input")
        if user_input is None:
            return {"output": "Error: 'input' key is missing from the request."}

        user_msg = HumanMessage(content=user_input)
        state = {"messages": [user_msg]}

        try:
            response_state = self.graph.invoke(state, config=config)
            messages = response_state.get("messages", [])
            if not messages:
                return {"output": "Error: The agent returned an empty response."}
            last_msg = messages[-1]
            return {"output": last_msg.content}
        except KeyError as e:
            logger.error("Missing expected key in graph response: %s", e)
            return {"output": f"Error: The agent encountered an internal data error ({e})."}
        except Exception as e:
            logger.exception("Unexpected error during graph invocation")
            return {"output": f"Error: An unexpected error occurred: {e}"}

SYSTEM_PROMPT = (
    "You are an expert Data Reconciliation Test Agent.\n"
    "Your objective is to guide users to compare two data files (Excel, JSON, or CSV), "
    "identify mismatches, missing rows, or duplicate records, and generate clear reports.\n\n"
    "Workflow:\n"
    "1. If the user provides both file paths in a single message (e.g. "
    "\"reconcile 'source.xlsx' vs 'target.csv'\"), extract both paths immediately "
    "and proceed to step 2. Do NOT ask for the second path again.\n"
    "2. Call the `analyze_files` tool with Source and Target paths to load and understand file structures.\n"
    "3. Review the outputs. If a primary key was not auto-detected (or if multiple candidates were found), "
     "   prompt the user to specify or confirm the column(s) to use "
     "as the primary key. You can support composite keys (comma-separated).\n"
    "4. Ask the user if they want exact matching (strict) or fuzzy matching "
    "(with custom tolerances for numeric or date differences).\n"
    "5. Call the `run_reconciliation` tool with the primary key and tolerance settings.\n"
    "6. Display a detailed summary of the reconciliation findings to the user.\n"
    "7. Call the `generate_report` tool to create the final report. "
    "Offer the user a choice between HTML (default) or Excel.\n"
    "8. Provide the final absolute file path link clearly to the user.\n\n"
    "IMPORTANT: When the user says 'compare', 'reconcile', 'find differences', "
    "'match files', or similar, you MUST proceed through the FULL workflow "
    "(steps 1-8) automatically. Do NOT stop after analysis. "
    "Only pause for user input if critical information (like primary key) is "
    "truly ambiguous or missing.\n\n"
    "Guiding Rules:\n"
    "- Never hallucinate file contents or counts. Rely ONLY on tool outputs.\n"
    "- If an execution step fails or a file is malformed, explain the issue clearly. Do not crash.\n"
    "- Terminology: Use 'Source' for the first file and 'Target' for the second file.\n"
    "- SCOPE RESTRICTION: You are ONLY a Data Reconciliation Agent. "
    "You MUST politely decline any question or request unrelated to "
    "data file reconciliation, comparison, or integrity testing.\n"
    "- If asked about general knowledge, weather, colors, programming help, "
    "or any non-reconciliation topic, respond with: "
    "'I am a Data Integrity Test Agent. I can only help with reconciling "
    "data files (XLSX, CSV, JSON, TXT, Parquet). Please provide Source "
    "and Target file paths for reconciliation.'\n"
    "Response Format:\n"
    "- Always respond with structured, multi-line, human-readable text.\n"
    "- Use bullet points (•), bold counts, and clear section headers.\n"
    "- Preserve and reformat the detailed output from tools — never collapse "
    "it into a single sentence or line.\n"
    "- For analysis results: show file names, row counts, column counts, "
    "aligned columns list, and key candidates in a clear formatted structure.\n"
    "- For reconciliation results: show matched count, mismatched count, "
    "missing in source, missing in target, duplicates, and severity breakdown "
    "in a readable bulleted or table format.\n"
)

def get_reconciliation_agent() -> ReconciliationAgentWrapper:
    """Configures and returns the LangChain tool-calling Agent graph wrapper."""
    logger.info("Initializing LangChain reconciliation agent using model %s", OPENAI_MODEL)

    # 1. Define LLM
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY
    )

    # 2. Define prompt structure
    system_prompt = SYSTEM_PROMPT

    # 3. Setup tools
    tools = [analyze_files, run_reconciliation, generate_report]

    # 4. Setup checkpointer for memory
    checkpointer = MemorySaver()

    # 5. Create compiled agent graph
    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer
    )

    # 6. Wrap for AgentExecutor backward compatibility
    executor = ReconciliationAgentWrapper(graph, tools, system_prompt)

    logger.info("Agent graph successfully initialized.")
    return executor
