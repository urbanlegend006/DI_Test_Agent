import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment
load_dotenv(override=True)
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("No API Key found in env!")
    sys.exit(1)

# Import LangChain latest APIs
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# Import tools
sys.path.append(os.path.abspath("."))
from tools.analyze_files import analyze_files
from tools.run_reconciliation import run_reconciliation
from tools.generate_report import generate_report

print("Imports successful!")

# Setup model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=api_key
)

# Setup tools
tools = [analyze_files, run_reconciliation, generate_report]

# Setup checkpointer
checkpointer = MemorySaver()

# Define system prompt
system_prompt = (
    "You are an expert Data Reconciliation Test Agent. "
    "You help users compare two data files (Excel, JSON, or CSV) using analyze_files, run_reconciliation, and generate_report tools."
)

# Create Agent Graph
graph = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=checkpointer
)

print("Agent graph created successfully!")

# Define wrapper class
class ReconciliationAgentWrapper:
    def __init__(self, graph, tools, system_prompt):
        self.graph = graph
        self.tools = tools
        self.system_prompt = system_prompt
        self.agent = self

    def get_prompts(self):
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt)
        ])
        return [prompt]

    def invoke(self, input_dict: dict, config: dict = None) -> dict:
        if config is None:
            config = {"configurable": {"thread_id": "default-session"}}
        elif "configurable" not in config:
            config["configurable"] = {"thread_id": "default-session"}
        elif "thread_id" not in config["configurable"]:
            config["configurable"]["thread_id"] = "default-session"
            
        user_msg = HumanMessage(content=input_dict["input"])
        state = {"messages": [user_msg]}
        
        response_state = self.graph.invoke(state, config=config)
        last_msg = response_state["messages"][-1]
        
        return {"output": last_msg.content}

wrapper = ReconciliationAgentWrapper(graph, tools, system_prompt)
print("Wrapper created successfully!")

# Test invocation
res = wrapper.invoke({"input": "Hello! What tools do you have access to?"})
print("Agent Response:")
print(res["output"])
