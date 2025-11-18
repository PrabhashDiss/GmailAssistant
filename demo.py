from agentkernel.cli import CLI
from agentkernel.langgraph import LangGraphModule
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor

from custom_agent import CustomAgent
from langgraph.prebuilt import create_react_agent
import json
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

model = ChatOpenAI(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    base_url="https://api.groq.com/openai/v1",
)


async def get_mcp_tools():
    # Load MCP configuration
    with open("config.json", "r") as f:
        config = json.load(f)

    # Initialize MCP client and get tools
    mcp_client = MultiServerMCPClient(config["mcpServers"])
    return await mcp_client.get_tools()

# Get MCP tools synchronously
mcp_tools = asyncio.run(get_mcp_tools())

# Gmail agent: Handles Gmail tasks via MCP tools
# Uses LangGraph's ReAct framework and MCP toolset for Gmail operations
gmail_agent = create_react_agent(
    name="gmail",
    tools=mcp_tools,
    model=model,
    prompt="You are a Gmail assistant. Use the MCP tools to read, compose, send, and manage emails. Ask clarifying questions only when necessary. Refuse non-Gmail unrelated tasks.",
)

# General agent: General agent for all queries
# Uses custom implementation to provide detailed explanations
general_agent = CustomAgent(
    name="general",
    description="Agent for general questions",
    model=model,
    system_prompt="You provide assistance with general queries. Give short and direct answers.",
    tool_functions=mcp_tools,
).graph

# LangGraph's inbuilt supervisor agent: Coordinates between math and general agents
# Routes queries to the appropriate specialized agent based on the question type
triage_agent = create_supervisor(
    model=model,
    agents=[gmail_agent, general_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a Gmail agent. Assign Gmail-related tasks (read, compose, send, manage emails) to this agent\n"
        "- a general agent. Assign general tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself. \n"
        "Display the response without asking follow up questions."
    ),
).compile(name="triage")

LangGraphModule([triage_agent, gmail_agent, general_agent])

if __name__ == "__main__":
    CLI.main()
