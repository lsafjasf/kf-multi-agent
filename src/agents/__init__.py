"""Specialist agent nodes and subgraphs.

Provides ``build_specialist_agent`` — the shared factory used to construct
order, logistics, and refund ReAct subgraphs.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph

from src.db.registry import set_db
from src.db.connection import DatabaseManager
from src.tools.handoff import build_handoff_tools_for
from src.state import AgentName


async def build_specialist_agent(
    *,
    model: BaseChatModel,
    db: DatabaseManager,
    agent_name: AgentName,
    system_prompt: str,
    domain_tools: list[BaseTool],
) -> CompiledStateGraph:
    """Build a specialist agent as a ReAct subgraph.

    This is the shared factory for order, logistics, and refund agents.
    Each agent gets:
    - Domain tools (order/logistics/refund operations)
    - Handoff tools (transfer to peer agents + escalate to human)

    Parameters
    ----------
    model : BaseChatModel
        The LLM to power the ReAct loop.
    db : DatabaseManager
        The database connection (injected into tool modules via ``set_db``).
    agent_name : AgentName
        The name of this agent (used for naming the subgraph and computing
        the correct set of peer handoff tools).
    system_prompt : str
        The agent's system prompt with tool descriptions and workflow rules.
    domain_tools : list[BaseTool]
        The tools this agent owns (e.g., order query, tracking, refund check).
    """
    set_db(db)

    handoff_tools = build_handoff_tools_for(agent_name)

    return create_react_agent(
        model=model,
        tools=domain_tools + handoff_tools,
        prompt=system_prompt,
        name=agent_name,
    )
