"""Handoff and escalation tool factory.

Each specialist agent gets handoff tools to transfer to peer agents
and an escalation tool to raise issues to human support.

The key mechanism: tools return ``Command(goto=..., graph=Command.PARENT)``
which signals the parent graph to route to the target node, breaking out
of the current agent's ReAct subgraph.

Uses ``ToolRuntime`` (available in langgraph >= 1.0) for injected state
and tool_call_id — no ``Annotated`` wrappers needed.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from src.state import AgentName

# Each specialist's handoff descriptions — tailored for the LLM.
HANDOFF_DESCRIPTIONS: dict[AgentName, str] = {
    "order_agent": (
        "Transfer to the ORDER SPECIALIST. Use this when the customer wants to: "
        "look up an order by ID, see all their orders, check order status, "
        "cancel an order, or ask about order details like price or items."
    ),
    "logistics_agent": (
        "Transfer to the LOGISTICS SPECIALIST. Use this when the customer wants to: "
        "track a package, get delivery estimates, find a shipment by order ID, "
        "report a lost package, or ask 'where is my order/package'."
    ),
    "refund_agent": (
        "Transfer to the REFUND SPECIALIST. Use this when the customer wants to: "
        "check refund eligibility, process a refund, check refund status, "
        "return an item, or asks 'can I get my money back'."
    ),
}


def make_handoff_tool(*, agent_name: AgentName):
    """Create a handoff tool that transfers control to *agent_name*.

    Uses ``ToolRuntime`` to access current graph state and tool_call_id.
    When called, returns ``Command(goto=..., graph=Command.PARENT)``
    which propagates up to the parent graph for routing.
    """
    tool_name = f"transfer_to_{agent_name}"
    description = HANDOFF_DESCRIPTIONS.get(agent_name, f"Transfer to {agent_name}.")

    @tool(tool_name, description=description)
    async def handoff(
        task_summary: Annotated[
            str,
            "A clear summary of what the next agent needs to handle. "
            "Include all relevant context: order IDs, tracking numbers, "
            "customer IDs, and what question the customer is asking.",
        ],
        runtime: ToolRuntime,
    ) -> Command:
        state = runtime.state

        # Use AIMessage (not ToolMessage) to avoid message-ordering issues
        # when the target agent's ReAct loop sees the tool message without
        # proper tool-call pairing in its API request context.
        handoff_msg = AIMessage(
            content=f"[Handoff to {agent_name}] Context: {task_summary}",
            name="handoff",
        )
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={
                "messages": [handoff_msg],
                "active_agent": agent_name,
                "retry_count": state.get("retry_count", 0) + 1,
            },
        )

    return handoff


def make_escalation_tool():
    """Create an escalation tool that transfers to human support.

    When the LLM calls this tool, control jumps to the human_agent node
    which creates a support ticket and resolves the conversation.
    """

    @tool(
        "escalate_to_human",
        description=(
            "Escalate the current issue to a HUMAN SUPPORT AGENT. "
            "Use this when: (1) you cannot resolve the issue after trying, "
            "(2) the customer explicitly demands to speak to a human, "
            "(3) the situation involves fraud, safety, or legal concerns, "
            "or (4) the refund amount is unusually large and needs manual approval. "
            "Always provide a detailed reason so the human agent has full context."
        ),
    )
    async def escalate(
        reason: Annotated[
            str,
            "Detailed reason for escalation. Include: what the customer wanted, "
            "what you tried, and why it needs human intervention.",
        ],
        runtime: ToolRuntime,
    ) -> Command:
        handoff_msg = AIMessage(
            content=f"[Escalating to human support] Reason: {reason}",
            name="escalation",
        )
        return Command(
            goto="human_agent",
            graph=Command.PARENT,
            update={
                "messages": [handoff_msg],
                "active_agent": "human_agent",
                "escalation_reason": reason,
            },
        )

    return escalate


def build_handoff_tools_for(agent_name: AgentName) -> list:
    """Return the full tool set: handoff to every *other* specialist + escalate.

    An agent never gets a ``transfer_to_self`` — that would be a no-op loop.
    """
    all_specialists: list[AgentName] = [
        "order_agent",
        "logistics_agent",
        "refund_agent",
    ]
    peers = [a for a in all_specialists if a != agent_name]

    tools = [make_handoff_tool(agent_name=peer) for peer in peers]
    tools.append(make_escalation_tool())
    return tools
