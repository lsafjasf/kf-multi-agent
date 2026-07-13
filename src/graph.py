"""Graph assembly — builds the complete customer service StateGraph.

This is the central orchestration file. It:
1. Creates the LLM model.
2. Builds each specialist agent as a subgraph (via create_react_agent).
3. Registers all nodes on a parent StateGraph.
4. Wires up the routing: supervisor → specialists → END / human / supervisor.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.state import CustomerServiceState
from src.config import Config
from src.supervisor import build_supervisor_node
from src.agents.order_agent import build_order_agent
from src.agents.logistics_agent import build_logistics_agent
from src.agents.refund_agent import build_refund_agent
from src.agents.human_agent import human_agent_node, set_db as human_set_db
from src.db.connection import DatabaseManager


# ── Post-specialist routing ────────────────────────────────────────
def route_after_specialist(state: CustomerServiceState) -> str:
    """Decide what happens after a specialist subgraph completes normally.

    This function fires ONLY when a specialist finished WITHOUT issuing a
    Command(goto=...) (i.e., the LLM produced a final answer and stopped).

    Handoff/Escalation via Command.PARENT bypasses this entirely — control
    jumps directly to the target node.
    """
    # Case 1: Already resolved (e.g., human_agent ran)
    if state.resolved:
        return "end"

    # Case 2: Specialist stopped but marked for escalation
    if state.escalation_reason:
        return "human_agent"

    # Case 3: Too many retries → escalate
    if state.retry_count >= state.max_retries:
        return "human_agent"

    # Case 4: Specialist finished normally → conversation resolved
    return "end"


# ── Graph builder ───────────────────────────────────────────────────
async def build_customer_service_graph(
    model,
    db: DatabaseManager,
) -> CompiledStateGraph:
    """Assemble and compile the full customer service graph.

    Returns a compiled graph ready for ``ainvoke()`` / ``astream()``.
    """

    # Wire DB for human_agent
    human_set_db(db)

    # ── Build specialist subgraphs ──────────────────────────────
    order_graph = await build_order_agent(model, db)
    logistics_graph = await build_logistics_agent(model, db)
    refund_graph = await build_refund_agent(model, db)

    # ── Build supervisor node ───────────────────────────────────
    supervisor_node = build_supervisor_node(model)

    # ── Assemble parent graph ───────────────────────────────────
    builder = StateGraph(CustomerServiceState)

    # Register nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("order_agent", order_graph)
    builder.add_node("logistics_agent", logistics_graph)
    builder.add_node("refund_agent", refund_graph)
    builder.add_node("human_agent", human_agent_node)

    # Entry point
    builder.add_edge(START, "supervisor")

    # After each specialist completes normally (no Command), evaluate next step.
    # The specialist agent nodes use subgraphs — when the subgraph finishes
    # naturally (ReAct loop ends with a final message), this conditional
    # edge fires to decide: END, escalate, or re-route.
    for agent_name in ("order_agent", "logistics_agent", "refund_agent"):
        builder.add_conditional_edges(
            agent_name,
            route_after_specialist,
            {
                "end": END,
                "supervisor": "supervisor",
                "human_agent": "human_agent",
            },
        )

    # Human agent always goes to END after creating a ticket
    builder.add_edge("human_agent", END)

    return builder.compile(checkpointer=MemorySaver())
