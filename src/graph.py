"""Graph assembly — builds the complete customer service StateGraph.

This is the central orchestration file. It:
1. Creates the LLM model.
2. Builds each specialist agent as a subgraph (via create_react_agent).
3. Registers all nodes on a parent StateGraph.
4. Wires up the routing: supervisor → specialists → END / human / supervisor.
5. Hooks in the three-layer memory system (L1/L2/L3).
"""

from __future__ import annotations

import aiosqlite
from pathlib import Path
from typing import Union

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.state import CustomerServiceState
from src.supervisor import build_supervisor_node
from src.agents.order_agent import build_order_agent
from src.agents.logistics_agent import build_logistics_agent
from src.agents.refund_agent import build_refund_agent
from src.agents.human_agent import human_agent_node, set_db as human_set_db
from src.db.connection import DatabaseManager
from src.memory.session_store import SessionStore
from src.memory.profile_store import ProfileStore
from src.memory.consolidator import MemoryConsolidator


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
        return "consolidate"

    # Case 2: Specialist stopped but marked for escalation
    if state.escalation_reason:
        return "human_agent"

    # Case 3: Too many retries → escalate
    if state.retry_count >= state.max_retries:
        return "human_agent"

    # Case 4: Specialist finished normally → consolidate memory
    return "consolidate"


# ── Memory consolidator node ────────────────────────────────────────
def _build_memory_consolidator_node(consolidator: MemoryConsolidator):
    """Return a node function that runs end-of-session memory consolidation.

    This node is ONLY reachable from the SuperGraph level.  Specialist
    sub-agents (ReAct subgraphs) have no access to it.
    """

    async def memory_consolidator_node(state: CustomerServiceState) -> dict:
        """Persist L2b summary + update L3 profile."""
        return await consolidator.consolidate(state)

    return memory_consolidator_node


# ── Graph builder ───────────────────────────────────────────────────
async def build_customer_service_graph(
    model,
    db: DatabaseManager,
    checkpoint_db_path: str | Path = "",
) -> tuple[CompiledStateGraph, aiosqlite.Connection | None]:
    """Assemble and compile the full customer service graph.

    Parameters
    ----------
    checkpoint_db_path:
        Path to the SQLite DB for checkpoint persistence (L1).
        If empty, falls back to in-memory ``MemorySaver``.

    Returns
    -------
    (graph, checkpoint_conn)
        *graph* is the compiled graph ready for ``ainvoke()`` / ``astream()``.
        *checkpoint_conn* is the aiosqlite connection backing the checkpointer
        (or ``None`` if MemorySaver was used).  **The caller must close this
        connection** when done: ``await checkpoint_conn.close()``.
    """

    # Wire DB for human_agent
    human_set_db(db)

    # ── Memory sub-system ────────────────────────────────────────
    session_store = SessionStore(db)
    profile_store = ProfileStore(db)
    consolidator = MemoryConsolidator(model, session_store, profile_store, db)

    # ── L1: Checkpointer ─────────────────────────────────────────
    checkpoint_conn: aiosqlite.Connection | None = None
    if checkpoint_db_path:
        ckpt_path = Path(checkpoint_db_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_conn = await aiosqlite.connect(str(ckpt_path))
        checkpointer = AsyncSqliteSaver(checkpoint_conn)
        await checkpointer.setup()
    else:
        checkpointer = MemorySaver()

    # ── Build specialist subgraphs ──────────────────────────────
    order_graph = await build_order_agent(model, db)
    logistics_graph = await build_logistics_agent(model, db)
    refund_graph = await build_refund_agent(model, db)

    # ── Build supervisor node ───────────────────────────────────
    supervisor_node = build_supervisor_node(model, consolidator)

    # ── Assemble parent graph ───────────────────────────────────
    builder = StateGraph(CustomerServiceState)

    # Register nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("order_agent", order_graph)
    builder.add_node("logistics_agent", logistics_graph)
    builder.add_node("refund_agent", refund_graph)
    builder.add_node("human_agent", human_agent_node)
    builder.add_node(
        "memory_consolidator",
        _build_memory_consolidator_node(consolidator),
    )

    # Entry point
    builder.add_edge(START, "supervisor")

    # After each specialist completes normally (no Command), evaluate next step.
    for agent_name in ("order_agent", "logistics_agent", "refund_agent"):
        builder.add_conditional_edges(
            agent_name,
            route_after_specialist,
            {
                "consolidate": "memory_consolidator",
                "supervisor": "supervisor",
                "human_agent": "human_agent",
            },
        )

    # Human agent → memory consolidation → END
    builder.add_edge("human_agent", "memory_consolidator")

    # Memory consolidator always goes to END
    builder.add_edge("memory_consolidator", END)

    return builder.compile(checkpointer=checkpointer), checkpoint_conn
