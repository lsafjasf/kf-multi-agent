"""Shared graph state for the customer service agent system.

Uses Pydantic BaseModel for defaults, validation, and ergonomic state updates.
The ``messages`` field uses LangGraph's ``add_messages`` reducer for
append-only-with-dedup semantics.
"""

from __future__ import annotations

from typing import Annotated, Literal
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


AgentName = Literal[
    "supervisor",
    "order_agent",
    "logistics_agent",
    "refund_agent",
    "human_agent",
]


class CustomerServiceState(BaseModel):
    """State flowing through the customer service graph.

    - ``messages`` uses ``add_messages`` (append-or-replace-by-ID).
    - All other fields use last-write-wins semantics.
    """

    # ── Chat history ──────────────────────────────────────────
    messages: Annotated[list[AnyMessage], add_messages] = Field(
        default_factory=list,
        description="Full conversation history.",
    )

    # ── Routing ───────────────────────────────────────────────
    active_agent: AgentName = Field(
        default="supervisor",
        description="Agent currently handling the conversation.",
    )
    next_agent: AgentName | None = Field(
        default=None,
        description="Agent to hand off to (set by handoff tools or supervisor).",
    )
    user_id: str | None = Field(
        default=None,
        description="Extracted customer ID for context propagation.",
    )
    order_id: str | None = Field(
        default=None,
        description="Extracted order ID for context propagation.",
    )

    # ── Escalation ────────────────────────────────────────────
    retry_count: int = Field(
        default=0,
        description="Consecutive handoff/retry count for current issue.",
    )
    max_retries: int = Field(
        default=3,
        description="Threshold after which auto-escalation fires.",
    )
    escalation_reason: str = Field(
        default="",
        description="Human-readable reason for escalation.",
    )
    support_ticket_id: str | None = Field(
        default=None,
        description="Generated ticket ID when escalated to human.",
    )

    # ── Memory (L2a / context) ───────────────────────────────
    # NOTE: L1 is managed by the checkpointer (AsyncSqliteSaver);
    #       L2b and L3 are persisted in SQLite and only injected
    #       into the Supervisor prompt — they are NOT stored here.
    session_id: str = Field(
        default="",
        description="Unique ID for the current conversation session.",
    )
    running_summary: str = Field(
        default="",
        description="L2a: running summary accumulated by the Supervisor "
        "during the session.  Ephemeral — dies with the session.",
    )
    memory_context: str = Field(
        default="",
        description="L2b + L3 formatted text injected into Supervisor's "
        "system prompt.  Loaded once at session start.",
    )
    session_started_at: float = Field(
        default=0.0,
        description="time.perf_counter() at session start for duration calc.",
    )

    # ── Resolution ────────────────────────────────────────────
    resolved: bool = Field(
        default=False,
        description="True when the customer issue is fully resolved.",
    )
