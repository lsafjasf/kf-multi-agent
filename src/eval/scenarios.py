"""Evaluation scenario catalog for the ShopFast multi-agent system.

Each scenario is an ``EvalScenario`` with message, user context, and optional
assertions.  Assertions are soft — when a field is ``None`` it is not checked.

Categories
----------
normal      Happy path: single-agent task, tools fire, answer returned.
handoff     Cross-agent transfers (order→logistics, logistics→refund, etc.).
edge        Boundary cases: outside refund window, already cancelled, etc.
error       Bogus input: non-existent IDs, gibberish — system must not crash.
escalation  Issues that should end at human_agent with a ticket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ScenarioCategory = Literal["normal", "handoff", "edge", "error", "escalation"]


@dataclass
class EvalScenario:
    """A single end-to-end evaluation case."""

    id: str
    title: str
    category: ScenarioCategory
    user_id: str
    message: str

    # ── Soft assertions (None = don't check) ─────────────────────────
    expected_agent: str | None = None       # final active_agent
    expected_resolved: bool | None = None   # did the conversation resolve?
    expected_ticket: bool = False           # should a support ticket be created?
    expected_tools: list[str] | None = None # tools that should have been called
    forbidden_tools: list[str] | None = None  # tools that should NOT be called
    agent_trace_contains: list[str] | None = None  # agents that should appear in path

    # ── Post-run DB checks ─────────────────────────────────────────
    db_checks: list[dict] = field(default_factory=list)
    # Each dict: {"sql": str, "params": list, "expected": dict | None, "description": str}


# ═══════════════════════════════════════════════════════════════════════════
#  Scenario catalog
# ═══════════════════════════════════════════════════════════════════════════

ALL_SCENARIOS: list[EvalScenario] = [
    # ── NORMAL ──────────────────────────────────────────────────────────
    EvalScenario(
        id="order_query_by_id",
        title="Order lookup by order ID",
        category="normal",
        user_id="CUST-001",
        message="What's the status of my order ORD-001?",
        expected_agent="order_agent",
        expected_resolved=False,
        expected_tools=["query_order_by_id"],
        forbidden_tools=["escalate_to_human"],
    ),
    EvalScenario(
        id="order_query_by_user",
        title="Order lookup by customer ID",
        category="normal",
        user_id="CUST-001",
        message="Show me all my recent orders. My customer ID is CUST-001.",
        expected_agent="order_agent",
        expected_resolved=False,
        expected_tools=["query_orders_by_user"],
    ),
    EvalScenario(
        id="cancel_pending_order",
        title="Cancel a pending order",
        category="normal",
        user_id="CUST-003",
        message="I changed my mind — please cancel order ORD-006.",
        expected_agent="order_agent",
        expected_resolved=False,
        forbidden_tools=["escalate_to_human"],
        # Agent may query first or cancel directly — both valid.
        # DB check is the real validation of correct behavior.
        db_checks=[
            {
                "sql": "SELECT status FROM orders WHERE id = ?",
                "params": ["ORD-006"],
                "expected": {"status": "cancelled"},
                "description": "ORD-006 should be cancelled",
            }
        ],
    ),
    EvalScenario(
        id="track_by_tracking_number",
        title="Track shipment by tracking number",
        category="normal",
        user_id="CUST-001",
        message="Track package TRK-10001 — where is it right now?",
        expected_agent="logistics_agent",
        expected_resolved=False,
        expected_tools=["track_shipment"],
    ),
    EvalScenario(
        id="shipment_by_order_id",
        title="Find shipment by order ID",
        category="normal",
        user_id="CUST-001",
        message="Can you look up the shipment for order ORD-002?",
        expected_agent="logistics_agent",
        expected_resolved=False,
        expected_tools=["query_shipment_by_order"],
    ),
    EvalScenario(
        id="estimate_delivery_date",
        title="Get delivery estimate",
        category="normal",
        user_id="CUST-001",
        message="When is my order ORD-002 going to arrive?",
        expected_agent="logistics_agent",
        expected_resolved=False,
        expected_tools=["estimate_delivery"],
    ),
    EvalScenario(
        id="check_refund_eligibility",
        title="Check if an order is refund-eligible",
        category="normal",
        user_id="CUST-002",
        message="Can I get a refund for ORD-003?",
        # Supervisor rules: mentions Order ID → order_agent first.
        # But "refund" keyword → refund_agent. Both paths valid.
        expected_resolved=False,
        forbidden_tools=["escalate_to_human"],
    ),
    EvalScenario(
        id="process_full_refund",
        title="Full refund flow (check + process)",
        category="normal",
        user_id="CUST-004",
        message="I received the wrong item in ORD-008. I want my money back.",
        expected_agent="refund_agent",
        expected_resolved=False,
        expected_tools=["check_refund_eligibility", "process_refund"],
        # ORD-008: delivered 5 days ago, refund-eligible, no prior refund
        # (in a fresh DB — may have one if running after other ORD-008 scenarios)
        db_checks=[
            {
                "sql": "SELECT status FROM orders WHERE id = ?",
                "params": ["ORD-008"],
                "expected": {"status": "refunded"},
                "description": "ORD-008 should be marked refunded",
            }
        ],
    ),
    EvalScenario(
        id="query_refund_status",
        title="Check existing refund status",
        category="normal",
        user_id="CUST-004",
        message="What's happening with refund REF-002? Has it been processed yet?",
        expected_agent="refund_agent",
        expected_resolved=False,
        expected_tools=["query_refund_status"],
    ),

    # ── HANDOFF ────────────────────────────────────────────────────────
    EvalScenario(
        id="order_to_logistics_handoff",
        title="Order agent hands off to logistics for tracking",
        category="handoff",
        user_id="CUST-001",
        message="I ordered ORD-002 and need detailed tracking. "
                "First check my order, then get me the shipment info.",
        expected_agent="logistics_agent",  # should end up at logistics
        expected_resolved=False,
        expected_tools=["query_shipment_by_order"],
    ),
    EvalScenario(
        id="order_to_refund_cancel_rejected",
        title="Cancel shipped order → offer refund transfer",
        category="handoff",
        user_id="CUST-001",
        message="Cancel ORD-002. I don't want it anymore.",
        # Supervisor may route "cancel" to order_agent or refund_agent.
        # Either handles it: order_agent sees shipped→offers refund,
        # refund_agent checks eligibility (shipped→eligible for cancellation).
        expected_resolved=False,
        forbidden_tools=["escalate_to_human"],
    ),
    EvalScenario(
        id="logistics_to_refund_lost_package",
        title="Lost package → investigate + offer refund",
        category="handoff",
        user_id="CUST-003",
        message="TRK-20003 has been stuck in exception status for 2 weeks! "
                "I paid $1,299 for this laptop. What are you going to do about it?",
        # Supervisor keyword-matches "refund" → may route to refund_agent
        # OR keyword-matches tracking → may route to logistics_agent.
        # Either path is acceptable as long as the system doesn't crash.
        expected_resolved=False,
        forbidden_tools=["escalate_to_human"],
        # NOTE: known limitation — supervisor routing is keyword-based and
        # may not always capture the need for escalation in high-value cases.
    ),

    # ── EDGE ───────────────────────────────────────────────────────────
    EvalScenario(
        id="cancel_shipped_order_fails",
        title="Try to cancel an already-shipped order",
        category="edge",
        user_id="CUST-003",
        message="Cancel ORD-005 right now.",
        # Supervisor may route "cancel" to order_agent or refund_agent
        # (both are reasonable — cancel involves money back).
        # Assertion: system must not crash, must not escalate.
        expected_resolved=False,
        forbidden_tools=["escalate_to_human"],
    ),
    EvalScenario(
        id="refund_outside_window",
        title="Refund denied — delivered > 30 days ago",
        category="edge",
        user_id="CUST-004",
        message="I want a full refund for ORD-007 please.",
        expected_agent="refund_agent",
        expected_resolved=False,
        expected_tools=["check_refund_eligibility"],
        forbidden_tools=["process_refund", "escalate_to_human"],
        # check_refund_eligibility returns eligible=false; agent explains
    ),
    EvalScenario(
        id="refund_already_refunded",
        title="Refund denied — order already refunded",
        category="edge",
        user_id="CUST-002",
        message="I want my money back for ORD-004.",
        expected_agent="refund_agent",
        expected_resolved=False,
        expected_tools=["check_refund_eligibility"],
        forbidden_tools=["process_refund"],
        # ORD-004 is already cancelled + REF-001 processed
    ),
    EvalScenario(
        id="ambiguous_tracking_and_refund",
        title="Ambiguous: mentions both tracking and refund",
        category="edge",
        user_id="CUST-002",
        message="ORD-003 was the wrong item and I want to know where "
                "my refund is. Has it shipped?",
        expected_resolved=False,
        # No tight assertion on which agent — just don't crash
    ),
    EvalScenario(
        id="no_order_id_vague",
        title="Vague complaint — no IDs given",
        category="edge",
        user_id="CUST-001",
        message="I ordered something last week and I haven't received it yet. Help!",
        expected_resolved=False,
        # Supervisor should route somewhere reasonable; system must not crash
    ),

    # ── ERROR ──────────────────────────────────────────────────────────
    EvalScenario(
        id="nonexistent_order",
        title="Non-existent order ID",
        category="error",
        user_id="CUST-001",
        message="What's the status of ORD-999?",
        expected_agent="order_agent",
        expected_resolved=False,
        expected_tools=["query_order_by_id"],
        forbidden_tools=["escalate_to_human"],
    ),
    EvalScenario(
        id="nonexistent_tracking",
        title="Non-existent tracking number",
        category="error",
        user_id="CUST-001",
        message="Track TRK-FAKE-99. I need an update ASAP.",
        expected_agent="logistics_agent",
        expected_resolved=False,
        expected_tools=["track_shipment"],
        forbidden_tools=["escalate_to_human"],
    ),
    EvalScenario(
        id="gibberish_input",
        title="Random gibberish input",
        category="error",
        user_id="CUST-002",
        message="asdfghjkl !@#$%^&*()",
        expected_resolved=False,
        forbidden_tools=["escalate_to_human"],
        # System should route somewhere and ask for clarification, not crash
    ),

    # ── ESCALATION ─────────────────────────────────────────────────────
    EvalScenario(
        id="high_value_lost_escalation",
        title="$1,299 laptop lost → escalate to human",
        category="escalation",
        user_id="CUST-003",
        message="My ORD-005 with tracking TRK-20003 has been stuck with an "
                "exception for 2 weeks. This is a $1,299 laptop. "
                "I'm furious — escalate this to your manager RIGHT NOW.",
        expected_agent="human_agent",
        expected_resolved=True,
        expected_ticket=True,
        # Stronger signal for escalation: explicitly demands manager
    ),
    EvalScenario(
        id="demand_human_agent",
        title="Customer explicitly demands a human",
        category="escalation",
        user_id="CUST-002",
        message="I'm done talking to robots. Connect me to a real person immediately.",
        expected_agent="human_agent",
        expected_resolved=True,
        expected_ticket=True,
        forbidden_tools=["query_order_by_id", "track_shipment"],
    ),
]
