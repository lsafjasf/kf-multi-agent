"""Human Agent — simulates creating a support ticket for escalated issues.

This is a plain node function (not a subgraph) that generates a ticket ID,
stores it in SQLite, and returns a polite resolution message.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from langchain_core.messages import AIMessage

from src.db.connection import DatabaseManager
from src.state import CustomerServiceState

_db: DatabaseManager | None = None


def set_db(db: DatabaseManager) -> None:
    global _db
    _db = db


async def human_agent_node(state: CustomerServiceState) -> dict:
    """Create a support ticket and notify the customer.

    Reads ``escalation_reason``, ``user_id``, ``order_id``, ``active_agent``
    from state. Generates a ticket, writes it to SQLite, and returns a
    final response message.
    """
    if _db is None:
        return {
            "messages": [
                AIMessage(
                    content="⚠️ Escalation system unavailable. Please contact "
                    "support@shopfast.com directly. We apologize for the inconvenience."
                )
            ],
            "resolved": True,
        }

    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now(timezone.utc).isoformat()

    customer_id = state.user_id or "UNKNOWN"
    order_id = state.order_id or "N/A"

    # Trace which agents handled this case
    agent_trace = str({
        "last_agent": state.active_agent,
        "retry_count": state.retry_count,
        "escalation_reason": state.escalation_reason,
    })

    await _db.execute(
        """INSERT INTO support_tickets
           (id, customer_id, order_id, issue_summary, escalation_reason, agent_trace, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, 'open', ?)""",
        (
            ticket_id,
            customer_id,
            order_id,
            state.escalation_reason or "No summary provided",
            state.escalation_reason or "Auto-escalated by system",
            agent_trace,
            now,
        ),
    )
    await _db.commit()

    message = (
        f"📋 **Support Ticket Created**\n\n"
        f"Your case has been escalated to our human support team.\n"
        f"- **Ticket ID**: {ticket_id}\n"
        f"- **Issue**: {state.escalation_reason}\n\n"
        f"A member of our team will reach out to you within **2 hours** "
        f"during business hours. You can reference ticket **{ticket_id}** "
        f"in any follow-up communication.\n\n"
        f"We apologize for the inconvenience and appreciate your patience.\n"
        f"— ShopFast Support Team"
    )

    return {
        "messages": [AIMessage(content=message)],
        "support_ticket_id": ticket_id,
        "active_agent": "human_agent",
        "resolved": True,
    }
