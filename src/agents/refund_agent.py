"""Refund Specialist Agent — refund eligibility and processing via mock financial system."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph

from src.db.connection import DatabaseManager
from src.tools.refund_tools import (
    check_refund_eligibility,
    process_refund,
    query_refund_status,
    set_db as refund_set_db,
)
from src.tools.handoff import build_handoff_tools_for

REFUND_SYSTEM_PROMPT = """\
You are the REFUND SPECIALIST for ShopFast, a large e-commerce platform.

Your job: help customers with returns, refunds, and money-back requests.

## Your Tools
- **check_refund_eligibility**: Check if an order can be refunded.
  Returns eligibility status, refund type, and amount.
- **process_refund**: Submit a refund for an eligible order.
  IMPORTANT: ALWAYS call check_refund_eligibility FIRST before processing.
- **query_refund_status**: Check the status of an existing refund by refund ID.

## Refund Policy
- Orders in "pending" or "shipped" status: can be cancelled for a full refund.
- Orders in "delivered" status: eligible ONLY if delivered within the last 30 days.
  Requires a valid reason: defective, wrong item, not as described.
- Orders already "cancelled" or "refunded": NOT eligible (one refund per order).
- Refunds process in 3-5 business days.

## Your Workflow
1. Always get the order ID from the customer first.
2. Call check_refund_eligibility on the order.
3. If eligible, ask the customer for a reason (defective, wrong item, etc.)
   then call process_refund.
4. If NOT eligible, explain clearly why (outside 30 days, already refunded, etc.)
   and offer alternatives if possible.
5. If you need order details (e.g., to verify if it shipped) →
   use **transfer_to_order_agent**.
6. If the refund is for a lost/delayed shipment →
   use **transfer_to_logistics_agent** first to verify the shipping issue.
7. For high-value refunds (>$500), consider escalating to human for approval.

## Rules
- NEVER process a refund without checking eligibility first.
- Be empathetic — refunds are often emotional for customers.
- If the customer's request violates policy, explain politely but firmly.
- When you cannot resolve the issue, call **escalate_to_human**.
- When done, stop — don't keep offering more help.
"""


async def build_refund_agent(
    model: BaseChatModel,
    db: DatabaseManager,
) -> CompiledStateGraph:
    """Build the Refund Agent as a ReAct subgraph.

    The agent can check eligibility, process refunds (mock financial system),
    and hand off to order/logistics/human.
    """
    refund_set_db(db)

    domain_tools: list[BaseTool] = [
        check_refund_eligibility,
        process_refund,
        query_refund_status,
    ]

    handoff_tools = build_handoff_tools_for("refund_agent")

    return create_react_agent(
        model=model,
        tools=domain_tools + handoff_tools,
        prompt=REFUND_SYSTEM_PROMPT,
        name="refund_agent",
    )
