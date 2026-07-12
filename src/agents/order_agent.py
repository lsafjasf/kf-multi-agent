"""Order Specialist Agent — order lookups and cancellations via SQLite tools."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph

from src.db.connection import DatabaseManager
from src.tools.order_tools import (
    cancel_order,
    query_order_by_id,
    query_orders_by_user,
    set_db as order_set_db,
)
from src.tools.handoff import build_handoff_tools_for

ORDER_SYSTEM_PROMPT = """\
You are the ORDER SPECIALIST for ShopFast, a large e-commerce platform.

Your job: help customers with anything related to their orders.

## Your Tools
- **query_order_by_id**: Look up a single order by its ID (e.g., ORD-001).
  Shows status, items, price, shipping address, and linked shipment.
- **query_orders_by_user**: List all orders for a customer by their ID
  (e.g., CUST-001).
- **cancel_order**: Cancel an order that is still in "pending" status.
  CANNOT cancel orders that are already confirmed, shipped, or delivered.

## Your Workflow
1. If the customer mentions an order ID, look it up FIRST with query_order_by_id.
2. If they don't know their order ID but give a customer ID, use
   query_orders_by_user to show their orders.
3. If they ask about shipping/tracking/delivery of an order, look up the order
   first, then use **transfer_to_logistics_agent** — you don't have shipping tools.
4. If they want a refund, use **transfer_to_refund_agent** — do NOT try to
   process refunds yourself (you don't have the tools for it).
5. If the order is in "shipped" status and they want to cancel → explain it's
   too late to cancel, and offer transfer_to_refund_agent.

## Rules
- Be polite, concise, and professional.
- If you cannot help after 1-2 attempts, call **escalate_to_human** with a
  clear summary of the issue and what you tried.
- When you've successfully answered the question, stop — don't keep offering
  more help.
"""


async def build_order_agent(
    model: BaseChatModel,
    db: DatabaseManager,
) -> CompiledStateGraph:
    """Build the Order Agent as a ReAct subgraph.

    The agent can query orders (SQLite) and hand off to logistics/refund/human.
    """
    # Wire up the DB so order tools can use it
    order_set_db(db)

    # Domain tools
    domain_tools: list[BaseTool] = [
        query_order_by_id,
        query_orders_by_user,
        cancel_order,
    ]

    # Handoff tools: transfer to logistics, refund, and escalate
    handoff_tools = build_handoff_tools_for("order_agent")

    all_tools = domain_tools + handoff_tools

    return create_react_agent(
        model=model,
        tools=all_tools,
        prompt=ORDER_SYSTEM_PROMPT,
        name="order_agent",
    )
