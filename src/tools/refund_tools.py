"""Refund tools — eligibility checks and refund processing.

Simulates a financial/accounting backend.
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

from langchain_core.tools import tool

from src.db.registry import get_db


@tool
async def check_refund_eligibility(order_id: Annotated[str, "The order ID to check, e.g. 'ORD-008'."]) -> str:
    """Check whether an order is eligible for a refund.

    Rules:
    - Orders in 'pending' or 'shipped' status: eligible (cancellation refund).
    - Orders in 'delivered' status: eligible if delivered within 30 days AND
      not already refunded.
    - Orders already 'cancelled' or 'refunded': not eligible.
    """
    db = get_db()
    row = await db.fetch_one(
        "SELECT id, customer_id, status, total_amount, delivered_at FROM orders WHERE id = ?",
        (order_id,),
    )
    if row is None:
        return json.dumps({"error": f"Order '{order_id}' not found."})

    status = row["status"]

    # Check for existing refund
    existing = await db.fetch_one(
        "SELECT id, status FROM refunds WHERE order_id = ? AND status != 'rejected'",
        (order_id,),
    )
    if existing:
        return json.dumps({
            "eligible": False,
            "reason": f"Order '{order_id}' already has a {existing['status']} refund (refund ID: {existing['id']}).",
        })

    if status in ("pending", "shipped"):
        return json.dumps({
            "eligible": True,
            "refund_type": "cancellation",
            "amount": row["total_amount"],
            "currency": "USD",
            "message": f"Order '{order_id}' is in '{status}' status — eligible for full cancellation refund.",
        })

    if status == "delivered":
        delivered_at_str = row.get("delivered_at")
        if not delivered_at_str:
            return json.dumps({
                "eligible": False,
                "reason": f"Order '{order_id}' is marked delivered but has no delivery date.",
            })

        delivered_at = datetime.fromisoformat(delivered_at_str)
        if delivered_at.tzinfo is None:
            delivered_at = delivered_at.replace(tzinfo=timezone.utc)
        days_since = (datetime.now(timezone.utc) - delivered_at).days

        if days_since > 30:
            return json.dumps({
                "eligible": False,
                "reason": f"Order '{order_id}' was delivered {days_since} days ago — "
                          f"outside the 30-day refund window.",
            })

        return json.dumps({
            "eligible": True,
            "refund_type": "return",
            "amount": row["total_amount"],
            "currency": "USD",
            "days_since_delivery": days_since,
            "message": f"Order '{order_id}' is eligible for return refund. "
                       f"Delivered {days_since} days ago (within 30-day window).",
        })

    return json.dumps({
        "eligible": False,
        "reason": f"Order '{order_id}' is in '{status}' status — not eligible for refund.",
    })


@tool
async def process_refund(
    order_id: Annotated[str, "The order ID to refund, e.g. 'ORD-008'."],
    reason: Annotated[str, "Reason for the refund: 'defective', 'wrong_item', 'not_as_described', 'late_delivery', 'lost_package', 'customer_request'."],
) -> str:
    """Process a refund for an eligible order.

    This simulates the financial backend — creates a refund record,
    updates the order status to 'refunded', and returns a refund ID.

    IMPORTANT: Always call check_refund_eligibility BEFORE this tool
    to confirm the order can be refunded.
    """
    db = get_db()

    # Small delay to simulate financial system processing
    await asyncio.sleep(random.uniform(0.5, 1.0))

    order = await db.fetch_one(
        "SELECT id, customer_id, total_amount, status FROM orders WHERE id = ?",
        (order_id,),
    )
    if order is None:
        return json.dumps({"error": f"Order '{order_id}' not found."})

    refund_id = f"REF-{uuid.uuid4().hex[:6].upper()}"
    now = datetime.now(timezone.utc).isoformat()

    async with db.transaction():
        await db.execute(
            """INSERT INTO refunds (id, order_id, customer_id, amount, reason, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
            (refund_id, order_id, order["customer_id"], order["total_amount"], reason, now),
        )
        await db.execute(
            "UPDATE orders SET status = 'refunded' WHERE id = ?",
            (order_id,),
        )

    return json.dumps({
        "success": True,
        "refund_id": refund_id,
        "order_id": order_id,
        "amount": order["total_amount"],
        "currency": "USD",
        "status": "pending",
        "message": f"Refund of ${order['total_amount']:.2f} for order '{order_id}' "
                   f"has been submitted. It will process in 3-5 business days.",
        "estimated_completion": (datetime.now(timezone.utc) + timedelta(days=5)).isoformat(),
    })


@tool
async def query_refund_status(refund_id: Annotated[str, "The refund ID, e.g. 'REF-001'."]) -> str:
    """Check the status of an existing refund.

    Returns the current status (pending/approved/rejected/processed),
    amount, reason, and processing timeline.
    """
    db = get_db()
    row = await db.fetch_one(
        "SELECT * FROM refunds WHERE id = ?", (refund_id,)
    )
    if row is None:
        return json.dumps({"error": f"Refund '{refund_id}' not found."})

    return json.dumps(dict(row), default=str)
