"""Order tools backed by SQLite — query orders, cancel orders."""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import tool

from src.db.connection import DatabaseManager


# Module-level db reference — set by the factory on graph construction.
_db: DatabaseManager | None = None


def set_db(db: DatabaseManager) -> None:
    """Set the module-level database manager. Called at graph build time."""
    global _db
    _db = db


def _get_db() -> DatabaseManager:
    if _db is None:
        raise RuntimeError("Database not initialized. Call set_db() first.")
    return _db


@tool
async def query_order_by_id(order_id: Annotated[str, "The order ID, e.g. 'ORD-001'."]) -> str:
    """Look up a single order by its unique ID.

    Returns full order details including status, items, total amount,
    shipping address, and dates.
    """
    db = _get_db()
    row = await db.fetch_one(
        """SELECT o.*, c.name AS customer_name, c.email AS customer_email
           FROM orders o JOIN customers c ON o.customer_id = c.id
           WHERE o.id = ?""",
        (order_id,),
    )
    if row is None:
        return json.dumps({"error": f"Order '{order_id}' not found."})

    # Also fetch related shipment if any
    shipment = await db.fetch_one(
        "SELECT * FROM shipments WHERE order_id = ?", (order_id,)
    )

    result = dict(row)
    result["shipment"] = dict(shipment) if shipment else None
    return json.dumps(result, default=str)


@tool
async def query_orders_by_user(user_id: Annotated[str, "The customer ID, e.g. 'CUST-001'."]) -> str:
    """List all orders for a given customer.

    Returns a list of orders with basic info (id, status, total, date).
    Use this to help a customer see their order history.
    """
    db = _get_db()
    rows = await db.fetch_all(
        """SELECT id, status, total_amount, currency, created_at
           FROM orders WHERE customer_id = ?
           ORDER BY created_at DESC""",
        (user_id,),
    )
    return json.dumps(rows, default=str)


@tool
async def cancel_order(order_id: Annotated[str, "The order ID to cancel, e.g. 'ORD-001'."]) -> str:
    """Cancel a pending order.

    Only orders in 'pending' status can be cancelled. Once an order is
    confirmed or shipped, cancellation is not possible — suggest a refund.
    """
    db = _get_db()
    row = await db.fetch_one(
        "SELECT id, status, total_amount FROM orders WHERE id = ?",
        (order_id,),
    )
    if row is None:
        return json.dumps({"error": f"Order '{order_id}' not found."})
    if row["status"] != "pending":
        return json.dumps({
            "error": f"Order '{order_id}' is in '{row['status']}' status and cannot be cancelled.",
            "suggestion": "If the order has already shipped, offer the refund process instead.",
        })

    await db.execute(
        "UPDATE orders SET status = 'cancelled' WHERE id = ?",
        (order_id,),
    )
    await db.commit()
    return json.dumps({
        "success": True,
        "message": f"Order '{order_id}' has been cancelled.",
        "refund_amount": row["total_amount"],
    })
