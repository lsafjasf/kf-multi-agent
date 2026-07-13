"""Logistics tools — shipment tracking via mock external carrier APIs."""

from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timedelta, timezone
from typing import Annotated

from langchain_core.tools import tool

from src.config import get_config
from src.db.connection import DatabaseManager


_db: DatabaseManager | None = None


def set_db(db: DatabaseManager) -> None:
    global _db
    _db = db


def _get_db() -> DatabaseManager:
    if _db is None:
        raise RuntimeError("Database not initialized. Call set_db() first.")
    return _db


async def _simulate_api_call() -> bool:
    """Simulate an external API call with configurable failure rate.

    Returns True on success, False on simulated failure.
    """
    config = get_config()
    delay = random.uniform(0.3, 0.8)
    await asyncio.sleep(delay)

    if random.random() < config.mock_api_failure_rate:
        return False
    return True


@tool
async def track_shipment(tracking_number: Annotated[str, "Carrier tracking number, e.g. 'TRK-10001'."]) -> str:
    """Track a shipment by its carrier tracking number.

    Simulates querying FedEx/UPS/USPS tracking API.
    Returns current location, status, and estimated delivery.
    """
    db = _get_db()
    if not await _simulate_api_call():
        return json.dumps({"error": "Carrier API temporarily unavailable. Please try again."})

    row = await db.fetch_one(
        """SELECT s.*, o.status AS order_status
           FROM shipments s JOIN orders o ON s.order_id = o.id
           WHERE s.tracking_number = ?""",
        (tracking_number,),
    )
    if row is None:
        return json.dumps({"error": f"Tracking number '{tracking_number}' not found."})

    result = dict(row)
    result["last_checked"] = datetime.now(timezone.utc).isoformat()
    return json.dumps(result, default=str)


@tool
async def query_shipment_by_order(order_id: Annotated[str, "The order ID, e.g. 'ORD-002'."]) -> str:
    """Find shipment information for a given order ID.

    Returns carrier, tracking number, status, and location details.
    """
    db = _get_db()
    row = await db.fetch_one(
        "SELECT * FROM shipments WHERE order_id = ?", (order_id,)
    )
    if row is None:
        return json.dumps({"error": f"No shipment found for order '{order_id}'."})
    return json.dumps(dict(row), default=str)


@tool
async def estimate_delivery(order_id: Annotated[str, "The order ID, e.g. 'ORD-002'."]) -> str:
    """Get the estimated delivery date for an order's shipment.

    Simulates querying the carrier's delivery prediction API.
    """
    db = _get_db()
    if not await _simulate_api_call():
        return json.dumps({"error": "Delivery estimation service unavailable. Please try again."})

    row = await db.fetch_one(
        "SELECT tracking_number, estimated_delivery, status FROM shipments WHERE order_id = ?",
        (order_id,),
    )
    if row is None:
        return json.dumps({"error": f"No shipment found for order '{order_id}'."})

    return json.dumps({
        "order_id": order_id,
        "tracking_number": row["tracking_number"],
        "estimated_delivery": row["estimated_delivery"],
        "shipment_status": row["status"],
        "note": "Estimates are updated in real-time and may change based on carrier conditions.",
    }, default=str)


@tool
async def report_lost_package(tracking_number: Annotated[str, "Tracking number of the lost package, e.g. 'TRK-20003'."]) -> str:
    """Report a package as lost and initiate a carrier investigation.

    Flags the shipment as 'lost_investigation' and creates a case.
    This should be used when a package has shown no tracking updates
    for an extended period.
    """
    db = _get_db()
    row = await db.fetch_one(
        "SELECT id, order_id, status, carrier FROM shipments WHERE tracking_number = ?",
        (tracking_number,),
    )
    if row is None:
        return json.dumps({"error": f"Tracking number '{tracking_number}' not found."})

    if row["status"] == "lost_investigation":
        return json.dumps({
            "message": f"Package '{tracking_number}' is already under investigation.",
            "order_id": row["order_id"],
        })

    await db.execute(
        "UPDATE shipments SET status = 'lost_investigation', last_update = ? WHERE tracking_number = ?",
        (datetime.now(timezone.utc).isoformat(), tracking_number),
    )
    await db.commit()
    return json.dumps({
        "success": True,
        "message": f"Lost package investigation opened for '{tracking_number}' "
                   f"(order {row['order_id']}) with {row['carrier']}.",
        "investigation_id": f"INV-{tracking_number}",
        "estimated_resolution": (datetime.now(timezone.utc) + timedelta(days=5)).isoformat(),
        "next_steps": "Carrier will investigate within 3-5 business days. "
                      "Consider offering a refund or replacement.",
    })
