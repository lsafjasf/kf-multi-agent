"""Seed data for the ShopFast demo database.

Provides 4 customers, 8 orders across all statuses, matching shipments and refunds.
Idempotent — uses INSERT OR IGNORE so it's safe to call repeatedly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.db.connection import DatabaseManager


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


def _days_ahead(n: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=n)).isoformat()


async def seed_database(db: DatabaseManager) -> None:
    """Insert seed data. Safe to call multiple times (INSERT OR IGNORE)."""

    async with db.transaction():
        # ── Customers ──────────────────────────────────────────────────
        await db.execute(
            "INSERT OR IGNORE INTO customers (id, name, email, phone) VALUES (?, ?, ?, ?)",
            ("CUST-001", "Alice Johnson", "alice@example.com", "+1-555-0101"),
        )
        await db.execute(
            "INSERT OR IGNORE INTO customers (id, name, email, phone) VALUES (?, ?, ?, ?)",
            ("CUST-002", "Bob Williams", "bob@example.com", "+1-555-0102"),
        )
        await db.execute(
            "INSERT OR IGNORE INTO customers (id, name, email, phone) VALUES (?, ?, ?, ?)",
            ("CUST-003", "Carol Davis", "carol@example.com", "+1-555-0103"),
        )
        await db.execute(
            "INSERT OR IGNORE INTO customers (id, name, email, phone) VALUES (?, ?, ?, ?)",
            ("CUST-004", "David Chen", "david@example.com", "+1-555-0104"),
        )

        # ── Orders ─────────────────────────────────────────────────────
        # ORD-001: pending — Alice
        await db.execute(
            "INSERT OR IGNORE INTO orders (id, customer_id, status, total_amount, currency, created_at, delivered_at, shipping_address, items) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ORD-001", "CUST-001", "pending", 89.97, "USD",
                _days_ago(1), None,
                "123 Main St, Springfield, IL 62701",
                '[{"name":"Bluetooth Headphones","qty":1,"price":59.99},{"name":"USB-C Cable","qty":2,"price":14.99}]',
            ),
        )

        # ORD-002: shipped — Alice (for logistics handoff demo)
        await db.execute(
            "INSERT OR IGNORE INTO orders (id, customer_id, status, total_amount, currency, created_at, delivered_at, shipping_address, items) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ORD-002", "CUST-001", "shipped", 45.50, "USD",
                _days_ago(3), None,
                "123 Main St, Springfield, IL 62701",
                '[{"name":"Phone Case","qty":1,"price":25.50},{"name":"Screen Protector","qty":1,"price":20.00}]',
            ),
        )

        # ORD-003: delivered — Bob (within 30 days, refund-eligible)
        await db.execute(
            "INSERT OR IGNORE INTO orders (id, customer_id, status, total_amount, currency, created_at, delivered_at, shipping_address, items) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ORD-003", "CUST-002", "delivered", 234.00, "USD",
                _days_ago(10), _days_ago(5),
                "456 Oak Ave, Portland, OR 97201",
                '[{"name":"Running Shoes","qty":1,"price":234.00}]',
            ),
        )

        # ORD-004: cancelled — Bob (refund already processed)
        await db.execute(
            "INSERT OR IGNORE INTO orders (id, customer_id, status, total_amount, currency, created_at, delivered_at, shipping_address, items) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ORD-004", "CUST-002", "cancelled", 15.99, "USD",
                _days_ago(14), None,
                "456 Oak Ave, Portland, OR 97201",
                '[{"name":"USB Cable","qty":1,"price":15.99}]',
            ),
        )

        # ORD-005: shipped — Carol (exception shipment, great demo case)
        await db.execute(
            "INSERT OR IGNORE INTO orders (id, customer_id, status, total_amount, currency, created_at, delivered_at, shipping_address, items) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ORD-005", "CUST-003", "shipped", 1299.00, "USD",
                _days_ago(14), None,
                "789 Pine Rd, Seattle, WA 98101",
                '[{"name":"Laptop - Pro Model","qty":1,"price":1299.00}]',
            ),
        )

        # ORD-006: pending — Carol
        await db.execute(
            "INSERT OR IGNORE INTO orders (id, customer_id, status, total_amount, currency, created_at, delivered_at, shipping_address, items) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ORD-006", "CUST-003", "pending", 67.50, "USD",
                _days_ago(2), None,
                "789 Pine Rd, Seattle, WA 98101",
                '[{"name":"Wireless Mouse","qty":1,"price":39.99},{"name":"Keyboard","qty":1,"price":27.51}]',
            ),
        )

        # ORD-007: delivered — David (45 days ago, NOT refund-eligible)
        await db.execute(
            "INSERT OR IGNORE INTO orders (id, customer_id, status, total_amount, currency, created_at, delivered_at, shipping_address, items) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ORD-007", "CUST-004", "delivered", 89.00, "USD",
                _days_ago(50), _days_ago(45),
                "321 Elm St, Austin, TX 73301",
                '[{"name":"Coffee Maker","qty":1,"price":89.00}]',
            ),
        )

        # ORD-008: delivered — David (5 days ago, refund-eligible)
        await db.execute(
            "INSERT OR IGNORE INTO orders (id, customer_id, status, total_amount, currency, created_at, delivered_at, shipping_address, items) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ORD-008", "CUST-004", "delivered", 29.99, "USD",
                _days_ago(7), _days_ago(5),
                "321 Elm St, Austin, TX 73301",
                '[{"name":"Desk Lamp","qty":1,"price":29.99}]',
            ),
        )

        # ── Shipments ──────────────────────────────────────────────────
        await db.execute(
            "INSERT OR IGNORE INTO shipments (id, order_id, carrier, tracking_number, status, estimated_delivery, shipped_at, delivered_at, last_update, current_location) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "SHIP-001", "ORD-002", "FedEx", "TRK-10001",
                "in_transit", _days_ahead(2), _days_ago(2), None,
                _now(), "Memphis, TN Distribution Center",
            ),
        )

        await db.execute(
            "INSERT OR IGNORE INTO shipments (id, order_id, carrier, tracking_number, status, estimated_delivery, shipped_at, delivered_at, last_update, current_location) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "SHIP-002", "ORD-003", "UPS", "TRK-10002",
                "delivered", _days_ago(3), _days_ago(7), _days_ago(5),
                _days_ago(5), "Portland, OR — Delivered",
            ),
        )

        await db.execute(
            "INSERT OR IGNORE INTO shipments (id, order_id, carrier, tracking_number, status, estimated_delivery, shipped_at, delivered_at, last_update, current_location) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "SHIP-003", "ORD-005", "UPS", "TRK-20003",
                "exception", _days_ago(3), _days_ago(10), None,
                _days_ago(5), "Seattle, WA — Sorting Facility (Exception: Weather Delay)",
            ),
        )

        await db.execute(
            "INSERT OR IGNORE INTO shipments (id, order_id, carrier, tracking_number, status, estimated_delivery, shipped_at, delivered_at, last_update, current_location) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "SHIP-004", "ORD-008", "USPS", "TRK-10004",
                "delivered", _days_ago(2), _days_ago(6), _days_ago(5),
                _days_ago(5), "Austin, TX — Delivered",
            ),
        )

        # ── Refunds ────────────────────────────────────────────────────
        await db.execute(
            "INSERT OR IGNORE INTO refunds (id, order_id, customer_id, amount, reason, status, created_at, processed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "REF-001", "ORD-004", "CUST-002", 15.99,
                "Changed mind, cancelled before shipment",
                "processed", _days_ago(13), _days_ago(12),
            ),
        )

        await db.execute(
            "INSERT OR IGNORE INTO refunds (id, order_id, customer_id, amount, reason, status, created_at, processed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "REF-002", "ORD-007", "CUST-004", 89.00,
                "Defective item — coffee maker doesn't turn on",
                "pending", _days_ago(2), None,
            ),
        )
