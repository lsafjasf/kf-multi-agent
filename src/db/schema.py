"""SQLite schema for the ShopFast e-commerce customer service system."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL,
    phone       TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id                TEXT PRIMARY KEY,
    customer_id       TEXT NOT NULL REFERENCES customers(id),
    status            TEXT NOT NULL CHECK(status IN (
                          'pending','confirmed','shipped',
                          'delivered','cancelled','refunded'
                      )),
    total_amount      REAL NOT NULL,
    currency          TEXT NOT NULL DEFAULT 'USD',
    created_at        TEXT NOT NULL,
    delivered_at      TEXT,
    shipping_address  TEXT NOT NULL,
    items             TEXT NOT NULL  -- JSON array of {name, qty, price}
);

CREATE TABLE IF NOT EXISTS shipments (
    id                  TEXT PRIMARY KEY,
    order_id            TEXT NOT NULL REFERENCES orders(id),
    carrier             TEXT NOT NULL,
    tracking_number     TEXT NOT NULL UNIQUE,
    status              TEXT NOT NULL CHECK(status IN (
                            'label_created','in_transit','out_for_delivery',
                            'delivered','exception','lost_investigation'
                        )),
    estimated_delivery  TEXT,
    shipped_at          TEXT,
    delivered_at        TEXT,
    last_update         TEXT NOT NULL,
    current_location    TEXT
);

CREATE TABLE IF NOT EXISTS refunds (
    id              TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL REFERENCES orders(id),
    customer_id     TEXT NOT NULL REFERENCES customers(id),
    amount          REAL NOT NULL,
    reason          TEXT NOT NULL,
    status          TEXT NOT NULL CHECK(status IN (
                        'pending','approved','rejected','processed'
                    )),
    created_at      TEXT NOT NULL,
    processed_at    TEXT
);

CREATE TABLE IF NOT EXISTS support_tickets (
    id                  TEXT PRIMARY KEY,
    customer_id         TEXT NOT NULL REFERENCES customers(id),
    order_id            TEXT,
    issue_summary       TEXT NOT NULL,
    escalation_reason   TEXT NOT NULL,
    agent_trace         TEXT NOT NULL DEFAULT '[]',
    status              TEXT NOT NULL DEFAULT 'open'
                        CHECK(status IN ('open','assigned','resolved')),
    created_at          TEXT NOT NULL
);

-- ── Memory: L2b historical session summaries ─────────────────
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    summary       TEXT NOT NULL,
    key_entities  TEXT NOT NULL DEFAULT '[]',
    resolution    TEXT NOT NULL CHECK(resolution IN (
                      'resolved','escalated','abandoned'
                  )),
    ticket_id     TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    duration_ms   INTEGER,
    weight        REAL NOT NULL DEFAULT 1.0,
    archived      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);

-- ── Memory: L3 user long-term profiles ───────────────────────
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id             TEXT PRIMARY KEY,
    preferences         TEXT NOT NULL DEFAULT '{}',
    common_issues       TEXT NOT NULL DEFAULT '[]',
    total_sessions      INTEGER NOT NULL DEFAULT 0,
    total_resolved      INTEGER NOT NULL DEFAULT 0,
    total_escalated     INTEGER NOT NULL DEFAULT 0,
    favorite_agent      TEXT,
    sentiment_trend     TEXT DEFAULT 'neutral',
    last_session_at     TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);
"""
