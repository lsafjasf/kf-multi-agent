"""Async SQLite connection manager with query helpers."""

from __future__ import annotations

import aiosqlite
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from src.db.schema import SCHEMA_SQL


class DatabaseManager:
    """Async SQLite connection manager.

    Usage:
        async with DatabaseManager("data/shopfast.db") as db:
            await db.init_schema()
            row = await db.fetch_one("SELECT * FROM orders WHERE id = ?", ("ORD-001",))
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.conn: aiosqlite.Connection | None = None

    async def __aenter__(self) -> "DatabaseManager":
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = await aiosqlite.connect(str(self.db_path))
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute("PRAGMA journal_mode=WAL")
        await self.conn.execute("PRAGMA foreign_keys=ON")
        return self

    async def __aexit__(self, *args: object) -> None:
        if self.conn:
            await self.conn.close()

    async def execute(self, sql: str, params: tuple = ()) -> None:
        """Execute a write statement. Caller must commit() or use transaction()."""
        if not self.conn:
            raise RuntimeError("Database not connected.")
        await self.conn.execute(sql, params)

    async def commit(self) -> None:
        """Commit the current transaction."""
        if not self.conn:
            raise RuntimeError("Database not connected.")
        await self.conn.commit()

    async def rollback(self) -> None:
        """Rollback the current transaction."""
        if not self.conn:
            raise RuntimeError("Database not connected.")
        await self.conn.rollback()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Async context manager: BEGIN → body → COMMIT (or ROLLBACK on error).

        Usage::

            async with db.transaction():
                await db.execute("INSERT INTO ...", (...))
                await db.execute("UPDATE ...", (...))
        """
        if not self.conn:
            raise RuntimeError("Database not connected.")
        await self.conn.execute("BEGIN")
        try:
            yield
            await self.conn.commit()
        except Exception:
            await self.conn.rollback()
            raise

    async def fetch_one(self, sql: str, params: tuple = ()) -> dict | None:
        """Fetch a single row as a dict, or None."""
        if not self.conn:
            raise RuntimeError("Database not connected.")
        cursor = await self.conn.execute(sql, params)
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        """Fetch all matching rows as a list of dicts."""
        if not self.conn:
            raise RuntimeError("Database not connected.")
        cursor = await self.conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def init_schema(self) -> None:
        """Create all tables if they don't exist."""
        if not self.conn:
            raise RuntimeError("Database not connected.")
        await self.conn.executescript(SCHEMA_SQL)
        await self.conn.commit()
