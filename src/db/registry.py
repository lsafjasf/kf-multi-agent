"""Shared DB registry — single module-level DatabaseManager reference.

Every module that needs DB access (tools, human_agent) imports
``get_db`` / ``set_db`` from here instead of duplicating the
module-level ``_db`` + ``set_db`` + ``_get_db`` pattern.
"""

from __future__ import annotations

from src.db.connection import DatabaseManager

_db: DatabaseManager | None = None


def set_db(db: DatabaseManager) -> None:
    """Set the shared database manager. Called at graph build time."""
    global _db
    _db = db


def get_db() -> DatabaseManager:
    """Return the shared database manager."""
    if _db is None:
        raise RuntimeError("Database not initialized. Call set_db() first.")
    return _db
