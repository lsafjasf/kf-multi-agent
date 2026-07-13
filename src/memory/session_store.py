"""L2b — Historical session summary storage with time-based decay.

Decay is computed **at read time** in business logic — SQLite has no
scheduled tasks.  The rules (tunable via Config):

- 0–30 days  → weight = 1.0  (full relevance)
- 30–60 days → weight = 0.5  (moderate relevance)
- 60–90 days → weight = 0.1  (low relevance)
- >90 days   → archived = 1  (excluded from active queries)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.db.connection import DatabaseManager


class SessionStore:
    """CRUD for the ``conversation_sessions`` table (L2b)."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    # ── Write ──────────────────────────────────────────────────────
    async def save(
        self,
        *,
        session_id: str,
        user_id: str,
        summary: str,
        key_entities: list[str] | None = None,
        resolution: str = "resolved",
        ticket_id: str | None = None,
        message_count: int = 0,
        duration_ms: int | None = None,
    ) -> None:
        """Persist a new historical session summary."""
        await self.db.execute(
            """INSERT OR REPLACE INTO conversation_sessions
               (id, user_id, summary, key_entities, resolution,
                ticket_id, message_count, duration_ms, weight, archived, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1.0, 0, ?)""",
            (
                session_id,
                user_id,
                summary,
                json.dumps(key_entities or [], ensure_ascii=False),
                resolution,
                ticket_id,
                message_count,
                duration_ms,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await self.db.commit()

    # ── Read (with decay) ──────────────────────────────────────────
    async def get_active_sessions(
        self, user_id: str, limit: int = 5
    ) -> list[dict]:
        """Return active (non-archived) sessions for *user_id*.

        We fetch more rows than *limit* because some may have decayed
        to zero weight during the read-side computation.
        """
        rows = await self.db.fetch_all(
            """SELECT * FROM conversation_sessions
               WHERE user_id = ? AND archived = 0
               ORDER BY created_at DESC
               LIMIT ?""",
            (user_id, limit * 3),
        )

        now = datetime.now(timezone.utc)
        results: list[dict] = []

        for row in rows:
            row = dict(row)
            created = datetime.fromisoformat(row["created_at"])
            days = (now - created).days

            if days > 90:
                # Mark for archival (best-effort; caller should also call
                # archive_old_sessions periodically).
                row["archived"] = 1
                row["weight"] = 0.0
            elif days > 60:
                row["weight"] = 0.1
            elif days > 30:
                row["weight"] = 0.5
            else:
                row["weight"] = 1.0

            if row["weight"] > 0:
                results.append(row)

        results.sort(key=lambda r: r["weight"], reverse=True)
        return results[:limit]

    async def archive_old_sessions(self, user_id: str) -> int:
        """Archive sessions older than 90 days. Returns count of archived rows."""
        cutoff = datetime.now(timezone.utc).isoformat()
        # We compute cutoff date 90 days ago for the SQL comparison.
        # Using a simple approach: archive anything where we've already
        # detected it as decayed in get_active_sessions and it's now
        # being finalized. Actually, let's use a direct SQL approach:
        from datetime import timedelta
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()

        await self.db.execute(
            """UPDATE conversation_sessions
               SET archived = 1, weight = 0.0
               WHERE user_id = ? AND archived = 0 AND created_at < ?""",
            (user_id, cutoff_date),
        )
        await self.db.commit()

        row = await self.db.fetch_one(
            "SELECT changes() as cnt",
        )
        return row["cnt"] if row else 0

    async def get_session_count(self, user_id: str) -> int:
        """Return total sessions (including archived) for a user."""
        row = await self.db.fetch_one(
            "SELECT COUNT(*) AS cnt FROM conversation_sessions WHERE user_id = ?",
            (user_id,),
        )
        return row["cnt"] if row else 0

    # ── Helpers ────────────────────────────────────────────────────
    @staticmethod
    def format_for_context(sessions: list[dict]) -> str:
        """Format active sessions into a compact text block for the Supervisor.

        Sessions are expected to be pre-sorted by weight (highest first).
        """
        if not sessions:
            return ""

        lines = ["## Recent Conversation History (from past sessions)"]
        for i, s in enumerate(sessions[:5], 1):
            weight_label = (
                "★" if s.get("weight", 0) >= 1.0 else
                "☆" if s.get("weight", 0) >= 0.5 else "·"
            )
            date = s.get("created_at", "unknown")[:10]
            lines.append(
                f"{i}. {weight_label} [{date}] {s.get('summary', '')[:300]}"
            )
            resolution = s.get("resolution", "?")
            ticket = s.get("ticket_id")
            if ticket:
                lines[-1] += f" (→ {resolution}, ticket {ticket})"
            else:
                lines[-1] += f" (→ {resolution})"

        return "\n".join(lines)
