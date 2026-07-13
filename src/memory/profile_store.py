"""L3 — User long-term profile storage.

Profiles accumulate across sessions without decay.  Only the Supervisor
reads and writes this layer.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.db.connection import DatabaseManager


class ProfileStore:
    """CRUD for the ``user_profiles`` table (L3)."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    # ── Read ───────────────────────────────────────────────────────
    async def get_profile(self, user_id: str) -> dict | None:
        """Return the stored profile dict or None."""
        if not user_id:
            return None
        row = await self.db.fetch_one(
            "SELECT * FROM user_profiles WHERE user_id = ?",
            (user_id,),
        )
        return dict(row) if row else None

    # ── Write / Upsert ─────────────────────────────────────────────
    async def ensure_exists(self, user_id: str) -> dict:
        """Create a default profile if none exists. Returns the profile dict."""
        if not user_id:
            return self._empty_profile(user_id)

        existing = await self.get_profile(user_id)
        if existing:
            return existing

        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """INSERT OR IGNORE INTO user_profiles
               (user_id, preferences, common_issues, total_sessions,
                total_resolved, total_escalated, favorite_agent,
                sentiment_trend, last_session_at, created_at, updated_at)
               VALUES (?, '{}', '[]', 0, 0, 0, NULL, 'neutral', NULL, ?, ?)""",
            (user_id, now, now),
        )
        await self.db.commit()
        return await self.get_profile(user_id) or self._empty_profile(user_id)

    async def upsert_profile(self, user_id: str, updates: dict) -> None:
        """Merge *updates* into the existing profile.

        *updates* is a flat dict of column → value.  JSON columns
        (preferences, common_issues) should be passed as serialised strings.
        """
        if not user_id:
            return

        await self.ensure_exists(user_id)

        now = datetime.now(timezone.utc).isoformat()
        set_clauses = ["updated_at = ?"]
        params: list = [now]

        for key, value in updates.items():
            if key in ("user_id", "created_at"):
                continue
            set_clauses.append(f"{key} = ?")
            params.append(value)

        params.append(user_id)
        sql = f"UPDATE user_profiles SET {', '.join(set_clauses)} WHERE user_id = ?"
        await self.db.execute(sql, tuple(params))
        await self.db.commit()

    async def increment_stats(
        self,
        user_id: str,
        *,
        resolved: bool = False,
        escalated: bool = False,
        agent: str = "",
    ) -> None:
        """Increment session counters after a conversation ends."""
        if not user_id:
            return

        await self.ensure_exists(user_id)

        profile = await self.get_profile(user_id)
        if profile is None:
            return

        now = datetime.now(timezone.utc).isoformat()

        total = int(profile.get("total_sessions", 0)) + 1
        total_resolved = int(profile.get("total_resolved", 0)) + (1 if resolved else 0)
        total_escalated = int(profile.get("total_escalated", 0)) + (1 if escalated else 0)

        # Track favorite agent (simple frequency counter embedded in JSON)
        if agent and agent not in ("supervisor", "human_agent", ""):
            common = json.loads(profile.get("common_issues", "[]"))
            found = False
            for entry in common:
                if entry.get("type") == agent:
                    entry["count"] = entry.get("count", 0) + 1
                    found = True
                    break
            if not found:
                common.append({"type": agent, "count": 1})
            common_issues = json.dumps(common, ensure_ascii=False)

            # Determine favorite agent
            best = max(common, key=lambda x: x["count"])
            favorite_agent = best["type"]
        else:
            common_issues = profile.get("common_issues", "[]")
            favorite_agent = profile.get("favorite_agent")

        await self.db.execute(
            """UPDATE user_profiles
               SET total_sessions = ?, total_resolved = ?, total_escalated = ?,
                   common_issues = ?, favorite_agent = ?,
                   last_session_at = ?, updated_at = ?
               WHERE user_id = ?""",
            (
                total, total_resolved, total_escalated,
                common_issues, favorite_agent,
                now, now, user_id,
            ),
        )
        await self.db.commit()

    # ── Helpers ────────────────────────────────────────────────────
    @staticmethod
    def _empty_profile(user_id: str) -> dict:
        return {
            "user_id": user_id,
            "preferences": "{}",
            "common_issues": "[]",
            "total_sessions": 0,
            "total_resolved": 0,
            "total_escalated": 0,
            "favorite_agent": None,
            "sentiment_trend": "neutral",
            "last_session_at": None,
            "created_at": "",
            "updated_at": "",
        }

    @staticmethod
    def format_for_context(profile: dict | None) -> str:
        """Format a user profile into a compact text block for the Supervisor."""
        if not profile:
            return ""

        parts = ["## User Profile (Long-term)"]

        total = profile.get("total_sessions", 0)
        resolved = profile.get("total_resolved", 0)
        escalated = profile.get("total_escalated", 0)

        if total > 0:
            parts.append(
                f"- Sessions: {total} total, {resolved} resolved, "
                f"{escalated} escalated"
            )

        fav = profile.get("favorite_agent")
        if fav:
            parts.append(f"- Most-used service: {fav}")

        sentiment = profile.get("sentiment_trend", "neutral")
        parts.append(f"- Sentiment trend: {sentiment}")

        # Common issues
        try:
            common = json.loads(profile.get("common_issues", "[]"))
        except (json.JSONDecodeError, TypeError):
            common = []
        if common:
            sorted_issues = sorted(common, key=lambda x: x.get("count", 0), reverse=True)
            issues_str = ", ".join(
                f"{i['type']}({i['count']}×)" for i in sorted_issues[:3]
            )
            parts.append(f"- Frequent topics: {issues_str}")

        last = profile.get("last_session_at", "")
        if last:
            parts.append(f"- Last visit: {last[:10]}")

        return "\n".join(parts)
