"""Memory consolidation — the glue between L1, L2, and L3.

This module is **only invoked from the SuperGraph level** (never from
specialist sub-agents).  It handles:

1. **L2a update** — rule-based running summary accumulation during the session.
2. **L2b persistence** — LLM-generated historical summary at session end.
3. **L3 update** — LLM-assisted incremental profile update at session end.
4. **Context builder** — format L2b + L3 into a prompt prefix for the Supervisor.
"""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.logging import get_logger
from src.retry import with_retries
from src.memory.session_store import SessionStore
from src.memory.profile_store import ProfileStore
from src.db.connection import DatabaseManager

if TYPE_CHECKING:
    from src.state import CustomerServiceState

logger = get_logger("memory")


# ── LLM Prompts ────────────────────────────────────────────────────────
SESSION_SUMMARY_PROMPT = """\
You are a memory consolidator for a customer service system.  Your job is to
read a full conversation transcript and produce a **concise session summary**
that will help a supervisor route similar future requests.

## Output Format
Reply with EXACTLY this JSON and nothing else:
{
  "summary": "<2-4 sentences capturing: what the customer needed, which agents helped, how it was resolved>",
  "key_entities": ["list", "of", "order/shipment/refund/ticket IDs mentioned"],
  "resolution_type": "resolved | escalated | abandoned"
}

## Rules
- summary: Be specific — mention concrete order IDs, tracking numbers, ticket IDs.
- key_entities: ONLY include IDs that actually appeared in the conversation.
- resolution_type:
  - "resolved" = the issue was solved by an agent
  - "escalated" = a support ticket was created (human_agent)
  - "abandoned" = the conversation ended without clear resolution
"""

PROFILE_UPDATE_PROMPT = """\
You maintain a long-term user profile for a customer service system.
Given the **existing profile** and a **new session summary**, produce an
**updated profile** that incorporates the new information.

## Existing Profile (JSON)
{existing_profile}

## New Session Summary
{new_summary}

## Output Format
Reply with EXACTLY this JSON and nothing else:
{
  "preferences": {{...}},
  "common_issues": [{{"type": "order | logistics | refund", "count": N}}],
  "sentiment_trend": "positive | neutral | negative"
}

## Rules
- preferences: Merge new observations with old ones.  If the customer expressed
  a language preference, contact channel preference, or any recurring need,
  record it here as key-value pairs.
- common_issues: Increment counts for the issue type(s) seen in this session.
  Keep existing entries, add new ones if a new type appears.
- sentiment_trend: Based on the tone of ALL sessions seen so far.  If the
  customer was angry/frustrated in this session, shift toward "negative".
  If they were satisfied/thankful, shift toward "positive".  If neutral or
  mixed, leave as is or return "neutral".
- DO NOT remove information from the existing profile — only ADD or UPDATE.
"""


class MemoryConsolidator:
    """Orchestrates L2a / L2b / L3 memory operations.

    Instantiated once at graph build time and passed to the Supervisor
    (for L2a updates + context building) and the memory_consolidator node
    (for L2b persistence + L3 updates).
    """

    def __init__(
        self,
        model: BaseChatModel,
        session_store: SessionStore,
        profile_store: ProfileStore,
        db: DatabaseManager,
    ) -> None:
        self.model = model
        self.session_store = session_store
        self.profile_store = profile_store
        self.db = db

        # Retry-wrapped version of model.ainvoke
        self._ainvoke = with_retries(max_attempts=3)(model.ainvoke)

    # ── L2a: Running summary (in-state, rule-based) ─────────────────
    async def update_running_summary(self, state: "CustomerServiceState") -> str:
        """Accumulate a short running summary of the current session.

        This is rule-based (no LLM call) — it stitches together agent
        transitions and key events from the state.  The Supervisor calls
        this each time it is re-activated.
        """
        current = state.running_summary or ""

        # Extract the most recent AI message content (last 200 chars)
        new_line = ""
        messages = state.messages
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai" and hasattr(msg, "content"):
                content = str(msg.content)
                if content and not content.startswith("[Handoff") and not content.startswith("[Escalating"):
                    new_line = content[:200].replace("\n", " ")
                    break

        active = state.active_agent
        if active and new_line:
            entry = f"[{active}] {new_line}"
            if current:
                # Keep last 3 entries to bound size
                entries = current.split("\n")
                entries.append(entry)
                current = "\n".join(entries[-3:])
            else:
                current = entry

        return current

    # ── L2b + L3: End-of-session consolidation (LLM) ────────────────
    async def consolidate(self, state: "CustomerServiceState") -> dict:
        """Run at session end (from ``memory_consolidator`` node).

        1. Generate L2b summary via LLM → persist.
        2. Update L3 profile via LLM → persist.
        3. Archive old sessions.
        """
        user_id = state.user_id or ""
        session_id = state.session_id or ""

        # Guard: no user_id → skip persistence
        if not user_id:
            return {
                "running_summary": "",
                "memory_context": "",
            }

        logger.debug("Consolidating session %s for user %s", session_id, user_id)

        # ── 1. Generate and persist L2b summary ──────────────────
        l2b_result = await self._generate_session_summary(state)
        if l2b_result:
            # Count messages (human + ai pairs)
            msg_count = sum(
                1 for m in state.messages
                if hasattr(m, "type") and m.type in ("human", "ai")
            )
            duration_ms = None
            if state.session_started_at > 0:
                duration_ms = int((time.perf_counter() - state.session_started_at) * 1000)

            resolution = "escalated" if state.support_ticket_id else "resolved"

            await self.session_store.save(
                session_id=session_id,
                user_id=user_id,
                summary=l2b_result["summary"],
                key_entities=l2b_result.get("key_entities", []),
                resolution=resolution,
                ticket_id=state.support_ticket_id,
                message_count=msg_count,
                duration_ms=duration_ms,
            )
            logger.info("L2b saved: session=%s resolution=%s", session_id, resolution)

        # ── 2. Update L3 profile ─────────────────────────────────
        await self._update_user_profile(user_id, l2b_result)
        await self.profile_store.increment_stats(
            user_id,
            resolved=bool(state.resolved),
            escalated=bool(state.support_ticket_id),
            agent=state.active_agent,
        )

        # ── 3. Archive old sessions ──────────────────────────────
        archived = await self.session_store.archive_old_sessions(user_id)
        if archived:
            logger.debug("Archived %d old sessions for user %s", archived, user_id)

        return {
            "running_summary": "",
            "memory_context": "",
        }

    # ── Internal helpers ────────────────────────────────────────────
    async def _generate_session_summary(self, state: "CustomerServiceState") -> dict | None:
        """Invoke LLM to summarise the full conversation."""
        # Build a transcript from messages
        transcript_lines: list[str] = []
        for msg in state.messages:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                role = "Customer" if msg.type == "human" else "Agent"
                content = str(msg.content)
                if content:
                    transcript_lines.append(f"{role}: {content[:500]}")
        transcript = "\n".join(transcript_lines)

        if not transcript.strip():
            return None

        try:
            # Use retry-wrapped invoke
            response = await self._ainvoke([
                SystemMessage(content=SESSION_SUMMARY_PROMPT),
                HumanMessage(content=f"Conversation transcript:\n\n{transcript}"),
            ])
            text = str(response.content) if hasattr(response, "content") else str(response)
            return self._parse_json_response(text)
        except Exception:
            logger.exception("Failed to generate session summary")
            return None

    async def _update_user_profile(
        self, user_id: str, session_summary: dict | None
    ) -> None:
        """Invoke LLM to merge the new session into the user profile."""
        if not session_summary:
            return

        existing = await self.profile_store.get_profile(user_id)
        existing_json = json.dumps(existing, ensure_ascii=False, default=str) if existing else "{}"
        new_summary_text = session_summary.get("summary", "")

        try:
            response = await self._ainvoke([
                SystemMessage(content=PROFILE_UPDATE_PROMPT),
                HumanMessage(content=(
                    f"Existing profile:\n{existing_json}\n\n"
                    f"New session summary:\n{new_summary_text}"
                )),
            ])
            text = str(response.content) if hasattr(response, "content") else str(response)
            updates = self._parse_json_response(text)
            if updates:
                await self.profile_store.upsert_profile(
                    user_id,
                    {
                        "preferences": json.dumps(updates.get("preferences", {}), ensure_ascii=False),
                        "common_issues": json.dumps(updates.get("common_issues", []), ensure_ascii=False),
                        "sentiment_trend": updates.get("sentiment_trend", "neutral"),
                    },
                )
                logger.debug("L3 profile updated for user %s", user_id)
        except Exception:
            logger.debug("Profile update skipped (best-effort)")

    # ── Context builder (for Supervisor injection) ──────────────────
    async def build_memory_context(self, user_id: str) -> str:
        """Load L2b + L3 and format them into a prompt prefix for the Supervisor.

        Called at the **start** of a new session (or first Supervisor activation).
        """
        if not user_id:
            return ""

        parts: list[str] = []

        # L3: User profile
        profile = await self.profile_store.get_profile(user_id)
        profile_text = ProfileStore.format_for_context(profile)
        if profile_text:
            parts.append(profile_text)

        # L2b: Active historical sessions
        sessions = await self.session_store.get_active_sessions(user_id)
        sessions_text = SessionStore.format_for_context(sessions)
        if sessions_text:
            parts.append(sessions_text)

        return "\n\n".join(parts) if parts else ""

    # ── Utility ─────────────────────────────────────────────────────
    @staticmethod
    def _parse_json_response(text: str) -> dict | None:
        """Extract a JSON object from an LLM response text."""
        # Try to find a JSON object in the response
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if not match:
            # Try to find a larger JSON object (nested braces)
            match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None
