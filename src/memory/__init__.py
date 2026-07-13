"""Three-layer memory system for the ShopFast customer service agent system.

Layers
------
L1  Short-term (conversation context)
    AsyncSqliteSaver checkpoints — full message history within a session.
    All agents can read; system auto-writes on every state transition.

L2a Current-session summary (in-state, ephemeral)
    ``CustomerServiceState.running_summary`` — Supervisor accumulates a
    running summary during the session.  Dies with the session.

L2b Historical session summaries (SQLite, cross-session with decay)
    ``conversation_sessions`` table — Supervisor writes a summary at session
    end.  Weights decay in business logic: 30 d → 0.5, 60 d → 0.1,
    90 d → archived.  **Only the Supervisor reads L2b.**

L3  User long-term profiles (SQLite, permanent accumulation)
    ``user_profiles`` table — incremental stats and preferences built over
    many sessions.  **Only the Supervisor reads and writes L3.**
"""

from src.memory.session_store import SessionStore
from src.memory.profile_store import ProfileStore
from src.memory.consolidator import MemoryConsolidator

__all__ = ["SessionStore", "ProfileStore", "MemoryConsolidator"]
