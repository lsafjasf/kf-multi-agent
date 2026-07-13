"""Application configuration via environment variables with sensible defaults.

All values are read from the environment at access time, not cached at import.
Call ``get_config()`` to get the singleton; use ``reload_config()`` to force a refresh.
"""

from __future__ import annotations

import os
from pathlib import Path


class Config:
    """Central configuration sourced from environment variables.

    Every attribute reads ``os.getenv`` at access time, so changing an
    environment variable and calling ``reload_config()`` (or creating a new
    Config) picks up the new value.
    """

    # ── Database ──────────────────────────────────────────────────
    @property
    def db_path(self) -> Path:
        return Path(os.getenv("DB_PATH", "data/shopfast.db"))

    # ── Model ─────────────────────────────────────────────────────
    @property
    def model_name(self) -> str:
        return os.getenv("MODEL_NAME", "gpt-4o")

    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "sk-...")

    @property
    def openai_base_url(self) -> str | None:
        return os.getenv("OPENAI_BASE_URL") or None

    @property
    def model_temperature(self) -> float:
        return float(os.getenv("MODEL_TEMPERATURE", "0.0"))

    # ── Escalation ────────────────────────────────────────────────
    @property
    def max_retries(self) -> int:
        return int(os.getenv("MAX_RETRIES", "3"))

    # ── Memory ────────────────────────────────────────────────────
    @property
    def checkpoint_db_path(self) -> Path:
        return Path(os.getenv("CHECKPOINT_DB_PATH", "data/checkpoints.db"))

    @property
    def session_ttl_hours(self) -> int:
        """Session TTL in hours. Sessions older than this are cleaned up.
        Default 72h (3 days).  Set to 0 to disable cleanup.
        """
        return int(os.getenv("SESSION_TTL_HOURS", "72"))

    @property
    def memory_max_recent_sessions(self) -> int:
        return int(os.getenv("MEMORY_MAX_RECENT_SESSIONS", "5"))

    @property
    def memory_decay_days_warn(self) -> int:
        return 30

    @property
    def memory_decay_days_low(self) -> int:
        return 60

    @property
    def memory_decay_days_archive(self) -> int:
        return 90

    # ── Mock API ──────────────────────────────────────────────────
    @property
    def mock_api_failure_rate(self) -> float:
        return float(os.getenv("MOCK_API_FAILURE_RATE", "0.0"))

    # ── CORS ──────────────────────────────────────────────────────
    @property
    def cors_origins(self) -> list[str]:
        raw = os.getenv("CORS_ORIGINS", "*")
        return [o.strip() for o in raw.split(",") if o.strip()]

    # ── Rate limiting ─────────────────────────────────────────────
    @property
    def rate_limit_per_minute(self) -> int:
        return int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))


_config: Config | None = None


def get_config() -> Config:
    """Return the singleton Config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> Config:
    """Force a fresh Config instance (re-reads all env vars)."""
    global _config
    _config = Config()
    return _config
