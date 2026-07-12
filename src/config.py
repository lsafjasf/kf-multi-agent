"""Application configuration via environment variables with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Central configuration sourced from environment variables."""

    # ── Database ──────────────────────────────────────────────────
    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("DB_PATH", "data/shopfast.db"))
    )

    # ── Model ─────────────────────────────────────────────────────
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "sk-...")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    model_temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.0"))

    # ── Escalation ────────────────────────────────────────────────
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))

    # ── Mock API ──────────────────────────────────────────────────
    mock_api_failure_rate: float = float(os.getenv("MOCK_API_FAILURE_RATE", "0.0"))


_config: Config | None = None


def get_config() -> Config:
    """Return the singleton Config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
