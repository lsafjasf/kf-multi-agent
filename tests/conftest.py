"""Shared fixtures for the ShopFast evaluation framework.

Provides:
- temp_db:  a fresh copy of the seed database (mutations don't corrupt the master)
- model:    ChatOpenAI instance from config
- graph:    compiled customer-service graph wired to the temp DB
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from src.config import get_config
from src.db.connection import DatabaseManager
from src.db.seed import seed_database
from src.graph import build_customer_service_graph


# ── Helpers ────────────────────────────────────────────────────────────
def _master_db() -> Path:
    """Path to the canonical seed database."""
    # PROJECT_ROOT/data/shopfast.db
    root = Path(__file__).resolve().parent.parent
    return root / "data" / "shopfast.db"


# ── Fixtures ───────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def temp_db_path():
    """Copy the master seed DB to a temp file.  Mutations stay isolated."""
    master = _master_db()
    if not master.exists():
        pytest.fail(
            f"Master database not found at {master}. "
            f"Run `python main.py --seed-only` first."
        )

    fd, tmp_path = tempfile.mkstemp(suffix=".db", prefix="shopfast_eval_")
    os.close(fd)

    shutil.copy2(master, tmp_path)

    yield tmp_path

    # Cleanup
    try:
        os.unlink(tmp_path)
    except OSError:
        pass


@pytest_asyncio.fixture
async def db(temp_db_path):
    """Async DatabaseManager wrapping the temp DB copy."""
    async with DatabaseManager(temp_db_path) as db_mgr:
        await db_mgr.init_schema()
        yield db_mgr


@pytest_asyncio.fixture
def model():
    """ChatOpenAI model from environment config."""
    config = get_config()
    if config.openai_api_key in ("sk-...", ""):
        pytest.skip("OPENAI_API_KEY not set — cannot create model")

    model_kwargs: dict = {
        "model": config.model_name,
        "temperature": config.model_temperature,
        "api_key": config.openai_api_key,
    }
    if config.openai_base_url:
        model_kwargs["base_url"] = config.openai_base_url

    return ChatOpenAI(**model_kwargs)


@pytest_asyncio.fixture
async def graph(model, db):
    """Compiled customer-service graph with temp DB and model."""
    g, _ckpt = await build_customer_service_graph(model, db)
    return g
