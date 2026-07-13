#!/usr/bin/env python3
"""ShopFast Web Server — FastAPI + SSE streaming for the chat frontend.

Usage:
    python server.py
    python server.py --port 8080

Then open http://localhost:8000 in your browser.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.config import get_config
from src.logging import get_logger
from src.db.connection import DatabaseManager
from src.db.seed import seed_database
from src.graph import build_customer_service_graph, start_checkpoint_cleanup_task

logger = get_logger("server")

# ── Global state ────────────────────────────────────────────────────
_graph = None
_db: DatabaseManager | None = None
_ckpt_conn = None
_started_at: float = 0.0
_cleanup_task: asyncio.Task | None = None


# ── Pydantic models ─────────────────────────────────────────────────
MAX_MESSAGE_LENGTH = 4000
MAX_USER_ID_LENGTH = 50
MAX_SESSION_ID_LENGTH = 64

VALID_USER_ID = r"^[A-Za-z0-9_-]+$"
VALID_SESSION_ID = r"^[A-Za-z0-9_-]+$"


class ChatRequest(BaseModel):
    """Validated request body for POST /chat."""
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    user_id: str = Field(default="CUST-001", max_length=MAX_USER_ID_LENGTH, pattern=VALID_USER_ID)
    session_id: str = Field(
        default_factory=lambda: f"web-{uuid.uuid4().hex[:12]}",
        max_length=MAX_SESSION_ID_LENGTH,
        pattern=VALID_SESSION_ID,
    )


# ── Rate limiter (in-memory) ────────────────────────────────────────
class RateLimiter:
    """Simple sliding-window rate limiter per client IP."""

    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clients: dict[str, list[float]] = defaultdict(list)

    def _prune(self, client_ip: str, now: float) -> None:
        cutoff = now - self.window_seconds
        self._clients[client_ip] = [
            t for t in self._clients[client_ip] if t > cutoff
        ]

    def is_allowed(self, client_ip: str) -> bool:
        now = time.monotonic()
        self._prune(client_ip, now)
        if len(self._clients[client_ip]) >= self.max_requests:
            return False
        self._clients[client_ip].append(now)
        return True


# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + model + graph.  Shutdown: close connections."""
    global _graph, _db, _ckpt_conn, _started_at, _cleanup_task

    cfg = get_config()

    # Validate API key
    if cfg.openai_api_key in ("sk-...", ""):
        logger.error("OPENAI_API_KEY environment variable is required.")
        sys.exit(1)

    # Init DB
    _db = DatabaseManager(cfg.db_path)
    await _db.__aenter__()
    await _db.init_schema()
    await seed_database(_db)
    logger.info("Database ready at %s", cfg.db_path)

    # Init model
    model_kwargs = {
        "model": cfg.model_name,
        "temperature": cfg.model_temperature,
        "api_key": cfg.openai_api_key,
    }
    if cfg.openai_base_url:
        model_kwargs["base_url"] = cfg.openai_base_url
    model = ChatOpenAI(**model_kwargs)
    logger.info("Model: %s", cfg.model_name)

    # Build graph
    _graph, _ckpt_conn = await build_customer_service_graph(
        model, _db, cfg.checkpoint_db_path,
    )
    logger.info("Agent graph ready")

    _started_at = time.monotonic()

    # Start session cleanup background task
    if _ckpt_conn and cfg.session_ttl_hours > 0:
        _cleanup_task = await start_checkpoint_cleanup_task(
            _ckpt_conn,
            ttl_hours=cfg.session_ttl_hours,
            interval_seconds=3600,  # run every hour
        )

    yield  # ── app runs here ──

    # Shutdown
    logger.info("Shutting down...")
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
    if _ckpt_conn:
        await _ckpt_conn.close()
    if _db:
        await _db.__aexit__(None, None, None)


# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="ShopFast Customer Service", lifespan=lifespan)
STATIC_DIR = Path(__file__).parent / "static"

# CORS
cfg = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Rate limiter
_rate_limiter = RateLimiter(max_requests=cfg.rate_limit_per_minute, window_seconds=60.0)


# ── Routes ───────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the chat frontend."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


@app.get("/health")
async def health():
    """Health check endpoint — returns system status."""
    db_ok = _db is not None and _db.conn is not None
    graph_ok = _graph is not None
    all_ok = db_ok and graph_ok

    uptime_seconds = time.monotonic() - _started_at if _started_at > 0 else 0

    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "healthy" if all_ok else "degraded",
            "database": "ok" if db_ok else "disconnected",
            "graph": "ok" if graph_ok else "not_initialized",
            "uptime_seconds": round(uptime_seconds, 1),
        },
    )


@app.post("/chat")
async def chat(request: Request):
    """Send a message and stream the agent response via SSE."""
    # Rate limit check
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")

    # Parse and validate request body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    try:
        chat_req = ChatRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")

    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized yet.")

    return StreamingResponse(
        _stream_chat(chat_req, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Streaming ────────────────────────────────────────────────────────
async def _stream_chat(
    chat_req: ChatRequest, request: Request
) -> AsyncIterator[str]:
    """Execute the graph and yield SSE events for each step.

    Monitors ``request.is_disconnected`` to cancel early when the
    client closes the connection.
    """
    config = {"configurable": {"thread_id": chat_req.session_id}}
    input_state = {
        "messages": [HumanMessage(content=chat_req.message)],
        "user_id": chat_req.user_id,
        "session_id": chat_req.session_id,
        "session_started_at": time.perf_counter(),
    }

    # Create a task so we can cancel it on disconnect
    stream_task: asyncio.Task | None = None

    async def _run_graph():
        """Collect streaming chunks into a list so they can be consumed."""
        results = []
        async for chunk in _graph.astream(
            input_state, config=config, stream_mode="updates"
        ):
            # Check for client disconnect mid-stream
            if await request.is_disconnected():
                raise asyncio.CancelledError("Client disconnected")
            results.append(chunk)
        return results

    try:
        stream_task = asyncio.create_task(_run_graph())

        last_agent = None
        all_chunks = []

        while not stream_task.done():
            # Poll with a short timeout to check disconnect
            try:
                done, _pending = await asyncio.wait(
                    [stream_task], timeout=0.1
                )
                if done:
                    all_chunks = await stream_task
                    break
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    stream_task.cancel()
                    yield _sse_event("error", "Connection closed by client.")
                    return

        if stream_task.done() and not all_chunks:
            all_chunks = await stream_task

        for chunk in all_chunks:
            for node_name, node_output in chunk.items():
                if node_name == "memory_consolidator":
                    continue  # hide internal node from UI

                # Agent transition event
                if node_name != last_agent:
                    last_agent = node_name
                    yield _sse_event("route", {"agent": node_name})

                # Tool calls
                msgs = node_output.get("messages", [])
                for msg in msgs:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield _sse_event("tool", {
                                "name": tc["name"],
                                "args": str(tc.get("args", {}))[:120],
                            })
                    elif hasattr(msg, "name") and msg.name in ("handoff", "escalation"):
                        yield _sse_event("handoff", {
                            "content": str(getattr(msg, "content", ""))[:200],
                        })

        # Read final state
        if not await request.is_disconnected():
            final = await _graph.aget_state(config)
            result = final.values if final else {}

            final_msgs = [
                m for m in result.get("messages", [])
                if hasattr(m, "content") and getattr(m, "type", "") == "ai" and m.content
            ]
            final_text = str(final_msgs[-1].content) if final_msgs else ""

            yield _sse_event("done", {
                "message": final_text,
                "agent": result.get("active_agent", "?"),
                "resolved": result.get("resolved", False),
                "ticket_id": result.get("support_ticket_id"),
            })

    except asyncio.CancelledError:
        logger.info("Stream cancelled for session %s (client disconnected)", chat_req.session_id)
        yield _sse_event("error", "Request cancelled.")
    except Exception:
        logger.exception("Error processing chat for session %s", chat_req.session_id)
        yield _sse_event(
            "error",
            "An internal error occurred. Our team has been notified. "
            "Please try again or contact support@shopfast.com.",
        )


# ── Helpers ──────────────────────────────────────────────────────────
def _sse_event(event: str, data) -> str:
    """Format a Server-Sent Event line."""
    payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ShopFast Web Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
