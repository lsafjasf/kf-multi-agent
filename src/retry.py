"""LLM retry utility — wraps model.ainvoke with exponential backoff.

Uses ``tenacity`` to retry on transient API errors (timeouts, rate limits,
server errors).  Non-retryable errors (auth, bad request) fail immediately.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.logging import get_logger

logger = get_logger("retry")

T = TypeVar("T")

# Exception types that are worth retrying.
RETRYABLE = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)

# Additional retryable exceptions from common HTTP libraries.
try:
    from httpx import HTTPStatusError, ReadTimeout, ConnectTimeout
    RETRYABLE = RETRYABLE + (HTTPStatusError, ReadTimeout, ConnectTimeout)
except ImportError:
    pass

try:
    from openai import APITimeoutError, APIConnectionError, InternalServerError
    RETRYABLE = RETRYABLE + (APITimeoutError, APIConnectionError, InternalServerError)
except ImportError:
    pass


def with_retries(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator: retry an async function on transient errors.

    Parameters
    ----------
    max_attempts : int
        Total attempts before giving up (default 3).
    min_wait : float
        Initial backoff in seconds (default 1s).
    max_wait : float
        Maximum backoff in seconds (default 10s).
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(RETRYABLE),
            before_sleep=before_sleep_log(logger, "WARNING"),
            reraise=True,
        )
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await func(*args, **kwargs)

        wrapper.__name__ = func.__name__  # preserve name for debugging
        return wrapper

    return decorator
