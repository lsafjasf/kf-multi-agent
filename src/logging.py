"""Structured logging utility.

Provides ``get_logger(name)`` returning a pre-configured logger.
Format: ``2026-07-13 10:30:00 [INFO   ] [server] message``
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with the project's standard format."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name("shopfast-stderr")

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-7s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger
