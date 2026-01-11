"""Logging utilities.

Key goal:
- Each step logs clearly so the caller can locate failures quickly.
- Keep logging config minimal; allow integration into the caller's logging if needed.
"""
from __future__ import annotations

import logging
import os

_DEFAULT_LEVEL = os.environ.get("LLMKIT_LOG_LEVEL", "INFO").upper()

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # If already configured elsewhere, do not attach handlers again.
    if logger.handlers:
        return logger

    logger.setLevel(_DEFAULT_LEVEL)

    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(name)s:%(lineno)d - %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)

    return logger

def log_step(logger: logging.Logger, step: str, msg: str):
    logger.info("[STEP %s] %s", step, msg)
