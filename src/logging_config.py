import logging
import os

LOG_LEVEL = os.getenv("LLMKIT_LOG_LEVEL", "INFO").upper()

logger = logging.getLogger("llmkit")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)

logger.setLevel(LOG_LEVEL)
