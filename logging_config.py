"""
Structured (JSON) logging setup, shared by every module in the project.
Usage:
    from app.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("something happened", extra={"session_id": sid})
"""
import logging
import sys

from pythonjsonlogger import jsonlogger

from app.config import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.handlers = [handler]

    # Quiet down noisy third-party libraries.
    for noisy in ("httpx", "urllib3", "chromadb", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
