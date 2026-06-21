"""Terminal workflow logging for FastAPI backend and RAG pipeline."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("ragrank.backend.workflow")

_MAX_CHARS = 2000
_STEP = 0


def setup_backend_logging() -> None:
    """Enable structured workflow logs in the API terminal."""
    if logger.handlers:
        return

    enabled = os.getenv("API_WORKFLOW_DEBUG", "true").lower() == "true"
    if not enabled:
        logger.setLevel(logging.WARNING)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _enabled() -> bool:
    setup_backend_logging()
    return logger.isEnabledFor(logging.INFO)


def _truncate(value: Any, limit: int = _MAX_CHARS) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [{len(value) - limit} more chars]"
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return [_truncate(item, 300 if "text" in item else limit) for item in value[:5]] + (
            [{"note": f"... {len(value) - 5} more items"}] if len(value) > 5 else []
        )
    if isinstance(value, dict):
        return {k: _truncate(v, 300 if k == "text" else limit) for k, v in value.items()}
    return value


def _dump(data: Any) -> str:
    if isinstance(data, (dict, list)):
        return json.dumps(_truncate(data), indent=2, default=str)
    return str(_truncate(data))


def log_banner(title: str) -> None:
    if not _enabled():
        return
    logger.info("=" * 72)
    logger.info(title)
    logger.info("=" * 72)


def log_endpoint_request(method: str, path: str, payload: Any = None) -> None:
    if not _enabled():
        return
    log_banner(f"FASTAPI REQUEST  {method} {path}")
    if payload is not None:
        logger.info("Incoming payload:\n%s", _dump(payload))


def log_endpoint_response(method: str, path: str, payload: Any, elapsed_ms: float) -> None:
    if not _enabled():
        return
    logger.info("-" * 72)
    logger.info("FASTAPI RESPONSE %s %s | %.1f ms", method, path, elapsed_ms)
    logger.info("Outgoing payload:\n%s", _dump(payload))
    logger.info("=" * 72)


def reset_pipeline_steps() -> None:
    global _STEP
    _STEP = 0


def log_pipeline_step(title: str, **details: Any) -> None:
    """Log a numbered pipeline stage with key/value details."""
    global _STEP
    if not _enabled():
        return
    _STEP += 1
    logger.info("")
    logger.info("[PIPELINE STEP %d] %s", _STEP, title)
    for key, value in details.items():
        if isinstance(value, (dict, list)):
            logger.info("  %s:\n%s", key, _dump(value))
        else:
            logger.info("  %s: %s", key, value)


def log_workflow(message: str, **details: Any) -> None:
    if not _enabled():
        return
    logger.info(message)
    for key, value in details.items():
        logger.info("  %s: %s", key, _truncate(value))
