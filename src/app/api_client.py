"""HTTP client for Streamlit dashboard with terminal request/response logging."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx

from src.config import get_config

logger = logging.getLogger("ragrank.dashboard.api")

_MAX_LOG_CHARS = 2000


def setup_api_logging() -> None:
    """Configure terminal logging for dashboard → FastAPI calls."""
    if logger.handlers:
        return

    enabled = os.getenv("DASHBOARD_API_DEBUG", "true").lower() == "true"
    if not enabled:
        logger.setLevel(logging.WARNING)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _truncate(value: Any, limit: int = _MAX_LOG_CHARS) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [{len(value) - limit} more chars]"
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k == "answer" and isinstance(v, str) and len(v) > limit:
                out[k] = _truncate(v, limit)
            elif k == "contexts" and isinstance(v, list):
                out[k] = [
                    {**c, "text": _truncate(c.get("text", ""), 300)} for c in v[:5]
                ]
                if len(v) > 5:
                    out[k].append({"note": f"... {len(v) - 5} more contexts"})
            else:
                out[k] = v
        return out
    return value


def _log_block(title: str, data: Any) -> None:
    if isinstance(data, (dict, list)):
        text = json.dumps(data, indent=2, default=str)
    else:
        text = str(data)
    logger.info("%s\n%s", title, text)


def _log_request(method: str, url: str, headers: dict, body: Any = None) -> None:
    logger.info("=" * 72)
    logger.info("STREAMLIT → FASTAPI REQUEST")
    logger.info("Method : %s", method)
    logger.info("URL    : %s", url)
    _log_block("Request headers", dict(headers))
    if body is not None:
        _log_block("Request payload", _truncate(body))


def _log_response(method: str, url: str, resp: httpx.Response, elapsed_ms: float) -> None:
    logger.info("-" * 72)
    logger.info("FASTAPI → STREAMLIT RESPONSE")
    logger.info("Method : %s", method)
    logger.info("URL    : %s", url)
    logger.info("Status : %s", resp.status_code)
    logger.info("Time   : %.1f ms", elapsed_ms)
    _log_block("Response headers", dict(resp.headers))
    try:
        body = resp.json()
        _log_block("Response payload", _truncate(body))
    except Exception:
        _log_block("Response body (raw)", _truncate(resp.text))
    logger.info("=" * 72)


class DashboardApiClient:
    """Calls FastAPI backend and logs every request/response to the terminal."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        cfg = get_config().app
        self.base_url = (base_url or cfg.api_base_url).rstrip("/")
        self.timeout = timeout or float(os.getenv("API_CLIENT_TIMEOUT", "600"))
        setup_api_logging()

    def get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            _log_request("GET", url, dict(client.headers))
            t0 = time.perf_counter()
            resp = client.get(url)
            _log_response("GET", url, resp, (time.perf_counter() - t0) * 1000)
            resp.raise_for_status()
            return resp.json()

    def post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            headers = dict(client.headers)
            headers["Content-Type"] = "application/json"
            _log_request("POST", url, headers, payload)
            t0 = time.perf_counter()
            resp = client.post(url, json=payload)
            _log_response("POST", url, resp, (time.perf_counter() - t0) * 1000)
            resp.raise_for_status()
            return resp.json()

    def upload(self, files: list) -> dict:
        url = f"{self.base_url}/ingest"
        multipart = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        file_info = [{"filename": f.name, "size_bytes": len(f.getvalue())} for f in files]
        with httpx.Client(timeout=max(self.timeout, 300.0)) as client:
            _log_request("POST", url, dict(client.headers), {"files": file_info})
            t0 = time.perf_counter()
            resp = client.post(url, files=multipart)
            _log_response("POST", url, resp, (time.perf_counter() - t0) * 1000)
            resp.raise_for_status()
            return resp.json()
