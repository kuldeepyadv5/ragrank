#!/usr/bin/env python3
"""Entry point to run API, Streamlit dashboard, or reranker training."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_api(host: str | None = None, port: int | None = None, debug_workflow: bool = True) -> None:
    from src.config import get_config

    cfg = get_config().app
    h = host or cfg.api_host
    p = port or cfg.api_port
    env = os.environ.copy()
    env["API_WORKFLOW_DEBUG"] = "true" if debug_workflow else "false"
    if debug_workflow:
        print(f"[api] Workflow debug logging ON → http://{h}:{p}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.app.main:app",
            "--host",
            h,
            "--port",
            str(p),
            "--reload",
        ],
        cwd=str(ROOT),
        check=True,
        env=env,
    )


def run_dashboard(port: int | None = None, debug_api: bool = True) -> None:
    import os

    from src.config import get_config

    cfg = get_config().app
    p = port or cfg.streamlit_port
    env = os.environ.copy()
    env["DASHBOARD_API_DEBUG"] = "true" if debug_api else "false"
    if debug_api:
        print(f"[dashboard] API debug logging ON → FastAPI at {cfg.api_base_url}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "src" / "app" / "dashboard.py"),
            "--server.port",
            str(p),
            "--server.address",
            "0.0.0.0",
        ],
        cwd=str(ROOT),
        check=True,
        env=env,
    )


def run_train(epochs: int | None = None) -> None:
    from src.reranker.train import train

    path = train()
    print(f"Training complete. Checkpoint: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise RAG System")
    sub = parser.add_subparsers(dest="command", required=True)

    api_parser = sub.add_parser("api", help="Start FastAPI backend")
    api_parser.add_argument("--host", default=None)
    api_parser.add_argument("--port", type=int, default=None)
    api_parser.add_argument(
        "--no-debug-workflow",
        action="store_true",
        help="Disable terminal workflow logging in the API server",
    )

    dash_parser = sub.add_parser("dashboard", help="Start Streamlit dashboard")
    dash_parser.add_argument("--port", type=int, default=None)
    dash_parser.add_argument(
        "--no-debug-api",
        action="store_true",
        help="Disable terminal logging of dashboard → FastAPI requests/responses",
    )

    train_parser = sub.add_parser("train", help="Train PyTorch reranker")
    train_parser.add_argument("--epochs", type=int, default=None)

    sub.add_parser("all", help="Print instructions to run api + dashboard")

    args = parser.parse_args()

    if args.command == "api":
        run_api(args.host, args.port, debug_workflow=not args.no_debug_workflow)
    elif args.command == "dashboard":
        run_dashboard(args.port, debug_api=not args.no_debug_api)
    elif args.command == "train":
        run_train(args.epochs)
    elif args.command == "all":
        print("Run in separate terminals:")
        print("  python run.py api")
        print("  python run.py dashboard")


if __name__ == "__main__":
    main()
