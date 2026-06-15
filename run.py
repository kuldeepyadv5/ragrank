#!/usr/bin/env python3
"""Entry point to run API, Streamlit dashboard, or reranker training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_api(host: str | None = None, port: int | None = None) -> None:
    from src.config import get_config

    cfg = get_config().app
    h = host or cfg.api_host
    p = port or cfg.api_port
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
    )


def run_dashboard(port: int | None = None) -> None:
    from src.config import get_config

    cfg = get_config().app
    p = port or cfg.streamlit_port
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

    dash_parser = sub.add_parser("dashboard", help="Start Streamlit dashboard")
    dash_parser.add_argument("--port", type=int, default=None)

    train_parser = sub.add_parser("train", help="Train PyTorch reranker")
    train_parser.add_argument("--epochs", type=int, default=None)

    sub.add_parser("all", help="Print instructions to run api + dashboard")

    args = parser.parse_args()

    if args.command == "api":
        run_api(args.host, args.port)
    elif args.command == "dashboard":
        run_dashboard(args.port)
    elif args.command == "train":
        run_train(args.epochs)
    elif args.command == "all":
        print("Run in separate terminals:")
        print("  python run.py api")
        print("  python run.py dashboard")


if __name__ == "__main__":
    main()
