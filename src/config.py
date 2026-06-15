"""Central configuration and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    """Read env var; treat blank values as unset."""
    return os.getenv(key) or default

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
MODELS_DIR = DATA_DIR / "models"
MLFLOW_DIR = DATA_DIR / "mlruns"
LOGS_DIR = DATA_DIR / "logs"

for _dir in (DATA_DIR, INDEX_DIR, MODELS_DIR, MLFLOW_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


@dataclass
class RerankerConfig:
    model_name: str = os.getenv("RERANKER_MODEL", "distilbert-base-uncased")
    max_length: int = int(os.getenv("RERANKER_MAX_LENGTH", "256"))
    batch_size: int = int(os.getenv("RERANKER_BATCH_SIZE", "16"))
    learning_rate: float = float(os.getenv("RERANKER_LR", "2e-5"))
    epochs: int = int(os.getenv("RERANKER_EPOCHS", "3"))
    loss: Literal["bce", "margin_mse"] = os.getenv(
        "RERANKER_LOSS", "bce"
    )  # type: ignore[assignment]
    margin: float = float(os.getenv("RERANKER_MARGIN", "0.5"))
    checkpoint_dir: Path = field(default_factory=lambda: MODELS_DIR / "reranker")


@dataclass
class RetrieverConfig:
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))
    use_ivf: bool = os.getenv("FAISS_USE_IVF", "false").lower() == "true"
    index_path: Path = field(default_factory=lambda: INDEX_DIR)


@dataclass
class LLMConfig:
    provider: Literal["qwen", "gemini", "openai", "ollama", "mock"] = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "mock")  # type: ignore[return-value]
    )
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen-plus"))
    qwen_api_key: str = field(
        default_factory=lambda: _env("DASHSCOPE_API_KEY") or _env("QWEN_API_KEY")
    )
    qwen_base_url: str = field(
        default_factory=lambda: os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    )
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "1024"))
    )


@dataclass
class TrackingConfig:
    backend: Literal["mlflow", "tensorboard", "none"] = os.getenv(
        "TRACKING_BACKEND", "mlflow"
    )  # type: ignore[assignment]
    mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", str(MLFLOW_DIR))
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT", "rag-reranker")


@dataclass
class AppConfig:
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")


@dataclass
class Config:
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    app: AppConfig = field(default_factory=AppConfig)


def get_config() -> Config:
    return Config()
