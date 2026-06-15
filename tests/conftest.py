"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_text_file(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "A cross-encoder reranker scores query-document pairs jointly. "
        "FAISS enables fast vector similarity search. "
        "RAG combines retrieval with language model generation."
    )
    return path


@pytest.fixture
def mock_encoder():
    encoder = MagicMock()

    def _encode(texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        return np.random.randn(n, 384).astype(np.float32)

    encoder.encode.side_effect = _encode
    return encoder
