"""Additional pipeline integration smoke test."""

from unittest.mock import MagicMock

import numpy as np

from src.generator.llm_client import LLMClient
from src.pipeline import RAGPipeline
from src.reranker.model import RerankerService
from src.retriever.vector_store import VectorStore


def test_pipeline_compare_rerank_modes(tmp_path, sample_text_file, mock_encoder):
    store = VectorStore(index_path=tmp_path / "compare_idx")
    store._encoder = mock_encoder

    pipeline = RAGPipeline(
        vector_store=store,
        reranker=RerankerService(model_name="distilbert-base-uncased"),
        llm=LLMClient(provider="mock"),
    )
    pipeline.ingest_file(str(sample_text_file))

    with_rerank = pipeline.query("cross-encoder", use_reranker=True)
    without_rerank = pipeline.query("cross-encoder", use_reranker=False)

    assert with_rerank.answer
    assert without_rerank.answer
    assert with_rerank.use_reranker is True
    assert without_rerank.use_reranker is False
