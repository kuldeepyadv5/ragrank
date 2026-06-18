"""Unit and integration tests for the Enterprise RAG system."""

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.evaluator.metrics import RAGEvaluator
from src.generator.llm_client import LLMClient
from src.pipeline import RAGPipeline
from src.reranker.model import NeuralReranker, RerankerService
from src.retriever.text_processor import TextProcessor
from src.retriever.vector_store import VectorStore
import torch


class TestNeuralReranker:
    def test_model_forward_shape(self):
        model = NeuralReranker("distilbert-base-uncased")
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        with torch.no_grad():
            scores = model(input_ids, attention_mask)

        assert scores.shape == (batch_size,)

    def test_reranker_service_scores_pairs(self):
        service = RerankerService(model_name="distilbert-base-uncased")
        docs = [
            "Cross-encoders process query and document together.",
            "Unrelated text about cooking recipes.",
        ]
        scores = service.score_pairs("What is a cross-encoder?", docs)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_rerank_orders_documents(self):
        service = RerankerService(model_name="distilbert-base-uncased")
        docs = [
            {"text": "PyTorch autograd computes gradients.", "chunk_id": "a"},
            {"text": "Cross-encoders rerank retrieved documents.", "chunk_id": "b"},
        ]
        ranked = service.rerank("cross-encoder reranking", docs, top_k=2)
        assert len(ranked) == 2
        assert "rerank_score" in ranked[0]


class TestRetriever:
    def test_text_chunking(self, sample_text_file):
        processor = TextProcessor(chunk_size=50, chunk_overlap=10)
        chunks = processor.process_file(sample_text_file)
        assert len(chunks) >= 1
        assert all(c.chunk_id for c in chunks)
        assert all(c.source == "sample.txt" for c in chunks)

    def test_vector_store_ingest_and_search(self, sample_text_file, mock_encoder, tmp_path):
        processor = TextProcessor(chunk_size=100, chunk_overlap=10)
        chunks = processor.process_file(sample_text_file)

        store = VectorStore(index_path=tmp_path / "index")
        store._encoder = mock_encoder

        added = store.add_chunks(chunks)
        assert added == len(chunks)
        assert store.num_chunks == len(chunks)

        results = store.search("cross-encoder reranker", top_k=2)
        assert len(results) <= 2
        assert "retrieval_score" in results[0]


class TestPipeline:
    def test_end_to_end_with_mocks(self, sample_text_file, mock_encoder, tmp_path):
        store = VectorStore(index_path=tmp_path / "idx")
        store._encoder = mock_encoder

        pipeline = RAGPipeline(
            vector_store=store,
            reranker=RerankerService(model_name="distilbert-base-uncased"),
            llm=LLMClient(provider="mock"),
        )
        count = pipeline.ingest_file(str(sample_text_file))
        assert count > 0

        result = pipeline.query("What is a cross-encoder?", run_eval=True)
        assert result.answer
        assert len(result.contexts) > 0
        assert "total_ms" in result.latency_ms
        assert result.eval_scores is not None


class TestFastAPI:
    @pytest.fixture
    def client(self, sample_text_file, mock_encoder, tmp_path):
        store = VectorStore(index_path=tmp_path / "api_idx")
        store._encoder = mock_encoder
        pipeline = RAGPipeline(
            vector_store=store,
            reranker=RerankerService(model_name="distilbert-base-uncased"),
            llm=LLMClient(provider="mock"),
        )
        chunks = pipeline.text_processor.process_file(sample_text_file)
        store.add_chunks(chunks)

        with patch("src.app.main._pipeline", pipeline):
            from src.app.main import app

            yield TestClient(app)

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_query(self, client):
        resp = client.post(
            "/query",
            json={"query": "cross-encoder", "use_reranker": True, "run_eval": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "latency_ms" in data

    def test_ingest(self, client):
        content = b"Enterprise RAG with PyTorch reranker and FAISS retrieval."
        resp = client.post(
            "/ingest",
            files=[("files", ("doc.txt", io.BytesIO(content), "text/plain"))],
        )
        assert resp.status_code == 200
        assert resp.json()["chunks_added"] >= 1


class TestLLMClient:
    def test_qwen_requires_api_key(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "")
        monkeypatch.setenv("QWEN_API_KEY", "")
        client = LLMClient(provider="qwen")
        with pytest.raises(ValueError, match="API key"):
            list(client.stream("hello"))

    def test_qwen_stream(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

        class FakeDelta:
            content = "Hello "

        class FakeChoice:
            delta = FakeDelta()

        class FakeChunk:
            choices = [FakeChoice()]

        class FakeStream:
            def __iter__(self):
                yield FakeChunk()

        class FakeCompletions:
            def create(self, **kwargs):
                return FakeStream()

        class FakeChat:
            completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, *args, **kwargs):
                self.chat = FakeChat()

        import openai

        monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
        client = LLMClient(provider="qwen", model="qwen-plus")
        tokens = list(client.stream("test prompt", system="be helpful"))
        assert tokens == ["Hello "]

    def test_ollama_chat_stream(self, monkeypatch):
        captured: dict = {}

        class FakeDelta:
            content = "Hi from Ollama "

        class FakeChoice:
            delta = FakeDelta()

        class FakeChunk:
            choices = [FakeChoice()]

        class FakeStream:
            def __iter__(self):
                yield FakeChunk()

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeStream()

        class FakeChat:
            completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, *args, **kwargs):
                captured["client_kwargs"] = kwargs
                self.chat = FakeChat()

        import openai

        monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
        client = LLMClient(provider="ollama", model="qwen3:8b")
        tokens = list(client.stream("Hello from Windows", system="be concise"))
        assert tokens == ["Hi from Ollama "]
        assert captured["client_kwargs"]["base_url"] == "http://localhost:11434/v1"
        assert captured["model"] == "qwen3:8b"


class TestEvaluator:
    def test_mock_evaluation(self):
        evaluator = RAGEvaluator(LLMClient(provider="mock"))
        result = evaluator.evaluate(
            query="What is FAISS?",
            answer="FAISS is a vector search library.",
            contexts=["FAISS enables fast similarity search."],
        )
        assert 0 <= result.context_precision <= 1
        assert 0 <= result.answer_faithfulness <= 1
        assert 0 <= result.answer_relevance <= 1

    def test_batch_evaluation(self):
        evaluator = RAGEvaluator(LLMClient(provider="mock"))
        report = evaluator.evaluate_batch(
            [
                {
                    "query": "test",
                    "answer": "answer",
                    "contexts": ["context"],
                }
            ]
        )
        assert report["count"] == 1
        assert "average_score" in report
