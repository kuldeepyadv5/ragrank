"""End-to-end RAG pipeline orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from src.config import get_config
from src.evaluator.metrics import RAGEvaluator
from src.generator.llm_client import LLMClient
from src.reranker.model import RerankerService
from src.retriever.text_processor import TextProcessor
from src.retriever.vector_store import VectorStore


RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the provided context. "
    "If the context is insufficient, say you don't know. Cite sources when possible."
)


@dataclass
class QueryResult:
    query: str
    answer: str
    contexts: list[dict]
    latency_ms: dict[str, float] = field(default_factory=dict)
    use_reranker: bool = True
    eval_scores: dict | None = None


class RAGPipeline:
    """Orchestrates retrieval, reranking, generation, and evaluation."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        reranker: RerankerService | None = None,
        llm: LLMClient | None = None,
        evaluator: RAGEvaluator | None = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.reranker = reranker or RerankerService()
        self.llm = llm or LLMClient()
        self.evaluator = evaluator or RAGEvaluator(self.llm)
        self.text_processor = TextProcessor()
        self._query_log: list[dict] = []

    def ingest_file(self, file_path: str) -> int:
        chunks = self.text_processor.process_file(file_path)
        return self.vector_store.add_chunks(chunks)

    def ingest_bytes(self, data: bytes, filename: str) -> int:
        import io

        chunks = self.text_processor.process_upload(io.BytesIO(data), filename)
        return self.vector_store.add_chunks(chunks)

    def query(
        self,
        question: str,
        use_reranker: bool = True,
        run_eval: bool = False,
    ) -> QueryResult:
        cfg = get_config().retriever
        latency: dict[str, float] = {}

        t0 = time.perf_counter()
        retrieved = self.vector_store.search(question, top_k=cfg.top_k)
        latency["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        if use_reranker and retrieved:
            contexts = self.reranker.rerank(question, retrieved, top_k=cfg.rerank_top_k)
        else:
            contexts = retrieved[: cfg.rerank_top_k]
        latency["rerank_ms"] = (time.perf_counter() - t1) * 1000

        context_texts = [c["text"] for c in contexts]
        print("\ndebugging context_texts: ", context_texts)
        context_block = "\n\n".join(
            f"[Source: {c.get('source', 'unknown')}, chunk={c.get('chunk_id', '?')}]\n{c['text']}"
            for c in contexts
        )
        prompt = f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"

        t2 = time.perf_counter()
        answer = self.llm.generate(prompt, system=RAG_SYSTEM_PROMPT)
        latency["generation_ms"] = (time.perf_counter() - t2) * 1000
        latency["total_ms"] = sum(latency.values())

        eval_scores = None
        if run_eval:
            t3 = time.perf_counter()
            eval_result = self.evaluator.evaluate(question, answer, context_texts)
            eval_scores = {
                "context_precision": eval_result.context_precision,
                "answer_faithfulness": eval_result.answer_faithfulness,
                "answer_relevance": eval_result.answer_relevance,
                "details": eval_result.details,
            }
            latency["eval_ms"] = (time.perf_counter() - t3) * 1000
            latency["total_ms"] = sum(
                v for k, v in latency.items() if k != "total_ms"
            ) + latency.get("eval_ms", 0)

        result = QueryResult(
            query=question,
            answer=answer,
            contexts=contexts,
            latency_ms=latency,
            use_reranker=use_reranker,
            eval_scores=eval_scores,
        )
        self._log_query(result)
        return result

    def stream_query(
        self,
        question: str,
        use_reranker: bool = True,
    ) -> Iterator[str | dict]:
        """Yield context metadata first, then streamed answer tokens."""
        cfg = get_config().retriever
        retrieved = self.vector_store.search(question, top_k=cfg.top_k)
        if use_reranker and retrieved:
            contexts = self.reranker.rerank(question, retrieved, top_k=cfg.rerank_top_k)
        else:
            contexts = retrieved[: cfg.rerank_top_k]

        yield {"type": "contexts", "data": contexts}

        context_block = "\n\n".join(
            f"[Source: {c.get('source', 'unknown')}]\n{c['text']}" for c in contexts
        )
        prompt = f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"

        for token in self.llm.stream(prompt, system=RAG_SYSTEM_PROMPT):
            yield token

    def get_eval_report(self) -> dict:
        runs = [
            {
                "query": r["query"],
                "answer": r["answer"],
                "contexts": [c["text"] for c in r["contexts"]],
            }
            for r in self._query_log
        ]
        report = self.evaluator.evaluate_batch(runs)
        report["latency_stats"] = self._latency_stats()
        return report

    def _log_query(self, result: QueryResult) -> None:
        self._query_log.append(
            {
                "query": result.query,
                "answer": result.answer,
                "contexts": result.contexts,
                "latency_ms": result.latency_ms,
                "use_reranker": result.use_reranker,
                "eval_scores": result.eval_scores,
            }
        )

    def _latency_stats(self) -> dict:
        if not self._query_log:
            return {}
        keys = ["retrieval_ms", "rerank_ms", "generation_ms", "total_ms"]
        stats = {}
        for key in keys:
            values = [
                r["latency_ms"].get(key, 0)
                for r in self._query_log
                if key in r["latency_ms"]
            ]
            if values:
                stats[key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
        return stats
