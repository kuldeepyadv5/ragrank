"""End-to-end RAG pipeline orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from src.app.workflow_log import log_pipeline_step, reset_pipeline_steps
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


@dataclass
class RetrieveResult:
    query: str
    retrieved: list[dict]
    reranked: list[dict]
    rank_changes: list[dict]
    latency_ms: dict[str, float] = field(default_factory=dict)
    settings: dict = field(default_factory=dict)


def _chunk_summary(chunks: list[dict]) -> list[dict]:
    return [
        {
            "chunk_id": c.get("chunk_id"),
            "source": c.get("source"),
            "page": c.get("page"),
            "score": c.get("rerank_score", c.get("retrieval_score")),
            "text_preview": (c.get("text") or "")[:120],
        }
        for c in chunks
    ]


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
        log_pipeline_step("INGEST from file", path=file_path)
        chunks = self.text_processor.process_file(file_path)
        log_pipeline_step("Text chunked", num_chunks=len(chunks))
        added = self.vector_store.add_chunks(chunks)
        log_pipeline_step("FAISS index updated", chunks_added=added, total=self.vector_store.num_chunks)
        return added

    def ingest_bytes(self, data: bytes, filename: str) -> int:
        import io

        log_pipeline_step(
            "INGEST from upload",
            filename=filename,
            size_bytes=len(data),
            indexed_before=self.vector_store.num_chunks,
        )
        chunks = self.text_processor.process_upload(io.BytesIO(data), filename)
        log_pipeline_step(
            "Text extracted & chunked",
            num_chunks=len(chunks),
            sample_chunk_ids=[c.chunk_id for c in chunks[:3]],
        )
        added = self.vector_store.add_chunks(chunks)
        log_pipeline_step(
            "Embeddings stored in FAISS",
            chunks_added=added,
            total_chunks=self.vector_store.num_chunks,
        )
        return added

    def retrieve_only(
        self,
        question: str,
        retrieval_top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> RetrieveResult:
        """Run FAISS retrieval + cross-encoder rerank without LLM generation."""
        reset_pipeline_steps()
        cfg = get_config().retriever
        k_retrieve = retrieval_top_k or cfg.top_k
        k_rerank = rerank_top_k or cfg.rerank_top_k
        latency: dict[str, float] = {}

        log_pipeline_step(
            "Retrieval-only query started",
            question=question,
            retrieval_top_k=k_retrieve,
            rerank_top_k=k_rerank,
        )

        t0 = time.perf_counter()
        retrieved = self.vector_store.search(question, top_k=k_retrieve)
        latency["retrieval_ms"] = (time.perf_counter() - t0) * 1000
        log_pipeline_step(
            "FAISS retrieval complete",
            candidates=len(retrieved),
            latency_ms=round(latency["retrieval_ms"], 1),
            top_results=_chunk_summary(retrieved[:5]),
        )

        t1 = time.perf_counter()
        reranked = (
            self.reranker.rerank(question, retrieved, top_k=k_rerank)
            if retrieved
            else []
        )
        latency["rerank_ms"] = (time.perf_counter() - t1) * 1000
        log_pipeline_step(
            "PyTorch rerank complete",
            selected=len(reranked),
            latency_ms=round(latency["rerank_ms"], 1),
            ranked_results=_chunk_summary(reranked),
        )

        faiss_rank = {c["chunk_id"]: i + 1 for i, c in enumerate(retrieved)}
        rerank_rank = {c["chunk_id"]: i + 1 for i, c in enumerate(reranked)}
        all_ids = set(faiss_rank) | set(rerank_rank)

        rank_changes = []
        for chunk_id in all_ids:
            fr = faiss_rank.get(chunk_id)
            rr = rerank_rank.get(chunk_id)
            rank_changes.append(
                {
                    "chunk_id": chunk_id,
                    "faiss_rank": fr,
                    "rerank_rank": rr,
                    "rank_delta": (fr - rr) if fr and rr else None,
                    "in_top_rerank": chunk_id in rerank_rank,
                }
            )
        rank_changes.sort(
            key=lambda x: (x["rerank_rank"] is None, x["rerank_rank"] or 9999)
        )

        latency["total_ms"] = latency["retrieval_ms"] + latency["rerank_ms"]
        log_pipeline_step("Retrieval-only query finished", latency_ms=latency)

        return RetrieveResult(
            query=question,
            retrieved=retrieved,
            reranked=reranked,
            rank_changes=rank_changes,
            latency_ms=latency,
            settings={
                "retrieval_top_k": k_retrieve,
                "rerank_top_k": k_rerank,
                "embedding_model": get_config().retriever.embedding_model,
                "reranker_model": get_config().reranker.model_name,
            },
        )

    def query(
        self,
        question: str,
        use_reranker: bool = True,
        run_eval: bool = False,
    ) -> QueryResult:
        reset_pipeline_steps()
        cfg = get_config()
        latency: dict[str, float] = {}

        log_pipeline_step(
            "RAG query started",
            question=question,
            use_reranker=use_reranker,
            run_eval=run_eval,
            llm_provider=cfg.llm.provider,
            llm_model=cfg.llm.model,
            retrieval_top_k=cfg.retriever.top_k,
            rerank_top_k=cfg.retriever.rerank_top_k,
        )

        t0 = time.perf_counter()
        retrieved = self.vector_store.search(question, top_k=cfg.retriever.top_k)
        latency["retrieval_ms"] = (time.perf_counter() - t0) * 1000
        log_pipeline_step(
            "FAISS retrieval complete",
            candidates=len(retrieved),
            latency_ms=round(latency["retrieval_ms"], 1),
            top_results=_chunk_summary(retrieved[:5]),
        )

        t1 = time.perf_counter()
        if use_reranker and retrieved:
            contexts = self.reranker.rerank(question, retrieved, top_k=cfg.retriever.rerank_top_k)
            stage = "PyTorch cross-encoder rerank"
        else:
            contexts = retrieved[: cfg.retriever.rerank_top_k]
            stage = "Rerank skipped (vector order kept)"
        latency["rerank_ms"] = (time.perf_counter() - t1) * 1000
        log_pipeline_step(
            stage,
            selected_chunks=len(contexts),
            latency_ms=round(latency["rerank_ms"], 1),
            ranked_results=_chunk_summary(contexts),
        )

        context_texts = [c["text"] for c in contexts]
        context_block = "\n\n".join(
            f"[Source: {c.get('source', 'unknown')}, chunk={c.get('chunk_id', '?')}]\n{c['text']}"
            for c in contexts
        )
        prompt = f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"

        log_pipeline_step(
            "LLM prompt built",
            system_prompt=RAG_SYSTEM_PROMPT,
            prompt_chars=len(prompt),
            context_chunks=len(contexts),
        )

        t2 = time.perf_counter()
        answer = self.llm.generate(prompt, system=RAG_SYSTEM_PROMPT)
        latency["generation_ms"] = (time.perf_counter() - t2) * 1000
        log_pipeline_step(
            "LLM generation complete",
            provider=cfg.llm.provider,
            model=cfg.llm.model,
            latency_ms=round(latency["generation_ms"], 1),
            answer_preview=answer[:300],
        )

        eval_scores = None
        if run_eval:
            t3 = time.perf_counter()
            log_pipeline_step("LLM-as-a-judge evaluation started")
            eval_result = self.evaluator.evaluate(question, answer, context_texts)
            eval_scores = {
                "context_precision": eval_result.context_precision,
                "answer_faithfulness": eval_result.answer_faithfulness,
                "answer_relevance": eval_result.answer_relevance,
                "details": eval_result.details,
            }
            latency["eval_ms"] = (time.perf_counter() - t3) * 1000
            log_pipeline_step(
                "Evaluation complete",
                latency_ms=round(latency["eval_ms"], 1),
                scores=eval_scores,
            )

        latency["total_ms"] = sum(v for k, v in latency.items() if k != "total_ms")

        result = QueryResult(
            query=question,
            answer=answer,
            contexts=contexts,
            latency_ms=latency,
            use_reranker=use_reranker,
            eval_scores=eval_scores,
        )
        log_pipeline_step("RAG query finished", latency_ms=latency)
        self._log_query(result)
        return result

    def stream_query(
        self,
        question: str,
        use_reranker: bool = True,
    ) -> Iterator[str | dict]:
        """Yield context metadata first, then streamed answer tokens."""
        reset_pipeline_steps()
        cfg = get_config()
        log_pipeline_step("RAG stream query started", question=question, use_reranker=use_reranker)

        retrieved = self.vector_store.search(question, top_k=cfg.retriever.top_k)
        log_pipeline_step("FAISS retrieval complete", candidates=len(retrieved))

        if use_reranker and retrieved:
            contexts = self.reranker.rerank(question, retrieved, top_k=cfg.retriever.rerank_top_k)
        else:
            contexts = retrieved[: cfg.retriever.rerank_top_k]
        log_pipeline_step("Context ready for stream", chunks=len(contexts))

        yield {"type": "contexts", "data": contexts}

        context_block = "\n\n".join(
            f"[Source: {c.get('source', 'unknown')}]\n{c['text']}" for c in contexts
        )
        prompt = f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"
        log_pipeline_step("Streaming tokens from LLM", model=cfg.llm.model)

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
