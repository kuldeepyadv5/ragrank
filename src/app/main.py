"""FastAPI backend with latency telemetry."""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.app.workflow_log import (
    log_endpoint_request,
    log_endpoint_response,
    setup_backend_logging,
)
from src.config import get_config
from src.pipeline import RAGPipeline

_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_backend_logging()
    global _pipeline
    _pipeline = RAGPipeline()
    yield
    _pipeline = None


app = FastAPI(
    title="Enterprise RAG API",
    description="RAG system with PyTorch cross-encoder reranker",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    use_reranker: bool = True
    run_eval: bool = False
    stream: bool = False


class QueryResponse(BaseModel):
    query: str
    answer: str
    contexts: list[dict[str, Any]]
    latency_ms: dict[str, float]
    use_reranker: bool
    eval_scores: dict[str, Any] | None = None


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    num_chunks: int
    llm_provider: str
    reranker_quantization: str = "none"
    embedding_quantization: str = "none"


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    retrieval_top_k: int | None = Field(default=None, ge=1, le=100)
    rerank_top_k: int | None = Field(default=None, ge=1, le=50)


class RetrieveResponse(BaseModel):
    query: str
    retrieved: list[dict[str, Any]]
    faiss_only: list[dict[str, Any]]
    reranked: list[dict[str, Any]]
    reranked_trained: list[dict[str, Any]]
    reranked_untrained: list[dict[str, Any]]
    rank_changes: list[dict[str, Any]]
    rank_changes_trained: list[dict[str, Any]]
    latency_ms: dict[str, float]
    settings: dict[str, Any]


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return _pipeline


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    t0 = time.perf_counter()
    log_endpoint_request("GET", "/health")
    cfg = get_config()
    pipeline = get_pipeline()
    resp = HealthResponse(
        status="ok",
        num_chunks=pipeline.vector_store.num_chunks,
        llm_provider=cfg.llm.provider,
        reranker_quantization=pipeline.reranker.quantization_mode,
        embedding_quantization=(
            pipeline.vector_store.quantization_mode
            if pipeline.vector_store.quantization_mode != "none"
            else ("int8" if pipeline.vector_store.quantize else "none")
        ),
    )
    log_endpoint_response("GET", "/health", resp.model_dump(), (time.perf_counter() - t0) * 1000)
    return resp


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
    t0 = time.perf_counter()
    file_meta = [{"filename": f.filename, "content_type": f.content_type} for f in files]
    log_endpoint_request("POST", "/ingest", {"files": file_meta})

    pipeline = get_pipeline()
    total_added = 0
    last_filename = ""

    for upload in files:
        data = await upload.read()
        if not data:
            continue
        filename = upload.filename or "upload.txt"
        added = pipeline.ingest_bytes(data, filename)
        total_added += added
        last_filename = filename

    if total_added == 0:
        raise HTTPException(status_code=400, detail="No valid content ingested")

    resp = IngestResponse(
        filename=last_filename,
        chunks_added=total_added,
        total_chunks=pipeline.vector_store.num_chunks,
    )
    log_endpoint_response("POST", "/ingest", resp.model_dump(), (time.perf_counter() - t0) * 1000)
    return resp


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """Retrieval + reranking only — no LLM call."""
    t0 = time.perf_counter()
    log_endpoint_request("POST", "/retrieve", req.model_dump())

    pipeline = get_pipeline()
    if pipeline.vector_store.num_chunks == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Use /ingest first.")

    result = pipeline.retrieve_only(
        req.query,
        retrieval_top_k=req.retrieval_top_k,
        rerank_top_k=req.rerank_top_k,
    )
    resp = RetrieveResponse(
        query=result.query,
        retrieved=result.retrieved,
        faiss_only=result.faiss_only,
        reranked=result.reranked,
        reranked_trained=result.reranked_trained,
        reranked_untrained=result.reranked_untrained,
        rank_changes=result.rank_changes,
        rank_changes_trained=result.rank_changes_trained,
        latency_ms=result.latency_ms,
        settings=result.settings,
    )
    log_endpoint_response("POST", "/retrieve", resp.model_dump(), (time.perf_counter() - t0) * 1000)
    return resp


@app.post("/query")
def query(req: QueryRequest):
    t0 = time.perf_counter()
    log_endpoint_request("POST", "/query", req.model_dump())

    pipeline = get_pipeline()
    if pipeline.vector_store.num_chunks == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Use /ingest first.")

    if req.stream:
        log_endpoint_response(
            "POST",
            "/query",
            {"mode": "stream", "query": req.query},
            (time.perf_counter() - t0) * 1000,
        )
        return _stream_query_response(pipeline, req)

    result = pipeline.query(
        req.query,
        use_reranker=req.use_reranker,
        run_eval=req.run_eval,
    )
    resp = QueryResponse(
        query=result.query,
        answer=result.answer,
        contexts=result.contexts,
        latency_ms=result.latency_ms,
        use_reranker=result.use_reranker,
        eval_scores=result.eval_scores,
    )
    log_endpoint_response("POST", "/query", resp.model_dump(), (time.perf_counter() - t0) * 1000)
    return resp


def _stream_query_response(pipeline: RAGPipeline, req: QueryRequest):
    def event_generator():
        t0 = time.perf_counter()
        for item in pipeline.stream_query(req.query, use_reranker=req.use_reranker):
            if isinstance(item, dict):
                yield f"data: {json.dumps(item)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'token', 'data': item})}\n\n"
        elapsed = (time.perf_counter() - t0) * 1000
        yield f"data: {json.dumps({'type': 'done', 'latency_ms': elapsed})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    t0 = time.perf_counter()
    log_endpoint_request("POST", "/query/stream", req.model_dump())

    pipeline = get_pipeline()
    if pipeline.vector_store.num_chunks == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Use /ingest first.")

    log_endpoint_response(
        "POST",
        "/query/stream",
        {"mode": "sse_stream", "query": req.query},
        (time.perf_counter() - t0) * 1000,
    )

    def event_generator():
        t0 = time.perf_counter()
        for item in pipeline.stream_query(req.query, use_reranker=req.use_reranker):
            if isinstance(item, dict):
                yield f"data: {json.dumps(item)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'token', 'data': item})}\n\n"
        elapsed = (time.perf_counter() - t0) * 1000
        yield f"data: {json.dumps({'type': 'done', 'latency_ms': elapsed})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/eval-report")
def eval_report() -> dict:
    t0 = time.perf_counter()
    log_endpoint_request("GET", "/eval-report")
    pipeline = get_pipeline()
    report = pipeline.get_eval_report()
    log_endpoint_response("GET", "/eval-report", report, (time.perf_counter() - t0) * 1000)
    return report


@app.get("/stats")
def stats() -> dict:
    t0 = time.perf_counter()
    log_endpoint_request("GET", "/stats")
    pipeline = get_pipeline()
    resp = {
        "num_chunks": pipeline.vector_store.num_chunks,
        "query_count": len(pipeline._query_log),
        "latency_stats": pipeline._latency_stats(),
    }
    log_endpoint_response("GET", "/stats", resp, (time.perf_counter() - t0) * 1000)
    return resp
