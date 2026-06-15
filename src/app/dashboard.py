"""Streamlit dashboard for RAG visualization and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import get_config

st.set_page_config(
    page_title="Enterprise RAG Dashboard",
    page_icon="🔍",
    layout="wide",
)

cfg = get_config()
API_BASE = cfg.app.api_base_url


def api_get(path: str) -> dict:
    with httpx.Client(timeout=120.0) as client:
        resp = client.get(f"{API_BASE}{path}")
        resp.raise_for_status()
        return resp.json()


def api_post(path: str, payload: dict) -> dict:
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{API_BASE}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()


def api_upload(files: list) -> dict:
    with httpx.Client(timeout=300.0) as client:
        multipart = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        resp = client.post(f"{API_BASE}/ingest", files=multipart)
        resp.raise_for_status()
        return resp.json()


def render_header():
    st.title("Enterprise RAG System")
    st.caption("PyTorch Cross-Encoder Reranker · FAISS Retrieval · LLM-as-a-Judge Eval")

    try:
        health = api_get("/health")
        st.success(
            f"API connected | {health['num_chunks']} chunks indexed | "
            f"LLM: {health['llm_provider']}"
        )
    except Exception as exc:
        st.error(f"Cannot reach API at {API_BASE}: {exc}")
        st.info("Start the API with: `python run.py api`")


def render_document_manager():
    st.header("Document Manager")
    uploaded = st.file_uploader(
        "Drag & drop PDF or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Ingest Documents", type="primary"):
        with st.spinner("Processing and indexing..."):
            try:
                result = api_upload(uploaded)
                st.success(
                    f"Ingested {result['chunks_added']} chunks "
                    f"(total: {result['total_chunks']})"
                )
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")


def render_rag_lab():
    st.header("RAG Comparison Lab")
    query = st.text_input("Enter your question", placeholder="What is a cross-encoder reranker?")

    col1, col2 = st.columns(2)
    if query and st.button("Run Comparison", type="primary"):
        with st.spinner("Running RAG pipeline..."):
            try:
                with_rerank = api_post(
                    "/query",
                    {"query": query, "use_reranker": True, "run_eval": True},
                )
                without_rerank = api_post(
                    "/query",
                    {"query": query, "use_reranker": False, "run_eval": True},
                )

                with col1:
                    st.subheader("With PyTorch Reranking")
                    st.markdown(with_rerank["answer"])
                    st.metric("Total Latency (ms)", f"{with_rerank['latency_ms'].get('total_ms', 0):.1f}")
                    if with_rerank.get("eval_scores"):
                        st.json(with_rerank["eval_scores"])
                    _render_context_table(with_rerank["contexts"])

                with col2:
                    st.subheader("Without Reranking")
                    st.markdown(without_rerank["answer"])
                    st.metric("Total Latency (ms)", f"{without_rerank['latency_ms'].get('total_ms', 0):.1f}")
                    if without_rerank.get("eval_scores"):
                        st.json(without_rerank["eval_scores"])
                    _render_context_table(without_rerank["contexts"])
            except Exception as exc:
                st.error(f"Query failed: {exc}")


def _render_context_table(contexts: list[dict]):
    if not contexts:
        st.info("No contexts retrieved")
        return
    rows = []
    for i, ctx in enumerate(contexts):
        rows.append(
            {
                "Rank": i + 1,
                "Score": ctx.get("rerank_score", ctx.get("retrieval_score", 0)),
                "Source": ctx.get("source", "?"),
                "Preview": ctx.get("text", "")[:120] + "...",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_telemetry():
    st.header("Live Latency & Evaluation Radar")

    try:
        stats = api_get("/stats")
        report = api_get("/eval-report")
    except Exception:
        st.warning("No telemetry data yet. Run some queries first.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Indexed Chunks", stats.get("num_chunks", 0))
    col2.metric("Queries Logged", stats.get("query_count", 0))
    col3.metric("Avg Context Precision", f"{report.get('context_precision', 0):.2f}")
    col4.metric("Avg Faithfulness", f"{report.get('answer_faithfulness', 0):.2f}")

    latency = stats.get("latency_stats", {})
    if latency:
        df = pd.DataFrame(
            [
                {"Stage": k.replace("_ms", ""), "Mean (ms)": v["mean"]}
                for k, v in latency.items()
            ]
        )
        fig_bar = px.bar(df, x="Stage", y="Mean (ms)", title="Mean Latency by Stage")
        st.plotly_chart(fig_bar, use_container_width=True)

    metrics = ["context_precision", "answer_faithfulness", "answer_relevance"]
    values = [report.get(m, 0) for m in metrics]
    if any(values):
        fig_radar = go.Figure(
            data=go.Scatterpolar(
                r=values + [values[0]],
                theta=[m.replace("_", " ").title() for m in metrics] + [metrics[0].replace("_", " ").title()],
                fill="toself",
                name="RAG Quality",
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Evaluation Radar",
        )
        st.plotly_chart(fig_radar, use_container_width=True)


def main():
    render_header()
    tab1, tab2, tab3 = st.tabs(["Documents", "RAG Lab", "Telemetry"])
    with tab1:
        render_document_manager()
    with tab2:
        render_rag_lab()
    with tab3:
        render_telemetry()


if __name__ == "__main__":
    main()
