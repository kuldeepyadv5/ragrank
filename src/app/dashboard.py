"""Streamlit dashboard for RAG visualization and evaluation."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app.api_client import DashboardApiClient, setup_api_logging
from src.config import get_config

st.set_page_config(
    page_title="Enterprise RAG Dashboard",
    page_icon="🔍",
    layout="wide",
)

setup_api_logging()
cfg = get_config()
API_BASE = cfg.app.api_base_url
api = DashboardApiClient(base_url=API_BASE)


def api_get(path: str) -> dict:
    return api.get(path)


def api_post(path: str, payload: dict) -> dict:
    return api.post(path, payload)


def api_upload(files: list) -> dict:
    return api.upload(files)


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
    run_eval = st.checkbox(
        "Run LLM evaluation (slower — adds extra Ollama calls)",
        value=False,
    )

    col1, col2 = st.columns(2)
    if query and st.button("Run Comparison", type="primary"):
        with st.spinner("Running RAG pipeline (local qwen3:8b can take several minutes)..."):
            try:
                with_rerank = api_post(
                    "/query",
                    {"query": query, "use_reranker": True, "run_eval": run_eval},
                )
                without_rerank = api_post(
                    "/query",
                    {"query": query, "use_reranker": False, "run_eval": run_eval},
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


def _render_context_table(contexts: list[dict], score_key: str = "rerank_score"):
    if not contexts:
        st.info("No contexts retrieved")
        return
    rows = []
    for i, ctx in enumerate(contexts):
        score = ctx.get(score_key, ctx.get("retrieval_score", 0))
        rows.append(
            {
                "Rank": i + 1,
                "Score": round(float(score), 4) if score is not None else None,
                "Chunk ID": ctx.get("chunk_id", "?"),
                "Source": ctx.get("source", "?"),
                "Page": ctx.get("page", "-"),
                "Preview": (ctx.get("text") or "")[:120] + "...",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_chunk_details(chunks: list[dict], label: str):
    st.subheader(label)
    _render_context_table(
        chunks,
        score_key="rerank_score" if any("rerank_score" in c for c in chunks) else "retrieval_score",
    )
    for i, ctx in enumerate(chunks):
        with st.expander(f"#{i + 1} {ctx.get('source', '?')} — {ctx.get('chunk_id', '')}"):
            st.markdown(ctx.get("text", ""))


def render_retrieval_check():
    st.header("RAG Retrieval Check")
    st.caption(
        "Compare FAISS retrieval vs trained reranker vs untrained (base) reranker — no LLM call."
    )

    cfg = get_config().retriever
    query = st.text_input(
        "Query",
        key="retrieval_query",
        placeholder="What is a cross-encoder reranker?",
    )

    col1, col2 = st.columns(2)
    with col1:
        retrieval_top_k = st.slider(
            "FAISS retrieval top-K",
            min_value=1,
            max_value=50,
            value=cfg.top_k,
        )
    with col2:
        rerank_top_k = st.slider(
            "Reranker top-K",
            min_value=1,
            max_value=20,
            value=cfg.rerank_top_k,
        )

    if query and st.button("Run Retrieval", type="primary", key="run_retrieval"):
        with st.spinner("Running FAISS + trained & untrained rerankers..."):
            try:
                result = api_post(
                    "/retrieve",
                    {
                        "query": query,
                        "retrieval_top_k": retrieval_top_k,
                        "rerank_top_k": rerank_top_k,
                    },
                )

                settings = result.get("settings", {})
                if not settings.get("trained_checkpoint_loaded"):
                    st.warning(
                        "No trained checkpoint found — run `python run.py train` first. "
                        "Trained and untrained results may look identical until you train."
                    )
                else:
                    st.success(f"Using trained checkpoint: {settings.get('trained_checkpoint')}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("FAISS (ms)", f"{result['latency_ms'].get('retrieval_ms', 0):.1f}")
                m2.metric("Trained rerank (ms)", f"{result['latency_ms'].get('rerank_trained_ms', 0):.1f}")
                m3.metric("Untrained rerank (ms)", f"{result['latency_ms'].get('rerank_untrained_ms', 0):.1f}")
                m4.metric("Total (ms)", f"{result['latency_ms'].get('total_ms', 0):.1f}")

                st.json(settings)

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    _render_chunk_details(
                        result["faiss_only"],
                        f"1. FAISS only (top {len(result['faiss_only'])})",
                    )
                with col_b:
                    _render_chunk_details(
                        result["reranked_trained"],
                        f"2. Trained reranker (top {len(result['reranked_trained'])})",
                    )
                with col_c:
                    _render_chunk_details(
                        result["reranked_untrained"],
                        f"3. Untrained / base reranker (top {len(result['reranked_untrained'])})",
                    )

                with st.expander(f"Full FAISS pool ({len(result['retrieved'])} candidates)"):
                    _render_chunk_details(
                        result["retrieved"],
                        "All retrieval candidates before reranking",
                    )

                st.subheader("Rank changes (FAISS → Trained reranker)")
                st.caption("Positive rank_delta = moved up after trained reranking")
                changes = result.get("rank_changes_trained", result.get("rank_changes", []))
                if changes:
                    df = pd.DataFrame(changes)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    promoted = [c for c in changes if c.get("rank_delta") and c["rank_delta"] > 0]
                    demoted = [c for c in changes if c.get("rank_delta") and c["rank_delta"] < 0]
                    c1, c2 = st.columns(2)
                    c1.metric("Chunks promoted (trained)", len(promoted))
                    c2.metric("Chunks demoted (trained)", len(demoted))

                st.subheader("Score comparison (top reranked chunks)")
                trained = result.get("reranked_trained", [])
                untrained = result.get("reranked_untrained", [])
                if trained or untrained:
                    chart_rows = []
                    for c in trained:
                        chart_rows.append(
                            {
                                "chunk": c.get("chunk_id", "")[:8],
                                "score": c.get("rerank_score", 0),
                                "model": "Trained",
                            }
                        )
                    for c in untrained:
                        chart_rows.append(
                            {
                                "chunk": c.get("chunk_id", "")[:8],
                                "score": c.get("rerank_score", 0),
                                "model": "Untrained",
                            }
                        )
                    fig = px.bar(
                        pd.DataFrame(chart_rows),
                        x="chunk",
                        y="score",
                        color="model",
                        barmode="group",
                        title="Trained vs untrained rerank scores",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Retrieval failed: {exc}")


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
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Documents", "RAG Lab", "Retrieval Check", "Telemetry"]
    )
    with tab1:
        render_document_manager()
    with tab2:
        render_rag_lab()
    with tab3:
        render_retrieval_check()
    with tab4:
        render_telemetry()


if __name__ == "__main__":
    main()
