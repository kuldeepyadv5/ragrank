# Enterprise RAG System with PyTorch Neural Reranker

A production-style **Retrieval-Augmented Generation (RAG)** system that combines:

- **FAISS** vector retrieval (bi-encoder embeddings)
- **Custom PyTorch cross-encoder reranker** (`distilbert-base-uncased` or `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **LLM generation** (Qwen via DashScope, or Ollama/mock for local dev)
- **LLM-as-a-judge evaluation** (faithfulness, relevance, context precision)
- **FastAPI** backend + **Streamlit** dashboard + **MLflow** experiment tracking

See **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** for the full design spec.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  FAISS Retriever │  ← sentence-transformers embeddings (bi-encoder)
└────────┬────────┘
         │ top-K chunks (default: 20)
         ▼
┌─────────────────┐
│ PyTorch Reranker │  ← cross-encoder scores query+doc jointly
└────────┬────────┘
         │ top-K reranked (default: 5)
         ▼
┌─────────────────┐
│   LLM Generator  │  ← Qwen (DashScope) / Ollama / mock
└────────┬────────┘
         ▼
    Answer + Latency + Eval Scores
```

---

## Project Structure

```
PyTorch/
├── src/
│   ├── config.py              # Environment & settings
│   ├── pipeline.py            # End-to-end RAG orchestration
│   ├── reranker/              # PyTorch cross-encoder (model, train, dataset)
│   ├── retriever/             # PDF/text chunking + FAISS vector store
│   ├── generator/             # LLM client (streaming)
│   ├── evaluator/             # LLM-as-a-judge metrics
│   └── app/                   # FastAPI + Streamlit dashboard
├── tests/                     # pytest unit & integration tests
├── data/                      # FAISS index, model checkpoints, MLflow logs
├── run.py                     # CLI entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── docs/
    └── PYTORCH_AND_INTERVIEW.md   # PyTorch concepts + interview Q&A
```

---

## Prerequisites

- Python 3.11+ (3.12 tested)
- Optional: NVIDIA GPU + CUDA for faster reranker training
- Optional: `DASHSCOPE_API_KEY` from [Alibaba DashScope](https://dashscope.aliyun.com/) for Qwen

### Qwen setup

1. Create a DashScope account and generate an API key.
2. In `.env`:
   ```env
   LLM_PROVIDER=qwen
   DASHSCOPE_API_KEY=sk-your-key-here
   LLM_MODEL=qwen-plus
   ```
3. Restart the API: `python run.py api`

**Popular Qwen models:** `qwen-turbo` (fast), `qwen-plus` (balanced), `qwen-max` (best quality), `qwen2.5-7b-instruct`, `qwen2.5-72b-instruct`

**Run Qwen locally (free, no API key):**
```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
```
Then run `ollama pull qwen2.5:7b` before starting the API.

---

## Setup

### 1. Create and activate virtual environment

**Windows (PowerShell):**
```powershell
cd c:\Users\kulde\OneDrive\Desktop\LLM\PyTorch
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
cd PyTorch
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

> First install may take several minutes (PyTorch, transformers, FAISS, Streamlit, MLflow).

### 3. Configure environment

```powershell
copy .env.example .env
```

Edit `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `qwen`, `ollama`, or `mock` | `qwen` |
| `DASHSCOPE_API_KEY` | Qwen / DashScope API key | — |
| `LLM_MODEL` | Qwen model name | `qwen-plus` |
| `QWEN_BASE_URL` | DashScope OpenAI-compatible endpoint | DashScope default |
| `RERANKER_MODEL` | HuggingFace cross-encoder base | `distilbert-base-uncased` |
| `RERANKER_LOSS` | `bce` or `margin_mse` | `bce` |
| `EMBEDDING_MODEL` | SentenceTransformer for retrieval | `all-MiniLM-L6-v2` |
| `RETRIEVAL_TOP_K` | Chunks retrieved from FAISS | `20` |
| `RERANK_TOP_K` | Chunks kept after reranking | `5` |

**Mock mode** works without any API keys — good for local development and tests.

---

## How to Run the Project

### Option A: CLI (recommended for development)

Open **two terminals** from the project root (with venv activated):

**Terminal 1 — FastAPI backend:**
```powershell
python run.py api
```
API docs: http://localhost:8000/docs

**Terminal 2 — Streamlit dashboard:**
```powershell
python run.py dashboard
```
Dashboard: http://localhost:8501

### Option B: Train the PyTorch reranker

```powershell
python run.py train
```

Checkpoint saved to: `data/models/reranker/best.pt`

Training logs go to MLflow at `./data/mlruns` (or TensorBoard if configured).

### Option C: Docker (API + Dashboard + MLflow)

```powershell
docker-compose up --build
```

| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000 |
| Streamlit | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

---

## Usage Examples

### Ingest documents (API)

```powershell
curl -X POST http://localhost:8000/ingest -F "files=@your_document.pdf"
```

Or upload PDFs via the **Documents** tab in the Streamlit dashboard.

### Query with reranking

```powershell
curl -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d '{"query": "What is a cross-encoder?", "use_reranker": true, "run_eval": true}'
```

### Query with streaming (SSE)

```powershell
curl -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d '{"query": "What is FAISS?", "use_reranker": true, "stream": true}'
```

Or use the dedicated endpoint: `POST /query/stream`

### Compare with vs without reranking

Use the **RAG Comparison Lab** tab in the dashboard — it runs the same query both ways side-by-side.

### View evaluation report

```powershell
curl http://localhost:8000/eval-report
```

---

## Run Tests

```powershell
pytest tests/ -v
```

All 13 tests cover reranker forward pass, FAISS ingestion, pipeline end-to-end, FastAPI endpoints, and evaluator metrics.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health + chunk count |
| `POST` | `/ingest` | Upload PDF/TXT files |
| `POST` | `/query` | Full RAG query (retrieve → rerank → generate) |
| `POST` | `/query/stream` | Streaming SSE response |
| `GET` | `/eval-report` | Aggregated LLM-judge metrics |
| `GET` | `/stats` | Latency statistics |

---

## Where PyTorch Is Used

PyTorch powers the **neural reranker** module only. Everything else uses standard Python libraries.

| File | PyTorch usage |
|------|---------------|
| `src/reranker/model.py` | `nn.Module`, `nn.Linear`, tensors, `@torch.no_grad()` |
| `src/reranker/dataset.py` | `Dataset`, `DataLoader`, `torch.tensor` |
| `src/reranker/train.py` | Training loop, `BCEWithLogitsLoss`, AdamW, AMP, `torch.save` |

For a deep dive with code walkthroughs and interview questions, see:

**[docs/PYTORCH_AND_INTERVIEW.md](docs/PYTORCH_AND_INTERVIEW.md)**

---

## Typical Workflow

1. **Setup** — install deps, copy `.env`, add `DASHSCOPE_API_KEY` for Qwen
2. **Start services** — `python run.py api` + `python run.py dashboard`
3. **Ingest** — upload a technical PDF via dashboard or `/ingest`
4. **Query** — ask questions in RAG Lab; compare reranked vs non-reranked results
5. **Train** (optional) — `python run.py train` to fine-tune the cross-encoder
6. **Evaluate** — check Telemetry tab or `/eval-report` for faithfulness/relevance scores

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Dashboard can't reach API | Ensure API is running; check `API_BASE_URL` in `.env` |
| `No documents indexed` | Upload files via `/ingest` or dashboard first |
| Slow first query | HuggingFace models download on first run (~250 MB) |
| pip timeout on Windows | `pip install --default-timeout=100 -r requirements.txt` |
| CUDA not used | Training falls back to CPU automatically; install CUDA PyTorch for GPU |

---

## License

Educational / portfolio project. HuggingFace model weights follow their respective licenses.
