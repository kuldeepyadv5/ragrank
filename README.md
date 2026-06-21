# RAGRank

**Trainable PyTorch reranker · INT8-quantized inference · FAISS retrieval · LLM generation**

RAGRank is an end-to-end **Retrieval-Augmented Generation (RAG)** stack where you control the ranking layer: fine-tune a cross-encoder on your domain, quantize it for fast CPU inference, and compare FAISS vs trained vs base reranking — all without touching the LLM when debugging retrieval.

> **Why this name?** **RAG** = retrieval-augmented generation. **Rank** = the cross-encoder reranking step that reorders FAISS candidates. Short, accurate, and matches the repo: [github.com/kuldeepyadv5/ragrank](https://github.com/kuldeepyadv5/ragrank).

Alternative names if you prefer something more descriptive: **NeuralRAG-Rank**, **PyTorch-RAG-Reranker**, **RankForge-RAG**.

---

## What makes this different

Most RAG tutorials stop at “embed + search + prompt.” RAGRank adds a **learnable reranking stage** you can train yourself:

| Stage | Model | Role |
|-------|--------|------|
| **Retrieve** | SentenceTransformer (bi-encoder) + FAISS | Fast approximate search — query and document encoded separately |
| **Rerank** | PyTorch cross-encoder (DistilBERT) | Slow but accurate — query + document encoded **together** |
| **Generate** | Ollama / Qwen / mock LLM | Answer from top reranked chunks |

Bi-encoders are fast but can miss nuance. Cross-encoders score each `(query, document)` pair jointly and usually rank better — at the cost of compute. RAGRank trains that reranker on **your** `(query, positive, negative)` examples, then optionally **quantizes** it so inference is smaller and faster on CPU.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────┐
│  FAISS + Bi-Encoder      │  sentence-transformers (optional INT8 on CPU)
│  RETRIEVAL_TOP_K = 20    │
└────────────┬─────────────┘
             │ candidate chunks
             ▼
┌──────────────────────────┐
│  PyTorch Cross-Encoder   │  custom trained best.pt (optional INT8 / FP16)
│  RERANK_TOP_K = 5        │
└────────────┬─────────────┘
             │ top contexts
             ▼
┌──────────────────────────┐
│  LLM (Ollama / Qwen)     │  answer generation
└────────────┬─────────────┘
             ▼
        Answer + latency + eval scores
```

**Debug path (no LLM):** the **Retrieval Check** tab and `POST /retrieve` run only retrieve + rerank and show three rankings side by side:

1. **FAISS only** — vector similarity order  
2. **Trained reranker** — uses `data/models/reranker/best.pt` (or `best_quantized.pt`)  
3. **Untrained / base reranker** — same architecture, no fine-tuned weights  

---

## How the custom reranker works

### Bi-encoder vs cross-encoder

```
Bi-encoder (retrieval):     embed(query) · embed(doc)     → fast, millions of docs
Cross-encoder (reranker):   score(query, doc) jointly      → slow, top-K only
```

FAISS returns ~20 candidates quickly. The cross-encoder re-scores each `(query, chunk)` pair and keeps the best 5 for the LLM.

### Model structure

- Base: `distilbert-base-uncased` (default) or `cross-encoder/ms-marco-MiniLM-L-6-v2`
- A linear head outputs one relevance logit per pair
- Training uses query / positive / negative triples from `src/reranker/dataset.py`

### Default training data (example)

Out of the box, training uses synthetic **automotive Q&A** pairs in `default_training_examples()` — e.g.:

| Query | Positive doc | Negative doc |
|-------|--------------|--------------|
| Best SUV under 15 lakh in India for a family of 5 | Mahindra XUV 3XO offers good safety… | BMW X5 is a luxury SUV priced above 1 crore |
| Best mileage petrol car under 10 lakh | Maruti Suzuki Baleno delivers ~22 kmpl… | Toyota Fortuner focuses on power… |

**Replace these with your domain** (HR policies, product docs, support tickets) before serious training. Each entry is a `QueryExample(query, positive, negative)`.

### Loss functions

| `RERANKER_LOSS` | Behavior |
|-----------------|----------|
| `bce` (default) | Binary cross-entropy — positive pairs → 1, negative → 0 |
| `margin_mse` | Push positive score above negative by margin `RERANKER_MARGIN` |

### Checkpoints after training

| File | Description |
|------|-------------|
| `data/models/reranker/best.pt` | Best FP32 weights (by validation MRR@10) |
| `data/models/reranker/best_quantized.pt` | INT8 export (~64% smaller) — auto-created when `RERANKER_QUANTIZE=true` |

Metrics logged to MLflow: `train_loss`, `mrr_at_10`, `ndcg_at_10`, `accuracy`.

---

## How quantization works

Quantization shrinks models and speeds up inference on CPU without retraining.

### Reranker

| Environment | Mode | Config |
|-------------|------|--------|
| CPU | **INT8 dynamic quantization** on `Linear` layers | `RERANKER_QUANTIZE=true` |
| CUDA GPU | **FP16** half-precision | `RERANKER_FP16_CUDA=true` |

**Load order at startup:**

1. If `best_quantized.pt` exists → load INT8 checkpoint  
2. Else load `best.pt` → quantize in memory (CPU) or cast to FP16 (GPU)  
3. Batched inference via `RERANKER_BATCH_SIZE` (default 16)

**Example size reduction (DistilBERT reranker):**

```
FP32 in-memory: 253 MB  →  INT8: 91 MB  (~64% smaller)
```

Export manually anytime:

```powershell
python run.py quantize
```

### Embeddings (SentenceTransformer)

When `EMBEDDING_QUANTIZE=true` and no GPU is detected, the underlying transformer is INT8-quantized on first load. Encoding uses batched `EMBEDDING_BATCH_SIZE` (default 32).

### What quantization does *not* affect

- FAISS index storage (already compact float vectors)
- Ollama LLM weights — use Ollama quants separately (e.g. `qwen3:8b-q4_0`)

---

## Project structure

```
RAGRank/
├── src/
│   ├── config.py                 # All env-based settings
│   ├── pipeline.py               # RAG + retrieve-only orchestration
│   ├── quantization.py           # Shared INT8 / FP16 helpers
│   ├── reranker/
│   │   ├── model.py              # NeuralReranker + RerankerService
│   │   ├── dataset.py            # Training pairs — EDIT THIS for your domain
│   │   ├── train.py              # Training loop + MLflow logging
│   │   └── quantize.py           # Export best_quantized.pt
│   ├── retriever/                # Chunking + FAISS vector store
│   ├── generator/                # LLM client (Ollama, Qwen, mock)
│   ├── evaluator/                # LLM-as-a-judge metrics
│   └── app/                      # FastAPI + Streamlit dashboard
├── data/
│   ├── faiss_index/              # Persisted FAISS index + chunks
│   ├── models/reranker/          # best.pt, best_quantized.pt
│   └── mlruns/                   # MLflow experiment logs
├── tests/
├── run.py                        # api | dashboard | train | quantize
├── requirements.txt
└── docs/PYTORCH_AND_INTERVIEW.md
```

---

## Prerequisites

- Python 3.11+ (3.12 tested)
- Optional: NVIDIA GPU + CUDA (faster training; FP16 reranker inference)
- **Local LLM:** [Ollama](https://ollama.com/) with `qwen3:8b` (recommended)
- **Cloud LLM:** DashScope API key for Qwen

---

## Setup

### 1. Virtual environment

**Windows (PowerShell):**

```powershell
cd c:\Users\kulde\OneDrive\Desktop\LLM\PyTorch
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment

```powershell
copy .env.example .env
```

**Recommended local setup (Ollama):**

```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_THINK=false
LLM_TIMEOUT=600

RERANKER_MODEL=distilbert-base-uncased
RERANKER_QUANTIZE=true
EMBEDDING_QUANTIZE=true
```

Or use `scripts/setup_ollama_qwen.bat` to pull the model.

**Mock mode** (`LLM_PROVIDER=mock`) — no API keys; good for tests and retrieval-only debugging.

### 3. Key configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `ollama`, `qwen`, `mock` | `mock` |
| `LLM_MODEL` | Model name | `qwen3:8b` |
| `RERANKER_MODEL` | HuggingFace base | `distilbert-base-uncased` |
| `RERANKER_EPOCHS` | Training epochs | `3` |
| `RERANKER_LOSS` | `bce` or `margin_mse` | `bce` |
| `RERANKER_QUANTIZE` | INT8 (CPU) / enable quant export | `true` |
| `RERANKER_FP16_CUDA` | FP16 on GPU | `true` |
| `RERANKER_BATCH_SIZE` | Reranker inference batch | `16` |
| `EMBEDDING_MODEL` | SentenceTransformer | `all-MiniLM-L6-v2` |
| `EMBEDDING_QUANTIZE` | INT8 embeddings on CPU | `true` |
| `RETRIEVAL_TOP_K` | FAISS candidates | `20` |
| `RERANK_TOP_K` | Chunks sent to LLM | `5` |
| `MLFLOW_ALLOW_FILE_STORE` | Required on Windows for local MLflow | set before train |

---

## How to run

Open **two terminals** (venv activated):

```powershell
# Terminal 1 — API
python run.py api

# Terminal 2 — Dashboard
python run.py dashboard
```

| Service | URL |
|---------|-----|
| FastAPI docs | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| MLflow UI | `./data/mlruns` (or `mlflow ui --backend-store-uri ./data/mlruns`) |

### Dashboard tabs

| Tab | Purpose |
|-----|---------|
| **Documents** | Upload PDF / TXT / MD → FAISS index |
| **RAG Lab** | Full pipeline with vs without reranking (+ optional LLM eval) |
| **Retrieval Check** | FAISS vs trained vs untrained reranker — **no LLM** |
| **Telemetry** | Latency charts + evaluation radar |

---

## Train the reranker (step by step)

### 1. Customize training data

Edit `src/reranker/dataset.py` → `default_training_examples()`. Add 20–100+ triples for your domain.

### 2. Run training

**Windows:**

```powershell
set MLFLOW_ALLOW_FILE_STORE=true
python run.py train
```

**Optional flags** (via env or code): `RERANKER_EPOCHS=5`, `RERANKER_LR=2e-5`, `RERANKER_LOSS=margin_mse`

### 3. Example output

```
Starting reranker training...
Epoch 1/3 | loss=0.6961 | train_acc=0.3438 | mrr_at_10=1.0000 | ndcg_at_10=1.0000 | accuracy=1.0000
Epoch 2/3 | loss=0.4123 | train_acc=0.8750 | mrr_at_10=1.0000 | ndcg_at_10=1.0000 | accuracy=1.0000
Epoch 3/3 | loss=0.2891 | train_acc=1.0000 | mrr_at_10=1.0000 | ndcg_at_10=1.0000 | accuracy=1.0000
Quantized reranker saved to data/models/reranker/best_quantized.pt
  FP32 in-memory: 253.2 MB -> INT8: 91.0 MB (64% smaller)
Training complete. Checkpoint: data/models/reranker/best.pt
```

### 4. Restart API

The API loads checkpoints at startup:

```powershell
python run.py api
```

Health response includes quantization mode:

```json
{
  "status": "ok",
  "num_chunks": 42,
  "llm_provider": "ollama",
  "reranker_quantization": "int8",
  "embedding_quantization": "int8"
}
```

### 5. Verify in Retrieval Check

1. Ingest documents  
2. Open **Retrieval Check**  
3. Run a query — compare columns 1 (FAISS), 2 (trained), 3 (untrained)  

If training helped, column 2 should rank relevant chunks higher than column 3.

---

## CLI reference

```powershell
python run.py api              # FastAPI backend (:8000)
python run.py dashboard        # Streamlit UI (:8501)
python run.py train            # Fine-tune reranker → best.pt + best_quantized.pt
python run.py quantize         # Re-export INT8 from existing best.pt
python run.py all              # Print run instructions
```

---

## API examples

### Ingest

```powershell
curl -X POST http://localhost:8000/ingest -F "files=@document.pdf"
```

### Full RAG query

```powershell
curl -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d '{"query": "Best SUV under 15 lakh?", "use_reranker": true, "run_eval": false}'
```

### Retrieval only (no LLM)

```powershell
curl -X POST http://localhost:8000/retrieve `
  -H "Content-Type: application/json" `
  -d '{"query": "electric SUV India", "retrieval_top_k": 20, "rerank_top_k": 5}'
```

Returns `retrieved`, `faiss_only`, `reranked_trained`, `reranked_untrained`, `rank_changes`, `latency_ms`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Status, chunk count, quantization modes |
| `POST` | `/ingest` | Upload PDF/TXT/MD |
| `POST` | `/retrieve` | Retrieve + rerank only (3-way comparison) |
| `POST` | `/query` | Full RAG pipeline |
| `POST` | `/query/stream` | Streaming SSE |
| `GET` | `/eval-report` | LLM-judge aggregate metrics |
| `GET` | `/stats` | Latency statistics |

---

## Tests

```powershell
pytest tests/ -v
```

Covers reranker forward pass, INT8 export, FAISS ingestion, pipeline E2E, FastAPI endpoints, and evaluator metrics.

---

## Typical workflow

1. **Setup** — venv, `pip install`, copy `.env`, start Ollama  
2. **Start** — `python run.py api` + `python run.py dashboard`  
3. **Ingest** — upload domain documents  
4. **Baseline** — **Retrieval Check** with untrained reranker  
5. **Train** — edit `dataset.py`, `python run.py train`, restart API  
6. **Compare** — Retrieval Check again (trained vs untrained)  
7. **Quantize** — automatic after train, or `python run.py quantize`  
8. **RAG Lab** — full answers with reranking on/off  
9. **Evaluate** — enable LLM-as-judge in RAG Lab or check Telemetry  

---

## Docker

```powershell
docker-compose up --build
```

| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000 |
| Streamlit | http://localhost:8501 |
| MLflow | http://localhost:5000 |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Dashboard can't reach API | Start `python run.py api`; check `API_BASE_URL` |
| `No documents indexed` | Ingest via Documents tab or `/ingest` |
| Trained vs untrained identical | No `best.pt` yet — run `python run.py train` and restart API |
| MLflow error on Windows | `set MLFLOW_ALLOW_FILE_STORE=true` before train |
| MLflow metric name error | Fixed — metrics use `mrr_at_10` not `mrr@10` |
| Query timeout with Ollama | Increase `LLM_TIMEOUT` / `API_CLIENT_TIMEOUT`; set `OLLAMA_THINK=false` |
| Slow first query | HuggingFace models download once (~250 MB) |
| Quantization not shown | Restart API after train; check `/health` |

---

## Further reading

- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** — full design spec  
- **[docs/PYTORCH_AND_INTERVIEW.md](docs/PYTORCH_AND_INTERVIEW.md)** — PyTorch concepts + interview Q&A  

---

## License

Educational / portfolio project. HuggingFace model weights follow their respective licenses.
