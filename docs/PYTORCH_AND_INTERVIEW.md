# PyTorch Deep Dive & Interview Questions

This document maps **every PyTorch concept used in this project** to the actual source code, then provides **interview-style Q&A** you can use to explain the system.

---

## Table of Contents

1. [PyTorch Concepts Used in This Project](#pytorch-concepts-used-in-this-project)
2. [Code Walkthrough: Cross-Encoder Reranker](#code-walkthrough-cross-encoder-reranker)
3. [Training Loop Explained](#training-loop-explained)
4. [Bi-Encoder vs Cross-Encoder](#bi-encoder-vs-cross-encoder)
5. [Interview Questions & Answers](#interview-questions--answers)

---

## PyTorch Concepts Used in This Project

| Concept | Where in code | What it does here |
|---------|---------------|-------------------|
| **`nn.Module`** | `src/reranker/model.py` → `NeuralReranker` | Base class for the reranker neural network |
| **`nn.Linear`** | `src/reranker/model.py` → `self.classifier` | Maps CLS hidden state → 1 relevance score |
| **`forward()`** | `src/reranker/model.py` | Defines the computation graph: encoder → CLS token → classifier |
| **Tensors** | Everywhere in reranker | `input_ids`, `attention_mask`, `label` are `torch.Tensor` |
| **`.to(device)`** | `model.py`, `train.py` | Moves model/data to GPU (`cuda`) or CPU |
| **`torch.no_grad()`** | `model.py` (inference), `train.py` (eval) | Disables gradient tracking during inference |
| **`model.eval()` / `model.train()`** | `model.py`, `train.py` | Switches dropout/batchnorm behavior |
| **`Dataset`** | `src/reranker/dataset.py` → `RerankerPairDataset` | Provides (query, doc, label) training samples |
| **`DataLoader`** | `src/reranker/dataset.py` → `build_dataloader()` | Batches samples for training |
| **`BCEWithLogitsLoss`** | `src/reranker/train.py` | Binary classification: relevant (1) vs irrelevant (0) |
| **`AdamW`** | `src/reranker/train.py` | Optimizer with decoupled weight decay |
| **`CosineAnnealingLR`** | `src/reranker/train.py` | Learning rate schedule |
| **Mixed Precision (AMP)** | `src/reranker/train.py` | `torch.amp.autocast` + `GradScaler` on CUDA |
| **`loss.backward()`** | `src/reranker/train.py` | Autograd computes gradients |
| **`torch.save()`** | `src/reranker/train.py` | Saves best checkpoint to `data/models/reranker/best.pt` |
| **`torch.load()`** | `src/reranker/model.py` | Loads trained weights at inference time |

### What does NOT use PyTorch in this project?

| Component | Library instead |
|-----------|-----------------|
| Vector retrieval (FAISS) | `sentence-transformers` + `faiss-cpu` (embeddings computed outside PyTorch training loop) |
| LLM generation | OpenAI / Gemini / Ollama APIs |
| PDF parsing | `pypdf` |
| API / Dashboard | FastAPI, Streamlit |

---

## Code Walkthrough: Cross-Encoder Reranker

### 1. `nn.Module` — the model class

**File:** `src/reranker/model.py`

```python
class NeuralReranker(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
```

**What to understand:**
- `nn.Module` is PyTorch's base class for any neural network.
- `super().__init__()` registers sub-modules so PyTorch tracks their parameters.
- `AutoModel.from_pretrained()` loads a pre-trained transformer (DistilBERT).
- `nn.Linear(768, 1)` is a single fully-connected layer: hidden size → 1 score.

**Interview one-liner:** *"We subclass `nn.Module`, stack a pre-trained transformer encoder with a linear head, and fine-tune it as a binary relevance classifier."*

---

### 2. `forward()` — the computation graph

```python
def forward(self, input_ids, attention_mask):
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    cls_representation = outputs.last_hidden_state[:, 0, :]
    score = self.classifier(cls_representation)
    return score.squeeze(-1)
```

**Step by step:**

1. **Input:** tokenized query + document pair → `input_ids` shape `[batch, seq_len]`
2. **Encoder:** DistilBERT runs self-attention over the **combined** query+doc tokens
3. **CLS token:** `[:, 0, :]` takes the first token's hidden state as the sentence-pair representation
4. **Classifier:** linear layer outputs one logit per pair
5. **Output:** relevance score (higher = more relevant)

**Why CLS token?** BERT-style models prepend a `[CLS]` token; its final hidden state is trained to summarize the entire input sequence.

---

### 3. Inference — `@torch.no_grad()` and `.eval()`

**File:** `src/reranker/model.py` → `RerankerService`

```python
self.model.eval()

@torch.no_grad()
def score_pairs(self, query, documents):
    ...
    score = self.model(input_ids, attention_mask).item()
```

- **`model.eval()`** — disables dropout; batch norm uses running stats.
- **`@torch.no_grad()`** — no gradient computation → faster inference, less memory.
- **`.item()`** — converts a single-element tensor to a Python float.

---

## Training Loop Explained

**File:** `src/reranker/train.py`

### Dataset design

**File:** `src/reranker/dataset.py`

Each training example has: `(query, positive_doc, negative_doc)`.

The dataset doubles each example into two samples:
- Index 0, 2, 4… → positive pair, label = 1.0
- Index 1, 3, 5… → negative pair, label = 0.0

This teaches the model: *"this doc is relevant to this query"* vs *"this doc is not."*

### Loss function

```python
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, labels)
```

**BCEWithLogitsLoss** = Sigmoid + Binary Cross Entropy in one numerically stable operation.

- Label `1.0` → model should output high score for relevant pairs
- Label `0.0` → model should output low score for irrelevant pairs

### Optimizer & scheduler

```python
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

- **AdamW:** adaptive learning rate + weight decay (standard for transformer fine-tuning)
- **CosineAnnealingLR:** LR starts high, smoothly decreases to near zero

### Mixed precision (GPU only)

```python
with torch.amp.autocast("cuda"):
    logits = model(input_ids, attention_mask)
    loss = criterion(logits, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Why?** FP16 computation is ~2× faster and uses less VRAM. `GradScaler` prevents underflow in small gradients.

On CPU, the code falls back to standard FP32 training.

### Validation metrics

| Metric | Meaning |
|--------|---------|
| **MRR@K** (Mean Reciprocal Rank) | If the first relevant doc is at rank 3, MRR = 1/3 |
| **NDCG@K** | Graded ranking quality; rewards putting relevant docs higher |

Best checkpoint saved when MRR@10 improves.

---

## Bi-Encoder vs Cross-Encoder

This project uses **both** — they serve different stages:

```
                    BI-ENCODER (retrieval)          CROSS-ENCODER (reranking)
                    ─────────────────────          ──────────────────────────
Input               Query and doc encoded          Query + doc encoded
                    separately                     TOGETHER

Interaction         None (just dot product         Full self-attention between
                    of two vectors)                query and doc tokens

Speed               Very fast (pre-compute         Slow (must run per pair)
                    all doc embeddings)

Used in project     FAISS + SentenceTransformer    PyTorch NeuralReranker
File                src/retriever/vector_store.py  src/reranker/model.py
```

**Why both?**
1. Bi-encoder retrieves top-20 from thousands of chunks quickly.
2. Cross-encoder reranks top-20 to top-5 with higher accuracy.

---

## Interview Questions & Answers

### PyTorch Fundamentals

**Q1: What is a tensor and how is it used here?**

A tensor is PyTorch's multi-dimensional array (like NumPy, but GPU-accelerable with autograd). In this project:
- `input_ids` — `[batch, seq_len]` integer tensor of token IDs
- `attention_mask` — `[batch, seq_len]` mask (1 = real token, 0 = padding)
- `label` — `[batch]` float tensor (0.0 or 1.0)

---

**Q2: What is autograd and where do we use it?**

Autograd automatically computes gradients via backpropagation. Used during **training** in `train.py`:

```python
loss.backward()   # computes ∂loss/∂weights for all parameters
optimizer.step()  # updates weights using those gradients
```

During **inference**, we disable it with `@torch.no_grad()`.

---

**Q3: Explain `nn.Module` vs a plain Python class.**

`nn.Module` provides:
- Automatic parameter registration (`model.parameters()`)
- `.to(device)` moves all parameters to GPU
- `.train()` / `.eval()` mode switching
- State dict save/load (`torch.save` / `torch.load`)

Our `NeuralReranker(nn.Module)` inherits all of this.

---

**Q4: What is the difference between `model.train()` and `model.eval()`?**

| Mode | Dropout | BatchNorm | Gradients |
|------|---------|-----------|-----------|
| `.train()` | Active | Uses batch stats | Computed |
| `.eval()` | Disabled | Uses running stats | Typically disabled |

We call `.train()` in the training loop and `.eval()` before inference/validation.

---

**Q5: Why use `BCEWithLogitsLoss` instead of `BCELoss`?**

`BCEWithLogitsLoss` applies sigmoid internally in a numerically stable way. Applying sigmoid separately then BCE can cause log(0) errors. Our model outputs raw logits (no sigmoid in forward), so this loss is the correct choice.

---

**Q6: What is mixed precision training?**

Training with FP16 (half precision) for forward/backward passes while keeping FP32 master weights. Benefits: faster training, less GPU memory. In `train.py`, we use `torch.amp.autocast("cuda")` and `GradScaler` — only activated when CUDA is available.

---

### RAG & System Design

**Q7: Explain the full RAG pipeline in this project.**

1. **Ingest:** PDF/text → chunked → embedded → stored in FAISS
2. **Retrieve:** query embedded → FAISS returns top-20 similar chunks (bi-encoder)
3. **Rerank:** PyTorch cross-encoder scores each (query, chunk) pair → top-5
4. **Generate:** top-5 chunks sent as context to LLM → answer
5. **Evaluate:** LLM-as-a-judge scores faithfulness, relevance, context precision

Orchestrated in `src/pipeline.py`.

---

**Q8: Why rerank if FAISS already returns similar documents?**

Bi-encoder retrieval is fast but approximate — it compares independent embeddings and misses fine-grained semantic matches. Cross-encoder reranking is slower but more accurate because query and document tokens attend to each other directly. Typical pattern: retrieve 20–100, rerank to 5.

---

**Q9: What is a cross-encoder vs bi-encoder?**

- **Bi-encoder:** `embed(query)` and `embed(doc)` separately → cosine similarity. Fast, but no token-level interaction.
- **Cross-encoder:** `model(query, doc)` jointly → single score. Slow, but captures nuanced relevance.

Our reranker is a cross-encoder (`NeuralReranker`). Our retriever uses a bi-encoder (`SentenceTransformer`).

---

**Q10: How do you evaluate RAG quality without human labels?**

We use **LLM-as-a-judge** (`src/evaluator/metrics.py`):
- **Context Precision** — are retrieved chunks actually useful?
- **Answer Faithfulness** — is the answer grounded in context (no hallucination)?
- **Answer Relevance** — does the answer address the question?

The judge LLM returns structured JSON scores 0.0–1.0.

---

**Q11: What are MRR@K and NDCG@K?**

Used to evaluate the reranker during training (`src/reranker/train.py`):

- **MRR@K:** average of 1/rank for the first relevant document within top-K
- **NDCG@K:** normalized discounted cumulative gain — penalizes relevant docs ranked low

Example: if positive doc moves from rank 2 → rank 1 after reranking, MRR improves from 0.5 → 1.0.

---

**Q12: How would you scale this to millions of documents?**

1. **Retrieval:** FAISS `IndexIVFFlat` or `IndexHNSW` (set `FAISS_USE_IVF=true`), shard across machines
2. **Reranking:** batch cross-encoder inference on GPU, limit to top-20 candidates
3. **Embeddings:** pre-compute and cache document embeddings offline
4. **Serving:** containerize with Docker, horizontal scale FastAPI replicas
5. **Training:** use larger cross-encoder base (`ms-marco-MiniLM-L-6-v2`), train on MS MARCO dataset

---

**Q13: What happens in `forward()` step by step for one query-document pair?**

1. Tokenizer converts `"What is FAISS?" + "FAISS is a vector search library"` → `input_ids`
2. DistilBERT encoder runs 6 transformer layers with self-attention
3. Extract hidden state of `[CLS]` token → 768-dim vector
4. Linear layer maps 768 → 1 logit
5. Higher logit = model predicts the pair is relevant

---

**Q14: Why DistilBERT and not full BERT?**

DistilBERT has 6 layers vs BERT's 12, ~40% smaller, ~60% faster, with ~97% of BERT's performance on many tasks. Good tradeoff for a reranker that runs on every query.

Configurable via `RERANKER_MODEL` in `.env`. For production reranking, `cross-encoder/ms-marco-MiniLM-L-6-v2` is a strong alternative (pre-trained on MS MARCO passage ranking).

---

**Q15: How does the Dataset/DataLoader pipeline work?**

```python
# dataset.py
class RerankerPairDataset(Dataset):
    def __getitem__(self, idx):
        # returns dict of tensors: input_ids, attention_mask, label

# train.py
loader = DataLoader(dataset, batch_size=16, shuffle=True)
for batch in loader:
    # batch["input_ids"] shape: [16, 256]
    # batch["label"] shape: [16]
```

`DataLoader` handles batching, shuffling, and optional multi-worker loading (`num_workers=0` on Windows for simplicity).

---

### Practical / Coding Questions

**Q16: Show how you would add a new training example.**

Edit or extend `default_training_examples()` in `src/reranker/dataset.py`:

```python
QueryExample(
    query="How does attention work?",
    positive="Attention computes weighted sums of value vectors based on query-key similarity.",
    negative="FAISS stores vectors in an inverted file index for fast search.",
)
```

Then run: `python run.py train`

---

**Q17: How do you load a trained checkpoint for inference?**

Automatically in `RerankerService.__init__`:

```python
ckpt = cfg.checkpoint_dir / "best.pt"
if ckpt.exists():
    state = torch.load(ckpt, map_location=self.device, weights_only=True)
    self.model.load_state_dict(state)
```

Train first with `python run.py train`, then restart the API — reranking will use fine-tuned weights.

---

**Q18: What would you monitor in production?**

This project tracks:
- **Latency:** retrieval_ms, rerank_ms, generation_ms, total_ms (in `pipeline.py`)
- **Quality:** context_precision, faithfulness, relevance (evaluator)
- **Training:** loss, MRR@K, NDCG@K (MLflow/TensorBoard)

Dashboard Telemetry tab and `/stats` endpoint expose these.

---

## Quick Reference: File → PyTorch Feature

```
src/reranker/model.py
  ├── nn.Module          → NeuralReranker class
  ├── nn.Linear          → classifier head
  ├── torch.Tensor       → inputs/outputs
  ├── .to(device)        → GPU/CPU placement
  ├── @torch.no_grad()   → inference mode
  └── torch.load()       → checkpoint loading

src/reranker/dataset.py
  ├── Dataset            → RerankerPairDataset
  ├── DataLoader         → build_dataloader()
  └── torch.tensor       → label creation

src/reranker/train.py
  ├── BCEWithLogitsLoss  → training loss
  ├── AdamW              → optimizer
  ├── CosineAnnealingLR  → LR schedule
  ├── torch.amp.autocast → mixed precision
  ├── GradScaler         → AMP gradient scaling
  ├── loss.backward()    → autograd
  ├── torch.save()       → checkpoint export
  └── @torch.no_grad()   → validation eval
```

---

## Study Path (recommended order)

1. Read `15.md` in project root — basic tensor/matrix intuition
2. Read `src/reranker/model.py` — understand `nn.Module` and `forward()`
3. Read `src/reranker/dataset.py` — understand `Dataset` / `DataLoader`
4. Read `src/reranker/train.py` — full training loop
5. Run `python run.py train` and watch loss/MRR in terminal
6. Run `pytest tests/test_reranker.py -v` — see tests as executable documentation
7. Practice explaining Q7, Q8, Q9, Q13 out loud — these come up most in ML/RAG interviews
