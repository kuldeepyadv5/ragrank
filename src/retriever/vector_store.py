"""FAISS vector store with SentenceTransformer embeddings."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import get_config
from src.retriever.text_processor import DocumentChunk


class VectorStore:
    """Manages chunk embeddings and FAISS similarity search."""

    def __init__(
        self,
        embedding_model: str | None = None,
        index_path: Path | None = None,
        use_ivf: bool | None = None,
    ):
        cfg = get_config().retriever
        self.embedding_model_name = embedding_model or cfg.embedding_model
        self.index_path = index_path or cfg.index_path
        self.use_ivf = use_ivf if use_ivf is not None else cfg.use_ivf
        self.top_k = cfg.top_k

        self._encoder: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._chunks: list[dict] = []
        self._dim: int | None = None

        self.index_path.mkdir(parents=True, exist_ok=True)
        self._load_if_exists()

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            raise RuntimeError("Vector index is empty. Ingest documents first.")
        return self._index

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def _load_if_exists(self) -> None:
        index_file = self.index_path / "index.faiss"
        meta_file = self.index_path / "chunks.pkl"
        if index_file.exists() and meta_file.exists():
            self._index = faiss.read_index(str(index_file))
            with meta_file.open("rb") as f:
                self._chunks = pickle.load(f)
            self._dim = self._index.d

    def _build_index(self, vectors: np.ndarray) -> faiss.Index:
        dim = vectors.shape[1]
        self._dim = dim

        if self.use_ivf and len(vectors) >= 100:
            nlist = min(64, max(4, len(vectors) // 10))
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.train(vectors)
            index.add(vectors)
            index.nprobe = min(8, nlist)
        else:
            index = faiss.IndexFlatL2(dim)
            index.add(vectors)
        return index

    def add_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        vectors = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)

        new_records = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source": c.source,
                "page": c.page,
                "metadata": c.metadata,
            }
            for c in chunks
        ]

        if self._index is None:
            self._index = self._build_index(vectors)
            self._chunks = new_records
        else:
            self._index.add(vectors)
            self._chunks.extend(new_records)

        self.save()
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.top_k
        if self._index is None or not self._chunks:
            return []

        query_vec = self.encoder.encode([query], convert_to_numpy=True)
        query_vec = np.asarray(query_vec, dtype=np.float32)
        k = min(k, len(self._chunks))
        distances, indices = self.index.search(query_vec, k)

        results: list[dict] = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = dict(self._chunks[idx])
            chunk["retrieval_score"] = float(-dist)
            chunk["retrieval_rank"] = rank
            results.append(chunk)
        return results

    def save(self) -> None:
        if self._index is None:
            return
        faiss.write_index(self._index, str(self.index_path / "index.faiss"))
        with (self.index_path / "chunks.pkl").open("wb") as f:
            pickle.dump(self._chunks, f)
        meta = {
            "embedding_model": self.embedding_model_name,
            "num_chunks": len(self._chunks),
            "dim": self._dim,
        }
        (self.index_path / "meta.json").write_text(json.dumps(meta, indent=2))

    def clear(self) -> None:
        self._index = None
        self._chunks = []
        self._dim = None
        for f in self.index_path.glob("*"):
            f.unlink()
