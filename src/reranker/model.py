"""Cross-Encoder neural reranker model."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.config import get_config


class NeuralReranker(nn.Module):
    """Cross-encoder that scores query-document pairs jointly."""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.model_name = model_name
        self._use_pretrained_head = "cross-encoder" in model_name.lower()

        if self._use_pretrained_head:
            from transformers import AutoModelForSequenceClassification

            self.encoder = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=1
            )
            self.classifier = None
        else:
            self.encoder = AutoModel.from_pretrained(model_name)
            self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self._use_pretrained_head:
            logits = self.encoder(input_ids=input_ids, attention_mask=attention_mask).logits
            return logits.squeeze(-1)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        score = self.classifier(cls_representation)
        return score.squeeze(-1)


class RerankerService:
    """Inference wrapper for loading and scoring query-document pairs."""

    def __init__(
        self,
        model_name: str | None = None,
        checkpoint_path: Path | None = None,
        device: str | None = None,
        load_checkpoint: bool = True,
    ):
        cfg = get_config().reranker
        self.model_name = model_name or cfg.model_name
        self.max_length = cfg.max_length
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = NeuralReranker(self.model_name).to(self.device)
        self.checkpoint_loaded = False
        self.checkpoint_path: Path | None = None

        if load_checkpoint:
            ckpt = checkpoint_path or (cfg.checkpoint_dir / "best.pt")
            if ckpt.exists():
                state = torch.load(ckpt, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
                self.checkpoint_loaded = True
                self.checkpoint_path = ckpt
        self.model.eval()

    @torch.no_grad()
    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []

        scores: list[float] = []
        for doc in documents:
            encoded = self.tokenizer(
                query,
                doc,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            score = self.model(input_ids, attention_mask).item()
            scores.append(score)
        return scores

    def rerank(
        self,
        query: str,
        documents: Sequence[dict],
        text_key: str = "text",
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank document dicts by cross-encoder score."""
        cfg = get_config().retriever
        k = top_k or cfg.rerank_top_k
        texts = [doc[text_key] for doc in documents]
        scores = self.score_pairs(query, texts)
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            {**doc, "rerank_score": score, "original_rank": i}
            for i, (doc, score) in enumerate(ranked[:k])
        ]
