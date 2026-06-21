"""Cross-Encoder neural reranker model."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.config import get_config
from src.quantization import apply_fp16, apply_int8_dynamic_quantization


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
        quantize: bool | None = None,
    ):
        cfg = get_config().reranker
        self.model_name = model_name or cfg.model_name
        self.max_length = cfg.max_length
        self.batch_size = cfg.batch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.quantize_enabled = cfg.quantize if quantize is None else quantize
        self.use_fp16 = cfg.use_fp16_on_cuda and self.device.type == "cuda"
        self.quantization_mode = "none"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = NeuralReranker(self.model_name)
        self.checkpoint_loaded = False
        self.checkpoint_path: Path | None = None

        ckpt_dir = cfg.checkpoint_dir
        fp32_ckpt = checkpoint_path or (ckpt_dir / "best.pt")
        quant_ckpt = ckpt_dir / "best_quantized.pt"

        if load_checkpoint:
            if checkpoint_path and "quantized" in checkpoint_path.stem and checkpoint_path.exists():
                self._load_quantized_checkpoint(checkpoint_path)
            elif self.quantize_enabled and quant_ckpt.exists() and checkpoint_path is None:
                self._load_quantized_checkpoint(quant_ckpt)
            elif fp32_ckpt.exists():
                state = torch.load(fp32_ckpt, map_location="cpu", weights_only=True)
                self.model.load_state_dict(state)
                self.checkpoint_loaded = True
                self.checkpoint_path = fp32_ckpt
                self._apply_runtime_quantization()
            else:
                self._apply_runtime_quantization()
        else:
            self._apply_runtime_quantization()

        if not self.use_fp16:
            self.model.to(self.device)
        self.model.eval()

    def _load_quantized_checkpoint(self, quant_ckpt: Path) -> None:
        self.model = apply_int8_dynamic_quantization(self.model)
        state = torch.load(quant_ckpt, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.checkpoint_loaded = True
        self.checkpoint_path = quant_ckpt
        self.quantization_mode = "int8"

    def _apply_runtime_quantization(self) -> None:
        if not self.quantize_enabled:
            return

        if self.use_fp16:
            self.model = apply_fp16(self.model.to(self.device))
            self.quantization_mode = "fp16"
            return

        if self.device.type == "cpu":
            self.model = apply_int8_dynamic_quantization(self.model)
            self.quantization_mode = "int8"

    @torch.no_grad()
    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []

        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch_docs = list(documents[start : start + self.batch_size])
            encoded = self.tokenizer(
                [query] * len(batch_docs),
                batch_docs,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            if self.use_fp16:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    batch_scores = self.model(input_ids, attention_mask)
            else:
                batch_scores = self.model(input_ids, attention_mask)

            scores.extend(batch_scores.cpu().tolist())

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
