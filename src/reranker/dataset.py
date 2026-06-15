"""PyTorch Dataset and DataLoader for reranker training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import get_config


@dataclass
class QueryExample:
    query: str
    positive: str
    negative: str


class RerankerPairDataset(Dataset):
    """Dataset of (query, document, label) triples for cross-encoder training."""

    def __init__(
        self,
        examples: list[QueryExample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples) * 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx // 2]
        is_positive = idx % 2 == 0
        document = example.positive if is_positive else example.negative
        label = 1.0 if is_positive else 0.0

        encoded = self.tokenizer(
            example.query,
            document,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def build_dataloader(
    examples: list[QueryExample],
    batch_size: int | None = None,
    shuffle: bool = True,
) -> DataLoader:
    cfg = get_config().reranker
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    dataset = RerankerPairDataset(examples, tokenizer, cfg.max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size or cfg.batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


class RerankerPairwiseDataset(Dataset):
    """Yields (query, positive, negative) triples for margin-based training."""

    def __init__(
        self,
        examples: list[QueryExample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]

        def _encode(query: str, document: str) -> dict[str, torch.Tensor]:
            encoded = self.tokenizer(
                query,
                document,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }

        pos = _encode(example.query, example.positive)
        neg = _encode(example.query, example.negative)
        return {
            "pos_input_ids": pos["input_ids"],
            "pos_attention_mask": pos["attention_mask"],
            "neg_input_ids": neg["input_ids"],
            "neg_attention_mask": neg["attention_mask"],
        }


def build_pairwise_dataloader(
    examples: list[QueryExample],
    batch_size: int | None = None,
    shuffle: bool = True,
) -> DataLoader:
    cfg = get_config().reranker
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    dataset = RerankerPairwiseDataset(examples, tokenizer, cfg.max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size or cfg.batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def default_training_examples() -> list[QueryExample]:
    """Minimal synthetic training set for bootstrapping."""
    return [
        QueryExample(
            query="What is a cross-encoder reranker?",
            positive="A cross-encoder feeds query and document together into a transformer.",
            negative="PyTorch tensors support automatic differentiation via autograd.",
        ),
        QueryExample(
            query="How does FAISS work?",
            positive="FAISS is a library for efficient similarity search over dense vectors.",
            negative="Streamlit is a framework for building interactive Python dashboards.",
        ),
        QueryExample(
            query="What metrics evaluate retrieval quality?",
            positive="MRR@K and NDCG@K are common ranking metrics for retrieval systems.",
            negative="AdamW is an optimizer with decoupled weight decay.",
        ),
    ]


def collate_validation_groups(
    examples: list[QueryExample],
) -> list[dict[str, Any]]:
    """Build validation groups for MRR/NDCG computation."""
    groups = []
    for ex in examples:
        groups.append(
            {
                "query": ex.query,
                "documents": [ex.positive, ex.negative],
                "relevance": [1.0, 0.0],
            }
        )
    return groups
