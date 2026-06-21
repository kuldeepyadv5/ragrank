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
            query="Best SUV under 15 lakh in India for a family of 5",
            positive="Mahindra XUV 3XO offers good safety ratings, spacious seating for 5, and is priced under 15 lakh.",
            negative="BMW X5 is a luxury SUV priced above 1 crore."
        ),

        QueryExample(
            query="Best mileage petrol car under 10 lakh",
            positive="Maruti Suzuki Baleno delivers around 22 kmpl mileage and falls within a 10 lakh budget.",
            negative="Toyota Fortuner focuses on power and space rather than fuel efficiency."
        ),

        QueryExample(
            query="Latest electric SUV launched in India",
            positive="Mahindra BE 6 is a recently launched electric SUV with modern features and long driving range.",
            negative="Hyundai Creta Petrol is an internal combustion vehicle."
        ),

        QueryExample(
            query="Safe car for highway driving",
            positive="Volkswagen Virtus has strong build quality and high safety ratings suitable for highway travel.",
            negative="A low-speed electric scooter is not designed for highway driving."
        ),

        QueryExample(
            query="Best car for Bangalore city traffic",
            positive="Hyundai i20 offers compact dimensions, automatic transmission options, and city-friendly driving.",
            negative="Force Gurkha is primarily designed for off-road adventures."
        ),

        QueryExample(
            query="Electric car with more than 400 km range",
            positive="Tata Curvv EV offers a claimed range exceeding 400 km on a full charge.",
            negative="Maruti WagonR Petrol does not provide electric driving range."
        ),

        QueryExample(
            query="Best SUV for off-road trips",
            positive="Mahindra Thar includes 4x4 capability, high ground clearance, and terrain modes.",
            negative="Honda Amaze is a compact sedan focused on city driving."
        ),

        QueryExample(
            query="7 seater car under 20 lakh",
            positive="Mahindra Scorpio N provides 7-seat capacity and fits within the specified budget.",
            negative="MG Comet EV seats only four passengers."
        ),

        QueryExample(
            query="Best luxury sedan in India",
            positive="Mercedes-Benz E-Class offers premium comfort, technology, and luxury features.",
            negative="Maruti Alto is an entry-level hatchback."
        ),

        QueryExample(
            query="Car suitable for daily office commute and occasional long drives",
            positive="Honda City provides comfort, reliability, and fuel efficiency for both city and highway use.",
            negative="Mahindra Bolero Camper is designed primarily for commercial utility."
        )
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
