"""Training script for the PyTorch cross-encoder reranker."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer

from src.config import get_config
from src.reranker.dataset import (
    QueryExample,
    build_dataloader,
    build_pairwise_dataloader,
    collate_validation_groups,
    default_training_examples,
)
from src.reranker.model import NeuralReranker


def compute_mrr_at_k(ranked_relevance: list[float], k: int = 10) -> float:
    for rank, rel in enumerate(ranked_relevance[:k], start=1):
        if rel > 0:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(ranked_relevance: list[float], k: int = 10) -> float:
    def dcg(rels: list[float]) -> float:
        return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))

    actual = dcg(ranked_relevance[:k])
    ideal = dcg(sorted(ranked_relevance, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (logits > 0).float()
    return (preds == labels).float().mean().item()


def margin_mse_loss(
    pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float
) -> torch.Tensor:
    """MSE loss pushing positive scores above negative by `margin`."""
    target = torch.full_like(pos_scores, margin)
    return nn.functional.mse_loss(pos_scores - neg_scores, target)


@torch.no_grad()
def evaluate_ranking(
    model: NeuralReranker,
    tokenizer,
    validation_groups: list[dict],
    device: torch.device,
    max_length: int,
    k: int = 10,
) -> dict[str, float]:
    model.eval()
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []
    correct = 0
    total = 0

    for group in validation_groups:
        query = group["query"]
        documents = group["documents"]
        relevance = group["relevance"]
        scores: list[float] = []

        for doc in documents:
            encoded = tokenizer(
                query,
                doc,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            score = model(input_ids, attention_mask).item()
            scores.append(score)

        if len(scores) >= 2:
            correct += int(scores[0] > scores[1])
            total += 1

        ranked = sorted(zip(scores, relevance), key=lambda x: x[0], reverse=True)
        ranked_relevance = [rel for _, rel in ranked]
        mrr_scores.append(compute_mrr_at_k(ranked_relevance, k))
        ndcg_scores.append(compute_ndcg_at_k(ranked_relevance, k))

    return {
        f"mrr_at_{k}": sum(mrr_scores) / max(len(mrr_scores), 1),
        f"ndcg_at_{k}": sum(ndcg_scores) / max(len(ndcg_scores), 1),
        "accuracy": correct / max(total, 1),
    }


def _get_scaler(device: torch.device):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


def _train_step_bce(
    model,
    batch,
    device,
    criterion,
    optimizer,
    scaler,
) -> tuple[torch.Tensor, float]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)
    optimizer.zero_grad()

    if scaler is not None:
        with torch.amp.autocast("cuda"):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    acc = compute_accuracy(logits.detach(), labels)
    return loss, acc


def _train_step_margin_mse(
    model,
    batch,
    device,
    optimizer,
    scaler,
    margin: float,
) -> tuple[torch.Tensor, float]:
    pos_ids = batch["pos_input_ids"].to(device)
    pos_mask = batch["pos_attention_mask"].to(device)
    neg_ids = batch["neg_input_ids"].to(device)
    neg_mask = batch["neg_attention_mask"].to(device)
    optimizer.zero_grad()

    if scaler is not None:
        with torch.amp.autocast("cuda"):
            pos_scores = model(pos_ids, pos_mask)
            neg_scores = model(neg_ids, neg_mask)
            loss = margin_mse_loss(pos_scores, neg_scores, margin)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        pos_scores = model(pos_ids, pos_mask)
        neg_scores = model(neg_ids, neg_mask)
        loss = margin_mse_loss(pos_scores, neg_scores, margin)
        loss.backward()
        optimizer.step()

    acc = (pos_scores.detach() > neg_scores.detach()).float().mean().item()
    return loss, acc


def train(
    examples: list[QueryExample] | None = None,
    output_dir: Path | None = None,
) -> Path:
    print("\nStarting reranker training...",output_dir)
    cfg = get_config().reranker
    tracking = get_config().tracking
    train_examples = examples or default_training_examples()
    val_groups = collate_validation_groups(train_examples)
    use_margin = cfg.loss == "margin_mse"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = NeuralReranker(cfg.model_name).to(device)

    if use_margin:
        loader = build_pairwise_dataloader(train_examples, shuffle=True)
    else:
        loader = build_dataloader(train_examples, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))
    criterion = nn.BCEWithLogitsLoss()
    scaler = _get_scaler(device)

    out_dir = output_dir or cfg.checkpoint_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = _init_tracker(tracking)
    best_mrr = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for batch in loader:
            if use_margin:
                loss, acc = _train_step_margin_mse(
                    model, batch, device, optimizer, scaler, cfg.margin
                )
            else:
                loss, acc = _train_step_bce(
                    model, batch, device, criterion, optimizer, scaler
                )

            epoch_loss += loss.item()
            epoch_acc += acc
            num_batches += 1

        scheduler.step()
        metrics = evaluate_ranking(
            model, tokenizer, val_groups, device, cfg.max_length
        )
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        metrics["train_accuracy"] = avg_acc

        _log_metrics(tracker, epoch, avg_loss, metrics)
        print(
            f"Epoch {epoch + 1}/{cfg.epochs} | loss={avg_loss:.4f} | "
            f"train_acc={avg_acc:.4f} | "
            + " | ".join(
                f"{k}={v:.4f}"
                for k, v in metrics.items()
                if k != "train_accuracy"
            )
        )

        if metrics.get("mrr_at_10", 0) >= best_mrr:
            best_mrr = metrics["mrr_at_10"]
            torch.save(model.state_dict(), best_path)

    _finalize_tracker(tracker)
    return best_path


def _init_tracker(tracking_cfg):
    if tracking_cfg.backend == "mlflow":
        import mlflow

        mlflow.set_tracking_uri(tracking_cfg.mlflow_uri)
        mlflow.set_experiment(tracking_cfg.experiment_name)
        mlflow.start_run()
        return ("mlflow", mlflow)
    if tracking_cfg.backend == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(get_config().tracking.mlflow_uri))
        return ("tensorboard", writer)
    return ("none", None)


def _log_metrics(tracker, epoch: int, loss: float, metrics: dict[str, float]) -> None:
    backend, obj = tracker
    if backend == "mlflow":
        obj.log_metric("train_loss", loss, step=epoch)
        for name, value in metrics.items():
            obj.log_metric(name, value, step=epoch)
    elif backend == "tensorboard":
        obj.add_scalar("train/loss", loss, epoch)
        for name, value in metrics.items():
            obj.add_scalar(f"val/{name}", value, epoch)


def _finalize_tracker(tracker) -> None:
    backend, obj = tracker
    if backend == "mlflow":
        obj.end_run()
    elif backend == "tensorboard":
        obj.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cross-encoder reranker")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--loss",
        choices=["bce", "margin_mse"],
        default=None,
        help="Training loss (overrides RERANKER_LOSS env var)",
    )
    args = parser.parse_args()

    cfg = get_config().reranker
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.loss is not None:
        cfg.loss = args.loss  # type: ignore[assignment]

    path = train(output_dir=args.output_dir)
    print(f"Saved best checkpoint to {path}")


if __name__ == "__main__":
    main()
