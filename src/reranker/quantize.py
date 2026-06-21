"""Export and load INT8-quantized reranker checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from src.config import get_config
from src.quantization import apply_int8_dynamic_quantization, model_size_mb
from src.reranker.model import NeuralReranker


def export_quantized_checkpoint(
    fp32_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Build INT8 checkpoint from FP32 weights (~4x smaller on disk)."""
    cfg = get_config().reranker
    src = fp32_path or (cfg.checkpoint_dir / "best.pt")
    if not src.exists():
        raise FileNotFoundError(f"No FP32 checkpoint at {src}. Run: python run.py train")

    dst = output_path or (cfg.checkpoint_dir / "best_quantized.pt")
    dst.parent.mkdir(parents=True, exist_ok=True)

    model = NeuralReranker(cfg.model_name)
    model.load_state_dict(torch.load(src, map_location="cpu", weights_only=True))
    model.eval()

    fp32_mb = model_size_mb(model)
    quant_model = apply_int8_dynamic_quantization(model)
    quant_mb = model_size_mb(quant_model)

    torch.save(quant_model.state_dict(), dst)
    print(
        f"Quantized reranker saved to {dst}\n"
        f"  FP32 in-memory: {fp32_mb:.1f} MB -> INT8: {quant_mb:.1f} MB "
        f"({100 * (1 - quant_mb / max(fp32_mb, 0.01)):.0f}% smaller)"
    )
    return dst


def export_all() -> None:
    """Quantize reranker checkpoint if present."""
    cfg = get_config().reranker
    fp32 = cfg.checkpoint_dir / "best.pt"
    if fp32.exists():
        export_quantized_checkpoint(fp32)
    else:
        print(f"No reranker checkpoint at {fp32}. Skipping reranker export.")
