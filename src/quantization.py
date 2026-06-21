"""Shared PyTorch quantization helpers for inference."""

from __future__ import annotations

import torch
import torch.nn as nn

LINEAR_LAYERS = {nn.Linear}


def model_size_mb(module: nn.Module) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in module.buffers())
    return (param_bytes + buffer_bytes) / (1024**2)


def apply_int8_dynamic_quantization(model: nn.Module) -> nn.Module:
    """INT8 dynamic quantization for Linear layers (CPU inference)."""
    model.eval()
    return torch.quantization.quantize_dynamic(
        model, LINEAR_LAYERS, dtype=torch.qint8
    )


def apply_fp16(model: nn.Module) -> nn.Module:
    """Half-precision weights for GPU inference."""
    model.eval()
    return model.half()
