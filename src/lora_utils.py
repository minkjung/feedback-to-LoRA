"""LoRA utilities: build per-layer module name list, apply / remove LoRA from frozen target."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LoRASpec:
    """Per-layer LoRA shape description. Gemma 4 mixes local/global attention
    with different q_proj dims, so dims must be tracked per (layer, module)."""

    num_layers: int
    target_modules: list[str]
    rank: int
    per_layer_in: dict[str, int]   # "layer_{i}_{module}" -> in_features
    per_layer_out: dict[str, int]  # "layer_{i}_{module}" -> out_features

    def layer_module_keys(self) -> list[str]:
        return list(self.per_layer_in.keys())

    def key(self, layer_idx: int, module_name: str) -> str:
        return f"layer_{layer_idx}_{module_name}"


def build_lora_spec(target_model: nn.Module, target_modules: list[str], rank: int) -> LoRASpec:
    base = getattr(target_model, "model", target_model)
    layers = _find_decoder_layers(base)
    num_layers = len(layers)

    per_layer_in: dict[str, int] = {}
    per_layer_out: dict[str, int] = {}
    for i, layer in enumerate(layers):
        for m in target_modules:
            linear = _find_named_linear(layer, m)
            if linear is None:
                # some layers may legitimately lack a module (rare); skip silently
                continue
            key = f"layer_{i}_{m}"
            per_layer_in[key] = linear.in_features
            per_layer_out[key] = linear.out_features

    if not per_layer_in:
        raise ValueError("build_lora_spec found no target linears in any decoder layer")

    return LoRASpec(
        num_layers=num_layers,
        target_modules=list(target_modules),
        rank=rank,
        per_layer_in=per_layer_in,
        per_layer_out=per_layer_out,
    )


def _find_decoder_layers(module: nn.Module) -> list[nn.Module]:
    # For multimodal models, prioritize language_model to avoid vision encoder layers
    lang = getattr(module, "language_model", None)
    if lang is not None:
        result = _find_decoder_layers(lang)
        if result:
            return result
    for name in ("layers", "decoder", "h", "blocks"):
        sub = getattr(module, name, None)
        if isinstance(sub, nn.ModuleList):
            return list(sub)
        if sub is not None and hasattr(sub, "layers"):
            inner = sub.layers
            if isinstance(inner, nn.ModuleList):
                return list(inner)
    for child in module.children():
        if isinstance(child, nn.ModuleList) and len(child) > 0:
            return list(child)
        result = _find_decoder_layers(child)
        if result:
            return result
    return []


def _find_named_linear(module: nn.Module, name: str) -> nn.Linear | None:
    for sub_name, sub in module.named_modules():
        if sub_name.endswith(name) and isinstance(sub, nn.Linear):
            return sub
    return None


@contextmanager
def lora_applied(
    target_model: nn.Module,
    spec: LoRASpec,
    lora_weights: dict[str, tuple[torch.Tensor, torch.Tensor]],
):
    """Context manager: patch each target Linear with a LoRA delta."""
    base = getattr(target_model, "model", target_model)
    layers = _find_decoder_layers(base)

    handles: list[tuple[nn.Linear, object]] = []
    try:
        for i, layer in enumerate(layers):
            for m in spec.target_modules:
                linear = _find_named_linear(layer, m)
                if linear is None:
                    continue
                key = f"layer_{i}_{m}"
                if key not in lora_weights:
                    continue
                A, B = lora_weights[key]
                if A.shape[-1] != linear.in_features or B.shape[0] != linear.out_features:
                    raise ValueError(
                        f"LoRA shape mismatch at {key}: "
                        f"A={tuple(A.shape)} B={tuple(B.shape)} vs "
                        f"linear in={linear.in_features} out={linear.out_features}"
                    )
                _patch_linear(linear, A, B, handles)
        yield
    finally:
        for linear, original_forward in handles:
            linear.forward = original_forward  # type: ignore[assignment]


def _patch_linear(
    linear: nn.Linear,
    A: torch.Tensor,
    B: torch.Tensor,
    handles: list,
) -> None:
    """Patch linear.forward: out = W(x) + (x @ A^T) @ B^T."""
    original_forward = linear.forward

    def patched(x: torch.Tensor, _orig=original_forward, _A=A, _B=B) -> torch.Tensor:
        out = _orig(x)
        # x: (..., in_features)
        # A: (rank, in_features),  A^T: (in_features, rank)
        # B: (out_features, rank), B^T: (rank, out_features)
        delta = (x @ _A.transpose(-1, -2).to(x.dtype)) @ _B.transpose(-1, -2).to(x.dtype)
        return out + delta

    linear.forward = patched  # type: ignore[assignment]
    handles.append((linear, original_forward))
