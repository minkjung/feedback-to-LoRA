"""LoRA utilities: build per-layer module name list, apply / remove LoRA from frozen target."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LoRASpec:
    """Static description of LoRA shape across all target layers."""

    num_layers: int
    target_modules: list[str]
    rank: int
    target_hidden_dims: dict[str, int]       # module name -> in_features
    target_out_dims: dict[str, int]          # module name -> out_features

    def layer_module_keys(self) -> list[str]:
        """Stable ordered keys: layer_{i}_{module}."""
        return [
            f"layer_{i}_{m}" for i in range(self.num_layers) for m in self.target_modules
        ]


def build_lora_spec(target_model: nn.Module, target_modules: list[str], rank: int) -> LoRASpec:
    """Inspect a target HF causal LM and produce the LoRA spec for the hypernetwork."""
    base = getattr(target_model, "model", target_model)
    layers = _find_decoder_layers(base)
    num_layers = len(layers)

    in_dims: dict[str, int] = {}
    out_dims: dict[str, int] = {}
    sample = layers[0]
    for m in target_modules:
        linear = _find_named_linear(sample, m)
        if linear is None:
            raise ValueError(f"Could not find linear `{m}` in decoder layer.")
        in_dims[m] = linear.in_features
        out_dims[m] = linear.out_features

    return LoRASpec(
        num_layers=num_layers,
        target_modules=list(target_modules),
        rank=rank,
        target_hidden_dims=in_dims,
        target_out_dims=out_dims,
    )


def _find_decoder_layers(module: nn.Module) -> list[nn.Module]:
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
def lora_applied(target_model: nn.Module, spec: LoRASpec, lora_weights: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    """
    Context manager: monkey-patch forward of each target Linear to add B @ A delta.
    Original weights are untouched. Restores on exit.
    """
    base = getattr(target_model, "model", target_model)
    layers = _find_decoder_layers(base)

    handles: list[tuple[nn.Linear, callable]] = []
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
                _patch_linear(linear, A, B, handles)
        yield
    finally:
        for linear, original_forward in handles:
            linear.forward = original_forward  # type: ignore[assignment]


def _patch_linear(linear: nn.Linear, A: torch.Tensor, B: torch.Tensor, handles: list) -> None:
    """Replace linear.forward with x -> x @ W^T + bias + (x @ A^T) @ B^T."""
    original_forward = linear.forward

    def patched(x: torch.Tensor, _orig=original_forward, _A=A, _B=B) -> torch.Tensor:
        out = _orig(x)
        # x: (..., in), A: (rank, in), B: (out, rank)
        delta = (x @ _A.transpose(-1, -2).to(x.dtype)) @ _B.transpose(-1, -2).to(x.dtype)
        return out + delta

    linear.forward = patched  # type: ignore[assignment]
    handles.append((linear, original_forward))
