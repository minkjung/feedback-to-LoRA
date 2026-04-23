"""Gemma 4 E2B backbone + per-(layer, module) projection heads -> LoRA weights.

Architecture mirrors Sakana's Doc-to-LoRA (2026) gating trick:
  A = bias_A + scaler_A * A_gen
  B = bias_B + scaler_B * B_gen
  delta = alpha * (B @ A)

- bias_A: small random init  -> prior LoRA even before any feedback signal
- bias_B: zero init            -> delta starts at 0 (base model untouched)
- scaler_A: init 1             -> A active immediately
- scaler_B: init 0             -> B gated off; only this tiny param has to move
                                 for B_gen to start contributing
- A_gen, B_gen: heads with Kaiming init (NOT zero) -> gradient flows freely
- alpha: per-(layer, module) learnable, init 1

This shape is critical: with a zero-init B head (our previous attempt), the
entire gradient has to push a fat Linear out of 0 -- it can't. Here the only
tiny parameter (scaler_B) that needs to move is 1-D, so gradient signal lands
cleanly on it.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from .lora_utils import LoRASpec


class FeedbackToLoRA(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        spec: LoRASpec,
        projection_hidden: int = 512,
        output_scale: float = 1.0,  # placeholder; per-(layer, module) alpha is the real scale
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.spec = spec
        self.output_scale = output_scale

        full = AutoModelForCausalLM.from_pretrained(backbone_name, dtype=dtype)
        base = full.model
        self.backbone = getattr(base, "language_model", base)
        for p in self.backbone.parameters():
            p.requires_grad = False
        cfg = full.config
        text_cfg = getattr(cfg, "text_config", cfg)
        backbone_hidden = text_cfg.hidden_size

        self.proj_A = nn.ModuleDict()
        self.proj_B = nn.ModuleDict()
        self.bias_A = nn.ParameterDict()
        self.bias_B = nn.ParameterDict()
        self.scaler_A = nn.ParameterDict()
        self.scaler_B = nn.ParameterDict()
        self.alpha = nn.ParameterDict()

        rank = spec.rank
        for key in spec.layer_module_keys():
            in_dim = spec.per_layer_in[key]
            out_dim = spec.per_layer_out[key]
            safe = _safe_key(key)

            # Heads produce A_gen (rank, in_dim) / B_gen (out_dim, rank) — standard init.
            self.proj_A[safe] = nn.Sequential(
                nn.Linear(backbone_hidden, projection_hidden),
                nn.GELU(),
                nn.Linear(projection_hidden, rank * in_dim),
            ).to(dtype)
            self.proj_B[safe] = nn.Sequential(
                nn.Linear(backbone_hidden, projection_hidden),
                nn.GELU(),
                nn.Linear(projection_hidden, rank * out_dim),
            ).to(dtype)

            # Static bias per (layer, module). Shape: A=(rank,in_dim), B=(out_dim,rank).
            # bias_A: small random so "initial LoRA" exists even with scaler_A=0.
            std = 0.2 / (in_dim * rank) ** 0.5
            self.bias_A[safe] = nn.Parameter(
                torch.normal(0.0, std, size=(rank, in_dim), dtype=dtype)
            )
            self.bias_B[safe] = nn.Parameter(torch.zeros(out_dim, rank, dtype=dtype))

            # Learnable gates — fp32 for stable AdamW updates.
            self.scaler_A[safe] = nn.Parameter(torch.ones((), dtype=torch.float32))
            self.scaler_B[safe] = nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.alpha[safe] = nn.Parameter(torch.ones((), dtype=torch.float32))

        self._last_generated: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        representation = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        representation = representation / representation.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        rep = representation[0] if representation.dim() == 2 else representation

        rank = self.spec.rank
        lora_weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for key in self.spec.layer_module_keys():
            in_dim = self.spec.per_layer_in[key]
            out_dim = self.spec.per_layer_out[key]
            safe = _safe_key(key)

            a_gen = self.proj_A[safe](rep).view(rank, in_dim)
            b_gen = self.proj_B[safe](rep).view(out_dim, rank)

            sA = self.scaler_A[safe].to(a_gen.dtype)
            sB = self.scaler_B[safe].to(b_gen.dtype)
            a = self.bias_A[safe] + sA * a_gen
            b = self.bias_B[safe] + sB * b_gen

            alpha = self.alpha[safe].to(a.dtype)
            lora_weights[key] = (alpha * a, alpha * b)

        self._last_generated = lora_weights
        return lora_weights

    def generated_l1_norm(self) -> torch.Tensor:
        """Mean L1 on generated (A,B). Keep very small so it doesn't choke scaler_B growth."""
        if not self._last_generated:
            return torch.zeros((), device=next(self.parameters()).device)
        total = None
        n = 0
        for A, B in self._last_generated.values():
            l1 = A.abs().mean() + B.abs().mean()
            total = l1 if total is None else total + l1
            n += 1
        return total / max(1, n)


def _safe_key(key: str) -> str:
    return key.replace(".", "_")
