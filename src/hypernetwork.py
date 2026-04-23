"""Gemma 4 E2B backbone + per-(layer, module) projection heads -> LoRA weights.

Key stability tricks (adopted from Doc-to-LoRA, Sakana 2026):
  - Separate projection heads for A and B matrices
  - B head initialized to zero -> LoRA delta starts at 0 (no base-model disruption)
  - Per-layer learnable scaling alpha (replaces fixed output_scale)
  - L2-normalize pooled representation before projection
  - generated_l1_norm() exposes A/B magnitudes for L1 regularization in the loss
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
        output_scale: float = 1.0,  # kept for compat; effective scale is per-layer alpha
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

        # Separate A and B heads per (layer, module) so we can zero-init B only.
        self.proj_A = nn.ModuleDict()
        self.proj_B = nn.ModuleDict()
        # Per-layer learnable alpha (replaces fixed output_scale). Init small.
        self.alpha = nn.ParameterDict()

        for key in spec.layer_module_keys():
            in_dim = spec.per_layer_in[key]
            out_dim = spec.per_layer_out[key]

            a_head = nn.Sequential(
                nn.Linear(backbone_hidden, projection_hidden),
                nn.GELU(),
                nn.Linear(projection_hidden, spec.rank * in_dim),
            ).to(dtype)
            b_head = nn.Sequential(
                nn.Linear(backbone_hidden, projection_hidden),
                nn.GELU(),
                nn.Linear(projection_hidden, spec.rank * out_dim),
            ).to(dtype)

            # Zero-init the B head's final linear so LoRA delta starts at 0.
            nn.init.zeros_(b_head[-1].weight)
            nn.init.zeros_(b_head[-1].bias)

            self.proj_A[_safe_key(key)] = a_head
            self.proj_B[_safe_key(key)] = b_head
            # alpha init: small positive so gradient can grow it. Fp32 for stable updates.
            self.alpha[_safe_key(key)] = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

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
        hidden = outputs.last_hidden_state  # (B, T, D)

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        representation = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # L2-normalize the pooled representation before projection heads (stabilizes head output).
        representation = representation / representation.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        rep = representation[0] if representation.dim() == 2 else representation

        lora_weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        rank = self.spec.rank
        for key in self.spec.layer_module_keys():
            in_dim = self.spec.per_layer_in[key]
            out_dim = self.spec.per_layer_out[key]

            safe = _safe_key(key)
            a_flat = self.proj_A[safe](rep)
            b_flat = self.proj_B[safe](rep)

            alpha = self.alpha[safe].to(a_flat.dtype)
            A = a_flat.view(rank, in_dim) * alpha
            B = b_flat.view(out_dim, rank) * alpha

            lora_weights[key] = (A, B)

        self._last_generated = lora_weights
        return lora_weights

    def generated_l1_norm(self) -> torch.Tensor:
        """Mean-per-module L1 norm of the last generated (A, B). Used as regularizer."""
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
    # nn.ModuleDict / ParameterDict keys can't contain "."; our keys already don't,
    # but keep a hook in case future keys include them.
    return key.replace(".", "_")
