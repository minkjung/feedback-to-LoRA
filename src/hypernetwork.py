"""Gemma 4 E2B backbone + per-(layer, module) MLP projection head -> LoRA weights."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from .lora_utils import LoRASpec


class FeedbackToLoRA(nn.Module):
    """
    Gemma 4 E2B backbone + MLP projection head.
    feedback text -> LoRA (A, B) per (layer, module).
    """

    def __init__(
        self,
        backbone_name: str,
        spec: LoRASpec,
        projection_hidden: int = 512,
        output_scale: float = 0.01,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.spec = spec
        self.output_scale = output_scale

        self.backbone = AutoModel.from_pretrained(backbone_name, dtype=dtype)
        cfg = self.backbone.config
        backbone_hidden = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size

        self.projections = nn.ModuleDict()
        for i in range(spec.num_layers):
            for m in spec.target_modules:
                in_dim = spec.target_hidden_dims[m]
                out_dim = spec.target_out_dims[m]
                proj_out = spec.rank * in_dim + spec.rank * out_dim  # A: (rank, in), B: (out, rank)
                self.projections[f"layer_{i}_{m}"] = nn.Sequential(
                    nn.Linear(backbone_hidden, projection_hidden),
                    nn.GELU(),
                    nn.Linear(projection_hidden, proj_out),
                ).to(dtype)

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

        # batch dim is 1 during inference / per-sample LoRA generation
        rep = representation[0] if representation.dim() == 2 else representation

        lora_weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        rank = self.spec.rank
        for key, proj in self.projections.items():
            # keys are "layer_{i}_{module}", and module may itself contain '_' (e.g. q_proj)
            parts = key.split("_", 2)
            module_name = parts[2]

            ab = proj(rep) * self.output_scale
            in_dim = self.spec.target_hidden_dims[module_name]
            out_dim = self.spec.target_out_dims[module_name]
            half = rank * in_dim
            A_flat, B_flat = ab[:half], ab[half:]
            A = A_flat.view(rank, in_dim)
            B = B_flat.view(out_dim, rank)
            lora_weights[key] = (A, B)

        return lora_weights
