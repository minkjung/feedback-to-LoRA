"""Perceiver-style hypernetwork (random init) for ablation against pretrained backbone."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from .lora_utils import LoRASpec


class CrossAttentionBlock(nn.Module):
    def __init__(self, latent_dim: int, kv_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            kdim=kv_dim,
            vdim=kv_dim,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, latent_dim),
        )

    def forward(self, latents: torch.Tensor, kv: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(latents)
        k = self.norm_kv(kv)
        # key_padding_mask: True == ignore
        attn_out, _ = self.attn(q, k, k, key_padding_mask=~kv_mask.bool())
        latents = latents + attn_out
        latents = latents + self.ff(self.norm_ff(latents))
        return latents


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(x)
        a, _ = self.attn(h, h, h)
        x = x + a
        x = x + self.ff(self.norm_ff(x))
        return x


class PerceiverToLoRA(nn.Module):
    """
    Doc-to-LoRA style hypernetwork with random init.
    Uses frozen target encoder for token features, then learnable Perceiver on top.
    Same forward I/O as FeedbackToLoRA.
    """

    def __init__(
        self,
        token_encoder_name: str,
        spec: LoRASpec,
        num_latents: int = 128,
        latent_dim: int = 768,
        num_self_attn_blocks: int = 8,
        num_heads: int = 8,
        ff_dim: int = 2048,
        projection_hidden: int = 512,
        output_scale: float = 0.01,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.spec = spec
        self.output_scale = output_scale

        # Frozen token encoder for feedback hidden states
        self.token_encoder = AutoModel.from_pretrained(token_encoder_name, torch_dtype=dtype)
        for p in self.token_encoder.parameters():
            p.requires_grad = False
        _cfg = self.token_encoder.config
        kv_dim = getattr(_cfg, "hidden_size", None) or _cfg.text_config.hidden_size

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        self.cross_attn = CrossAttentionBlock(latent_dim, kv_dim, num_heads, ff_dim)
        self.self_attn = nn.ModuleList(
            [SelfAttentionBlock(latent_dim, num_heads, ff_dim) for _ in range(num_self_attn_blocks)]
        )

        self.projections = nn.ModuleDict()
        for i in range(spec.num_layers):
            for m in spec.target_modules:
                in_dim = spec.target_hidden_dims[m]
                out_dim = spec.target_out_dims[m]
                proj_out = spec.rank * in_dim + spec.rank * out_dim
                self.projections[f"layer_{i}_{m}"] = nn.Sequential(
                    nn.Linear(latent_dim, projection_hidden),
                    nn.GELU(),
                    nn.Linear(projection_hidden, proj_out),
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        with torch.no_grad():
            kv = self.token_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state  # (B, T, kv_dim)

        B = input_ids.size(0)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1).to(kv.dtype)

        latents = self.cross_attn(latents, kv, attention_mask)
        for block in self.self_attn:
            latents = block(latents)

        # mean-pool latents -> single representation per sample
        rep = latents.mean(dim=1)
        rep = rep[0] if rep.dim() == 2 else rep

        lora_weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        rank = self.spec.rank
        for key, proj in self.projections.items():
            parts = key.split("_", 2)
            module_name = parts[2]
            in_dim = self.spec.target_hidden_dims[module_name]
            out_dim = self.spec.target_out_dims[module_name]

            ab = proj(rep) * self.output_scale
            half = rank * in_dim
            A_flat, B_flat = ab[:half], ab[half:]
            A = A_flat.view(rank, in_dim)
            B = B_flat.view(out_dim, rank)
            lora_weights[key] = (A, B)

        return lora_weights
