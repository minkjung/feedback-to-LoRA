"""
Step 3: Evaluate.
- Correction accuracy
- Generalization rate
- ICL baseline
- Doc-to-LoRA baseline (Perceiver checkpoint, optional)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from _common import load_config, resolve
from src.dataset import FeedbackDataset
from src.evaluator import Evaluator
from src.hypernetwork import FeedbackToLoRA
from src.perceiver import PerceiverToLoRA
from src.target_model import TargetModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", default=None, help="Path to ours/best.pt (defaults to checkpoint_dir/best.pt)")
    p.add_argument("--d2l-checkpoint", default=None, help="Path to Perceiver baseline checkpoint")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    target = TargetModel(
        cfg["target_model"],
        cfg["lora_target_modules"],
        cfg["lora_rank"],
    ).to(args.device)

    hn_tokenizer = AutoTokenizer.from_pretrained(cfg["hypernetwork_backbone"])
    if hn_tokenizer.pad_token_id is None:
        hn_tokenizer.pad_token = hn_tokenizer.eos_token

    hypernetwork = FeedbackToLoRA(
        backbone_name=cfg["hypernetwork_backbone"],
        spec=target.spec,
        output_scale=cfg["lora_output_scale"],
        dtype=torch.bfloat16,
    )
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (
        resolve(cfg["paths"]["checkpoint_dir"]) / "best.pt"
    )
    state = torch.load(ckpt_path, map_location="cpu")
    hypernetwork.load_state_dict(state["model"])
    hypernetwork = hypernetwork.to(args.device)

    d2l = None
    if args.d2l_checkpoint:
        d2l = PerceiverToLoRA(
            token_encoder_name=cfg["target_model"],
            spec=target.spec,
            num_latents=cfg["perceiver"]["num_latents"],
            latent_dim=cfg["perceiver"]["latent_dim"],
            num_self_attn_blocks=cfg["perceiver"]["num_self_attn_blocks"],
            num_heads=cfg["perceiver"]["num_heads"],
            ff_dim=cfg["perceiver"]["ff_dim"],
            output_scale=cfg["lora_output_scale"],
            dtype=torch.bfloat16,
        )
        d2l_state = torch.load(args.d2l_checkpoint, map_location="cpu")
        d2l.load_state_dict(d2l_state["model"])

    test_ds = FeedbackDataset(
        resolve(cfg["paths"]["test_split"]),
        hn_tokenizer,
        max_feedback_length=cfg["max_feedback_length"],
        num_related_queries=cfg["num_related_queries"],
    )

    evaluator = Evaluator(
        hypernetwork=hypernetwork,
        target=target,
        device=args.device,
        d2l_hypernetwork=d2l,
    )
    results = evaluator.evaluate(test_ds)

    out_dir = resolve(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "correction_accuracy": results.correction_accuracy,
        "generalization_rate": results.generalization_rate,
        "icl_accuracy": results.icl_accuracy,
        "d2l_accuracy": results.d2l_accuracy,
        "n_test": len(results.per_sample),
    }
    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "eval_per_sample.jsonl", "w", encoding="utf-8") as f:
        for r in results.per_sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
