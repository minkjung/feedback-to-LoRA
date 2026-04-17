"""
Step 4: Ablation — train Perceiver-from-scratch hypernetwork (Doc-to-LoRA style)
with the SAME data and config as step 2, then evaluate via step 3 metrics.

Output: outputs/checkpoints/perceiver_best.pt + outputs/results/ablation_*.json
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
from src.perceiver import PerceiverToLoRA
from src.target_model import TargetModel
from src.trainer import TrainConfig, Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    target = TargetModel(
        cfg["target_model"],
        cfg["lora_target_modules"],
        cfg["lora_rank"],
    ).to(args.device)

    # tokenizer for the perceiver token encoder = target tokenizer
    hn_tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"])
    if hn_tokenizer.pad_token_id is None:
        hn_tokenizer.pad_token = hn_tokenizer.eos_token

    perceiver = PerceiverToLoRA(
        token_encoder_name=cfg["target_model"],
        spec=target.spec,
        num_latents=cfg["perceiver"]["num_latents"],
        latent_dim=cfg["perceiver"]["latent_dim"],
        num_self_attn_blocks=cfg["perceiver"]["num_self_attn_blocks"],
        num_heads=cfg["perceiver"]["num_heads"],
        ff_dim=cfg["perceiver"]["ff_dim"],
        output_scale=cfg["lora_output_scale"],
        dtype=torch.bfloat16,
    ).to(args.device)

    train_ds = FeedbackDataset(
        resolve(cfg["paths"]["train_split"]),
        hn_tokenizer,
        max_feedback_length=cfg["max_feedback_length"],
        num_related_queries=cfg["num_related_queries"],
    )
    val_ds = FeedbackDataset(
        resolve(cfg["paths"]["val_split"]),
        hn_tokenizer,
        max_feedback_length=cfg["max_feedback_length"],
        num_related_queries=cfg["num_related_queries"],
    )

    ckpt_dir = resolve(cfg["paths"]["checkpoint_dir"]) / "perceiver"
    train_cfg = TrainConfig(
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        max_epochs=cfg["max_epochs"],
        early_stopping_patience=cfg["early_stopping_patience"],
        warmup_ratio=cfg["warmup_ratio"],
        max_grad_norm=cfg["max_grad_norm"],
        eval_every_steps=cfg["eval_every_steps"],
        checkpoint_dir=str(ckpt_dir),
    )

    if not args.skip_train:
        trainer = Trainer(
            hypernetwork=perceiver,
            target=target,
            train_ds=train_ds,
            val_ds=val_ds,
            config=train_cfg,
            device=args.device,
        )
        trainer.fit()
        print(f"perceiver best val: {trainer.best_val:.4f}")

    if not args.skip_eval:
        ckpt_path = ckpt_dir / "best.pt"
        state = torch.load(ckpt_path, map_location="cpu")
        perceiver.load_state_dict(state["model"])

        test_ds = FeedbackDataset(
            resolve(cfg["paths"]["test_split"]),
            hn_tokenizer,
            max_feedback_length=cfg["max_feedback_length"],
            num_related_queries=cfg["num_related_queries"],
        )
        evaluator = Evaluator(hypernetwork=perceiver, target=target, device=args.device)
        results = evaluator.evaluate(test_ds)

        out_dir = resolve(cfg["paths"]["results_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "perceiver_correction_accuracy": results.correction_accuracy,
            "perceiver_generalization_rate": results.generalization_rate,
            "icl_accuracy": results.icl_accuracy,
            "n_test": len(results.per_sample),
        }
        with open(out_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
