"""
Step 2: Train FeedbackToLoRA hypernetwork.

Input:  data/splits/{train,val}.jsonl
Output: outputs/checkpoints/best.pt
"""

from __future__ import annotations

import argparse

import torch

from _common import load_config, resolve
from src.dataset import FeedbackDataset
from src.hypernetwork import FeedbackToLoRA
from src.target_model import TargetModel
from src.trainer import TrainConfig, Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Target model (frozen, also provides tokenizer for the target)
    target = TargetModel(
        cfg["target_model"],
        cfg["lora_target_modules"],
        cfg["lora_rank"],
    ).to(args.device)

    # Hypernetwork
    from transformers import AutoTokenizer
    hn_tokenizer = AutoTokenizer.from_pretrained(cfg["hypernetwork_backbone"])
    if hn_tokenizer.pad_token_id is None:
        hn_tokenizer.pad_token = hn_tokenizer.eos_token

    hypernetwork = FeedbackToLoRA(
        backbone_name=cfg["hypernetwork_backbone"],
        spec=target.spec,
        output_scale=cfg["lora_output_scale"],
        dtype=torch.bfloat16,
    ).to(args.device)

    # Datasets
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

    train_cfg = TrainConfig(
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        max_epochs=cfg["max_epochs"],
        early_stopping_patience=cfg["early_stopping_patience"],
        warmup_ratio=cfg["warmup_ratio"],
        max_grad_norm=cfg["max_grad_norm"],
        eval_every_steps=cfg["eval_every_steps"],
        checkpoint_dir=str(resolve(cfg["paths"]["checkpoint_dir"])),
    )

    trainer = Trainer(
        hypernetwork=hypernetwork,
        target=target,
        train_ds=train_ds,
        val_ds=val_ds,
        config=train_cfg,
        device=args.device,
    )
    history = trainer.fit()
    print(f"done. best val loss: {trainer.best_val:.4f}")
    print(f"checkpoint at {train_cfg.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
