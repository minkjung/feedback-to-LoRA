"""
Step 2: Train FeedbackToLoRA hypernetwork.

Input:  HuggingFace Hub (james-kernel/feedback-to-lora-step1)
Output: outputs/checkpoints/best.pt  +  HF hub upload
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from _common import load_config, resolve
from src.dataset import FeedbackDataset
from src.hypernetwork import FeedbackToLoRA
from src.target_model import TargetModel
from src.trainer import TrainConfig, Trainer

HF_DATA_REPO = "james-kernel/feedback-to-lora-step1"
HF_CKPT_REPO = "james-kernel/feedback-to-lora-checkpoints"


def download_splits(cfg: dict) -> None:
    from huggingface_hub import hf_hub_download
    for fname, key in [
        ("train.jsonl", "train_split"),
        ("val.jsonl", "val_split"),
        ("test.jsonl", "test_split"),
    ]:
        dest = resolve(cfg["paths"][key])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            path = hf_hub_download(repo_id=HF_DATA_REPO, filename=fname, repo_type="dataset")
            import shutil
            shutil.copy(path, dest)
            print(f"downloaded {fname} -> {dest}")


def upload_checkpoint(ckpt_path: Path) -> None:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(HF_CKPT_REPO, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(ckpt_path),
            path_in_repo=ckpt_path.name,
            repo_id=HF_CKPT_REPO,
            repo_type="model",
        )
        print(f"uploaded checkpoint -> {HF_CKPT_REPO}/{ckpt_path.name}")
    except Exception as e:
        print(f"checkpoint upload failed (non-fatal): {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-dir", default=None, help="override config checkpoint_dir")
    p.add_argument("--stop-instance", default=None,
                   help="vast.ai instance id to stop on exit (success or error)")
    return p.parse_args()


def stop_vastai(instance_id: str) -> None:
    import subprocess
    try:
        subprocess.run(["vastai", "stop", "instance", instance_id], check=False)
        print(f"[autostop] stopped vast.ai instance {instance_id}")
    except Exception as e:
        print(f"[autostop] failed: {e}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.checkpoint_dir:
        cfg["paths"]["checkpoint_dir"] = args.checkpoint_dir

    download_splits(cfg)

    target = TargetModel(
        cfg["target_model"],
        cfg["lora_target_modules"],
        cfg["lora_rank"],
    ).to(args.device)

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

    should_stop = False
    try:
        trainer.fit()
        print(f"done. best val loss: {trainer.best_val:.4f}")
        ckpt_path = Path(train_cfg.checkpoint_dir) / "best.pt"
        upload_checkpoint(ckpt_path)
        should_stop = True  # normal completion
    except KeyboardInterrupt:
        print("\n[autostop] Ctrl+C detected, instance will stay alive")
        should_stop = False
    except Exception:
        import traceback
        traceback.print_exc()
        should_stop = True  # unexpected error: stop to avoid burning money

    if should_stop and args.stop_instance:
        stop_vastai(args.stop_instance)


if __name__ == "__main__":
    main()
