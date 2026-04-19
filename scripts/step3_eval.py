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

HF_CKPT_REPO = "james-kernel/feedback-to-lora-checkpoints"
HF_RESULTS_REPO = "james-kernel/feedback-to-lora-results"
HF_DATA_REPO = "james-kernel/feedback-to-lora-step1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", default=None,
                   help="Path to ours/best.pt. If omitted, downloads best_light.pt from HF.")
    p.add_argument("--d2l-checkpoint", default=None, help="Path to Perceiver baseline checkpoint")
    p.add_argument("--device", default="cuda")
    p.add_argument("--exp-id", default=None,
                   help="experiment id, used to tag results on HF")
    p.add_argument("--no-upload", action="store_true",
                   help="skip uploading results to HF hub")
    return p.parse_args()


def infer_projection_hidden(state: dict) -> int:
    """Projection head shape is (projection_hidden, backbone_hidden). Read first dim."""
    for k, v in state["model"].items():
        if k.endswith(".0.weight") and k.startswith("projections."):
            return v.shape[0]
    raise ValueError("could not infer projection_hidden from checkpoint")


def ensure_checkpoint(local_path: Path | None) -> Path:
    if local_path and local_path.exists():
        return local_path
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HF_CKPT_REPO, filename="best_light.pt", repo_type="model")
    print(f"[ckpt] downloaded best_light.pt from {HF_CKPT_REPO} -> {path}")
    return Path(path)


def ensure_test_split(cfg: dict) -> None:
    dest = resolve(cfg["paths"]["test_split"])
    if dest.exists():
        return
    from huggingface_hub import hf_hub_download
    import shutil
    dest.parent.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(HF_DATA_REPO, filename="test.jsonl", repo_type="dataset")
    shutil.copy(path, dest)
    print(f"[data] downloaded test.jsonl -> {dest}")


def upload_results(out_dir: Path, exp_id: str | None) -> None:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(HF_RESULTS_REPO, repo_type="dataset", exist_ok=True)
        prefix = f"{exp_id}/" if exp_id else ""
        for fname in ("eval_summary.json", "eval_per_sample.jsonl"):
            fp = out_dir / fname
            if not fp.exists():
                continue
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=f"{prefix}{fname}",
                repo_id=HF_RESULTS_REPO,
                repo_type="dataset",
            )
            print(f"[hub] uploaded {fname} -> {HF_RESULTS_REPO}/{prefix}{fname}")
    except Exception as e:
        print(f"[hub] upload failed (non-fatal): {e}")


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

    ckpt_path = ensure_checkpoint(Path(args.checkpoint) if args.checkpoint else None)
    state = torch.load(ckpt_path, map_location="cpu")
    projection_hidden = infer_projection_hidden(state)
    print(f"[ckpt] projection_hidden={projection_hidden}")

    hypernetwork = FeedbackToLoRA(
        backbone_name=cfg["hypernetwork_backbone"],
        spec=target.spec,
        projection_hidden=projection_hidden,
        output_scale=cfg["lora_output_scale"],
        dtype=torch.bfloat16,
    )
    # strict=False because backbone.* keys are absent (frozen, reloaded from HF)
    hypernetwork.load_state_dict(state["model"], strict=False)
    hypernetwork = hypernetwork.to(args.device)

    ensure_test_split(cfg)

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

    if not args.no_upload:
        upload_results(out_dir, args.exp_id)


if __name__ == "__main__":
    main()
