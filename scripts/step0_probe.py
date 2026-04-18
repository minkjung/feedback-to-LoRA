"""
Step 0: Probe E4B on SimpleQA Verified to identify wrong answers.

Input:  data/raw/simpleqa_verified.jsonl  (download from Kaggle)
Output: data/processed/e4b_wrong_answers.jsonl

Each output row: {"question_id", "query", "gold", "model_answer", "is_wrong": bool}
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from _common import load_config, resolve
from src.dataset import load_jsonl, write_jsonl
from src.evaluator import is_correct
from src.target_model import TargetModel

HF_CHECKPOINT_REPO = "minkjung/feedback-to-lora-step0-checkpoint"
PUSH_EVERY = 50


def push_to_hub(results: list[dict], repo_id: str) -> None:
    try:
        from huggingface_hub import HfApi
        import tempfile
        api = HfApi()
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp_path = f.name
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="checkpoint.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
        )
        os.unlink(tmp_path)
        print(f"[hub] pushed {len(results)} rows -> {repo_id}")
    except Exception as e:
        print(f"[hub] push failed (non-fatal): {e}")


def load_from_hub(repo_id: str) -> list[dict]:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename="checkpoint.jsonl", repo_type="dataset")
        return load_jsonl(Path(path))
    except Exception:
        return []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--limit", type=int, default=None, help="Probe only first N questions (for smoke test).")
    p.add_argument("--no-hub", action="store_true", help="Disable HuggingFace Hub checkpointing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    raw_path = resolve(cfg["paths"]["raw_simpleqa"])
    out_path = resolve(cfg["paths"]["e4b_wrong_answers"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"SimpleQA Verified not found at {raw_path}. "
            "Download from https://www.kaggle.com/benchmarks/deepmind/simpleqa-verified"
        )

    use_hub = not args.no_hub and bool(os.environ.get("HF_TOKEN"))

    items = load_jsonl(raw_path)
    if args.limit:
        items = items[: args.limit]
    print(f"loaded {len(items)} simpleqa items")

    target = TargetModel(
        cfg["target_model"],
        cfg["lora_target_modules"],
        cfg["lora_rank"],
    ).to("cuda")

    # Resume from hub or local checkpoint
    checkpoint_path = out_path.with_suffix(".checkpoint.jsonl")
    results: list[dict] = []
    if use_hub:
        results = load_from_hub(HF_CHECKPOINT_REPO)
        if results:
            print(f"resuming from hub: {len(results)} done")
    if not results and checkpoint_path.exists():
        results = load_jsonl(checkpoint_path)
        if results:
            print(f"resuming from local checkpoint: {len(results)} done")

    done_ids = {str(r["question_id"]) for r in results}
    wrong = sum(1 for r in results if r["is_wrong"])
    since_last_push = 0

    for item in tqdm(items, desc="probe E4B"):
        query = item.get("question") or item.get("query")
        gold = item.get("answer") or item.get("gold")
        qid = item.get("id") or item.get("question_id") or item.get("uid")

        if str(qid) in done_ids:
            continue

        ans = target.generate(query)
        wrong_flag = not is_correct(ans, gold)
        if wrong_flag:
            wrong += 1
        results.append({
            "question_id": qid,
            "query": query,
            "gold": gold,
            "model_answer": ans,
            "is_wrong": wrong_flag,
        })
        write_jsonl(checkpoint_path, results)
        since_last_push += 1

        if use_hub and since_last_push >= PUSH_EVERY:
            push_to_hub(results, HF_CHECKPOINT_REPO)
            since_last_push = 0

    if use_hub:
        push_to_hub(results, HF_CHECKPOINT_REPO)

    print(f"wrong answers: {wrong}/{len(items)} ({100*wrong/max(1,len(items)):.1f}%)")
    write_jsonl(out_path, [r for r in results if r["is_wrong"]])
    print(f"wrote wrong-only file -> {out_path}")
    write_jsonl(out_path.with_suffix(".all.jsonl"), results)


if __name__ == "__main__":
    main()
