"""
Step 0: Probe E4B on SimpleQA Verified to identify wrong answers.

Input:  data/raw/simpleqa_verified.jsonl  (download from Kaggle)
Output: data/processed/e4b_wrong_answers.jsonl

Each output row: {"question_id", "query", "gold", "model_answer", "is_wrong": bool}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from _common import load_config, resolve
from src.dataset import load_jsonl, write_jsonl
from src.evaluator import is_correct
from src.target_model import TargetModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--limit", type=int, default=None, help="Probe only first N questions (for smoke test).")
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

    items = load_jsonl(raw_path)
    if args.limit:
        items = items[: args.limit]
    print(f"loaded {len(items)} simpleqa items")

    target = TargetModel(
        cfg["target_model"],
        cfg["lora_target_modules"],
        cfg["lora_rank"],
    ).to("cuda")

    results: list[dict] = []
    wrong = 0
    for item in tqdm(items, desc="probe E4B"):
        query = item.get("question") or item.get("query")
        gold = item.get("answer") or item.get("gold")
        qid = item.get("id") or item.get("question_id") or item.get("uid")

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

    print(f"wrong answers: {wrong}/{len(items)} ({100*wrong/max(1,len(items)):.1f}%)")
    write_jsonl(out_path, [r for r in results if r["is_wrong"]])
    print(f"wrote wrong-only file -> {out_path}")
    write_jsonl(out_path.with_suffix(".all.jsonl"), results)


if __name__ == "__main__":
    main()
