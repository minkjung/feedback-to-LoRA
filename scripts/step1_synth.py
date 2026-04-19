"""
Step 1: Synthesize feedback dataset.

Input:  HuggingFace Hub (james-kernel/feedback-to-lora-step0-checkpoint)
Output: data/processed/feedback_dataset.jsonl
        data/splits/{train,val,test}.jsonl

For each wrong answer, GPT generates:
  1. correction feedback in K different styles
  2. N related queries that probe the same underlying fact
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from _common import load_config, resolve
from src.dataset import load_jsonl, write_jsonl


FEEDBACK_PROMPT = """You write a short correction message a user might send to a chatbot.

The chatbot answered "{wrong}" but the correct answer is "{gold}". The user's original question was: "{query}".

Write the correction in this style: {style_desc}

Output ONLY the correction message, no explanation, no quotes."""


RELATED_QUERIES_PROMPT = """The fact: question "{query}" -> answer "{gold}".

Generate {n} related but DIFFERENT questions that depend on the same underlying fact.
They must be answerable from the same fact. Vary the angle (paraphrase, sub-question, what/why/when, etc).

Return JSON array of strings, no other text. Example: ["Q1", "Q2", "Q3"]"""


STYLE_DESCRIPTIONS = {
    "direct":          "direct and short, like 'No, the answer is X.'",
    "conversational":  "casual conversational, like 'hmm that's not right, isn't it X?'",
    "terse":           "very terse, just the corrected fact, like 'X.'",
    "partial":         "polite partial correction acknowledging some part may be ok, like 'close, but actually X'",
    "multilingual":    "in English, brief and direct: 'That's wrong, it should be X.'",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=20)
    return p.parse_args()


def call_openai(client, model: str, prompt: str) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "none"},
    )
    return resp.output_text.strip()


def gold_leak_filter(feedback: str, gold: str) -> bool:
    if not feedback.strip():
        return False
    if len(feedback.strip()) < len(gold.strip()) + 2:
        return False
    return True


def process_item(client, model, item, styles, augment_factor, n_related, rng_seed):
    rng = random.Random(rng_seed)
    query = item["query"]
    gold = item["gold"]
    wrong = item["model_answer"]
    rows = []

    try:
        rq_text = call_openai(client, model, RELATED_QUERIES_PROMPT.format(query=query, gold=gold, n=n_related))
        related_queries = json.loads(rq_text)
        if not isinstance(related_queries, list):
            related_queries = []
    except Exception:
        related_queries = []

    chosen_styles = rng.choices(styles, k=augment_factor)
    for style in chosen_styles:
        try:
            fb = call_openai(client, model, FEEDBACK_PROMPT.format(
                wrong=wrong, gold=gold, query=query,
                style_desc=STYLE_DESCRIPTIONS[style],
            ))
        except Exception:
            continue
        if not gold_leak_filter(fb, gold):
            continue
        rows.append({
            "question_id": item.get("question_id"),
            "query": query,
            "gold": gold,
            "wrong": wrong,
            "feedback": fb,
            "feedback_style": style,
            "related_queries": related_queries,
        })

    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("`openai` package is required. pip install openai") from e

    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY environment variable.")

    client = OpenAI()

    out_path = resolve(cfg["paths"]["feedback_dataset"])

    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download(
        repo_id="james-kernel/feedback-to-lora-step0-checkpoint",
        filename="checkpoint.jsonl",
        repo_type="dataset",
    )
    all_items = load_jsonl(Path(ckpt))
    items = [r for r in all_items if r.get("is_wrong")]
    print(f"loaded {len(items)} wrong-answer items from HF hub")

    if args.limit:
        items = items[: args.limit]

    styles = cfg["feedback_styles"]
    augment_factor = cfg["feedback_augment_factor"]
    n_related = cfg["num_related_queries"]
    api_model = cfg["synth_api_model"]
    seed = cfg["seed"]

    rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_item, client, api_model, item, styles, augment_factor, n_related, seed + i): i
            for i, item in enumerate(items)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="synth"):
            try:
                rows.extend(future.result())
            except Exception as e:
                print(f"item failed: {e}")

    write_jsonl(out_path, rows)
    print(f"wrote {len(rows)} feedback rows -> {out_path}")

    by_qid: dict[str, list[dict]] = {}
    for r in rows:
        by_qid.setdefault(str(r["question_id"]), []).append(r)
    qids = list(by_qid.keys())
    rng = random.Random(seed)
    rng.shuffle(qids)

    n = len(qids)
    n_train = int(n * cfg["train_ratio"])
    n_val = int(n * cfg["val_ratio"])
    train_q = set(qids[:n_train])
    val_q = set(qids[n_train:n_train + n_val])
    test_q = set(qids[n_train + n_val:])

    def collect(qset):
        out = []
        for q in qset:
            out.extend(by_qid[q])
        return out

    write_jsonl(resolve(cfg["paths"]["train_split"]), collect(train_q))
    write_jsonl(resolve(cfg["paths"]["val_split"]), collect(val_q))
    write_jsonl(resolve(cfg["paths"]["test_split"]), collect(test_q))
    print(f"splits: train={len(train_q)} val={len(val_q)} test={len(test_q)} (by question_id)")


if __name__ == "__main__":
    main()
