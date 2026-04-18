"""
Step 1: Synthesize feedback dataset.

Input:  data/processed/e4b_wrong_answers.jsonl
Output: data/processed/feedback_dataset.jsonl
        data/splits/{train,val,test}.jsonl

For each wrong answer, GPT-5 Nano produces:
  1. correction feedback in K different styles (direct / conversational / terse / partial / multilingual)
  2. N related queries that probe the same underlying fact

Filter: drop rows where the gold answer leaks verbatim from the feedback being too literal
(we want correction signal, not lookup); also drop empties.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
    return p.parse_args()


def call_openai(client, model: str, prompt: str) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "none"},
    )
    return resp.output_text.strip()


def gold_leak_filter(feedback: str, gold: str) -> bool:
    """Reject if feedback merely echoes gold; we still want gold inside, but not be only gold."""
    if not feedback.strip():
        return False
    if len(feedback.strip()) < len(gold.strip()) + 2:
        return False
    return True


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

    in_path = resolve(cfg["paths"]["e4b_wrong_answers"])
    out_path = resolve(cfg["paths"]["feedback_dataset"])
    items = load_jsonl(in_path)
    if args.limit:
        items = items[: args.limit]
    print(f"loaded {len(items)} wrong-answer items")

    styles = cfg["feedback_styles"]
    augment_factor = cfg["feedback_augment_factor"]
    n_related = cfg["num_related_queries"]
    api_model = cfg["synth_api_model"]

    rng = random.Random(cfg["seed"])
    rows: list[dict] = []

    for item in tqdm(items, desc="synth"):
        query = item["query"]
        gold = item["gold"]
        wrong = item["model_answer"]

        # related queries (one batch per fact)
        try:
            rq_text = call_openai(
                client, api_model,
                RELATED_QUERIES_PROMPT.format(query=query, gold=gold, n=n_related),
            )
            related_queries = json.loads(rq_text)
            if not isinstance(related_queries, list):
                related_queries = []
        except Exception as e:
            print(f"related queries failed: {e}")
            related_queries = []

        # feedback per style, repeated augment_factor times
        chosen_styles = rng.choices(styles, k=augment_factor)
        for style in chosen_styles:
            try:
                fb = call_openai(
                    client, api_model,
                    FEEDBACK_PROMPT.format(
                        wrong=wrong, gold=gold, query=query,
                        style_desc=STYLE_DESCRIPTIONS[style],
                    ),
                )
            except Exception as e:
                print(f"feedback failed: {e}")
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

    write_jsonl(out_path, rows)
    print(f"wrote {len(rows)} feedback rows -> {out_path}")

    # split (group by question_id so train/test don't share facts)
    by_qid: dict[str, list[dict]] = {}
    for r in rows:
        by_qid.setdefault(str(r["question_id"]), []).append(r)
    qids = list(by_qid.keys())
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
