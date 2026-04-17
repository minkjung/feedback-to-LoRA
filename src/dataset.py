"""Dataset / DataLoader for feedback->LoRA training and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class FeedbackSample:
    query: str
    gold: str
    feedback: str
    related_queries: list[str]


def load_jsonl(path: str | Path) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: str | Path, items: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class FeedbackDataset(Dataset):
    """
    Each sample yields:
        feedback_ids, feedback_mask  (tokenized for hypernetwork)
        feedback (raw str)            (for teacher prompt)
        query (raw str), gold (raw str), related_queries (list[str])
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_feedback_length: int = 256,
        num_related_queries: int = 5,
    ):
        self.items = load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_feedback_length = max_feedback_length
        self.num_related_queries = num_related_queries

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        feedback = item["feedback"]
        enc = self.tokenizer(
            feedback,
            truncation=True,
            max_length=self.max_feedback_length,
            return_tensors="pt",
        )
        related = item.get("related_queries", [])[: self.num_related_queries]
        return {
            "feedback_ids": enc["input_ids"][0],
            "feedback_mask": enc["attention_mask"][0],
            "feedback": feedback,
            "query": item["query"],
            "gold": item["gold"],
            "related_queries": related,
        }


def collate_singletons(batch: list[dict]) -> dict:
    """
    Per-sample LoRA generation -> we keep batch_size logically 1 in the model loop,
    but accumulate over a list. The DataLoader batches into a list of dicts.
    """
    return {
        "samples": batch,
    }


def pad_batch(
    sequences: list[torch.Tensor],
    pad_value: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(s.size(0) for s in sequences)
    padded = torch.full((len(sequences), max_len), pad_value, dtype=sequences[0].dtype)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, s in enumerate(sequences):
        padded[i, : s.size(0)] = s
        mask[i, : s.size(0)] = 1
    return padded, mask
