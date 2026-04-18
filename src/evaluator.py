"""Evaluation: Correction Accuracy + Generalization Rate, with ICL and Doc-to-LoRA baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from tqdm import tqdm

from .dataset import FeedbackDataset
from .target_model import TargetModel


def normalize(text: str) -> str:
    import re
    text = text.lower().strip()
    text = re.sub(r",(?=\d)", "", text)  # remove thousands separators: 120,000 -> 120000
    text = re.sub(r"\s+", " ", text)
    return text


def is_correct(answer: str, gold: str) -> bool:
    """Substring match after normalization. LLM judge can be plugged in later."""
    a = normalize(answer)
    g = normalize(gold)
    if not g:
        return False
    return g in a


@dataclass
class EvalResults:
    correction_accuracy: float
    generalization_rate: float
    icl_accuracy: float
    d2l_accuracy: float | None
    per_sample: list[dict]


class Evaluator:
    def __init__(
        self,
        hypernetwork: torch.nn.Module,
        target: TargetModel,
        device: str = "cuda",
        d2l_hypernetwork: torch.nn.Module | None = None,
    ):
        self.hypernetwork = hypernetwork.to(device).eval()
        self.target = target.to(device)
        self.device = device
        self.d2l_hypernetwork = (
            d2l_hypernetwork.to(device).eval() if d2l_hypernetwork is not None else None
        )

    @torch.no_grad()
    def _gen_lora(self, hn: torch.nn.Module, sample: dict):
        ids = sample["feedback_ids"].unsqueeze(0).to(self.device)
        mask = sample["feedback_mask"].unsqueeze(0).to(self.device)
        return hn(ids, mask)

    @torch.no_grad()
    def evaluate(self, dataset: FeedbackDataset) -> EvalResults:
        per_sample = []
        correction, generalization, icl, d2l = [], [], [], []

        for sample in tqdm(dataset, desc="eval"):
            lora = self._gen_lora(self.hypernetwork, sample)

            # 1. correction
            ans = self.target.generate(sample["query"], lora)
            c = is_correct(ans, sample["gold"])
            correction.append(c)

            # 2. generalization
            gen_scores = []
            for rq in sample["related_queries"]:
                ra = self.target.generate(rq, lora)
                gen_scores.append(is_correct(ra, sample["gold"]))
            gen_rate = (sum(gen_scores) / len(gen_scores)) if gen_scores else 0.0
            generalization.append(gen_rate)

            # 3. ICL upper bound
            icl_ans = self.target.generate_with_context(sample["query"], sample["feedback"])
            icl.append(is_correct(icl_ans, sample["gold"]))

            # 4. Doc-to-LoRA baseline
            d2l_correct = None
            if self.d2l_hypernetwork is not None:
                d2l_lora = self._gen_lora(self.d2l_hypernetwork, sample)
                d2l_ans = self.target.generate(sample["query"], d2l_lora)
                d2l_correct = is_correct(d2l_ans, sample["gold"])
                d2l.append(d2l_correct)

            per_sample.append({
                "query": sample["query"],
                "gold": sample["gold"],
                "feedback": sample["feedback"],
                "answer_ours": ans,
                "answer_icl": icl_ans,
                "correction": c,
                "generalization": gen_rate,
                "icl": icl[-1],
                "d2l": d2l_correct,
            })

        return EvalResults(
            correction_accuracy=_mean(correction),
            generalization_rate=_mean(generalization),
            icl_accuracy=_mean(icl),
            d2l_accuracy=_mean(d2l) if d2l else None,
            per_sample=per_sample,
        )


def _mean(values: Iterable) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))
