"""Gemma 4 E4B wrapper. Frozen target acting as both teacher (with feedback) and student (with LoRA)."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lora_utils import LoRASpec, build_lora_spec, lora_applied


class TargetModel:
    def __init__(self, model_name: str, lora_target_modules: list[str], lora_rank: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model: nn.Module = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.spec: LoRASpec = build_lora_spec(self.model, lora_target_modules, lora_rank)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to(self, device: str | torch.device) -> "TargetModel":
        self.model.to(device)
        return self

    # ---------- prompt formatting ----------

    def _format_query(self, query: str) -> str:
        messages = [
            {"role": "system", "content": "Answer in as few words as possible. If you don't know, say 'I don't know'."},
            {"role": "user", "content": query},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _format_query_with_feedback(self, query: str, feedback: str) -> str:
        messages = [
            {"role": "system", "content": "Answer in as few words as possible. If you don't know, say 'I don't know'."},
            {"role": "user", "content": f"{feedback}\n\n{query}"},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ---------- forward (returns logits) ----------

    def teacher_forward(self, query: str, feedback: str) -> torch.Tensor:
        prompt = self._format_query_with_feedback(query, feedback)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        return out.logits

    def student_forward(
        self,
        query: str,
        lora_weights: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        prompt = self._format_query(query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with lora_applied(self.model, self.spec, lora_weights):
            out = self.model(**inputs)
        return out.logits

    # ---------- text generation ----------

    @torch.no_grad()
    def generate(
        self,
        query: str,
        lora_weights: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        prompt = self._format_query(query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if lora_weights is None:
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        else:
            with lora_applied(self.model, self.spec, lora_weights):
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_with_context(self, query: str, feedback: str, max_new_tokens: int = 128) -> str:
        prompt = self._format_query_with_feedback(query, feedback)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
