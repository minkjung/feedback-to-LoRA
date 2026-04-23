"""Gemma 4 E4B wrapper. Frozen target acting as both teacher (with feedback) and student (with LoRA)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lora_utils import LoRASpec, build_lora_spec, lora_applied, lora_merged


@dataclass
class TeacherTrace:
    """Output of teacher_generate_and_score: the teacher's answer tokens and
    its logits at every answer position (for distillation)."""
    prompt_ids: torch.Tensor          # (1, P)  query-only student prompt ids
    answer_ids: torch.Tensor          # (A,)    teacher-generated answer tokens
    answer_logits: torch.Tensor       # (A, V)  teacher logits at each answer position


class TargetModel:
    def __init__(self, model_name: str, lora_target_modules: list[str], lora_rank: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model: nn.Module = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
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

    # ---------- distillation training ops ----------

    @torch.no_grad()
    def teacher_generate_and_score(
        self,
        query: str,
        feedback: str,
        max_new_tokens: int = 32,
    ) -> TeacherTrace:
        """Teacher sees feedback+query, greedily generates an answer, and returns
        the answer token ids plus teacher logits at every answer position.

        The returned prompt_ids is the *student* prompt (query without feedback),
        used later for teacher-forcing the student."""
        device = self.device

        teacher_prompt = self._format_query_with_feedback(query, feedback)
        teacher_inputs = self.tokenizer(teacher_prompt, return_tensors="pt").to(device)
        out = self.model.generate(
            **teacher_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        # scores: tuple of (1, V) per generated token (post-processed logits)
        # sequences: (1, P_t + A)
        P_t = teacher_inputs["input_ids"].shape[1]
        sequences = out.sequences
        answer_ids = sequences[0, P_t:]                              # (A,)
        # out.scores[i] is logits for answer_ids[i]
        answer_logits = torch.stack(list(out.scores), dim=0).squeeze(1)  # (A, V)

        # Drop trailing EOS/pad tokens so we only distill real answer content.
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id
        keep = []
        for i, tok in enumerate(answer_ids.tolist()):
            if tok in (eos, pad):
                break
            keep.append(i)
        if keep:
            answer_ids = answer_ids[keep]
            answer_logits = answer_logits[keep]

        student_prompt = self._format_query(query)
        student_prompt_ids = self.tokenizer(student_prompt, return_tensors="pt").input_ids.to(device)

        return TeacherTrace(
            prompt_ids=student_prompt_ids,
            answer_ids=answer_ids.to(device),
            answer_logits=answer_logits.to(device),
        )

    def student_score_answer(
        self,
        trace: TeacherTrace,
        lora_weights: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Teacher-force the student on prompt+answer and return student logits
        at every answer position, shape (A, V). Gradients flow through lora_weights."""
        return self.student_score_answer_batch([trace], lora_weights)[0]

    def student_score_answer_batch(
        self,
        traces: list[TeacherTrace],
        lora_weights: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Batched version: one forward pass for all traces. Returns list of (A_i, V) tensors."""
        vocab_size = (
            self.model.config.text_config.vocab_size
            if hasattr(self.model.config, "text_config")
            else self.model.config.vocab_size
        )
        empty = torch.empty(0, vocab_size, device=self.device)

        valid = [(i, t) for i, t in enumerate(traces) if t.answer_ids.numel() > 0]
        if not valid:
            return [empty for _ in traces]

        # Build padded batch: each row = [prompt_ids | answer_ids], left-pad to same length.
        seqs = []
        for _, t in valid:
            seq = torch.cat([t.prompt_ids[0], t.answer_ids], dim=0)  # (P_i + A_i,)
            seqs.append(seq)

        max_len = max(s.shape[0] for s in seqs)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(len(seqs), max_len, dtype=torch.long, device=self.device)
        for row, seq in enumerate(seqs):
            L = seq.shape[0]
            input_ids[row, max_len - L :] = seq
            attention_mask[row, max_len - L :] = 1

        with lora_applied(self.model, self.spec, lora_weights):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # (B, max_len, V)

        results: list[torch.Tensor] = [empty for _ in traces]
        for row, (orig_idx, t) in enumerate(valid):
            P = t.prompt_ids.shape[1]
            A = t.answer_ids.shape[0]
            # Left-padded: answer starts at max_len - A, prompt answer boundary at max_len - A - 1
            start = max_len - A - 1  # position that predicts answer_ids[0]
            results[orig_idx] = logits[row, start : start + A, :]  # (A, V)

        return results

    # ---------- legacy eval APIs ----------

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
        max_new_tokens: int = 32,
    ) -> str:
        prompt = self._format_query(query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if lora_weights is None:
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        else:
            with lora_merged(self.model, self.spec, lora_weights):
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_with_context(self, query: str, feedback: str, max_new_tokens: int = 32) -> str:
        prompt = self._format_query_with_feedback(query, feedback)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
