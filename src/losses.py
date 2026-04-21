"""Sequence-level KL distillation: student teacher-forced on teacher's answer tokens."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_sequence_loss(
    student_logits: torch.Tensor,   # (A, V)
    teacher_logits: torch.Tensor,   # (A, V)
    temperature: float = 1.0,
) -> torch.Tensor:
    """Position-wise KL(teacher || student) averaged over answer tokens.

    Student is teacher-forced on the teacher's greedy answer sequence, so at
    every position it sees ground-truth context and only has to match the
    teacher's next-token distribution.
    """
    if student_logits.numel() == 0 or teacher_logits.numel() == 0:
        return student_logits.new_zeros(())

    s = student_logits / temperature
    t = teacher_logits / temperature

    log_s = F.log_softmax(s, dim=-1)
    p_t = F.softmax(t, dim=-1)

    # F.kl_div with batchmean averages over the "batch" dim (here = positions).
    return F.kl_div(log_s, p_t, reduction="batchmean") * (temperature ** 2)


# Kept for backward-compat with any straggling imports.
def kl_distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    s_last = student_logits[:, -1, :]
    t_last = teacher_logits[:, -1, :]
    return kl_sequence_loss(s_last, t_last, temperature)
