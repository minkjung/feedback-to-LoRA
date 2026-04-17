"""KL divergence between student (LoRA-applied) and teacher (feedback-in-context)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    KL(teacher || student) is the standard distillation direction;
    we use F.kl_div which expects log-probs as input and probs as target.

    Logits may have different sequence lengths (student prompt is shorter, since it
    excludes the feedback context). Compare only the final-token next-token distribution,
    which is what the model would emit as its first answer token.
    """
    s_last = student_logits[:, -1, :] / temperature
    t_last = teacher_logits[:, -1, :] / temperature

    log_s = F.log_softmax(s_last, dim=-1)
    p_t = F.softmax(t_last, dim=-1)

    return F.kl_div(log_s, p_t, reduction="batchmean") * (temperature ** 2)
