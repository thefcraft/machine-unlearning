import torch
import torch.nn.functional as F


def kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
):
    teacher_probs = F.softmax(
        teacher_logits / temperature,
        dim=1,
    )

    student_log_probs = F.log_softmax(
        student_logits / temperature,
        dim=1,
    )

    return F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature**2)
