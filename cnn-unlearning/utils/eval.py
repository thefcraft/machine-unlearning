import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def eval_model_accuracy[T](
    model: Module,
    data_loader: DataLoader[T],
    device: torch.device,
) -> float:
    correct = 0
    total = 0

    model.eval()
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x).argmax(1)

        correct += (pred == y).sum()
        total += y.size(0)
    return (correct / total).__float__()


@torch.no_grad()
def eval_avg_confidence[T](
    model: Module,
    data_loader: DataLoader[T],
    device: torch.device,
) -> float:
    model.eval()

    total_confidence: float = 0.0
    total = 0

    for x, _ in data_loader:
        x = x.to(device)

        logits = model(x)

        probs = F.softmax(
            logits,
            dim=1,
        )

        confidence = probs.max(dim=1).values

        total_confidence += confidence.sum()  # type: ignore
        total += confidence.size(0)

    return (total_confidence / total).__float__()


@torch.no_grad()
def eval_model_roc_auc[T](
    model: Module,
    data_loader: DataLoader[T],
    device: torch.device,
) -> float:
    model.eval()

    y_true_list: list[np.ndarray] = []
    y_score_list: list[np.ndarray] = []

    for x, y in data_loader:
        x = x.to(device)

        logits = model(x)
        probs = F.softmax(logits, dim=1)

        y_true_list.append(y.cpu().numpy())
        y_score_list.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_score = np.concatenate(y_score_list)

    present_classes = np.unique(y_true)

    # ROC-AUC undefined for a single class
    if len(present_classes) < 2:
        return float("nan")

    # Keep only probability columns for classes that exist
    y_score_present = y_score[:, present_classes]
    y_score_present /= y_score_present.sum(axis=1, keepdims=True)


    return float(
        roc_auc_score(
            y_true,
            y_score_present,
            labels=present_classes,
            multi_class="ovr",
            average="macro",
        )
    )