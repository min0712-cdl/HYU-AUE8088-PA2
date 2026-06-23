"""Evaluation metrics for multi-task scene classification.

Primary:    Average Macro-F1 across the 3 attributes.
Secondary:  Average mean-Average-Precision across the 3 attributes.
Mandatory:  Per-attribute Confusion Matrices.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.datasets.bdd_attr import (
    ATTRIBUTES,
    NUM_CLASSES,
    SCENE_CLASSES,
    TIMEOFDAY_CLASSES,
    WEATHER_CLASSES,
)


CLASS_NAMES = {
    "weather": WEATHER_CLASSES,
    "scene": SCENE_CLASSES,
    "timeofday": TIMEOFDAY_CLASSES,
}


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=np.arange(num_classes), zero_division=0))


def per_attribute_macro_f1(
    preds: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> dict[str, float]:
    return {
        a: macro_f1(targets[a], preds[a], NUM_CLASSES[a])
        for a in ATTRIBUTES
    }


def average_macro_f1(
    preds: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> float:
    """Primary metric — Avg over the 3 attributes."""
    per = per_attribute_macro_f1(preds, targets)
    return float(np.mean(list(per.values())))


def per_attribute_mAP(
    probs: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """One-vs-rest mAP per attribute. ``probs[a]`` has shape (N, C_a)."""
    out = {}
    for a in ATTRIBUTES:
        n_classes = NUM_CLASSES[a]
        y_true_oh = np.eye(n_classes)[targets[a]]
        # average='macro' gives mean of per-class AP — i.e. mAP for this attr.
        out[a] = float(
            average_precision_score(y_true_oh, probs[a], average="macro")
        )
    return out


def average_mAP(
    probs: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> float:
    return float(np.mean(list(per_attribute_mAP(probs, targets).values())))


def per_attribute_accuracy(
    preds: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Top-1 accuracy for each attribute."""
    return {
        a: float(np.mean(np.asarray(preds[a]) == np.asarray(targets[a])))
        for a in ATTRIBUTES
    }


def per_class_accuracy(
    preds: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> dict[str, dict]:
    """Per-class recall, reported as class accuracy in the assignment."""
    out = {}
    for a in ATTRIBUTES:
        class_scores = []
        supports = []
        for class_index in range(NUM_CLASSES[a]):
            mask = np.asarray(targets[a]) == class_index
            support = int(mask.sum())
            supports.append(support)
            class_scores.append(
                float(np.mean(np.asarray(preds[a])[mask] == class_index))
                if support > 0 else float("nan")
            )
        out[a] = {
            "class": CLASS_NAMES[a],
            "accuracy": class_scores,
            "support": supports,
        }
    return out


def worst_class_accuracy(
    preds: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> dict[str, dict]:
    """Lowest supported per-class accuracy for each attribute."""
    class_metrics = per_class_accuracy(preds, targets)
    out = {}
    for a in ATTRIBUTES:
        scores = np.asarray(class_metrics[a]["accuracy"], dtype=np.float64)
        valid = ~np.isnan(scores)
        if not valid.any():
            out[a] = {"class": None, "accuracy": float("nan"), "support": 0}
            continue
        valid_indices = np.flatnonzero(valid)
        class_index = int(valid_indices[np.argmin(scores[valid])])
        out[a] = {
            "class": CLASS_NAMES[a][class_index],
            "accuracy": float(scores[class_index]),
            "support": class_metrics[a]["support"][class_index],
        }
    return out


def confusion_matrices(
    preds: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    normalize: str = "true",
) -> dict[str, np.ndarray]:
    """Returns a dict ``{attribute: cm}`` of normalized confusion matrices."""
    out = {}
    for a in ATTRIBUTES:
        out[a] = confusion_matrix(
            targets[a],
            preds[a],
            labels=np.arange(NUM_CLASSES[a]),
            normalize=normalize,
        )
    return out


def per_class_prf(
    preds: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> dict[str, dict]:
    out = {}
    for a in ATTRIBUTES:
        p, r, f, sup = precision_recall_fscore_support(
            targets[a],
            preds[a],
            labels=np.arange(NUM_CLASSES[a]),
            zero_division=0,
        )
        out[a] = {
            "class": CLASS_NAMES[a],
            "precision": p.tolist(),
            "recall": r.tolist(),
            "f1": f.tolist(),
            "support": sup.tolist(),
        }
    return out


def evaluation_report(
    preds: Mapping[str, np.ndarray],
    probs: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> dict:
    """Build every metric required by the assignment report."""
    per_map = per_attribute_mAP(probs, targets)
    return {
        "avg_macro_f1": average_macro_f1(preds, targets),
        "per_macro_f1": per_attribute_macro_f1(preds, targets),
        "avg_map": float(np.mean(list(per_map.values()))),
        "per_map": per_map,
        "top1_accuracy": per_attribute_accuracy(preds, targets),
        "worst_class_accuracy": worst_class_accuracy(preds, targets),
        "per_class_accuracy": per_class_accuracy(preds, targets),
        "per_class_prf": per_class_prf(preds, targets),
        "confusion_matrices": confusion_matrices(preds, targets, normalize="true"),
    }


def print_evaluation_report(report: Mapping, title: str = "Evaluation") -> None:
    """Print a compact notebook-friendly summary and per-class tables."""
    print(f"\n{title}")
    print(f"Avg-MF1={report['avg_macro_f1']:.5f}  Avg-mAP={report['avg_map']:.5f}")
    for a in ATTRIBUTES:
        worst = report["worst_class_accuracy"][a]
        print(
            f"{a}: MF1={report['per_macro_f1'][a]:.5f} "
            f"mAP={report['per_map'][a]:.5f} "
            f"Top-1={report['top1_accuracy'][a]:.5f} "
            f"Worst={worst['class']}:{worst['accuracy']:.5f}"
        )
        prf = report["per_class_prf"][a]
        class_acc = report["per_class_accuracy"][a]
        print("  class | precision | recall | f1 | accuracy | support")
        for index, class_name in enumerate(prf["class"]):
            print(
                f"  {class_name} | {prf['precision'][index]:.5f} | "
                f"{prf['recall'][index]:.5f} | {prf['f1'][index]:.5f} | "
                f"{class_acc['accuracy'][index]:.5f} | {prf['support'][index]}"
            )


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[dict, dict, dict, list[str]]:
    """Run inference and collect per-attribute argmax preds + softmax probs.

    Returns:
        preds:    {attr: (N,) np.int64}
        probs:    {attr: (N, C_attr) np.float32}
        targets:  {attr: (N,) np.int64}
        ids:      list of image_ids in order
    """
    model.eval()
    out_logits = {a: [] for a in ATTRIBUTES}
    out_target = {a: [] for a in ATTRIBUTES}
    ids: list[str] = []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x)  # dict of {attr: (B, C_attr)}
        for a in ATTRIBUTES:
            out_logits[a].append(logits[a].cpu())
            out_target[a].append(batch[a])
        ids.extend(batch["image_id"])

    preds, probs, targets = {}, {}, {}
    for a in ATTRIBUTES:
        logit = torch.cat(out_logits[a], dim=0)
        probs[a] = torch.softmax(logit, dim=-1).numpy().astype(np.float32)
        preds[a] = logit.argmax(dim=-1).numpy().astype(np.int64)
        targets[a] = torch.cat(out_target[a], dim=0).numpy().astype(np.int64)
    return preds, probs, targets, ids
