"""Metrics shared by every model (plan §7.1). Macro-F1 is primary."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def _ratio(num, den):
    return float(num / den) if den else float("nan")


def compute_metrics(y_true, y_pred, names, benign):
    """y_true / y_pred are class ids (positions in `names`); `benign` is the benign entry of `names`.

    attack_recall and fpr collapse the classes to benign vs attack, so they are defined for multiclass too.
    """
    labels = list(range(len(names)))
    benign_id = names.index(benign)
    true_attack, pred_attack = y_true != benign_id, y_pred != benign_id
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "attack_recall": _ratio(np.sum(pred_attack & true_attack), np.sum(true_attack)),
        "fpr": _ratio(np.sum(pred_attack & ~true_attack), np.sum(~true_attack)),
        "per_class_f1": {str(n): float(f) for n, f in zip(names, per_class)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
