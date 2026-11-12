import math

import numpy as np
import pytest

from src.eval.metrics import compute_metrics


def test_binary_metrics_by_hand():
    names = [0, 1]  # 0 = benign
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    m = compute_metrics(y_true, y_pred, names, 0)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["attack_recall"] == pytest.approx(3 / 4)
    assert m["fpr"] == pytest.approx(1 / 4)
    assert m["macro_f1"] == pytest.approx(0.75)  # both classes: P = R = 0.75
    assert np.array(m["confusion_matrix"]).sum() == 8


def test_multiclass_collapses_to_benign_vs_attack():
    names = ["benign", "dos", "worms"]
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 2, 1, 2, 0])  # dos->worms is still a caught attack
    m = compute_metrics(y_true, y_pred, names, "benign")
    assert m["attack_recall"] == pytest.approx(3 / 4)
    assert m["fpr"] == pytest.approx(1 / 2)
    assert set(m["per_class_f1"]) == set(names)
    assert m["macro_f1"] == pytest.approx(np.mean(list(m["per_class_f1"].values())))


def test_no_benign_rows_gives_nan_fpr():
    m = compute_metrics(np.array([1, 1]), np.array([1, 0]), [0, 1], 0)
    assert math.isnan(m["fpr"])
