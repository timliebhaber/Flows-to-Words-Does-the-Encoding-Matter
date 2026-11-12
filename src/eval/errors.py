"""Error analysis from results/run_details/<run_id>.json (plan §7.1: per-class F1 + confusion matrix)."""

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("analysis")


def aggregate_details(rows, details_dir):
    """{(model, encoding, dataset): {"classes", "per_class_f1" (mean over seeds), "confusion" (summed), "n"}}.

    `rows` are results.csv rows of within-dataset runs. A run without a details file is skipped with a warning.
    """
    out = {}
    for (model, encoding, dataset), g in rows.groupby(["model", "encoding", "test_dataset"]):
        classes, f1s, confusion, n_params = None, [], None, None
        for run_id in g["run_id"]:
            path = Path(details_dir) / f"{run_id}.json"
            if not path.exists():
                log.warning("no run details for %s; skipped in the error analysis", run_id)
                continue
            d = json.loads(path.read_text())
            if classes is None:
                classes, confusion = d["class_names"], np.zeros((len(d["class_names"]),) * 2, dtype=np.int64)
            elif d["class_names"] != classes:
                raise ValueError(f"{run_id}: class names differ from the other seeds of the same run")
            f1s.append([d["per_class_f1"][c] for c in classes])
            n_params = d.get("n_params", n_params)   # text models only
            confusion += np.asarray(d["confusion_matrix"], dtype=np.int64)
        if f1s:
            out[(model, encoding, dataset)] = {"classes": classes, "per_class_f1": np.mean(f1s, axis=0),
                                               "confusion": confusion, "n": len(f1s), "n_params": n_params}
    return out


def row_normalised(confusion):
    """Each row (true class) divided by its total, so cells are shares of that class; empty rows stay 0."""
    totals = confusion.sum(axis=1, keepdims=True)
    return np.divide(confusion, totals, out=np.zeros(confusion.shape), where=totals > 0)


def top_confusions(confusion, classes, k):
    """The k largest off-diagonal mix-ups as (true, predicted, share of the true class, count), largest share first."""
    shares = row_normalised(confusion)
    pairs = [(shares[i, j], confusion[i, j], i, j) for i in range(len(classes)) for j in range(len(classes))
             if i != j and confusion[i, j] > 0]
    pairs.sort(key=lambda p: (-p[0], -p[1]))
    return [(classes[i], classes[j], float(s), int(c)) for s, c, i, j in pairs[:k]]


def hard_classes(per_dataset, threshold):
    """Classes whose best per-class F1 over all methods stays below `threshold`: [(class, best F1)]."""
    best = {}
    for agg in per_dataset.values():
        for c, f in zip(agg["classes"], agg["per_class_f1"]):
            best[c] = max(best.get(c, 0.0), float(f))
    return sorted(((c, f) for c, f in best.items() if f < threshold), key=lambda cf: cf[1])
