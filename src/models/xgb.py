"""XGBoost baseline on the cleaned table (plan §6.1).

Usage (from the repo root):
    uv run python -m src.models.xgb --config configs/xgb.yaml --stage tune    # grid on val only
    uv run python -m src.models.xgb --config configs/xgb.yaml --stage final   # best params x seeds, test once
Optional: --dataset <name> and/or --task <name> to run a subset.
"""

import argparse
import itertools
import json
import logging
import time
from datetime import date
from pathlib import Path

from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import load_model_config, split_roles
from src.data.processed import class_names, encode_target, load_splits, model_frame
from src.eval.metrics import compute_metrics
from src.eval.results import results_run_ids, upsert_rows, write_run_details

log = logging.getLogger("xgb")


def sample_weight(y, class_weight):
    return None if class_weight == "none" else compute_sample_weight(class_weight, y)


def fit(X_tr, y_tr, X_val, y_val, params, seed, class_weight, eval_metric):
    """Fit one model. With a val set, it is used for early stopping (if `params` enables it)."""
    model = XGBClassifier(**params, random_state=seed, eval_metric=eval_metric)
    eval_set = None if X_val is None else [(X_val, y_val)]
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr, sample_weight=sample_weight(y_tr, class_weight), eval_set=eval_set, verbose=False)
    return model, time.perf_counter() - t0


def _xy(df, cols, target_col, names, train_df):
    return model_frame(df, cols["features"], cols["string_features"], train_df), encode_target(df, target_col, names)


def _task_columns(cfg, task):
    cols = cfg["data"]["columns"]
    spec = cfg["tasks"][task]
    return cols[spec["target"]], cols[spec["benign"]]


def tune(cfg, dataset, task):
    """Grid search on the validation split. Never reads the test split."""
    train, val, _ = split_roles(cfg["data"])
    splits = load_splits(cfg["data"], dataset)
    cols = cfg["data"]["columns"]
    target_col, benign = _task_columns(cfg, task)
    names = class_names(splits[train], target_col)
    X_tr, y_tr = _xy(splits[train], cols, target_col, names, splits[train])
    X_va, y_va = _xy(splits[val], cols, target_col, names, splits[train])

    grid = cfg["tuning"]["grid"]
    scores = []
    for values in itertools.product(*grid.values()):
        combo = dict(zip(grid, values))
        class_weight = combo.pop("class_weight")
        model, secs = fit(X_tr, y_tr, X_va, y_va, {**cfg["params"], **combo}, cfg["seeds"][0], class_weight,
                          cfg["tasks"][task]["eval_metric"])
        m = compute_metrics(y_va, model.predict(X_va), names, benign)
        scores.append({"params": combo, "class_weight": class_weight, "val_macro_f1": m["macro_f1"],
                       "best_iteration": int(model.best_iteration), "train_time_s": round(secs, 2)})
        log.info("[%s/%s] %s cw=%s -> val macro-F1 %.4f (%d trees, %.0fs)", dataset, task, combo,
                 class_weight, m["macro_f1"], model.best_iteration + 1, secs)
    best = max(scores, key=lambda s: s["val_macro_f1"])  # first one wins ties
    return {"grid": scores, "best": best}


def _save_tuning(path, dataset, task, result):
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault(dataset, {})[task] = result
    path.write_text(json.dumps(data, indent=2) + "\n")


def tune_done(cfg, dataset, task):
    """A tuning result exists for (dataset, task) (used by src/run.py)."""
    path = Path(cfg["outputs"]["tuning_file"])
    data = json.loads(path.read_text()) if path.exists() else {}
    return "best" in data.get(dataset, {}).get(task, {})


def _load_best(path, dataset, task):
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    if task not in data.get(dataset, {}):
        raise FileNotFoundError(f"no tuning result for {dataset}/{task} in {path}; run --stage tune first")
    return data[dataset][task]["best"]


def _run_id(cfg, task, dataset, test_ds, seed):
    return "-".join([cfg["tasks"][task]["experiment"], cfg["model_name"], cfg["encoding"], dataset, test_ds, task,
                     f"s{seed}"])


def _test_sets(cfg, dataset, task):
    return cfg["data"]["datasets"] if cfg["tasks"][task]["cross_dataset"] else [dataset]


def final_done(cfg, dataset, task):
    """Every seed x test set of (dataset, task) is in results.csv (used by src/run.py)."""
    existing = results_run_ids(cfg["outputs"]["results_file"])
    return all(_run_id(cfg, task, dataset, t, s) in existing
               for s in cfg["seeds"] for t in _test_sets(cfg, dataset, task))


def run_final(cfg, dataset, task, test_cache):
    """Best params x seeds; evaluate on the test split(s) once. Returns results.csv rows."""
    train, val, test = split_roles(cfg["data"])
    spec = cfg["tasks"][task]
    cols = cfg["data"]["columns"]
    target_col, benign = _task_columns(cfg, task)
    best = _load_best(cfg["outputs"]["tuning_file"], dataset, task)
    params = {**cfg["params"], **best["params"]}

    splits = load_splits(cfg["data"], dataset)
    names = class_names(splits[train], target_col)
    X_tr, y_tr = _xy(splits[train], cols, target_col, names, splits[train])
    X_va, y_va = _xy(splits[val], cols, target_col, names, splits[train])
    test_sets = _test_sets(cfg, dataset, task)

    rows = []
    for seed in cfg["seeds"]:
        model, secs = fit(X_tr, y_tr, X_va, y_va, params, seed, best["class_weight"], spec["eval_metric"])
        for test_ds in test_sets:
            if test_ds not in test_cache:
                test_cache[test_ds] = load_splits(cfg["data"], test_ds)[test]
            X_te, y_te = _xy(test_cache[test_ds], cols, target_col, names, splits[train])
            t0 = time.perf_counter()
            pred = model.predict(X_te)
            infer_ms = (time.perf_counter() - t0) * 1000 / len(X_te) * cfg["infer_time_per_rows"]
            m = compute_metrics(y_te, pred, names, benign)
            run_id = _run_id(cfg, task, dataset, test_ds, seed)
            note = (f"{best['params']} class_weight={best['class_weight']} trees={model.best_iteration + 1}"
                    .replace(",", ";"))
            rows.append({"run_id": run_id, "date": date.today().isoformat(), "experiment": spec["experiment"],
                         "encoding": cfg["encoding"], "model": cfg["model_name"], "train_dataset": dataset,
                         "test_dataset": test_ds, "task": task, "seed": seed, "n_train": len(X_tr),
                         "max_len": "", "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                         "attack_recall": m["attack_recall"], "fpr": m["fpr"], "train_time_s": round(secs, 3),
                         "infer_time_ms": round(infer_ms, 4), "notes": note})
            write_run_details(cfg["outputs"]["run_details_dir"], run_id,
                              {"class_names": [str(n) for n in names], "params": params,
                               "class_weight": best["class_weight"], "best_iteration": int(model.best_iteration),
                               "n_test": len(X_te), **m})
            log.info("[%s/%s s%d] test on %s: macro-F1 %.4f", dataset, task, seed, test_ds, m["macro_f1"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="model config YAML, e.g. configs/xgb.yaml")
    parser.add_argument("--stage", required=True, choices=["tune", "final"])
    parser.add_argument("--dataset", help="one dataset (default: all in the data config)")
    parser.add_argument("--task", help="one task (default: all in the model config)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_model_config(args.config)
    datasets = [args.dataset] if args.dataset else cfg["data"]["datasets"]
    tasks = [args.task] if args.task else list(cfg["tasks"])
    for name, allowed in [("dataset", cfg["data"]["datasets"]), ("task", list(cfg["tasks"]))]:
        chosen = getattr(args, name)
        if chosen and chosen not in allowed:
            parser.error(f"unknown {name} {chosen!r}; choose from {allowed}")

    test_cache = {}
    for dataset in datasets:
        for task in tasks:
            if args.stage == "tune":
                _save_tuning(cfg["outputs"]["tuning_file"], dataset, task, tune(cfg, dataset, task))
            else:
                upsert_rows(cfg["outputs"]["results_file"], cfg["results_columns"],
                            run_final(cfg, dataset, task, test_cache))


if __name__ == "__main__":
    main()
