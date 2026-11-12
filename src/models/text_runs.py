"""Tuning and final runs shared by the text classifiers (small BERT, DistilBERT).

A model module supplies a `Backend`:
    prepare(cfg, encoding, dataset, task, splits) -> dict with tok (a `tokenizers.Tokenizer`, truncation set),
        meta (incl. max_len), names, benign, pad_id, data ({train/val split: (ids, y)}), optional notes
    build_model(cfg, prep) -> a Hugging Face sequence classifier
Everything else lives here: LR tuning on val (first seed), final seeds on test (touched once, after tuning),
results.csv rows, run details, and skipping finished work.

`side_runs` in the model config: an encoding listed there runs only on its datasets/tasks/seeds and uses the LR
tuned for its `lr_from` encoding (e.g. DistilBERT's 8-hex hash variants). `dataset_encodings`: a dataset listed
there runs only those encodings (e.g. Edge-IIoTset as published: E1 only).
"""

import argparse
import json
import logging
import os
import time
from collections import namedtuple
from datetime import date
from pathlib import Path

from src.config import split_roles
from src.data.processed import class_names, encode_target, load_splits
from src.encoders import encode_split
from src.eval.metrics import compute_metrics
from src.eval.results import results_run_ids, upsert_rows, write_run_details
from src.models.text_train import (
    make_loader,
    predict,
    set_seed,
    tokenize,
    train_and_select,
)

log = logging.getLogger("text_runs")

Backend = namedtuple("Backend", ["prepare", "build_model"])


def task_cols(cfg, task):
    cols, spec = cfg["data"]["columns"], cfg["tasks"][task]
    return cols[spec["target"]], cols[spec["benign"]]


def train_val_texts(cfg, encoding, dataset, task, splits):
    """(class names from train, benign class, {split: (texts, y)} for train and val). Never touches test."""
    train, val, _ = split_roles(cfg["data"])
    target, benign = task_cols(cfg, task)
    names = class_names(splits[train], target)
    out = {s: (encode_split(cfg["enc"], encoding, dataset, splits[s]), encode_target(splits[s], target, names))
           for s in (train, val)}
    return names, benign, out


def fit(cfg, backend, prep, lr, seed):
    train, val, _ = split_roles(cfg["data"])
    set_seed(seed, cfg["training"]["deterministic"])
    model = backend.build_model(cfg, prep)
    (tr_ids, y_tr), (va_ids, y_va) = prep["data"][train], prep["data"][val]
    _, curve, best_epoch, secs = train_and_select(model, tr_ids, y_tr, va_ids, y_va, prep["names"], prep["benign"],
                                                  lr, seed, cfg["training"], prep["pad_id"])
    return model, curve, best_epoch, secs


# ---- which runs exist ----

def run_allowed(cfg, encoding, dataset, task):
    only = cfg["dataset_encodings"].get(dataset)
    if only is not None and encoding not in only:
        return False
    side = cfg["side_runs"].get(encoding)
    return side is None or (dataset in side["datasets"] and task in side["tasks"])


def seeds_for(cfg, encoding):
    side = cfg["side_runs"].get(encoding)
    return side["seeds"] if side else cfg["seeds"]


def tuning_key(cfg, dataset, task):
    """(dataset, task) whose tuning result is used for this run: itself (full) or the pilot (pilot scope)."""
    if cfg["tuning"]["scope"] == "pilot":
        return cfg["tuning"]["pilot"]["dataset"], cfg["tuning"]["pilot"]["task"]
    return dataset, task


def tuning_source(cfg, encoding, dataset, task):
    """(encoding, dataset, task) whose tuned LR this run uses."""
    side = cfg["side_runs"].get(encoding)
    return (side["lr_from"] if side else encoding, *tuning_key(cfg, dataset, task))


def tune_combos(cfg, combos):
    """Unique (encoding, dataset, task) combinations to tune for the requested runs."""
    return list(dict.fromkeys(tuning_source(cfg, e, d, t) for e, d, t in combos))


def tune_units(cfg, units):
    """src/run.py hook: requested {encoding, dataset, task} units -> the units that are actually tuned."""
    combos = [(u["encoding"], u["dataset"], u["task"]) for u in units]
    combos = tune_combos(cfg, [c for c in combos if run_allowed(cfg, *c)])
    return [{"encoding": e, "dataset": d, "task": t} for e, d, t in combos]


def final_units(cfg, units):
    """src/run.py hook: drop side-run encodings outside their datasets/tasks."""
    return [u for u in units if run_allowed(cfg, u["encoding"], u["dataset"], u["task"])]


# ---- tuning (val only) ----

def _read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def tune_done(cfg, encoding, dataset, task):
    """Every LR of the grid has been tried for this tuning unit (used by src/run.py)."""
    entry = _read_json(cfg["outputs"]["tuning_file"]).get(encoding, {}).get(dataset, {}).get(task, {})
    return bool(entry.get("best"))


def tune_one(cfg, backend, encoding, dataset, task, force):
    """Every LR in the grid with the first seed; saved after each LR. Returns the tuning entry."""
    path = Path(cfg["outputs"]["tuning_file"])
    seed = cfg["seeds"][0]
    prep = None
    for lr in cfg["tuning"]["lr_grid"]:
        done = _read_json(path).get(encoding, {}).get(dataset, {}).get(task, {}).get("runs", {})
        if str(lr) in done and not force:
            continue
        if prep is None:
            prep = backend.prepare(cfg, encoding, dataset, task, load_splits(cfg["data"], dataset))
        model, curve, best_epoch, secs = fit(cfg, backend, prep, lr, seed)
        best_f1 = max(c["val_macro_f1"] for c in curve)
        log.info("[tune %s/%s/%s] lr=%g -> val macro-F1 %.4f (epoch %d, %.0fs)", encoding, dataset, task, lr,
                 best_f1, best_epoch, secs)
        data = _read_json(path)
        entry = data.setdefault(encoding, {}).setdefault(dataset, {}).setdefault(task, {"runs": {}})
        entry["runs"][str(lr)] = {"lr": lr, "seed": seed, "best_epoch": best_epoch, "val_macro_f1": best_f1,
                                  "curve": curve, "train_time_s": round(secs, 1),
                                  "n_params": sum(p.numel() for p in model.parameters())}
        runs = entry["runs"]
        entry["best"] = max(runs.values(), key=lambda r: r["val_macro_f1"]) if len(runs) == len(
            cfg["tuning"]["lr_grid"]) else None
        path.write_text(json.dumps(data, indent=1) + "\n")
    return _read_json(path)[encoding][dataset][task]


def best_tuning(cfg, encoding, dataset, task):
    source = tuning_source(cfg, encoding, dataset, task)
    tune_enc, tune_ds, tune_task = source
    best = _read_json(cfg["outputs"]["tuning_file"]).get(tune_enc, {}).get(tune_ds, {}).get(tune_task, {}).get("best")
    if not best:
        raise RuntimeError(f"no complete tuning for {'/'.join(source)}; run --stage tune first")
    return best


# ---- final runs (test touched once) ----

def _run_id(cfg, task, encoding, train_ds, test_ds, seed):
    return "-".join([cfg["tasks"][task]["experiment"], cfg["model_name"], encoding, train_ds, test_ds, task, f"s{seed}"])


def _test_sets(cfg, dataset, task):
    return cfg["data"]["datasets"] if cfg["tasks"][task]["cross_dataset"] else [dataset]


def final_done(cfg, encoding, dataset, task, seed):
    existing = results_run_ids(cfg["outputs"]["results_file"])
    return all(_run_id(cfg, task, encoding, dataset, t, seed) in existing for t in _test_sets(cfg, dataset, task))


def final_seeds_done(cfg, encoding, dataset, task):
    """Every seed of (encoding, dataset, task) is in results.csv (used by src/run.py)."""
    return all(final_done(cfg, encoding, dataset, task, s) for s in seeds_for(cfg, encoding))


def final_one(cfg, backend, encoding, dataset, task, seed, test_cache):
    best = best_tuning(cfg, encoding, dataset, task)
    train, _, test = split_roles(cfg["data"])
    prep = backend.prepare(cfg, encoding, dataset, task, load_splits(cfg["data"], dataset))
    model, curve, best_epoch, secs = fit(cfg, backend, prep, best["lr"], seed)
    target, _ = task_cols(cfg, task)
    tc, spec = cfg["training"], cfg["tasks"][task]
    notes = " ".join([f"lr={best['lr']} (tuned on {'/'.join(tuning_key(cfg, dataset, task))})",
                      f"best_epoch={best_epoch}", *([f"lr_from={cfg['side_runs'][encoding]['lr_from']}"]
                                                    if encoding in cfg["side_runs"] else []),
                      *([prep["notes"]] if prep.get("notes") else [])])
    rows = []
    for test_ds in _test_sets(cfg, dataset, task):
        if test_ds not in test_cache:
            test_cache[test_ds] = load_splits(cfg["data"], test_ds)[test]
        df = test_cache[test_ds]
        # other datasets are encoded and tokenized with the TRAINING dataset's encoder state and tokenizer
        ids = tokenize(prep["tok"], encode_split(cfg["enc"], encoding, dataset, df))
        y = encode_target(df, target, prep["names"])
        loader = make_loader(ids, y, tc["eval_batch_size"], False, seed, prep["pad_id"])
        t0 = time.perf_counter()
        pred = predict(model, loader, tc["device"], tc["amp_dtype"])
        infer_ms = (time.perf_counter() - t0) * 1000 / len(df) * cfg["infer_time_per_rows"]
        m = compute_metrics(y, pred, prep["names"], prep["benign"])
        run_id = _run_id(cfg, task, encoding, dataset, test_ds, seed)
        rows.append({"run_id": run_id, "date": date.today().isoformat(), "experiment": spec["experiment"],
                     "encoding": encoding, "model": cfg["model_name"], "train_dataset": dataset,
                     "test_dataset": test_ds, "task": task, "seed": seed, "n_train": len(prep["data"][train][1]),
                     "max_len": prep["meta"]["max_len"], "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                     "attack_recall": m["attack_recall"], "fpr": m["fpr"], "train_time_s": round(secs, 1),
                     "infer_time_ms": round(infer_ms, 4), "notes": notes})
        write_run_details(cfg["outputs"]["run_details_dir"], run_id,
                          {"class_names": [str(n) for n in prep["names"]], "lr": best["lr"], "curve": curve,
                           "best_epoch": best_epoch, "n_params": sum(p.numel() for p in model.parameters()),
                           "max_len": prep["meta"]["max_len"], "tokenizer_meta": prep["meta"], "n_test": len(df),
                           **m})
        log.info("[final %s/%s/%s s%d] test on %s: macro-F1 %.4f", encoding, dataset, task, seed, test_ds,
                 m["macro_f1"])
    upsert_rows(cfg["outputs"]["results_file"], cfg["results_columns"], rows)


# ---- command line ----

def add_cli_args(parser, stages):
    parser.add_argument("--config", required=True, help="model config YAML")
    parser.add_argument("--stage", required=True, choices=stages)
    parser.add_argument("--encoding")
    parser.add_argument("--dataset")
    parser.add_argument("--task")
    parser.add_argument("--force", action="store_true", help="redo finished runs")


def requested_combos(cfg, args, parser):
    """(encoding, dataset, task) combinations matching the filters (side runs only where configured)."""
    pick = {"encoding": cfg["encodings"], "dataset": cfg["data"]["datasets"], "task": list(cfg["tasks"])}
    for name, allowed in pick.items():
        chosen = getattr(args, name)
        if chosen and chosen not in allowed:
            parser.error(f"unknown {name} {chosen!r}; choose from {allowed}")
        if chosen:
            pick[name] = [chosen]
    return [(e, d, t) for e in pick["encoding"] for d in pick["dataset"] for t in pick["task"]
            if run_allowed(cfg, e, d, t)]


def run_stages(cfg, backend, stage, combos, force):
    if cfg["training"]["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required by deterministic cuBLAS
    if stage in ("tune", "all"):
        t0 = time.time()
        to_tune = tune_combos(cfg, combos)
        for i, (e, d, t) in enumerate(to_tune, 1):
            tune_one(cfg, backend, e, d, t, force)
            log.info("tuning progress %d/%d combos, %.0f min elapsed", i, len(to_tune), (time.time() - t0) / 60)
    if stage in ("final", "all"):
        todo = [(e, d, t, s) for e, d, t in combos for s in seeds_for(cfg, e)
                if force or not final_done(cfg, e, d, t, s)]
        t0, test_cache = time.time(), {}
        for i, (e, d, t, s) in enumerate(todo, 1):
            final_one(cfg, backend, e, d, t, s, test_cache)
            per_run = (time.time() - t0) / i
            log.info("final progress %d/%d runs, ETA %.0f min", i, len(todo), per_run * (len(todo) - i) / 60)


def main(backend, load_cfg, description):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cli_args(parser, ["tune", "final", "all"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_cfg(args.config)
    run_stages(cfg, backend, args.stage, requested_combos(cfg, args, parser), args.force)
