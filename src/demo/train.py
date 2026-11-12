"""Demo models for the live demo (configs/demo.yaml; the app is src/demo/app.py).

Trains XGBoost and small BERT on the demo dataset x task with the benchmark's tuned settings and the first seed,
and saves each under `model_dir` with a meta.json (class names, settings, val/test macro-F1 of this copy).
Never writes results.csv or run_details: these models only show predictions live; the official numbers are the
3-seed benchmark runs.

Usage (from the repo root; needs xgb_tune, tokenizers and bert_small_tune):
    uv run python -m src.demo.train --config configs/demo.yaml [--force]
"""

import argparse
import json
import logging
from pathlib import Path

from xgboost import XGBClassifier

from src.config import load_demo_config, split_roles
from src.data.processed import class_names, encode_target, load_splits
from src.encoders import encode_split
from src.eval.metrics import compute_metrics
from src.models import bert_small, text_runs, xgb
from src.models.text_train import make_loader, predict, tokenize

log = logging.getLogger("demo")


def model_dirs(cfg):
    """{(model, encoding): directory} of every configured demo model (XGBoost's encoding is its config's)."""
    root, x, b = Path(cfg["model_dir"]), cfg["xgb_cfg"], cfg["bert_cfg"]
    keys = ([(x["model_name"], x["encoding"])] if cfg["xgb"] else []) + [(b["model_name"], e) for e in cfg["encodings"]]
    return {(m, e): root / f"{m}_{e}" for m, e in keys}


def is_done(cfg):
    """Every demo model has its meta file (written last; used by src/run.py)."""
    return all((d / cfg["meta_file"]).exists() for d in model_dirs(cfg).values())


def read_meta(cfg, directory):
    return json.loads((Path(directory) / cfg["meta_file"]).read_text())


def train_xgb(cfg, splits, out_dir):
    xcfg, ds, task = cfg["xgb_cfg"], cfg["dataset"], cfg["task"]
    train, val, test = split_roles(xcfg["data"])
    cols = xcfg["data"]["columns"]
    target, benign = xgb._task_columns(xcfg, task)
    best = xgb._load_best(xcfg["outputs"]["tuning_file"], ds, task)
    names = class_names(splits[train], target)
    X = {s: xgb._xy(splits[s], cols, target, names, splits[train]) for s in (train, val, test)}
    model, secs = xgb.fit(*X[train], *X[val], {**xcfg["params"], **best["params"]}, cfg["seed"],
                          best["class_weight"], xcfg["tasks"][task]["eval_metric"])
    scores = {s: compute_metrics(X[s][1], model.predict(X[s][0]), names, benign)["macro_f1"] for s in (val, test)}
    model.save_model(out_dir / cfg["xgb_model_file"])
    return {"params": best["params"], "class_weight": best["class_weight"], "trees": int(model.best_iteration) + 1,
            "class_names": [str(n) for n in names], "benign": benign, "train_time_s": round(secs, 1),
            **{f"{s}_macro_f1": v for s, v in scores.items()}}


def load_xgb(cfg, directory):
    model = XGBClassifier()
    model.load_model(Path(directory) / cfg["xgb_model_file"])
    return model


def train_bert(cfg, encoding, splits, out_dir):
    bcfg, ds, task = cfg["bert_cfg"], cfg["dataset"], cfg["task"]
    _, val, test = split_roles(bcfg["data"])
    best = text_runs.best_tuning(bcfg, encoding, ds, task)
    prep = bert_small.prepare(bcfg, encoding, ds, task, splits)
    model, curve, best_epoch, secs = text_runs.fit(bcfg, bert_small.BACKEND, prep, best["lr"], cfg["seed"])
    target, _ = text_runs.task_cols(bcfg, task)
    tc = bcfg["training"]
    ids = tokenize(prep["tok"], encode_split(bcfg["enc"], encoding, ds, splits[test]))
    y = encode_target(splits[test], target, prep["names"])
    pred = predict(model, make_loader(ids, y, tc["eval_batch_size"], False, cfg["seed"], prep["pad_id"]),
                   tc["device"], tc["amp_dtype"])
    model.save_pretrained(out_dir)
    return {"lr": best["lr"], "best_epoch": best_epoch, "curve": curve, "max_len": prep["meta"]["max_len"],
            "n_params": sum(p.numel() for p in model.parameters()), "class_names": [str(n) for n in prep["names"]],
            "benign": prep["benign"], "train_time_s": round(secs, 1),
            f"{val}_macro_f1": max(c["val_macro_f1"] for c in curve),
            f"{test}_macro_f1": compute_metrics(y, pred, prep["names"], prep["benign"])["macro_f1"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="demo config YAML, e.g. configs/demo.yaml")
    parser.add_argument("--force", action="store_true", help="retrain models that already exist")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_demo_config(args.config)
    splits = None
    for (model, encoding), out_dir in model_dirs(cfg).items():
        if (out_dir / cfg["meta_file"]).exists() and not args.force:
            log.info("[%s/%s] exists, skipped", model, encoding)
            continue
        splits = splits or load_splits(cfg["bert_cfg"]["data"], cfg["dataset"])
        out_dir.mkdir(parents=True, exist_ok=True)
        if model == cfg["xgb_cfg"]["model_name"]:
            meta = train_xgb(cfg, splits, out_dir)
        else:
            meta = train_bert(cfg, encoding, splits, out_dir)
        meta = {"model": model, "encoding": encoding, "dataset": cfg["dataset"], "task": cfg["task"],
                "seed": cfg["seed"], **meta}
        (out_dir / cfg["meta_file"]).write_text(json.dumps(meta, indent=1) + "\n")
        log.info("[%s/%s] saved to %s: %s", model, encoding, out_dir,
                 {k: round(v, 4) for k, v in meta.items() if k.endswith("_macro_f1")})


if __name__ == "__main__":
    main()
