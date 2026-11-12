"""Fine-tuned DistilBERT on encoded flows, with its own pretrained WordPiece tokenizer (plan §6.3).

Usage (from the repo root; run src.encoders.build first for the E2 bin edges):
    uv run python -m src.models.distilbert --config configs/distilbert.yaml --stage smoke   # timing, no results
    uv run python -m src.models.distilbert --config configs/distilbert.yaml --stage tune    # LR grid, val only
    uv run python -m src.models.distilbert --config configs/distilbert.yaml --stage final   # best LR x seeds, test
    --stage all runs tune + final. Optional filters: --encoding --dataset --task. Finished work is skipped unless
    --force. Or simply: uv run ftw distilbert

max_len = min(512, 99th percentile of train token lengths). E1/E2 rows are longer than 512 WordPieces, so they are
truncated (the last features in column order are cut); the share of truncated rows goes into notes/run details.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import load_distilbert_config, split_roles
from src.data.processed import load_splits
from src.models import text_runs
from src.models.text_train import autocast, make_loader, predict, set_seed, tokenize

log = logging.getLogger("distilbert")


def load_tokenizer(cfg):
    """(a fresh copy of the pretrained fast tokenizer's backend, pad id). Adds [CLS]/[SEP], lowercases."""
    hf = AutoTokenizer.from_pretrained(cfg["pretrained"])
    return Tokenizer.from_str(hf.backend_tokenizer.to_str()), hf.pad_token_id


def prepare(cfg, encoding, dataset, task, splits):
    """Tokenized train/val with max_len measured on train. Never touches the test split."""
    train = split_roles(cfg["data"])[0]
    tok, pad_id = load_tokenizer(cfg)
    names, benign, texts = text_runs.train_val_texts(cfg, encoding, dataset, task, splits)
    lengths = np.array([len(ids) for ids in tokenize(tok, texts[train][0])])  # train only, untruncated
    tk = cfg["tokenizer"]
    max_len = int(min(tk["max_len_cap"], np.ceil(np.percentile(lengths, tk["coverage_percentile"]))))
    tok.enable_truncation(max_len)
    meta = {"max_len": max_len, "p50": int(np.percentile(lengths, 50)), "max": int(lengths.max()),
            "share_truncated": round(float(np.mean(lengths > max_len)), 4)}
    data = {s: (tokenize(tok, t), y) for s, (t, y) in texts.items()}
    return {"tok": tok, "meta": meta, "names": names, "benign": benign, "pad_id": pad_id, "data": data,
            "notes": f"truncated={100 * meta['share_truncated']:.1f}%"}


def build_model(cfg, prep):
    model = AutoModelForSequenceClassification.from_pretrained(cfg["pretrained"], num_labels=len(prep["names"]),
                                                               **cfg["model"])
    if cfg["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    return model


BACKEND = text_runs.Backend(prepare=prepare, build_model=build_model)


# ---- smoke: timing and memory on a train subsample (no results.csv rows) ----

def _subsample(df, n, seed):
    return df.sample(n=min(n, len(df)), random_state=seed)


def time_training(cfg, prep, batch_size):
    """Seconds per optimizer step and peak GPU memory for `steps` steps at `batch_size` (OOM -> None)."""
    sm, tc = cfg["smoke"], cfg["training"]
    train = split_roles(cfg["data"])[0]
    ids, y = prep["data"][train]
    device = tc["device"]
    set_seed(cfg["seeds"][0], tc["deterministic"])
    model = optimizer = None
    try:
        model = build_model(cfg, prep).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=sm["lr"], weight_decay=tc["weight_decay"])
        loader = make_loader(ids, y, batch_size, True, cfg["seeds"][0], prep["pad_id"])
        model.train()
        torch.cuda.reset_peak_memory_stats()
        step, t0 = 0, None
        while step < sm["warmup_steps"] + sm["steps"]:
            for input_ids, mask, ys in loader:
                if step == sm["warmup_steps"]:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                with autocast(device, tc["amp_dtype"]):
                    logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), ys.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step == sm["warmup_steps"] + sm["steps"]:
                    break
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / sm["steps"], torch.cuda.max_memory_allocated() / 2**30, model
    except torch.cuda.OutOfMemoryError:
        return None, None, None
    finally:
        del optimizer
        torch.cuda.empty_cache()


def smoke(cfg):
    sm, tc = cfg["smoke"], cfg["training"]
    train, val, _ = split_roles(cfg["data"])
    seed = cfg["seeds"][0]
    splits = load_splits(cfg["data"], sm["dataset"])
    full_rows = {s: len(splits[s]) for s in (train, val)}
    sub = {train: _subsample(splits[train], sm["train_rows"], seed), val: _subsample(splits[val], sm["eval_rows"], seed)}
    out = {"device": torch.cuda.get_device_name(), "dataset": sm["dataset"], "task": sm["task"],
           "full_rows": full_rows, "settings": sm, "gradient_checkpointing": tc["gradient_checkpointing"],
           "results": []}
    for enc in sm["encodings"]:
        prep = prepare(cfg, enc, sm["dataset"], sm["task"], sub)
        val_ids, y_val = prep["data"][val]
        for bs in sm["batch_sizes"]:
            secs, peak_gb, model = time_training(cfg, prep, bs)
            row = {"encoding": enc, "max_len": prep["meta"]["max_len"], "p50_tokens": prep["meta"]["p50"],
                   "share_truncated": prep["meta"]["share_truncated"], "batch_size": bs}
            if secs is None:
                row["oom"] = True
                log.info("[smoke %s bs=%d] out of memory", enc, bs)
            else:
                loader = make_loader(val_ids, y_val, tc["eval_batch_size"], False, seed, prep["pad_id"])
                t0 = time.perf_counter()
                predict(model, loader, tc["device"], tc["amp_dtype"])
                torch.cuda.synchronize()
                eval_rps = len(y_val) / (time.perf_counter() - t0)
                train_rps = bs / secs
                row.update(oom=False, s_per_step=round(secs, 4), train_rows_per_s=round(train_rps, 1),
                           eval_rows_per_s=round(eval_rps, 1), peak_gb=round(peak_gb, 2),
                           min_per_epoch=round((full_rows[train] / train_rps + full_rows[val] / eval_rps) / 60, 1))
                log.info("[smoke %s bs=%d] %.0f train rows/s, %.0f eval rows/s, peak %.2f GB, ~%.0f min/epoch",
                         enc, bs, train_rps, eval_rps, peak_gb, row["min_per_epoch"])
            del model
            torch.cuda.empty_cache()
            out["results"].append(row)
    o = cfg["outputs"]
    Path(o["smoke_json"]).write_text(json.dumps(out, indent=2) + "\n")
    Path(o["smoke_md"]).write_text(render_smoke(out))
    log.info("wrote %s", o["smoke_md"])


def render_smoke(out):
    train, val = out["full_rows"]
    sm, rows = out["settings"], out["full_rows"]
    about = (f"{out['device']}; {out['dataset']} {out['task']}; bf16 autocast; gradient checkpointing "
             f"{out['gradient_checkpointing']}. Timed on {sm['train_rows']} random train rows, {sm['steps']} steps "
             f"per batch size. min/epoch = one pass over full train ({rows[train]:,}) + val ({rows[val]:,}) at the "
             "measured speed. Generated by `src/models/distilbert.py --stage smoke`.")
    lines = ["# DistilBERT smoke run", "", about, "",
             "| Encoding | max_len | truncated | batch | s/step | train rows/s | eval rows/s | peak GB | min/epoch |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in out["results"]:
        head = f"| {r['encoding']} | {r['max_len']} | {100 * r['share_truncated']:.0f}% | {r['batch_size']} |"
        lines.append(head + (" OOM | | | | |" if r["oom"] else
                             f" {r['s_per_step']} | {r['train_rows_per_s']} | {r['eval_rows_per_s']} | "
                             f"{r['peak_gb']} | {r['min_per_epoch']} |"))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    text_runs.add_cli_args(parser, ["smoke", "tune", "final", "all"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_distilbert_config(args.config)
    if args.stage == "smoke":
        smoke(cfg)
        return
    text_runs.run_stages(cfg, BACKEND, args.stage, text_runs.requested_combos(cfg, args, parser), args.force)


if __name__ == "__main__":
    main()
