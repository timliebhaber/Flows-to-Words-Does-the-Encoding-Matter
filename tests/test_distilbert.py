"""DistilBERT backend + side runs on tiny synthetic data (CPU, seconds; tiny random model, cached tokenizer)."""

import copy

import pytest
import yaml
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from src.config import (
    ConfigError,
    load_bert_config,
    load_distilbert_config,
    load_model_config,
    split_roles,
)
from src.models import distilbert, text_runs
from tests.conftest import REPO_ROOT
from tests.test_bert_small import TrainValOnly, _processed

CONFIG = REPO_ROOT / "configs" / "distilbert.yaml"


def tiny_model(cfg, prep):
    return DistilBertForSequenceClassification(DistilBertConfig(
        vocab_size=prep["tok"].get_vocab_size(), dim=32, n_layers=1, n_heads=2, hidden_dim=64,
        max_position_embeddings=cfg["tokenizer"]["max_len_cap"], num_labels=len(prep["names"])))


TINY_BACKEND = text_runs.Backend(prepare=distilbert.prepare, build_model=tiny_model)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    cfg = copy.deepcopy(load_distilbert_config(CONFIG))
    cfg["training"].update(device="cpu", batch_size=32, eval_batch_size=64, epochs=1)
    cfg["tuning"]["lr_grid"] = [0.001]
    cfg["outputs"]["tuning_file"] = str(tmp_path / "tuning.json")
    cols = cfg["data"]["columns"]
    train, val, test = split_roles(cfg["data"])
    splits = TrainValOnly({train: _processed(cols, 200, 0), val: _processed(cols, 60, 1),
                           test: _processed(cols, 60, 2)})
    monkeypatch.setattr(text_runs, "load_splits", lambda data_cfg, dataset: splits)
    return cfg, splits


def test_task_blocks_and_seeds_match_other_models():
    dist = load_distilbert_config(CONFIG)
    bert = load_bert_config(REPO_ROOT / "configs" / "bert_small.yaml")
    xgb = load_model_config(REPO_ROOT / "configs" / "xgb.yaml")["tasks"]
    assert dist["tasks"] == bert["tasks"]
    for task, spec in dist["tasks"].items():
        assert {k: xgb[task][k] for k in spec} == spec
    assert dist["results_columns"] == bert["results_columns"]


def test_max_len_from_train_and_truncation(tiny):
    cfg, splits = tiny
    train, val, _ = split_roles(cfg["data"])
    prep = distilbert.prepare(cfg, "e3_kv", "synthetic", "binary", splits)
    assert prep["meta"]["max_len"] <= cfg["tokenizer"]["max_len_cap"]
    assert prep["meta"]["share_truncated"] <= 1 - cfg["tokenizer"]["coverage_percentile"] / 100 + 1e-9
    assert prep["tok"].encode("dst port=80").tokens[0] == "[CLS]"

    cfg["tokenizer"]["max_len_cap"] = 20  # E3 rows are far longer: every row is cut at the cap
    prep = distilbert.prepare(cfg, "e3_kv", "synthetic", "binary", splits)
    assert prep["meta"]["max_len"] == 20 and prep["meta"]["share_truncated"] == 1.0
    assert prep["notes"] == "truncated=100.0%"
    for split in (train, val):
        ids, y = prep["data"][split]
        assert {len(i) for i in ids} == {20} and len(ids) == len(y) == len(splits[split])
        assert all(i[-1] == prep["tok"].token_to_id("[SEP]") for i in ids)


def test_tune_runs_on_train_val_and_resumes(tiny, monkeypatch):
    cfg, _ = tiny
    entry = text_runs.tune_one(cfg, TINY_BACKEND, "e1_ppfle", "synthetic", "binary", force=False)
    assert entry["best"]["lr"] == 0.001 and len(entry["best"]["curve"]) == cfg["training"]["epochs"]
    calls = []
    monkeypatch.setattr(text_runs, "fit", lambda *a: calls.append(a))
    text_runs.tune_one(cfg, TINY_BACKEND, "e1_ppfle", "synthetic", "binary", force=False)
    assert calls == []


def test_side_runs_restricted_and_reuse_main_lr(tiny):
    cfg, _ = tiny
    side = cfg["side_runs"]
    assert side, "configs/distilbert.yaml defines no side runs"
    combos = [(e, d, t) for e in cfg["encodings"] for d in cfg["data"]["datasets"] for t in cfg["tasks"]]
    allowed = [c for c in combos if text_runs.run_allowed(cfg, *c)]
    main = [e for e in cfg["encodings"] if e not in side]
    n_side = sum(len(s["datasets"]) * len(s["tasks"]) for s in side.values())
    assert len(allowed) == len(main) * len(cfg["data"]["datasets"]) * len(cfg["tasks"]) + n_side
    pilot = cfg["tuning"]["pilot"]
    assert text_runs.tune_combos(cfg, allowed) == [(e, pilot["dataset"], pilot["task"]) for e in main]

    enc, spec = next(iter(side.items()))
    ds, task = spec["datasets"][0], spec["tasks"][0]
    units = [{"encoding": enc, "dataset": d, "task": t} for d in cfg["data"]["datasets"] for t in cfg["tasks"]]
    assert text_runs.final_units(cfg, units) == [{"encoding": enc, "dataset": ds, "task": task}]
    assert text_runs.seeds_for(cfg, enc) == spec["seeds"]
    text_runs.tune_one(cfg, TINY_BACKEND, spec["lr_from"], pilot["dataset"], pilot["task"], force=False)
    assert text_runs.best_tuning(cfg, enc, ds, task)["lr"] == cfg["tuning"]["lr_grid"][0]


@pytest.mark.parametrize("change, message", [
    (lambda side, enc: side[enc].update(lr_from=enc), "lr_from"),
    (lambda side, enc: side[enc].update(datasets=["nope"]), "unknown datasets"),
    (lambda side, enc: side[enc].update(seeds=[99]), "subset of seeds"),
])
def test_invalid_side_runs(tmp_path, change, message):
    raw = yaml.safe_load(CONFIG.read_text())
    change(raw["side_runs"], next(iter(raw["side_runs"])))
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match=message):
        load_distilbert_config(path)


def test_render_smoke_marks_oom():
    out = {"device": "GPU", "dataset": "d", "task": "multiclass", "gradient_checkpointing": False,
           "settings": {"train_rows": 10, "steps": 2}, "full_rows": {"train": 100, "val": 20},
           "results": [{"encoding": "e1", "max_len": 512, "share_truncated": 1.0, "batch_size": 64, "oom": True},
                       {"encoding": "e1", "max_len": 512, "share_truncated": 1.0, "batch_size": 16, "oom": False,
                        "s_per_step": 0.5, "train_rows_per_s": 32.0, "eval_rows_per_s": 90.0, "peak_gb": 3.1,
                        "min_per_epoch": 0.1}]}
    text = distilbert.render_smoke(out)
    assert "| e1 | 512 | 100% | 64 | OOM |" in text and "| 3.1 | 0.1 |" in text
