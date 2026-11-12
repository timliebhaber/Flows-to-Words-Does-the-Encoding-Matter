"""Tokenizer + small BERT on tiny synthetic data (CPU, seconds)."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.config import load_bert_config, load_model_config, split_roles
from src.models import bert_small, text_runs
from src.tokenizers import train_bpe
from tests.conftest import REPO_ROOT, make_raw


class FakeTok:
    def get_vocab_size(self):
        return 5000


class TrainValOnly(dict):
    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("read the test split")
        return super().__getitem__(key)


def _processed(cols, n, seed):
    df = make_raw(cols, {"Benign": n, "DoS": n // 2}, seed=seed, value_range=50)
    df[cols["class"]] = df[cols["attack"]].str.lower()
    df[cols["features"]] = 0  # constant everywhere except one easy signal column
    df["IN_BYTES"] = df[cols["class"]].map({"benign": 11, "dos": 999})
    return df


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    cfg = copy.deepcopy(load_bert_config(REPO_ROOT / "configs" / "bert_small.yaml"))
    cfg["tokenizer"].update(vocab_size=300, dir=str(tmp_path / "tok"))
    cfg["model"].update(num_hidden_layers=1, hidden_size=32, num_attention_heads=2, intermediate_size=64)
    cfg["training"].update(device="cpu", batch_size=32, eval_batch_size=64, epochs=1, amp_dtype="bfloat16")
    cfg["tuning"]["lr_grid"] = [0.001]
    cfg["outputs"]["tuning_file"] = str(tmp_path / "tuning.json")
    cols = cfg["data"]["columns"]
    train, val, test = split_roles(cfg["data"])
    splits = TrainValOnly({train: _processed(cols, 200, 0), val: _processed(cols, 60, 1),
                           test: _processed(cols, 60, 2)})
    for mod in (text_runs, train_bpe):
        monkeypatch.setattr(mod, "load_splits", lambda data_cfg, dataset: splits)
    return cfg


def test_task_blocks_match_xgb():
    bert = load_bert_config(REPO_ROOT / "configs" / "bert_small.yaml")["tasks"]
    xgb = load_model_config(REPO_ROOT / "configs" / "xgb.yaml")["tasks"]
    for task, spec in bert.items():
        assert {k: xgb[task][k] for k in spec} == spec


def test_tokenizer_train_only_and_max_len(tiny):
    meta = train_bpe.build_one(tiny, "e3_kv", "synthetic")
    tok, loaded = train_bpe.load_tokenizer(tiny, "e3_kv", "synthetic")
    assert loaded == meta and meta["share_truncated"] <= 1 - tiny["tokenizer"]["coverage_percentile"] / 100 + 1e-9
    enc = tok.encode("dst port=80")
    assert enc.tokens[0] == tiny["tokenizer"]["special_tokens"]["cls"]
    assert enc.tokens[-1] == tiny["tokenizer"]["special_tokens"]["sep"]
    assert len(tok.encode("x " * 10_000).ids) == meta["max_len"]  # truncation at max_len


def test_model_has_configured_size():
    cfg = load_bert_config(REPO_ROOT / "configs" / "bert_small.yaml")
    prep = {"tok": FakeTok(), "meta": {"max_len": 256}, "names": list(range(10)), "pad_id": 0}
    model = bert_small.build_model(cfg, prep)
    c = model.config
    m = cfg["model"]
    assert (c.num_hidden_layers, c.hidden_size, c.num_attention_heads, c.intermediate_size) == (
        m["num_hidden_layers"], m["hidden_size"], m["num_attention_heads"], m["intermediate_size"])
    assert model.classifier.out_features == 10


def test_tune_runs_on_train_val_and_resumes(tiny, monkeypatch):
    # Mechanics only: whether a model learns is checked on real data (a toy model this small may not).
    train_bpe.build_one(tiny, "e3_kv", "synthetic")
    entry = text_runs.tune_one(tiny, bert_small.BACKEND, "e3_kv", "synthetic", "binary", force=False)
    assert entry["best"]["lr"] == 0.001 and len(entry["best"]["curve"]) == tiny["training"]["epochs"]
    assert 0 <= entry["best"]["val_macro_f1"] <= 1
    calls = []
    monkeypatch.setattr(text_runs, "fit", lambda *a: calls.append(a))
    text_runs.tune_one(tiny, bert_small.BACKEND, "e3_kv", "synthetic", "binary", force=False)
    assert calls == []  # finished LR is skipped
    saved = json.loads(Path(tiny["outputs"]["tuning_file"]).read_text())
    assert list(saved["e3_kv"]["synthetic"]["binary"]["runs"]) == ["0.001"]


def test_same_seed_same_model(tiny):
    tiny["training"]["deterministic"] = True
    train_bpe.build_one(tiny, "e3_kv", "synthetic")
    prep = bert_small.prepare(tiny, "e3_kv", "synthetic", "binary", text_runs.load_splits(None, None))
    a = text_runs.fit(tiny, bert_small.BACKEND, prep, 0.001, 0)[0]
    b = text_runs.fit(tiny, bert_small.BACKEND, prep, 0.001, 0)[0]
    for (k, va), vb in zip(a.state_dict().items(), b.state_dict().values()):
        assert np.allclose(va.numpy(), vb.numpy()), k


def test_pilot_scope_tunes_once_per_encoding():
    cfg = load_bert_config(REPO_ROOT / "configs" / "bert_small.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["tuning"]["scope"] = "pilot"
    combos = [(e, d, t) for e in cfg["encodings"] for d in cfg["data"]["datasets"] for t in cfg["tasks"]]
    pilot = cfg["tuning"]["pilot"]
    assert text_runs.tune_combos(cfg, combos) == [(e, pilot["dataset"], pilot["task"]) for e in cfg["encodings"]]
    cfg["tuning"]["scope"] = "full"
    assert text_runs.tune_combos(cfg, combos) == combos
