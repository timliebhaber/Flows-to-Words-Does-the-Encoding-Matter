import copy
import os

import numpy as np
import pandas as pd
import pytest

from src.config import load_model_config, split_roles
from src.data.processed import class_names, encode_target, model_frame
from src.eval.leakage import flagged, single_feature_scores
from src.models import xgb
from tests.conftest import REPO_ROOT, make_raw

# XGBoost 3.4.1 (CUDA build): a multiclass fit looks for a GPU even with device="cpu" and fails when the GPU is hidden
# (CUDA_VISIBLE_DEVICES=""), e.g. while a training run owns it. Binary fits are unaffected. Seen 2026-11-01.
needs_visible_gpu = pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
                                       reason="XGBoost 3.4.1 multiclass fit probes CUDA even on CPU")


@pytest.fixture(scope="module")
def mcfg():
    cfg = copy.deepcopy(load_model_config(REPO_ROOT / "configs" / "xgb.yaml"))
    cfg["params"].update(n_estimators=20, n_jobs=1, device="cpu")  # small, fast, runs without a GPU
    cfg["leakage"]["params"].update(n_estimators=10, n_jobs=1)
    cfg["tuning"]["grid"] = {"max_depth": [3], "learning_rate": [0.3], "class_weight": ["none", "balanced"]}
    return cfg


def _processed(cols, n, seed):
    df = make_raw(cols, {"Benign": n, "DoS": n // 2, "Worms": n // 4}, seed=seed, value_range=50)
    df[cols["class"]] = df[cols["attack"]].str.lower()
    # make the task learnable: one feature carries signal
    df["IN_BYTES"] = df[cols["class"]].map({"benign": 10, "dos": 20, "worms": 30}) + df["IN_BYTES"] % 3
    return df


class TrainValOnly(dict):
    """Splits dict that fails loudly if anything reads the test split."""
    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("tuning read the test split")
        return super().__getitem__(key)


@pytest.mark.parametrize("task", [pytest.param("multiclass", marks=needs_visible_gpu), "binary"])
def test_tune_uses_only_train_and_val(mcfg, monkeypatch, task):
    cols = mcfg["data"]["columns"]
    train, val, test = split_roles(mcfg["data"])
    splits = TrainValOnly({train: _processed(cols, 300, 0), val: _processed(cols, 100, 1),
                           test: _processed(cols, 100, 2)})
    monkeypatch.setattr(xgb, "load_splits", lambda data_cfg, dataset: splits)
    result = xgb.tune(mcfg, "synthetic", task)
    assert len(result["grid"]) == 2
    assert result["best"]["val_macro_f1"] > 0.9  # signal feature is learnable


@needs_visible_gpu
def test_seed_changes_model_when_subsampling(mcfg):
    cols = mcfg["data"]["columns"]
    df = _processed(cols, 400, 0)
    X = df[cols["features"]]
    names = class_names(df, cols["class"])
    y = encode_target(df, cols["class"], names)
    params = {**mcfg["params"], "max_depth": 3, "learning_rate": 0.3}
    params.pop("early_stopping_rounds")
    p = [xgb.fit(X, y, None, None, params, s, "none", "mlogloss")[0].predict_proba(X) for s in (0, 0, 1)]
    assert np.array_equal(p[0], p[1])       # same seed -> identical
    assert not np.array_equal(p[0], p[2])   # different seed -> different (subsample < 1)


def test_leakage_flags_label_copy_not_noise(mcfg):
    cols = mcfg["data"]["columns"]
    tr, ev = _processed(cols, 300, 0), _processed(cols, 100, 1)
    for df in (tr, ev):
        df["LEAK"] = df[cols["label"]]
        df["NOISE"] = np.random.default_rng(0).integers(0, 50, len(df))
        df["TEXT_LEAK"] = np.where(df[cols["label"]] == cols["benign_label_value"], "0", "0.0")  # Edge-IIoTset style
    lk = {**mcfg["leakage"], "params": {**mcfg["leakage"]["params"], "enable_categorical": True}}
    scores = single_feature_scores(tr, ev, ["LEAK", "NOISE", "TEXT_LEAK"], ["TEXT_LEAK"], cols["label"],
                                   cols["benign_label_value"], lk, "logloss")
    assert flagged(scores, mcfg["leakage"]["flag_threshold"]) == ["LEAK", "TEXT_LEAK"]


def test_model_frame_uses_train_categories():
    train = pd.DataFrame({"m": ["GET", "0", "POST"], "n": [1.0, 2.0, 3.0]})
    other = pd.DataFrame({"m": ["POST", "PUT"], "n": [4.0, 5.0]})
    X = model_frame(other, ["m", "n"], ["m"], train)
    assert list(X["m"].cat.categories) == ["0", "GET", "POST"]
    assert X["m"].isna().tolist() == [False, True]  # unseen in train -> missing
    assert X["n"].tolist() == [4.0, 5.0]


def test_encode_target_rejects_unseen_class():
    with pytest.raises(ValueError):
        encode_target(pd.DataFrame({"c": ["a", "z"]}), "c", ["a", "b"])
