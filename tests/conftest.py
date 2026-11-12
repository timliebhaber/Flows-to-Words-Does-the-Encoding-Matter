"""Shared fixtures: the real config (column lists etc.) and a synthetic raw-data generator."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = REPO_ROOT / "configs" / "data.yaml"


@pytest.fixture(scope="session")
def cfg():
    return load_config(DATA_CONFIG)


@pytest.fixture(scope="session")
def cols(cfg):
    return cfg["columns"]


def make_raw(cols, class_sizes, seed, value_range):
    """Synthetic raw frame with every raw column (features, leaky, Label, Attack).

    `class_sizes` maps raw attack name -> rows; the benign class gets the benign label.
    Feature values are random ints in [0, value_range), so a small range creates duplicates.
    """
    rng = np.random.default_rng(seed)
    attack = np.concatenate([[name] * n for name, n in class_sizes.items()])
    rng.shuffle(attack)
    n = len(attack)
    df = pd.DataFrame({c: rng.integers(0, value_range, n) for c in cols["features"]})
    for c in cols["float_features"]:
        df[c] = df[c].astype(float)
    for c in cols["leaky"]:
        df[c] = rng.integers(0, 1_000_000, n)
    df[cols["attack"]] = attack
    benign_raw = [k for k in class_sizes if k.lower() == cols["benign_class"]]
    df[cols["label"]] = np.where(np.isin(attack, benign_raw), cols["benign_label_value"], 1)
    return df
