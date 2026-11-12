"""Reading the processed splits written by src/data/pipeline.py (shared by all models)."""

from pathlib import Path

import numpy as np
import pandas as pd


def split_paths(data_cfg, dataset):
    """{split name: Parquet path} for one dataset (written by src/data/pipeline.py)."""
    paths = data_cfg["paths"]
    out = Path(paths["processed_dir"]) / dataset
    return {s: out / f"{s}{paths['split_file_suffix']}" for s in data_cfg["split"]}


def load_splits(data_cfg, dataset):
    """{split name: DataFrame} for one dataset."""
    files = split_paths(data_cfg, dataset)
    missing = [str(f) for f in files.values() if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{missing} not found; run `python -m src.data.pipeline` first")
    return {s: pd.read_parquet(f) for s, f in files.items()}


def model_frame(df, features, string_features, train_df):
    """Feature table for XGBoost: text columns become categoricals with the TRAIN split's values as categories
    (values unseen in train become missing). Numeric columns are passed through."""
    X = df[features].copy()
    for c in [c for c in string_features if c in features]:
        X[c] = pd.Categorical(X[c], categories=sorted(train_df[c].unique()))
    return X


def class_names(train_df, target_col):
    """Sorted target values seen in the train split; position = class id."""
    return sorted(train_df[target_col].unique().tolist())


def encode_target(df, target_col, names):
    """Target column -> integer ids following `names`. Values unseen in train raise."""
    unknown = sorted(set(df[target_col].unique().tolist()) - set(names))
    if unknown:
        raise ValueError(f"target values not in the train classes: {unknown}")
    return df[target_col].map({n: i for i, n in enumerate(names)}).to_numpy(dtype=np.int64)
