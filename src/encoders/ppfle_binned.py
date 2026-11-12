"""E2: PPFLE on quantile bins. Numeric columns -> hash(column$binK); categorical columns hashed raw (plan §5).

Bins are fitted on the train split of ONE dataset. For each numeric column: the unique quantiles at
1/n, 2/n, ..., 1 of the train values (values that occur in train); every quantile but the last is an upper bin edge. A value goes to the
first bin whose edge is >= the value; values above all edges go to the last bin. So bin 0 holds the point
mass at the minimum (often 0), and values outside the train range land in the edge bins.
"""

import json
from pathlib import Path

import numpy as np

from src.encoders.common import column_values, hasher, is_missing, join_rows, map_column
from src.encoders.ppfle import hash_raw_column

NEEDS_FIT = True


def numeric_columns(cfg):
    categorical = set(cfg["binning"]["categorical_columns"])
    return [c for c in cfg["data"]["columns"]["features"] if c not in categorical]


def fit(train_df, cfg):
    n_bins = cfg["binning"]["n_bins"]
    levels = np.linspace(0, 1, n_bins + 1)[1:]
    edges = {}
    for c in numeric_columns(cfg):
        x = train_df[c].to_numpy(dtype=float)
        x = x[~np.isnan(x)]
        edges[c] = np.unique(np.quantile(x, levels, method=cfg["binning"]["quantile_method"]))[:-1].tolist() if len(x) else []
    return {"n_bins": n_bins, "edges": edges}


def bin_index(values, edges):
    return np.searchsorted(np.asarray(edges, dtype=float), np.asarray(values, dtype=float), side="left")


def encode(df, cfg, state):
    fmt, h = cfg["format"], hasher(cfg["hash"]["algorithm"], cfg["hash"]["length"])
    prefix, sep, na = cfg["binning"]["bin_prefix"], fmt["separator"], fmt["na_token"]
    features = cfg["data"]["columns"]["features"]
    values = column_values(df, features)
    columns = []
    for c in features:
        if c not in state["edges"]:
            columns.append(hash_raw_column(c, values[c], cfg))
            continue
        bins = bin_index(values[c], state["edges"][c]).tolist()
        labels = [na if is_missing(v) else f"{prefix}{b}" for v, b in zip(values[c], bins)]
        columns.append(map_column(labels, lambda label, c=c: h(f"{c}{sep}{label}")))
    return join_rows(columns, fmt["token_joiner"])


def state_path(cfg, dataset):
    return Path(cfg["binning"]["state_dir"]) / f"{dataset}.json"


def save_state(state, cfg, dataset):
    path = state_path(cfg, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fit_dataset": dataset, **state}, indent=1) + "\n")


def load_state(cfg, dataset):
    path = state_path(cfg, dataset)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m src.encoders.build` first")
    return json.loads(path.read_text())
