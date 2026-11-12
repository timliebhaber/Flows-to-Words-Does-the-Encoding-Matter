"""Checks on the real pipeline outputs in data/processed/ (skipped for datasets not yet processed)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from tests.conftest import DATA_CONFIG, REPO_ROOT

CFGS = [load_config(DATA_CONFIG), load_config(REPO_ROOT / "configs" / "data_edge.yaml")]   # NetFlow, Edge-IIoTset
CASES = [(cfg, d) for cfg in CFGS for d in cfg["datasets"]]


def _load(cfg, dataset):
    out = Path(cfg["paths"]["processed_dir"]) / dataset
    files = {s: out / f"{s}{cfg['paths']['split_file_suffix']}" for s in cfg["split"]}
    if not all(f.exists() for f in files.values()):
        pytest.skip(f"{dataset} not processed")
    return {s: pd.read_parquet(f) for s, f in files.items()}


@pytest.mark.parametrize("cfg, dataset", CASES, ids=[d for _, d in CASES])
def test_processed_dataset(cfg, dataset):
    cols = cfg["columns"]
    parts = _load(cfg, dataset)
    expected_cols = [cols["row_id"]] + cols["features"] + [cols["label"], cols["attack"], cols["class"]]
    for part in parts.values():
        assert list(part.columns) == expected_cols  # fixed order, no leaky column
    all_rows = pd.concat(parts.values())
    if cfg["cleaning"]["inf_to_missing"]:
        assert not np.isinf(all_rows[cols["float_features"]]).any().any()
    assert all_rows[cols["row_id"]].is_unique
    assert not all_rows.duplicated(subset=cols["features"] + [cols["class"]]).any()
    classes = set(all_rows[cols["class"]])
    for part in parts.values():
        assert set(part[cols["class"]]) == classes
    value_map = cfg["dataset_configs"][dataset]["string_value_map"]
    for c in cols["string_features"]:  # rewritten spellings (e.g. Edge's "0.0" placeholder) are gone
        assert not all_rows[c].isin(list(value_map)).any()

    stats = json.loads((Path(cfg["paths"]["stats_dir"]) / f"{dataset}.json").read_text())
    cc = stats["class_counts"]
    for c in stats["classes_kept_whole"]:  # rare classes entered the sample whole
        assert cc["sampled"][c] == cc["raw"][c]
    assert stats["rows"]["sampled"] == sum(cc["target"].values())


def test_edge_versions_share_the_sample():
    """Both Edge-IIoTset versions sample the same raw rows; they differ only in the placeholder spelling."""
    edge = CFGS[1]
    fixed, raw = (pd.concat(_load(edge, d).values()) for d in edge["datasets"])
    row_id = edge["columns"]["row_id"]
    stats = [json.loads((Path(edge["paths"]["stats_dir"]) / f"{d}.json").read_text()) for d in edge["datasets"]]
    assert stats[0]["class_counts"]["sampled"] == stats[1]["class_counts"]["sampled"]
    shared = fixed.merge(raw, on=row_id, suffixes=("_fixed", "_published"))  # not "_raw": tcp.ack_raw exists
    assert len(shared) > 0.99 * min(len(fixed), len(raw))  # only dedupe conflicts differ
    for c in edge["columns"]["float_features"]:
        assert np.allclose(shared[f"{c}_fixed"], shared[f"{c}_published"], equal_nan=True)
