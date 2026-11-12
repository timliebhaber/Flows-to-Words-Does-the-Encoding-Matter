import numpy as np
import pandas as pd
import pytest

from src.data import clean, sample_split
from tests.conftest import make_raw


def _prepared(cols, class_sizes, value_range, seed=0):
    raw = make_raw(cols, class_sizes, seed=seed, value_range=value_range)
    df = clean.drop_leaky(raw, cols["leaky"])
    df[cols["class"]] = df[cols["attack"]].str.lower()
    df[cols["row_id"]] = np.arange(len(df))
    return clean.enforce_schema(df, cols)


def test_rows_differing_only_in_leaky_columns_are_duplicates(cols):
    raw = make_raw(cols, {"Benign": 2}, seed=0, value_range=1000)
    raw.iloc[1] = raw.iloc[0]
    raw.loc[1, cols["leaky"]] = raw.loc[0, cols["leaky"]] + 1  # differ only in IPs/ports/timestamps
    df = clean.drop_leaky(raw, cols["leaky"])
    df[cols["class"]] = "benign"
    out, info = sample_split.dedupe(df, cols["features"], cols["class"], "drop_all")
    assert len(out) == 1 and info["exact_duplicates_removed"] == 1


@pytest.mark.parametrize("policy,expected_rows", [("drop_all", 1), ("keep", 3)])
def test_conflicts_follow_policy(cols, policy, expected_rows):
    df = _prepared(cols, {"Benign": 3}, value_range=10**6)
    df.loc[1, cols["features"]] = df.loc[0, cols["features"]].to_numpy()
    df.loc[1, cols["class"]] = "dos"  # same features, different class -> conflict
    out, info = sample_split.dedupe(df, cols["features"], cols["class"], policy)
    assert info["conflict_rows"] == 2
    assert len(out) == expected_rows


def test_drop_small_classes(cols):
    df = pd.DataFrame({cols["class"]: ["a"] * 30 + ["b"] * 5})
    out, small = sample_split.drop_small_classes(df, cols["class"], 20)
    assert small == {"b": 5} and set(out[cols["class"]]) == {"a"}


@pytest.fixture
def splits(cfg, cols):
    df = _prepared(cols, {"Benign": 6000, "DoS": 2000, "Worms": 100}, value_range=4)
    df, _ = sample_split.dedupe(df, cols["features"], cols["class"], "drop_all")
    return df, sample_split.split(df, cfg["split"], cols["class"], cols["row_id"], cfg["seeds"]["split"])


def test_split_is_disjoint_and_complete(cols, splits):
    df, parts = splits
    ids = [set(p[cols["row_id"]]) for p in parts.values()]
    assert sum(len(i) for i in ids) == len(df)
    assert set.union(*ids) == set(df[cols["row_id"]])
    key = cols["features"] + [cols["class"]]
    all_rows = pd.concat(parts.values())
    assert not all_rows.duplicated(subset=key).any()  # no identical row in two splits


def test_split_ratios_and_class_proportions(cfg, cols, splits):
    df, parts = splits
    overall = df[cols["class"]].value_counts(normalize=True)
    for name, part in parts.items():
        assert len(part) / len(df) == pytest.approx(cfg["split"][name], abs=0.01)
        props = part[cols["class"]].value_counts(normalize=True)
        assert set(props.index) == set(overall.index)  # every class in every split
        for c in overall.index:
            assert props[c] == pytest.approx(overall[c], abs=0.01)


def test_split_is_deterministic(cfg, cols, splits):
    df, parts = splits
    again = sample_split.split(df, cfg["split"], cols["class"], cols["row_id"], cfg["seeds"]["split"])
    for name in parts:
        assert parts[name].equals(again[name])
