import numpy as np
import pytest

from src.data import clean, load, sample_split
from tests.conftest import make_raw


def test_targets_keep_rare_whole_and_hit_sample_size():
    counts = {"benign": 1_000_000, "a": 50_000, "b": 30_000, "rare": 500, "tiny": 7}
    t = sample_split.compute_class_targets(counts, 10_000, 1_000, [])
    assert sum(t.values()) == 10_000
    assert t["rare"] == 500 and t["tiny"] == 7
    rest = {c: counts[c] for c in ["benign", "a", "b"]}
    for c, n in rest.items():  # natural proportions among the non-rare classes
        assert t[c] / sum(t[k] for k in rest) == pytest.approx(n / sum(rest.values()), abs=1e-3)


def test_targets_keep_all_classes():
    counts = {"benign": 50_000, "ddos": 7_000_000, "dos": 8_000_000, "theft": 1_600}
    t = sample_split.compute_class_targets(counts, 200_000, 10_000, ["benign"])
    assert t["benign"] == 50_000 and t["theft"] == 1_600
    assert sum(t.values()) == 200_000


def test_targets_errors_and_small_data():
    with pytest.raises(ValueError):
        sample_split.compute_class_targets({"a": 900, "b": 900}, 1_000, 1_000, [])
    counts = {"a": 300, "b": 20_000}
    assert sample_split.compute_class_targets(counts, 50_000, 1_000, []) == counts


def test_select_row_ids_exact_and_deterministic():
    class_ids = {"a": np.arange(0, 1000, 2), "b": np.arange(1, 1000, 2), "c": np.array([7])}
    targets = {"a": 50, "b": 20, "c": 1}
    ids = sample_split.select_row_ids(class_ids, targets, seed=1)
    assert len(ids) == 71 and len(set(ids)) == 71 and np.all(np.diff(ids) > 0)
    assert np.sum(ids % 2 == 0) == 50 and 7 in ids
    assert np.array_equal(ids, sample_split.select_row_ids(class_ids, targets, seed=1))
    assert not np.array_equal(ids, sample_split.select_row_ids(class_ids, targets, seed=2))


def _sample_from_csv(path, cols, label_map, block_size):
    raw_ids = load.scan_classes(path, cols["attack"], block_size)
    class_ids = clean.group_ids_by_class(raw_ids, label_map)
    counts = {c: len(v) for c, v in class_ids.items()}
    targets = sample_split.compute_class_targets(counts, 400, 50, [])
    ids = sample_split.select_row_ids(class_ids, targets, seed=3)
    df = load.read_rows(path, ids, cols["features"] + [cols["label"], cols["attack"]],
                        load.column_types(cols), block_size, cols["row_id"])
    return df, targets


def test_streamed_sample_is_independent_of_block_size(tmp_path, cols):
    raw = make_raw(cols, {"Benign": 3000, "DoS": 600, "Worms": 30}, seed=0, value_range=1000)
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    label_map = {"Benign": "benign", "DoS": "dos", "Worms": "worms"}

    small, _ = _sample_from_csv(path, cols, label_map, block_size=16 * 1024)
    large, _ = _sample_from_csv(path, cols, label_map, block_size=64 * 1024 * 1024)
    assert small.equals(large)
    assert len(small) == 400
    assert (small[cols["attack"]] == "Worms").sum() == 30  # rare class kept whole
    assert not set(cols["leaky"]) & set(small.columns)     # leaky columns never read
    # row ids point at the right raw rows
    expected = raw.iloc[small[cols["row_id"]]].reset_index(drop=True)
    for c in cols["features"] + [cols["attack"]]:
        assert (small[c].to_numpy() == expected[c].to_numpy()).all(), c
