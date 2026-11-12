import numpy as np
import pandas as pd
import pytest

from src.data import clean
from tests.conftest import make_raw


def test_drop_leaky_removes_all_and_keeps_dst_port(cols):
    raw = make_raw(cols, {"Benign": 10, "DoS": 5}, seed=0, value_range=5)
    out = clean.drop_leaky(raw, cols["leaky"])
    clean.assert_no_leaky(out, cols["leaky"])
    assert "L4_DST_PORT" in out.columns
    assert set(cols["features"]) <= set(out.columns)
    with pytest.raises(ValueError):
        clean.assert_no_leaky(raw, cols["leaky"])


def test_normalize_labels_maps_and_rejects_unknown():
    label_map = {"Benign": "benign", "ddos": "ddos", "DDoS": "ddos", "Infilteration": "infiltration"}
    s = pd.Series(["Benign", "DDoS", "ddos", "Infilteration"])
    assert clean.normalize_labels(s, label_map).tolist() == ["benign", "ddos", "ddos", "infiltration"]
    with pytest.raises(ValueError, match="Worms"):
        clean.normalize_labels(pd.Series(["Benign", "Worms"]), label_map)


def test_group_ids_by_class_merges_spellings():
    raw_ids = {"ddos": np.array([5, 1]), "DDoS": np.array([3]), "Benign": np.array([0])}
    out = clean.group_ids_by_class(raw_ids, {"ddos": "ddos", "DDoS": "ddos", "Benign": "benign"})
    assert out["ddos"].tolist() == [1, 3, 5]
    with pytest.raises(ValueError):
        clean.group_ids_by_class({"x": np.array([0])}, {"Benign": "benign"})


def test_inf_to_missing_only_touches_inf(cols):
    f = cols["float_features"][0]
    df = pd.DataFrame({f: [1.5, np.inf, -np.inf, np.nan]})
    assert clean.count_inf(df, [f]) == {f: 2}
    out = clean.inf_to_missing(df, [f])
    assert out[f].iloc[0] == 1.5 and out[f].isna().sum() == 3
    assert np.isinf(df[f]).sum() == 2  # input not modified


def test_label_consistency(cols):
    df = pd.DataFrame({cols["label"]: [0, 1, 1], cols["class"]: ["benign", "dos", "benign"]})
    with pytest.raises(ValueError, match="1 rows"):
        clean.check_label_consistency(df, cols["label"], cols["class"], cols["benign_class"],
                                      cols["benign_label_value"])


def test_enforce_schema_fixes_order_and_rejects_extra(cols):
    raw = make_raw(cols, {"Benign": 4}, seed=0, value_range=5)
    df = clean.drop_leaky(raw, cols["leaky"])
    df[cols["class"]] = "benign"
    df[cols["row_id"]] = range(len(df))
    shuffled = df[list(reversed(df.columns))]
    out = clean.enforce_schema(shuffled, cols)
    assert list(out.columns) == ([cols["row_id"]] + cols["features"]
                                 + [cols["label"], cols["attack"], cols["class"]])
    with pytest.raises(ValueError, match="unexpected"):
        clean.enforce_schema(raw.assign(**{cols["class"]: "benign", cols["row_id"]: 0}), cols)
