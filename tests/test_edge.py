"""Edge-IIoTset support: config inheritance, text features, the placeholder fix, E3/E4 with dotted names, run filters."""

import copy
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
import yaml

from src import encoders
from src.config import (
    ConfigError,
    _read_yaml,
    load_bert_config,
    load_config,
    load_distilbert_config,
    load_encoder_config,
    load_model_config,
    validate_encoder_config,
)
from src.data import clean, load
from src.models import text_runs
from tests.conftest import REPO_ROOT

CONFIGS = REPO_ROOT / "configs"


@pytest.fixture(scope="module")
def edge():
    return load_config(CONFIGS / "data_edge.yaml")


@pytest.fixture(scope="module")
def ecfg():
    return load_encoder_config(CONFIGS / "encoders_edge.yaml")


def test_extends_inherits_and_replaces_whole_sections(tmp_path):
    parent, child = tmp_path / "parent.yaml", tmp_path / "child.yaml"
    parent.write_text(yaml.safe_dump({"a": {"x": 1, "y": 2}, "b": 3}))
    child.write_text(yaml.safe_dump({"extends": str(parent), "a": {"x": 9}}))
    assert _read_yaml(child) == {"a": {"x": 9}, "b": 3}


def test_edge_shares_sampling_and_split_with_netflow(edge):
    netflow = load_config(CONFIGS / "data.yaml")
    for section in ["seeds", "sampling", "cleaning", "dedupe", "split", "io"]:
        assert edge[section] == netflow[section]
    assert set(edge["datasets"]).isdisjoint(netflow["datasets"])
    assert edge["paths"]["stats_summary_file"] != netflow["paths"]["stats_summary_file"]


def test_edge_columns(edge):
    cols = edge["columns"]
    for name in ["frame.time", "ip.src_host", "ip.dst_host", "tcp.payload", "http.file_data", "http.request.full_uri"]:
        assert name in cols["leaky"]  # leaky column for Edge-IIoTset
    assert len(cols["features"]) == 46
    assert set(cols["float_features"]) | set(cols["string_features"]) == set(cols["features"])
    fixed, raw = (edge["dataset_configs"][d] for d in edge["datasets"])
    assert fixed["raw_file"] == raw["raw_file"] and fixed["label_map"] == raw["label_map"]
    assert fixed["string_value_map"] and not raw["string_value_map"]


def test_edge_raw_header_matches(edge):
    f = Path(edge["paths"]["raw_dir"]) / edge["dataset_configs"][edge["datasets"][0]]["raw_file"]
    if not f.exists():
        pytest.skip("Edge-IIoTset raw file not present")
    load.check_header(f, edge["columns"])


def test_string_features_read_as_text(edge):
    cols = edge["columns"]
    types = load.column_types(cols)
    assert all(types[c] == pa.string() for c in cols["string_features"])
    assert all(types[c] == pa.float64() for c in cols["float_features"])


def test_map_string_values_unifies_placeholder():
    df = pd.DataFrame({"t": ["0", "0.0", "MQTT", "0.0"], "n": [0.0, 0.0, 1.0, 2.0]})
    out, changed = clean.map_string_values(df, ["t"], {"0.0": "0"})
    assert out["t"].tolist() == ["0", "0", "MQTT", "0"] and changed == {"t": 2}
    assert df["t"].tolist() == ["0", "0.0", "MQTT", "0.0"]  # input untouched
    same, none = clean.map_string_values(df, ["t"], {})
    assert same.equals(df) and none == {}


def _row(ecfg, **overrides):
    cols = ecfg["data"]["columns"]
    row = {c: 1.0 for c in cols["float_features"]} | {c: "0" for c in cols["string_features"]} | overrides
    return pd.DataFrame([row])[cols["features"]]


def test_e3_e4_with_dotted_column_names(ecfg):
    df = _row(ecfg, **{"tcp.flags": 24.0, "mqtt.topic": "Temperature_and_Humidity"})
    e4 = encoders.encode(ecfg, "e4_sentence", df, None)[0]
    assert "flags 24.0 (ACK flag 1.0)" in e4 and "MQTT topic was Temperature_and_Humidity" in e4
    e3 = encoders.encode(ecfg, "e3_kv", df, None)[0]
    assert e3.split()[0] == "arp.opcode=1.0" and "mqtt.topic=Temperature_and_Humidity" in e3


def test_placeholder_spelling_changes_e1_tokens(ecfg):
    """Why the fix matters for PPFLE: "0" and "0.0" hash to different tokens."""
    a, b = (encoders.encode(ecfg, "e1_ppfle", _row(ecfg, **{"mqtt.topic": v}), None)[0].split() for v in ("0", "0.0"))
    i = ecfg["data"]["columns"]["features"].index("mqtt.topic")
    assert a[i] != b[i] and a[:i] + a[i + 1:] == b[:i] + b[i + 1:]


def test_e2_must_hash_text_features_raw(ecfg):
    bad = copy.deepcopy(ecfg)
    bad["binning"]["categorical_columns"].remove("mqtt.topic")
    with pytest.raises(ConfigError, match="string feature"):
        validate_encoder_config(bad)


def test_edge_model_configs():
    bert = load_bert_config(CONFIGS / "bert_small_edge.yaml")
    dist = load_distilbert_config(CONFIGS / "distilbert_edge.yaml")
    xgb = load_model_config(CONFIGS / "xgb_edge.yaml")
    assert bert["tasks"] == dist["tasks"]
    for task, spec in bert["tasks"].items():
        assert {k: xgb["tasks"][task][k] for k in spec} == spec
    assert not any(s["cross_dataset"] for s in xgb["tasks"].values())
    assert xgb["params"]["enable_categorical"] and xgb["leakage"]["params"]["enable_categorical"]
    netflow = {"bert": load_bert_config(CONFIGS / "bert_small.yaml"),
               "dist": load_distilbert_config(CONFIGS / "distilbert.yaml"),
               "xgb": load_model_config(CONFIGS / "xgb.yaml")}
    for name, cfg in [("bert", bert), ("dist", dist), ("xgb", xgb)]:
        same = ["seeds", "results_columns"] + (["tuning"] if name == "xgb" else ["model", "training", "tokenizer"])
        for section in same:
            assert cfg[section] == netflow[name][section], f"{name}: {section} differs from NetFlow"
        # files that list only one config's datasets must not overwrite the NetFlow ones
        shared = {"results_file", "run_details_dir"}
        for key, path in cfg["outputs"].items():
            if key not in shared:
                assert path != netflow[name]["outputs"][key], f"{name}: outputs.{key}"


def test_dataset_encodings_limit_runs():
    bert = load_bert_config(CONFIGS / "bert_small_edge.yaml")
    dist = load_distilbert_config(CONFIGS / "distilbert_edge.yaml")

    def allowed(cfg):
        return [(e, d, "multiclass") for e in cfg["encodings"] for d in cfg["data"]["datasets"]
                if text_runs.run_allowed(cfg, e, d, "multiclass")]

    raw = [e for e, d, _ in allowed(bert) if d == "edge_iiotset_raw"]
    assert raw == bert["dataset_encodings"]["edge_iiotset_raw"]
    assert all(d == "edge_iiotset" for _, d, _ in allowed(dist))
    pilot = bert["tuning"]["pilot"]
    assert text_runs.tune_combos(bert, allowed(bert)) == [(e, pilot["dataset"], pilot["task"]) for e in bert["encodings"]]


def test_invalid_dataset_encodings(tmp_path):
    raw = yaml.safe_load((CONFIGS / "bert_small_edge.yaml").read_text())
    raw["dataset_encodings"] = {"edge_iiotset_raw": ["nope"]}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="dataset_encodings"):
        load_bert_config(path)
