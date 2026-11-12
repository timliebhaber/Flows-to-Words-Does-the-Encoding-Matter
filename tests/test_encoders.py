"""Encoders E1-E4 against the exact examples in plan §5, plus the encoder invariants."""

import copy

import numpy as np
import pandas as pd
import pytest

from src import encoders
from src.config import ConfigError, load_encoder_config, validate_encoder_config
from src.encoders import ppfle_binned
from src.encoders.common import hasher, rounded_text
from tests.conftest import REPO_ROOT

# Plan §5 example row and its published outputs.
PLAN_ROW = {"PROTOCOL": 6, "L7_PROTO": 7.0, "L4_DST_PORT": 80, "IN_BYTES": 1520, "OUT_BYTES": 40,
            "IN_PKTS": 5, "OUT_PKTS": 1, "FLOW_DURATION_MILLISECONDS": 12}
PLAN_E1_START = "4fbfe692dd218a7c 4bbe6778ca7f0d4a 965debace268a1e7 24c24537f5a7676c"
PLAN_E3 = ("PROTOCOL=6 L7_PROTO=7.0 L4_DST_PORT=80 IN_BYTES=1520 OUT_BYTES=40 IN_PKTS=5 OUT_PKTS=1 "
           "FLOW_DURATION_MILLISECONDS=12")


@pytest.fixture(scope="module")
def ecfg():
    return load_encoder_config(REPO_ROOT / "configs" / "encoders.yaml")


@pytest.fixture
def plan_cfg(ecfg):
    """Config restricted to the plan's 8 example columns, in the plan's order."""
    cfg = copy.deepcopy(ecfg)
    cfg["data"]["columns"]["features"] = list(PLAN_ROW)
    cfg["kv_text"]["use_aliases"] = False  # the plan's E3 example uses the raw column names
    return cfg


def full_row(cfg, **overrides):
    """One row with every feature (0 / 0.0), then overrides."""
    floats = set(cfg["data"]["columns"]["float_features"])
    row = {c: (0.0 if c in floats else 0) for c in cfg["data"]["columns"]["features"]}
    row.update(overrides)
    return pd.DataFrame([row])


def enc_name(cfg, kind):
    return next(name for name, k in cfg["encodings"].items() if k == kind)


def test_e1_reproduces_plan_hashes(plan_cfg):
    text = encoders.encode(plan_cfg, enc_name(plan_cfg, "ppfle"), pd.DataFrame([PLAN_ROW]), None)[0]
    assert text.startswith(PLAN_E1_START)
    assert len(text.split()) == len(PLAN_ROW)


def test_e1_neighbours_unrelated_e2_neighbours_equal(plan_cfg):
    df = pd.DataFrame([PLAN_ROW, {**PLAN_ROW, "IN_BYTES": 1521}])
    e1 = encoders.encode(plan_cfg, enc_name(plan_cfg, "ppfle"), df, None)
    assert e1[0] != e1[1] and "fd5f11ee9d86f4c0" in e1[1]  # plan: IN_BYTES$1521
    state = {"edges": {"IN_BYTES": [100.0, 2000.0]}}
    e2 = encoders.encode(plan_cfg, enc_name(plan_cfg, "ppfle_binned"), df, state)
    assert e2[0] == e2[1]


def test_e2_reproduces_plan_bin_hashes(plan_cfg):
    # edges chosen so IN_BYTES=1520 -> bin7 and OUT_BYTES=40 -> bin2, as in plan §5
    state = {"edges": {"IN_BYTES": [0, 100, 200, 300, 400, 500, 600, 2000, 3000],
                       "OUT_BYTES": [10, 20, 50, 90]}}
    tokens = encoders.encode(plan_cfg, enc_name(plan_cfg, "ppfle_binned"), pd.DataFrame([PLAN_ROW]),
                             state)[0].split()
    assert "69378708c96232be" in tokens and "95ed1196a120e1c9" in tokens
    assert "4fbfe692dd218a7c" in tokens  # PROTOCOL (no edges) is hashed raw


def test_e2_bins_fit_on_train_and_edge_bins(ecfg):
    train = pd.concat([full_row(ecfg, IN_BYTES=0)] * 70 + [full_row(ecfg, IN_BYTES=v) for v in range(1, 31)])
    state = ppfle_binned.fit(train, ecfg)
    edges = state["edges"]["IN_BYTES"]
    assert edges[0] == 0  # zeros (70%) get their own bin
    assert set(state["edges"]).isdisjoint(ecfg["binning"]["categorical_columns"])
    assert ppfle_binned.bin_index([0, 1, 10_000, -5], edges).tolist() == [0, 1, len(edges), 0]
    assert len(edges) + 1 <= ecfg["binning"]["n_bins"]


def test_e3_matches_plan_string(plan_cfg):
    assert encoders.encode(plan_cfg, enc_name(plan_cfg, "kv_text"), pd.DataFrame([PLAN_ROW]), None)[0] == PLAN_E3


def test_rounding():
    assert rounded_text(0.4795134443, 2, "NA") == "0.48"
    assert rounded_text(12.000000, 2, "NA") == "12.0"
    assert rounded_text(1520, 2, "NA") == "1520"
    assert rounded_text(float("nan"), 2, "NA") == "NA"


def test_fixed_column_order_and_determinism(ecfg):
    df = pd.concat([full_row(ecfg, IN_BYTES=v, L7_PROTO=7.37, PROTOCOL=17) for v in (5, 900)], ignore_index=True)
    shuffled = df[list(reversed(df.columns))]
    for enc in ecfg["encodings"]:
        state = encoders.fit_state(ecfg, enc, df)
        assert encoders.encode(ecfg, enc, df, state) == encoders.encode(ecfg, enc, shuffled, state), enc


def test_missing_becomes_na(ecfg):
    f = ecfg["data"]["columns"]["float_features"][-1]
    df = full_row(ecfg, **{f: np.nan})
    sep, na = ecfg["format"]["separator"], ecfg["format"]["na_token"]
    h = hasher(ecfg["hash"]["algorithm"], ecfg["hash"]["length"])
    texts = {enc: encoders.encode(ecfg, enc, df, encoders.fit_state(ecfg, enc, df))[0] for enc in ecfg["encodings"]}
    assert h(f"{f}{sep}{na}") in texts[enc_name(ecfg, "ppfle")]
    assert h(f"{f}{sep}{na}") in texts[enc_name(ecfg, "ppfle_binned")]
    kv = ecfg["kv_text"]
    shown = kv["aliases"][f] if kv["use_aliases"] else f
    assert f"{shown}{kv['kv_separator']}{na}" in texts[enc_name(ecfg, "kv_text")]
    assert f"({na} bytes per second)" in texts[enc_name(ecfg, "sentence")]


def test_e4_names_and_all_values(ecfg):
    name = enc_name(ecfg, "sentence")
    t = encoders.encode(ecfg, name, full_row(ecfg, PROTOCOL=6, L7_PROTO=7.37, L4_DST_PORT=8443, IN_BYTES=31337),
                        None)[0]
    assert "TCP flow" in t and "application protocol HTTP" in t  # 7.37 -> master code 7
    assert "8443" in t and "31337" in t and "{" not in t
    t = encoders.encode(ecfg, name, full_row(ecfg, PROTOCOL=250, L7_PROTO=99999.0), None)[0]
    assert "IP protocol 250" in t and "application protocol code 99999.0" in t


def test_template_must_cover_every_feature(ecfg):
    bad = copy.deepcopy(ecfg)
    bad["sentence"]["template"] = bad["sentence"]["template"][:-1]
    with pytest.raises(ConfigError, match="missing"):
        validate_encoder_config(bad)
    bad = copy.deepcopy(ecfg)
    bad["sentence"]["template"].append("Again {IN_BYTES}.")
    with pytest.raises(ConfigError, match="repeated"):
        validate_encoder_config(bad)


def test_encode_split_uses_saved_train_state(ecfg, tmp_path):
    cfg = copy.deepcopy(ecfg)
    cfg["binning"]["state_dir"] = str(tmp_path)
    name = enc_name(cfg, "ppfle_binned")
    train = pd.concat([full_row(cfg, IN_BYTES=v) for v in range(50)], ignore_index=True)
    state = encoders.fit_state(cfg, name, train)
    ppfle_binned.save_state(state, cfg, "ds_a")
    other = full_row(cfg, IN_BYTES=10_000)  # another dataset's row, encoded with ds_a's edges
    assert encoders.encode_split(cfg, name, "ds_a", other) == encoders.encode(cfg, name, other, state)
    with pytest.raises(FileNotFoundError):
        encoders.encode_split(cfg, name, "ds_missing", other)


def test_e3_aliases(ecfg):
    cfg = copy.deepcopy(ecfg)
    cfg["kv_text"]["use_aliases"] = True
    t = encoders.encode(cfg, enc_name(cfg, "kv_text"), full_row(cfg, L4_DST_PORT=80), None)[0]
    port_alias = cfg["kv_text"]["aliases"]["L4_DST_PORT"]
    assert t.startswith(f"{port_alias}{cfg['kv_text']['kv_separator']}80 ")
    assert "L4_DST_PORT" not in t


def test_short_hash_variants_are_prefixes_with_same_bins(ecfg, tmp_path):
    cfg = copy.deepcopy(ecfg)
    cfg["binning"]["state_dir"] = str(tmp_path)
    variants = {n: s for n, s in cfg["encodings"].items() if isinstance(s, dict)}
    assert variants, "configs/encoders.yaml defines no override variants"
    train = pd.concat([full_row(cfg, IN_BYTES=v, OUT_BYTES=3 * v) for v in range(50)], ignore_index=True)
    for name, spec in variants.items():
        base = enc_name(cfg, spec["kind"])
        state = encoders.fit_state(cfg, base, train)
        if state is not None:
            assert encoders.fit_state(cfg, name, train) == state  # bins do not depend on the hash
            ppfle_binned.save_state(state, cfg, "ds_a")
        short = encoders.encode_split(cfg, name, "ds_a", train)[7].split()
        full = encoders.encode_split(cfg, base, "ds_a", train)[7].split()
        length = spec["hash"]["length"]
        assert [len(t) for t in short] == [length] * len(full) and [f[:length] for f in full] == short
    assert cfg["hash"]["length"] == ecfg["hash"]["length"]  # overrides never leak into the shared config


@pytest.mark.parametrize("spec", [{"hash": {"length": 8}}, {"kind": "ppfle", "hash": {"bits": 8}},
                                  {"kind": "ppfle", "data": {"x": 1}}])
def test_invalid_encoding_override(ecfg, spec):
    bad = copy.deepcopy(ecfg)
    bad["encodings"]["broken"] = spec
    with pytest.raises(ConfigError):
        validate_encoder_config(bad)
