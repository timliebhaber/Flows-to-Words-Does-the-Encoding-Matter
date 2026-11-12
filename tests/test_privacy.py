"""Experiment 4 dictionary attack (src/eval/privacy.py) on a toy frame with the real column set."""

import copy
import json
from pathlib import Path

import pytest

from src.config import ConfigError, load_privacy_config
from src.encoders import encoding_cfg, fit_state, ppfle_binned
from src.encoders.ppfle import hash_raw_column
from src.eval import privacy
from tests.conftest import REPO_ROOT, make_raw

PRIVACY_CONFIG = REPO_ROOT / "configs" / "privacy.yaml"


@pytest.fixture(scope="module")
def real_cfg():
    return load_privacy_config(PRIVACY_CONFIG)


@pytest.fixture
def cfg(real_cfg, tmp_path):
    c = copy.deepcopy(real_cfg)
    c["bruteforce_range"] = [0, 100]                          # toy values are < 50
    c["encoder"]["binning"]["state_dir"] = str(tmp_path / "bins")
    c["outputs"].update(json=str(tmp_path / "privacy.json"), table=str(tmp_path / "privacy"),
                        tex=str(tmp_path / "privacy.tex"))
    c["numbers"].update(tex=str(tmp_path / "numbers.tex"), md=str(tmp_path / "numbers.md"))
    c["figures"] = {**c["figures"], "dir": str(tmp_path / "figures"), "formats": ["png"], "dpi": 40}
    return c


@pytest.fixture
def toy(cfg):
    cols = cfg["data"]["columns"]
    df = make_raw(cols, {"Benign": 40, "DoS": 20}, seed=0, value_range=50)[cols["features"]]
    df.loc[0, "SRC_TO_DST_SECOND_BYTES"] = 3.14159             # a float no integer guess can hit
    state = fit_state(cfg["encoder"], "e2_ppfle_bins", df)
    ppfle_binned.save_state(state, encoding_cfg(cfg["encoder"], "e2_ppfle_bins"), "toy")
    return df


def test_find_reproduces_encoder_hashes(cfg):
    ecfg = encoding_cfg(cfg["encoder"], "e1_ppfle")
    tokens = hash_raw_column("L4_DST_PORT", [80, 443], ecfg)
    found = privacy.find("L4_DST_PORT", privacy.int_range(0, 1000, False), set(tokens), ecfg)
    assert found == {tokens[0]: "80", tokens[1]: "443"}
    float_token = hash_raw_column("L7_PROTO", [7.178], ecfg)[0]
    assert privacy.find("L7_PROTO", privacy.domain({"dotted_range": [0, 511]}, True), {float_token}, ecfg) == \
        {float_token: "7.178"}


def test_candidates_follow_the_column_type():
    assert list(privacy.int_range(5, 6, True)) == ["5.0", "6.0"]
    assert list(privacy.int_range(5, 6, False)) == ["5", "6"]
    assert {"10.16", "7.0", "0.5"} <= privacy.domain({"dotted_range": [0, 20]}, True)
    assert privacy.domain({"values": ["0", "GET"], "hex_range": [0, 1, 8]}, False) == {"0", "GET", "0x00000000",
                                                                                     "0x00000001"}


def test_attack_levels_are_cumulative_and_correct(cfg, toy):
    features = cfg["data"]["columns"]["features"]
    floats = [c for c in cfg["data"]["columns"]["float_features"] if c not in cfg["low_cardinality"]]
    n_low, n, rows = len(cfg["low_cardinality"]), len(features), len(toy)
    no_public = {c: set() for c in features}
    r = privacy.attack(cfg, "toy", "e1_ppfle", toy, no_public)
    lv = r["levels"]
    assert r["wrong"] == 0
    assert lv["enumerate"]["token_share"] == pytest.approx(n_low / n)
    # every numeric value up to the range is found (floats spelled 6.0), except the one non-integer float
    assert lv["bruteforce"]["token_share"] == pytest.approx(1 - 1 / (n * rows))
    assert lv["public"]["token_share"] == lv["bruteforce"]["token_share"]
    assert lv["bruteforce"]["row_share"] == pytest.approx(1 - 1 / rows)

    public = dict(no_public, **{c: {str(v) for v in toy[c]} for c in floats})           # public data has them all
    lv = privacy.attack(cfg, "toy", "e1_ppfle", toy, public)["levels"]
    assert lv["public"]["token_share"] == 1.0 and lv["public"]["row_share"] == 1.0

    e2 = privacy.attack(cfg, "toy", "e2_ppfle_bins", toy, no_public)
    assert e2["wrong"] == 0 and e2["levels"]["enumerate"]["token_share"] == 1.0         # bins + categoricals


def test_is_done_reads_the_json(cfg, tmp_path):
    assert not privacy.is_done(cfg, "toy")
    done = {"toy": {"encodings": {e: {} for e in cfg["encodings"]}}}
    (tmp_path / "privacy.json").write_text(json.dumps(done))
    assert privacy.is_done(cfg, "toy") and not privacy.is_done(cfg, "other")


def test_config_rejects_bad_levels(tmp_path):
    import yaml
    raw = yaml.safe_load(PRIVACY_CONFIG.read_text())
    raw["levels"] = raw["levels"][::-1]
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="levels"):
        load_privacy_config(path)


def test_render_writes_tables_and_numbers(cfg, toy):
    features = cfg["data"]["columns"]["features"]
    r = privacy.attack(cfg, "toy", "e1_ppfle", toy, {c: set() for c in features})
    results = {"toy": {"n_rows": len(toy), "encodings": {e: {**r, "seconds": 1.0} for e in cfg["encodings"]}}}
    privacy.render(cfg, results)
    numbers = Path(cfg["numbers"]["tex"]).read_text()
    assert r"\resset{privacy.toy.e1_ppfle.bruteforce.token_share}" in numbers
    assert r"\resset{privacy.e1_ppfle.enumerate.token_share.max}" in numbers
    assert Path(cfg["outputs"]["tex"]).read_text().startswith("% Experiment 4")


def test_levels_may_be_a_subset(cfg, toy):
    sub = {**cfg, "levels": cfg["levels"][:2], "public_datasets": []}
    r = privacy.attack(sub, "toy", "e1_ppfle", toy, {})
    assert list(r["levels"]) == ["enumerate", "bruteforce"] and r["wrong"] == 0


def test_real_edge_config_loads():
    c = load_privacy_config(REPO_ROOT / "configs" / "privacy_edge.yaml")
    assert [lv["name"] for lv in c["levels"]] == ["enumerate", "bruteforce"] and c["numbers"]["prefix"] != "privacy"

