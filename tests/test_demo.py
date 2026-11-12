"""Live demo (configs/demo.yaml): config checks, the demo's computations (src/demo/logic.py) and a click-through of
the Streamlit app on the real files (skipped while the processed data or the demo models are missing)."""

import copy
import sys

import pytest
import yaml

from src.config import ConfigError, load_demo_config
from src.data.processed import load_splits, split_paths
from src.demo import logic, train
from src.eval.privacy import candidates, int_range
from tests.conftest import REPO_ROOT

DEMO_CONFIG = REPO_ROOT / "configs" / "demo.yaml"
APP = REPO_ROOT / "src" / "demo" / "app.py"


@pytest.fixture(scope="module")
def cfg():
    return load_demo_config(DEMO_CONFIG)


def test_model_dirs_one_per_model(cfg):
    dirs = train.model_dirs(cfg)
    assert next(iter(dirs)) == (cfg["xgb_cfg"]["model_name"], cfg["xgb_cfg"]["encoding"])
    assert [e for m, e in dirs if m == cfg["bert_cfg"]["model_name"]] == cfg["encodings"]
    assert len(set(dirs.values())) == len(dirs)


@pytest.mark.parametrize("change, message", [
    (lambda c: c.update(dataset="nope"), "dataset/task"),
    (lambda c: c.update(encodings=["e1_ppfle_h8"]), "without small BERT tokenizers"),
    (lambda c: c["app"].update(attack_encoding="e3_kv"), "must be attacked"),
    (lambda c: c["app"]["column_info"].pop("PROTOCOL"), "one entry per feature"),
    (lambda c: c["story"]["highlight"].append("NOPE"), "story features"),
    (lambda c: c["story"]["classifiers"].pop(), "one entry per demo model"),
    (lambda c: c["story"]["encodings"].pop("e4_sentence"), "story.encodings"),
    (lambda c: c["story"]["port_guess"].update(column="IN_BYTES"), "low_cardinality"),
])
def test_invalid_demo_config(tmp_path, change, message):
    raw = yaml.safe_load(DEMO_CONFIG.read_text())
    bad = copy.deepcopy(raw)
    change(bad)
    path = tmp_path / "demo.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ConfigError, match=message):
        load_demo_config(path)


def test_attack_candidates_per_level(cfg):
    pcfg = cfg["privacy_cfg"]
    ecfg = pcfg["encoder"]
    na = ecfg["format"]["na_token"]
    port = candidates(pcfg, ecfg, "L4_DST_PORT", "enumerate", set(), {})
    assert port[0] == na and "80" in port
    assert candidates(pcfg, ecfg, "L4_DST_PORT", "bruteforce", set(), {}) == []   # enumerated, not brute-forced
    assert list(candidates(pcfg, ecfg, "IN_BYTES", "bruteforce", set(), {})) == list(
        int_range(*pcfg["bruteforce_range"], False))
    assert candidates(pcfg, ecfg, "IN_BYTES", "public", {"IN_BYTES"}, {"IN_BYTES": {"1"}}) == []   # bin label


@pytest.fixture(scope="module")
def test_split(cfg):
    data = cfg["bert_cfg"]["data"]
    if not split_paths(data, cfg["dataset"])["test"].exists():
        pytest.skip("processed data missing (uv run ftw data)")
    return load_splits(data, cfg["dataset"])["test"]


def rows(cfg, test, n):
    features = cfg["bert_cfg"]["data"]["columns"]["features"]
    step = max(len(test) // n, 1)
    return [test.loc[[i], features] for i in test.index[::step][:n]]


def test_feature_spans_match_the_encoders(cfg, test_split):
    enc, features = cfg["bert_cfg"]["enc"], cfg["bert_cfg"]["data"]["columns"]["features"]
    _, ecfg, _, _ = logic.attack_setup(cfg)
    aliases = enc["kv_text"]["aliases"]
    for row in rows(cfg, test_split, 25):
        for e in cfg["app"]["show_encodings"]:
            text, spans = logic.feature_spans(enc, e, cfg["dataset"], row)   # raises if the rebuilt text differs
            assert set(spans) == set(features)
            assert all(0 <= a < b <= len(text) for a, b in spans.values())
        text, spans = logic.feature_spans(enc, "e1_ppfle", cfg["dataset"], row)
        port = "L4_DST_PORT"
        assert text[slice(*spans[port])] == logic.fingerprint(ecfg, port, row[port].iloc[0])
        text, spans = logic.feature_spans(enc, "e3_kv", cfg["dataset"], row)
        assert text[slice(*spans[port])] == f"{aliases[port]}={row[port].iloc[0]}"


def test_mystery_draws_mix_normal_and_attacks(cfg, test_split):
    labels = logic.classes_of(cfg, test_split)
    drawn = [labels.loc[logic.draw_flow(cfg, test_split, k)] for k in range(400)]
    attacks = [c for c in drawn if c != logic.benign_class(cfg)]
    assert abs(len(attacks) / len(drawn) - cfg["mystery"]["attack_share"]) < 0.1
    assert len(set(attacks)) == len(set(labels)) - 1   # every attack type shows up
    assert labels.loc[logic.draw_flow(cfg, test_split, 3, "scanning")] == "scanning"


def test_every_class_has_a_plain_explanation(cfg, test_split):
    assert set(logic.classes_of(cfg, test_split)) <= set(cfg["story"]["classes"])


def test_attack_on_one_flow_is_checked_against_the_truth(cfg, test_split):
    row = rows(cfg, test_split, 1)[0]
    results = list(logic.crack_flow(cfg, row, {}))
    assert [r["column"] for r in results] == cfg["privacy_cfg"]["data"]["columns"]["features"]
    assert all(r["correct"] for r in results if r["guess"] is not None)
    col = cfg["story"]["port_guess"]["column"]
    _, ecfg, _, _ = logic.attack_setup(cfg)
    value, tried, _ = logic.crack_port(cfg, col, logic.fingerprint(ecfg, col, row[col].iloc[0]))
    assert value == str(row[col].iloc[0]) and tried == 65536


def test_findings_numbers_come_from_the_report(cfg):
    if not all((REPO_ROOT / p).exists() for p in cfg["numbers_files"]):
        pytest.skip("report numbers not generated (uv run ftw analysis privacy)")
    numbers = logic.load_numbers(cfg["numbers_files"])
    for f in cfg["story"]["findings"]:
        for template in (f["big"], f["text"]):
            assert set(logic.placeholders(template)) <= set(numbers), f["title"]


def test_fill_prints_values_and_percentages():
    numbers = {"a": ("0.978 ± 0.003", "0.9781"), "b": ("5", "5")}
    assert logic.fill("{a} | {a|pct} % | {b} of {c}", numbers, 1, "…") == "0.978 ± 0.003 | 97.8 % | 5 of …"


def test_app_clicks_through(cfg, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    if not train.is_done(cfg):
        pytest.skip("demo models not trained (uv run ftw demo)")
    monkeypatch.setattr(sys, "argv", ["app", "--config", str(DEMO_CONFIG)])
    at = AppTest.from_file(str(APP), default_timeout=300).run()

    def html():   # everything drawn with st.html, without the stylesheet
        return " ".join(h.proto.body for h in at.html if not h.proto.body.lstrip().startswith("<style>"))

    assert not at.exception
    assert "ftw-conn" in html()                                   # step 1: the connection card
    at.segmented_control(key="step").set_value(1).run()
    assert not at.exception and html().count("<mark>") == len(cfg["app"]["show_encodings"])   # step 2: highlight
    at.segmented_control(key="step").set_value(2).run()
    guess = next(p for p in at.pills if p.key.startswith("guess_"))
    guess.set_value("ddos").run()                                 # step 3: guess, then the classifiers answer
    classifiers = len(cfg["story"]["classifiers"])                # once in the reveal, once in "What if" (collapsed)
    assert not at.exception and html().count("ftw-card ftw-det") == 2 * classifiers
    assert "Scoreboard" in html()
    slider = next(s for s in at.select_slider if s.key.startswith("wi_IN_PKTS"))
    slider.set_value(cfg["story"]["what_if"]["IN_PKTS"]["options"][-1]).run()   # what if: many more packets
    assert not at.exception and html().count("ftw-card ftw-det") == 2 * classifiers
    at.segmented_control(key="step").set_value(3).run()
    next(b for b in at.button if "every port" in b.label).click().run()   # step 4: brute-force the port
    assert not at.exception and "tried all" in html()
    at.segmented_control(key="step").set_value(4).run()
    next(b for b in at.button if b.label.startswith("Row ")).click().run()   # step 5: findings + leak quiz
    assert not at.exception and html().count('ftw-card ftw-find') == len(cfg["story"]["findings"])
    assert cfg["story"]["missing_number"] not in html()
