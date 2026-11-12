"""Analysis (src/eval/analysis.py + helpers) on a tiny synthetic results.csv; never reads the real results."""

import copy
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ConfigError, load_analysis_config
from src.eval import analysis
from src.eval.aggregate import (
    ResultsReadError,
    Stat,
    cross_summary,
    fmt_stat,
    load_results,
    make_stat,
    within_rows,
)
from src.eval.errors import aggregate_details, hard_classes, top_confusions
from src.eval.hypotheses import compare, h2, h3, verdict

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_CONFIG = REPO_ROOT / "configs" / "analysis.yaml"
CLASSES = ["a", "benign", "b"]


@pytest.fixture(scope="module")
def real_cfg():
    return load_analysis_config(ANALYSIS_CONFIG)


@pytest.fixture
def cfg(real_cfg, tmp_path):
    c = copy.deepcopy(real_cfg)
    c["inputs"].update(results_file=str(tmp_path / "results.csv"), run_details_dir=str(tmp_path / "details"))
    c["figures"].update(dir=str(tmp_path / "figures"), formats=["png"], dpi=40)
    c["outputs"] = {k: str(tmp_path / Path(v).name) for k, v in c["outputs"].items()}
    c["outputs"]["tex_dir"] = str(tmp_path / "tex")
    c["numbers"].update(tex=str(tmp_path / "tex" / "numbers.tex"), md=str(tmp_path / "numbers.md"), sources=[])
    return c


def _row(cfg, experiment, model, encoding, train, test, task, seed, f1, recall=0.9, fpr=0.1):
    r = {c: "" for c in cfg["inputs"]["required_columns"]}
    r.update(run_id=f"{experiment}-{model}-{encoding}-{train}-{test}-{task}-s{seed}", experiment=experiment,
             encoding=encoding, model=model, train_dataset=train, test_dataset=test, task=task, seed=seed,
             macro_f1=f1, accuracy=f1 + 0.05, attack_recall=recall, fpr=fpr, max_len="" if model == "xgboost" else 64,
             train_time_s=10.0, infer_time_ms=1.0)
    return r


def _details(cfg, row, confusion):
    d = Path(cfg["inputs"]["run_details_dir"])
    d.mkdir(parents=True, exist_ok=True)
    f1 = {c: float(confusion[i][i] / max(1, sum(confusion[i]))) for i, c in enumerate(CLASSES)}
    (d / f"{row['run_id']}.json").write_text(json.dumps(
        {"class_names": CLASSES, "per_class_f1": f1, "confusion_matrix": confusion}))


def write_results(cfg, rows):
    pd.DataFrame(rows).to_csv(cfg["inputs"]["results_file"], index=False)


def synthetic_rows(cfg, with_edge=False):
    ds = cfg["cross"]["datasets"]
    rows = []
    for seed, f1 in enumerate([0.70, 0.72, 0.71]):
        rows.append(_row(cfg, "exp2", "xgboost", "none", ds[0], ds[0], "multiclass", seed, f1))
    for seed, f1 in enumerate([0.50, 0.52, 0.48]):
        rows.append(_row(cfg, "exp2", "bert_small", "e1_ppfle", ds[0], ds[0], "multiclass", seed, f1))
    rows.append(_row(cfg, "exp2", "distilbert", "e1_ppfle", ds[0], ds[0], "multiclass", 0, 0.60))
    rows.append(_row(cfg, "exp2", "distilbert", "e1_ppfle_h8", ds[0], ds[0], "multiclass", 0, 0.62))
    for seed in (0, 1):                      # complete 4x4 XGBoost grid, 2 seeds
        for tr in ds:
            for te in ds:
                rows.append(_row(cfg, "exp3", "xgboost", "none", tr, te, "binary", seed,
                                 0.98 if tr == te else 0.5 + 0.1 * seed))
    for tr in ds[:2]:                        # incomplete small-BERT grid: must not count
        for te in ds:
            rows.append(_row(cfg, "exp3", "bert_small", "e1_ppfle", tr, te, "binary", 0, 0.9))
    if with_edge:
        edge = "edge_iiotset"
        rows += [_row(cfg, "exp1", "xgboost", "none", edge, edge, "multiclass", s, 0.9) for s in range(3)]
    for r in rows:
        if r["task"] == "multiclass":
            _details(cfg, r, [[8, 1, 1], [0, 10, 0], [5, 0, 5]])
    return rows


# ---- config ----

def test_real_config_loads(real_cfg):
    assert set(real_cfg["cross"]["datasets"]) <= set(real_cfg["datasets"])


@pytest.mark.parametrize("break_it, message", [
    (lambda c: c["hypotheses"].pop("tie_margin"), "missing keys"),
    (lambda c: c["hypotheses"].update(baseline_model="nope"), "unknown models"),
    (lambda c: c["cross"].update(datasets=["nope"]), "cross.datasets"),
    (lambda c: c["within"].update(metrics=["nope"]), "required_columns"),
])
def test_config_errors(tmp_path, break_it, message):
    import yaml
    raw = yaml.safe_load(ANALYSIS_CONFIG.read_text())
    break_it(raw)
    path = tmp_path / "a.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True))
    with pytest.raises(ConfigError, match=message):
        load_analysis_config(path)


# ---- aggregation ----

def test_stats_and_formatting(real_cfg):
    fmt = real_cfg["format"]
    s = make_stat([0.70, 0.72, 0.71], fmt["std_ddof"])
    assert s.n == 3 and s.mean == pytest.approx(0.71) and s.std == pytest.approx(0.01)
    one = make_stat([0.6], fmt["std_ddof"])
    assert np.isnan(one.std)
    assert fmt_stat(one, fmt) == f"0.600{fmt['one_seed_marker']}"
    assert fmt_stat(s, fmt) == "0.710 ± 0.010"
    assert fmt_stat(None, fmt) == fmt["pending"]


def test_load_results_rejects_broken_files(cfg):
    rows = synthetic_rows(cfg)
    write_results(cfg, rows + [rows[0]])
    with pytest.raises(ResultsReadError, match="duplicate"):
        load_results(cfg)
    broken = dict(rows[1], run_id="x", seed="")
    write_results(cfg, rows + [broken])
    with pytest.raises(ResultsReadError, match="incomplete"):
        load_results(cfg)


def test_cross_summary_averages_per_seed_and_skips_incomplete_grids(cfg):
    write_results(cfg, synthetic_rows(cfg))
    df = load_results(cfg)
    rows = df[df["experiment"] == "exp3"]
    s = cross_summary(rows, cfg["cross"]["datasets"], "macro_f1", cfg["format"]["std_ddof"])
    assert set(s) == {("xgboost", "none")}            # the incomplete small-BERT grid is left out
    assert s[("xgboost", "none")]["diag"].mean == pytest.approx(0.98)
    off = s[("xgboost", "none")]["off"]
    assert (off.mean, off.n) == (pytest.approx(0.55), 2) and off.std == pytest.approx(np.std([0.5, 0.6], ddof=1))


def test_exp2_table_marks_one_seed_and_fills_edge_when_it_arrives(cfg):
    write_results(cfg, synthetic_rows(cfg))
    table, _, _ = analysis.exp2_table(cfg, within_rows(load_results(cfg), cfg), "macro_f1")
    edge_col = list(cfg["datasets"]).index("edge_iiotset")
    xgb = dict(table["rows"])[cfg["methods"][0]["label"]]
    assert xgb[edge_col].stat is None and "pending" in analysis.render_md(table, cfg["format"])
    md = analysis.render_md(table, cfg["format"])
    assert "0.600†" in md and "**0.710 ± 0.010**" in md
    tex = analysis.render_tex(table, cfg["format"])
    assert r"\textbf{\resor{exp2.xgboost.none.nf_unsw_nb15.macro_f1}{pending}}" in tex
    body = re.sub(r"\\res(or|diff)\{[^{}]*\}\{[^{}]*\}", "", "\n".join(tex.splitlines()[1:]))
    assert not re.search(r"\d\.\d", body)                     # no number typed into the LaTeX table

    write_results(cfg, synthetic_rows(cfg, with_edge=True))
    table, _, _ = analysis.exp2_table(cfg, within_rows(load_results(cfg), cfg), "macro_f1")
    assert dict(table["rows"])[cfg["methods"][0]["label"]][edge_col].stat.mean == pytest.approx(0.9)


def test_side_runs_compare_with_main_encoding(cfg):
    write_results(cfg, synthetic_rows(cfg))
    table, records = analysis.side_runs_table(cfg, within_rows(load_results(cfg), cfg), "macro_f1")
    diff = table["rows"][0][1][2]
    assert len(table["rows"]) == 1 and diff.text == "+0.020"
    assert diff.tex == r"\resdiff{exp2.distilbert.e1_ppfle_h8.nf_unsw_nb15.macro_f1}{exp2.distilbert.e1_ppfle.nf_unsw_nb15.macro_f1}"
    assert records.loc[0, "main_encoding"] == "e1_ppfle"


# ---- error analysis ----

def test_details_sum_confusions_over_seeds(cfg):
    write_results(cfg, synthetic_rows(cfg))
    aggs = aggregate_details(within_rows(load_results(cfg), cfg), cfg["inputs"]["run_details_dir"])
    agg = aggs[("xgboost", "none", cfg["cross"]["datasets"][0])]
    assert agg["n"] == 3 and agg["confusion"][2][0] == 15      # 5 per seed x 3 seeds
    top = top_confusions(agg["confusion"], agg["classes"], 2)
    assert top[0][:2] == ("b", "a") and top[0][2] == pytest.approx(0.5)
    assert [c for c, _ in hard_classes({"m": agg}, 0.6)] == ["b"]


# ---- hypotheses ----

def test_compare_uses_margin_and_seed_std():
    assert compare(Stat(0.80, 0.001, 3), Stat(0.75, 0.001, 3), 0.01) == 1
    assert compare(Stat(0.80, 0.10, 3), Stat(0.75, 0.001, 3), 0.01) == 0     # within one std: tie
    assert compare(Stat(0.70, float("nan"), 1), Stat(0.75, 0.001, 3), 0.01) == -1
    assert verdict([]) == "no data yet"
    assert verdict([("x", True), ("y", True)]) == "supported"
    assert verdict([("x", False)]) == "not supported"
    assert verdict([("x", True), ("y", False)]).startswith("mixed (1 of 2")


def test_h2_and_h3_verdicts(cfg):
    ds = list(cfg["datasets"])
    within = {("xgboost", "none", ds[0]): Stat(0.90, 0.01, 3), ("bert_small", "e1_ppfle", ds[0]): Stat(0.85, 0.01, 3),
              ("xgboost", "none", ds[1]): Stat(0.80, 0.01, 3), ("distilbert", "e3_kv", ds[1]): Stat(0.90, float("nan"), 1)}
    r = h2(cfg, within)
    assert r["verdict"].startswith("mixed (1 of 2") and r["tentative"]

    off = {("bert_small", "e1_ppfle"): Stat(0.40, 0.01, 3), ("bert_small", "e3_kv"): Stat(0.50, 0.01, 3)}
    cross = {m: {k: {"diag": v, "off": v} for k, v in off.items()} for m in cfg["cross"]["metrics"]}
    r = h3(cfg, cross)
    assert r["verdict"] == "supported" and not r["tentative"]


# ---- full run + done check ----

def test_run_writes_everything_and_done_follows_results(cfg):
    write_results(cfg, synthetic_rows(cfg, with_edge=True))
    assert not analysis.is_done(cfg)
    answers, written, stems = analysis.run(cfg)
    assert [a["id"] for a in answers] == ["RQ1", "H1", "H2", "H3", "H4"]
    assert all(Path(p).exists() for p in written)
    assert "exp3_heatmap_xgboost_none" in stems and "exp2_within_macro_f1" in stems
    assert analysis.is_done(cfg)
    numbers = Path(cfg["numbers"]["tex"]).read_text()
    for key in ["exp2.xgboost.none.nf_unsw_nb15.macro_f1", "exp3.xgboost.none.off.macro_f1", "hyp.H2.verdict",
                "seeds.xgboost", "time.xgboost.train.median_s", "perclass.xgboost.none.nf_unsw_nb15.b",
                "confusion.xgboost.none.nf_unsw_nb15.b.a", "exp1.xgboost.none.edge_iiotset.multiclass.accuracy"]:
        assert rf"\resset{{{key}}}" in numbers, key
    assert (Path(cfg["outputs"]["tex_dir"]) / "exp3_cross.tex").exists()
    later = time.time() + 5
    os.utime(cfg["inputs"]["results_file"], (later, later))    # a training run appended a row
    assert not analysis.is_done(cfg)


def test_tex_escape():
    assert analysis.tex_escape("ddos_hoic 50% & more") == r"ddos\_hoic 50\% \& more"


def test_runs_never_planned_show_not_run(cfg, tmp_path):
    (tmp_path / "m.yaml").write_text("dataset_encodings:\n  edge_iiotset_raw: [e1_ppfle]\n")
    cfg["replication"]["planned_from"] = {"bert_small": str(tmp_path / "m.yaml")}
    skip = analysis.not_planned(cfg)
    assert ("bert_small", "e2_ppfle_bins", "edge_iiotset_raw") in skip
    assert ("bert_small", "e1_ppfle", "edge_iiotset_raw") not in skip
    assert not any(m == "xgboost" for m, _, _ in skip)


def test_encoding_examples_table(cfg, tmp_path):
    src = tmp_path / "enc.json"
    src.write_text(json.dumps({"example": {"e1_ppfle": "ab12 " * 100, "e4_sentence": "A short flow_x."}}))
    cfg["examples"].update(source=str(src), tex=str(tmp_path / "ex.tex"), max_chars=20)
    (path,) = analysis.encoding_examples(cfg)
    tex = path.read_text()
    assert r"\texttt{\small ab12 ab12 ab12 ab12 \dots}" in tex       # cut at a word boundary, typewriter font
    assert r"A short flow\_x." in tex and r"\dots" not in tex.split("E4 sentence")[1]
    cfg["examples"]["source"] = str(tmp_path / "missing.json")
    assert analysis.encoding_examples(cfg) == []
