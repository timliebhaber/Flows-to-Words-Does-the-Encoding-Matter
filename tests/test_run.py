"""Pipeline orchestrator (src/run.py): plan order, dependencies, filters, --force guard, execution, done hooks.

Planner tests use a fake pipeline whose hooks live in this module; no real step is ever run.
"""

import copy
import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src import run
from src.config import (
    ConfigError,
    load_bert_config,
    load_distilbert_config,
    load_model_config,
    load_pipeline_config,
)
from src.data import pipeline as data_pipeline
from src.data.processed import split_paths
from src.data.stats import dataset_stats_path
from src.eval import leakage
from src.models import text_runs, xgb
from tests.conftest import REPO_ROOT

PIPELINE_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"

# ---- fake pipeline: config path = stage name, DONE = finished (stage, unit) pairs ----

FAKE_VALUES = {"datasets": ["a", "b"], "encodings": ["e1", "e2"], "tasks": {"multi": {}, "bin": {}}}
DONE = set()


def fake_load(path):
    return {**FAKE_VALUES, "name": path}


def _key(name, unit):
    return name, tuple(sorted(unit.items()))


def fake_done(cfg, **unit):
    return _key(cfg["name"], unit) in DONE


def fake_pilot(cfg, units):
    """Like bert_small pilot tuning: every unit maps to dataset a / task multi."""
    return list({u["encoding"]: {**u, "dataset": "a", "task": "multi"} for u in units}.values())


def _stage(name, dims, depends_on, est, units=None, test_split=False, force_flag=None):
    return {"name": name, "module": f"fake.{name}", "config": name, "loader": "tests.test_run:fake_load",
            "args": ["--stage", name], "dims": dims, "units": units, "done": "tests.test_run:fake_done",
            "force_flag": force_flag, "test_split": test_split, "depends_on": depends_on, "est_minutes": est}


FAKE_PIPELINE = {
    "default_target": "all",
    "dims": {"dataset": "--dataset", "encoding": "--encoding", "task": "--task"},
    "show_command": ["uv", "run", "python", "-m"],
    "groups": {"model": ["fit", "tune", "final"], "all": ["prep", "fit", "tune", "final"]},
    "stages": [
        _stage("prep", ["dataset"], [], 1),
        _stage("report", [], ["prep"], 5),
        _stage("fit", ["encoding", "dataset"], ["prep"], {"e1": 2, "e2": 3}),
        _stage("tune", ["encoding", "dataset", "task"], ["fit"], 10, units="tests.test_run:fake_pilot"),
        _stage("final", ["encoding", "dataset", "task"], ["tune"], 20, test_split=True, force_flag="--force"),
    ],
}


@pytest.fixture()
def pcfg(tmp_path, monkeypatch):
    # the hooks are imported by name; make that name this already-loaded module (one shared DONE set)
    monkeypatch.setitem(sys.modules, "tests.test_run", sys.modules[__name__])
    DONE.clear()
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(FAKE_PIPELINE, sort_keys=False))
    return load_pipeline_config(path)


def _mark_done(pcfg, name, **filters):
    stage = next(s for s in pcfg["stages"] if s["name"] == name)
    for unit in run.candidate_units(stage, fake_load(name), filters):
        DONE.add(_key(name, unit))


def _plan(pcfg, targets=(), **kw):
    return run.build_plan(pcfg, list(targets), **kw)


def _names(plan):
    return [item["stage"]["name"] for item in plan]


# ---- the real configs/pipeline.yaml ----

def test_real_pipeline_config_is_consistent():
    pcfg = load_pipeline_config(PIPELINE_CONFIG)
    names = [s["name"] for s in pcfg["stages"]]
    assert "leakage" in names and "leakage" not in pcfg["groups"][pcfg["default_target"]]
    configs = run.StageConfigs()
    for stage in pcfg["stages"]:
        for key in ["loader", "done"] + (["units"] if stage["units"] else []):
            assert callable(run.resolve(stage[key]))
        module = importlib.import_module(stage["module"])
        # the CLI may live in a project module it delegates to (e.g. bert_small -> text_runs)
        used = [module] + [v for v in vars(module).values() if inspect.ismodule(v) and v.__name__.startswith("src.")]
        source = "".join(inspect.getsource(m) for m in used)
        for flag in [pcfg["dims"][d] for d in stage["dims"]] + [stage["force_flag"]] + stage["args"][:1]:
            if flag:
                assert f'"{flag}"' in source, f"{stage['module']} has no {flag} option"
        for unit in run.candidate_units(stage, configs.get(stage), {}):
            assert run.est_minutes(stage, unit) >= 0  # an estimate for every unit (e.g. every encoding)


def test_real_bert_tune_units_follow_pilot_scope():
    pcfg = load_pipeline_config(PIPELINE_CONFIG)
    stage = next(s for s in pcfg["stages"] if s["name"] == "bert_small_tune")
    cfg = load_bert_config(REPO_ROOT / "configs" / "bert_small.yaml")
    units = run.candidate_units(stage, cfg, {"dataset": cfg["data"]["datasets"][-1]})
    pilot = cfg["tuning"]["pilot"]
    assert len(units) == len(cfg["encodings"])
    assert all(u["dataset"] == pilot["dataset"] and u["task"] == pilot["task"] for u in units)


# ---- config validation ----

@pytest.mark.parametrize("change, message", [
    (lambda p: p["stages"][0]["depends_on"].append("fit"), "listed before"),
    (lambda p: p["groups"].update(prep=["prep"]), "same name as a stage"),
    (lambda p: p["stages"][1].update(done="no_colon"), "package.module:function"),
    (lambda p: p["groups"]["all"].append("nope"), "unknown stages"),
    (lambda p: p["stages"][0].pop("est_minutes"), "missing keys"),
])
def test_invalid_pipeline_config(tmp_path, change, message):
    bad = copy.deepcopy(FAKE_PIPELINE)
    change(bad)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ConfigError, match=message):
        load_pipeline_config(path)


# ---- planning ----

def test_plan_follows_stage_order_and_slices(pcfg):
    assert _names(_plan(pcfg)) == ["prep", "fit", "tune", "final"]  # default target, report not included
    assert _names(_plan(pcfg, ["final", "prep"])) == ["prep", "fit", "tune", "final"]
    _mark_done(pcfg, "prep")
    assert _names(_plan(pcfg, start="fit", stop="tune")) == ["fit", "tune"]
    with pytest.raises(run.PlanError, match="unknown target"):
        _plan(pcfg, ["nope"])
    with pytest.raises(run.PlanError, match="no stage selected"):
        _plan(pcfg, ["prep"], start="fit")


def test_unfinished_dependencies_are_added_and_labelled(pcfg):
    plan = {item["stage"]["name"]: item for item in _plan(pcfg, ["final"])}
    assert list(plan) == ["prep", "fit", "tune", "final"]
    assert plan["prep"]["needed_by"] == ["fit"] and plan["tune"]["needed_by"] == ["final"]
    assert plan["final"]["needed_by"] == []

    for name in ["prep", "fit", "tune"]:
        _mark_done(pcfg, name)
    assert _names(_plan(pcfg, ["final"])) == ["final"]  # finished dependencies are not added


def test_dependencies_are_checked_without_filters(pcfg):
    _mark_done(pcfg, "prep", dataset="a")
    plan = _plan(pcfg, ["fit"], filters={"dataset": "a"})
    prep = plan[0]
    assert prep["stage"]["name"] == "prep" and prep["pending"] == [{"dataset": "b"}]
    assert plan[1]["pending"] == [{"encoding": "e1", "dataset": "a"}, {"encoding": "e2", "dataset": "a"}]


def test_filters_apply_only_to_stages_with_that_dim(pcfg):
    for name in ["prep", "fit", "tune"]:
        _mark_done(pcfg, name)
    plan = {i["stage"]["name"]: i for i in _plan(pcfg, ["all"], filters={"encoding": "e1", "dataset": "b"})}
    assert plan["prep"]["units"] == [{"dataset": "b"}]
    assert plan["fit"]["units"] == [{"encoding": "e1", "dataset": "b"}]
    assert plan["tune"]["units"] == [{"encoding": "e1", "dataset": "a", "task": "multi"}]  # pilot hook
    assert plan["final"]["pending"] == [{"encoding": "e1", "dataset": "b", "task": t} for t in ["multi", "bin"]]
    with pytest.raises(run.PlanError, match="unknown dataset"):
        _plan(pcfg, filters={"dataset": "zzz"})


def test_force_on_test_stage_needs_explicit_target(pcfg):
    for name in ["prep", "fit", "tune", "final"]:
        _mark_done(pcfg, name)
    with pytest.raises(run.PlanError, match="re-evaluate the test set"):
        _plan(pcfg, force=True)
    with pytest.raises(run.PlanError, match="re-evaluate the test set"):
        _plan(pcfg, ["all"], start="tune", force=True)
    plan = _plan(pcfg, ["final"], force=True)
    assert _names(plan) == ["final"] and len(plan[0]["pending"]) == 8 and plan[0]["force"]
    assert _names(_plan(pcfg, ["model"], force=True)) == ["fit", "tune", "final"]
    assert len(_plan(pcfg, ["prep"], force=True)[0]["pending"]) == 2  # no test split: default rules suffice


def test_forced_stage_does_not_force_added_dependencies(pcfg):
    _mark_done(pcfg, "prep", dataset="a")
    plan = _plan(pcfg, ["fit"], force=True)
    assert [(i["stage"]["name"], i["force"], len(i["pending"])) for i in plan] == [("prep", False, 1),
                                                                                   ("fit", True, 4)]


def test_commands(pcfg):
    final = _plan(pcfg, ["final"], force=True)[-1]
    unit = {"encoding": "e1", "dataset": "a", "task": "bin"}
    assert run.module_args(pcfg, final, unit) == ["fake.final", "--config", "final", "--stage", "final",
                                                  "--encoding", "e1", "--dataset", "a", "--task", "bin", "--force"]
    report = run.build_plan(pcfg, ["report"])[-1]
    assert run.module_args(pcfg, report, {}) == ["fake.report", "--config", "report", "--stage", "report"]
    assert run.show(pcfg, ["m", "x y"]) == "uv run python -m m 'x y'"


def test_render_plan(pcfg):
    text = run.render_plan(pcfg, _plan(pcfg, ["fit"], filters={"encoding": "e2"}), {"encoding": "e2"})
    assert "added: needed by fit" in text and "~8 min" in text  # 2 prep x 1 + 2 fit x 3
    assert "uv run python -m fake.fit --config fit --stage fit --encoding e2 --dataset b" in text
    for name in ["prep", "fit"]:
        _mark_done(pcfg, name)
    assert "Nothing to do" in run.render_plan(pcfg, _plan(pcfg, ["fit"]), {})


# ---- execution ----

class Proc:
    def __init__(self, returncode):
        self.returncode = returncode


def test_dry_run_starts_no_process(pcfg, tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run started a process")

    monkeypatch.setattr(subprocess, "run", forbidden)
    path = tmp_path / "pipeline.yaml"
    assert run.main(["--config", str(path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Commands, in order:" in out and "fake.final" in out


def test_execute_skips_units_finished_meanwhile_and_stops_on_failure(pcfg):
    plan = _plan(pcfg, ["prep"]) + _plan(pcfg, ["report"])[-1:]
    calls, said = [], []

    def runner(cmd):
        calls.append(cmd)
        if len(calls) == 1:
            DONE.add(_key("prep", {"dataset": "b"}))  # e.g. finished by a parallel run
            return Proc(0)
        return Proc(3)

    code = run.execute(pcfg, plan, run.StageConfigs(), run=runner, say=said.append)
    assert code == 3
    assert [c[2:] for c in calls] == [["fake.prep", "--config", "prep", "--stage", "prep", "--dataset", "a"],
                                      ["fake.report", "--config", "report", "--stage", "report"]]
    assert any("already done, skipped" in s for s in said) and "FAILED" in said[-1]


# ---- the real done hooks, on temporary files ----

def test_data_done_hook(cfg, tmp_path):
    data = copy.deepcopy(cfg)
    data["paths"].update(processed_dir=str(tmp_path / "processed"), stats_dir=str(tmp_path / "stats"))
    ds = data["datasets"][0]
    assert not data_pipeline.is_done(data, ds)
    for p in split_paths(data, ds).values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    assert not data_pipeline.is_done(data, ds)  # stats still missing
    stats = dataset_stats_path(data["paths"]["stats_dir"], ds)
    stats.parent.mkdir(parents=True)
    stats.touch()
    assert data_pipeline.is_done(data, ds)


def _xgb_cfg(tmp_path):
    cfg = copy.deepcopy(load_model_config(REPO_ROOT / "configs" / "xgb.yaml"))
    cfg["outputs"].update(results_file=str(tmp_path / "results.csv"), tuning_file=str(tmp_path / "tuning.json"),
                          leakage_json=str(tmp_path / "leakage.json"))
    return cfg


def test_xgb_and_leakage_done_hooks(tmp_path):
    cfg = _xgb_cfg(tmp_path)
    ds = cfg["data"]["datasets"][0]
    binary = next(t for t, spec in cfg["tasks"].items() if spec["cross_dataset"])
    assert not xgb.tune_done(cfg, ds, binary) and not leakage.is_done(cfg, ds)
    Path(cfg["outputs"]["tuning_file"]).write_text(json.dumps({ds: {binary: {"best": {}}}}))
    assert xgb.tune_done(cfg, ds, binary)
    Path(cfg["outputs"]["leakage_json"]).write_text(json.dumps({ds: {t: {} for t in cfg["tasks"]}}))
    assert leakage.is_done(cfg, ds)

    ids = [xgb._run_id(cfg, binary, ds, t, s) for s in cfg["seeds"] for t in cfg["data"]["datasets"]]
    pd.DataFrame({"run_id": ids[:-1]}).to_csv(cfg["outputs"]["results_file"], index=False)
    assert not xgb.final_done(cfg, ds, binary)  # one seed x test set missing
    pd.DataFrame({"run_id": ids}).to_csv(cfg["outputs"]["results_file"], index=False)
    assert xgb.final_done(cfg, ds, binary)


def test_bert_small_done_hooks(tmp_path):
    cfg = copy.deepcopy(load_bert_config(REPO_ROOT / "configs" / "bert_small.yaml"))
    cfg["outputs"].update(results_file=str(tmp_path / "results.csv"), tuning_file=str(tmp_path / "tuning.json"))
    enc, ds, task = cfg["encodings"][0], cfg["data"]["datasets"][0], next(iter(cfg["tasks"]))
    Path(cfg["outputs"]["tuning_file"]).write_text(json.dumps({enc: {ds: {task: {"runs": {}, "best": None}}}}))
    assert not text_runs.tune_done(cfg, enc, ds, task)  # grid not finished
    ids = [text_runs._run_id(cfg, task, enc, ds, t, s) for s in cfg["seeds"]
           for t in text_runs._test_sets(cfg, ds, task)]
    pd.DataFrame({"run_id": ids[1:]}).to_csv(cfg["outputs"]["results_file"], index=False)
    assert not text_runs.final_seeds_done(cfg, enc, ds, task)
    pd.DataFrame({"run_id": ids}).to_csv(cfg["outputs"]["results_file"], index=False)
    assert text_runs.final_seeds_done(cfg, enc, ds, task)


def test_more_seeds_later_runs_only_new_seeds(tmp_path, monkeypatch):
    """1 seed now, more later (e.g. on the cluster): finished seed-0 runs are kept, only new seeds run."""
    cfg = copy.deepcopy(load_distilbert_config(REPO_ROOT / "configs" / "distilbert.yaml"))
    cfg["outputs"]["results_file"] = str(tmp_path / "results.csv")
    enc, ds = "e3_kv", cfg["data"]["datasets"][0]
    task = next(t for t, spec in cfg["tasks"].items() if spec["cross_dataset"])
    cfg["seeds"] = [0]
    ids = [text_runs._run_id(cfg, task, enc, ds, t, 0) for t in text_runs._test_sets(cfg, ds, task)]
    pd.DataFrame({"run_id": ids}).to_csv(cfg["outputs"]["results_file"], index=False)
    assert text_runs.final_seeds_done(cfg, enc, ds, task)

    cfg["seeds"] = [0, 1, 2]
    assert not text_runs.final_seeds_done(cfg, enc, ds, task)
    calls = []
    monkeypatch.setattr(text_runs, "final_one", lambda cfg, backend, e, d, t, s, cache: calls.append(s))
    text_runs.run_stages(cfg, None, "final", [(enc, ds, task)], force=False)
    assert calls == [1, 2]
