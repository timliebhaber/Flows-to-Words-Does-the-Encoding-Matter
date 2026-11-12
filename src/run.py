"""One entry point for the whole pipeline. Stages, order, "done" checks and time estimates: configs/pipeline.yaml.

Usage (from the repo root; `uv run ftw` is the same as `uv run python -m src.run`):
    uv run ftw --list                         # groups and stages
    uv run ftw --dry-run                      # plan: what is done, what is left, rough time, exact commands
    uv run ftw                                # run everything that is left (the config's default_target)
    uv run ftw xgb bert_small                 # only these groups/stages
    uv run ftw --from encoders --to tokenizers
    uv run ftw bert_small_final --encoding e3_kv --task binary --dry-run
    uv run ftw data --dataset nf_unsw_nb15 --force

Each pending unit (e.g. one dataset, or one encoding x dataset x task) runs as the documented per-module
command in a subprocess. Finished units are skipped, so after a crash the same command continues where it
stopped. Stages that a requested stage depends on are added automatically while they still have work to do.
"""

import argparse
import importlib
import itertools
import shlex
import subprocess
import sys
import time

from src.config import ConfigError, load_pipeline_config

# The one path in code (decided 2026-10-28): where the pipeline config lives. Everything else is in that file.
DEFAULT_CONFIG = "configs/pipeline.yaml"


class PlanError(ValueError):
    pass


def resolve(ref):
    """'package.module:function' -> the function."""
    module, func = ref.split(":")
    return getattr(importlib.import_module(module), func)


class StageConfigs:
    """Loads each stage's own config once (keyed by loader and path)."""

    def __init__(self):
        self._cache = {}

    def get(self, stage):
        key = (stage["loader"], stage["config"])
        if key not in self._cache:
            self._cache[key] = resolve(stage["loader"])(stage["config"])
        return self._cache[key]


def dim_values(cfg, dim):
    """Values of a unit dimension, read from a stage's own loaded config."""
    if dim == "dataset":
        return list(cfg.get("data", cfg)["datasets"])  # the data config has `datasets` itself, others under `data`
    if dim == "encoding":
        return list(cfg["encodings"])
    return list(cfg["tasks"])


# ---- planning (no side effects) ----

def select(pcfg, targets, start=None, stop=None):
    """Stage names requested by targets (groups or stages) and --from/--to, in stage order."""
    order = [s["name"] for s in pcfg["stages"]]
    names = set()
    for t in targets or [pcfg["default_target"]]:
        if t in pcfg["groups"]:
            names |= set(pcfg["groups"][t])
        elif t in order:
            names.add(t)
        else:
            raise PlanError(f"unknown target {t!r}; groups: {list(pcfg['groups'])}; stages: {order}")
    for flag, name in [("--from", start), ("--to", stop)]:
        if name and name not in order:
            raise PlanError(f"{flag} {name!r} is not a stage; stages: {order}")
    lo, hi = order.index(start) if start else 0, order.index(stop) if stop else len(order) - 1
    picked = [n for n in order[lo:hi + 1] if n in names]
    if not picked:
        raise PlanError("no stage selected (check the targets and --from/--to)")
    return picked


def explicit_stages(pcfg, targets):
    """Stages named on the command line, directly or through a group other than the default target."""
    out = set()
    for t in targets:
        if t in pcfg["groups"]:
            if t != pcfg["default_target"]:
                out |= set(pcfg["groups"][t])
        else:
            out.add(t)
    return out


def candidate_units(stage, cfg, filters):
    """Units of a stage after filters; a filter on a dimension the stage does not have is ignored."""
    axes = []
    for dim in stage["dims"]:
        values = dim_values(cfg, dim)
        chosen = filters.get(dim)
        if chosen:
            if chosen not in values:
                raise PlanError(f"stage {stage['name']}: unknown {dim} {chosen!r}; choose from {values}")
            values = [chosen]
        axes.append(values)
    units = [dict(zip(stage["dims"], combo)) for combo in itertools.product(*axes)]
    return resolve(stage["units"])(cfg, units) if stage["units"] else units


def unit_done(stage, cfg, unit):
    return bool(resolve(stage["done"])(cfg, **unit))


def est_minutes(stage, unit):
    est = stage["est_minutes"]
    if not isinstance(est, dict):
        return est
    hits = [est[v] for v in unit.values() if v in est]
    if len(hits) != 1:
        raise PlanError(f"stage {stage['name']}: est_minutes has no single entry for unit {unit}")
    return hits[0]


def _item(stage, configs, filters, force, needed_by):
    cfg = configs.get(stage)
    units = candidate_units(stage, cfg, filters)
    pending = list(units) if force else [u for u in units if not unit_done(stage, cfg, u)]
    return {"stage": stage, "units": units, "pending": pending, "force": force, "needed_by": needed_by}


def build_plan(pcfg, targets, start=None, stop=None, filters=None, force=False, configs=None):
    """Ordered plan items: the selected stages plus any unfinished stage they depend on."""
    stages = {s["name"]: s for s in pcfg["stages"]}
    configs = configs or StageConfigs()
    filters = filters or {}
    picked = select(pcfg, targets, start, stop)
    if force:
        named = explicit_stages(pcfg, targets)
        blocked = [n for n in picked if stages[n]["test_split"] and n not in named]
        if blocked:
            raise PlanError(f"--force would re-evaluate the test set in {blocked}; name these stages (or their "
                            "group) explicitly to confirm")
    items = {n: _item(stages[n], configs, filters, force, []) for n in picked}

    # Dependencies are checked without filters: e.g. binary runs test on every dataset's split.
    checked_done, queue = set(), list(picked)
    while queue:
        name = queue.pop(0)
        for dep in stages[name]["depends_on"]:
            if dep in items:
                if items[dep]["needed_by"] and name not in items[dep]["needed_by"]:
                    items[dep]["needed_by"].append(name)
                continue
            if dep in checked_done:
                continue
            item = _item(stages[dep], configs, {}, False, [name])
            if item["pending"]:
                items[dep] = item
                queue.append(dep)
            else:
                checked_done.add(dep)
    return [items[s["name"]] for s in pcfg["stages"] if s["name"] in items]


def module_args(pcfg, item, unit):
    """Arguments after `python -m` for one unit."""
    stage = item["stage"]
    flags = [x for dim in stage["dims"] for x in (pcfg["dims"][dim], unit[dim])]
    force = [stage["force_flag"]] if item["force"] and stage["force_flag"] else []
    return [stage["module"], "--config", stage["config"], *stage["args"], *flags, *force]


def show(pcfg, args):
    return shlex.join([*pcfg["show_command"], *args])


def fmt_minutes(m):
    if m == 0:
        return "-"
    if m < 1:
        return "<1 min"
    return f"~{m:.0f} min" if m < 90 else f"~{m / 60:.1f} h"


def fmt_unit(unit):
    return "/".join(str(v) for v in unit.values()) or "all"


def render_plan(pcfg, plan, filters):
    shown = ", ".join(f"{k}={v}" for k, v in filters.items() if v) or "none"
    lines = [f"Plan (filters: {shown})", f"  {'stage':<18} {'done':>7} {'todo':>5} {'est.':>9}  note"]
    total = 0
    for item in plan:
        stage, n_units, n_todo = item["stage"], len(item["units"]), len(item["pending"])
        minutes = sum(est_minutes(stage, u) for u in item["pending"])
        total += minutes
        notes = []
        if item["needed_by"]:
            notes.append(f"added: needed by {', '.join(item['needed_by'])}")
        if item["force"]:
            notes.append("--force: redo all" + ("; RE-EVALUATES THE TEST SET" if stage["test_split"] else ""))
        lines.append(f"  {stage['name']:<18} {f'{n_units - n_todo}/{n_units}':>7} {n_todo:>5} "
                     f"{fmt_minutes(minutes):>9}  {'; '.join(notes)}")
    todo = [(item, u) for item in plan for u in item["pending"]]
    if not todo:
        lines.append("Nothing to do: everything requested is already done.")
        return "\n".join(lines)
    lines.append(f"Total: {len(todo)} step(s), {fmt_minutes(total)} (rough, measured on the RTX A3000 laptop GPU)")
    lines.append("Commands, in order:")
    lines += [f"  {show(pcfg, module_args(pcfg, item, u))}" for item, u in todo]
    return "\n".join(lines)


def render_list(pcfg):
    lines = ["Groups:"]
    lines += [f"  {g:<12} {' '.join(m)}" + ("   (default)" if g == pcfg["default_target"] else "")
              for g, m in pcfg["groups"].items()]
    lines.append("Stages (in run order):")
    for s in pcfg["stages"]:
        dims = f"per {' x '.join(s['dims'])}" if s["dims"] else "one unit"
        after = f", after {', '.join(s['depends_on'])}" if s["depends_on"] else ""
        lines.append(f"  {s['name']:<18} {dims}{after}{'  [reads test split]' if s['test_split'] else ''}")
    return "\n".join(lines)


# ---- execution ----

def _say(msg):
    print(msg, flush=True)


def execute(pcfg, plan, configs, run=None, say=_say):
    """Run every pending unit in order; re-check "done" just before each one. Returns an exit code."""
    run = run or subprocess.run
    todo = [(item, u) for item in plan for u in item["pending"]]
    left = sum(est_minutes(item["stage"], u) for item, u in todo)
    t0 = time.time()
    for i, (item, unit) in enumerate(todo, 1):
        stage = item["stage"]
        tag = f"[{i}/{len(todo)}] {stage['name']} {fmt_unit(unit)}"
        left -= est_minutes(stage, unit)
        if not item["force"] and unit_done(stage, configs.get(stage), unit):
            say(f"{tag}: already done, skipped")
            continue
        args = module_args(pcfg, item, unit)
        say(f"{tag}: $ {show(pcfg, args)}")
        t1 = time.time()
        code = run([sys.executable, "-m", *args]).returncode
        if code != 0:
            say(f"{tag}: FAILED (exit code {code}). Fix the error and rerun the same command; "
                "finished steps are skipped.")
            return code
        say(f"{tag}: done in {fmt_minutes((time.time() - t1) / 60)}; elapsed {fmt_minutes((time.time() - t0) / 60)},"
            f" about {fmt_minutes(left)} left")
    say("All requested steps are done.")
    return 0


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=DEFAULT_CONFIG)
    known, _ = pre.parse_known_args(argv)
    try:
        pcfg = load_pipeline_config(known.config)
    except (ConfigError, OSError) as e:
        sys.exit(f"cannot load pipeline config: {e}")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", help=f"groups or stages (default: {pcfg['default_target']}); "
                                                   "see --list")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="pipeline config YAML")
    parser.add_argument("--from", dest="start", metavar="STAGE", help="first stage to consider")
    parser.add_argument("--to", dest="stop", metavar="STAGE", help="last stage to consider")
    for dim, flag in pcfg["dims"].items():
        parser.add_argument(flag, dest=dim, help=f"only this {dim} (stages without a {dim} ignore it)")
    parser.add_argument("--force", action="store_true", help="redo the selected stages even if done")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--list", action="store_true", help="list groups and stages and exit")
    args = parser.parse_args(argv)

    if args.list:
        _say(render_list(pcfg))
        return 0
    filters = {dim: getattr(args, dim) for dim in pcfg["dims"]}
    configs = StageConfigs()
    try:
        plan = build_plan(pcfg, args.targets, args.start, args.stop, filters, args.force, configs)
        _say(render_plan(pcfg, plan, filters))
    except PlanError as e:
        parser.error(str(e))
    if args.dry_run or not any(item["pending"] for item in plan):
        return 0
    return execute(pcfg, plan, configs)


if __name__ == "__main__":
    sys.exit(main())
