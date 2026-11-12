"""Rule-based answers to RQ1 and H1-H4 (plan §2) from the aggregated results.

Rules (thresholds in configs/analysis.yaml `hypotheses`): two cells differ only if |delta| > max(tie_margin, the
larger seed std of the two); otherwise they tie. Each hypothesis is a list of checks; all hold -> "supported",
none -> "not supported", else "mixed". Checks that use a single-seed cell make the verdict "tentative".
"""

import math

from src.eval.aggregate import fmt_stat


def compare(a, b, margin):
    """+1 if a is clearly larger than b, -1 if clearly smaller, 0 for a tie."""
    threshold = max([margin] + [s for s in (a.std, b.std) if not math.isnan(s)])
    delta = a.mean - b.mean
    return 0 if abs(delta) <= threshold else (1 if delta > 0 else -1)


def verdict(checks):
    if not checks:
        return "no data yet"
    held = sum(ok for _, ok in checks)
    if held == len(checks):
        return "supported"
    if held == 0:
        return "not supported"
    return f"mixed ({held} of {len(checks)} checks hold)"


def method_labels(cfg):
    return {(m["model"], m["encoding"]): m["label"] for m in cfg["methods"]}


def _baseline_key(cfg):
    model = cfg["hypotheses"]["baseline_model"]
    return next((m["model"], m["encoding"]) for m in cfg["methods"] if m["model"] == model)


def _dataset_cells(cfg, within, dataset):
    labels = method_labels(cfg)
    return {(m, e): s for (m, e, d), s in within.items() if d == dataset and (m, e) in labels}


def _result(hid, claim, checks, tentative, notes=()):
    return {"id": hid, "claim": claim, "checks": checks, "verdict": verdict(checks),
            "tentative": tentative, "notes": list(notes)}


def h1(cfg, within):
    """PPFLE does well within one dataset: best E1 model within `well_margin` of the best method, per dataset."""
    hyp, fmt, labels = cfg["hypotheses"], cfg["format"], method_labels(cfg)
    checks, notes, tentative = [], [], False
    for dataset, name in cfg["datasets"].items():
        cells = _dataset_cells(cfg, within, dataset)
        e1 = {k: s for k, s in cells.items() if k[1] == hyp["ppfle_encoding"] and k[0] in hyp["text_models"]}
        if not e1:
            continue
        best_e1 = max(e1, key=lambda k: e1[k].mean)
        best = max(cells, key=lambda k: cells[k].mean)
        gap = cells[best].mean - e1[best_e1].mean
        ok = gap <= hyp["well_margin"]
        checks.append(((f"{name}: best E1 = {labels[best_e1]} {fmt_stat(e1[best_e1], fmt)}, best method = "
                       f"{labels[best]} {fmt_stat(cells[best], fmt)}, gap {gap:.{fmt['decimals']}f} "
                       f"({'≤' if ok else '>'} {hyp['well_margin']})"), ok))
        tentative |= min(e1[best_e1].n, cells[best].n) == 1
        for model in hyp["text_models"]:
            own = sorted((s.mean, e) for (m, e), s in cells.items() if m == model)
            if (model, hyp["ppfle_encoding"]) in cells:
                rank = len(own) - [e for _, e in own].index(hyp["ppfle_encoding"])
                notes.append(f"{name}: E1 ranks {rank} of {len(own)} encodings for {cfg['models'][model]}")
    return _result("H1", "PPFLE does well within one dataset.", checks, tentative, notes)


def h2(cfg, within):
    """XGBoost is as good or better within one dataset: XGBoost >= best text model (or tied), per dataset."""
    hyp, fmt, labels = cfg["hypotheses"], cfg["format"], method_labels(cfg)
    base = _baseline_key(cfg)
    checks, tentative = [], False
    for dataset, name in cfg["datasets"].items():
        cells = _dataset_cells(cfg, within, dataset)
        text = {k: s for k, s in cells.items() if k[0] in hyp["text_models"]}
        if base not in cells or not text:
            continue
        best = max(text, key=lambda k: text[k].mean)
        c = compare(cells[base], text[best], hyp["tie_margin"])
        word = {1: "better", 0: "ties", -1: "worse"}[c]
        checks.append(((f"{name}: {labels[base]} {fmt_stat(cells[base], fmt)} vs best text model {labels[best]} "
                       f"{fmt_stat(text[best], fmt)}: {labels[base]} {word} "
                       f"(delta {cells[base].mean - text[best].mean:+.{fmt['decimals']}f})"), c >= 0))
        tentative |= min(cells[base].n, text[best].n) == 1
    return _result("H2", "XGBoost is as good or better within one dataset.", checks, tentative)


def h3(cfg, cross):
    """Readable or binned encodings transfer better than raw PPFLE: off-diagonal macro-F1 vs E1, per text model."""
    hyp, fmt, labels = cfg["hypotheses"], cfg["format"], method_labels(cfg)
    primary = cfg["cross"]["metrics"][0]
    off = {k: v["off"] for k, v in cross[primary].items()}
    checks, notes, tentative = [], [], False
    for model in hyp["text_models"]:
        e1 = off.get((model, hyp["ppfle_encoding"]))
        if e1 is None:
            continue
        for alt in hyp["alternative_encodings"]:
            s = off.get((model, alt))
            if s is None:
                continue
            c = compare(s, e1, hyp["tie_margin"])
            word = {1: "better", 0: "tie", -1: "worse"}[c]
            checks.append(((f"{labels[(model, alt)]} {fmt_stat(s, fmt)} vs E1 {fmt_stat(e1, fmt)}: {word} "
                           f"(delta {s.mean - e1.mean:+.{fmt['decimals']}f})"), c > 0))
            tentative |= min(s.n, e1.n) == 1
    base = _baseline_key(cfg)
    if base in off:
        notes.append(f"Reference: {labels[base]} off-diagonal {cfg['format']['metric_labels'][primary]} "
                     f"{fmt_stat(off[base], fmt)}")
    for metric in cfg["cross"]["metrics"][1:]:
        parts = [f"{labels[k]} {fmt_stat(v['off'], fmt)}" for k, v in cross[metric].items() if k in labels]
        notes.append(f"Off-diagonal {cfg['format']['metric_labels'][metric]}: " + "; ".join(parts))
    return _result("H3", "Readable (key=value, sentence) or grouped (binned) encodings transfer better across "
                   "datasets than raw PPFLE.", checks, tentative, notes)


def h4(cfg, within, cross):
    """DistilBERT gains on readable encodings but not on hashes: DistilBERT - small BERT, within and cross."""
    hyp, fmt = cfg["hypotheses"], cfg["format"]
    pre, scratch = hyp["pretrained_model"], hyp["scratch_model"]
    primary = cfg["cross"]["metrics"][0]
    scopes = [(name, {(m, e): s for (m, e, d), s in within.items() if d == dataset})
              for dataset, name in cfg["datasets"].items()]
    scopes.append(("cross-dataset (off-diagonal)", {k: v["off"] for k, v in cross[primary].items()}))
    checks, tentative = [], False
    for name, cells in scopes:
        for enc in hyp["readable_encodings"] + hyp["hashed_encodings"]:
            a, b = cells.get((pre, enc)), cells.get((scratch, enc))
            if a is None or b is None:
                continue
            c = compare(a, b, hyp["tie_margin"])
            readable = enc in hyp["readable_encodings"]
            ok = c > 0 if readable else c <= 0
            word = {1: "gain", 0: "tie", -1: "loss"}[c]
            want = "expect a gain" if readable else "expect no gain"
            checks.append(((f"{name}, {cfg['encodings'][enc]} ({want}): {cfg['models'][pre]} {fmt_stat(a, fmt)} vs "
                           f"{cfg['models'][scratch]} {fmt_stat(b, fmt)}: {word} "
                           f"(delta {a.mean - b.mean:+.{fmt['decimals']}f})"), ok))
            tentative |= min(a.n, b.n) == 1
    return _result("H4", "The pretrained DistilBERT benefits from readable encodings but gains nothing on hashes.",
                   checks, tentative)


def rq1(cfg, replication):
    """Replication: our E1 small BERT accuracy vs the paper's, on each data version."""
    rep, fmt = cfg["replication"], cfg["format"]
    paper = rep["paper"]
    checks, notes, tentative = [], [], False
    for version, vname in rep["versions"].items():
        stats = replication.get((paper["model"], paper["encoding"], version, paper["task"]))
        if stats is None:
            continue
        acc = stats["accuracy"]
        diff = acc.mean - paper["accuracy"]
        line = (f"{vname}: accuracy {fmt_stat(acc, fmt)} vs paper {paper['accuracy']} "
                f"(diff {diff:+.{fmt['decimals']}f})")
        if version == paper["version"]:
            checks.append((line + f", reproduced if |diff| ≤ {paper['close_margin']}",
                           abs(diff) <= paper["close_margin"]))
            tentative |= acc.n == 1
        else:
            notes.append(line)
    for (model, encoding, version, task), stats in sorted(replication.items()):
        if model == cfg["hypotheses"]["baseline_model"]:
            notes.append(f"{cfg['models'][model]} {task}, {rep['versions'][version]}: accuracy "
                         f"{fmt_stat(stats['accuracy'], fmt)}, macro-F1 {fmt_stat(stats['macro_f1'], fmt)}")
    return _result("RQ1", f"{paper['label']} reaches {paper['accuracy']} accuracy on Edge-IIoTset; does it "
                   "reproduce?", checks, tentative, notes)


def evaluate(cfg, within, cross, replication):
    """All answers in order: RQ1, H1-H4. `within`: {(model, enc, dataset): Stat} of the primary metric;
    `cross`: {metric: {(model, enc): {"diag", "off"}}}; `replication`: {(model, enc, dataset, task): {metric: Stat}}."""
    return [rq1(cfg, replication), h1(cfg, within), h2(cfg, within), h3(cfg, cross), h4(cfg, within, cross)]


def render_summary(cfg, answers, sources):
    hyp, fmt = cfg["hypotheses"], cfg["format"]
    lines = ["# Analysis summary: RQ1 and H1-H4", "",
             "Generated by `src/eval/analysis.py` from `results/results.csv`; rerun it after new results.", "",
             (f"Rules: two numbers differ only if |delta| > max({hyp['tie_margin']}, the larger seed std of the two); "
             f"otherwise they tie. Cells marked {fmt['one_seed_marker']} come from one seed (no std), so verdicts that "
             f"use them are tentative. Within-dataset numbers are test macro-F1 (multiclass), cross-dataset numbers "
             f"are the mean off-diagonal binary macro-F1."), ""]
    for a in answers:
        tag = " (tentative: uses 1-seed results)" if a["tentative"] else ""
        lines += [f"## {a['id']}: {a['claim']}", "", f"**Verdict: {a['verdict']}**{tag}", ""]
        lines += [f"- {'✓' if ok else '✗'} {text}" for text, ok in a["checks"]]
        if a["notes"]:
            lines += ["", "Context:", ""] + [f"- {n}" for n in a["notes"]]
        lines.append("")
    lines += ["## Sources", ""] + [f"- {s}" for s in sources] + [""]
    return "\n".join(lines)
