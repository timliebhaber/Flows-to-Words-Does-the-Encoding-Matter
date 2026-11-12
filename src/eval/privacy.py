"""Experiment 4 (plan §7.2, RQ4): dictionary attack on PPFLE tokens.

The attacker hashes guessed `column$value` strings exactly as the encoder does and matches them against the tokens
of each dataset's test split. Three cumulative levels (configs/privacy.yaml):
  L1 enumerate every value of the low-cardinality columns (ports, protocol, flags, nDPI codes, ...), the NA token
     and, for E2, the bin labels bin0..bin{n-1};
  L2 brute-force every other integer column over a fixed range;
  L3 a public-data dictionary: every value seen in the other NetFlow datasets.
Reports the share of test tokens and of whole rows recovered, per column, level and encoding. A recovered E1 token
reveals the exact value; a recovered E2 bin token reveals the bin index (a quantile range of the private train data).
Every recovered token is checked against the true value. CPU only, ~30 s per dataset.

Usage (from the repo root):
    uv run python -m src.eval.privacy --config configs/privacy.yaml [--dataset nf_unsw_nb15] [--render-only]
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_privacy_config
from src.data.processed import split_paths
from src.encoders import encode_split, encoding_cfg, kind
from src.encoders.common import is_missing, raw_text
from src.encoders.ppfle_binned import bin_index, load_state, numeric_columns
from src.eval import figures
from src.eval.numbers import Numbers, tex_escape

log = logging.getLogger("privacy")


# ---- candidates ----

def int_range(lo, hi, is_float):
    """Every integer lo..hi spelled as the encoder stores it (6 or 6.0)."""
    return (str(float(v)) for v in range(lo, hi + 1)) if is_float else map(str, range(lo, hi + 1))


def domain(spec, is_float):
    """L1 candidates of one low-cardinality column: the union of its spec's kinds (configs/privacy.yaml)."""
    out = set()
    if "range" in spec:
        out.update(int_range(*spec["range"], is_float))
    if "dotted_range" in spec:
        lo, hi = spec["dotted_range"]
        out.update(str(float(f"{m}.{s}")) for m in range(lo, hi + 1) for s in range(lo, hi + 1))
    if "hex_range" in spec:
        lo, hi, width = spec["hex_range"]
        out.update(f"0x{v:0{width}x}" for v in range(lo, hi + 1))
    out.update(spec.get("values", []))
    return out


def find(column, candidates, targets, ecfg):
    """{token: plaintext} for every candidate whose hash, computed like the encoder's, is one of `targets`."""
    fmt, h = ecfg["format"], ecfg["hash"]
    base = hashlib.new(h["algorithm"], f"{column}{fmt['separator']}".encode())
    found = {}
    if not targets:
        return found
    for text in candidates:
        d = base.copy()
        d.update(text.encode())
        token = d.hexdigest()[:h["length"]]
        if token in targets:
            found[token] = text
    return found


def truth_texts(df, column, ecfg, state):
    """The plaintext behind each row's token: the stored value (E1, E2 categorical) or the bin label (E2 numeric)."""
    na = ecfg["format"]["na_token"]
    values = df[column].tolist()
    if state is not None and column in state["edges"]:
        prefix = ecfg["binning"]["bin_prefix"]
        bins = bin_index(values, state["edges"][column]).tolist()
        return [na if is_missing(v) else f"{prefix}{b}" for v, b in zip(values, bins)]
    return [raw_text(v, na) for v in values]


def public_values(cfg, dataset):
    """L3 dictionary: {column: set of plaintexts} from the public split of every OTHER public dataset."""
    features, na = cfg["data"]["columns"]["features"], cfg["encoder"]["format"]["na_token"]
    out = {c: set() for c in features}
    for other in cfg["public_datasets"]:
        if other == dataset:
            continue
        df = pd.read_parquet(split_paths(cfg["data"], other)[cfg["public_split"]], columns=features)
        for c in features:
            out[c].update(raw_text(v, na) for v in df[c].drop_duplicates().tolist())
    return out


# ---- attack ----

def binned_columns(cfg, encoding):
    """Columns whose tokens hide a bin label instead of the value (E2 numeric columns); empty for E1."""
    enc = cfg["encoder"]
    return set(numeric_columns(encoding_cfg(enc, encoding))) if kind(enc, encoding) == "ppfle_binned" else set()


def candidates(cfg, ecfg, column, level, binned, public):
    """Plaintext guesses for one column at one attack level (`levels` in configs/privacy.yaml)."""
    cols = cfg["data"]["columns"]
    is_float, low = column in cols["float_features"], cfg["low_cardinality"]
    if level == "enumerate":
        bins = [f"{ecfg['binning']['bin_prefix']}{k}" for k in range(ecfg["binning"]["n_bins"])]
        return ([ecfg["format"]["na_token"]] + (bins if column in binned else [])
                + (list(domain(low[column], is_float)) if column in low else []))
    if level == "bruteforce":
        # every other numeric column, spelled like its type (Edge-IIoTset stores integers as floats: "6.0")
        skip = column in binned or column in cols["string_features"] or column in low
        return [] if skip else int_range(*cfg["bruteforce_range"], is_float)
    return [] if column in binned else public[column]


def attack(cfg, dataset, encoding, df, public):
    """Shares recovered per level for one dataset x encoding (`df` = the target split)."""
    enc = cfg["encoder"]
    ecfg = encoding_cfg(enc, encoding)
    features = cfg["data"]["columns"]["features"]
    binned = binned_columns(cfg, encoding)
    state = load_state(ecfg, dataset) if binned else None
    tokens = np.array([t.split(ecfg["format"]["token_joiner"]) for t in encode_split(enc, encoding, dataset, df)])
    if tokens.shape[1] != len(features):
        raise ValueError(f"{encoding}: expected {len(features)} tokens per row, got {tokens.shape[1]}")

    levels = [lv["name"] for lv in cfg["levels"]]
    recovered = {lv: np.zeros(tokens.shape, dtype=bool) for lv in levels}
    wrong = 0
    for i, col in enumerate(features):
        targets, found = set(tokens[:, i]), {}
        truth = np.array(truth_texts(df, col, ecfg, state))
        for lv in levels:
            found.update(find(col, candidates(cfg, ecfg, col, lv, binned, public), targets - found.keys(), ecfg))
            hit = np.isin(tokens[:, i], list(found))
            recovered[lv][:, i] = hit
        guessed = np.array([found.get(t, "") for t in tokens[:, i]])
        wrong += int(np.sum(hit & (guessed != truth)))
    out = {"levels": {}, "wrong": wrong}
    for lv in levels:
        r = recovered[lv]
        out["levels"][lv] = {"token_share": float(r.mean()), "row_share": float(r.all(axis=1).mean()),
                             "columns": {c: float(r[:, i].mean()) for i, c in enumerate(features)}}
    return out


def run_dataset(cfg, dataset):
    features = cfg["data"]["columns"]["features"]
    df = pd.read_parquet(split_paths(cfg["data"], dataset)[cfg["target_split"]], columns=features)
    public = public_values(cfg, dataset) if "public" in [lv["name"] for lv in cfg["levels"]] else {}
    result = {"n_rows": len(df), "encodings": {}}
    for encoding in cfg["encodings"]:
        t0 = time.time()
        r = attack(cfg, dataset, encoding, df, public)
        r["seconds"] = round(time.time() - t0, 1)
        if r["wrong"]:
            raise RuntimeError(f"{dataset}/{encoding}: {r['wrong']} recovered tokens disagree with the true value")
        result["encodings"][encoding] = r
        log.info("[%s/%s] %.0fs; tokens recovered per level: %s", dataset, encoding, r["seconds"],
                 {lv: round(v["token_share"], 3) for lv, v in r["levels"].items()})
    return result


# ---- outputs ----

def _label(cfg, dataset):
    """Display name: `dataset_labels` of the privacy config, else the analysis config's dataset name."""
    return cfg["dataset_labels"].get(dataset, cfg["display"]["datasets"].get(dataset, dataset))


def pooled_columns(cfg, results, encoding, level):
    """Per-column share pooled over datasets (weighted by rows)."""
    rows = {d: r["n_rows"] for d, r in results.items()}
    total = sum(rows.values())
    return {c: sum(r["encodings"][encoding]["levels"][level]["columns"][c] * rows[d] for d, r in results.items()) / total
            for c in cfg["data"]["columns"]["features"]}


def render(cfg, results):
    """Markdown, LaTeX and CSV tables + figures from all datasets in `results`."""
    out, disp = cfg["outputs"], cfg["display"]
    levels = cfg["levels"]
    records = [{"dataset": d, "encoding": e, "level": lv["name"], "token_share": r["encodings"][e]["levels"][lv["name"]]["token_share"],
                "row_share": r["encodings"][e]["levels"][lv["name"]]["row_share"], "n_rows": r["n_rows"]}
               for d, r in results.items() for e in cfg["encodings"] for lv in levels]
    stem = Path(out["table"])
    stem.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(stem.with_suffix(".csv"), index=False)

    head = ["Dataset", "Encoding"] + [f"{lv['name']}: {k}" for lv in levels for k in ("% tokens", "% rows")]
    pd_ = cfg["numbers"]["percent_decimals"]
    shares = ("token_share", "row_share")
    rows = [[_label(cfg, d), disp["encodings"].get(e, e)]
            + [f"{100 * r['encodings'][e]['levels'][lv['name']][k]:.{pd_}f}" for lv in levels for k in shares]
            for d, r in results.items() for e in cfg["encodings"]]
    tex_rows = [[tex_escape(_label(cfg, d)), tex_escape(disp["encodings"].get(e, e))]
                + [rf"\resor{{{cfg['numbers']['prefix']}.{d}.{e}.{lv['name']}.{k}}}{{}}" for lv in levels for k in shares]
                for d in results for e in cfg["encodings"]]
    level_text = "; ".join(lv["label"] for lv in levels)
    note = (f"Levels are cumulative ({level_text}). % rows = rows whose every token is recovered. A recovered E1 token "
            "is the exact value; a recovered E2 token is the bin index (a quantile range of the private train data). "
            "Every recovered token was checked against the true value (0 wrong).")
    md = ["# Experiment 4: dictionary attack on PPFLE (test split)", "", note, "",
          "| " + " | ".join(head) + " |", "|" + "|".join(["---", "---"] + ["---:"] * (len(head) - 2)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    per_col = {f"{disp['encodings'].get(e, e)} {lv['name']}": pooled_columns(cfg, results, e, lv["name"])
               for e in cfg["encodings"] for lv in levels}
    md += ["", "## Share of test tokens recovered per column (pooled over datasets, weighted by rows)", "",
           "| Column | " + " | ".join(per_col) + " |", "|" + "|".join(["---"] + ["---:"] * len(per_col)) + "|"]
    md += [f"| {c} | " + " | ".join(f"{100 * v[c]:.1f}" for v in per_col.values()) + " |"
           for c in cfg["data"]["columns"]["features"]]
    stem.with_suffix(".md").write_text("\n".join(md + [""]))
    tex = [("% Experiment 4: dictionary attack on PPFLE: generated by src/eval/privacy.py, do not edit. Cells are \\resor "
           f"macros (values in {Path(cfg['numbers']['tex']).name})."),
           r"\begin{tabular}{ll" + "r" * (len(head) - 2) + "}", r"\toprule",
           " & ".join(tex_escape(h) for h in head) + r" \\", r"\midrule"]
    tex += [" & ".join(r) + r" \\" for r in tex_rows]
    Path(out["tex"]).parent.mkdir(parents=True, exist_ok=True)
    Path(out["tex"]).write_text("\n".join(tex + [r"\bottomrule", r"\end{tabular}", ""]))
    write_numbers(cfg, results)
    return plot(cfg, results, per_col)


def write_numbers(cfg, results):
    """Every attack result as a report macro (<p> = numbers.prefix): <p>.<dataset>.<encoding>.<level>.<token_share|row_share> (in %),
    the min/max over datasets (<p>.<encoding>.<level>.<share>.min/max), per-column shares pooled over datasets
    (<p>.col.<encoding>.<level>.<column>) and the attack settings."""
    num, p = Numbers(cfg["numbers"]), cfg["numbers"]["prefix"]
    shares = ("token_share", "row_share")
    for d, r in results.items():
        num.value(f"{p}.{d}.n_rows", r["n_rows"])
        for e, er in r["encodings"].items():
            num.value(f"{p}.{d}.{e}.wrong", er["wrong"])
            num.value(f"{p}.{d}.{e}.seconds", er["seconds"])
            for lv, lr in er["levels"].items():
                for s in shares:
                    num.percent(f"{p}.{d}.{e}.{lv}.{s}", lr[s])
    for e in cfg["encodings"]:
        for lv in cfg["levels"]:
            for s in shares:
                values = [r["encodings"][e]["levels"][lv["name"]][s] for r in results.values() if e in r["encodings"]]
                if values:
                    num.percent(f"{p}.{e}.{lv['name']}.{s}.min", min(values))
                    num.percent(f"{p}.{e}.{lv['name']}.{s}.max", max(values))
            for c, v in pooled_columns(cfg, results, e, lv["name"]).items():
                num.percent(f"{p}.col.{e}.{lv['name']}.{c}", v)
            num.text(f"{p}.level.{lv['name']}.label", lv["label"])
    num.value(f"{p}.n_datasets", len(results))
    num.value(f"{p}.n_low_cardinality", len(cfg["low_cardinality"]))
    num.value(f"{p}.bruteforce.min", cfg["bruteforce_range"][0])
    num.value(f"{p}.bruteforce.max", cfg["bruteforce_range"][1])
    num.write("Privacy attack numbers (src/eval/privacy.py)")


def plot(cfg, results, per_col):
    fc, disp, style = cfg["figures"], cfg["display"], cfg["level_style"]
    figures.setup(fc)
    features = cfg["data"]["columns"]["features"]
    values = np.array([[v[c] for v in per_col.values()] for c in features])
    d = fc["annot_decimals"]
    annot = [[f"{x:.{d}f}" for x in row] for row in values]
    cw, ch = fc["sizes"]["cell"]
    mw, mh = fc["sizes"]["margins"]
    fig, ax = plt.subplots(figsize=(mw + cw * len(per_col), mh + ch * len(features)))
    img = figures.heatmap(ax, values, annot, features, list(per_col), fc)
    ax.set_title("Share of test tokens recovered, per column")
    fig.colorbar(img, ax=ax, fraction=0.05, pad=0.02)
    stems = [f"{cfg['outputs']['figure']}_columns"]
    figures.save(fig, fc["dir"], stems[0], fc)

    labels = [f"{_label(cfg, ds)}, {disp['encodings'].get(e, e)}" for ds in results for e in cfg["encodings"]]
    w, h = fc["sizes"]["offdiag_panel"]
    fig, ax = plt.subplots(figsize=(w * 2, h))
    for k, lv in enumerate(cfg["levels"]):
        xs = [100 * results[ds]["encodings"][e]["levels"][lv["name"]]["token_share"] for ds in results for e in cfg["encodings"]]
        offset = (k - (len(cfg["levels"]) - 1) / 2) * fc["style"]["dodge"]
        ax.plot(xs, [y + offset for y in range(len(xs))], linestyle="", marker=style["markers"][k], color=style["colors"][k],
                markersize=fc["marker_size"], label=lv["label"])
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(cfg["outputs"]["label"])
    ax.grid(axis="x", color=fc["grid"], linewidth=fc["style"]["line_width"])
    ax.set_axisbelow(True)
    fig.legend(loc="lower center", ncol=len(cfg["levels"]), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    stems.append(f"{cfg['outputs']['figure']}_overall")
    figures.save(fig, fc["dir"], stems[1], fc)
    return stems


def load_results(cfg):
    path = Path(cfg["outputs"]["json"])
    return json.loads(path.read_text()) if path.exists() else {}


def is_done(cfg, dataset):
    """`dataset` has attack results for every configured encoding (used by src/run.py)."""
    done = load_results(cfg).get(dataset, {}).get("encodings", {})
    return all(e in done for e in cfg["encodings"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="privacy config YAML, e.g. configs/privacy.yaml")
    parser.add_argument("--dataset", help="one dataset (default: all in the data config)")
    parser.add_argument("--render-only", action="store_true",
                        help="only rewrite tables, numbers and figures from the saved JSON (no attack)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    cfg = load_privacy_config(args.config)
    datasets = cfg["data"]["datasets"]
    if args.dataset and args.dataset not in datasets:
        parser.error(f"unknown dataset {args.dataset!r}; choose from {datasets}")
    results = load_results(cfg)
    path = Path(cfg["outputs"]["json"])
    for dataset in [] if args.render_only else [args.dataset] if args.dataset else datasets:
        results[dataset] = run_dataset(cfg, dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=1) + "\n")
    ordered = {d: results[d] for d in datasets if d in results}
    stems = render(cfg, ordered)
    log.info("wrote %s.{md,csv}, %s, %s and figures %s", cfg["outputs"]["table"], cfg["outputs"]["tex"],
             cfg["numbers"]["tex"], stems)


if __name__ == "__main__":
    main()
