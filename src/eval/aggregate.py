"""Reading results.csv for the analysis and averaging runs over seeds (plan §7). Read-only.

Training runs may rewrite results.csv while the analysis reads it, so `load_results` checks the file did not
change during the read and that every row is complete; otherwise it asks for a rerun.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class ResultsReadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Stat:
    """A metric over seeds: mean, std (NaN for a single seed) and the number of seeds."""
    mean: float
    std: float
    n: int


def make_stat(values, ddof):
    values = np.asarray(values, dtype=float)
    std = float(np.std(values, ddof=ddof)) if len(values) > ddof else float("nan")
    return Stat(float(np.mean(values)), std, len(values))


def fmt_stat(stat, fmt, plus_minus=" ± ", marker=None):
    """'0.671 ± 0.004' for several seeds, '0.556†' for one seed, `fmt.pending` for no result."""
    if stat is None:
        return fmt["pending"]
    d = fmt["decimals"]
    if stat.n == 1:
        return f"{stat.mean:.{d}f}{fmt['one_seed_marker'] if marker is None else marker}"
    return f"{stat.mean:.{d}f}{plus_minus}{stat.std:.{d}f}"


def _signature(path):
    st = Path(path).stat()
    return st.st_mtime_ns, st.st_size


def load_results(cfg):
    """results.csv as a DataFrame (metrics as floats, seed as int). Raises ResultsReadError on a half-written file."""
    inp = cfg["inputs"]
    path = inp["results_file"]
    before = _signature(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if _signature(path) != before:
        raise ResultsReadError(f"{path} changed while it was read (a training run is writing it); rerun in a moment")
    missing = sorted(set(inp["required_columns"]) - set(df.columns))
    if missing:
        raise ResultsReadError(f"{path}: missing columns {missing}")
    if (df["run_id"] == "").any() or df["run_id"].duplicated().any():
        raise ResultsReadError(f"{path}: empty or duplicate run_id (half-written file?); rerun in a moment")
    metrics = sorted(set(cfg["within"]["metrics"]) | set(cfg["cross"]["metrics"]) | set(cfg["replication"]["metrics"]))
    for col in metrics:
        df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")
    seeds = pd.to_numeric(df["seed"], errors="coerce")
    if seeds.isna().any() or df[cfg["within"]["metrics"]].isna().any().any():
        raise ResultsReadError(f"{path}: incomplete rows (half-written file?); rerun in a moment")
    df["seed"] = seeds.astype(int)
    return df


def within_rows(df, cfg):
    """Exp 2 rows (+ the Edge-IIoTset exp1 rows): within-dataset, configured task and datasets."""
    w = cfg["within"]
    keep = (df["experiment"].isin(w["experiments"]) & (df["task"] == w["task"])
            & (df["train_dataset"] == df["test_dataset"]) & df["test_dataset"].isin(list(cfg["datasets"])))
    return df[keep]


def cross_rows(df, cfg):
    c = cfg["cross"]
    keep = ((df["experiment"] == c["experiment"]) & (df["task"] == c["task"])
            & df["train_dataset"].isin(c["datasets"]) & df["test_dataset"].isin(c["datasets"]))
    return df[keep]


def replication_rows(df, cfg):
    r = cfg["replication"]
    keep = ((df["experiment"] == r["experiment"]) & df["task"].isin(r["tasks"])
            & (df["train_dataset"] == df["test_dataset"]) & df["test_dataset"].isin(list(r["versions"])))
    return df[keep]


def cell_stats(rows, keys, metric, ddof):
    """{tuple(keys values): Stat of `metric` over the rows (seeds) of that group}."""
    return {k if isinstance(k, tuple) else (k,): make_stat(g[metric].dropna(), ddof)
            for k, g in rows.groupby(keys) if g[metric].notna().any()}


def cross_summary(rows, datasets, metric, ddof):
    """{(model, encoding): {"diag": Stat, "off": Stat}} for the cross-dataset grid.

    Per seed, average the diagonal (train = test) and the off-diagonal cells; then mean/std over seeds. A seed counts
    only when its grid is complete, so a half-finished grid never mixes into the average.
    """
    k = len(datasets)
    out = {}
    for (model, encoding), g in rows.groupby(["model", "encoding"]):
        per_seed = {"diag": [], "off": []}
        for _, s in g.groupby("seed"):
            s = s.dropna(subset=[metric])
            if len(s) != k * k or s[["train_dataset", "test_dataset"]].duplicated().any():
                continue
            same = s["train_dataset"] == s["test_dataset"]
            per_seed["diag"].append(s.loc[same, metric].mean())
            per_seed["off"].append(s.loc[~same, metric].mean())
        if per_seed["off"]:
            out[(model, encoding)] = {part: make_stat(v, ddof) for part, v in per_seed.items()}
    return out
