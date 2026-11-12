"""Writing results: one row per run in results/results.csv, plus per-run detail JSON."""

import json
import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

try:
    import fcntl
except ImportError:          # Windows: no fcntl; runs there are single-process, so no lock is needed
    fcntl = None


@contextmanager
def _locked(path):
    """Exclusive lock on <path>.lock, so parallel jobs (e.g. cluster jobs) never lose each other's rows."""
    if fcntl is None:
        yield
        return
    with open(f"{path}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def upsert_rows(path, columns, rows):
    """Add `rows` (dicts with exactly `columns`) to the CSV; a row with an existing run_id replaces it.

    Read-modify-write happens under a lock, and the file is replaced atomically, so readers never see half a file."""
    for row in rows:
        if set(row) != set(columns):
            raise ValueError(f"row keys differ from results_columns: {sorted(set(row) ^ set(columns))}")
    new = pd.DataFrame(rows, columns=columns).astype(str)
    if new["run_id"].duplicated().any():
        raise ValueError("duplicate run_id within the new rows")
    with _locked(path):
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        if list(existing.columns) != list(columns):
            raise ValueError(f"{path}: header {list(existing.columns)} differs from config results_columns")
        kept = existing[~existing["run_id"].isin(new["run_id"])]
        tmp = f"{path}.tmp{os.getpid()}"
        pd.concat([kept, new], ignore_index=True).to_csv(tmp, index=False)
        os.replace(tmp, path)


def results_run_ids(path):
    """Set of run_ids already in the results CSV (empty if the file does not exist)."""
    path = Path(path)
    return set(pd.read_csv(path, usecols=["run_id"], dtype=str)["run_id"]) if path.exists() else set()


def write_run_details(directory, run_id, details):
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = Path(directory) / f"{run_id}.json"
    path.write_text(json.dumps(details, indent=2) + "\n")
    return path
