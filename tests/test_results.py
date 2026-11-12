import pandas as pd
import pytest

from src.eval.results import upsert_rows

COLUMNS = ["run_id", "macro_f1", "notes"]


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text(",".join(COLUMNS) + "\n")
    return path


def test_upsert_adds_and_replaces(csv):
    upsert_rows(csv, COLUMNS, [{"run_id": "a", "macro_f1": 0.5, "notes": ""},
                               {"run_id": "b", "macro_f1": 0.6, "notes": "x"}])
    upsert_rows(csv, COLUMNS, [{"run_id": "a", "macro_f1": 0.9, "notes": "rerun"}])
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    assert list(df.columns) == COLUMNS
    assert sorted(df["run_id"]) == ["a", "b"]
    assert df.set_index("run_id").loc["a", "macro_f1"] == "0.9"


def test_upsert_rejects_bad_rows_and_header(csv, tmp_path):
    with pytest.raises(ValueError):
        upsert_rows(csv, COLUMNS, [{"run_id": "a", "macro_f1": 0.5}])
    with pytest.raises(ValueError):
        upsert_rows(csv, ["run_id", "other"], [{"run_id": "a", "other": 1}])


def _write_many(csv, worker, n):
    for i in range(n):
        upsert_rows(csv, COLUMNS, [{"run_id": f"w{worker}-{i}", "macro_f1": 0.5, "notes": ""}])


def test_upsert_parallel_writers_lose_nothing(csv):
    import multiprocessing as mp
    procs = [mp.get_context("fork").Process(target=_write_many, args=(csv, w, 25)) for w in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    assert len(df) == 100 and df["run_id"].is_unique


def test_real_results_header_matches_config():
    from src.config import load_model_config
    from tests.conftest import REPO_ROOT
    cfg = load_model_config(REPO_ROOT / "configs" / "xgb.yaml")
    header = (REPO_ROOT / cfg["outputs"]["results_file"]).read_text().splitlines()[0].split(",")
    assert header == cfg["results_columns"]
