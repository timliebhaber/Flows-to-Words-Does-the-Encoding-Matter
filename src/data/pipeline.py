"""Raw CSV -> cleaned, sampled, deduplicated, split Parquet + statistics (plan §4).

Usage (from the repo root):
    uv run python -m src.data.pipeline --config configs/data.yaml --dataset nf_unsw_nb15
    uv run python -m src.data.pipeline --config configs/data.yaml --all
"""

import argparse
import logging
import time
from pathlib import Path

from src.config import load_config
from src.data import clean, load, sample_split, stats
from src.data.processed import split_paths

log = logging.getLogger("pipeline")


def run_dataset(cfg, dataset):
    ds = cfg["dataset_configs"][dataset]
    cols, paths = cfg["columns"], cfg["paths"]
    block = cfg["io"]["read_block_size_bytes"]
    raw_path = Path(paths["raw_dir"]) / ds["raw_file"]
    t0 = time.time()

    load.check_header(raw_path, cols)

    log.info("[%s] pass 1: scanning classes in %s", dataset, raw_path)
    raw_ids = load.scan_classes(raw_path, cols["attack"], block)
    class_ids = clean.group_ids_by_class(raw_ids, ds["label_map"])
    raw_counts = {c: len(ids) for c, ids in class_ids.items()}
    log.info("[%s] %d raw rows, %d classes (%.0fs)", dataset, sum(raw_counts.values()), len(raw_counts),
             time.time() - t0)

    targets = sample_split.compute_class_targets(raw_counts, cfg["sampling"]["sample_size"],
                                                 cfg["sampling"]["rare_class_threshold"],
                                                 ds["keep_all_classes"])
    row_ids = sample_split.select_row_ids(class_ids, targets, cfg["seeds"]["sampling"])
    del raw_ids, class_ids

    log.info("[%s] pass 2: reading %d selected rows (leaky columns not read)", dataset, len(row_ids))
    read_cols = cols["features"] + [cols["label"], cols["attack"]]
    df = load.read_rows(raw_path, row_ids, read_cols, load.column_types(cols), block, cols["row_id"])
    df = clean.drop_leaky(df, cols["leaky"])
    clean.assert_no_leaky(df, cols["leaky"])
    df[cols["class"]] = clean.normalize_labels(df[cols["attack"]], ds["label_map"])
    clean.check_label_consistency(df, cols["label"], cols["class"], cols["benign_class"],
                                  cols["benign_label_value"])
    sampled = clean.enforce_schema(df, cols)
    sampled, mapped_counts = clean.map_string_values(sampled, cols["string_features"], ds["string_value_map"])
    inf_counts = clean.count_inf(sampled, cols["float_features"])
    if cfg["cleaning"]["inf_to_missing"]:
        sampled = clean.inf_to_missing(sampled, cols["float_features"])

    deduped, dedupe_info = sample_split.dedupe(sampled, cols["features"], cols["class"],
                                               cfg["dedupe"]["conflict_policy"])
    log.info("[%s] dedupe: %s -> %d rows", dataset, dedupe_info, len(deduped))
    kept, small = sample_split.drop_small_classes(deduped, cols["class"], cfg["dedupe"]["min_rows_per_class"])
    if small:
        log.warning("[%s] dropped classes too small to stratify: %s", dataset, small)

    splits = sample_split.split(kept, cfg["split"], cols["class"], cols["row_id"], cfg["seeds"]["split"])
    out_dir = Path(paths["processed_dir"]) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(out_dir / f"{name}{paths['split_file_suffix']}", index=False)
    log.info("[%s] wrote %s to %s", dataset, {n: len(p) for n, p in splits.items()}, out_dir)

    settings = {"sample_size": cfg["sampling"]["sample_size"],
                "rare_class_threshold": cfg["sampling"]["rare_class_threshold"],
                "keep_all_classes": ds["keep_all_classes"],
                "string_value_map": ds["string_value_map"],
                "inf_to_missing": cfg["cleaning"]["inf_to_missing"],
                "conflict_policy": cfg["dedupe"]["conflict_policy"],
                "min_rows_per_class": cfg["dedupe"]["min_rows_per_class"],
                "split": cfg["split"], "seeds": cfg["seeds"]}
    st = stats.build_stats(dataset, ds, settings, raw_counts, targets, sampled, deduped, dedupe_info,
                           small, splits, cols, inf_counts, mapped_counts)
    stats.write_dataset_stats(st, paths["stats_dir"])
    log.info("[%s] done in %.0fs", dataset, time.time() - t0)


def is_done(cfg, dataset):
    """Every split Parquet and the stats JSON of `dataset` exist (used by src/run.py)."""
    files = [*split_paths(cfg, dataset).values(), stats.dataset_stats_path(cfg["paths"]["stats_dir"], dataset)]
    return all(f.exists() for f in files)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="data config YAML, e.g. configs/data.yaml")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--dataset", help="one dataset name from the config's `datasets` list")
    which.add_argument("--all", action="store_true", help="every dataset in the config's `datasets` list")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    if args.dataset and args.dataset not in cfg["datasets"]:
        parser.error(f"unknown dataset {args.dataset!r}; choose from {cfg['datasets']}")
    for dataset in cfg["datasets"] if args.all else [args.dataset]:
        run_dataset(cfg, dataset)
    summary = stats.write_summary(cfg["paths"]["stats_dir"], cfg["datasets"], cfg["paths"]["stats_summary_file"],
                                  list(cfg["split"]), cfg["columns"]["benign_label_value"])
    log.info("summary table: %s", summary)


if __name__ == "__main__":
    main()
