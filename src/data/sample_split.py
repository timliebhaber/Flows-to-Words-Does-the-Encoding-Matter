"""Stratified sampling (rare classes kept whole), dedupe, stratified split."""

import numpy as np
from sklearn.model_selection import train_test_split


def compute_class_targets(class_counts, sample_size, rare_threshold, keep_all_classes):
    """How many rows to sample from each class.

    Classes with fewer than `rare_threshold` rows, and classes in `keep_all_classes`, are kept whole.
    The remaining budget is shared proportionally among the other classes (largest-remainder
    rounding, so the total is exactly `sample_size` unless the data is smaller).
    """
    whole = {c: n for c, n in class_counts.items() if n < rare_threshold or c in keep_all_classes}
    rest = {c: n for c, n in class_counts.items() if c not in whole}
    budget = sample_size - sum(whole.values())
    if budget < 0:
        raise ValueError(f"classes kept whole ({sum(whole.values())} rows) exceed sample_size {sample_size}")
    total_rest = sum(rest.values())
    if budget >= total_rest:
        return dict(class_counts)

    quotas = {c: budget * n / total_rest for c, n in rest.items()}
    targets = {c: int(q) for c, q in quotas.items()}
    leftover = budget - sum(targets.values())
    by_remainder = sorted(rest, key=lambda c: (-(quotas[c] - targets[c]), c))
    for c in by_remainder[:leftover]:
        targets[c] += 1
    targets.update(whole)
    return targets


def select_row_ids(class_ids, targets, seed):
    """Pick exactly targets[c] row ids from each class; returns all picked ids sorted.

    Depends only on the ids, targets and seed (not on how the file was read).
    """
    rng = np.random.default_rng(seed)
    picked = []
    for c in sorted(class_ids):
        ids, k = class_ids[c], targets[c]
        picked.append(ids if k == len(ids) else rng.choice(ids, size=k, replace=False))
    return np.sort(np.concatenate(picked))


def dedupe(df, feature_cols, class_col, conflict_policy):
    """Remove exact duplicates (features + class), keeping the first (lowest raw row id) row.

    Then handle conflicts (same features, different classes) according to `conflict_policy`.
    Must run AFTER leaky columns are dropped.
    """
    is_dup = df.duplicated(subset=feature_cols + [class_col], keep="first")
    df = df[~is_dup]
    n_classes = df.groupby(feature_cols, dropna=False, sort=False)[class_col].transform("nunique")
    is_conflict = n_classes > 1
    info = {"exact_duplicates_removed": int(is_dup.sum()),
            "conflict_rows": int(is_conflict.sum()),
            "conflict_rows_removed": 0}
    if conflict_policy == "drop_all":
        df = df[~is_conflict]
        info["conflict_rows_removed"] = info["conflict_rows"]
    elif conflict_policy != "keep":
        raise ValueError(f"unknown conflict_policy {conflict_policy!r}")
    return df.reset_index(drop=True), info


def drop_small_classes(df, class_col, min_rows):
    """Drop classes with fewer than `min_rows` rows (cannot be stratified). Returns (df, {class: n})."""
    counts = df[class_col].value_counts()
    small = counts[counts < min_rows]
    return df[~df[class_col].isin(small.index)].reset_index(drop=True), {c: int(n) for c, n in small.items()}


def split(df, ratios, class_col, row_id_col, seed):
    """Stratified 3-way split. `ratios` is {split name: fraction} in order (e.g. train, val, test).

    Each part is sorted by raw row id so files are stable and readable.
    """
    (n1, r1), (n2, r2), (n3, r3) = ratios.items()
    first, rest = train_test_split(df, train_size=r1, stratify=df[class_col], random_state=seed)
    second, third = train_test_split(rest, train_size=r2 / (r2 + r3), stratify=rest[class_col],
                                     random_state=seed)
    return {name: part.sort_values(row_id_col).reset_index(drop=True)
            for name, part in [(n1, first), (n2, second), (n3, third)]}
