"""Cleaning: leaky columns, label normalization, schema (fixed column order)."""

import numpy as np


def drop_leaky(df, leaky):
    return df.drop(columns=[c for c in leaky if c in df.columns])


def assert_no_leaky(df, leaky):
    left = [c for c in leaky if c in df.columns]
    if left:
        raise ValueError(f"leaky columns still present: {left}")


def count_inf(df, columns):
    return {c: int(n) for c, n in np.isinf(df[columns]).sum().items() if n}


def inf_to_missing(df, columns):
    """Replace +/-inf with NaN in `columns` (returns a new frame)."""
    df = df.copy()
    df[columns] = df[columns].replace([np.inf, -np.inf], np.nan)
    return df


def map_string_values(df, columns, value_map):
    """Rewrite spellings in text columns (e.g. Edge-IIoTset's '0.0' -> '0' placeholder).

    Returns (new frame, {column: number of values changed}).
    """
    df = df.copy()
    changed = {}
    for c in columns:
        hit = df[c].isin(list(value_map))
        if hit.any():
            df.loc[hit, c] = df.loc[hit, c].map(value_map)
            changed[c] = int(hit.sum())
    return df, changed


def _check_known(values, label_map):
    unknown = sorted(set(values) - set(label_map))
    if unknown:
        raise ValueError(f"attack values missing from label_map: {unknown}")


def normalize_labels(attack, label_map):
    """Map raw attack names (Series) to normalized class names. Unknown names raise."""
    _check_known(attack.unique(), label_map)
    return attack.map(label_map)


def group_ids_by_class(raw_ids, label_map):
    """{raw attack value: ids} -> {normalized class: sorted ids} (merges spellings of one class)."""
    _check_known(raw_ids, label_map)
    grouped = {}
    for raw, ids in raw_ids.items():
        grouped.setdefault(label_map[raw], []).append(ids)
    return {c: np.sort(np.concatenate(parts)) for c, parts in grouped.items()}


def check_label_consistency(df, label_col, class_col, benign_class, benign_value):
    """Binary label and attack class must agree: label == benign_value <=> class == benign_class."""
    bad = (df[label_col] == benign_value) != (df[class_col] == benign_class)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows where '{label_col}' disagrees with '{class_col}'")


def enforce_schema(df, cols):
    """Fixed column order: row id, features (config order), label, raw attack, class."""
    order = [cols["row_id"]] + cols["features"] + [cols["label"], cols["attack"], cols["class"]]
    missing = [c for c in order if c not in df.columns]
    extra = [c for c in df.columns if c not in order]
    if missing or extra:
        raise ValueError(f"schema mismatch: missing {missing}, unexpected {extra}")
    return df[order].reset_index(drop=True)
