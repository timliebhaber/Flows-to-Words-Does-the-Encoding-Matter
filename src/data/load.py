"""Streaming CSV reads (pyarrow). The raw files are too large to load whole."""

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv


def read_header(path):
    with open(path) as f:
        return f.readline().strip().split(",")


def check_header(path, cols):
    """The raw header must be exactly features + leaky + label + attack (in any order)."""
    header = set(read_header(path))
    expected = set(cols["features"] + cols["leaky"] + [cols["label"], cols["attack"]])
    if header != expected:
        raise ValueError(f"{path}: header mismatch. Missing {sorted(expected - header)}, "
                         f"unexpected {sorted(header - expected)}")


def column_types(cols):
    """Explicit Arrow types for every kept raw column (streaming readers must not infer per block)."""
    floats, strings = set(cols["float_features"]), set(cols["string_features"])
    types = {c: (pa.string() if c in strings else pa.float64() if c in floats else pa.int64())
             for c in cols["features"]}
    types[cols["label"]] = pa.int64()
    types[cols["attack"]] = pa.string()
    return types


def _open(path, columns, types, block_size):
    return pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=block_size),
        convert_options=pacsv.ConvertOptions(include_columns=columns,
                                             column_types={c: types[c] for c in columns}),
    )


def scan_classes(path, attack_col, block_size):
    """Pass 1: read only the attack column.

    Returns {raw attack value: sorted np.ndarray of 0-based data-row ids}.
    """
    reader = _open(path, [attack_col], {attack_col: pa.string()}, block_size)
    parts, offset = {}, 0
    for batch in reader:
        col = batch.column(0)
        if col.null_count:
            raise ValueError(f"{path}: {col.null_count} empty '{attack_col}' values near row {offset}")
        enc = col.dictionary_encode()
        codes = enc.indices.to_numpy()
        for k, name in enumerate(enc.dictionary.to_pylist()):
            parts.setdefault(name, []).append(np.flatnonzero(codes == k) + offset)
        offset += batch.num_rows
    return {name: np.concatenate(p) for name, p in parts.items()}


def read_rows(path, row_ids, columns, types, block_size, row_id_col):
    """Pass 2: stream `columns` only and keep the rows whose id is in sorted `row_ids`.

    Columns not listed (the leaky ones) are never read. Adds `row_id_col` with the raw row id.
    """
    reader = _open(path, columns, types, block_size)
    tables, offset = [], 0
    for batch in reader:
        lo, hi = np.searchsorted(row_ids, [offset, offset + batch.num_rows])
        if hi > lo:
            ids = row_ids[lo:hi]
            table = pa.Table.from_batches([batch]).take(pa.array(ids - offset))
            tables.append(table.append_column(row_id_col, pa.array(ids, pa.int64())))
        offset += batch.num_rows
    if offset <= row_ids[-1]:
        raise ValueError(f"{path}: selected row id {row_ids[-1]} beyond file end ({offset} rows)")
    return pa.concat_tables(tables).to_pandas()
