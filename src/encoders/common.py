"""Value formatting and hashing shared by the encoders.

Every encoder walks `columns.features` in config order and formats each column as a whole
(column-wise is much faster than row-wise for ~100k rows x 47 columns).
"""

import hashlib
import math


def is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def raw_text(value, na_token):
    """Value as stored (E1/E2 hash input): 1520 -> '1520', 7.0 -> '7.0'."""
    return na_token if is_missing(value) else str(value)


def rounded_text(value, decimals, na_token):
    """E3/E4 text: floats rounded to at most `decimals` places (0.4795 -> '0.48', 7.0 -> '7.0')."""
    if is_missing(value):
        return na_token
    return str(round(value, decimals)) if isinstance(value, float) else str(value)


def map_column(values, fn):
    """Apply fn once per distinct value of a column (python list) and return the mapped list."""
    cache = {}
    out = []
    for v in values:
        key = "nan" if is_missing(v) else v
        if key not in cache:
            cache[key] = fn(v)
        out.append(cache[key])
    return out


def hasher(algorithm, length):
    def hash_text(text):
        return hashlib.new(algorithm, text.encode()).hexdigest()[:length]
    return hash_text


def join_rows(columns, joiner):
    """Per-column token lists -> one string per row."""
    return [joiner.join(parts) for parts in zip(*columns)]


def column_values(df, features):
    """{feature: python list} (ints stay int, floats stay float, NaN stays NaN)."""
    return {c: df[c].tolist() for c in features}
