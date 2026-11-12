"""E3: plain `COLUMN=value` text, floats rounded (plan §5)."""

from src.encoders.common import column_values, join_rows, map_column, rounded_text

NEEDS_FIT = False


def encode(df, cfg, state):
    fmt, kv = cfg["format"], cfg["kv_text"]
    features = cfg["data"]["columns"]["features"]
    values = column_values(df, features)
    columns = []
    for c in features:
        name = kv["aliases"][c] if kv["use_aliases"] and c in kv["aliases"] else c
        columns.append(map_column(values[c], lambda v, name=name: (
            f"{name}{kv['kv_separator']}{rounded_text(v, fmt['float_decimals'], fmt['na_token'])}")))
    return join_rows(columns, fmt["token_joiner"])
