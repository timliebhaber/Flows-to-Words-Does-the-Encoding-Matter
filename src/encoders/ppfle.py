"""E1: PPFLE as published. Each feature -> hash(column$value), space-joined (plan §5)."""

from src.encoders.common import column_values, hasher, join_rows, map_column, raw_text

NEEDS_FIT = False


def hash_raw_column(name, values, cfg):
    fmt, h = cfg["format"], hasher(cfg["hash"]["algorithm"], cfg["hash"]["length"])
    return map_column(values, lambda v: h(f"{name}{fmt['separator']}{raw_text(v, fmt['na_token'])}"))


def encode(df, cfg, state):
    features = cfg["data"]["columns"]["features"]
    values = column_values(df, features)
    return join_rows([hash_raw_column(c, values[c], cfg) for c in features], cfg["format"]["token_joiner"])
