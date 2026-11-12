"""Encoder registry. Model code only needs `encode_split`."""

from src.encoders import kv_text, ppfle, ppfle_binned, sentence

KINDS = {"ppfle": ppfle, "ppfle_binned": ppfle_binned, "kv_text": kv_text, "sentence": sentence}


def kind(cfg, encoding):
    """An `encodings` entry is a kind, or {kind: ..., <section>: {overrides}} (e.g. a shorter hash)."""
    spec = cfg["encodings"][encoding]
    return spec["kind"] if isinstance(spec, dict) else spec


def encoding_cfg(cfg, encoding):
    """The config one encoding sees: top-level sections with that encoding's overrides applied."""
    spec = cfg["encodings"][encoding]
    if not isinstance(spec, dict):
        return cfg
    return {**cfg, **{section: {**cfg[section], **values} for section, values in spec.items() if section != "kind"}}


def module(cfg, encoding):
    name = kind(cfg, encoding)
    if name not in KINDS:
        raise ValueError(f"encoding {encoding!r}: unknown kind {name!r}; choose from {list(KINDS)}")
    return KINDS[name]


def fit_state(cfg, encoding, train_df):
    """Fitted state from a TRAIN split (None for encoders that need no fitting)."""
    mod = module(cfg, encoding)
    return mod.fit(train_df, encoding_cfg(cfg, encoding)) if mod.NEEDS_FIT else None


def encode(cfg, encoding, df, state):
    return module(cfg, encoding).encode(df, encoding_cfg(cfg, encoding), state)


def encode_split(cfg, encoding, fit_dataset, df):
    """Texts for `df`, using the state fitted on `fit_dataset`'s train split (saved by src.encoders.build)."""
    mod, ecfg = module(cfg, encoding), encoding_cfg(cfg, encoding)
    state = mod.load_state(ecfg, fit_dataset) if mod.NEEDS_FIT else None
    return mod.encode(df, ecfg, state)
