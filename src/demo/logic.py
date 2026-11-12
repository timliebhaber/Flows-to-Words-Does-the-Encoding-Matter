"""Everything the live demo computes, without Streamlit (src/demo/app.py draws it; tests/test_demo.py checks it).

- `draw_flow`: a random flow (benign or attack, attack types drawn evenly) or one of a chosen class.
- `feature_spans`: where each feature sits in the E1-E4 text of a row, so one feature can be highlighted in all
  four texts. The text is rebuilt from the encoder config and must equal `encode_split`'s output.
- `hash_inputs`: the value text behind each E1/E2 token (stored value or E2 bin label).
- `predict_proba`: one row through XGBoost or small BERT.
- `crack_port` / `crack_flow`: the Exp 4 dictionary attack (src/eval/privacy.py) on one port or one whole flow.
- `load_numbers` / `fill`: report numbers (`\\resset` lines of report/generated/numbers_*.tex) for the findings.
"""

import re
import time
from pathlib import Path

import numpy as np
import torch

from src.data.processed import model_frame
from src.encoders import encode_split, encoding_cfg, kind
from src.encoders.common import column_values, rounded_text
from src.encoders.ppfle import hash_raw_column
from src.encoders.ppfle_binned import load_state
from src.encoders.sentence import _FORMATTER, _name
from src.eval.privacy import binned_columns, candidates, find, truth_texts

# ---- flows ----


def classes_of(cfg, df):
    cols = cfg["bert_cfg"]["data"]["columns"]
    return df[cols[cfg["bert_cfg"]["tasks"][cfg["task"]]["target"]]]


def benign_class(cfg):
    cols = cfg["bert_cfg"]["data"]["columns"]
    return cols[cfg["bert_cfg"]["tasks"][cfg["task"]]["benign"]]


def draw_flow(cfg, test, draw, wanted=None):
    """Index of a test flow. `wanted` = a class; None = mystery: an attack with probability `mystery.attack_share`,
    its type drawn evenly over the attack classes (so rare types show up too), else a normal flow."""
    rng = np.random.default_rng(cfg["app"]["random_seed"] + draw)
    labels = classes_of(cfg, test)
    if wanted is None:
        benign = benign_class(cfg)
        attacks = sorted(set(labels) - {benign})
        wanted = attacks[rng.integers(len(attacks))] if rng.random() < cfg["mystery"]["attack_share"] else benign
    pool = test.index[labels == wanted]
    return pool[rng.integers(len(pool))]


# ---- where each feature sits in each encoding ----

_OPEN, _CLOSE = "\x02", "\x03"   # markers around one feature's text while the E3/E4 text is rebuilt


def _marked(feature, text):
    return f"{_OPEN}{feature}{_CLOSE}{text}{_OPEN}{_CLOSE}"


def _marked_text(ecfg, encoding, row):
    """The E3/E4 text of a one-row frame with every feature's part wrapped in markers (same rules as the encoder)."""
    fmt, features = ecfg["format"], ecfg["data"]["columns"]["features"]
    decimals, na = fmt["float_decimals"], fmt["na_token"]
    values = {c: v[0] for c, v in column_values(row, features).items()}
    if kind(ecfg, encoding) == "kv_text":
        kv = ecfg["kv_text"]
        parts = []
        for c in features:
            name = kv["aliases"][c] if kv["use_aliases"] and c in kv["aliases"] else c
            parts.append(_marked(c, f"{name}{kv['kv_separator']}{rounded_text(values[c], decimals, na)}"))
        return fmt["token_joiner"].join(parts)
    sent = ecfg["sentence"]
    fields = {c: _marked(c, rounded_text(values[c], decimals, na)) for c in features}
    for name, spec in sent["derived"].items():
        src = spec["source"]
        fields[name] = _marked(src, _name(values[src], ecfg[spec["names"]], spec["unknown"], decimals, na))
    return _FORMATTER.vformat(fmt["token_joiner"].join(sent["template"]), (), fields)


def _strip_markers(marked):
    """(plain text, {feature: (start, end)}) from a marked text."""
    spans, out, pos, i = {}, [], 0, 0
    while i < len(marked):
        if marked[i] == _OPEN:
            close = marked.index(_CLOSE, i)
            feature = marked[i + 1:close]
            if feature:
                end = marked.index(_OPEN, close)
                text = marked[close + 1:end]
                spans[feature] = (pos, pos + len(text))
                out.append(text)
                pos += len(text)
                i = marked.index(_CLOSE, end) + 1
                continue
            i = close + 1
            continue
        out.append(marked[i])
        pos += 1
        i += 1
    return "".join(out), spans


def feature_spans(ecfg_all, encoding, dataset, row):
    """(text, {feature: (start, end)}) for a one-row frame. The text is always `encode_split`'s; the spans come
    from token order (E1/E2: one token per feature) or a marked rebuild (E3/E4) that must give the same text."""
    text = encode_split(ecfg_all, encoding, dataset, row)[0]
    ecfg = encoding_cfg(ecfg_all, encoding)
    if kind(ecfg_all, encoding) in ("ppfle", "ppfle_binned"):
        spans, pos = {}, 0
        joiner = ecfg["format"]["token_joiner"]
        for c, token in zip(ecfg["data"]["columns"]["features"], text.split(joiner)):
            spans[c] = (pos, pos + len(token))
            pos += len(token) + len(joiner)
        return text, spans
    rebuilt, spans = _strip_markers(_marked_text(ecfg, encoding, row))
    if rebuilt != text:
        raise ValueError(f"{encoding}: highlight text differs from the encoder's text")
    return text, spans


def is_hashed(ecfg_all, encoding):
    return kind(ecfg_all, encoding) in ("ppfle", "ppfle_binned")


def hash_inputs(ecfg_all, encoding, dataset, row):
    """{feature: value text that goes into the hash} of a one-row frame for E1/E2: the stored value, or the bin
    label of an E2 numeric feature (the same plaintext Exp 4 checks recovered tokens against)."""
    ecfg = encoding_cfg(ecfg_all, encoding)
    state = load_state(ecfg, dataset) if kind(ecfg_all, encoding) == "ppfle_binned" else None
    return {c: truth_texts(row, c, ecfg, state)[0] for c in ecfg["data"]["columns"]["features"]}


# ---- models ----


def predict_proba(cfg, entry, row, text, tok, train, device):
    """(class probabilities, milliseconds) of one demo model for one flow."""
    meta, model = entry["meta"], entry["model"]
    t0 = time.perf_counter()
    if meta["model"] == cfg["xgb_cfg"]["model_name"]:
        cols = cfg["xgb_cfg"]["data"]["columns"]
        probs = model.predict_proba(model_frame(row, cols["features"], cols["string_features"], train))[0]
    else:
        ids = torch.tensor([tok.encode(text).ids], device=device)
        with torch.no_grad():
            probs = torch.softmax(model(input_ids=ids).logits.float(), -1)[0].cpu().numpy()
    return probs, (time.perf_counter() - t0) * 1000


# ---- the hash attack ----


def attack_setup(cfg):
    """(privacy config, encoding config, binned columns, E2 state) of the attacked encoding."""
    pcfg, enc = cfg["privacy_cfg"], cfg["app"]["attack_encoding"]
    ecfg = encoding_cfg(pcfg["encoder"], enc)
    binned = binned_columns(pcfg, enc)
    return pcfg, ecfg, binned, (load_state(ecfg, cfg["dataset"]) if binned else None)


def fingerprint(ecfg, column, value):
    """The token the encoder writes for `column` = `value` (E1 rule)."""
    return hash_raw_column(column, [value], ecfg)[0]


def crack_port(cfg, column, token):
    """Hash every value of `column`'s known domain (privacy config, level 1) and look for `token`.
    Returns (value or None, number of values tried, seconds)."""
    pcfg, ecfg, binned, _ = attack_setup(cfg)
    t0 = time.perf_counter()
    guesses = list(candidates(pcfg, ecfg, column, "enumerate", binned, {}))
    hit = find(column, guesses, {token}, ecfg).get(token)
    values = len(guesses) - (ecfg["format"]["na_token"] in guesses)   # the "missing" guess is not a port
    return hit, values, time.perf_counter() - t0


def crack_flow(cfg, row, public):
    """Yield one dict per feature of a one-row frame: column, token, recovered text (or None), level label and
    whether it equals the true plaintext. Levels are tried in the privacy config's order, as in Exp 4."""
    pcfg, ecfg, binned, state = attack_setup(cfg)
    features = pcfg["data"]["columns"]["features"]
    joiner = ecfg["format"]["token_joiner"]
    tokens = encode_split(pcfg["encoder"], cfg["app"]["attack_encoding"], cfg["dataset"], row)[0].split(joiner)
    for col, token in zip(features, tokens):
        guess, level = None, None
        for lv in pcfg["levels"]:
            hit = find(col, candidates(pcfg, ecfg, col, lv["name"], binned, public), {token}, ecfg)
            if hit:
                guess, level = hit[token], lv["label"]
                break
        yield {"column": col, "token": token, "guess": guess, "level": level,
               "correct": guess is not None and guess == truth_texts(row, col, ecfg, state)[0]}


# ---- report numbers ----

_RESSET = re.compile(r"\\resset\{([^{}]*)\}\{(.*)\}\{([^{}]*)\}\s*$")
_LATEX = [(r"$\pm$", "±"), (r"\,", " "), (r"\%", "%"), ("$", "")]


def load_numbers(paths):
    """{key: (text, raw)} from generated numbers files (one `\\resset{key}{text}{raw}` per line)."""
    numbers = {}
    for path in paths:
        if not Path(path).exists():
            continue
        for line in Path(path).read_text().splitlines():
            if m := _RESSET.match(line):
                text = m[2]
                for a, b in _LATEX:
                    text = text.replace(a, b)
                numbers[m[1]] = (text, m[3])
    return numbers


_PLACEHOLDER = re.compile(r"\{([^{}|]+)(?:\|(pct))?\}")


def fill(template, numbers, pct_decimals, missing):
    """`template` with each {key} replaced by the key's printed value and {key|pct} by 100 * raw value."""
    def one(m):
        if m[1] not in numbers:
            return missing
        text, raw = numbers[m[1]]
        return f"{100 * float(raw):.{pct_decimals}f}" if m[2] else text
    return _PLACEHOLDER.sub(one, template)


def placeholders(template):
    return [m[1] for m in _PLACEHOLDER.finditer(template)]


# ---- display helpers ----


def human_bytes(n, units):
    """1536 -> '1.5 KB' (units from the config, base 1024)."""
    n = float(n)
    for unit in units[:-1]:
        if abs(n) < 1024:
            return f"{n:.0f} {unit}" if unit == units[0] else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} {units[-1]}"


def human_ms(ms):
    ms = float(ms)
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"
