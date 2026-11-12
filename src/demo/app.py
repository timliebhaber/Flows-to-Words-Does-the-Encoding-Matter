"""Live demo for the station session (configs/demo.yaml): "Can language models read network flows?"

Start (from the repo root; train the demo models first with `uv run ftw demo`):
    uv run streamlit run src/demo/app.py -- --config configs/demo.yaml
Open a given step directly with ?step=N (1-5), e.g. http://localhost:8501/?step=3

Five steps: (1) one real network flow from the test split, (2) the same flow encoded live as E1-E4 text, under the
original features, with one feature highlighted in all four, (3) guess the class and compare with three trained
classifiers, then a what-if analysis, (4) a live dictionary attack on the PPFLE hashes, (5) the project's results,
with numbers read from the report's generated files. Computation lives in src/demo/logic.py, HTML in src/demo/ui.py.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import BertForSequenceClassification

from src.config import load_demo_config, split_roles
from src.data.processed import load_splits, split_paths
from src.demo import logic, ui
from src.demo.train import load_xgb, model_dirs, read_meta
from src.encoders import encode_split, encoding_cfg, kind
from src.encoders.common import raw_text
from src.eval.privacy import public_values
from src.tokenizers.train_bpe import load_tokenizer


def config_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="demo config YAML, e.g. configs/demo.yaml")
    return parser.parse_known_args(sys.argv[1:])[0].config


# ---- loading (once per app process) ----

@st.cache_resource
def load_everything(path):
    cfg = load_demo_config(path)
    app, bcfg = cfg["app"], cfg["bert_cfg"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if app["device"] == "auto" else app["device"]
    splits = load_splits(bcfg["data"], cfg["dataset"])
    tokenizers = {e: load_tokenizer(bcfg, e, cfg["dataset"])[0] for e in app["show_encodings"]}
    models, missing = {}, []
    for (model, encoding), d in model_dirs(cfg).items():
        if not (d / cfg["meta_file"]).exists():
            missing.append(f"{model}/{encoding}")
            continue
        if model == cfg["xgb_cfg"]["model_name"]:
            obj = load_xgb(cfg, d)
            obj.set_params(device=app["xgb_device"])
        else:
            obj = BertForSequenceClassification.from_pretrained(d).to(device).eval()
            tokenizers.setdefault(encoding, load_tokenizer(bcfg, encoding, cfg["dataset"])[0])
        models[(model, encoding)] = {"meta": read_meta(cfg, d), "model": obj}
    return cfg, device, splits, tokenizers, models, missing


@st.cache_resource
def load_public(path):
    cfg = load_demo_config(path)
    pcfg = cfg["privacy_cfg"]
    return public_values(pcfg, cfg["dataset"]) if "public" in [lv["name"] for lv in pcfg["levels"]] else {}


@st.cache_data
def load_numbers(path):
    return logic.load_numbers(load_demo_config(path)["numbers_files"])


@st.cache_data
def leak_table(path):
    """{column: {(benign|attack, spelling): share of rows}} per text column of the as-published Edge data."""
    cfg = load_demo_config(path)
    leak, data = cfg["leak"], cfg["leak_data"]
    cols = data["columns"]
    vmap = data["dataset_configs"][leak["fixed_dataset"]]["string_value_map"]
    spellings = list(dict.fromkeys([*vmap, *vmap.values()]))
    df = pd.read_parquet(split_paths(data, leak["raw_dataset"])[leak["split"]],
                         columns=cols["string_features"] + [cols["class"]])
    benign = df[cols["class"]] == cols["benign_class"]
    table = {c: {**{("benign", s): float((df.loc[benign, c] == s).mean()) for s in spellings},
                 **{("attack", s): float((df.loc[~benign, c] == s).mean()) for s in spellings}}
             for c in cols["string_features"]}
    return table, spellings


# ---- session state: one flow at a time ----

def new_flow(mode, wanted=None):
    """Draw the next flow and forget everything about the previous one (guess, tries, attack, what-if)."""
    ss = st.session_state
    ss.draw = ss.get("draw", 0) + 1
    ss.mode, ss.wanted = mode, wanted
    ss.revealed, ss.guess, ss.scored = mode == "explore", None, False
    ss.tries, ss.cracked_port, ss.attack, ss.leak_pick = [], None, None, None


def pick_class():
    if st.session_state.get("explore_pick"):
        new_flow("explore", st.session_state.explore_pick)


def make_guess(key):
    st.session_state.guess, st.session_state.revealed = st.session_state.get(key), True


def go(step):
    st.session_state.step = step


def next_button(story, step):
    if step + 1 < len(story["steps"]):
        st.button(f"Next: {story['steps'][step + 1]} →", type="primary", on_click=go, args=(step + 1,),
                  key=f"next_{step}")


# ---- helpers ----

def class_label(story, c):
    info = story["classes"].get(c, {"emoji": "❔", "name": c})
    return f"{info['emoji']} {info['name']}"


def service(story, port):
    names = story["port_names"]
    return (names[port], f"port {port}") if port in names else ("Unknown service", f"port {port}")


def features_of(cfg):
    return cfg["bert_cfg"]["data"]["columns"]["features"]


def encoding_info(cfg, encoding):
    """story.encodings entry with the encoder settings filled into its `about` text."""
    ecfg = encoding_cfg(cfg["bert_cfg"]["enc"], encoding)
    info = dict(cfg["story"]["encodings"][encoding])
    info["about"] = info["about"].format(hash_length=ecfg["hash"]["length"], separator=ecfg["format"]["separator"],
                                         n_bins=ecfg["binning"]["n_bins"], decimals=ecfg["format"]["float_decimals"])
    return info


def connection_html(cfg, row):
    story, enc = cfg["story"], cfg["bert_cfg"]["enc"]
    card = story["card"]
    v = {k: row[col].iloc[0] for k, col in card.items()}
    units = story["byte_units"]
    name, port_text = service(story, int(v["port"]))
    proto = enc["protocol_names"].get(int(v["protocol"]), f"IP protocol {int(v['protocol'])}")
    app_name = enc["l7_names"].get(int(float(v["app"])), "unknown")
    stats = [("Duration", card["duration"], logic.human_ms(v["duration"])),
             ("Application (nDPI)", card["app"], app_name),
             ("Longest packet", card["biggest"], logic.human_bytes(v["biggest"], units)),
             ("Minimum TTL", card["ttl"], int(v["ttl"]))]
    out = (int(v["pkts_out"]), logic.human_bytes(v["bytes_out"], units), f"{card['pkts_out']} · {card['bytes_out']}")
    back = (int(v["pkts_back"]), logic.human_bytes(v["bytes_back"], units),
            f"{card['pkts_back']} · {card['bytes_back']}")
    return ui.connection(name, port_text, proto, out, back, stats)


def predictions(cfg, row, tokenizers, models, train, device):
    """[(classifier, class names, probabilities)] for one flow, in the configured order."""
    out, texts = [], {}
    for d in cfg["story"]["classifiers"]:
        entry = models.get((d["model"], d["encoding"]))
        if entry is None:
            continue
        enc = d["encoding"]
        if enc in cfg["bert_cfg"]["enc"]["encodings"] and enc not in texts:
            texts[enc] = encode_split(cfg["bert_cfg"]["enc"], enc, cfg["dataset"], row)[0]
        probs, _ = logic.predict_proba(cfg, entry, row, texts.get(enc), tokenizers.get(enc), train, device)
        out.append((d, entry["meta"]["class_names"], probs))
    return out


def top_rows(story, names, probs, k):
    top = np.argsort(probs)[::-1][:k]
    return [(class_label(story, names[i]), float(probs[i])) for i in top]


def pieces_html(text, tok, colors):
    """The text with every BPE token on its own background colour (what the model actually reads)."""
    spans = [(s, e) for s, e in tok.encode(text).offsets if e > s]
    out = [f'<span style="background:{colors[i % len(colors)]};border-radius:3px;padding:0 1px">'
           f"{ui.e(text[s:e])}</span>" for i, (s, e) in enumerate(spans)]
    return "".join(out)


# ---- the five steps ----

def step_meet(cfg, row, truth):
    story = cfg["story"]
    features = features_of(cfg)
    st.html(ui.lead(f"A <b>flow</b> is the record a network sensor (NetFlow) writes for one connection between two "
                    f"hosts: destination port, protocol, packets and bytes in each direction, duration, TCP flags, "
                    f"packet timing. {cfg['display']['datasets'][cfg['dataset']]} describes every flow with "
                    f"<b>{len(features)} features</b>. Here is one from the test split."))
    left, right = st.columns([1, 3], vertical_alignment="center")
    left.button("Draw a random flow", type="primary", icon="🎲", on_click=new_flow, args=("mystery",),
                width="stretch")
    with right:
        if st.session_state.revealed:
            info = story["classes"].get(truth, {"name": truth, "about": ""})
            st.html(ui.pill(f"Label: {class_label(story, truth)} · {info['about']}",
                            "good" if truth == logic.benign_class(cfg) else "bad"))
        else:
            st.html(ui.pill("❓ Label hidden: you classify this flow in step 3", "hidden"))
    st.html(connection_html(cfg, row))
    with st.expander("Pick a flow of a specific class", icon="🔎"):
        st.pills("Class", list(story["classes"]), format_func=lambda c: class_label(story, c),
                 key="explore_pick", on_change=pick_class, label_visibility="collapsed")
        st.html(ui.class_list(story["classes"]))
    with st.expander(f"All {len(features)} features of this flow", icon="📋"):
        table = row.T.rename(columns={row.index[0]: "value"}).astype(str)
        table["description"] = table.index.map(cfg["app"]["column_info"])
        st.dataframe(table, width="stretch", height=420,
                     column_config={"description": st.column_config.TextColumn(width="large")})
    next_button(story, 0)


def transform_rows(cfg, encs, texts, plains, feature, value):
    """[(encoding info, how (html), output (html))]: how each encoding writes one feature."""
    enc_cfg = cfg["bert_cfg"]["enc"]
    rows = []
    for enc in encs:
        info = encoding_info(cfg, enc)
        text, spans = texts[enc]
        a, b = spans[feature]
        out = f"<b>{ui.e(text[a:b])}</b>"
        if enc in plains:
            ecfg = encoding_cfg(enc_cfg, enc)
            plain = plains[enc][feature]
            how = (f'{ui.e(ecfg["hash"]["algorithm"])}("{ui.e(feature)}{ui.e(ecfg["format"]["separator"])}'
                   f'{ui.e(plain)}")[:{ecfg["hash"]["length"]}]')
            if plain != value:
                how = f"quantile bin of {ui.e(value)} = {ui.e(plain)}<br>{how}"
            elif kind(enc_cfg, enc) == "ppfle_binned":
                how = f"categorical feature, not binned<br>{how}"
        elif kind(enc_cfg, enc) == "kv_text":
            how = "short name, value rounded"
        else:
            how = "value placed in the sentence template"
            lo = text.rfind(" ", 0, max(a - 30, 0)) + 1
            hi = text.find(" ", min(b + 30, len(text)))
            hi = len(text) if hi < 0 else hi
            out = (("…" if lo else "") + ui.e(text[lo:a]) + out + ui.e(text[b:hi])
                   + ("…" if hi < len(text) else ""))
        rows.append((info, how, out))
    return rows


def words_board(cfg, row, tokenizers, texts, plains, counts, done, current, hl, show_pieces):
    """Original flow + how one feature is encoded + the four texts, after `done` features (`current` = the feature
    being encoded now, None once the whole flow is encoded)."""
    app, features = cfg["app"], features_of(cfg)
    encs = app["show_encodings"]
    finished = done == len(features)
    values = {c: raw_text(row[c].iloc[0], cfg["bert_cfg"]["enc"]["format"]["na_token"]) for c in features}
    status = {c: "done" if i < done else "pending" for i, c in enumerate(features)}
    if current:
        status[current] = "now"
    grid_status = {c: ("" if s == "pending" else s) for c, s in status.items()} if not finished else {hl: "hl"}
    ds = cfg["display"]["datasets"][cfg["dataset"]]
    head = (f'<div class="ftw-row"><div class="ftw-kicker">Original flow · {ui.e(ds)} test split · row '
            f'{row.index[0]}</div>{ui.progress(done, len(features))}</div>')
    html = [ui.card(head + ui.flow_grid(features, values, app["column_info"], grid_status))]
    shown = current or (hl if finished else None)
    if shown:
        html.append(ui.transform(shown, values[shown], app["column_info"][shown],
                                 transform_rows(cfg, encs, texts, plains, shown, values[shown])))
    cards = []
    mono = {e for e in encs if kind(cfg["bert_cfg"]["enc"], e) != "sentence"}   # sentences read better unspaced
    for e in encs:
        text, spans = texts[e]
        body = (pieces_html(text, tokenizers[e], app["piece_colors"]) if finished and show_pieces
                else ui.encoded_text(text, spans, status, hl if finished else None))
        cards.append(ui.encoding_card(encoding_info(cfg, e), body, e in mono,
                                      counts[e] if finished else None, max(counts.values())))
    html.append(ui.encoding_grid(cards))
    return "".join(html)


def step_words(cfg, row, tokenizers):
    story, enc_cfg, app, ss = cfg["story"], cfg["bert_cfg"]["enc"], cfg["app"], st.session_state
    st.html(ui.lead("Language models take <b>text</b> as input, so every flow has to be serialised first. We compare "
                    "four encodings of the same flow: two based on hashes (the published PPFLE and our binned "
                    "variant) and two readable ones. Watch the flow below being encoded feature by feature."))
    encs, features = app["show_encodings"], features_of(cfg)
    texts = {e: logic.feature_spans(enc_cfg, e, cfg["dataset"], row) for e in encs}
    plains = {e: logic.hash_inputs(enc_cfg, e, cfg["dataset"], row) for e in encs if logic.is_hashed(enc_cfg, e)}
    counts = {e: len(tokenizers[e].encode(texts[e][0]).ids) for e in encs}
    hl = st.pills("Highlight one feature in all four texts", story["highlight"], default=story["highlight"][0],
                  key="hl")
    a, b, _ = st.columns([1.2, 1.5, 4], vertical_alignment="center")
    replay = a.button("Encode again", icon="🔁", width="stretch")
    show_pieces = b.toggle("Show BPE tokens", key="pieces")
    board = st.empty()
    if replay or ss.get("encoded") != ss.draw:
        for i, feature in enumerate(features):
            board.html(words_board(cfg, row, tokenizers, texts, plains, counts, i, feature, None, False))
            time.sleep(app["encode_step_s"])
        ss.encoded = ss.draw
    board.html(words_board(cfg, row, tokenizers, texts, plains, counts, len(features), None, hl, show_pieces))
    st.html(ui.takeaway("A hash carries no meaning by itself: the model can only learn that a token such as "
                        f"<code>{ui.e(texts[encs[0]][0].split()[0])}</code> tends to occur in some classes, and every "
                        "unseen value is a new token. Binning makes nearby values share a token; readable text keeps "
                        "numbers and names that a pretrained model can relate. Step 3 compares the predictions."))
    next_button(story, 1)


def step_detect(cfg, row, truth, tokenizers, models, missing, train, device):
    story, ss = cfg["story"], st.session_state
    if missing:
        st.warning(f"Demo models not trained yet: {', '.join(missing)}. Run `uv run ftw demo` (~6 min GPU).")
        return
    st.html(connection_html(cfg, row))
    if not ss.revealed:
        st.html(ui.lead("<b>Your turn.</b> Is this flow benign, or which attack class is it? Hint: look at the "
                        "destination port, the packet counts in each direction and the duration."))
        key = f"guess_{ss.draw}"
        st.pills("Your guess", list(story["classes"]), format_func=lambda c: class_label(story, c), key=key,
                 on_change=make_guess, args=(key,), label_visibility="collapsed")
        with st.expander("Class definitions", icon="📖"):
            st.html(ui.class_list(story["classes"]))
        st.button("Skip, show the predictions", icon="🙈", on_click=make_guess, args=("no_guess",))
        return
    preds = predictions(cfg, row, tokenizers, models, train, device)
    verdicts = [names[int(np.argmax(p))] for _, names, p in preds]
    you = ss.guess if ss.mode == "mystery" else None
    if you is not None and not ss.scored:   # count this round once
        sc = ss.setdefault("score", {"rounds": 0, "you": 0, "det": [0] * len(preds)})
        sc["rounds"] += 1
        sc["you"] += int(you == truth)
        sc["det"] = [n + int(v == truth) for n, v in zip(sc["det"], verdicts)]
        ss.scored = True
    info = story["classes"].get(truth, {"name": truth, "about": ""})
    badge = "" if you is None else ui.pill(
        f"Your guess: {class_label(story, you)} · {'correct' if you == truth else 'wrong'}",
        "good" if you == truth else "bad")
    st.html(ui.truth(info.get("emoji", "❔"), truth == logic.benign_class(cfg), f"True label: {info['name']}",
                     info["about"], badge))
    for col, (d, names, p), v in zip(st.columns(len(preds)), preds, verdicts):
        col.html(ui.classifier(d, class_label(story, v), v == truth, top_rows(story, names, p, cfg["app"]["top_k"])))
    if "score" in ss and ss.score["rounds"]:
        sc = ss.score
        st.html(ui.card('<div class="ftw-kicker">🏆 Scoreboard · correct / flows this session</div>' + ui.scoreboard(
            [("🙋 You", sc["you"], sc["rounds"], True)]
            + [(f"{d['emoji']} {d['name']}", n, sc["rounds"], False) for (d, _, _), n in zip(preds, sc["det"])])))
    st.button("Draw another flow", type="primary", icon="🎲", on_click=new_flow, args=("mystery",))
    what_if(cfg, row, verdicts, tokenizers, models, train, device)
    next_button(story, 2)


def what_if(cfg, row, verdicts, tokenizers, models, train, device):
    story, ss = cfg["story"], st.session_state
    with st.expander("What-if analysis: edit features and re-run the classifiers", icon="🔧"):
        st.caption("Edited flows can fall outside the training distribution, so the classifiers may disagree. Which "
                   "edits flip a prediction shows what each model relies on.")
        tag = f"{ss.draw}_{ss.get('wi_reset', 0)}"
        changed = row.copy()
        cols = st.columns(len(story["what_if"]) + 1)
        units = story["byte_units"]
        formats = {"bytes": lambda x: logic.human_bytes(x, units), "ms": logic.human_ms, "count": lambda x: f"{x:,}"}
        for col, (feature, spec) in zip(cols, story["what_if"].items()):
            current = row[feature].iloc[0].item()
            options = sorted({*spec["options"], current})
            value = col.select_slider(spec["label"], options, value=current, format_func=formats[spec["format"]],
                                      key=f"wi_{feature}_{tag}")
            changed[feature] = pd.Series([value], index=changed.index).astype(row[feature].dtype)
        port_col = story["what_if_port"]
        current = int(row[port_col].iloc[0])
        ports = sorted({*story["port_names"], current})
        port = cols[-1].selectbox(port_col, ports, index=ports.index(current), key=f"wi_port_{tag}",
                                  format_func=lambda p: f"{p} · {service(story, p)[0]}")
        changed[port_col] = pd.Series([port], index=changed.index).astype(row[port_col].dtype)
        preds = predictions(cfg, changed, tokenizers, models, train, device)
        for col, (d, names, p), before in zip(st.columns(len(preds)), preds, verdicts):
            v = names[int(np.argmax(p))]
            col.html(ui.classifier(d, class_label(story, v), None, top_rows(story, names, p, cfg["app"]["top_k"]),
                                   changed=v != before))
        st.button("Reset", icon="🔄",
                  on_click=lambda: ss.update(wi_reset=ss.get("wi_reset", 0) + 1))


def step_crack(path, cfg, row):
    story, ss = cfg["story"], st.session_state
    pcfg, ecfg, _, _ = logic.attack_setup(cfg)
    col = story["port_guess"]["column"]
    port = row[col].iloc[0]
    target = logic.fingerprint(ecfg, col, port)
    example = story["port_guess"]["hints"][0]
    sep, length, algo = ecfg["format"]["separator"], ecfg["hash"]["length"], ecfg["hash"]["algorithm"]
    badge = cfg["story"]["encodings"][cfg["app"]["attack_encoding"]]["badge"]
    st.html(ui.lead("SecurityBERT calls PPFLE <b>privacy-preserving</b>: only hashes leave the network, never raw "
                    "values. But the hash is <b>unsalted</b> and the recipe is public, so anyone can hash candidate "
                    f"values and compare: a <b>dictionary attack</b>. Every {ui.e(badge)} token is computed like this:"))
    st.html(ui.card(f'<div class="ftw-recipe">{ui.e(algo)}("{ui.e(col)}{ui.e(sep)}{example}")[:{length}] → '
                    f'<b>{ui.e(logic.fingerprint(ecfg, col, example))}</b></div>'
                    '<div style="margin-top:10px;color:var(--muted)">Same input, same token, on any machine. No secret '
                    'key is involved.</div>'))
    left, right = st.columns([3, 2])
    with left:
        st.html(ui.card(f'<div class="ftw-kicker">🎯 Target: this flow\'s {ui.e(col)} token</div>'
                        f'<div class="ftw-target">{ui.e(target)}</div>'
                        '<div style="color:var(--muted);margin-top:6px">Guess the destination port: try a common one '
                        'or any number from 0 to 65,535.</div>'))
        g1, g2 = st.columns([2, 1], vertical_alignment="bottom")
        guess = g1.number_input("Your guess", min_value=0, max_value=65535, step=1, value=None,
                                key=f"pg_{ss.draw}", placeholder="e.g. 80")
        hint = st.pills("Common ports", story["port_guess"]["hints"], key=f"hint_{ss.draw}", wrap=True,
                        format_func=lambda p: f"{p} · {service(story, p)[0]}")
        if g2.button("Hash it", width="stretch") and guess is not None:
            ss.tries.append(int(guess))
        if hint is not None and (not ss.tries or ss.tries[-1] != hint):
            ss.tries.append(int(hint))
        hit = False
        for t in reversed(ss.tries[-6:]):
            fp = logic.fingerprint(ecfg, col, t)
            ok = fp == target
            hit |= ok
            st.html(f'<div class="ftw-try">{t:>5} → {ui.e(fp)} '
                    f'{"<span class=ok>✓ match</span>" if ok else "<span class=no>✗</span>"}</div>')
        if hit and not ss.get(f"party_{ss.draw}"):
            ss[f"party_{ss.draw}"] = True
            st.balloons()
    with right:
        if st.button("Brute-force every port", type="primary", icon="⚡", width="stretch"):
            ss.cracked_port = logic.crack_port(cfg, col, target)
        if ss.cracked_port:
            value, tries, secs = ss.cracked_port
            name = service(story, int(value))[0] if value is not None else ""
            st.html(ui.card(f'<div class="ftw-kicker">⚡ Brute force</div>'
                            f'<div style="font-size:1.05rem">tried all <b>{tries:,}</b> candidate ports in '
                            f'<b>{secs:.2f} s</b></div><div class="ftw-target" style="margin-top:6px">{ui.e(value)}'
                            f'</div><div style="color:var(--muted)">{ui.e(name)}</div>'))
    features = pcfg["data"]["columns"]["features"]
    st.markdown(f"#### 🗝️ Dictionary attack on all {len(features)} features")
    st.caption(" · ".join(lv["label"] for lv in pcfg["levels"]) + " (the levels of Exp 4, tried in this order)")
    if st.button("Run the attack on this flow", type="primary", icon="🔓"):
        board, status = st.empty(), st.empty()
        results, t0 = [], time.perf_counter()
        public = load_public(path)
        board.html(ui.attack_tiles(results, features, now=features[0]))
        for i, r in enumerate(logic.crack_flow(cfg, row, public)):
            results.append(r)
            nxt = features[i + 1] if i + 1 < len(features) else None
            board.html(ui.attack_tiles(results, features, now=nxt))
            status.caption(f"{sum(x['guess'] is not None for x in results)} of {len(results)} features recovered so "
                           "far…")
        status.empty()
        ss.attack = (results, time.perf_counter() - t0)
    elif ss.attack:
        st.html(ui.attack_tiles(ss.attack[0], features))
    if ss.attack:
        results, secs = ss.attack
        found = sum(r["guess"] is not None for r in results)
        right_ = sum(r["correct"] for r in results)
        st.html(ui.takeaway(f"<b>{found} of {len(results)}</b> features recovered in <b>{secs:.0f} s</b> on this "
                            f"laptop, <b>{right_}</b> of them verified against the true value. Hashing values that "
                            "can be enumerated hides nothing; a secret key (HMAC) would stop this attack."))
    next_button(story, 3)


def leak_quiz(path, cfg):
    table, spellings = leak_table(path)
    gap = {c: max(t[("benign", s)] - t[("attack", s)] for s in spellings) for c, t in table.items()}
    column = max(gap, key=gap.get)
    shares = table[column]
    normal = max(spellings, key=lambda s: shares[("benign", s)])
    attack = max(spellings, key=lambda s: shares[("attack", s)])
    ds = cfg["display"]["datasets"][cfg["leak"]["fixed_dataset"]]
    st.html(ui.card(f'<div class="ftw-kicker">🧩 Quiz · label leak</div><div style="font-size:1.08rem">Two rows from '
                    f'the published {ui.e(ds)} file, one benign and one attack. Which one is the attack?</div>'))
    rows = [normal, attack][::(-1) ** st.session_state.draw]   # order changes with every new flow
    for col, letter, spelling in zip(st.columns(2), "AB", rows):
        if col.button(f'Row {letter}:  {column} = "{spelling}"', width="stretch", key=f"leak_{letter}"):
            st.session_state.leak_pick = spelling
    if st.session_state.get("leak_pick") is not None:
        pct = cfg["app"]["percent_decimals"]
        letter = "AB"[rows.index(attack)]
        verdict = "Correct. " if st.session_state.leak_pick == attack else f"It was row {letter}. "
        st.html(ui.takeaway(
            f"{verdict}In the published file, this empty field is <code>\"{ui.e(normal)}\"</code> in "
            f"{100 * shares[('benign', normal)]:.{pct}f} % of benign rows but <code>\"{ui.e(attack)}\"</code> in "
            f"{100 * shares[('attack', attack)]:.{pct}f} % of attack rows. A classifier can learn the formatting "
            "instead of the traffic: a label leak."))


def step_findings(path, cfg):
    story, app = cfg["story"], cfg["app"]
    numbers = load_numbers(path)
    st.html(ui.lead(f"What we found across <b>{len(cfg['display']['datasets'])} datasets</b>, four encodings and "
                    "three models (XGBoost, small BERT, DistilBERT):"))
    fill = lambda t: logic.fill(t, numbers, app["percent_decimals"], story["missing_number"])
    for pair in (story["findings"][:2], story["findings"][2:]):
        for col, f in zip(st.columns(2), pair):
            col.html(ui.finding(f, fill(f["big"]), fill(f["text"])))
    leak_quiz(path, cfg)
    with st.expander("Charts and verdicts from the report", icon="📊"):
        heading = None
        for ln in Path(app["summary_file"]).read_text().splitlines():
            if ln.startswith("## "):
                heading = ln[3:]
            elif heading and (m := re.match(r"\*\*Verdict: (.*?)\*\*(.*)", ln)):
                st.markdown(f"- **{heading}** → **{m[1]}**{m[2]}")
        for file, caption in app["figures"]:
            st.image(file, caption=caption, width="stretch")


def main():
    path = config_path()
    st.set_page_config(page_title=load_demo_config(path)["app"]["title"], page_icon="🛡️", layout="wide")
    st.html(ui.CSS)
    cfg, device, splits, tokenizers, models, missing = load_everything(path)
    load_public(path)   # warm the attack dictionary before the first visitor runs the attack
    story, ss = cfg["story"], st.session_state
    train_name, _, test_name = split_roles(cfg["bert_cfg"]["data"])
    test, train = splits[test_name], splits[train_name]
    if "draw" not in ss:
        new_flow("mystery")
    idx = logic.draw_flow(cfg, test, ss.draw, ss.wanted)
    row = test.loc[[idx], features_of(cfg)]
    truth = logic.classes_of(cfg, test).loc[idx]
    classes = logic.classes_of(cfg, test).nunique()
    st.html(ui.hero("Flows to Words · Does the encoding matter?", cfg["app"]["title"],
                    "We serialise network flow records as text in four encodings, train language models on them and "
                    "compare them with XGBoost on the raw table: within one dataset, across datasets, and for "
                    "privacy.",
                    [f"Demo data: {cfg['display']['datasets'][cfg['dataset']]} test split",
                     f"{len(features_of(cfg))} features per flow", f"{classes} classes",
                     "Classifiers: " + ", ".join(d["name"] for d in story["classifiers"])]))
    if "step" not in ss:
        asked = st.query_params.get("step", "1")
        ss.step = min(max(int(asked) - 1, 0), len(story["steps"]) - 1) if asked.isdigit() else 0
    step = st.segmented_control("Step", list(range(len(story["steps"]))), key="step", required=True,
                                format_func=lambda i: f"{i + 1}  ·  {story['steps'][i]}",
                                label_visibility="collapsed", width="stretch")
    st.space("small")
    if step == 0:
        step_meet(cfg, row, truth)
    elif step == 1:
        step_words(cfg, row, tokenizers)
    elif step == 2:
        step_detect(cfg, row, truth, tokenizers, models, missing, train, device)
    elif step == 3:
        step_crack(path, cfg, row)
    else:
        step_findings(path, cfg)


main()
