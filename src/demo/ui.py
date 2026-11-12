"""HTML and CSS of the live demo (src/demo/app.py). Pure functions: data in, HTML string out.

No external fonts or scripts, so the app also works offline at the station. Colours that mean something (the
encodings and the three classifiers) come from configs/demo.yaml; the rest is the stylesheet below.
"""

from html import escape

CSS = """
<style>
:root {
  --ink: #0f172a; --ink-2: #334155; --muted: #5a6478; --faint: #8a93a6; --line: #e2e7ef; --card: #ffffff;
  --bg: #f4f6fa; --tint: #f7f9fc; --accent: #184f95; --accent-2: #2a78d6; --accent-soft: #e8f0fb;
  --good: #0f7b55; --good-soft: #e2f4ec; --bad: #c03b12; --bad-soft: #fdeee6; --mark: #ffe27a;
  --mono: ui-monospace, "JetBrains Mono", "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  --shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -6px rgba(15,23,42,.08);
}
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
  display: none !important; }
.block-container { padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1480px; }
.ftw-mono { font-family: var(--mono); }

/* header */
.ftw-hero { position: relative; overflow: hidden; border-radius: 22px; padding: 26px 34px 24px 34px; color: #fff;
  background: radial-gradient(1200px 300px at 85% -40%, rgba(255,255,255,.18), transparent 60%),
              linear-gradient(115deg, #0c2340 0%, #164a8c 55%, #2a78d6 100%);
  box-shadow: 0 12px 32px -12px rgba(12,35,64,.45); margin-bottom: 18px; }
.ftw-hero::after { content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .5;
  background-image: radial-gradient(rgba(255,255,255,.16) 1px, transparent 1px); background-size: 18px 18px;
  -webkit-mask-image: linear-gradient(90deg, transparent 35%, #000 100%); mask-image: linear-gradient(90deg, transparent 35%, #000 100%); }
.ftw-hero .kicker { font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase;
  color: #a9c8f2; }
.ftw-hero h1 { font-size: 2.35rem; line-height: 1.12; margin: .3rem 0 .45rem 0; color: #fff; letter-spacing: -.02em;
  padding: 0; font-weight: 800; }
.ftw-hero p { font-size: 1.08rem; color: #dbe7f8; margin: 0; max-width: 960px; line-height: 1.5; }
.ftw-hero .chips { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; position: relative; z-index: 1; }
.ftw-hero .chip { font-size: .82rem; font-weight: 600; padding: 4px 11px; border-radius: 999px;
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22); color: #eef4fc; }

.ftw-lead { font-size: 1.12rem; color: var(--ink-2); margin: .3rem 0 1rem 0; max-width: 1060px; line-height: 1.6; }
.ftw-lead b { color: var(--ink); }
.ftw-lead code, .ftw-takeaway code, .ftw-card code { font-family: var(--mono); font-size: .88em; background: var(--bg);
  border: 1px solid var(--line); border-radius: 6px; padding: 1px 6px; color: var(--ink); }
.ftw-card { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 18px 20px;
  box-shadow: var(--shadow); margin-bottom: 14px; }
.ftw-kicker { font-size: .74rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
  color: var(--faint); margin-bottom: 8px; }
.ftw-row { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.ftw-pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: .92rem;
  background: var(--accent-soft); color: var(--accent); }
.ftw-pill.good { background: var(--good-soft); color: var(--good); }
.ftw-pill.bad { background: var(--bad-soft); color: var(--bad); }
.ftw-pill.hidden { background: #efeafb; color: #553c9a; }
.ftw-takeaway { border: 1px solid #cfe0f6; border-left: 4px solid var(--accent); background: var(--accent-soft);
  padding: 14px 18px; border-radius: 12px; font-size: 1.04rem; color: var(--ink); margin: 6px 0 16px 0;
  line-height: 1.55; }

/* flow card (step 1, 3) */
.ftw-conn { display: grid; grid-template-columns: 1fr 2.3fr 1fr; align-items: center; gap: 18px; }
.ftw-node { text-align: center; padding: 16px 10px; border-radius: 14px; background: var(--tint);
  border: 1px solid var(--line); }
.ftw-node .role { font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--faint); }
.ftw-node .name { font-weight: 800; font-size: 1.25rem; margin-top: 4px; color: var(--ink); }
.ftw-node .sub { color: var(--muted); font-size: .88rem; font-family: var(--mono); }
.ftw-wire { display: flex; flex-direction: column; gap: 16px; }
.ftw-arrow { position: relative; text-align: center; font-size: .98rem; color: var(--ink-2); padding-bottom: 9px; }
.ftw-arrow::after { content: ""; position: absolute; left: 3%; right: 3%; bottom: 0; height: 3px; border-radius: 3px;
  background: linear-gradient(90deg, #bcd5f5, var(--accent-2)); }
.ftw-arrow.back::after { background: linear-gradient(270deg, #bcd5f5, var(--accent-2)); }
.ftw-arrow .head { position: absolute; bottom: -8px; font-size: .9rem; color: var(--accent-2); }
.ftw-arrow .head.r { right: 1%; } .ftw-arrow .head.l { left: 1%; }
.ftw-arrow b { font-size: 1.12rem; color: var(--ink); }
.ftw-arrow .f { font-family: var(--mono); font-size: .74rem; color: var(--faint); margin-left: 6px; }
.ftw-proto { text-align: center; font-size: .9rem; color: var(--muted); }
.ftw-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 18px; }
.ftw-stat { background: var(--tint); border: 1px solid var(--line); border-radius: 12px; padding: 10px 14px; }
.ftw-stat .k { color: var(--muted); font-size: .82rem; }
.ftw-stat .k .f { font-family: var(--mono); font-size: .7rem; color: var(--faint); margin-left: 4px; }
.ftw-stat .v { font-size: 1.28rem; font-weight: 700; color: var(--ink); }
.ftw-classes { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 4px 18px; }
.ftw-classes div { font-size: .93rem; color: var(--muted); padding: 3px 0; }
.ftw-classes b { color: var(--ink); }

/* original flow grid (step 2) */
.ftw-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(196px, 1fr)); gap: 6px; }
.ftw-feat { border: 1px solid var(--line); background: var(--tint); border-radius: 9px; padding: 5px 9px;
  transition: background .2s, border-color .2s; }
.ftw-feat .n { font-family: var(--mono); font-size: .68rem; color: var(--faint); white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.ftw-feat .v { font-family: var(--mono); font-size: .92rem; font-weight: 700; color: var(--ink); white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
.ftw-feat.done { background: #fff; }
.ftw-feat.done .v { color: var(--muted); font-weight: 600; }
.ftw-feat.now { background: var(--accent-soft); border-color: var(--accent-2); box-shadow: 0 0 0 3px rgba(42,120,214,.18); }
.ftw-feat.now .v, .ftw-feat.hl .v { color: var(--accent); }
.ftw-feat.hl { background: #fff8d6; border-color: #f0c94a; box-shadow: 0 0 0 3px rgba(240,201,74,.25); }
.ftw-progress { font-size: .88rem; color: var(--muted); }
.ftw-progress .bar { display: inline-block; width: 140px; height: 6px; border-radius: 6px; background: var(--line);
  vertical-align: middle; margin-left: 8px; overflow: hidden; }
.ftw-progress .bar i { display: block; height: 100%; background: var(--accent-2); border-radius: 6px; }

/* how one feature is encoded (step 2) */
.ftw-xform .feat { font-family: var(--mono); font-size: 1.05rem; color: var(--ink); margin-bottom: 10px; }
.ftw-xform .feat b { color: var(--accent); }
.ftw-xform .feat .about { font-family: "Source Sans", "Source Sans Pro", sans-serif; color: var(--muted); font-size: .9rem; margin-left: 10px; }
.ftw-xrow { display: grid; grid-template-columns: 2.6em minmax(0, .85fr) 1.2em minmax(0, 1.15fr); align-items: center;
  gap: 10px; padding: 6px 0; border-top: 1px dashed var(--line); }
.ftw-xrow .how { font-family: var(--mono); font-size: .86rem; color: var(--muted); overflow-wrap: anywhere; }
.ftw-xrow .arrow { color: var(--faint); text-align: center; }
.ftw-xrow .out { font-family: var(--mono); font-size: .9rem; color: var(--ink); overflow-wrap: anywhere; }
.ftw-xrow .out b { color: var(--c); }
.ftw-badge { display: inline-flex; align-items: center; justify-content: center; min-width: 2.3em; padding: 2px 7px;
  border-radius: 7px; font-weight: 800; font-size: .8rem; color: #fff; background: var(--c); letter-spacing: .02em; }

/* encodings (step 2) */
.ftw-encgrid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.ftw-enc { border-top: 4px solid var(--c); margin-bottom: 0; }
.ftw-enc .hd { display: flex; align-items: center; gap: 10px; }
.ftw-enc h4 { margin: 0; padding: 0; font-size: 1.12rem; color: var(--ink); }
.ftw-enc .tag { margin-left: auto; font-size: .76rem; font-weight: 700; color: var(--c); background: var(--tint);
  border: 1px solid var(--line); border-radius: 999px; padding: 2px 10px; white-space: nowrap; }
.ftw-enc .about { color: var(--muted); font-size: .9rem; margin: 8px 0 10px 0; min-height: 2.8em; line-height: 1.45; }
.ftw-text { background: var(--tint); border: 1px solid var(--line); border-radius: 10px; padding: 11px 13px;
  font-size: .9rem; line-height: 1.75; word-break: break-word; color: var(--ink-2); }
.ftw-text.mono { font-family: var(--mono); font-size: .8rem; line-height: 1.85; }
.ftw-text mark { background: var(--mark); color: var(--ink); padding: 1px 3px; border-radius: 4px;
  box-shadow: 0 0 0 2px var(--mark); font-weight: 700; }
.ftw-text .now { background: var(--c); color: #fff; border-radius: 4px; padding: 1px 3px; font-weight: 700; }
.ftw-text .pend { color: transparent; background: #e6eaf1; border-radius: 4px; }
.ftw-meter { margin-top: 10px; font-size: .84rem; color: var(--muted); display: flex; align-items: center; gap: 10px; }
.ftw-meter .bar { flex: 1; height: 6px; border-radius: 6px; background: var(--line); overflow: hidden; }
.ftw-meter .fill { height: 100%; border-radius: 6px; background: var(--c); }

/* classifiers (step 3) */
.ftw-det { border-top: 4px solid var(--c); }
.ftw-det .who { font-weight: 800; font-size: 1.1rem; color: var(--ink); }
.ftw-det .reads { color: var(--faint); font-size: .85rem; margin-bottom: 12px; }
.ftw-det .verdict { font-size: 1.5rem; font-weight: 800; color: var(--ink); margin: 0 0 8px 0; }
.ftw-det .changed { display: inline-block; margin-left: 8px; font-size: .72rem; font-weight: 700; color: #fff;
  background: var(--bad); border-radius: 999px; padding: 2px 9px; vertical-align: middle; }
.ftw-bar { display: grid; grid-template-columns: 8.5em 1fr 3.2em; align-items: center; gap: 8px; margin: 5px 0;
  font-size: .88rem; color: var(--muted); }
.ftw-bar .track { height: 8px; background: var(--line); border-radius: 8px; overflow: hidden; }
.ftw-bar .fill { height: 100%; background: var(--c); border-radius: 8px; }
.ftw-bar span:last-child { text-align: right; font-variant-numeric: tabular-nums; }
.ftw-truth { display: flex; align-items: center; gap: 16px; }
.ftw-truth .big { font-size: 2.4rem; line-height: 1; }
.ftw-truth .tag { font-size: .8rem; font-weight: 800; letter-spacing: .08em; padding: 6px 12px; border-radius: 8px; }
.ftw-truth .tag.good { background: var(--good-soft); color: var(--good); }
.ftw-truth .tag.bad { background: var(--bad-soft); color: var(--bad); }
.ftw-truth .t { font-size: 1.45rem; font-weight: 800; color: var(--ink); }
.ftw-truth .s { color: var(--muted); }
.ftw-score { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
.ftw-score .tile { background: var(--tint); border: 1px solid var(--line); border-radius: 12px; padding: 10px 14px;
  text-align: center; }
.ftw-score .tile.you { background: var(--accent-soft); border-color: #cfe0f6; }
.ftw-score .n { font-size: 1.7rem; font-weight: 800; color: var(--ink); font-variant-numeric: tabular-nums; }
.ftw-score .n small { font-size: 1rem; color: var(--faint); font-weight: 600; }
.ftw-score .l { font-size: .84rem; color: var(--muted); }

/* hash attack (step 4) */
.ftw-recipe { font-family: var(--mono); font-size: 1rem; background: var(--tint); border: 1px solid var(--line);
  border-radius: 10px; padding: 11px 15px; display: inline-block; color: var(--ink-2); }
.ftw-recipe b { color: var(--accent); }
.ftw-target { font-family: var(--mono); font-size: 1.7rem; font-weight: 700; letter-spacing: .05em; color: var(--accent); }
.ftw-try { font-family: var(--mono); font-size: .92rem; margin: 3px 0; color: var(--ink-2); }
.ftw-try .ok { color: var(--good); font-weight: 700; } .ftw-try .no { color: var(--bad); }
.ftw-tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(176px, 1fr)); gap: 7px; }
.ftw-tile { border-radius: 10px; padding: 6px 10px; background: var(--tint); border: 1px solid var(--line);
  min-height: 62px; }
.ftw-tile .c { font-family: var(--mono); font-size: .68rem; color: var(--faint); white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.ftw-tile .h { font-family: var(--mono); font-size: .76rem; color: #98a2b3; }
.ftw-tile .v { font-family: var(--mono); font-size: 1rem; font-weight: 800; color: var(--good); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.ftw-tile .lv { font-size: .66rem; font-weight: 700; color: var(--good); opacity: .75; text-transform: uppercase;
  letter-spacing: .06em; }
.ftw-tile.found { background: var(--good-soft); border-color: #b5e2cd; }
.ftw-tile.miss { background: var(--bad-soft); border-color: #f4c7b2; }
.ftw-tile.miss .v { color: var(--bad); font-size: .85rem; }
.ftw-tile.now { border-color: var(--accent-2); box-shadow: 0 0 0 3px rgba(42,120,214,.18); background: #fff; }

/* results (step 5) */
.ftw-find { border-top: 4px solid var(--accent-2); }
.ftw-find-top { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.ftw-find-top .e { font-size: 1.7rem; line-height: 1; }
.ftw-find-top .ftw-kicker { margin-bottom: 0; }
.ftw-find h3 { margin: 0 0 8px 0; padding: 0; font-size: 1.22rem; color: var(--ink); }
.ftw-find .big { font-size: 2.5rem; font-weight: 800; color: var(--accent); line-height: 1.05;
  font-variant-numeric: tabular-nums; }
.ftw-find .bl { color: var(--faint); font-size: .88rem; margin: 2px 0 12px 0; }
.ftw-find p { color: var(--ink-2); font-size: .98rem; line-height: 1.6; margin: 0; }

/* streamlit widgets */
div[data-testid="stButtonGroup"] button p { font-size: 1rem; }
button[data-variant="segmented_control"] { padding-top: .55rem; padding-bottom: .55rem; background: var(--card); }
button[data-variant="segmented_control"] p { color: var(--muted); font-weight: 600; }
button[data-variant="segmented_control"][aria-checked="true"] { background: var(--accent) !important;
  border-color: var(--accent) !important; }
button[data-variant="segmented_control"][aria-checked="true"] p { color: #fff !important; }
.stButton > button { border-radius: 10px; font-weight: 600; padding: .48rem 1.05rem; }
div[data-testid="stExpander"] details { border-radius: 14px; border-color: var(--line); background: var(--card); }
@media (max-width: 900px) {
  .ftw-encgrid { grid-template-columns: 1fr; }
  .ftw-conn { grid-template-columns: 1fr; }
  .ftw-stats, .ftw-score { grid-template-columns: repeat(2, 1fr); }
  .ftw-xrow { grid-template-columns: 2.6em 1fr; }
  .ftw-xrow .arrow { display: none; }
}
</style>
"""


def e(text):
    return escape(str(text))


def card(inner, extra_class="", style=""):
    return f'<div class="ftw-card {extra_class}" style="{style}">{inner}</div>'


def hero(kicker, title, subtitle, chips):
    chip_html = "".join(f'<span class="chip">{e(c)}</span>' for c in chips)
    return (f'<div class="ftw-hero"><div class="kicker">{e(kicker)}</div><h1>{e(title)}</h1><p>{e(subtitle)}</p>'
            f'<div class="chips">{chip_html}</div></div>')


def lead(html_text):
    """A short intro paragraph; `html_text` may contain <b>/<code> (trusted, written in app.py)."""
    return f'<div class="ftw-lead">{html_text}</div>'


def takeaway(html_text):
    return f'<div class="ftw-takeaway">{html_text}</div>'


def pill(text, kind=""):
    return f'<span class="ftw-pill {kind}">{e(text)}</span>'


def badge(info):
    return f'<span class="ftw-badge" style="--c:{info["color"]}">{e(info["badge"])}</span>'


def connection(service, port_text, proto, out, back, stats):
    """Client -> server picture with the traffic in each direction (`out`/`back` = (packets, bytes, feature names)),
    then stat tiles [(label, feature, value)]."""
    tiles = "".join(f'<div class="ftw-stat"><div class="k">{e(k)}<span class="f">{e(f)}</span></div>'
                    f'<div class="v">{e(v)}</div></div>' for k, f, v in stats)
    return card(f"""
      <div class="ftw-conn">
        <div class="ftw-node"><div class="role">Source</div><div class="name">Client</div>
          <div class="sub" style="font-family:inherit">initiated the flow</div></div>
        <div class="ftw-wire">
          <div class="ftw-arrow"><b>{e(out[0])}</b> packets · <b>{e(out[1])}</b><span class="f">{e(out[2])}</span>
            <span class="head r">▶</span></div>
          <div class="ftw-arrow back"><b>{e(back[0])}</b> packets · <b>{e(back[1])}</b><span class="f">{e(back[2])}</span>
            <span class="head l">◀</span></div>
          <div class="ftw-proto">over <b>{e(proto)}</b></div>
        </div>
        <div class="ftw-node"><div class="role">Destination</div><div class="name">{e(service)}</div>
          <div class="sub">{e(port_text)}</div></div>
      </div>
      <div class="ftw-stats">{tiles}</div>""")


def class_list(classes):
    """{class: {emoji, name, about}} as a two-column list of definitions."""
    return ('<div class="ftw-classes">' + "".join(f'<div>{e(i["emoji"])} <b>{e(i["name"])}</b> · {e(i["about"])}</div>'
                                                  for i in classes.values()) + "</div>")


def flow_grid(features, values, info, status):
    """The original flow as one tile per feature. `status[feature]` = '' | done | now | hl."""
    tiles = "".join(f'<div class="ftw-feat {status.get(c, "")}" title="{e(info[c])}"><div class="n">{e(c)}</div>'
                    f'<div class="v">{e(values[c])}</div></div>' for c in features)
    return f'<div class="ftw-grid">{tiles}</div>'


def progress(done, total):
    return (f'<span class="ftw-progress">{done} / {total} features encoded'
            f'<span class="bar"><i style="width:{100 * done / max(total, 1):.0f}%"></i></span></span>')


def encoded_text(text, spans, status, highlight=None):
    """`text` with each feature's span styled by `status[feature]` (pending | now | done); `highlight` = a feature
    shown in <mark>. Text outside the spans (joiners, sentence template) is always shown."""
    out, pos = [], 0
    for feature, (a, b) in sorted(spans.items(), key=lambda kv: kv[1][0]):
        out.append(e(text[pos:a]))
        part, state = e(text[a:b]), status.get(feature, "done")
        if feature == highlight and state == "done":
            out.append(f"<mark>{part}</mark>")
        elif state in ("pending", "now"):
            out.append(f'<span class="{"pend" if state == "pending" else "now"}">{part}</span>')
        else:
            out.append(part)
        pos = b
    out.append(e(text[pos:]))
    return "".join(out)


def encoding_card(info, text_html, mono, n_tokens=None, max_tokens=None):
    """One encoding: badge, name, explanation, the text and (when given) its length in BPE tokens."""
    meter = ""
    if n_tokens is not None:
        width = 100 * n_tokens / max(max_tokens, 1)
        meter = (f'<div class="ftw-meter"><span><b>{n_tokens}</b> BPE tokens</span>'
                 f'<div class="bar"><div class="fill" style="width:{width:.0f}%"></div></div></div>')
    return card(f"""
      <div class="hd">{badge(info)}<h4>{e(info['emoji'])} {e(info['name'])}</h4><span class="tag">{e(info['tag'])}</span></div>
      <div class="about">{e(info['about'])}</div>
      <div class="ftw-text {'mono' if mono else ''}">{text_html}</div>{meter}""", "ftw-enc", f"--c:{info['color']}")


def encoding_grid(cards):
    return f'<div class="ftw-encgrid">{"".join(cards)}</div>'


def transform(feature, value, about, rows):
    """How one feature is written by each encoding. rows = [(encoding info, how (html), output (html))]."""
    lines = "".join(f'<div class="ftw-xrow" style="--c:{info["color"]}">{badge(info)}<div class="how">{how}</div>'
                    f'<div class="arrow">→</div><div class="out">{out}</div></div>' for info, how, out in rows)
    return card(f'<div class="ftw-kicker">One feature, four encodings</div>'
                f'<div class="feat">{e(feature)} = <b>{e(value)}</b><span class="about">{e(about)}</span></div>'
                f"{lines}", "ftw-xform")


def bars(rows, color):
    """[(label, probability)] as horizontal bars."""
    out = []
    for label, p in rows:
        out.append(f'<div class="ftw-bar" style="--c:{color}"><span>{e(label)}</span>'
                   f'<div class="track"><div class="fill" style="width:{100 * p:.0f}%"></div></div>'
                   f'<span>{100 * p:.0f} %</span></div>')
    return "".join(out)


def classifier(d, verdict, correct, bar_rows, changed=False):
    mark = "" if correct is None else (pill("✓ correct", "good") if correct else pill("✗ wrong", "bad"))
    flag = '<span class="changed">prediction changed</span>' if changed else ""
    return card(f"""
      <div class="who">{e(d['emoji'])} {e(d['name'])}</div><div class="reads">{e(d['reads'])}</div>
      <div class="ftw-kicker" style="margin-bottom:2px">Prediction</div>
      <div class="verdict">{e(verdict)}{flag}</div>{mark}
      <div style="margin-top:12px">{bars(bar_rows, d['color'])}</div>""",
                "ftw-det", f"--c:{d['color']}")


def truth(emoji, is_benign, title, sub, badge_html=""):
    tag = '<span class="tag good">BENIGN</span>' if is_benign else '<span class="tag bad">ATTACK</span>'
    return card(f'<div class="ftw-truth"><div class="big">{e(emoji)}</div>{tag}<div><div class="t">{e(title)}</div>'
                f'<div class="s">{e(sub)}</div></div><div style="margin-left:auto">{badge_html}</div></div>')


def scoreboard(tiles):
    """[(label, correct, rounds, is_you)]."""
    inner = "".join(f'<div class="tile {"you" if you else ""}"><div class="n">{c}<small> / {n}</small></div>'
                    f'<div class="l">{e(label)}</div></div>' for label, c, n, you in tiles)
    return f'<div class="ftw-score">{inner}</div>'


def attack_tiles(results, features, now=None):
    """One tile per feature: waiting (hash only), recovered (value + attack level) or not recovered."""
    out = []
    by_col = {r["column"]: r for r in results}
    for c in features:
        r = by_col.get(c)
        if r is None:
            out.append(f'<div class="ftw-tile {"now" if c == now else ""}"><div class="c">{e(c)}</div>'
                       f'<div class="h">{"hashing candidates…" if c == now else "waiting"}</div></div>')
        elif r["guess"] is not None:
            out.append(f'<div class="ftw-tile found"><div class="c">{e(c)}</div>'
                       f'<div class="h">{e(r["token"])}</div><div class="v">{e(r["guess"])}</div>'
                       f'<div class="lv" title="{e(r["level"])}">{e(r["level"].split(":")[0])}</div></div>')
        else:
            out.append(f'<div class="ftw-tile miss"><div class="c">{e(c)}</div>'
                       f'<div class="h">{e(r["token"])}</div><div class="v">not recovered</div></div>')
    return f'<div class="ftw-tiles">{"".join(out)}</div>'


def finding(f, big, text):
    return card(f"""
      <div class="ftw-find-top"><span class="e">{e(f['emoji'])}</span><span class="ftw-kicker">{e(f['kicker'])}</span></div><h3>{e(f['title'])}</h3>
      <div class="big">{e(big)}</div><div class="bl">{e(f['big_label'])}</div><p>{e(text)}</p>""", "ftw-find")
