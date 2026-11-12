"""Numbers for the report as LaTeX macros, so no result is ever typed into the text (Daniel's rule, 2026-11-01).

Every value gets a key such as `exp2.xgboost.none.nf_unsw_nb15.macro_f1`. The generated .tex file has one line per key:
    \resset{<key>}{<text as printed>}{<raw number or empty>}
report/main.tex defines \resset, \res{key} (printed text; a red ?? if the key is missing) and \resnum{key} (the raw
number, for \fpeval arithmetic such as differences). A Markdown twin lists every key and value for looking them up.
"""

import json
import math
import re
from glob import glob
from pathlib import Path

from src.config import _read_yaml

TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
KEY_FORBIDDEN = set("\\{}%#")
USED_KEY = re.compile(r"\\res(?:num|percentof)?\{([^{}]+)\}|\\res(?:diff|pct)\{([^{}]+)\}\{([^{}]+)\}")


def tex_escape(text):
    return "".join(TEX_SPECIAL.get(ch, ch) for ch in str(text))


class Numbers:
    """Collects key -> (LaTeX text, raw number) pairs. `nc` is a config `numbers` section."""

    def __init__(self, nc):
        self.nc = nc
        self.items = {}

    def add(self, key, text, raw=None):
        key = str(key)
        if KEY_FORBIDDEN & set(key):
            raise ValueError(f"number key {key!r} contains one of {sorted(KEY_FORBIDDEN)}")
        if key in self.items and self.items[key] != (text, raw):
            raise ValueError(f"number key {key!r} set twice with different values")
        self.items[key] = (text, raw)

    def text(self, key, value):
        self.add(key, tex_escape(value))

    def value(self, key, v, decimals=None):
        """A number: ints with thousands separators, floats rounded to `decimals` (default float_decimals)."""
        if isinstance(v, bool):
            return self.text(key, "yes" if v else "no")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return
        if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and decimals == 0):
            self.add(key, f"{int(v):,}".replace(",", "{,}"), int(v))
        else:
            d = self.nc["float_decimals"] if decimals is None else decimals
            if decimals is None and v and abs(v) < 10 ** -d:   # a learning rate from a config: 2e-05 -> $2 \times 10^{-5}$
                mantissa, exponent = f"{v:.{d}e}".split("e")
                self.add(key, f"${float(mantissa):g} \\times 10^{{{int(exponent)}}}$", float(v))
            else:
                self.add(key, f"{v:.{d}f}", float(v))

    def percent(self, key, share):
        """A share 0-1 written as a percentage number (the text adds the % sign)."""
        self.value(key, 100 * share, self.nc["percent_decimals"])

    def stat(self, key, s, fmt):
        """A mean over seeds: `key` = 'mean ± std' (or mean + one-seed marker), plus key.mean/.std/.n."""
        d = fmt["decimals"]
        if s.n == 1:
            self.add(key, f"{s.mean:.{d}f}{fmt['tex_one_seed_marker']}", s.mean)
        else:
            self.add(key, f"{s.mean:.{d}f} $\\pm$ {s.std:.{d}f}", s.mean)
        self.value(f"{key}.mean", s.mean, d)
        if not math.isnan(s.std):
            self.value(f"{key}.std", s.std, d)
        self.value(f"{key}.n", s.n)

    def flatten(self, prefix, obj, list_keys=()):
        """Every number (and short string) inside a JSON/YAML object, keyed by its path. Lists of records are keyed
        by the values of `list_keys` when the records have them, otherwise by position."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                self.flatten(f"{prefix}.{k}", v, list_keys)
        elif isinstance(obj, list):
            self.value(f"{prefix}.count", len(obj))
            for i, v in enumerate(obj):
                if isinstance(v, dict) and list_keys and all(k in v for k in list_keys):
                    sub = ".".join(str(v[k]) for k in list_keys)
                    self.flatten(f"{prefix}.{sub}", {k: x for k, x in v.items() if k not in list_keys}, list_keys)
                else:
                    self.flatten(f"{prefix}.{i}", v, list_keys)
        elif isinstance(obj, (int, float)):
            self.value(prefix, obj)
        elif isinstance(obj, str) and len(obj) <= self.nc["max_string_length"]:
            self.text(prefix, obj)

    def add_sources(self, sources):
        """Flatten the stats/config files listed in the config (`sources`); missing files are skipped."""
        for src in sources:
            paths = sorted(Path().glob(src["glob"])) if "glob" in src else [Path(src["path"])]
            for path in paths:
                if not path.exists():
                    continue
                obj = json.loads(path.read_text()) if path.suffix == ".json" else _read_yaml(path)
                prefix = f"{src['prefix']}.{path.stem}" if "glob" in src else src["prefix"]
                self.flatten(prefix, obj, src.get("list_keys", ()))

    def write(self, title):
        nc = self.nc
        keys = sorted(self.items)
        tex = [f"% {title}: generated, do not edit. Use \\res{{key}} in the text; keys are listed in {nc['md']}.",
               f"\\def\\resdecimals{{{nc['float_decimals']}}}", f"\\def\\respctdecimals{{{nc['percent_decimals']}}}"]
        tex += [f"\\{nc['macro']}{{{k}}}{{{self.items[k][0]}}}{{{'' if self.items[k][1] is None else repr(self.items[k][1])}}}"
                for k in keys]
        md = [f"# {title}", "", (f"Generated; {len(keys)} keys. In LaTeX: `\\res{{key}}` prints the value, "
              "`\\resnum{key}` gives the raw number (e.g. for `\\resdiff{a}{b}`)."), "", "| Key | Value |", "|---|---|"]
        md += [f"| `{k}` | {self.items[k][0].replace('|', '/')} |" for k in keys]
        for path, lines in ((nc["tex"], tex), (nc["md"], md)):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("\n".join(lines) + "\n")
        return [Path(nc["tex"]), Path(nc["md"])]


def missing_report_keys(nc):
    """Keys (other than <pattern> placeholders) used as \\res{...} (or \\resnum/\\resdiff/\\respct) in `report_files` (paths or
    glob patterns) that no file in `all_numbers` defines."""
    defined = set()
    for path in map(Path, nc["all_numbers"]):
        if path.exists():
            defined |= set(re.findall(r"^\\" + nc["macro"] + r"\{([^{}]+)\}", path.read_text(), flags=re.MULTILINE))
    used = set()
    for path in sorted({Path(p) for pattern in nc["report_files"] for p in glob(pattern)}):
        if path.is_file():
            text = path.read_text()
            if path.suffix == ".tex":                  # LaTeX comments and macro definitions (#1) are not uses
                text = re.sub(r"(?<!\\)%.*", "", text)
            used |= {k for groups in USED_KEY.findall(text) for k in groups if k and not set("<#") & set(k)}
    return sorted(used - defined)
