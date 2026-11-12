"""Report numbers as LaTeX macros (src/eval/numbers.py)."""

import json

import pytest

from src.eval.aggregate import Stat
from src.eval.numbers import Numbers, missing_report_keys, tex_escape

NC = {"tex": None, "md": None, "macro": "resset", "float_decimals": 3, "percent_decimals": 1,
      "max_string_length": 10}
FMT = {"decimals": 3, "tex_one_seed_marker": r"\textsuperscript{\dag}"}


def test_values_are_formatted_and_keep_the_raw_number():
    n = Numbers(NC)
    n.value("a", 200000)
    n.value("b", 0.99969)
    n.percent("c", 0.9445)
    n.value("d", True)
    assert n.items["a"] == ("200{,}000", 200000)
    assert n.items["b"] == ("1.000", 0.99969)
    assert n.items["c"][0] == "94.5" and n.items["c"][1] == pytest.approx(94.45)
    assert n.items["d"] == ("yes", None)
    n.value("lr", 2e-05)
    assert n.items["lr"] == (r"$2 \times 10^{-5}$", 2e-05)


def test_stats_get_mean_std_and_seed_count():
    n = Numbers(NC)
    n.stat("x", Stat(0.671, 0.003, 3), FMT)
    n.stat("y", Stat(0.556, float("nan"), 1), FMT)
    assert n.items["x"][0] == r"0.671 $\pm$ 0.003" and n.items["x.std"][0] == "0.003" and n.items["x.n"][1] == 3
    assert n.items["y"][0] == r"0.556\textsuperscript{\dag}" and "y.std" not in n.items


def test_flatten_keys_records_and_skips_long_strings():
    n = Numbers(NC)
    n.flatten("t", {"rows": [{"encoding": "e1", "dataset": "d", "max_len": 427}], "s": "short", "l": "x" * 11},
              list_keys=["encoding", "dataset"])
    assert n.items["t.rows.e1.d.max_len"] == ("427", 427)
    assert n.items["t.rows.count"] == ("1", 1)
    assert n.items["t.s"] == ("short", None) and "t.l" not in n.items


def test_conflicts_and_bad_keys_are_errors():
    n = Numbers(NC)
    n.value("k", 1)
    n.value("k", 1)                       # the same value twice is fine
    with pytest.raises(ValueError, match="twice"):
        n.value("k", 2)
    with pytest.raises(ValueError, match="contains"):
        n.value("bad{key", 1)


def test_sources_and_write(tmp_path):
    src = tmp_path / "stats.json"
    src.write_text(json.dumps({"ds": {"rows": 5}}))
    nc = {**NC, "tex": str(tmp_path / "n.tex"), "md": str(tmp_path / "n.md")}
    n = Numbers(nc)
    n.add_sources([{"prefix": "s", "path": str(src)}, {"prefix": "m", "path": str(tmp_path / "missing.json")}])
    n.text("name", "ddos_hoic & co")
    n.write("Test")
    tex = (tmp_path / "n.tex").read_text()
    assert r"\resset{s.ds.rows}{5}{5}" in tex
    assert r"\resset{name}{ddos\_hoic \& co}{}" in tex
    assert "`s.ds.rows`" in (tmp_path / "n.md").read_text()
    assert tex_escape("50%") == r"50\%"


def test_missing_report_keys(tmp_path):
    (tmp_path / "n.tex").write_text("\\resset{a.b}{1}{1}\n\\resset{c}{2}{2}\n")
    (tmp_path / "r.tex").write_text(r"x \res{a.b} y \resdiff{c}{d.e} \respct{a.b}{c} \resnum{f}")
    nc = {"macro": "resset", "all_numbers": [str(tmp_path / "n.tex")], "report_files": [str(tmp_path / "r.tex")]}
    assert missing_report_keys(nc) == ["d.e", "f"]
    (tmp_path / "sections").mkdir()
    (tmp_path / "sections" / "s1.tex").write_text(r"\res{g} % \res{commented}")
    nc["report_files"].append(str(tmp_path / "sections" / "*.tex"))   # glob patterns are expanded
    assert missing_report_keys(nc) == ["d.e", "f", "g"]
