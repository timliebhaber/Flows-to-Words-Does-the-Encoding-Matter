"""Figures for the analysis (matplotlib, Agg backend: no display, CPU only). Colours, sizes and formats come from
configs/analysis.yaml `figures` (dataviz reference palette: one colour + marker per model, one blue ramp for heatmaps).
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.eval.errors import row_normalised


def setup(fc):
    plt.rcParams.update({
        "font.size": fc["font_size"], "text.color": fc["text_primary"], "axes.labelcolor": fc["text_secondary"],
        "axes.edgecolor": fc["grid"], "xtick.color": fc["text_secondary"], "ytick.color": fc["text_secondary"],
        "axes.titlesize": fc["font_size"], "axes.titlecolor": fc["text_primary"], "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": fc["surface"], "savefig.facecolor": fc["surface"],
        "axes.facecolor": fc["surface"], "pdf.fonttype": 42, "legend.frameon": False,
    })


def save(fig, directory, stem, fc):
    Path(directory).mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in fc["formats"]:
        path = Path(directory) / f"{stem}.{ext}"
        # no creation date in PDFs, so an unchanged figure is an unchanged file (clean git diffs and Overleaf pushes)
        fig.savefig(path, dpi=fc["dpi"], bbox_inches="tight", metadata={"CreationDate": None} if ext == "pdf" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def _cmap(fc):
    return LinearSegmentedColormap.from_list("ramp", fc["sequential_ramp"]).with_extremes(bad=fc["surface"])


def heatmap(ax, values, annot, rows, cols, fc, outline_diag=False):
    """values: 2-D array (NaN = no result); annot: same-shape strings ('' = none). Returns the image."""
    lo, hi = fc["heatmap_range"]
    img = ax.imshow(np.ma.masked_invalid(values), cmap=_cmap(fc), vmin=lo, vmax=hi, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if annot[i][j]:
                v = values[i, j]
                dark = not math.isnan(v) and (v - lo) / (hi - lo) > fc["dark_cell_threshold"]
                ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=fc["annot_font_size"],
                        color=fc["text_on_dark"] if dark else fc["text_primary"])
    if outline_diag:
        for k in range(min(values.shape)):
            ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, edgecolor=fc["text_primary"],
                                   linewidth=fc["style"]["outline_width"]))
    ax.set_xticks(range(len(cols)), cols, rotation=90)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    return img


def _stat_annot(s, d, marker):
    if s is None:
        return ""
    return f"{s.mean:.{d}f}{marker}" if s.n == 1 else f"{s.mean:.{d}f}\n±{s.std:.{d}f}"


def _grid(cells, model, encoding, datasets, d, marker):
    values = np.full((len(datasets), len(datasets)), np.nan)
    annot = [[""] * len(datasets) for _ in datasets]
    for i, tr in enumerate(datasets):
        for j, te in enumerate(datasets):
            s = cells.get((model, encoding, tr, te))
            if s is not None:
                values[i, j], annot[i][j] = s.mean, _stat_annot(s, d, marker)
    return values, annot


def exp3_heatmaps(cfg, cells, metric):
    """One train x test heatmap per method + one combined grid (rows = models). Returns the figure stems written."""
    fc, ds = cfg["figures"], cfg["cross"]["datasets"]
    names = [cfg["datasets"][d] for d in ds]
    d, marker, label = fc["annot_decimals"], cfg["format"]["one_seed_marker"], cfg["format"]["metric_labels"][metric]
    methods = [m for m in cfg["methods"] if any(k[:2] == (m["model"], m["encoding"]) for k in cells)]
    stems = []
    for m in methods:
        fig, ax = plt.subplots(figsize=fc["sizes"]["heatmap"])
        img = heatmap(ax, *_grid(cells, m["model"], m["encoding"], ds, d, marker), names, names, fc, outline_diag=True)
        ax.set_title(f"{m['label']}: {label}")
        ax.set_xlabel("test dataset")
        ax.set_ylabel("train dataset")
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        stem = f"exp3_heatmap_{m['model']}_{m['encoding']}"
        save(fig, fc["dir"], stem, fc)
        stems.append(stem)
    if methods:
        models = list(dict.fromkeys(m["model"] for m in methods))
        per_model = {mo: [m for m in methods if m["model"] == mo] for mo in models}
        ncols = max(len(v) for v in per_model.values())
        w, h = fc["sizes"]["grid_panel"]
        fig, axes = plt.subplots(len(models), ncols, figsize=(w * ncols, h * len(models)), squeeze=False)
        img = None
        for r, mo in enumerate(models):
            for c in range(ncols):
                ax = axes[r][c]
                if c >= len(per_model[mo]):
                    ax.axis("off")
                    continue
                m = per_model[mo][c]
                img = heatmap(ax, *_grid(cells, mo, m["encoding"], ds, d, marker), names, names, fc, outline_diag=True)
                ax.set_title(m["label"])
                if c:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel("train dataset")
                if r < len(models) - 1 and c < len(per_model[models[r + 1]]):
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("test dataset")
        fig.colorbar(img, ax=axes, fraction=0.02, pad=0.02, label=label)
        save(fig, fc["dir"], "exp3_heatmaps_grid", fc)
        stems.append("exp3_heatmaps_grid")
    return stems


def _model_handles(cfg, single_seed_models):
    fc = cfg["figures"]
    handles = []
    for mo, name in cfg["models"].items():
        color = fc["model_colors"][mo]
        hollow = mo in single_seed_models
        handles.append(Line2D([], [], linestyle="", marker=fc["model_markers"][mo], markersize=fc["marker_size"],
                              color=color, markerfacecolor=fc["surface"] if hollow else color,
                              label=f"{name} (1 seed)" if hollow else name))
    return handles


def _dot(ax, x, y, s, model, fc):
    color = fc["model_colors"][model]
    ax.errorbar(x, y, xerr=None if math.isnan(s.std) else s.std, fmt=fc["model_markers"][model], color=color,
                markersize=fc["marker_size"], markerfacecolor=fc["surface"] if s.n == 1 else color,
                capsize=fc["style"]["capsize"], elinewidth=fc["style"]["line_width"])


def _min_span(ax, span):
    """Widen the x-axis to at least `span` around its centre, so tiny seed stds don't look large."""
    lo, hi = ax.get_xlim()
    if hi - lo < span:
        mid = (lo + hi) / 2
        ax.set_xlim(mid - span / 2, mid + span / 2)


def _single_seed_models(cfg, stats_by_model):
    return {mo for mo, stats in stats_by_model.items() if stats and all(s.n == 1 for s in stats)}


def exp2_dotplot(cfg, cells, metric):
    """Within-dataset results: one panel per dataset, encodings on y, one colour/marker per model, std as bars."""
    fc = cfg["figures"]
    datasets = [d for d in cfg["datasets"] if any(k[2] == d for k in cells)]
    if not datasets:
        return []
    encs = list(dict.fromkeys(m["encoding"] for m in cfg["methods"]))
    models = list(dict.fromkeys(m["model"] for m in cfg["methods"]))
    w, h = fc["sizes"]["dot_panel"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(w * len(datasets), h), sharey=True, squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        for i, mo in enumerate(models):
            offset = (i - (len(models) - 1) / 2) * fc["style"]["dodge"]
            for j, enc in enumerate(encs):
                s = cells.get((mo, enc, dataset))
                if s is not None:
                    _dot(ax, s.mean, j + offset, s, mo, fc)
        ax.set_title(cfg["datasets"][dataset])
        ax.grid(axis="x", color=fc["grid"], linewidth=fc["style"]["line_width"])
        ax.set_axisbelow(True)
        ax.set_xlabel(cfg["format"]["metric_labels"][metric])
        _min_span(ax, fc["min_axis_span"])
    axes[0][0].set_yticks(range(len(encs)), [cfg["encodings"][e] for e in encs])
    axes[0][0].invert_yaxis()
    single = _single_seed_models(cfg, {mo: [s for k, s in cells.items() if k[0] == mo] for mo in models})
    fig.legend(handles=_model_handles(cfg, single), loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    stem = f"exp2_within_{metric}"
    save(fig, fc["dir"], stem, fc)
    return [stem]


def exp3_offdiag(cfg, summary):
    """Mean off-diagonal value per method, one panel per cross metric (same 0-1 axis)."""
    fc = cfg["figures"]
    metrics = [m for m in cfg["cross"]["metrics"] if summary[m]]
    if not metrics:
        return []
    methods = [m for m in cfg["methods"] if (m["model"], m["encoding"]) in summary[metrics[0]]]
    w, h = fc["sizes"]["offdiag_panel"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(w * len(metrics), h), sharey=True, squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for y, m in enumerate(methods):
            s = summary[metric].get((m["model"], m["encoding"]))
            if s is not None:
                _dot(ax, s["off"].mean, y, s["off"], m["model"], fc)
        ax.set_xlim(fc["heatmap_range"])
        ax.grid(axis="x", color=fc["grid"], linewidth=fc["style"]["line_width"])
        ax.set_axisbelow(True)
        ax.set_title(f"off-diagonal {cfg['format']['metric_labels'][metric]}")
    axes[0][0].set_yticks(range(len(methods)), [m["label"] for m in methods])
    axes[0][0].invert_yaxis()
    stats = {mo: [v["off"] for k, v in summary[metrics[0]].items() if k[0] == mo] for mo in cfg["models"]}
    used = {m["model"] for m in methods}
    handles = [hd for hd, mo in zip(_model_handles(cfg, _single_seed_models(cfg, stats)), cfg["models"]) if mo in used]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, fc["dir"], "exp3_offdiag", fc)
    return ["exp3_offdiag"]


def perclass_heatmap(cfg, dataset, entries):
    """entries: [(label, aggregate)] for one dataset; rows = methods, columns = classes, cells = per-class F1."""
    fc = cfg["figures"]
    classes = entries[0][1]["classes"]
    values = np.array([[dict(zip(a["classes"], a["per_class_f1"])).get(c, np.nan) for c in classes]
                       for _, a in entries])
    d = fc["annot_decimals"]
    annot = [["" if math.isnan(v) else f"{v:.{d}f}" for v in row] for row in values]
    cw, ch = fc["sizes"]["cell"]
    mw, mh = fc["sizes"]["margins"]
    fig, ax = plt.subplots(figsize=(mw + cw * len(classes), mh + ch * len(entries)))
    img = heatmap(ax, values, annot, [lab for lab, _ in entries], classes, fc)
    ax.set_title(f"{cfg['datasets'][dataset]}: per-class F1 (mean over seeds)")
    fig.colorbar(img, ax=ax, fraction=0.03, pad=0.02)
    stem = f"perclass_f1_{dataset}"
    save(fig, fc["dir"], stem, fc)
    return [stem]


def confusion_panels(cfg, dataset, entries):
    """entries: [(label, aggregate)]; row-normalised confusion matrices (counts summed over seeds), side by side."""
    fc = cfg["figures"]
    n = len(entries[0][1]["classes"])
    side = fc["sizes"]["confusion_cell"] * n + fc["sizes"]["margins"][1]
    fig, axes = plt.subplots(1, len(entries), figsize=(side * len(entries) + fc["sizes"]["margins"][0], side),
                             squeeze=False)
    d, img = fc["annot_decimals"], None
    for c, (ax, (label, agg)) in enumerate(zip(axes[0], entries)):
        shares = np.where(agg["confusion"] > 0, row_normalised(agg["confusion"]), np.nan)  # empty cells: surface
        annot = [[f"{v:.{d}f}".lstrip("0") if v >= fc["confusion_min_share"] else "" for v in row]
                 for row in np.nan_to_num(shares)]
        img = heatmap(ax, shares, annot, agg["classes"], agg["classes"], fc)
        ax.set_title(label)
        ax.set_xlabel("predicted")
        if c:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("true")
    fig.suptitle(f"{cfg['datasets'][dataset]}: confusion matrices (share of each true class)")
    fig.colorbar(img, ax=axes, fraction=0.02, pad=0.02)
    stem = f"confusion_{dataset}"
    save(fig, fc["dir"], stem, fc)
    return [stem]
