"""
seanet/analysis/figures.py - the benchmark, top-k, WebTraffic and efficiency figures.

The rule every figure in this file obeys
----------------------------------------
ONE FIGURE ANSWERS ONE QUESTION. Before a figure is written down, its question is written down (it
is a required argument of style.save, and it ends up in the metadata JSON and the LaTeX caption). If
a question cannot be phrased, the figure does not belong in the paper. That is why this file draws
about thirty figures and not the two hundred that "every metric x every model x every grouping"
would give - most of those would answer the same question twice.

Sections
--------
    1. the benchmark bands   : who is strongest, in non-overlapping accuracy groups
    2. the top-k analysis    : the best k models, together and metric by metric
    3. WebTraffic            : our headline dataset on its own
    4. efficiency + Pareto   : quality against cost (params, size, FLOPs, latency, memory)
    5. ablation              : what each part of the model contributes

The statistical figures (ranks, significance, heatmaps) live in figures_stats.py.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from seanet.analysis import data as PD
from seanet.analysis import stats as PS
from seanet.analysis import style as S

# --------------------------------------------------------------------------------------
# What each metric is called, which way is good, and how to print it.
# One table, used by every figure here - so a metric is labelled identically everywhere.
# --------------------------------------------------------------------------------------
METRICS: Dict[str, Dict] = {
    "web_acc":    {"label": "Accuracy", "axis": "WebTraffic accuracy",
                   "higher": True, "fmt": "{:.3f}", "paper": "acc", "scale": 1.0},
    "web_aopcr":  {"label": "AOPCR", "axis": "AOPCR (interpretability)",
                   "higher": True, "fmt": "{:.2f}", "paper": "aopcr", "scale": 1.0},
    "web_ndcg":   {"label": "NDCG@n", "axis": "NDCG@n (explanation quality)",
                   "higher": True, "fmt": "{:.3f}", "paper": "ndcg", "scale": 1.0},
    "web_loss":   {"label": "Loss", "axis": "WebTraffic test loss",
                   "higher": False, "fmt": "{:.3f}", "paper": "loss", "scale": 1.0},
    "ucr85_acc":  {"label": "UCR-85 accuracy", "axis": "mean accuracy over 85 UCR datasets",
                   "higher": True, "fmt": "{:.4f}", "paper": None, "scale": 1.0},
    "params":     {"label": "Parameters", "axis": "parameters (thousands)",
                   "higher": False, "fmt": "{:.0f}K", "paper": None, "scale": 1e-3},
    "size_mb":    {"label": "Model size", "axis": "model size (MB)",
                   "higher": False, "fmt": "{:.2f}", "paper": None, "scale": 1.0},
    "flops_m":    {"label": "FLOPs", "axis": "FLOPs per series (millions)",
                   "higher": False, "fmt": "{:.1f}", "paper": None, "scale": 1.0},
    "infer_ms":   {"label": "Inference time", "axis": "inference time per series (ms)",
                   "higher": False, "fmt": "{:.3f}", "paper": None, "scale": 1.0},
    "peak_mem_mb": {"label": "GPU memory", "axis": "peak GPU memory (MB)",
                    "higher": False, "fmt": "{:.1f}", "paper": None, "scale": 1.0},
}

# the metrics that only exist once scripts/profile_models.py has been run
COST_METRICS = ["params", "size_mb", "flops_m", "infer_ms", "peak_mem_mb"]


def _values(df: pd.DataFrame, column: str) -> np.ndarray:
    """One metric column, scaled the way we plot it (params in thousands, everything else as-is)."""
    return pd.to_numeric(df[column], errors="coerce").to_numpy(float) * METRICS[column]["scale"]


def _colours(df: pd.DataFrame) -> List[str]:
    """Blue for the models we propose, orange for our rerun of the paper's own backbones."""
    is_ours = PD.ours_mask(df)
    return [S.C_OURS if o else S.C_BASELINE for o in is_ours]


def _rank_by(lb: pd.DataFrame, column: str) -> pd.DataFrame:
    """Sort by one metric, best first, dropping models that have no value for it."""
    return (lb.dropna(subset=[column])
              .sort_values(column, ascending=not METRICS[column]["higher"])
              .reset_index(drop=True))


def _ours_legend(ax, reference: Optional[float] = None) -> None:
    """
    The legend, placed ABOVE the plot so it can never sit on top of a data point.

    matplotlib's loc="best" still lands on the data when a plot is crowded, and a legend covering a
    value label is the single most common figure mistake. Putting it outside the axes costs a few
    millimetres of height and removes the problem for good.
    """
    handles = [plt.Line2D([0], [0], marker="o", ls="none", color=S.C_OURS, ms=5),
               plt.Line2D([0], [0], marker="o", ls="none", color=S.C_BASELINE, ms=5)]
    labels = ["Ours", "Reproduced baseline"]
    if reference is not None and pd.notna(reference):
        handles.append(plt.Line2D([0], [0], ls="--", lw=1.2, color=S.C_MILLET))
        labels.append("MILLET (published)")
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=len(labels), frameon=False, fontsize=7.5, borderaxespad=0.4)


# ======================================================================================
# 1. BENCHMARK BANDS - who is strongest, in non-overlapping groups
# ======================================================================================
def auto_thresholds(values: np.ndarray) -> List[float]:
    """
    Choose the two accuracy cut points automatically, instead of hard-coding 0.94 / 0.90.

    Why automatic: a threshold that is right for today's 57 models is wrong after the next sweep. We
    want "the strongest models" to stay a small, readable group no matter how the field of models
    changes, so we let the DATA choose:

        cut 1 = the 85th percentile  -> roughly the top 15% of models: the paper's main figure
        cut 2 = the 60th percentile  -> the next group down: the "strong models" figure

    Percentiles are used rather than fixed numbers because they adapt: if every model improves, the
    main figure still shows the leaders rather than suddenly showing everyone.

    The chosen values are rounded to two decimals so the caption can state a clean number, and the
    figure title always prints the threshold it used - nothing is hidden from the reader.

    values : the metric values of every model.
    returns : [high cut, low cut], highest first.
    """
    clean = values[~np.isnan(values)]
    if len(clean) < 4:                                       # too few models to split meaningfully
        return [float(np.min(clean)) if len(clean) else 0.0]
    high = float(np.round(np.percentile(clean, 85), 2))
    low = float(np.round(np.percentile(clean, 60), 2))
    if low >= high:                                          # everything bunched together
        low = float(np.round(high - 0.02, 2))
    return [high, low]


def _bands(sub: pd.DataFrame, column: str, cuts: List[float]) -> List[tuple]:
    """
    Cut the ranked models into NON-OVERLAPPING groups.

    This is the heart of the figure-1 / figure-2 / figure-3 split the paper needs. With accuracy and
    cuts [0.94, 0.90]:

        band 1 : acc >= 0.94              -> Figure 1 (main paper)
        band 2 : 0.90 <= acc < 0.94       -> Figure 2, WITHOUT band 1's models
        band 3 : acc < 0.90               -> Figure 3 (appendix)

    Each band takes only what the bands above it did not, so a model appears in exactly one figure.
    That is what makes Figure 2 "the next group down" instead of Figure 1 redrawn with extras.

    returns : (band number, human title, the rows) - empty bands are skipped.
    """
    higher = METRICS[column]["higher"]
    vals = pd.to_numeric(sub[column], errors="coerce") * METRICS[column]["scale"]
    out, taken = [], pd.Series(False, index=sub.index)
    for i, cut in enumerate(cuts):
        inside = (vals >= cut) if higher else (vals <= cut)
        mask = inside & ~taken
        taken = taken | mask
        if mask.any():
            if i == 0:
                title = f"{METRICS[column]['label']} $\\geq$ {cut:g}"
            else:
                prev = cuts[i - 1]
                title = (f"{cut:g} $\\leq$ {METRICS[column]['label']} $<$ {prev:g}" if higher
                         else f"{prev:g} $<$ {METRICS[column]['label']} $\\leq$ {cut:g}")
            out.append((i + 1, title, sub[mask]))
    rest = sub[~taken]
    if len(rest):
        sign = "<" if higher else ">"
        out.append((len(cuts) + 1, f"{METRICS[column]['label']} ${sign}$ {cuts[-1]:g}", rest))
    return out


def _hbar(ax, frame: pd.DataFrame, column: str, reference: Optional[float] = None,
          show_names: bool = True) -> None:
    """
    Draw one ranked comparison: one row per model, best at the top.

    WHY A DOT PLOT AND NOT BARS. Our models sit in a very narrow band - the top thirteen span
    0.940 to 0.954 accuracy. Drawn as bars starting at zero, all thirteen look identical and the
    figure says nothing. The usual fix, cutting the axis so it starts near the data, is exactly the
    thing you must NOT do with bars: a bar means "this much", so a cut axis makes a 1% difference
    look like a 3x difference and reviewers rightly call it out.

    A Cleveland dot plot solves both problems honestly. A dot marks a POSITION, not a quantity, so
    the axis is free to start wherever the data is, and small real differences become visible
    without exaggerating anything. The faint guide line only leads the eye from the name to the dot.

    Horizontal layout is kept because model names are long words: on a horizontal axis they sit flat
    and read normally instead of being rotated at column width.
    """
    names = [S.shorten(m) for m in frame["model"]]
    values = _values(frame, column)
    colours = _colours(frame)
    y = np.arange(len(names))

    finite = values[~np.isnan(values)]
    if len(finite) == 0:
        return
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if reference is not None and pd.notna(reference):        # the reference line must fit too
        lo, hi = min(lo, float(reference)), max(hi, float(reference))
    span = hi - lo if hi > lo else max(abs(hi), 1.0) * 0.1
    x_left = max(0.0, lo - span * 0.35) if lo >= 0 else lo - span * 0.35
    x_right = hi + span * 0.55                               # room on the right for the value labels

    # the guide line: name -> dot, so the eye can track a long row without a ruler
    for yi, v in zip(y, values):
        if not np.isnan(v):
            ax.plot([x_left, v], [yi, yi], color="#DDDDDD", lw=0.7, zorder=2)
    ax.scatter(values, y, c=colours, s=46, zorder=4, edgecolors="white", linewidths=0.7)

    for yi, v in zip(y, values):
        if not np.isnan(v):
            ax.text(v + span * 0.045, yi, METRICS[column]["fmt"].format(v),
                    va="center", ha="left", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels(names if show_names else [])
    ax.set_ylim(len(names) - 0.5, -0.5)                      # best model at the TOP
    ax.set_xlim(x_left, x_right)
    ax.set_xlabel(METRICS[column]["axis"])
    ax.grid(axis="x", alpha=0.35, zorder=0)
    ax.grid(axis="y", visible=False)

    if reference is not None and pd.notna(reference):
        ax.axvline(float(reference), ls="--", lw=1.2, color=S.C_MILLET, zorder=3,
                   label=f"MILLET (paper): {float(reference):.3f}")


def benchmark_bands(lb: pd.DataFrame, column: str = "web_acc") -> List[Dict]:
    """
    FIGURES 1-3: every model ranked on one metric, split into non-overlapping strength bands.

    Figure 1 (main paper)  : the strongest models only.
    Figure 2 (main paper)  : the next competitive group, excluding Figure 1's models.
    Figure 3 (appendix)    : everything remaining.

    The threshold is chosen from the data (see auto_thresholds) and printed in the title.
    """
    sub = _rank_by(lb, column)
    if sub.empty:
        return []
    paper = PD.paper_baseline()
    ref = paper.get(METRICS[column]["paper"]) if METRICS[column]["paper"] else None
    cuts = auto_thresholds(_values(sub, column))
    entries = []

    for band_no, title, frame in _bands(sub, column, cuts):
        n = len(frame)
        # height grows with the number of models so the bars never squash together
        height = max(1.6, 0.23 * n + 0.85)
        width = S.FULL_WIDTH if n > 8 else S.COL_WIDTH * 1.55
        fig, ax = plt.subplots(figsize=(width, height))
        _hbar(ax, frame, column, reference=ref)
        _ours_legend(ax, reference=ref)                       # above the plot, never over the data
        ax.set_title(f"{title}  ({n} model{'s' if n != 1 else ''})", pad=22)

        section = "main" if band_no <= 2 else "appendix"
        rank_word = {1: "strongest", 2: "competitive", 3: "remaining"}.get(band_no, "remaining")
        entries.append(S.save(
            fig, section, f"fig{band_no}_benchmark_{rank_word}_{column}",
            title=f"Benchmark band {band_no}: {rank_word} models ({METRICS[column]['label']})",
            # NOTE: keep the $...$ around \geq / \leq / <. Stripping them leaves a bare \geq
            # in normal text, and LaTeX then stops with "Missing $ inserted".
            caption=(f"{METRICS[column]['label']} of the {rank_word} models on WebTraffic "
                     f"({n} models, {title}). Bars are sorted best-first; blue marks "
                     f"models proposed in this work and orange our reproduction of the published "
                     f"backbones. Labels read \\texttt{{encoder\\_\\_pooling}}, abbreviated, where sea\\_ "
                     f"marks a component introduced here and mil\\_ one reused unchanged from "
                     f"MILLET. "
                     f"The dashed line is the "
                     f"accuracy reported in the MILLET paper. "
                     f"Bands are disjoint, so each model appears in exactly one of "
                     f"Figures 1--3."),
            question=f"Which models are the {rank_word} on {METRICS[column]['label']}?"))
    return entries


# ======================================================================================
# 2. TOP-K ANALYSIS - the best k models, together and one metric at a time
# ======================================================================================
def _top_k(lb: pd.DataFrame, k: int, column: str = "web_acc") -> pd.DataFrame:
    """The best k models on one metric. WebTraffic accuracy by default: every model has it."""
    return _rank_by(lb, column).head(k)


def topk_multimetric(lb: pd.DataFrame, k: int, section: str = "main") -> Optional[Dict]:
    """
    THE CONCLUSION FIGURE: the top-k models on every metric at once.

    One small panel per metric, the same k models in the same order on every panel, so the reader
    can follow one model straight down the figure and see where it wins and where it pays. This is
    the figure the conclusion section points at to justify the recommended model.
    """
    top = _top_k(lb, k)
    if top.empty:
        return None
    columns = [c for c in ["web_acc", "web_aopcr", "web_ndcg", "web_loss",
                           "ucr85_acc", "params", "flops_m", "infer_ms"]
               if c in lb.columns and top[c].notna().any()]
    if not columns:
        return None

    n_cols = min(4, len(columns))
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(S.FULL_WIDTH, 1.55 * n_rows + 0.35 * k))
    axes = np.atleast_1d(axes).ravel()
    paper = PD.paper_baseline()

    for ax, col in zip(axes, columns):
        ref = paper.get(METRICS[col]["paper"]) if METRICS[col]["paper"] else None
        _hbar(ax, top, col, reference=ref)
        ax.set_title(METRICS[col]["label"])
        ax.set_xlabel("")                                    # the title already names the metric
        if ax is not axes[0]:                                # model names once, on the left column
            pass
    for ax in axes[len(columns):]:
        ax.axis("off")
    # keep the model names only on the leftmost panel of each row - repeating them wastes width
    for i, ax in enumerate(axes[:len(columns)]):
        if i % n_cols != 0:
            ax.set_yticklabels([])

    fig.suptitle(f"Top-{k} models across every metric (ranked by WebTraffic accuracy)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return S.save(
        fig, section, f"topk{k}_multimetric",
        title=f"Top-{k} models across all metrics",
        caption=(f"The top-{k} models, ranked by WebTraffic accuracy, compared on every available "
                 f"metric. Panels share the same model order, so a single model can be followed "
                 f"across quality (accuracy, AOPCR, NDCG, loss) and cost (parameters, FLOPs, "
                 f"latency). Blue marks models proposed in this work."),
        question=f"Which of the top-{k} models is preferable once every metric is considered?")


def topk_per_metric(lb: pd.DataFrame, k: int, section: str = "appendix") -> List[Dict]:
    """
    The same top-k models, one metric per figure, drawn at full size.

    Use these when a single metric deserves its own place in the paper or a slide; use
    topk_multimetric when the whole story should fit in one picture.
    """
    top = _top_k(lb, k)
    if top.empty:
        return []
    paper = PD.paper_baseline()
    entries = []
    for col, spec in METRICS.items():
        if col not in lb.columns or not top[col].notna().any():
            continue
        ref = paper.get(spec["paper"]) if spec["paper"] else None
        fig, ax = plt.subplots(figsize=(S.COL_WIDTH * 1.5, max(1.5, 0.26 * k + 0.8)))
        _hbar(ax, top, col, reference=ref)
        ax.set_title(f"Top-{k}: {spec['label']}")
        entries.append(S.save(
            fig, section, f"topk{k}_{col}",
            title=f"Top-{k} models - {spec['label']}",
            caption=(f"{spec['label']} of the top-{k} models (ranked by WebTraffic accuracy). "
                     f"{'Higher' if spec['higher'] else 'Lower'} is better."),
            question=f"How do the top-{k} models compare on {spec['label']} alone?"))
    return entries


# ======================================================================================
# 3. WEBTRAFFIC - our headline dataset, the only one with per-timestep ground truth
# ======================================================================================
def webtraffic_metrics(lb: pd.DataFrame, top_n: int = 10) -> List[Dict]:
    """
    WebTraffic on its own, one figure per metric.

    WebTraffic earns its own section because it is the ONLY dataset that ships per-timestep ground
    truth. Everywhere else, an explanation cannot be checked; here it can, which is what makes NDCG
    meaningful at all. The paper shows the strongest models; the appendix version shows all of them.
    """
    paper = PD.paper_baseline()
    entries = []
    for col in ["web_acc", "web_aopcr", "web_ndcg", "web_loss"]:
        if col not in lb.columns:
            continue
        ranked = _rank_by(lb, col)
        if ranked.empty:
            continue
        spec = METRICS[col]
        ref = paper.get(spec["paper"]) if spec["paper"] else None

        # (a) the paper version: the strongest models only
        top = ranked.head(top_n)
        fig, ax = plt.subplots(figsize=(S.COL_WIDTH * 1.55, 0.24 * len(top) + 0.9))
        _hbar(ax, top, col, reference=ref)
        _ours_legend(ax, reference=ref)
        ax.set_title(f"WebTraffic: {spec['label']} (top {len(top)})", pad=22)
        entries.append(S.save(
            fig, "web", f"web_top{top_n}_{col}",
            title=f"WebTraffic {spec['label']} - top {top_n}",
            caption=(f"{spec['axis']} for the {len(top)} best models. WebTraffic is the only dataset "
                     f"with per-timestep ground truth, so it is the only setting where explanation "
                     f"quality can be measured directly rather than estimated. "
                     f"{'Higher' if spec['higher'] else 'Lower'} is better."),
            question=f"Which models explain and classify WebTraffic best on {spec['label']}?"))

        # (b) the appendix version: every model, for completeness
        if len(ranked) > top_n:
            fig, ax = plt.subplots(figsize=(S.FULL_WIDTH, 0.19 * len(ranked) + 1.0))
            _hbar(ax, ranked, col, reference=ref)
            _ours_legend(ax, reference=ref)
            ax.set_title(f"WebTraffic: {spec['label']} (all {len(ranked)} models)", pad=22)
            entries.append(S.save(
                fig, "appendix", f"appendix_web_all_{col}",
                title=f"WebTraffic {spec['label']} - every model",
                caption=(f"{spec['axis']} for all {len(ranked)} evaluated models, sorted best-first. "
                         f"The main-paper version of this figure shows only the top {top_n}."),
                question=f"How does the complete field of models compare on WebTraffic {spec['label']}?"))
    return entries


# ======================================================================================
# 4. EFFICIENCY AND PARETO - is the small model actually worth choosing?
# ======================================================================================
def pareto_scatter(lb: pd.DataFrame, cost: str, quality: str = "web_acc",
                   section: str = "main") -> Optional[Dict]:
    """
    Quality against cost, with the Pareto front drawn through it.

    This is the figure that makes the paper's central claim checkable. A model on the front is one
    that nothing else beats on BOTH axes - so choosing anything behind the front is strictly worse.
    If our small model sits on the front next to a much larger baseline, "smaller and just as good"
    stops being a claim and becomes a picture.

    cost : the x-axis (params / size_mb / flops_m / infer_ms / peak_mem_mb).
    quality : the y-axis (accuracy by default).
    """
    if cost not in lb.columns or quality not in lb.columns:
        return None
    sub = lb.dropna(subset=[cost, quality]).copy()
    if len(sub) < 3:
        return None

    sub["_x"] = _values(sub, cost)
    sub["_y"] = _values(sub, quality)
    on_front = PS.pareto_front(sub, "_x", "_y", x_lower_better=True,
                               y_higher_better=METRICS[quality]["higher"])

    fig, ax = plt.subplots(figsize=(S.COL_WIDTH * 1.5, S.COL_WIDTH * 1.5 * S.GOLDEN + 0.4))
    is_ours = PD.ours_mask(sub)

    # everything that is NOT on the front: small, hollow, quiet
    for ours, marker, label in ((True, "o", "Ours"), (False, "s", "Reproduced baseline")):
        mask = (is_ours == ours) & ~on_front
        if mask.any():
            ax.scatter(sub.loc[mask, "_x"], sub.loc[mask, "_y"], marker=marker, s=26,
                       facecolors="none", edgecolors=S.C_OURS if ours else S.C_BASELINE,
                       linewidths=0.9, label=f"{label} (dominated)", zorder=3)
    # the Pareto-optimal models: filled, larger, and joined by the front line
    for ours, marker, label in ((True, "o", "Ours"), (False, "s", "Reproduced baseline")):
        mask = (is_ours == ours) & on_front
        if mask.any():
            ax.scatter(sub.loc[mask, "_x"], sub.loc[mask, "_y"], marker=marker, s=58,
                       color=S.C_OURS if ours else S.C_BASELINE, edgecolors="white",
                       linewidths=0.7, label=f"{label} (Pareto-optimal)", zorder=5)

    front = sub[on_front].sort_values("_x")
    ax.plot(front["_x"], front["_y"], ls="--", lw=1.1, color=S.C_GOOD, zorder=4,
            label="Pareto front")

    # A LOG x-axis when the cost values span more than one order of magnitude. Parameters run from
    # about 41k to 506k here, so on a linear axis every small model is squashed into one thin strip
    # at the left and the front is unreadable. A log axis gives each factor-of-two the same width,
    # which is the right way to look at cost anyway - "twice as big" matters, "+1000 weights" does not.
    finite = sub["_x"][sub["_x"] > 0]
    if len(finite) and float(finite.max()) / float(finite.min()) > 5:
        ax.set_xscale("log")
        # matplotlib's default log ticks would print "10^2" and nothing else, which is useless when
        # the whole range is 41-506. So we place our own ticks at round numbers inside the range and
        # print them as plain numbers - a reader should never have to do mental arithmetic on an axis.
        from matplotlib.ticker import FixedLocator, FuncFormatter
        nice = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        lo_x, hi_x = float(finite.min()), float(finite.max())
        ticks = [t for t in nice if lo_x * 0.85 <= t <= hi_x * 1.15]
        if len(ticks) >= 2:
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_minor_locator(FixedLocator([]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))

    # headroom above the highest point, so a label can never collide with the title
    y_lo, y_hi = float(sub["_y"].min()), float(sub["_y"].max())
    y_span = (y_hi - y_lo) or max(abs(y_hi), 1.0) * 0.1
    ax.set_ylim(y_lo - y_span * 0.08, y_hi + y_span * 0.22)

    # name only the models ON the front - labelling all of them would be an unreadable pile. Even
    # so the front models often sit close together, so we STAGGER the labels: each one is pushed a
    # little further up than the last, and the cycle resets every few points. That keeps them
    # readable without needing an external label-placement library.
    offsets = [(7, -13), (7, 7), (-7, -13), (7, 18)]
    for n, (_, row) in enumerate(front.iterrows()):
        ax.annotate(S.shorten(row["model"], 14), (row["_x"], row["_y"]),
                    textcoords="offset points", xytext=offsets[n % len(offsets)],
                    fontsize=6.2, color="#444444",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#BBBBBB",
                                    shrinkA=0, shrinkB=2))

    ax.set_xlabel(METRICS[cost]["axis"])
    ax.set_ylabel(METRICS[quality]["axis"])
    ax.set_title(f"{METRICS[quality]['label']} vs {METRICS[cost]['label']}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=6.5)
    return S.save(
        fig, section, f"pareto_{quality}_vs_{cost}",
        title=f"{METRICS[quality]['label']} vs {METRICS[cost]['label']} (Pareto)",
        caption=(f"{METRICS[quality]['axis']} against {METRICS[cost]['axis']}. Filled markers joined "
                 f"by the dashed line are Pareto-optimal: no other model is both cheaper and better. "
                 f"Hollow markers are dominated and therefore never the right choice. Only "
                 f"Pareto-optimal models are labelled."),
        question=(f"Does a smaller model stay competitive once "
                  f"{METRICS[cost]['label'].lower()} is taken into account?"))


def efficiency_figures(lb: pd.DataFrame) -> List[Dict]:
    """
    Every quality-versus-cost figure the available data supports.

    Parameters and model size always exist. FLOPs, inference time and GPU memory only exist once
    `python scripts/profile_models.py` has been run - if it has not, those figures are skipped with a
    printed note rather than drawn from missing data.
    """
    entries = []
    # accuracy vs parameters is the headline efficiency claim, so it goes in the main paper
    main = pareto_scatter(lb, "params", "web_acc", section="main")
    if main:
        entries.append(main)
    for cost in ["size_mb", "flops_m", "infer_ms", "peak_mem_mb"]:
        entry = pareto_scatter(lb, cost, "web_acc", section="appendix")
        if entry:
            entries.append(entry)
    # quality-versus-quality: does being accurate cost interpretability?
    for other in ["web_aopcr", "web_ndcg"]:
        if other in lb.columns and lb[other].notna().sum() >= 3:
            entry = _quality_tradeoff(lb, "web_acc", other)
            if entry:
                entries.append(entry)
    return entries


def _quality_tradeoff(lb: pd.DataFrame, x: str, y: str) -> Optional[Dict]:
    """
    Two quality metrics against each other - accuracy against interpretability.

    This asks the question the whole MILLET line of work exists for: does making a model more
    accurate cost you the explanation quality? A cloud with no trend means the two goals are
    independent, which is a genuinely useful result to report.
    """
    sub = lb.dropna(subset=[x, y])
    if len(sub) < 4:
        return None
    fig, ax = plt.subplots(figsize=(S.COL_WIDTH * 1.4, S.COL_WIDTH * 1.4 * S.GOLDEN + 0.3))
    is_ours = PD.ours_mask(sub)
    for ours, marker, colour, label in ((True, "o", S.C_OURS, "Ours"),
                                        (False, "s", S.C_BASELINE, "Reproduced baseline")):
        mask = is_ours == ours
        if mask.any():
            ax.scatter(_values(sub[mask], x), _values(sub[mask], y), marker=marker, s=34,
                       color=colour, alpha=0.85, edgecolors="white", linewidths=0.5, label=label)
    # Spearman correlation: do the two metrics ORDER the models the same way?
    rho = sub[[x, y]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel(METRICS[x]["axis"])
    ax.set_ylabel(METRICS[y]["axis"])
    ax.set_title(f"{METRICS[x]['label']} vs {METRICS[y]['label']}  "
                 f"(Spearman $\\rho$ = {rho:.2f})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=7)
    return S.save(
        fig, "stats", f"tradeoff_{x}_vs_{y}",
        title=f"{METRICS[x]['label']} vs {METRICS[y]['label']}",
        caption=(f"{METRICS[x]['axis']} against {METRICS[y]['axis']} for every evaluated model. "
                 f"Spearman rank correlation is {rho:.2f}: a value near zero means the two "
                 f"objectives are largely independent, so gains on one do not have to be paid for "
                 f"on the other."),
        question=f"Does improving {METRICS[x]['label'].lower()} cost {METRICS[y]['label'].lower()}?")


# ======================================================================================
# 5. ABLATION - what does each part of the model contribute?
# ======================================================================================
def _group_box(lb: pd.DataFrame, group_col: str, metric: str, title: str,
               name: str, question: str, caption: str) -> Optional[Dict]:
    """
    One box plot per group - the honest way to compare families of models.

    A bar of group means would hide how much the members disagree. A box plot shows the median (the
    line), the middle half of the models (the box), and the full spread (the whiskers), so a group
    that looks good only because of one lucky member cannot hide.
    """
    # a missing column must skip this one figure, never bring the whole run down
    if metric not in lb.columns or group_col not in lb.columns:
        return None
    sub = lb.dropna(subset=[metric, group_col])
    if sub.empty:
        return None
    groups = sub.groupby(group_col)[metric]
    # order the groups by their median, best first - so the figure reads as a ranking
    order = groups.median().sort_values(ascending=not METRICS[metric]["higher"]).index.tolist()
    data = [groups.get_group(g).to_numpy(float) * METRICS[metric]["scale"] for g in order]
    if len(order) < 2:
        return None

    fig, ax = plt.subplots(figsize=(S.FULL_WIDTH * 0.75, 0.34 * len(order) + 1.2))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6,
                    medianprops=dict(color=S.C_MILLET, lw=1.3),
                    flierprops=dict(marker="o", ms=2.5, mfc=S.C_NEUTRAL, mec="none"))
    for patch in bp["boxes"]:
        patch.set_facecolor(S.C_OURS)
        patch.set_alpha(0.35)
        patch.set_edgecolor(S.C_OURS)
    # every model as a faint dot on top of its box, so small groups stay honest
    for i, values in enumerate(data, start=1):
        ax.scatter(values, np.full(len(values), i) + np.random.uniform(-0.12, 0.12, len(values)),
                   s=9, color=S.C_MILLET, alpha=0.45, zorder=4)
    ax.set_yticklabels([f"{g}  (n={len(d)})" for g, d in zip(order, data)])
    ax.set_xlabel(METRICS[metric]["axis"])
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", visible=False)
    return S.save(fig, "ablation", name, title=title, caption=caption, question=question)


def ablation_figures(lb: pd.DataFrame) -> List[Dict]:
    """
    What each half of the model contributes: the encoder, the pooling head, and how new the model is.

    Because every model id carries its parts ("<config>__<encoder>__<pooling>") and every part is
    tagged sea_ / mil_, these groupings need no hand-written lists - they are read off the names.
    """
    entries = []
    metric = "web_acc"

    entry = _group_box(
        lb, "encoder", metric,
        title="Accuracy by encoder (pooling head varies within each group)",
        name="ablation_encoder",
        question="Which encoder gives the best accuracy, independently of the pooling head?",
        caption=("WebTraffic accuracy grouped by encoder. Each box covers every model built on that "
                 "encoder, so the spread shows how much the choice of pooling head still matters "
                 "once the encoder is fixed. Dots are individual models; n is the group size."))
    if entry:
        entries.append(entry)

    entry = _group_box(
        lb, "pooling", metric,
        title="Accuracy by pooling head (encoder varies within each group)",
        name="ablation_pooling",
        question="Which pooling head gives the best accuracy, independently of the encoder?",
        caption=("WebTraffic accuracy grouped by MIL pooling head. Heads prefixed mil\\_ are reused "
                 "unchanged from MILLET; heads prefixed sea\\_ are proposed in this work. Dots are "
                 "individual models; n is the group size."))
    if entry:
        entries.append(entry)

    entry = _group_box(
        lb, "origin", metric,
        title="Accuracy by how much of the model is new",
        name="ablation_origin",
        question="Do models with more new components actually perform better?",
        caption=("WebTraffic accuracy grouped by provenance: millet (both halves reproduced from the "
                 "paper), half-ours (one new half), ours (both halves new). This isolates whether "
                 "the gains come from the new components or from the shared training recipe."))
    if entry:
        entries.append(entry)

    entry = _encoder_pooling_heatmap(lb, metric)
    if entry:
        entries.append(entry)
    return entries


def _encoder_pooling_heatmap(lb: pd.DataFrame, metric: str = "web_acc") -> Optional[Dict]:
    """
    The full encoder x pooling grid as a heatmap - which COMBINATION works, not which part.

    A grid answers something the two box plots cannot: whether a head that is mediocre on average is
    actually excellent on one particular encoder. Empty cells are combinations never trained, and are
    left blank rather than filled with a zero.
    """
    sub = lb.dropna(subset=[metric, "encoder", "pooling"])
    if sub.empty:
        return None
    grid = sub.pivot_table(index="encoder", columns="pooling", values=metric, aggfunc="max")
    if grid.shape[0] < 2 or grid.shape[1] < 2:
        return None
    # sort so the best encoder is at the top and the best head on the left - a sorted heatmap
    # reads as a ranking, an unsorted one is just a grid of colours
    grid = grid.loc[grid.max(axis=1).sort_values(ascending=False).index,
                    grid.max(axis=0).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(S.FULL_WIDTH * 0.85, 0.34 * len(grid) + 1.6))
    # "viridis" is perceptually uniform and stays readable in grayscale: equal steps in value look
    # like equal steps in colour, and its lightness increases monotonically.
    im = ax.imshow(grid.to_numpy(float), cmap="viridis", aspect="auto")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels([c.replace("sea_", "").replace("mil_", "") for c in grid.columns],
                       rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels([r.replace("sea_", "").replace("mil_", "") for r in grid.index], fontsize=7)
    ax.grid(visible=False)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            if pd.notna(v):
                # white text on the dark half of the colour map, black on the light half
                dark = v < np.nanmean(grid.to_numpy(float))
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6,
                        color="white" if dark else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label=METRICS[metric]["axis"])
    ax.set_title("Best accuracy for each encoder x pooling combination")
    ax.set_xlabel("pooling head")
    ax.set_ylabel("encoder")
    return S.save(
        fig, "ablation", "ablation_encoder_pooling_grid",
        title="Encoder x pooling accuracy grid",
        caption=("Best WebTraffic accuracy achieved by each encoder and pooling-head combination. "
                 "Blank cells were never trained. Rows and columns are sorted by their best value, "
                 "so strong components appear top-left. Prefixes are dropped from the labels for "
                 "space: all encoders and heads named here are sea\\_ (ours) except those marked "
                 "otherwise in the text."),
        question="Which encoder and pooling-head COMBINATION works best, rather than which part?")
