"""
seanet/report.py - turning the result tables into figures you can put in a report.

What this file is for:
    After a sweep finishes, seanet/results.py has written the numbers (results.csv,
    comparison_vs_millet.csv, summary.csv). This module draws them. It only reads those tables and
    saves PNGs - all the arithmetic (the means, the win/tie/loss records) lives in results.py, so
    the figures and the tables can never disagree.

What it draws:
    Per model, into results/SEA_NET/<model_id>/figures/:
      - results.png       : the model on its own (accuracy / loss / AOPCR spread, accuracy vs length)
      - acc_scatter.png   : our accuracy vs MILLET's, one point per dataset (above the line = we win)
      - loss_scatter.png  : our loss vs MILLET's                          (BELOW the line = we win)
      - aopcr_scatter.png : our AOPCR vs MILLET's                         (above the line = we win)
      - win_tie_loss.png  : the win/tie/loss bars for accuracy, loss and AOPCR
      - means.png         : our mean vs MILLET's mean for all three metrics, side by side
      - acc_diff.png      : the per-dataset accuracy gap, sorted (green = we win)

    Once, into results/SEA_NET/figures/:
      - model_comparison.png : every model's mean accuracy / loss / AOPCR next to MILLET's
      - webtraffic_acc/aopcr/ndcg.png : every model on WebTraffic (one metric per figure)
      - webtraffic_tier_ge95/94/.../90.png : the PAPER figures - models grouped by accuracy tier
        (>=95%, >=94%, ... >=90% on WebTraffic), each tier next to the MILLET paper line. Fewer
        models per figure = clean and readable, instead of one meshed-up wall of bars.
      - winner_dashboard.png : the single best model (good on WebTraffic AND UCR) shown in detail

    All the comparison figures use only the 85 datasets MILLET published - the only ones where a
    fair one-to-one comparison exists.

Input:
    The CSVs under results/SEA_NET/ (data_summary.csv + each model's results).
Output:
    PNG figures + each model's summary table (written via seanet.results.write_summary).

Related files:
    - seanet/results.py -> load_results / build_comparison / summarise_model / compare_models (the
      data this module plots) and all the per-model paths.
    - seanet/data.py    -> read_our_csv() (tolerant CSV reader) and DATA_SUMMARY_CSV.
    - main.py ("report" command) -> calls generate_report().
    - analysis.ipynb    -> a thin notebook that calls generate_report() and shows the saved figures.
"""
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")                                        # headless: we save PNGs (works on Grid5000)
import matplotlib.pyplot as plt                              # noqa: E402
import pandas as pd

from seanet import data as D
from seanet import results as R

# the shared figure (not tied to one model) lives with the other shared outputs
SHARED_FIGURES_DIR = R.SHARED_FIGURES_DIR
DATA_SUMMARY_FIG = os.path.join(SHARED_FIGURES_DIR, "data_summary.png")

# The MILLET baseline models: an encoder MILLET published, with MILLET's own pooling head. Anything
# else is one of OUR new pooling heads, drawn in orange so it stands out in the comparison figure.
# We list only the "<encoder>_<pooling>" part here, because a model id now starts with the config
# file name (e.g. "seanet__mstcn_sep_additive"); is_baseline() strips that prefix before comparing,
# so adding a new config file never means editing this set.
BASELINE_MODELS = {"inceptiontime_conjunctive", "fcn_conjunctive", "resnet_conjunctive",
                   "mstcn_sep_conjunctive", "mstcn_sep_additive"}


def is_baseline(model_id: str) -> bool:
    """True if a model id's encoder_pooling part is one of MILLET's baseline combinations."""
    return model_id.split("__")[-1] in BASELINE_MODELS       # drop the "<config>__" prefix first


# the reran PAPER baselines use one of these encoder backbones (our own encoders all start "mstcn_sep")
PAPER_BACKBONES = ("inceptiontime", "fcn", "resnet")


def is_paper_baseline(model_id: str) -> bool:
    """
    True ONLY for our rerun of the paper baselines (InceptionTime / FCN / ResNet backbones).

    Unlike is_baseline (which also flags our encoder when it uses a MILLET pooling head), this checks
    the ENCODER, so seanet_conjunctive / seanet (additive) count as OUR models here - which is what we
    want when we colour "our models vs the reran baselines" on the WebTraffic figure.
    """
    encoder = model_id.split("__")[-1].split("_")[0]         # first word of "<encoder>_<pooling>"
    return encoder in PAPER_BACKBONES


def short_labels(models: List[str]) -> Dict[str, str]:
    """
    Give every model a short code m1, m2, ... in the order given.

    Model ids are long, so a comparison figure with many models turns into a wall of text. We map each
    full id to a tiny code and print the mapping as a legend, so the bars stay readable even with many
    models. The order is kept stable so the SAME code means the SAME model in every panel.

    models : the list of full model ids.
    returns : {full model id -> "m1"/"m2"/...}.
    """
    return {m: f"m{i}" for i, m in enumerate(models, start=1)}

OURS_COLOUR = "steelblue"
MILLET_COLOUR = "indianred"


def _to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Make sure the given columns are numbers (our CSVs can have stray spaces)."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _save(fig, path: str) -> str:
    """Save a figure, close it (so memory is freed), and return its path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def load_data_summary() -> pd.DataFrame:
    """Load data_summary.csv (facts about the datasets), or an empty frame if it is not there yet."""
    if not os.path.exists(D.DATA_SUMMARY_CSV):
        return pd.DataFrame()
    return _to_num(D.read_our_csv(D.DATA_SUMMARY_CSV),
                   ["n_train", "n_test", "series_length", "n_classes", "imbalance_ratio"])


# --------------------------------------------------------------------------------------
# 1. The shared figure: what the datasets look like (nothing to do with any model)
# --------------------------------------------------------------------------------------
def plot_data_summary(summary: pd.DataFrame, path: str = DATA_SUMMARY_FIG) -> str:
    """Draw the dataset overview (series length, #classes, train size, adjusted-folder count)."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].hist(summary["series_length"].dropna(), bins=30, color=OURS_COLOUR)
    ax[0, 0].set_title("Series length (T)"); ax[0, 0].set_xlabel("T"); ax[0, 0].set_ylabel("# datasets")
    ax[0, 1].hist(summary["n_classes"].dropna(), bins=30, color="seagreen")
    ax[0, 1].set_title("Number of classes"); ax[0, 1].set_xlabel("classes")
    ax[1, 0].hist(summary["n_train"].dropna(), bins=30, color=MILLET_COLOUR)
    ax[1, 0].set_title("Train set size"); ax[1, 0].set_xlabel("# train series"); ax[1, 0].set_ylabel("# datasets")
    counts = summary["used_adjusted_folder"].astype(str).value_counts()
    ax[1, 1].bar(counts.index, counts.values, color=["gray", "orange"][:len(counts)])
    ax[1, 1].set_title("Used the adjusted folder?"); ax[1, 1].set_ylabel("# datasets")
    fig.suptitle("Dataset summary"); fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------------------
# 2. One model's own figures
# --------------------------------------------------------------------------------------
def plot_model_results(results: pd.DataFrame, model_id: str, figdir: str) -> str:
    """Draw one model's own results: how accuracy / loss / AOPCR are spread, and accuracy vs length."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].hist(results["test_acc"].dropna(), bins=20, color=OURS_COLOUR)
    ax[0, 0].set_title("Test accuracy"); ax[0, 0].set_xlabel("accuracy"); ax[0, 0].set_ylabel("# datasets")
    ax[0, 1].hist(results["test_loss"].dropna(), bins=20, color=MILLET_COLOUR)
    ax[0, 1].set_title("Test loss (lower is better)"); ax[0, 1].set_xlabel("loss")
    ax[1, 0].hist(results["test_aopcr"].dropna(), bins=20, color="seagreen")
    ax[1, 0].set_title("AOPCR (interpretability)"); ax[1, 0].set_xlabel("AOPCR"); ax[1, 0].set_ylabel("# datasets")
    ax[1, 1].scatter(results["series_length"], results["test_acc"], s=18, alpha=0.7, color="purple")
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_title("Accuracy vs series length"); ax[1, 1].set_xlabel("T (log scale)"); ax[1, 1].set_ylabel("accuracy")
    fig.suptitle(f"{model_id} - own results over {len(results)} datasets"); fig.tight_layout()
    return _save(fig, os.path.join(figdir, "results.png"))


def _scatter_vs_millet(overlap: pd.DataFrame, metric: str, model_id: str, figdir: str,
                       lower_is_better: bool) -> str:
    """
    Draw one "ours vs MILLET" scatter: one point per dataset, plus the y=x line and both means.

    The y=x line is the whole point of the picture: a point off that line is a dataset where the two
    models really differ, and which SIDE it falls on says who won. The big star marks the two means,
    so you can see the overall verdict and the per-dataset spread in the same glance.

    overlap : the 85 datasets that have a MILLET baseline.
    metric : "acc", "loss" or "aopcr".
    model_id : the model being drawn (for the title / axis labels).
    figdir : where to save.
    lower_is_better : True for loss - it flips which side of the line is a win.
    returns : the saved figure path.
    """
    ours, theirs = overlap[f"ours_{metric}"], overlap[f"millet_{metric}"]
    win_side = "BELOW" if lower_is_better else "above"
    fig = plt.figure(figsize=(6.2, 6.2))
    plt.scatter(theirs, ours, s=25, alpha=0.7, color=OURS_COLOUR, label="one dataset")

    lo = float(min(theirs.min(), ours.min()))                 # make the box square so y=x is a true diagonal
    hi = float(max(theirs.max(), ours.max()))
    pad = (hi - lo) * 0.05 or 0.02
    lo, hi = lo - pad, hi + pad
    plt.plot([lo, hi], [lo, hi], "r--", lw=1, label="equal (y = x)")
    plt.scatter([theirs.mean()], [ours.mean()], marker="*", s=320, color="gold",
                edgecolor="black", zorder=5, label="the two means")

    plt.xlabel(f"MILLET {metric}"); plt.ylabel(f"{model_id} {metric}")
    plt.title(f"{metric}: {model_id} vs MILLET\n({len(overlap)} datasets; {win_side} the line = we win)")
    plt.xlim(lo, hi); plt.ylim(lo, hi); plt.grid(alpha=0.3); plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    return _save(fig, os.path.join(figdir, f"{metric}_scatter.png"))


def plot_win_tie_loss(overlap: pd.DataFrame, model_id: str, figdir: str) -> str:
    """Draw the win / tie / loss bars for all three metrics (accuracy, loss, AOPCR)."""
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, metric in zip(ax, R.COMPARED_METRICS):
        vc = overlap[f"{metric}_outcome"].value_counts().reindex(["win", "tie", "loss"]).fillna(0)
        bars = a.bar(vc.index, vc.values, color=["green", "gray", "red"])
        a.set_title(f"{metric}: {vc['win']:.0f}/{vc['tie']:.0f}/{vc['loss']:.0f}")
        a.set_ylabel("# datasets")
        for b, v in zip(bars, vc.values):
            a.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"{model_id} vs MILLET, over the {len(overlap)} datasets MILLET published "
                 f"(win / tie / loss)")
    fig.tight_layout()
    return _save(fig, os.path.join(figdir, "win_tie_loss.png"))


def plot_means(overlap: pd.DataFrame, model_id: str, figdir: str) -> str:
    """
    Draw our mean next to MILLET's mean for accuracy, loss and AOPCR - the headline verdict.

    Each metric gets its own panel because they live on different scales (accuracy is 0..1, AOPCR
    can be 15), so putting them on one axis would squash two of them flat.
    """
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, (metric, (_csv, _band, lower)) in zip(ax, R.COMPARED_METRICS.items()):
        ours = float(overlap[f"ours_{metric}"].mean())
        theirs = float(overlap[f"millet_{metric}"].mean())
        bars = a.bar([0, 1], [ours, theirs], color=[OURS_COLOUR, MILLET_COLOUR])
        a.set_title(f"mean {metric}" + ("  (lower is better)" if lower else ""))
        a.set_xticks([0, 1])                                  # set the ticks before their labels
        a.set_xticklabels([model_id, "MILLET"], rotation=15, ha="right", fontsize=8)
        for b, v in zip(bars, [ours, theirs]):
            a.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"{model_id} vs MILLET - means over the {len(overlap)} datasets MILLET published")
    fig.tight_layout()
    return _save(fig, os.path.join(figdir, "means.png"))


def plot_acc_diff(overlap: pd.DataFrame, model_id: str, figdir: str) -> str:
    """Draw the per-dataset accuracy gap (ours - MILLET), sorted, green where we win."""
    d = overlap.dropna(subset=["acc_diff"]).sort_values("acc_diff")
    fig = plt.figure(figsize=(7, max(3, 0.28 * len(d))))
    plt.barh(d["dataset"], d["acc_diff"], color=["green" if v > 0 else "red" for v in d["acc_diff"]])
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel(f"accuracy difference ({model_id} - MILLET)")
    plt.title(f"Per-dataset accuracy gap (green = we win)")
    plt.tight_layout()
    return _save(fig, os.path.join(figdir, "acc_diff.png"))


def plot_model_figures(model_id: str, verbose: bool = True) -> List[str]:
    """
    Draw every figure for ONE model, into results/SEA_NET/<model_id>/figures/.

    model_id : the model to draw, e.g. "mstcn_sep_additive".
    verbose : print what was written.
    returns : the list of saved figure paths.
    """
    figdir = R.figures_dir(model_id)
    results = _to_num(R.load_results(model_id),
                      ["test_acc", "test_loss", "test_aopcr", "test_ndcg", "series_length",
                       "n_classes", "params", "model_size_mb", "train_time_s"])
    paths: List[str] = []
    if results.empty:
        if verbose:
            print(f"  {model_id}: no results yet - nothing to draw.")
        return paths

    paths.append(plot_model_results(results, model_id, figdir))

    cmp = _to_num(R.build_comparison(model_id, verbose=False),
                  [f"{side}_{m}" for m in R.COMPARED_METRICS for side in ("ours", "millet")]
                  + [f"{m}_diff" for m in R.COMPARED_METRICS])
    overlap = R.overlap_rows(cmp)
    if len(overlap):                                          # the MILLET comparison needs the 85
        for metric, (_csv, _band, lower) in R.COMPARED_METRICS.items():
            paths.append(_scatter_vs_millet(overlap, metric, model_id, figdir, lower))
        paths.append(plot_win_tie_loss(overlap, model_id, figdir))
        paths.append(plot_means(overlap, model_id, figdir))
        paths.append(plot_acc_diff(overlap, model_id, figdir))
    elif verbose:
        print(f"  {model_id}: no dataset with a MILLET baseline finished yet - "
              f"skipping the comparison figures.")
    return paths


# --------------------------------------------------------------------------------------
# 3. The cross-model figure: which encoder+pooling wins?
# --------------------------------------------------------------------------------------
def plot_model_comparison(cross: pd.DataFrame, figdir: str = SHARED_FIGURES_DIR) -> List[str]:
    """
    Draw every model's mean accuracy / loss / AOPCR next to MILLET's, over the 85 published datasets.

    MILLET gets its own red bar in each panel, so "did we beat the baseline?" is answered by looking
    at whether a bar clears it. Our new pooling heads are orange; the baseline-style models are blue.

    cross : the cross-model frame from seanet.results.compare_models().
    figdir : where to save the PNG.
    returns : the saved figure path in a list (empty if there is nothing to plot).
    """
    if cross.empty or "mean_acc_ours" not in cross.columns:
        return []
    # Model ids are long ("<config>__<encoder>_<pooling>"), so with many models the x-axis becomes an
    # unreadable wall of text. Instead we give every model a SHORT code (m1, m2, ...) - fixed once here
    # so the same code means the same model in all three panels - and print a legend under the figure.
    code = short_labels(list(cross["model"]))                # {full model id -> "m1"/"m2"/...}
    fig, ax = plt.subplots(1, 3, figsize=(13, 5))
    for a, (metric, (_csv, _band, lower)) in zip(ax, R.COMPARED_METRICS.items()):
        col = f"mean_{metric}_ours"
        s = cross.dropna(subset=[col]).sort_values(col, ascending=bool(lower))   # best first
        names = [code[m] for m in s["model"]] + ["MILLET"]   # short codes on the axis
        values = list(s[col]) + [float(s[f"mean_{metric}_millet"].iloc[0])]
        colours = [OURS_COLOUR if is_baseline(m) else "darkorange" for m in s["model"]]
        colours.append(MILLET_COLOUR)
        bars = a.bar(range(len(names)), values, color=colours)
        a.set_xticks(range(len(names)))
        a.set_xticklabels(names, rotation=0, ha="center", fontsize=8)
        a.set_title(f"mean {metric}" + ("  (lower is better)" if lower else ""))
        for b, v in zip(bars, values):
            if pd.notna(v):
                a.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    n = int(cross["millet85_datasets"].max())
    fig.suptitle(f"Model comparison over the {n} datasets MILLET published   "
                 f"[orange = our new pooling heads, red = MILLET]")
    # the legend: "m1 = <config name>" per model, laid out in columns under the plots
    legend = "   ".join(f"{code[m]} = {m.split('__')[0]}" for m in cross['model'])
    fig.tight_layout(rect=(0, 0.10, 1, 1))                   # leave a strip at the bottom for the legend
    fig.text(0.01, 0.01, "legend:  " + legend, fontsize=7, va="bottom", family="monospace", wrap=True)
    return [_save(fig, os.path.join(figdir, "model_comparison.png"))]


# --------------------------------------------------------------------------------------
# 3b. The WebTraffic-only figure (our headline dataset)
# --------------------------------------------------------------------------------------
def _webtraffic_legend_text(code: Dict[str, str], models: List[str], per_line: int = 4) -> tuple:
    """Build the 'm1 = name   m2 = name ...' legend, wrapped to a few entries per line."""
    entries = [f"{code[m]} = {m.split('__')[0]}" for m in models]
    lines = ["   ".join(entries[i:i + per_line]) for i in range(0, len(entries), per_line)]
    return "\n".join(lines), len(lines)


def _plot_webtraffic_metric(df, paper: Dict[str, float], code: Dict[str, str], col: str,
                            title: str, target, figdir: str) -> Optional[str]:
    """
    Draw ONE WebTraffic metric (accuracy / AOPCR / NDCG) as its own full-size figure.

    Orange bars = our models, blue bars = our rerun of the paper baselines. The red dashed line is the
    MILLET PAPER number (the bar to beat); the green dotted line is our target (accuracy only). The
    x-axis uses short m1/m2 codes with a legend underneath, so it stays readable with many models.
    """
    s = df.dropna(subset=[col])
    if s.empty:
        return None
    models = list(s["model"])
    names = [code[m] for m in models]
    values = list(s[col])
    colours = [OURS_COLOUR if is_paper_baseline(m) else "darkorange" for m in models]
    n = len(names)
    legend_text, n_lines = _webtraffic_legend_text(code, models)
    bottom = min(0.45, 0.06 + 0.028 * n_lines)               # leave room at the bottom for the legend

    # width grows with the number of models so the bars never get squashed (that was the whole point)
    fig, a = plt.subplots(figsize=(max(11, 0.5 * n), 6))
    bars = a.bar(range(n), values, color=colours)
    a.set_xticks(range(n))
    a.set_xticklabels(names, fontsize=8)
    a.set_ylabel(title)
    a.set_title(f"WebTraffic {title}   "
                f"[orange = our models, blue = reran baselines, red dashed = MILLET paper]", fontsize=11)
    for b, v in zip(bars, values):
        a.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    if col in paper:                                         # the paper baseline as a line to beat
        a.axhline(paper[col], ls="--", color=MILLET_COLOUR, lw=1.6, label=f"MILLET paper ({paper[col]:.3f})")
    if target is not None:                                   # our goal (only meaningful for accuracy)
        a.axhline(target, ls=":", color="green", lw=1.3, label=f"target ({target})")
    a.legend(fontsize=9, loc="lower left")
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.text(0.01, 0.01, "legend:  " + legend_text, fontsize=7, va="bottom", family="monospace")
    return _save(fig, os.path.join(figdir, f"webtraffic_{col}.png"))


def plot_webtraffic_comparison(figdir: str = SHARED_FIGURES_DIR) -> List[str]:
    """
    Draw the WebTraffic comparison as THREE separate figures - accuracy, AOPCR and NDCG - so each one is
    full-size and easy to read in detail (instead of one cramped 3-in-1 panel).

    Files: webtraffic_acc.png, webtraffic_aopcr.png, webtraffic_ndcg.png (all under results/SEA_NET/
    figures/). Everything is rebuilt from whatever has finished, so new models appear automatically. The
    short m1/m2 codes are shared across the three figures, so m3 means the same model in all of them.
    """
    df = R.webtraffic_table()
    if df.empty:
        return []
    paper = R.webtraffic_paper_baseline()
    code = short_labels(list(df["model"]))                   # one shared code map -> same code in all 3
    panels = [("acc", "accuracy", 0.96), ("aopcr", "AOPCR", None), ("ndcg", "NDCG", None)]
    paths = []
    for col, title, target in panels:
        path = _plot_webtraffic_metric(df, paper, code, col, title, target, figdir)
        if path:
            paths.append(path)
    return paths


# --------------------------------------------------------------------------------------
# 3c. Paper figures: accuracy TIERS on WebTraffic + the WINNER dashboard
#
# Problem this solves: with sv1..sv5 we now have MANY models, so one bar chart of everyone is a
# meshed-up wall of bars. Fix = show the models in ACCURACY TIERS. For each threshold (>=95%, >=94%,
# ... >=90% on WebTraffic) we draw ONE clean figure of just the models that clear that bar, next to
# the MILLET paper number. Higher tiers have few models, so they read cleanly - perfect for a paper.
#
# Colours use the Okabe-Ito colour-blind-safe palette (the standard for scientific figures), so the
# figures stay readable in print and for colour-blind readers.
# --------------------------------------------------------------------------------------
# Okabe-Ito: blue = our models, orange = our rerun of the paper baselines, black = the MILLET line.
OK_BLUE = "#0072B2"      # our new models
OK_ORANGE = "#E69F00"    # our rerun of the paper baselines (InceptionTime / FCN / ResNet)
OK_GREEN = "#009E73"     # the "target" line (accuracy goal)
OK_BLACK = "#000000"     # the MILLET-paper reference line (the bar to beat)

# The WebTraffic accuracy tiers we draw, high to low (cumulative: >=0.95, >=0.94, ... >=0.90).
ACC_TIERS = [0.95, 0.94, 0.93, 0.92, 0.91, 0.90]


def _short_name(model_id: str) -> str:
    """Turn a long model id ('seanet_x__mstcn_sep_dualstream') into just its config name for a label."""
    return model_id.split("__")[0]


def _load_ucr_means() -> pd.DataFrame:
    """
    Read the cross-model table (model_comparison.csv) so tier/winner figures can also show the UCR
    mean accuracy (over the 85 MILLET datasets). It is written by `python main.py results`/`report`.

    returns : a DataFrame indexed by model id with mean_acc_ours / mean_acc_millet (empty if missing).
    """
    if not os.path.exists(R.MODEL_COMPARISON_CSV):
        return pd.DataFrame()
    df = D.read_our_csv(R.MODEL_COMPARISON_CSV)
    if "model" not in df.columns:
        return pd.DataFrame()
    for col in ["mean_acc_ours", "mean_acc_millet"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("model")


def _hbar_panel(ax, names, values, colours, title, ref=None, ref_label=None,
                target=None, target_label=None, fmt="{:.3f}"):
    """
    Draw ONE metric as a horizontal bar chart (one bar per model), best model on top.

    Horizontal bars are used on purpose: model names are long, and on a horizontal axis they read
    left-to-right instead of being rotated. `ref` is the MILLET paper value (a vertical line = the
    bar to beat); `target` is our goal line (accuracy only).

    ax : the subplot to draw on.   names/values/colours : one entry per model (already ordered).
    title : panel title (say the metric + which direction is good).
    ref / target : optional vertical reference lines.   fmt : how to print the value at each bar end.
    """
    y = range(len(names))
    bars = ax.barh(list(y), values, color=colours, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()                                        # best model (first row) at the TOP
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    for b, v in zip(bars, values):
        if pd.notna(v):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt.format(v),
                    va="center", ha="left", fontsize=7)
    if ref is not None and pd.notna(ref):
        ax.axvline(ref, ls="--", color=OK_BLACK, lw=1.4, zorder=2, label=ref_label)
    if target is not None:
        ax.axvline(target, ls=":", color=OK_GREEN, lw=1.4, zorder=2, label=target_label)
    if (ref is not None and ref_label) or (target is not None and target_label):
        ax.legend(fontsize=7, loc="lower right")


def _plot_one_tier(sub: pd.DataFrame, threshold: float, paper: Dict[str, float],
                   ucr: pd.DataFrame, figdir: str) -> str:
    """
    Draw one accuracy tier: the models with WebTraffic acc >= threshold, side by side, on 4-6 metrics.

    Panels (each a horizontal bar chart, models sorted by accuracy): accuracy, AOPCR, NDCG, params (K),
    size (MB), and - if we have full-sweep numbers - the UCR-85 mean accuracy. Higher-is-better metrics
    (acc, aopcr, ndcg, ucr) show the MILLET paper number as a black dashed line; lower-is-better metrics
    (params, size) have no line (smaller bar = smaller model = better).

    sub : the models in this tier (already sorted best-accuracy-first).   threshold : e.g. 0.95.
    paper : MILLET's published WebTraffic numbers.   ucr : the cross-model UCR means (may be empty).
    returns : the saved figure path.
    """
    models = list(sub["model"])
    names = [_short_name(m) for m in models]
    colours = [OK_ORANGE if is_paper_baseline(m) else OK_BLUE for m in models]

    # decide which panels to draw. UCR mean only if at least one model in this tier has it.
    ucr_acc = [float(ucr.loc[m, "mean_acc_ours"]) if (len(ucr) and m in ucr.index
               and pd.notna(ucr.loc[m, "mean_acc_ours"])) else float("nan") for m in models]
    have_ucr = any(pd.notna(v) for v in ucr_acc)
    ucr_ref = float(ucr["mean_acc_millet"].dropna().iloc[0]) if (len(ucr)
              and "mean_acc_millet" in ucr.columns and ucr["mean_acc_millet"].notna().any()) else None

    # (column, title, values, reference line, target line, number format)
    panels = [
        ("acc",   "WebTraffic accuracy (higher better)", list(sub["acc"]),
         paper.get("acc"), 0.96, "{:.3f}"),
        ("aopcr", "AOPCR - interpretability (higher better)", list(sub["aopcr"]),
         paper.get("aopcr"), None, "{:.2f}"),
        ("ndcg",  "NDCG (higher better)", list(sub["ndcg"]),
         paper.get("ndcg"), None, "{:.3f}"),
        ("params", "params in thousands (lower better)", [v / 1e3 for v in sub["params"]],
         None, None, "{:.0f}K"),
        ("size",  "model size MB (lower better)", list(sub["size_mb"]),
         None, None, "{:.2f}"),
    ]
    if have_ucr:
        panels.append(("ucr", "UCR-85 mean accuracy (higher better)", ucr_acc, ucr_ref, None, "{:.3f}"))

    n_panels = len(panels)
    # height grows with the number of models so the bars never get squashed
    fig, axes = plt.subplots(1, n_panels, figsize=(3.1 * n_panels, max(3.0, 0.5 * len(models) + 1.4)))
    if n_panels == 1:
        axes = [axes]
    for ax, (col, title, values, ref, target, fmt) in zip(axes, panels):
        ref_label = "MILLET paper" if (ref is not None and pd.notna(ref)) else None
        target_label = "target 0.96" if target is not None else None
        _hbar_panel(ax, names, values, colours, title, ref=ref, ref_label=ref_label,
                    target=target, target_label=target_label, fmt=fmt)
    n = len(models)
    fig.suptitle(f"WebTraffic tier: accuracy >= {threshold:.2f}   ({n} model{'s' if n != 1 else ''})   "
                 f"[blue = our models, orange = reran baselines, black dashed = MILLET paper]",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, os.path.join(figdir, f"webtraffic_tier_ge{int(round(threshold * 100))}.png"))


def plot_webtraffic_tiers(figdir: str = SHARED_FIGURES_DIR) -> List[str]:
    """
    Draw the WebTraffic accuracy-tier figures (>=95%, >=94%, ... >=90%), one clean figure per tier.

    We skip a tier if it has no models, and we skip a tier whose model list is IDENTICAL to the tier
    just above it (drawing the same picture twice adds nothing) - so you get one figure per DISTINCT
    group of models, which is exactly what makes them paper-ready.

    returns : the list of saved figure paths (one per distinct tier).
    """
    df = R.webtraffic_table()                                # already sorted best-accuracy-first
    if df.empty:
        return []
    paper = R.webtraffic_paper_baseline()
    ucr = _load_ucr_means()
    paths: List[str] = []
    last_members = None
    for t in ACC_TIERS:
        sub = df[df["acc"] >= t]
        if sub.empty:
            continue
        members = tuple(sub["model"])                        # who is in this tier
        if members == last_members:                          # same models as the tier above -> skip
            continue
        paths.append(_plot_one_tier(sub, t, paper, ucr, figdir))
        last_members = members
    return paths


def pick_winner(df: pd.DataFrame, ucr: pd.DataFrame) -> Optional[str]:
    """
    Choose the single best model: good on WebTraffic AND good on the UCR benchmark.

    Simple rule (easy to explain in the paper): among models that have BOTH a WebTraffic accuracy and
    a UCR-85 mean accuracy, pick the highest average of the two. If no model has UCR numbers yet
    (only the fast WebTraffic screen has run), fall back to the best WebTraffic accuracy.

    df : the WebTraffic table.   ucr : the cross-model UCR means (may be empty).
    returns : the winning model id, or None if there is nothing to pick from.
    """
    if df.empty:
        return None
    both = [m for m in df["model"] if len(ucr) and m in ucr.index
            and pd.notna(ucr.loc[m, "mean_acc_ours"])]
    if both:
        web = df.set_index("model")["acc"]
        score = {m: 0.5 * float(web[m]) + 0.5 * float(ucr.loc[m, "mean_acc_ours"]) for m in both}
        return max(score, key=score.get)
    return str(df.iloc[0]["model"])                          # WebTraffic-only fallback: best accuracy


def plot_winner_dashboard(figdir: str = SHARED_FIGURES_DIR) -> List[str]:
    """
    Draw ONE detailed figure for the winning model, so a reader can see all its headline numbers at
    once: WebTraffic accuracy / AOPCR / NDCG vs the MILLET paper, the UCR-85 mean accuracy vs MILLET,
    and its size (params + MB). This is the "hero" figure for the paper.

    returns : the saved figure path in a list (empty if no model has results yet).
    """
    df = R.webtraffic_table()
    if df.empty:
        return []
    paper = R.webtraffic_paper_baseline()
    ucr = _load_ucr_means()
    winner = pick_winner(df, ucr)
    if winner is None:
        return []
    row = df[df["model"] == winner].iloc[0]
    has_ucr = len(ucr) and winner in ucr.index and pd.notna(ucr.loc[winner, "mean_acc_ours"])

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))

    # Row 1: WebTraffic, ours (blue) vs MILLET paper (grey), one metric per panel (different scales).
    web_panels = [("acc", "WebTraffic accuracy", "{:.3f}"),
                  ("aopcr", "WebTraffic AOPCR", "{:.2f}"),
                  ("ndcg", "WebTraffic NDCG", "{:.3f}")]
    for a, (col, title, fmt) in zip(ax[0], web_panels):
        ours = float(row[col]) if pd.notna(row[col]) else float("nan")
        theirs = paper.get(col, float("nan"))
        bars = a.bar([0, 1], [ours, theirs], color=[OK_BLUE, "#999999"], zorder=3)
        a.set_xticks([0, 1]); a.set_xticklabels(["winner", "MILLET\npaper"], fontsize=9)
        a.set_title(title, fontsize=11); a.grid(axis="y", alpha=0.25, zorder=0)
        for b, v in zip(bars, [ours, theirs]):
            if pd.notna(v):
                a.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)

    # Row 2, panel 1: UCR-85 mean accuracy, ours vs MILLET (only if the full sweep has run).
    a = ax[1, 0]
    if has_ucr:
        ours = float(ucr.loc[winner, "mean_acc_ours"])
        theirs = float(ucr.loc[winner, "mean_acc_millet"]) if pd.notna(ucr.loc[winner, "mean_acc_millet"]) else float("nan")
        bars = a.bar([0, 1], [ours, theirs], color=[OK_BLUE, "#999999"], zorder=3)
        a.set_xticks([0, 1]); a.set_xticklabels(["winner", "MILLET"], fontsize=9)
        a.set_title("UCR-85 mean accuracy", fontsize=11); a.grid(axis="y", alpha=0.25, zorder=0)
        for b, v in zip(bars, [ours, theirs]):
            if pd.notna(v):
                a.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    else:
        a.axis("off")
        a.text(0.5, 0.5, "UCR-85 mean:\nrun the full sweep\nto fill this in", ha="center", va="center",
               fontsize=10, color="gray")

    # Row 2, panel 2: model footprint (params + size), our "small model" goal.
    a = ax[1, 1]
    params_k = float(row["params"]) / 1e3 if pd.notna(row["params"]) else float("nan")
    size_mb = float(row["size_mb"]) if pd.notna(row["size_mb"]) else float("nan")
    bars = a.bar([0, 1], [params_k, size_mb], color=[OK_BLUE, OK_ORANGE], zorder=3)
    a.set_xticks([0, 1]); a.set_xticklabels(["params (K)", "size (MB)"], fontsize=9)
    a.set_title("Model footprint (smaller = better)", fontsize=11); a.grid(axis="y", alpha=0.25, zorder=0)
    for b, v, fmt in zip(bars, [params_k, size_mb], ["{:.0f}K", "{:.2f}MB"]):
        if pd.notna(v):
            a.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)

    # Row 2, panel 3: a plain-words summary box (the headline numbers, spelled out).
    a = ax[1, 2]; a.axis("off")
    lines = [f"WINNER: {_short_name(winner)}",
             f"encoder+pooling: {winner.split('__')[-1]}",
             "",
             f"WebTraffic acc : {row['acc']:.3f}" + (f"   (MILLET {paper['acc']:.3f})" if 'acc' in paper else ""),
             f"WebTraffic AOPCR: {row['aopcr']:.2f}" + (f"   (MILLET {paper['aopcr']:.2f})" if 'aopcr' in paper else ""),
             f"WebTraffic NDCG : {row['ndcg']:.3f}" if pd.notna(row['ndcg']) else "WebTraffic NDCG : n/a"]
    if has_ucr:
        lines.append(f"UCR-85 mean acc : {float(ucr.loc[winner, 'mean_acc_ours']):.3f}")
    lines += ["", f"params: {int(row['params']):,}" if pd.notna(row['params']) else "params: n/a",
              f"size  : {size_mb:.2f} MB" if pd.notna(size_mb) else "size  : n/a"]
    a.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle(f"Winner model: {_short_name(winner)}   "
                 f"(best on WebTraffic + UCR)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return [_save(fig, os.path.join(figdir, "winner_dashboard.png"))]


# --------------------------------------------------------------------------------------
# 4. The whole report
# --------------------------------------------------------------------------------------
def generate_report(models: Optional[List[str]] = None, verbose: bool = True) -> Dict:
    """
    Do everything: refresh every model's comparison + summary, draw all their figures, and draw the
    cross-model comparison.

    models : which models to report on (default: every model that has results).
    verbose : if True, print the rankings and where things were written.
    returns : a dict with "models" (each model's summary dict) and "figures" (list of PNG paths).
    """
    models = R.discover_models() if models is None else models
    figures: List[str] = []

    # the shared dataset overview (about the data, so it is drawn once, not per model)
    summary = load_data_summary()
    if len(summary):
        figures.append(plot_data_summary(summary))

    if not models:
        if verbose:
            print("No model has any results yet. Run e.g. `python main.py train --model seanet`.")
        return {"models": [], "figures": figures}

    # compare_models refreshes each model's comparison_vs_millet.csv + summary.csv, then ranks them
    cross = R.compare_models(models=models, verbose=False)

    if verbose:
        print("=== SEA-Net report ===")
    for model_id in models:
        if verbose:
            print(f"\n--- {model_id} ---")
        figures += plot_model_figures(model_id, verbose=verbose)
        if verbose:
            perf = R.summarise_model(model_id)
            for key, value in perf.items():
                print(f"  {key:24s}: {value}")

    figures += plot_model_comparison(cross)
    figures += plot_webtraffic_comparison()                  # our headline dataset, vs the paper baseline
    figures += plot_webtraffic_tiers()                       # the clean accuracy-tier figures (paper)
    figures += plot_winner_dashboard()                       # the single "winner" hero figure (paper)

    if verbose:
        print("\n=== which model wins? ===")
        R._print_model_comparison(cross, R.MODEL_COMPARISON_CSV)
        print(f"\nwrote {len(figures)} figures:")
        for p in figures:
            print("  ", p)
    return {"models": cross.to_dict("records") if len(cross) else [], "figures": figures}
