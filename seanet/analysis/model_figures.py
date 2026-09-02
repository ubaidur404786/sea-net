"""
seanet/analysis/model_figures.py - turning the result tables into figures you can put in a report.

What this file is for:
    After a sweep finishes, seanet/results.py has written the numbers (results.csv,
    comparison_vs_millet.csv, summary.csv). This module draws them. It only reads those tables and
    saves PNGs - all the arithmetic (the means, the win/tie/loss records) lives in results.py, so
    the figures and the tables can never disagree.

What it draws:
    Per model, into results/top_results/SEA_NET/<model_id>/figures/:
      - results.png       : the model on its own (accuracy / loss / AOPCR spread, accuracy vs length)
      - acc_scatter.png   : our accuracy vs MILLET's, one point per dataset (above the line = we win)
      - loss_scatter.png  : our loss vs MILLET's                          (BELOW the line = we win)
      - aopcr_scatter.png : our AOPCR vs MILLET's                         (above the line = we win)
      - win_tie_loss.png  : the win/tie/loss bars for accuracy, loss and AOPCR
      - means.png         : our mean vs MILLET's mean for all three metrics, side by side
      - acc_diff.png      : the per-dataset accuracy gap, sorted (green = we win)

    Once, into results/top_results/SEA_NET/figures/:
      - model_comparison.png : every model's mean accuracy / loss / AOPCR next to MILLET's
      - webtraffic_acc/aopcr/ndcg.png : every model on WebTraffic (one metric per figure)
      - webtraffic_tier_ge95/94/.../90.png : the PAPER figures - models grouped by accuracy tier
        (>=95%, >=94%, ... >=90% on WebTraffic), each tier next to the MILLET paper line. Fewer
        models per figure = clean and readable, instead of one meshed-up wall of bars.
      - winner_dashboard.png : the single best model (good on WebTraffic AND UCR) shown in detail

    All the comparison figures use only the 85 datasets MILLET published - the only ones where a
    fair one-to-one comparison exists.

Input:
    The CSVs under results/top_results/SEA_NET/ (data_summary.csv + each model's results).
Output:
    PNG figures + each model's summary table (written via seanet.results.write_summary).

Related files:
    - seanet/results.py -> load_results / build_comparison / summarise_model / compare_models (the
      data this module plots) and all the per-model paths.
    - seanet/data.py    -> read_our_csv() (tolerant CSV reader) and data_summary_csv().
    - main.py ("report" command) -> calls generate_report().
    - main.py ("report") -> the command that calls generate_report().
"""
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")                                        # headless: we save PNGs (works on Grid5000)
import matplotlib.pyplot as plt                              # noqa: E402
import pandas as pd

from seanet import data as D
from seanet import results as R
# these three read a model id; they live in config.py, which is torch-free, so importing it here
# keeps this module torch-free too (it must stay importable without torch).
from seanet.config import is_millet, split_model_id

# the shared figure (not tied to one model) lives with the other shared outputs
SHARED_FIGURES_DIR = R.SHARED_FIGURES_DIR
DATA_SUMMARY_FIG = os.path.join(SHARED_FIGURES_DIR, "data_summary.png")

# Telling OUR work from MILLET's, straight off the model id.
#
# A model id is "<config>__<encoder>__<pooling>", and every encoder / pooling name carries its own
# origin tag: "sea_" = ours, "mil_" = MILLET's, reused unchanged (see seanet/config.py). So these
# two questions are now just a prefix check - there is no hand-written list of model names to keep
# up to date any more, and a brand-new encoder or head is classified correctly the day it is added.


def is_baseline(model_id: str) -> bool:
    """
    True if NOTHING in this model is ours - a MILLET encoder AND a MILLET pooling head.

    These are the paper's own models, rerun here under our training recipe. Everything else has at
    least one new part, and is drawn in orange so it stands out in the comparison figures.
    """
    _config, encoder, pooling = split_model_id(model_id)
    return is_millet(encoder) and is_millet(pooling)


def is_paper_baseline(model_id: str) -> bool:
    """
    True ONLY for our rerun of the paper baselines - the ones using a MILLET ENCODER
    (InceptionTime / FCN / ResNet), whatever pooling head sits on top.

    Unlike is_baseline (which needs BOTH halves to be MILLET's), this asks only about the encoder.
    That is what we want when colouring "our encoders vs the paper's encoders" on the WebTraffic
    figure: seanet_conjunctive (our encoder + MILLET's head) counts as OURS there.
    """
    _config, encoder, _pooling = split_model_id(model_id)
    return is_millet(encoder)


def is_fully_ours(model_id: str) -> bool:
    """
    True when BOTH halves are ours - our encoder AND our pooling head.

    This is the strictest test, and it is the one that matters for the paper's headline model. A
    model like seanet_conjunctive uses our encoder but MILLET's Conjunctive head, so it cannot be
    presented as "our architecture"; it belongs in the ablation table instead.
    """
    _config, encoder, pooling = split_model_id(model_id)
    return (not is_millet(encoder)) and (not is_millet(pooling))


# Our own rerun of MILLET: the SAME architecture, trained by us, under our recipe. This - not the
# published table - is the fair reference for our numbers, because it was measured with our code,
# our splits and our metric implementations. (Their published AOPCR in particular is ~5.7x larger
# than what our AOPCR code produces for the very same model, so the two cannot share an axis.)
MILLET_RERUN_ID = "millet__mil_inceptiontime__mil_conjunctive"


def millet_rerun_row(cross: pd.DataFrame) -> Optional[pd.Series]:
    """
    Find our MILLET rerun in the cross-model table, so figures can draw it as the reference line.

    cross : the frame from seanet.results.compare_models() (indexed by position, with a "model" col).
    returns : that row, or None if MILLET has not been swept yet.
    """
    if cross.empty or "model" not in cross.columns:
        return None
    hit = cross[cross["model"] == MILLET_RERUN_ID]
    return hit.iloc[0] if len(hit) else None


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
    summary_csv = D.data_summary_csv()
    if not os.path.exists(summary_csv):
        return pd.DataFrame()
    return _to_num(D.read_our_csv(summary_csv),
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
    Draw every figure for ONE model, into results/top_results/SEA_NET/<model_id>/figures/.

    model_id : the model to draw, e.g. "seanet__sea_mstcn_sep__mil_additive".
    verbose : print what was written.
    returns : the list of saved figure paths.
    """
    figdir = R.figures_dir(model_id)
    results = _to_num(R.mean_over_seeds(R.load_results(model_id)),   # several seeds -> their average
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
def plot_model_comparison(cross: pd.DataFrame, figdir: str = SHARED_FIGURES_DIR,
                          top_n: int = 15) -> List[str]:
    """
    Draw the best models' mean accuracy / loss / AOPCR over the 85 datasets MILLET published.

    HOW THIS FIGURE IS BUILT (and why the old one was unreadable):
      - The old version put ALL ~66 swept models on a vertical x-axis, renamed them m1..m66, and
        printed the m1 = name mapping as a paragraph under the plots. The paragraph overlapped the
        bars and no reader could match a bar to a model. Nothing about that is fixable by tweaking
        font sizes: 66 bars is simply too many for one figure.
      - Now we show the TOP N only (default 15), as HORIZONTAL bars, with the real config names on
        the axis. Long names read fine sideways, so no codes and no legend paragraph are needed.
      - All three panels SHARE one order (ranked by accuracy) and the names are printed ONCE, on the
        left. So you read one model straight across the row and see all three of its numbers - the
        way a results table reads. Sorting each panel separately would force the names to be
        repeated three times and would stop you tracking a model across panels.

    THE TWO REFERENCE LINES
      solid black  = our MILLET RERUN (same architecture, our code, our recipe) - the fair bar to beat.
      dashed grey  = MILLET's PUBLISHED number, shown for accuracy and loss only.
    AOPCR deliberately gets NO published line: their published AOPCR is about 5.7x larger than what
    our AOPCR code computes for the very same model, so drawing it would squash every bar to nothing
    and invite a false conclusion. That is exactly what the old figure did.

    cross : the cross-model frame from seanet.results.compare_models().
    figdir : where to save the PNG.  top_n : how many models to draw.
    returns : the saved figure path in a list (empty if there is nothing to plot).
    """
    if cross.empty or "mean_acc_ours" not in cross.columns:
        return []
    ranked = cross.dropna(subset=["mean_acc_ours"]).sort_values("mean_acc_ours", ascending=False)
    if ranked.empty:
        return []
    n_total = len(ranked)
    rerun = millet_rerun_row(cross)
    # Always keep two models on the figure even if accuracy alone would cut them: the MILLET rerun
    # (it is the reference line) and the headline model (it is chosen on four criteria, not accuracy
    # alone, so it can easily sit outside the accuracy top 15 - and a figure that leaves out the
    # model the paper is about is not much use).
    keep = {MILLET_RERUN_ID}
    winner = pick_winner(R.webtraffic_table(), _load_ucr_means())
    if winner:
        keep.add(winner)
    shown = ranked.head(top_n)
    extra = ranked[ranked["model"].isin(keep - set(shown["model"]))]
    if len(extra):
        shown = pd.concat([shown, extra])
    shown = shown.sort_values("mean_acc_ours", ascending=False)

    models = list(shown["model"])
    # mark the headline model in its label, so a reader can find it without counting rows
    names = [_short_name(m) + ("  *" if m == winner else "") for m in models]
    # colour says WHAT KIND of model it is, never its rank: blue = ours end to end, orange = has at
    # least one MILLET part, black = the MILLET rerun itself.
    colours = [OK_BLACK if m == MILLET_RERUN_ID else (OK_BLUE if is_fully_ours(m) else OK_ORANGE)
               for m in models]

    panels = [("acc", "mean accuracy (higher better)", "{:.3f}", False),
              ("loss", "mean loss (lower better)", "{:.3f}", True),
              # the "no published line" warning lives in the title, not floating in the panel, where
              # it would sit on top of a bar
              ("aopcr", "mean AOPCR (higher better)\nno published line - different scale",
               "{:.3f}", False)]
    # NOT sharey: every panel draws the same models in the same order, so the rows line up anyway,
    # and sharing the axis would make the three panels fight over one set of tick labels.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, max(4.0, 0.34 * len(models) + 2.0)))
    for i, (a, (metric, title, fmt, _lower)) in enumerate(zip(axes, panels)):
        values = [float(v) if pd.notna(v) else float("nan")
                  for v in shown[f"mean_{metric}_ours"]]
        ref = float(rerun[f"mean_{metric}_ours"]) if (rerun is not None
              and pd.notna(rerun.get(f"mean_{metric}_ours"))) else None
        # the published line is only meaningful where their metric matches our implementation
        pub = None
        if metric != "aopcr" and pd.notna(shown[f"mean_{metric}_millet"]).any():
            pub = float(shown[f"mean_{metric}_millet"].dropna().iloc[0])
        # the reference lines are explained once in the subtitle, so no legend box is needed inside
        # a panel whose bars already run its full width
        _hbar_panel(a, names, values, colours, title, ref=ref, ref_label=None,
                    target=pub, target_label=None, fmt=fmt, legend=False)
        if i > 0:                                            # names once, on the left panel only
            a.tick_params(labelleft=False)

    fig.suptitle(
        f"Top {top_n} of {n_total} swept models, mean over the 85 datasets MILLET published\n"
        f"bars: blue = ours end to end,  orange = uses a MILLET part,  black = our MILLET rerun"
        + ("   ( * = headline model )" if winner in models else "") + "\n"
        f"lines: black dashed = our MILLET rerun,  green dotted = MILLET published",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
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

    Files: webtraffic_acc.png, webtraffic_aopcr.png, webtraffic_ndcg.png (all under results/top_results/SEA_NET/
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
# Problem this solves: we now have MANY models, so one bar chart of everyone is a
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
    """Turn a long model id ('seanet_x__sea_mstcn_sep__sea_dualstream_conjunctive') into just its config name for a label."""
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
                target=None, target_label=None, fmt="{:.3f}", legend=True):
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
    if legend and ((ref is not None and ref_label) or (target is not None and target_label)):
        ax.legend(fontsize=7, loc="lower right")

    # Leave room on the right for the value printed at each bar end, and for a reference line that
    # sits past the longest bar. Without this the numbers run off the panel or land on top of the
    # dashed line - which is what the old figure did.
    finite = [v for v in list(values) + [ref, target] if v is not None and pd.notna(v)]
    if finite:
        top = max(max(finite), 0.0)
        ax.set_xlim(min(0.0, min(finite)), top * 1.18 if top > 0 else 1.0)


def _millet_rerun_web(df: pd.DataFrame) -> Dict[str, float]:
    """
    Pull our MILLET rerun's WebTraffic numbers out of the WebTraffic table, to use as the reference
    line in the band figures.

    We use the RERUN and not the published table on purpose. The published WebTraffic AOPCR is about
    12.8 while our AOPCR code scores the very same architecture near 1.5, so a published line would
    sit far off the right of every panel and make our models look hopeless for no real reason. The
    rerun was measured with our code, so it shares a scale with every bar next to it.

    df : the WebTraffic table from R.webtraffic_table().
    returns : {"acc": .., "aopcr": .., "ndcg": ..} - only the keys we actually have.
    """
    hit = df[df["model"] == MILLET_RERUN_ID]
    if hit.empty:
        return {}
    r = hit.iloc[0]
    return {k: float(r[k]) for k in ("acc", "aopcr", "ndcg") if k in df.columns and pd.notna(r[k])}


def _plot_one_tier(sub: pd.DataFrame, low: float, high: Optional[float], paper: Dict[str, float],
                   ucr: pd.DataFrame, figdir: str) -> str:
    """
    Draw one accuracy BAND: the models whose WebTraffic accuracy is in [low, high).

    TWO THINGS WERE FIXED HERE.
      1. Every panel used to repeat the model names down its own y-axis. With 6 panels and 25 models
         that is 150 labels for 25 models, and it left almost no room for the bars. The names are now
         printed ONCE, on the leftmost panel; the other panels line up with it row for row.
      2. The AOPCR panel used to draw MILLET's PUBLISHED AOPCR (about 12.8 on WebTraffic) as its
         reference line. Our AOPCR code puts the same model near 1.5, so every real bar was squashed
         into the left edge under a line nothing could reach. That line is gone - the panel now shows
         only our own numbers, which are all on one scale and can honestly be compared to each other.

    sub : the models in this band (sorted best-accuracy-first).
    low / high : the band edges; high is None for the top band.
    paper : MILLET's published WebTraffic numbers.   ucr : the cross-model UCR means (may be empty).
    returns : the saved figure path.
    """
    models = list(sub["model"])
    names = [_short_name(m) for m in models]
    colours = [OK_ORANGE if is_paper_baseline(m) else OK_BLUE for m in models]

    # decide which panels to draw. UCR mean only if at least one model in this band has it.
    ucr_acc = [float(ucr.loc[m, "mean_acc_ours"]) if (len(ucr) and m in ucr.index
               and pd.notna(ucr.loc[m, "mean_acc_ours"])) else float("nan") for m in models]
    have_ucr = any(pd.notna(v) for v in ucr_acc)
    ucr_ref = float(ucr.loc[MILLET_RERUN_ID, "mean_acc_ours"]) if (len(ucr)
              and MILLET_RERUN_ID in ucr.index
              and pd.notna(ucr.loc[MILLET_RERUN_ID, "mean_acc_ours"])) else None

    # (title, values, reference line, ref label, number format). Reference lines are our MILLET
    # RERUN wherever we have it - the same code and recipe, so the numbers share a scale.
    # Short titles on purpose. The full wording ("WebTraffic accuracy (higher better)") was long
    # enough that neighbouring titles printed over each other; the direction now lives in the arrow.
    panels = [
        ("accuracy ↑", list(sub["acc"]), paper.get("acc"), "{:.3f}"),
        ("AOPCR ↑", list(sub["aopcr"]), paper.get("aopcr"), "{:.2f}"),
        ("NDCG ↑", list(sub["ndcg"]), paper.get("ndcg"), "{:.3f}"),
        ("params K ↓", [v / 1e3 for v in sub["params"]], None, "{:.0f}K"),
        ("size MB ↓", list(sub["size_mb"]), None, "{:.2f}"),
    ]
    if have_ucr:
        panels.append(("UCR-85 acc ↑", ucr_acc, ucr_ref, "{:.3f}"))

    n_panels = len(panels)
    # the leftmost panel is wider because it is the only one carrying the model names
    widths = [3.0] + [1.9] * (n_panels - 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(sum(widths) + 1.2,
                                                   max(3.2, 0.42 * len(models) + 2.0)),
                             gridspec_kw={"width_ratios": widths})
    if n_panels == 1:
        axes = [axes]
    for i, (ax, (title, values, ref, fmt)) in enumerate(zip(axes, panels)):
        # the reference line means the same thing in every panel, so it is named once (panel 0) and
        # explained in the subtitle - six copies of the same legend box just ate the plot area
        _hbar_panel(ax, names, values, colours, title, ref=ref,
                    ref_label="MILLET rerun" if i == 0 else None, fmt=fmt)
        if i > 0:                                            # names once, on the left panel only
            ax.tick_params(labelleft=False)

    n = len(models)
    band = f"{low:.2f} - {high:.2f}" if high is not None else f"{low:.2f} and above"
    fig.suptitle(f"WebTraffic accuracy {band}   ({n} model{'s' if n != 1 else ''})\n"
                 f"blue = our encoder, orange = a MILLET encoder, dashed line = our MILLET rerun"
                 f"    ({chr(8593)} higher is better, {chr(8595)} lower is better)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.subplots_adjust(wspace=0.28)
    tag = f"ge{int(round(low * 100))}" if high is None else \
          f"{int(round(low * 100))}to{int(round(high * 100))}"
    return _save(fig, os.path.join(figdir, f"webtraffic_band_{tag}.png"))


def plot_webtraffic_tiers(figdir: str = SHARED_FIGURES_DIR) -> List[str]:
    """
    Draw the WebTraffic accuracy figures as NON-OVERLAPPING bands, so every model appears once.

    WHAT CHANGED AND WHY: the tiers used to be cumulative - ">= 0.95", then ">= 0.94", then ">= 0.93"
    and so on. Each figure therefore contained everything in the figure above it, so the best models
    were redrawn in every single file (the >=0.90 figure repeated all 25 of them). You could not put
    two of those figures in a paper without printing the same bars twice.

    Now the bands are [0.95, ...], [0.94, 0.95), [0.93, 0.94), ... - each model lands in exactly ONE
    figure, chosen by its own accuracy. Anything below the lowest edge goes into a final "below"
    figure so nothing is silently dropped.

    returns : the list of saved figure paths (one per non-empty band).
    """
    df = R.webtraffic_table()                                # already sorted best-accuracy-first
    if df.empty:
        return []
    paper = _millet_rerun_web(df)                            # our rerun, NOT the published table
    ucr = _load_ucr_means()
    paths: List[str] = []
    edges = sorted(ACC_TIERS, reverse=True)                  # e.g. [0.95, 0.94, ..., 0.90]
    high: Optional[float] = None                             # the top band has no upper edge
    for low in edges:
        sub = df[(df["acc"] >= low) & (df["acc"] < high)] if high is not None else df[df["acc"] >= low]
        if not sub.empty:
            paths.append(_plot_one_tier(sub, low, high, paper, ucr, figdir))
        high = low
    rest = df[df["acc"] < edges[-1]]                          # everything under the lowest edge
    if not rest.empty:
        paths.append(_plot_one_tier(rest, 0.0, edges[-1], paper, ucr, figdir))
    return paths


def pick_winner(df: pd.DataFrame, ucr: pd.DataFrame) -> Optional[str]:
    """
    Choose the model to put on the front page of the paper.

    WHY THE OLD RULE WAS WRONG. It scored 0.5 * WebTraffic accuracy + 0.5 * UCR accuracy - accuracy
    only. That picked seanet_conjunctive, which uses OUR encoder but MILLET's OWN Conjunctive pooling
    head. We cannot headline a model whose pooling head is the baseline's; a reviewer would say we
    only changed half the architecture. It also ignored AOPCR, which is the metric our whole
    interpretability claim rests on, and ignored model size, which is our other selling point.

    THE NEW RULE, in four steps:
      1. Only models with a full UCR sweep can win - one WebTraffic screen is not enough evidence.
      2. Prefer models that are OURS END TO END (our encoder AND our pooling). If any exist, models
         reusing a MILLET part are dropped from the contest. They still appear in the tables as
         ablations, they just cannot be the headline.
      3. Rank the survivors on four things that matter equally: UCR-85 accuracy, UCR-85 AOPCR,
         WebTraffic NDCG, and smallness (fewest parameters).
      4. Average the four ranks and take the best. This is a Borda count: it needs no invented
         weights, it cannot be dominated by one metric with a big numeric range (AOPCR moves in
         units, accuracy in hundredths), and it is one sentence to justify in the paper.

    df : the WebTraffic table.   ucr : the cross-model UCR means (may be empty).
    returns : the winning model id, or None if there is nothing to pick from.
    """
    if df.empty:
        return None
    web = df.set_index("model")
    eligible = [m for m in df["model"]
                if len(ucr) and m in ucr.index and pd.notna(ucr.loc[m, "mean_acc_ours"])
                and m != MILLET_RERUN_ID and not is_baseline(m)]
    if not eligible:                                          # no full sweep yet -> WebTraffic only
        return str(df.iloc[0]["model"])

    ours_only = [m for m in eligible if is_fully_ours(m)]
    contest = ours_only or eligible                           # step 2

    # step 3: one column per criterion, all written so that BIGGER = BETTER
    crit = pd.DataFrame(index=contest)
    crit["ucr_acc"] = [float(ucr.loc[m, "mean_acc_ours"]) for m in contest]
    crit["ucr_aopcr"] = [float(ucr.loc[m, "mean_aopcr_ours"])
                         if "mean_aopcr_ours" in ucr.columns
                         and pd.notna(ucr.loc[m, "mean_aopcr_ours"]) else float("nan")
                         for m in contest]
    crit["ndcg"] = [float(web.loc[m, "ndcg"]) if pd.notna(web.loc[m, "ndcg"]) else float("nan")
                    for m in contest]
    crit["small"] = [-float(web.loc[m, "params"]) if pd.notna(web.loc[m, "params"]) else float("nan")
                     for m in contest]                        # negative, so bigger = fewer params

    # step 4: average rank. rank(pct=True) puts every criterion on the same 0-1 scale, so a metric
    # that happens to use bigger numbers cannot shout down the others.
    ranks = crit.rank(pct=True, na_option="bottom")
    return str(ranks.mean(axis=1).idxmax())


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
    rerun_web = _millet_rerun_web(df)                        # our rerun's WebTraffic numbers
    published = R.webtraffic_paper_baseline()                # their published numbers
    ucr = _load_ucr_means()
    winner = pick_winner(df, ucr)
    if winner is None:
        return []
    row = df[df["model"] == winner].iloc[0]
    has_ucr = len(ucr) and winner in ucr.index and pd.notna(ucr.loc[winner, "mean_acc_ours"])
    rerun_ucr = ucr.loc[MILLET_RERUN_ID] if (len(ucr) and MILLET_RERUN_ID in ucr.index) else None

    def _pair(a, ours, theirs, title, fmt, note=None):
        """One small 'ours vs MILLET rerun' panel. Kept local: it is only used here."""
        if note:
            title = f"{title}\n({note})"
        bars = a.bar([0, 1], [ours, theirs], color=[OK_BLUE, OK_BLACK], zorder=3)
        a.set_xticks([0, 1])
        a.set_xticklabels(["ours", "MILLET\nrerun"], fontsize=9)
        a.set_title(title, fontsize=11)
        a.grid(axis="y", alpha=0.25, zorder=0)
        for b, v in zip(bars, [ours, theirs]):
            if pd.notna(v):
                a.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                       ha="center", va="bottom", fontsize=9)
        # a note goes in the TITLE, never inside the axes - the bars fill the panel, so floating
        # text lands on top of them and becomes unreadable

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))

    # Row 1: the three WebTraffic metrics, ours vs our MILLET RERUN.
    # We compare against the rerun, not the published table, because the published AOPCR (about 12.8)
    # was produced by different code from ours (which scores their own model near 1.5). Putting the
    # two on one axis - as this figure used to - makes our model look 8x worse than a baseline it
    # actually beats.
    for a, (col, title, fmt) in zip(ax[0], [("acc", "WebTraffic accuracy", "{:.3f}"),
                                            ("aopcr", "WebTraffic AOPCR", "{:.2f}"),
                                            ("ndcg", "WebTraffic NDCG", "{:.3f}")]):
        ours = float(row[col]) if pd.notna(row[col]) else float("nan")
        note = "published AOPCR uses a different scale" if col == "aopcr" else None
        _pair(a, ours, rerun_web.get(col, float("nan")), title, fmt, note=note)

    # Row 2, panel 1: UCR-85 mean accuracy, ours vs the rerun.
    a = ax[1, 0]
    if has_ucr and rerun_ucr is not None:
        _pair(a, float(ucr.loc[winner, "mean_acc_ours"]),
              float(rerun_ucr["mean_acc_ours"]), "UCR-85 mean accuracy", "{:.3f}")
    elif has_ucr:
        _pair(a, float(ucr.loc[winner, "mean_acc_ours"]), float("nan"),
              "UCR-85 mean accuracy", "{:.3f}", note="MILLET not swept yet")
    else:
        a.axis("off")
        a.text(0.5, 0.5, "UCR-85 mean:\nrun the full sweep\nto fill this in", ha="center",
               va="center", fontsize=10, color="gray")

    # Row 2, panel 2: parameter count, ours vs the rerun. Params and megabytes used to share this
    # axis - 269 next to 3.54 - so the size bar was invisible. Size now lives in the text box.
    a = ax[1, 1]
    params_k = float(row["params"]) / 1e3 if pd.notna(row["params"]) else float("nan")
    their_k = float(rerun_web.get("params", float("nan"))) / 1e3 if "params" in rerun_web else \
        (float(df[df["model"] == MILLET_RERUN_ID]["params"].iloc[0]) / 1e3
         if (df["model"] == MILLET_RERUN_ID).any() else float("nan"))
    _pair(a, params_k, their_k, "parameters, thousands (lower better)", "{:.0f}K")

    # Row 2, panel 3: a plain-words summary box (the headline numbers, spelled out).
    a = ax[1, 2]; a.axis("off")
    size_mb = float(row["size_mb"]) if pd.notna(row["size_mb"]) else float("nan")
    _cfg, w_encoder, w_pooling = split_model_id(winner)
    both_ours = "yes" if is_fully_ours(winner) else "NO - reuses a MILLET part"
    lines = [f"WINNER: {_short_name(winner)}",
             f"encoder: {w_encoder}  ({'MILLET' if is_millet(w_encoder) else 'ours'})",
             f"pooling: {w_pooling}  ({'MILLET' if is_millet(w_pooling) else 'ours'})",
             f"ours end to end: {both_ours}",
             "",
             "                    ours     MILLET rerun"]
    for col, label, fmt in [("acc", "WebTraffic acc  ", "{:>7.3f}"),
                            ("aopcr", "WebTraffic AOPCR", "{:>7.2f}"),
                            ("ndcg", "WebTraffic NDCG ", "{:>7.3f}")]:
        mine = fmt.format(float(row[col])) if pd.notna(row[col]) else "    n/a"
        theirs = fmt.format(rerun_web[col]) if col in rerun_web else "    n/a"
        lines.append(f"{label} {mine}   {theirs}")
    if has_ucr:
        theirs = f"{float(rerun_ucr['mean_acc_ours']):>7.3f}" if rerun_ucr is not None else "    n/a"
        lines.append(f"UCR-85 mean acc  {float(ucr.loc[winner, 'mean_acc_ours']):>7.3f}   {theirs}")
    lines += ["",
              f"params: {int(row['params']):,}" if pd.notna(row["params"]) else "params: n/a",
              f"size  : {size_mb:.2f} MB" if pd.notna(size_mb) else "size  : n/a"]
    if "acc" in published:
        lines += ["", f"(MILLET published WebTraffic acc {published['acc']:.3f};",
                  " their AOPCR is on a different scale, so it is",
                  " not compared here)"]
    a.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")

    fig.suptitle(f"Headline model: {_short_name(winner)}   "
                 f"(best average rank over UCR accuracy, UCR AOPCR, NDCG and size)", fontsize=13)
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
