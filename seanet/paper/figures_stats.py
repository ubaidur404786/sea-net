"""
seanet/paper/figures_stats.py - the "is the difference real?" figures.

These are the figures a reviewer looks for before believing any table of means. They all need the
full per-dataset matrix (model x dataset), so they cover only the models that finished the complete
sweep - every caption says how many models and how many datasets, so the scope is never hidden.

    1. critical difference diagram : which models are statistically indistinguishable
    2. average rank                : who is best overall, robust to one easy dataset
    3. win / tie / loss            : how often we beat the published baseline, dataset by dataset
    4. improvement over baseline   : by how much, on average
    5. pairwise win matrix         : who beats whom, head to head
    6. dataset x model heatmap     : where the wins actually come from
    7. metric correlation          : do our metrics measure different things?
    8. accuracy distribution       : the spread over datasets, not just the mean
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from seanet.paper import stats as PS
from seanet.paper import style as S


def _labels(models: List[str]) -> List[str]:
    """Short, readable names for a figure axis."""
    return [S.shorten(m, 20) for m in models]


# ======================================================================================
# 1. The critical difference diagram - the single most expected figure in a TSC paper
# ======================================================================================
def critical_difference(mat: pd.DataFrame, higher_better: bool = True,
                        alpha: float = 0.05, metric_name: str = "accuracy") -> Optional[Dict]:
    """
    The critical difference diagram (Demsar 2006, drawn the Ismail Fawaz way).

    HOW TO READ IT: models sit on a horizontal axis of average rank, best on the LEFT. A thick
    horizontal bar joins any group of models whose differences are NOT statistically significant.
    Two models sharing a bar cannot be claimed to differ - no matter what their means say.

    HOW IT IS BUILT: rank the models on every dataset, average the ranks, then run a Wilcoxon
    signed-rank test on every pair and correct all those p-values with Holm's step-down method (see
    seanet/paper/stats.py). The bars are the maximal groups with no significant difference inside.

    mat : the model x dataset matrix, already reduced to the datasets every model shares.
    higher_better : True for accuracy / AOPCR, False for loss.
    alpha : the significance level for the whole family of tests.
    returns : the metadata entry, or None if there is not enough data (or SciPy is missing).
    """
    if mat.empty or len(mat) < 3:
        return None
    ranks = PS.average_ranks(mat, higher_better)
    order = list(ranks.index)                                # best (lowest rank) first
    pairs = PS.wilcoxon_holm(mat, higher_better, alpha)
    bars = PS.cliques(order, pairs, alpha) if not pairs.empty else []

    n_models, n_datasets = len(order), mat.shape[1]
    half = int(np.ceil(n_models / 2))                        # best half left, worst half right

    # --- the drawing area, in DATA coordinates -------------------------------------------
    # The axis runs left (best rank) to right (worst rank). Model names sit in a margin outside
    # the rank range on each side, so a name can never land on top of the axis or a bar.
    lo, hi = float(np.floor(ranks.min())), float(np.ceil(ranks.max()))
    span = max(hi - lo, 1.0)
    margin = span * 0.55                                     # room for the names, in rank units
    x_left, x_right = lo - margin, hi + margin

    # y is laid out downwards from the axis: first the clique bars, then one row per model.
    y_axis = 0.0
    bar_gap = 0.55
    row_gap = 1.0
    y_first_row = -(len(bars) + 1) * bar_gap - 0.6
    y_bottom = y_first_row - (half - 1) * row_gap - 0.9

    fig_h = 1.05 + 0.20 * half + 0.11 * len(bars)
    fig, ax = plt.subplots(figsize=(S.FULL_WIDTH, fig_h))
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_axis + 1.25)
    ax.axis("off")
    ax.grid(False)                                           # the house style turns grids on; not here

    # --- the rank axis ---
    ax.plot([lo, hi], [y_axis, y_axis], color="black", lw=1.0, zorder=3)
    for tick in np.arange(np.ceil(lo), np.floor(hi) + 1):
        ax.plot([tick, tick], [y_axis, y_axis + 0.18], color="black", lw=0.9, zorder=3)
        ax.text(tick, y_axis + 0.28, f"{int(tick)}", ha="center", va="bottom", fontsize=7.5)
    ax.text((lo + hi) / 2, y_axis + 0.72, "average rank (lower is better)",
            ha="center", va="bottom", fontsize=7.5, color="#555555")

    # --- one connector per model: down from its rank, then out to its name ---
    for i, model in enumerate(order):
        rank = float(ranks[model])
        left_side = i < half
        row = i if left_side else i - half
        y = y_first_row - row * row_gap
        x_end = x_left if left_side else x_right
        ax.plot([rank, rank], [y_axis, y], color=S.C_NEUTRAL, lw=0.8, zorder=2)
        ax.plot([rank, x_end], [y, y], color=S.C_NEUTRAL, lw=0.8, zorder=2)
        parts = str(model).split("__")
        ours = not parts[1].startswith("mil_") if len(parts) > 1 else True
        if str(model).startswith("MILLET"):
            ours = False
        ax.text(x_end, y + 0.14, f"{_labels([model])[0]} ({rank:.2f})",
                ha="left" if left_side else "right", va="bottom", fontsize=7,
                color=S.C_OURS if ours else S.C_BASELINE)

    # --- the "not significantly different" bars, stacked just under the axis ---
    for k, (i, j) in enumerate(bars):
        y = y_axis - (k + 1) * bar_gap
        ax.plot([ranks[order[i]], ranks[order[j]]], [y, y],
                color=S.C_MILLET, lw=3.0, solid_capstyle="round", zorder=4)

    note = "" if not pairs.empty else "  (SciPy missing: significance bars omitted)"
    ax.set_title(f"Critical difference diagram - {metric_name} over {n_datasets} datasets\n"
                 f"{n_models} models, Wilcoxon signed-rank with Holm correction "
                 f"($\\alpha$={alpha}){note}", fontsize=8.5)

    return S.save(
        fig, "main", f"fig4_critical_difference_{metric_name}",
        title=f"Critical difference diagram ({metric_name})",
        caption=(f"Critical difference diagram over {n_datasets} datasets for the {n_models} models "
                 f"with a complete sweep. Models are placed by average rank ({metric_name}), best on "
                 f"the left. A thick horizontal bar joins models whose pairwise differences are not "
                 f"statistically significant under a Wilcoxon signed-rank test with Holm correction "
                 f"at alpha={alpha}; models sharing a bar cannot be separated by this evidence. "
                 f"Blue names are models proposed in this work."),
        question=f"Which differences in {metric_name} are statistically significant?")


# ======================================================================================
# 2. Average rank
# ======================================================================================
def average_rank(mat: pd.DataFrame, higher_better: bool = True,
                 metric_name: str = "accuracy") -> Optional[Dict]:
    """
    Every model's average rank, best first.

    Why rank and not mean accuracy: a mean can be dragged up by one dataset where a model happens to
    do unusually well. A rank cannot - being best on an easy dataset earns exactly the same rank 1 as
    being best on a hard one. So average rank is the fairer "who is best overall" summary, and it is
    what the critical difference diagram is built on.
    """
    if mat.empty or len(mat) < 2:
        return None
    ranks = PS.average_ranks(mat, higher_better)
    models = list(ranks.index)
    ours = [not str(m).split("__")[1].startswith("mil_") if "__" in str(m) else True for m in models]
    colours = [S.C_OURS if o else S.C_BASELINE for o in ours]

    fig, ax = plt.subplots(figsize=(S.COL_WIDTH * 1.55, 0.24 * len(models) + 0.9))
    y = np.arange(len(models))
    bars = ax.barh(y, ranks.to_numpy(float), color=colours, height=0.7, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(_labels(models))
    ax.invert_yaxis()
    ax.set_xlabel(f"average rank over {mat.shape[1]} datasets (1 = best)")
    ax.grid(axis="x", alpha=0.35)
    ax.grid(axis="y", visible=False)
    S.annotate_bars(ax, bars, ranks.to_numpy(float), fmt="{:.2f}")
    ax.set_title(f"Average rank ({metric_name})")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=S.C_OURS, ec="none"),
               plt.Rectangle((0, 0), 1, 1, fc=S.C_BASELINE, ec="none")]
    ax.legend(handles, ["Ours", "Reproduced baseline"], loc="lower right")
    return S.save(
        fig, "stats", f"stats_average_rank_{metric_name}",
        title=f"Average rank ({metric_name})",
        caption=(f"Average {metric_name} rank of each fully-swept model over {mat.shape[1]} datasets "
                 f"(rank 1 is best on a dataset). Ranking is robust to a single easy dataset in a "
                 f"way that a mean is not."),
        question=f"Which model is best overall on {metric_name}, robust to individual datasets?")


# ======================================================================================
# 3 + 4. Against the published baseline: how often, and by how much
# ======================================================================================
def win_tie_loss(mat: pd.DataFrame, baseline: pd.Series, metric_name: str = "accuracy",
                 higher_better: bool = True, section: str = "main") -> Optional[Dict]:
    """
    Win / tie / loss against the MILLET paper, one stacked bar per model.

    A difference smaller than 0.005 accuracy counts as a TIE, not a win: on a test set of a few
    hundred series that gap is noise, and calling it a win would overstate the result. The same band
    is used by the project's existing tables, so this figure agrees with them.
    """
    if mat.empty or baseline.empty:
        return None
    wtl = PS.win_tie_loss(mat, baseline, higher_better=higher_better)
    if wtl.empty:
        return None
    wtl = wtl.sort_values("win", ascending=False).reset_index(drop=True)

    models = list(wtl["model"])
    y = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(S.COL_WIDTH * 1.6, 0.24 * len(models) + 1.0))
    # a stacked bar: wins, then ties starting where wins ended, then losses on top of both
    ax.barh(y, wtl["win"], color=S.C_GOOD, height=0.7, label="win", zorder=3)
    ax.barh(y, wtl["tie"], left=wtl["win"], color=S.C_NEUTRAL, height=0.7, label="tie", zorder=3)
    ax.barh(y, wtl["loss"], left=wtl["win"] + wtl["tie"], color=S.C_BAD, height=0.7,
            label="loss", zorder=3)
    for i, r in wtl.iterrows():
        if r["win"]:
            ax.text(r["win"] / 2, i, int(r["win"]), ha="center", va="center",
                    fontsize=6.5, color="white")
        if r["loss"]:
            ax.text(r["win"] + r["tie"] + r["loss"] / 2, i, int(r["loss"]), ha="center",
                    va="center", fontsize=6.5, color="white")
    ax.set_yticks(y)
    ax.set_yticklabels(_labels(models))
    ax.invert_yaxis()
    ax.set_xlabel(f"datasets (out of {int(wtl['n_datasets'].max())})")
    ax.grid(axis="x", alpha=0.35)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Win / tie / loss vs MILLET ({metric_name})")
    ax.legend(loc="lower right", ncol=3, fontsize=7)
    return S.save(
        fig, section, f"fig5_win_tie_loss_{metric_name}",
        title=f"Win / tie / loss against MILLET ({metric_name})",
        caption=(f"Per-dataset {metric_name} record of each fully-swept model against the published "
                 f"MILLET results, over the {int(wtl['n_datasets'].max())} datasets both report. "
                 f"Differences within 0.005 count as ties, since that is below the resolution of "
                 f"these test sets. Sorted by number of wins."),
        question=f"How often does each model actually beat the published baseline on {metric_name}?")


def improvement_over_baseline(mat: pd.DataFrame, baseline: pd.Series,
                              metric_name: str = "accuracy") -> Optional[Dict]:
    """
    The average per-dataset gap to the published baseline - by how much, not just how often.

    Win/tie/loss counts how often we win; this says by how much. Both are needed: many small wins and
    a few large losses can look good in a count and bad here, and that combination is exactly what a
    reviewer will probe.
    """
    if mat.empty or baseline.empty:
        return None
    gaps = PS.improvement_over(mat, baseline)
    if gaps.empty:
        return None
    models = list(gaps.index)
    colours = [S.C_GOOD if v > 0 else S.C_BAD for v in gaps]

    fig, ax = plt.subplots(figsize=(S.COL_WIDTH * 1.55, 0.24 * len(models) + 0.9))
    y = np.arange(len(models))
    bars = ax.barh(y, gaps.to_numpy(float), color=colours, height=0.7, zorder=3)
    ax.axvline(0, color=S.C_MILLET, lw=1.0)                  # zero = exactly the baseline
    ax.set_yticks(y)
    ax.set_yticklabels(_labels(models))
    ax.invert_yaxis()
    ax.set_xlabel(f"mean {metric_name} improvement over MILLET (positive = better)")
    ax.grid(axis="x", alpha=0.35)
    ax.grid(axis="y", visible=False)
    S.annotate_bars(ax, bars, gaps.to_numpy(float), fmt="{:+.4f}")
    ax.set_title(f"Average improvement over MILLET ({metric_name})")
    return S.save(
        fig, "stats", f"stats_improvement_over_millet_{metric_name}",
        title=f"Average improvement over MILLET ({metric_name})",
        caption=(f"Mean per-dataset {metric_name} difference against the published MILLET results. "
                 f"Green bars improve on the baseline, red bars fall short. Read together with the "
                 f"win/tie/loss figure: this shows the size of the gap, that one shows how often it "
                 f"goes each way."),
        question=f"By how much does each model improve on the baseline in {metric_name}?")


# ======================================================================================
# 5 + 6. Heatmaps: who beats whom, and where the wins come from
# ======================================================================================
def pairwise_win_matrix(mat: pd.DataFrame, higher_better: bool = True,
                        metric_name: str = "accuracy") -> Optional[Dict]:
    """
    A square heatmap of head-to-head wins: cell (row, column) = datasets the row model wins.

    This answers something the ranking cannot: a model can have a mediocre average rank yet still
    beat one specific rival almost every time. That pattern is visible here and nowhere else.
    """
    if mat.empty or len(mat) < 2:
        return None
    order = list(PS.average_ranks(mat, higher_better).index)
    wins = PS.pairwise_wins(mat.loc[order], higher_better).loc[order, order]

    n = len(order)
    fig, ax = plt.subplots(figsize=(S.FULL_WIDTH * 0.9, S.FULL_WIDTH * 0.9))
    im = ax.imshow(wins.to_numpy(float), cmap="RdYlGn", vmin=0, vmax=mat.shape[1])
    ax.set_xticks(range(n)); ax.set_xticklabels(_labels(order), rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(_labels(order), fontsize=6)
    ax.grid(visible=False)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, int(wins.iat[i, j]), ha="center", va="center", fontsize=5.2)
    fig.colorbar(im, ax=ax, shrink=0.75, label=f"datasets won (of {mat.shape[1]})")
    ax.set_title(f"Head-to-head wins ({metric_name})\nrow beats column on this many datasets",
                 fontsize=9)
    return S.save(
        fig, "stats", f"stats_pairwise_wins_{metric_name}",
        title=f"Pairwise win matrix ({metric_name})",
        caption=(f"Head-to-head {metric_name} record over {mat.shape[1]} datasets: the cell in row "
                 f"i and column j counts the datasets on which model i beats model j. Models are "
                 f"ordered by average rank, so a well-ordered matrix is green above the diagonal."),
        question="Which models beat which other models directly, rather than on average?")


def dataset_model_heatmap(mat: pd.DataFrame, baseline: Optional[pd.Series] = None,
                          metric_name: str = "accuracy") -> Optional[Dict]:
    """
    The full model x dataset picture, as a heatmap of the gap to the baseline.

    We plot the DIFFERENCE from MILLET rather than raw accuracy on purpose. Raw accuracy is dominated
    by how hard each dataset is - the map would just show easy and hard columns and tell you nothing
    about the models. The difference removes that, so what is left is exactly what we want to see:
    where each model gains and where it loses.
    """
    if mat.empty:
        return None
    if baseline is not None and not baseline.empty:
        shared = [d for d in mat.columns if d in baseline.index]
        grid = mat[shared].astype(float).sub(baseline[shared].astype(float), axis=1)
        label = f"{metric_name} minus MILLET (positive = better)"
        cmap, centre = "RdBu_r", True
    else:
        grid = mat.astype(float)
        label, cmap, centre = metric_name, "viridis", False
    if grid.empty:
        return None

    # rows ordered by mean gap (best model at the top), columns by how hard the dataset was for us
    grid = grid.loc[grid.mean(axis=1).sort_values(ascending=False).index]
    grid = grid[grid.mean(axis=0).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(S.FULL_WIDTH, 0.22 * len(grid) + 1.8))
    if centre:
        # a diverging colour map must be centred on zero, or "no difference" would not be white
        limit = float(np.nanpercentile(np.abs(grid.to_numpy(float)), 98))
        im = ax.imshow(grid.to_numpy(float), cmap=cmap, aspect="auto", vmin=-limit, vmax=limit)
    else:
        im = ax.imshow(grid.to_numpy(float), cmap=cmap, aspect="auto")
    ax.set_yticks(range(len(grid)))
    ax.set_yticklabels(_labels(list(grid.index)), fontsize=6.5)
    ax.set_xticks([])                                        # 85 dataset names never fit; see caption
    ax.set_xlabel(f"{grid.shape[1]} UCR datasets (sorted by mean gap across models)")
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, shrink=0.85, label=label)
    ax.set_title(f"Per-dataset {metric_name} against MILLET")
    return S.save(
        fig, "appendix", f"appendix_dataset_model_heatmap_{metric_name}",
        title=f"Dataset x model heatmap ({metric_name})",
        caption=(f"Per-dataset {metric_name} difference from the published MILLET results for every "
                 f"fully-swept model ({grid.shape[0]} models x {grid.shape[1]} datasets). Blue is "
                 f"better than the baseline, red is worse, white is equal. Datasets are ordered by "
                 f"mean difference, so systematic strengths and weaknesses appear as vertical bands. "
                 f"Dataset names are omitted for space; the ordering is in the accompanying CSV."),
        question="Where do the gains and losses against the baseline actually come from?")


def metric_correlation(lb: pd.DataFrame) -> Optional[Dict]:
    """
    How much our metrics agree with each other, as a Spearman correlation heatmap.

    If accuracy and NDCG correlated almost perfectly, reporting both would be padding. If they do
    not, then interpretability really is a separate axis and the paper needs both - which is a
    result in itself. Either way the reader should be able to check it.
    """
    columns = [c for c in ["web_acc", "web_aopcr", "web_ndcg", "web_loss",
                           "ucr85_acc", "params", "size_mb", "flops_m", "infer_ms"]
               if c in lb.columns and lb[c].notna().sum() >= 5]
    if len(columns) < 3:
        return None
    corr = PS.metric_correlation(lb, columns)
    nice = [c.replace("web_", "").replace("ucr85_", "UCR ").replace("_m", "")
             .replace("_mb", "").replace("_ms", "") for c in columns]

    fig, ax = plt.subplots(figsize=(S.COL_WIDTH * 1.5, S.COL_WIDTH * 1.4))
    im = ax.imshow(corr.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(nice))); ax.set_xticklabels(nice, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(nice))); ax.set_yticklabels(nice, fontsize=7)
    ax.grid(visible=False)
    for i in range(len(nice)):
        for j in range(len(nice)):
            v = corr.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman $\\rho$")
    ax.set_title("Do our metrics measure different things?", fontsize=9)
    return S.save(
        fig, "stats", "stats_metric_correlation",
        title="Metric correlation",
        caption=("Spearman rank correlation between every reported metric across all evaluated "
                 "models. Rank correlation is used because what matters is whether two metrics order "
                 "the models the same way. Values near zero identify genuinely independent axes, "
                 "which is why they are reported separately rather than combined into one score."),
        question="Are our metrics measuring different things, or repeating each other?")


def accuracy_distribution(mat: pd.DataFrame, metric_name: str = "accuracy") -> Optional[Dict]:
    """
    The spread of each model's per-dataset scores, as box plots - not just the mean.

    A mean hides everything about consistency. Two models can share a mean while one is steady across
    every dataset and the other swings wildly; the steady one is the better model and only a
    distribution shows it.

    Note on what this is NOT: these boxes show variation ACROSS DATASETS, not across repeated runs.
    Every model here was trained once (a single seed), so this figure says nothing about run-to-run
    variance - the caption states that explicitly so no reader mistakes one for the other.
    """
    if mat.empty or len(mat) < 2:
        return None
    order = list(PS.average_ranks(mat, higher_better=True).index)
    data = [mat.loc[m].dropna().to_numpy(float) for m in order]

    fig, ax = plt.subplots(figsize=(S.COL_WIDTH * 1.6, 0.26 * len(order) + 1.0))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.62,
                    medianprops=dict(color=S.C_MILLET, lw=1.2),
                    flierprops=dict(marker="o", ms=2.2, mfc=S.C_NEUTRAL, mec="none"))
    for patch in bp["boxes"]:
        patch.set_facecolor(S.C_OURS); patch.set_alpha(0.35); patch.set_edgecolor(S.C_OURS)
    ax.set_yticklabels(_labels(order), fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"per-dataset {metric_name} over {mat.shape[1]} datasets")
    ax.grid(axis="x", alpha=0.35); ax.grid(axis="y", visible=False)
    ax.set_title(f"Distribution of {metric_name} across datasets")
    return S.save(
        fig, "appendix", f"appendix_{metric_name}_distribution",
        title=f"Per-dataset {metric_name} distribution",
        caption=(f"Distribution of {metric_name} across {mat.shape[1]} datasets for each fully-swept "
                 f"model, ordered by average rank. The box spans the interquartile range and the "
                 f"line is the median, so consistency is visible alongside central tendency. Note "
                 f"that this shows variation across DATASETS: each model was trained once, so these "
                 f"boxes do not represent run-to-run variance."),
        question=f"How consistent is each model across datasets, not just on average?")
