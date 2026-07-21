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
    fig, ax = plt.subplots(1, 3, figsize=(13, 5))
    for a, (metric, (_csv, _band, lower)) in zip(ax, R.COMPARED_METRICS.items()):
        col = f"mean_{metric}_ours"
        s = cross.dropna(subset=[col]).sort_values(col, ascending=bool(lower))   # best first
        names = list(s["model"]) + ["MILLET"]
        values = list(s[col]) + [float(s[f"mean_{metric}_millet"].iloc[0])]
        colours = [OURS_COLOUR if is_baseline(m) else "darkorange" for m in s["model"]]
        colours.append(MILLET_COLOUR)
        bars = a.bar(range(len(names)), values, color=colours)
        a.set_xticks(range(len(names)))
        a.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
        a.set_title(f"mean {metric}" + ("  (lower is better)" if lower else ""))
        for b, v in zip(bars, values):
            if pd.notna(v):
                a.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    n = int(cross["millet85_datasets"].max())
    fig.suptitle(f"Model comparison over the {n} datasets MILLET published   "
                 f"[orange = our new pooling heads, red = MILLET]")
    fig.tight_layout()
    return [_save(fig, os.path.join(figdir, "model_comparison.png"))]


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

    if verbose:
        print("\n=== which model wins? ===")
        R._print_model_comparison(cross, R.MODEL_COMPARISON_CSV)
        print(f"\nwrote {len(figures)} figures:")
        for p in figures:
            print("  ", p)
    return {"models": cross.to_dict("records") if len(cross) else [], "figures": figures}
