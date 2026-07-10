"""
seanet/report.py - collecting the experiment outputs into paper-ready tables and figures.

What this file is for:
    After a sweep finishes, we want the results in a form that goes straight into a report or paper:
    a summary table, the SEA-Net vs MILLET comparison, and a set of figures. This module builds all
    of that from the CSV files SEA-Net writes. It used to live inside analysis.ipynb; moving it here
    means it is reusable code (the notebook AND `python main.py report` both call it), so there is
    one place that draws the figures, not two.

    It reads (never writes) the result CSVs, and produces:
      - figures  -> results/SEA_NET/figures/*.png (data summary, results, and the MILLET comparison),
      - a summary table -> results/SEA_NET/summary.csv and summary.md (paper-ready headline numbers),
      - the per-dataset comparison -> results/SEA_NET/comparison_vs_millet.csv (via seanet.results).

Input:
    The three CSVs under results/SEA_NET/ (data_summary.csv, results.csv, and the MILLET baseline).
Output:
    PNG figures + a summary table; the functions also return the values so the notebook can show them.

Related files:
    - seanet/data.py    -> read_our_csv() (tolerant CSV reader) and DATA_SUMMARY_CSV.
    - seanet/results.py -> load_results() and build_comparison() (the data this module plots).
    - main.py ("report" command) -> calls generate_report().
    - analysis.ipynb    -> a thin notebook that calls generate_report() and shows the saved figures.
"""
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")                                        # headless: we save PNGs (works on Grid5000)
import matplotlib.pyplot as plt                              # noqa: E402
import pandas as pd

from seanet import data as D
from seanet import results as R

FIGURES_DIR = os.path.join("results", "SEA_NET", "figures")
SUMMARY_CSV = os.path.join("results", "SEA_NET", "summary.csv")
SUMMARY_MD = os.path.join("results", "SEA_NET", "summary.md")


def _to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Make sure the given columns are numbers (our CSVs can have stray spaces)."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_frames() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three data frames every report is built from.

    returns : (data_summary, results, comparison_vs_millet). Any of them can be empty if that CSV
              has not been produced yet.
    """
    if os.path.exists(D.DATA_SUMMARY_CSV):
        summary = _to_num(D.read_our_csv(D.DATA_SUMMARY_CSV),
                          ["n_train", "n_test", "series_length", "n_classes", "imbalance_ratio"])
    else:
        summary = pd.DataFrame()
    results = _to_num(R.load_results(),
                      ["test_acc", "test_aopcr", "test_ndcg", "series_length", "n_classes",
                       "params", "model_size_mb", "train_time_s"])
    comparison = _to_num(R.build_comparison(verbose=False),      # also refreshes comparison_vs_millet.csv
                         ["ours_acc", "millet_acc", "acc_diff", "ours_aopcr", "millet_aopcr", "aopcr_diff"])
    return summary, results, comparison


# --------------------------------------------------------------------------------------
# Figures (each one saves a PNG and returns its path)
# --------------------------------------------------------------------------------------
def plot_data_summary(summary: pd.DataFrame, figdir: str = FIGURES_DIR) -> str:
    """Draw the dataset overview (series length, #classes, train size, adjusted-folder count)."""
    os.makedirs(figdir, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].hist(summary["series_length"].dropna(), bins=30, color="steelblue")
    ax[0, 0].set_title("Series length (T)"); ax[0, 0].set_xlabel("T"); ax[0, 0].set_ylabel("# datasets")
    ax[0, 1].hist(summary["n_classes"].dropna(), bins=30, color="seagreen")
    ax[0, 1].set_title("Number of classes"); ax[0, 1].set_xlabel("classes")
    ax[1, 0].hist(summary["n_train"].dropna(), bins=30, color="indianred")
    ax[1, 0].set_title("Train set size"); ax[1, 0].set_xlabel("# train series"); ax[1, 0].set_ylabel("# datasets")
    counts = summary["used_adjusted_folder"].astype(str).value_counts()
    ax[1, 1].bar(counts.index, counts.values, color=["gray", "orange"][:len(counts)])
    ax[1, 1].set_title("Used the adjusted folder?"); ax[1, 1].set_ylabel("# datasets")
    fig.suptitle("Dataset summary"); fig.tight_layout()
    path = os.path.join(figdir, "data_summary.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_results(results: pd.DataFrame, figdir: str = FIGURES_DIR) -> str:
    """Draw SEA-Net's own results (accuracy + AOPCR distributions, accuracy vs length / #classes)."""
    os.makedirs(figdir, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].hist(results["test_acc"].dropna(), bins=20, color="steelblue")
    ax[0, 0].set_title("Test accuracy"); ax[0, 0].set_xlabel("accuracy"); ax[0, 0].set_ylabel("# datasets")
    ax[0, 1].hist(results["test_aopcr"].dropna(), bins=20, color="seagreen")
    ax[0, 1].set_title("AOPCR (interpretability)"); ax[0, 1].set_xlabel("AOPCR")
    ax[1, 0].scatter(results["series_length"], results["test_acc"], s=18, alpha=0.7)
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_title("Accuracy vs series length"); ax[1, 0].set_xlabel("T (log scale)"); ax[1, 0].set_ylabel("accuracy")
    ax[1, 1].scatter(results["n_classes"], results["test_acc"], s=18, alpha=0.7, color="purple")
    ax[1, 1].set_title("Accuracy vs number of classes"); ax[1, 1].set_xlabel("classes"); ax[1, 1].set_ylabel("accuracy")
    fig.suptitle("SEA-Net results"); fig.tight_layout()
    path = os.path.join(figdir, "results.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_comparison(comparison: pd.DataFrame, figdir: str = FIGURES_DIR) -> List[str]:
    """
    Draw the SEA-Net vs MILLET figures (accuracy scatter, win/tie/loss bars, AOPCR scatter, gap bar).

    In the scatter plots a point ABOVE the red y=x line means SEA-Net wins on that dataset.
    Only the datasets that have a MILLET baseline are plotted.
    """
    os.makedirs(figdir, exist_ok=True)
    overlap = comparison[comparison["acc_outcome"] != "no_baseline"].copy()
    paths: List[str] = []
    if len(overlap) == 0:
        return paths

    # accuracy scatter (above the line = SEA-Net wins)
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(overlap["millet_acc"], overlap["ours_acc"], s=25, alpha=0.7)
    plt.plot([0, 1], [0, 1], "r--", lw=1)
    plt.xlabel("MILLET accuracy"); plt.ylabel("SEA-Net accuracy"); plt.title("Accuracy: SEA-Net vs MILLET")
    plt.xlim(0, 1.02); plt.ylim(0, 1.02); plt.grid(alpha=0.3); plt.tight_layout()
    p = os.path.join(figdir, "acc_scatter.png"); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # win / tie / loss bars, for accuracy and AOPCR
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, col, title in [(ax[0], "acc_outcome", "Accuracy"), (ax[1], "aopcr_outcome", "AOPCR")]:
        vc = overlap[col].value_counts().reindex(["win", "tie", "loss"]).fillna(0)
        a.bar(vc.index, vc.values, color=["green", "gray", "red"])
        a.set_title(title + ": SEA-Net vs MILLET"); a.set_ylabel("# datasets")
    fig.tight_layout()
    p = os.path.join(figdir, "win_tie_loss.png"); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # AOPCR scatter
    fig = plt.figure(figsize=(6, 6))
    lim = max(overlap["millet_aopcr"].max(), overlap["ours_aopcr"].max()) * 1.05
    plt.scatter(overlap["millet_aopcr"], overlap["ours_aopcr"], s=25, alpha=0.7, color="seagreen")
    plt.plot([0, lim], [0, lim], "r--", lw=1)
    plt.xlabel("MILLET AOPCR"); plt.ylabel("SEA-Net AOPCR"); plt.title("AOPCR: SEA-Net vs MILLET")
    plt.grid(alpha=0.3); plt.tight_layout()
    p = os.path.join(figdir, "aopcr_scatter.png"); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # per-dataset accuracy gap (green = we win), sorted
    d = overlap.sort_values("acc_diff")
    fig = plt.figure(figsize=(7, max(3, 0.28 * len(d))))
    plt.barh(d["dataset"], d["acc_diff"], color=["green" if v > 0 else "red" for v in d["acc_diff"]])
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("accuracy difference (SEA-Net - MILLET)")
    plt.title("Per-dataset accuracy gap (green = we win)"); plt.tight_layout()
    p = os.path.join(figdir, "acc_diff.png"); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)
    return paths


# --------------------------------------------------------------------------------------
# Tables / summaries
# --------------------------------------------------------------------------------------
def performance_summary(results: pd.DataFrame, comparison: pd.DataFrame) -> Dict:
    """
    Build the headline numbers (a small dict): how many datasets, mean accuracy / AOPCR, WebTraffic
    NDCG, model footprint, and the win/tie/loss record vs MILLET. This is the paper-ready summary.

    results : the SEA-Net results frame.
    comparison : the SEA-Net vs MILLET comparison frame.
    returns : an ordered dict of metric name -> value.
    """
    out: Dict = {"datasets_with_results": int(len(results))}
    if len(results):
        out["mean_test_accuracy"] = round(float(results["test_acc"].mean()), 4)
        out["mean_aopcr"] = round(float(results["test_aopcr"].mean()), 4)
        web = results[results["dataset"] == "WebTraffic"]
        if len(web) and pd.notna(web["test_ndcg"].iloc[0]):
            out["webtraffic_ndcg"] = round(float(web["test_ndcg"].iloc[0]), 4)
        out["params"] = int(results["params"].iloc[0])
        out["model_size_mb"] = round(float(results["model_size_mb"].iloc[0]), 3)
    if comparison is not None and len(comparison):
        overlap = comparison[comparison["acc_outcome"] != "no_baseline"]
        out["millet_overlap_datasets"] = int(len(overlap))
        if len(overlap):
            av = overlap["acc_outcome"].value_counts()
            pv = overlap["aopcr_outcome"].value_counts()
            out["acc_win_tie_loss"] = f"{av.get('win', 0)}/{av.get('tie', 0)}/{av.get('loss', 0)}"
            out["aopcr_win_tie_loss"] = f"{pv.get('win', 0)}/{pv.get('tie', 0)}/{pv.get('loss', 0)}"
            out["acc_mean_ours_vs_millet"] = f"{overlap['ours_acc'].mean():.4f} / {overlap['millet_acc'].mean():.4f}"
            out["aopcr_mean_ours_vs_millet"] = f"{overlap['ours_aopcr'].mean():.3f} / {overlap['millet_aopcr'].mean():.3f}"
    return out


def write_summary_table(perf: Dict, out_csv: str = SUMMARY_CSV, out_md: str = SUMMARY_MD) -> Tuple[str, str]:
    """
    Write the headline summary as a CSV and a Markdown table (both paper-ready).

    perf : the dict from performance_summary.
    out_csv, out_md : where to write.
    returns : (csv path, md path).
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame([{"metric": k, "value": v} for k, v in perf.items()]).to_csv(out_csv, index=False)
    with open(out_md, "w") as f:
        f.write("# SEA-Net results summary\n\n| metric | value |\n|---|---|\n")
        for key, value in perf.items():
            f.write(f"| {key} | {value} |\n")
    return out_csv, out_md


def generate_report(figdir: str = FIGURES_DIR, verbose: bool = True) -> Dict:
    """
    Do everything: load the CSVs, draw all figures, and write the summary table.

    figdir : where to save the PNG figures.
    verbose : if True, print the headline summary and where things were written.
    returns : a dict with "summary" (the headline numbers) and "figures" (list of PNG paths).
    """
    summary, results, comparison = load_frames()
    figures: List[str] = []
    if len(summary):
        figures.append(plot_data_summary(summary, figdir))
    if len(results):
        figures.append(plot_results(results, figdir))
    if len(comparison):
        figures += plot_comparison(comparison, figdir)

    perf = performance_summary(results, comparison)
    csv_path, md_path = write_summary_table(perf)

    if verbose:
        print("=== SEA-Net report ===")
        for key, value in perf.items():
            print(f"  {key:26s}: {value}")
        print(f"\nwrote {len(figures)} figures -> {figdir}/")
        for p in figures:
            print("  ", p)
        print(f"wrote summary table -> {csv_path} and {md_path}")
    return {"summary": perf, "figures": figures}
