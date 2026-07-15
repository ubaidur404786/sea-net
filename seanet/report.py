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
      - the per-dataset comparison -> results/SEA_NET/comparison_vs_millet.csv (via seanet.results),
      - the pooling-head comparison figures -> from results/SEA_NET/pooling_benchmark.csv, IF that file
        exists (i.e. you have run `python main.py benchmark`); skipped quietly otherwise.

Input:
    The CSVs under results/SEA_NET/ (data_summary.csv, results.csv, the MILLET baseline, and the
    optional pooling_benchmark.csv).
Output:
    PNG figures + a summary table; the functions also return the values so the notebook can show them.

Related files:
    - seanet/data.py      -> read_our_csv() (tolerant CSV reader) and DATA_SUMMARY_CSV.
    - seanet/results.py   -> load_results() and build_comparison() (the data this module plots).
    - seanet/benchmark.py -> writes pooling_benchmark.csv (the pooling-head comparison this module plots).
    - main.py ("report" command) -> calls generate_report().
    - analysis.ipynb      -> a thin notebook that calls generate_report() and shows the saved figures.
"""
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                                        # headless: we save PNGs (works on Grid5000)
import matplotlib.pyplot as plt                              # noqa: E402
import pandas as pd

from seanet import data as D
from seanet import results as R

FIGURES_DIR = os.path.join("results", "SEA_NET", "figures")
SUMMARY_CSV = os.path.join("results", "SEA_NET", "summary.csv")
SUMMARY_MD = os.path.join("results", "SEA_NET", "summary.md")
# where the pooling benchmark writes its results (mirrors seanet/benchmark.BENCHMARK_CSV; defined here
# so report.py stays free of the heavier benchmark/train imports - it only needs the path string).
BENCHMARK_CSV = os.path.join("results", "SEA_NET", "pooling_benchmark.csv")

# our new pooling heads (highlighted in the benchmark figure so they stand out from the baselines)
NEW_POOLING_HEADS = {"classwise_conjunctive", "softmax_conjunctive", "adaptive_classwise"}


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


def load_benchmark(path: str = BENCHMARK_CSV) -> pd.DataFrame:
    """
    Load the pooling-head benchmark frame (from `python main.py benchmark`), or an empty frame.

    The file is append-only, so a re-run of the same (dataset, pooling) pair can appear twice; we keep
    the last row for each pair (newest wins), exactly like results.load_results does for results.csv.

    path : the pooling_benchmark.csv file.
    returns : a DataFrame (empty if the benchmark has not been run yet).
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    df = _to_num(D.read_our_csv(path),
                 ["test_acc", "test_bal_acc", "test_auroc", "test_loss", "test_aopcr", "test_ndcg",
                  "params", "train_time_s"])
    if {"dataset", "pooling"}.issubset(df.columns):
        df = df.drop_duplicates(["dataset", "pooling"], keep="last").reset_index(drop=True)
    return df


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


def plot_pooling_benchmark(bench: pd.DataFrame, figdir: str = FIGURES_DIR) -> List[str]:
    """
    Draw the pooling-head comparison from pooling_benchmark.csv. Two figures:
      1) pooling_benchmark.png            - mean test accuracy and mean AOPCR per head (bars, best first),
      2) pooling_benchmark_by_dataset.png - a per-dataset test-accuracy heatmap (head x dataset).
    Our new heads (classwise_conjunctive / softmax_conjunctive / adaptive_classwise) are drawn in orange
    so they stand out from the MILLET baselines.

    bench : the benchmark frame from load_benchmark().
    figdir : where to save the PNGs.
    returns : the list of saved figure paths (empty if there is nothing to plot).
    """
    os.makedirs(figdir, exist_ok=True)
    paths: List[str] = []
    if bench.empty or "pooling" not in bench.columns:
        return paths

    n_datasets = bench["dataset"].nunique()
    # mean of each metric over the datasets, per pooling head, best accuracy first
    agg = bench.groupby("pooling").agg(
        mean_acc=("test_acc", "mean"),
        mean_aopcr=("test_aopcr", "mean"),
    ).sort_values("mean_acc", ascending=False)

    # figure 1: mean accuracy + mean AOPCR bars (each panel sorted by its own metric, best first)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, col, title in [(ax[0], "mean_acc", "Mean test accuracy"),
                          (ax[1], "mean_aopcr", "Mean AOPCR (interpretability)")]:
        s = agg[col].sort_values(ascending=False)
        colors = ["darkorange" if p in NEW_POOLING_HEADS else "steelblue" for p in s.index]
        bars = a.bar(range(len(s)), s.values, color=colors)
        a.set_xticks(range(len(s))); a.set_xticklabels(s.index, rotation=30, ha="right")
        a.set_title(title); a.set_ylabel(col)
        for b, v in zip(bars, s.values):                        # write the value on top of each bar
            if pd.notna(v):
                a.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Pooling head comparison (mean over {n_datasets} datasets)   [orange = our new heads]")
    fig.tight_layout()
    p = os.path.join(figdir, "pooling_benchmark.png"); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # figure 2: per-dataset accuracy heatmap (rows = pooling head ordered best-first, cols = dataset)
    pivot = bench.pivot_table(index="pooling", columns="dataset", values="test_acc", aggfunc="mean")
    pivot = pivot.reindex(agg.index)                            # rows in the same best-first order as above
    vmin, vmax = float(pivot.min().min()), float(pivot.max().max())
    fig, ax = plt.subplots(figsize=(1.4 * len(pivot.columns) + 3, 0.6 * len(pivot.index) + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):                             # annotate each cell with its accuracy
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.notna(v):
                shade = (v - vmin) / (vmax - vmin + 1e-9)       # dark cell -> white text, light cell -> black
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if shade < 0.5 else "black")
    fig.colorbar(im, ax=ax, label="test accuracy")
    ax.set_title("Test accuracy per pooling head x dataset")
    fig.tight_layout()
    p = os.path.join(figdir, "pooling_benchmark_by_dataset.png")
    fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)
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


def per_model_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (model, settings): how far it got, how well it did, and when it last ran.

    Now that several models can each sweep the archive, a single mean over results.csv would blend
    them into a meaningless number. This keeps them apart, so the README can show "seanet did 129
    datasets at 0.83; seanet_acp did 129 at 0.84" instead of one mixed average.

    results : the results frame (every model, every setting).
    returns : a DataFrame sorted by mean accuracy, best first (empty if there are no results).
    """
    if results.empty or "model" not in results.columns:
        return pd.DataFrame()
    df = results.copy()
    if "settings" not in df.columns:
        df["settings"] = ""
    grouped = df.groupby(["model", "settings"], dropna=False)
    out = grouped.agg(
        datasets=("dataset", "nunique"),
        mean_acc=("test_acc", "mean"),
        mean_aopcr=("test_aopcr", "mean"),
        params=("params", "median"),
        last_run=("run_at", "max"),
    ).reset_index()
    out["mean_acc"] = out["mean_acc"].round(4)
    out["mean_aopcr"] = out["mean_aopcr"].round(4)
    out["params"] = out["params"].astype("Int64")
    return out.sort_values("mean_acc", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# README auto-update
#
# The README's results section is generated from the CSVs, between the two markers below, so it can
# never drift away from what was actually measured. Everything outside the markers is hand-written
# and is never touched. If the markers are missing (a fresh clone of the old README), the section is
# appended once at the end and picked up in place from then on.
# --------------------------------------------------------------------------------------
README_PATH = "README.md"
README_BEGIN = "<!-- BEGIN AUTO-RESULTS (generated by `python main.py report`) -->"
README_END = "<!-- END AUTO-RESULTS -->"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Build a GitHub-flavoured markdown table from headers + already-stringified rows."""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def render_readme_section(perf: Dict, per_model: pd.DataFrame, figures: List[str]) -> str:
    """
    Build the markdown that goes between the README markers: headline numbers, a per-model table,
    and the figures that exist right now.

    perf : the headline dict from performance_summary.
    per_model : the table from per_model_summary.
    figures : the PNG paths produced by this report run.
    returns : the markdown string (without the markers themselves).
    """
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        "## Results",
        "",
        f"_Generated by `python main.py report` on {stamp} from `results/SEA_NET/results.csv`._",
        f"_Do not edit this section by hand - it is rewritten on every report._",
        "",
    ]

    if perf:
        parts += ["### Headline (best model per dataset)", "",
                  _md_table(["metric", "value"], [[str(k), str(v)] for k, v in perf.items()]), ""]

    if len(per_model):
        rows = [[str(r["model"]), f"`{r['settings']}`", str(r["datasets"]),
                 f"{r['mean_acc']:.4f}" if pd.notna(r["mean_acc"]) else "-",
                 f"{r['mean_aopcr']:.4f}" if pd.notna(r["mean_aopcr"]) else "-",
                 f"{int(r['params']):,}" if pd.notna(r["params"]) else "-",
                 str(r["last_run"])[:16] if pd.notna(r["last_run"]) else "-"]
                for _, r in per_model.iterrows()]
        parts += ["### Every model we have run", "",
                  "One row per model + settings fingerprint. Change a hyperparameter and the "
                  "fingerprint changes, so a retrained recipe appears as its own row rather than "
                  "overwriting the old one.", "",
                  _md_table(["model", "settings", "datasets", "mean acc", "mean AOPCR",
                             "params", "last run"], rows), ""]

    existing = [p for p in figures if os.path.exists(p)]
    if existing:
        parts += ["### Figures", ""]
        for path in existing:
            title = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
            parts += [f"**{title}**", "", f"![{title}]({path.replace(os.sep, '/')})", ""]

    return "\n".join(parts).rstrip() + "\n"


def update_readme(perf: Dict, per_model: pd.DataFrame, figures: List[str],
                  path: str = README_PATH) -> Optional[str]:
    """
    Rewrite the README's auto-generated results section in place.

    Only the text between README_BEGIN and README_END is replaced; the rest of the README (the
    project description, the architecture write-up, the how-to-run instructions) is preserved
    byte for byte. If the markers are not there yet, the section is appended once.

    perf : headline numbers. per_model : the per-model table. figures : PNG paths.
    path : the README to update.
    returns : the path written, or None if the README does not exist.
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()

    block = f"{README_BEGIN}\n{render_readme_section(perf, per_model, figures)}{README_END}"
    start, end = text.find(README_BEGIN), text.find(README_END)
    if start != -1 and end != -1:                                # replace between the markers
        text = text[:start] + block + text[end + len(README_END):]
    else:                                                        # first time: append the section
        text = text.rstrip() + "\n\n" + block + "\n"
    with open(path, "w") as f:
        f.write(text)
    return path


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


def generate_report(figdir: str = FIGURES_DIR, verbose: bool = True,
                    readme: bool = True) -> Dict:
    """
    Do everything: load the CSVs, draw all figures, write the summary table, and refresh the README.

    figdir : where to save the PNG figures.
    verbose : if True, print the headline summary and where things were written.
    readme : if True, rewrite the README's auto-generated results section from these numbers.
    returns : a dict with "summary" (headline numbers), "figures" (PNG paths), "per_model" (table).
    """
    summary, results, comparison = load_frames()
    figures: List[str] = []
    if len(summary):
        figures.append(plot_data_summary(summary, figdir))
    if len(results):
        figures.append(plot_results(results, figdir))
    if len(comparison):
        figures += plot_comparison(comparison, figdir)

    # optional: the pooling-head comparison, only if `python main.py benchmark` has been run
    bench = load_benchmark()
    if len(bench):
        figures += plot_pooling_benchmark(bench, figdir)

    # The headline is computed over the BEST model per dataset, not over raw results.csv: once more
    # than one model has swept the archive, averaging every row together would mix models and mean
    # nothing. per_model_summary below keeps each model's own record visible next to it.
    best = _to_num(R.build_best_results(verbose=False),
                   ["test_acc", "test_aopcr", "test_ndcg", "params", "model_size_mb"])
    headline_frame = best if len(best) else results
    perf = performance_summary(headline_frame, comparison)
    per_model = per_model_summary(results)
    csv_path, md_path = write_summary_table(perf)

    readme_path = update_readme(perf, per_model, figures) if readme else None

    if verbose:
        print("=== SEA-Net report ===")
        for key, value in perf.items():
            print(f"  {key:26s}: {value}")
        if len(per_model):                                       # each model's own record
            print("\n  per model (model | settings | datasets | mean acc | last run):")
            for _, r in per_model.iterrows():
                acc = f"{r['mean_acc']:.4f}" if pd.notna(r["mean_acc"]) else "  -   "
                print(f"    {str(r['model']):24s} {str(r['settings']):10s} "
                      f"{int(r['datasets']):>4d}  {acc}  {str(r['last_run'])[:16]}")
        if len(bench):                                          # short pooling ranking, best first
            print("\n  pooling benchmark (mean test_acc over "
                  f"{bench['dataset'].nunique()} datasets):")
            rank = bench.groupby("pooling")["test_acc"].mean().sort_values(ascending=False)
            for pooling, acc in rank.items():
                mark = "  <- new" if pooling in NEW_POOLING_HEADS else ""
                print(f"    {pooling:24s} {acc:.4f}{mark}")
        print(f"\nwrote {len(figures)} figures -> {figdir}/")
        for p in figures:
            print("  ", p)
        print(f"wrote summary table -> {csv_path} and {md_path}")
        if readme_path:
            print(f"updated README results section -> {readme_path}")
    return {"summary": perf, "figures": figures, "per_model": per_model}
