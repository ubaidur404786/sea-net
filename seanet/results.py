"""
seanet/results.py - saving results and comparing to MILLET.

What this file is for:
    Two jobs, both just pandas + csv:
      1. Remember what has finished and store each dataset's numbers, so a long run can stop and
         resume without redoing work.
      2. Compare our UCR numbers to the MILLET paper's numbers, dataset by dataset, and say where
         we win / tie / lose.

Files it writes (each model in its OWN folder results/SEA_NET/<model>/, e.g. results/SEA_NET/seanet/):
    - results.csv               : one row of metrics per finished dataset (append-only).
    - done.txt                  : plain-text list of finished dataset names (the "what is done" list).
    - comparison_vs_millet.csv  : our numbers next to MILLET's, with win/tie/loss columns.
    (+ one shared results/SEA_NET/model_comparison.csv, written by compare_models, ranking the models.)

    Per-model folders are what let us sweep several pooling heads (seanet, seanet_conjunctive,
    seanet_acp, ...) over ALL datasets without clobbering each other: results.csv / done.txt are
    keyed by dataset name only, so two models sharing one folder would overwrite each other's rows.

Files it reads:
    - results.csv / done.txt    : our own outputs.
    - results/UCR/InceptionTime/test_acc.csv and test_aopcr.csv : the MILLET paper's published
      numbers (already in the repo). We use the Conjunctive-InceptionTime model as the baseline.

Related files:
    - main.py calls result_exists() to skip finished datasets, save_result_row() after each
      dataset, and build_comparison() at the end (and for the "results" command).
    - analysis.ipynb reads results.csv and calls build_comparison() to make figures.

Note on this machine: some tool keeps re-aligning .csv files (padding columns with spaces), which
once corrupted results.csv mid-run and made the sweep restart from scratch. To be safe:
  - the "what is done" list is a plain .txt file (done.txt), which the tool leaves alone, and
  - results.csv is written append-only (never read-then-rewritten), so it can't be corrupted mid-write.
"""
import os
from typing import Dict

import numpy as np
import pandas as pd

from seanet.data import UCR_128_DATASETS, read_our_csv

# --------------------------------------------------------------------------------------
# Paths + column order
# --------------------------------------------------------------------------------------
# Each model writes into its OWN folder: results/SEA_NET/<model>/. Use the *_for(model) helpers below
# to build the paths; the bare constants (RESULTS_CSV, ...) are kept for old callers and point at the
# DEFAULT model's folder (seanet).
RESULTS_ROOT = os.path.join("results", "SEA_NET")     # everything SEA-Net writes lives under here
DEFAULT_MODEL = "seanet"                              # the model whose folder the plain sweep uses


def results_dir_for(model: str = None) -> str:
    """The folder that holds one model's result files: results/SEA_NET/<model>/ (default: seanet)."""
    return os.path.join(RESULTS_ROOT, model or DEFAULT_MODEL)


def results_csv_for(model: str = None) -> str:
    """That model's metrics table (one row per finished dataset)."""
    return os.path.join(results_dir_for(model), "results.csv")


def done_txt_for(model: str = None) -> str:
    """That model's 'what is finished' list (plain text, one dataset name per line)."""
    return os.path.join(results_dir_for(model), "done.txt")


def comparison_csv_for(model: str = None) -> str:
    """That model's per-dataset comparison against the MILLET baseline."""
    return os.path.join(results_dir_for(model), "comparison_vs_millet.csv")


# Backward-compatible module constants (the DEFAULT model's paths). Prefer the *_for(model) helpers.
RESULTS_CSV = results_csv_for()                                           # our metrics, one row per dataset
COMPARISON_CSV = comparison_csv_for()                                     # our numbers vs MILLET
DONE_TXT = done_txt_for()                                                 # plain-text list of finished datasets

# Where the MILLET paper's published numbers live (one column per model and repetition).
MILLET_UCR_DIR = os.path.join("results", "UCR", "InceptionTime")
BASELINE_MODEL = "ConjunctiveInceptionTime"   # the MILLET model we compare against (it has 5 reps)

# The column order for results.csv, so the file is always laid out the same way.
RESULT_COLUMNS = [
    "dataset", "model", "seed", "device", "params", "model_size_mb",
    "n_train", "n_val", "n_test", "series_length", "n_classes", "lambda_entropy",
    "test_acc", "test_bal_acc", "test_auroc", "test_loss", "test_aopcr", "test_ndcg",
    "train_time_s",
]

# How close counts as a "tie" (anything smaller than this is not a real win or loss).
ACC_TIE_BAND = 0.005    # 0.5% for accuracy
AOPCR_TIE_BAND = 0.1    # AOPCR is on a ~0..15 scale, so 0.1 is a small band


# --------------------------------------------------------------------------------------
# 1. Remembering what is done + saving results
# --------------------------------------------------------------------------------------
def load_done(model: str = None) -> set:
    """
    Read a model's done.txt into a set of dataset names.

    model : which model's list to read (default: seanet).
    returns : set of finished dataset names (empty set if the file does not exist yet).
    """
    path = done_txt_for(model)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}   # one name per line, ignore blanks


def mark_done(name: str, model: str = None) -> None:
    """
    Add a dataset name to a model's done.txt (only if it is not already there).

    name : the finished dataset name.  model : which model's list (default: seanet).
    returns : nothing.
    """
    path = done_txt_for(model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if name not in load_done(model):                          # avoid writing the same name twice
        with open(path, "a") as f:
            f.write(name + "\n")


def result_exists(name: str, model: str = None) -> bool:
    """
    Say whether a dataset is already finished FOR THIS MODEL. We check done.txt (a plain-text file
    that the CSV aligner cannot corrupt), not results.csv.

    name : dataset name.  model : which model's list (default: seanet).
    returns : True if the dataset is in that model's done.txt.
    """
    return name in load_done(model)


def load_results(model: str = None, path: str = None) -> pd.DataFrame:
    """
    Read a model's results.csv into a DataFrame.

    Because results.csv is append-only, the same dataset could appear more than once (if it was
    re-run), so we keep only the last row for each dataset.

    model : which model's results to read (default: seanet).
    path : read this exact file instead (overrides model); handy for tests.
    returns : a DataFrame of results (empty but correctly-columned if the file is missing).
    """
    path = path or results_csv_for(model)
    if not os.path.exists(path):
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = read_our_csv(path)                                   # tolerant read (handles aligned csv)
    if "dataset" in df.columns:
        df = df.drop_duplicates("dataset", keep="last").reset_index(drop=True)   # last row wins
    return df


def save_result_row(row: Dict, model: str = None, path: str = None) -> None:
    """
    Append one dataset's result row to a model's results.csv and record the dataset in its done.txt.

    We append instead of rewriting the whole file, so the CSV-aligning tool cannot corrupt it
    while we write. If a dataset ends up with more than one row (from a re-run), load_results
    keeps the last one.

    row : a results-row dict from train_one / score_model.
    model : which model's folder to write into (default: seanet).
    path : write this exact file instead (overrides model).
    returns : nothing.
    """
    path = path or results_csv_for(model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([row]).reindex(columns=RESULT_COLUMNS)   # put columns in the fixed order
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)   # append (header only if new file)
    mark_done(row["dataset"], model)                          # remember it is done (in the same model's list)


# --------------------------------------------------------------------------------------
# 2. Comparing our numbers to MILLET
# --------------------------------------------------------------------------------------
def millet_baseline(metric_csv: str) -> pd.Series:
    """
    Read one MILLET metric file and return, per dataset, the mean of the 5 Conjunctive reps.

    The paper reports 5 runs (reps 0..4). We average them for a fair single number. The
    "...Ensemble 0" column is left out because it is an ensemble of the 5, not a single model.

    metric_csv : file name inside results/UCR/InceptionTime/, e.g. "test_acc.csv".
    returns : a Series indexed by dataset name, values = the mean baseline metric.
    """
    df = pd.read_csv(os.path.join(MILLET_UCR_DIR, metric_csv))
    df.columns = df.columns.str.strip()                       # clean any padded header names
    df["Dataset"] = df["Dataset"].str.strip()
    reps = [f"{BASELINE_MODEL} {i}" for i in range(5)]        # the 5 columns to average
    return df.set_index("Dataset")[reps].mean(axis=1)


def _outcome(diff: float, band: float) -> str:
    """
    Turn a (ours - baseline) difference into "win" / "tie" / "loss".

    diff : ours minus baseline (positive = we are higher).
    band : how far from 0 still counts as a tie.
    returns : "win", "tie", or "loss".
    """
    if diff > band:
        return "win"
    if diff < -band:
        return "loss"
    return "tie"


def build_comparison(model: str = None, results_path: str = None, out: str = None,
                     verbose: bool = True) -> pd.DataFrame:
    """
    Join one model's UCR results to the MILLET baseline by dataset name and write its
    comparison_vs_millet.csv (inside that model's own folder).

    Every UCR dataset we have a result for shows up:
      - the 85 that the paper also reports  -> get millet_acc / millet_aopcr + win/tie/loss.
      - the ~43 the paper skipped           -> get a blank baseline and outcome "no_baseline".
    WebTraffic is left out here on purpose (it is not a UCR dataset; it is checked separately by
    `python main.py webtraffic`).

    model : which model to compare (default: seanet).
    results_path : the results csv to read (overrides the model-derived path).
    out : where to write the comparison csv (overrides the model-derived path).
    verbose : if True, also print the win/tie/loss summary.
    returns : the comparison DataFrame (also saved to `out`).
    """
    results_path = results_path or results_csv_for(model)
    out = out or comparison_csv_for(model)
    res = load_results(model=model, path=results_path)
    ucr_set = set(UCR_128_DATASETS)
    res = res[res["dataset"].isin(ucr_set)].copy()            # keep UCR datasets only (drop WebTraffic)

    base_acc = millet_baseline("test_acc.csv")                # MILLET accuracy per dataset
    base_aopcr = millet_baseline("test_aopcr.csv")            # MILLET AOPCR per dataset

    rows = []
    for _, r in res.iterrows():                               # go through each dataset we ran
        name = r["dataset"]
        has_baseline = name in base_acc.index                 # is this one of the paper's 85?
        m_acc = float(base_acc[name]) if has_baseline else np.nan
        m_aopcr = float(base_aopcr[name]) if name in base_aopcr.index else np.nan
        o_acc = float(r["test_acc"]) if pd.notna(r["test_acc"]) else np.nan
        o_aopcr = float(r["test_aopcr"]) if pd.notna(r["test_aopcr"]) else np.nan
        rows.append({
            "dataset": name,
            "ours_acc": round(o_acc, 4),
            "millet_acc": round(m_acc, 4) if has_baseline else np.nan,
            "acc_diff": round(o_acc - m_acc, 4) if has_baseline else np.nan,
            "acc_outcome": _outcome(o_acc - m_acc, ACC_TIE_BAND) if has_baseline else "no_baseline",
            "ours_aopcr": round(o_aopcr, 4),
            "millet_aopcr": round(m_aopcr, 4) if has_baseline else np.nan,
            "aopcr_diff": round(o_aopcr - m_aopcr, 4) if has_baseline else np.nan,
            "aopcr_outcome": _outcome(o_aopcr - m_aopcr, AOPCR_TIE_BAND) if has_baseline else "no_baseline",
            "params": int(r["params"]) if pd.notna(r.get("params")) else np.nan,
            "model_size_mb": r.get("model_size_mb"),
        })

    # sort so the paper's 85 come first (in the paper's order), then the rest alphabetically
    paper_order = list(base_acc.index)
    order_key = {name: i for i, name in enumerate(paper_order)}
    cmp = pd.DataFrame(rows)
    if not cmp.empty:
        cmp["_k"] = cmp["dataset"].map(lambda n: order_key.get(n, len(paper_order)))
        cmp = cmp.sort_values(["_k", "dataset"]).drop(columns="_k").reset_index(drop=True)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cmp.to_csv(out, index=False)

    if verbose:
        _print_comparison_summary(cmp, out)
    return cmp


def _print_comparison_summary(cmp: pd.DataFrame, out: str) -> None:
    """
    Print a short win/tie/loss summary over the datasets that have a MILLET baseline.

    cmp : the comparison DataFrame from build_comparison.
    out : the path the comparison was written to (just for the printed message).
    returns : nothing.
    """
    if cmp.empty:                                             # nothing run yet
        print("No UCR results yet - run the sweep first (nothing to compare).")
        return
    overlap = cmp[cmp["acc_outcome"] != "no_baseline"]        # only the datasets with a baseline
    n_no_base = int((cmp["acc_outcome"] == "no_baseline").sum())
    print(f"UCR datasets evaluated: {len(cmp)}  "
          f"(with MILLET baseline: {len(overlap)}, without: {n_no_base})")
    if len(overlap) > 0:
        acc_counts = overlap["acc_outcome"].value_counts()
        aopcr_counts = overlap["aopcr_outcome"].value_counts()
        # accuracy line: win/tie/loss counts + the two mean accuracies
        print(f"  accuracy  vs MILLET  -> win/tie/loss = "
              f"{acc_counts.get('win', 0)}/{acc_counts.get('tie', 0)}/{acc_counts.get('loss', 0)}"
              f"   | mean acc  ours {overlap['ours_acc'].mean():.4f} vs MILLET {overlap['millet_acc'].mean():.4f}")
        # AOPCR line: same idea for the interpretability metric
        print(f"  AOPCR     vs MILLET  -> win/tie/loss = "
              f"{aopcr_counts.get('win', 0)}/{aopcr_counts.get('tie', 0)}/{aopcr_counts.get('loss', 0)}"
              f"   | mean AOPCR ours {overlap['ours_aopcr'].mean():.3f} vs MILLET {overlap['millet_aopcr'].mean():.3f}")
    print(f"  wrote {out}")


# --------------------------------------------------------------------------------------
# 3. Comparing several models (pooling heads) to each other
# --------------------------------------------------------------------------------------
MODEL_COMPARISON_CSV = os.path.join(RESULTS_ROOT, "model_comparison.csv")   # the cross-model ranking


def discover_result_models() -> list:
    """
    List the model names that have a results folder with a results.csv, so we can compare them.

    Looks for results/SEA_NET/<name>/results.csv. Shared folders (figures/, interpretation/, logs/)
    are skipped because they have no results.csv.

    returns : a sorted list of model names (empty if nothing has been swept yet).
    """
    if not os.path.isdir(RESULTS_ROOT):
        return []
    return [name for name in sorted(os.listdir(RESULTS_ROOT))
            if os.path.exists(os.path.join(RESULTS_ROOT, name, "results.csv"))]


def compare_models(models: list = None, out: str = MODEL_COMPARISON_CSV, verbose: bool = True) -> pd.DataFrame:
    """
    Build a head-to-head table ACROSS models (e.g. the pooling heads): one row per model with its
    mean accuracy / AOPCR and its win/tie/loss record against the MILLET baseline, over the UCR
    datasets. This is the "which pooling wins overall" table.

    It reuses build_comparison(model=...) per model (which also refreshes each model's own
    comparison_vs_millet.csv), so the per-model and cross-model numbers always agree.

    models : which models to include (default: every model with a results folder).
    out : where to write the table (default: results/SEA_NET/model_comparison.csv).
    verbose : if True, print the ranking.
    returns : the cross-model DataFrame (also saved to `out`), best mean accuracy first.
    """
    models = models if models is not None else discover_result_models()
    rows = []
    for m in models:
        cmp = build_comparison(model=m, verbose=False)        # also (re)writes that model's comparison csv
        if cmp.empty:
            continue
        overlap = cmp[cmp["acc_outcome"] != "no_baseline"]    # only datasets the paper also reports
        av = overlap["acc_outcome"].value_counts()
        pv = overlap["aopcr_outcome"].value_counts()
        rows.append({
            "model": m,
            "n_datasets": int(len(cmp)),
            "n_with_baseline": int(len(overlap)),
            "mean_acc": round(float(cmp["ours_acc"].mean()), 4),
            "mean_aopcr": round(float(cmp["ours_aopcr"].mean()), 4),
            "acc_win_tie_loss": f"{av.get('win', 0)}/{av.get('tie', 0)}/{av.get('loss', 0)}",
            "acc_ours_vs_millet": (f"{overlap['ours_acc'].mean():.4f} / {overlap['millet_acc'].mean():.4f}"
                                   if len(overlap) else ""),
            "aopcr_win_tie_loss": f"{pv.get('win', 0)}/{pv.get('tie', 0)}/{pv.get('loss', 0)}",
            "aopcr_ours_vs_millet": (f"{overlap['ours_aopcr'].mean():.3f} / {overlap['millet_aopcr'].mean():.3f}"
                                     if len(overlap) else ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("mean_acc", ascending=False).reset_index(drop=True)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False)
    if verbose:
        _print_model_comparison(df, out)
    return df


def _print_model_comparison(df: pd.DataFrame, out: str) -> None:
    """Print the cross-model head-to-head ranking (best mean accuracy first)."""
    if df.empty:
        print("No model results yet - run `python main.py train --model <name>` first.")
        return
    print(f"Cross-model comparison over {len(df)} model(s) (best mean UCR accuracy first):")
    print(f"  {'model':22s} {'mean_acc':>9s} {'acc W/T/L':>11s} {'mean_aopcr':>11s} {'aopcr W/T/L':>12s}")
    for _, r in df.iterrows():
        print(f"  {r['model']:22s} {r['mean_acc']:>9.4f} {r['acc_win_tie_loss']:>11s} "
              f"{r['mean_aopcr']:>11.4f} {r['aopcr_win_tie_loss']:>12s}")
    print(f"  wrote {out}")
