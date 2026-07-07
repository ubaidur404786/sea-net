"""
seanet/results.py - saving results and comparing to MILLET.

What this file is for:
    Two jobs, both just pandas + csv:
      1. Remember what has finished and store each dataset's numbers, so a long run can stop and
         resume without redoing work.
      2. Compare our UCR numbers to the MILLET paper's numbers, dataset by dataset, and say where
         we win / tie / lose.

Files it writes (all under results/SEA_NET/):
    - results.csv               : one row of metrics per finished dataset (append-only).
    - done.txt                  : plain-text list of finished dataset names (the "what is done" list).
    - comparison_vs_millet.csv  : our numbers next to MILLET's, with win/tie/loss columns.

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
RESULTS_CSV = os.path.join("results", "SEA_NET", "results.csv")           # our metrics, one row per dataset
COMPARISON_CSV = os.path.join("results", "SEA_NET", "comparison_vs_millet.csv")  # our numbers vs MILLET
DONE_TXT = os.path.join("results", "SEA_NET", "done.txt")                 # plain-text list of finished datasets

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
def load_done() -> set:
    """
    Read done.txt into a set of dataset names.

    returns : set of finished dataset names (empty set if done.txt does not exist yet).
    """
    if not os.path.exists(DONE_TXT):
        return set()
    with open(DONE_TXT) as f:
        return {line.strip() for line in f if line.strip()}   # one name per line, ignore blanks


def mark_done(name: str) -> None:
    """
    Add a dataset name to done.txt (only if it is not already there).

    name : the finished dataset name.
    returns : nothing.
    """
    os.makedirs(os.path.dirname(DONE_TXT), exist_ok=True)
    if name not in load_done():                               # avoid writing the same name twice
        with open(DONE_TXT, "a") as f:
            f.write(name + "\n")


def result_exists(name: str, path: str = RESULTS_CSV) -> bool:
    """
    Say whether a dataset is already finished. We check done.txt (a plain-text file that the CSV
    aligner cannot corrupt), not results.csv.

    name : dataset name.
    path : kept for a consistent signature; not used.
    returns : True if the dataset is in done.txt.
    """
    return name in load_done()


def load_results(path: str = RESULTS_CSV) -> pd.DataFrame:
    """
    Read results.csv into a DataFrame.

    Because results.csv is append-only, the same dataset could appear more than once (if it was
    re-run), so we keep only the last row for each dataset.

    path : the results csv file.
    returns : a DataFrame of results (empty but correctly-columned if the file is missing).
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = read_our_csv(path)                                   # tolerant read (handles aligned csv)
    if "dataset" in df.columns:
        df = df.drop_duplicates("dataset", keep="last").reset_index(drop=True)   # last row wins
    return df


def save_result_row(row: Dict, path: str = RESULTS_CSV) -> None:
    """
    Append one dataset's result row to results.csv and record the dataset in done.txt.

    We append instead of rewriting the whole file, so the CSV-aligning tool cannot corrupt it
    while we write. If a dataset ends up with more than one row (from a re-run), load_results
    keeps the last one.

    row : a results-row dict from train_one.
    path : the results csv file.
    returns : nothing.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([row]).reindex(columns=RESULT_COLUMNS)   # put columns in the fixed order
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)   # append (header only if new file)
    mark_done(row["dataset"])                                 # remember it is done


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


def build_comparison(results_path: str = RESULTS_CSV, out: str = COMPARISON_CSV,
                     verbose: bool = True) -> pd.DataFrame:
    """
    Join our UCR results to the MILLET baseline by dataset name and write comparison_vs_millet.csv.

    Every UCR dataset we have a result for shows up:
      - the 85 that the paper also reports  -> get millet_acc / millet_aopcr + win/tie/loss.
      - the ~43 the paper skipped           -> get a blank baseline and outcome "no_baseline".
    WebTraffic is left out here on purpose (it is not a UCR dataset; it is checked separately by
    `python main.py webtraffic`).

    results_path : the results csv to read.
    out : where to write the comparison csv.
    verbose : if True, also print the win/tie/loss summary.
    returns : the comparison DataFrame (also saved to `out`).
    """
    res = load_results(results_path)
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
