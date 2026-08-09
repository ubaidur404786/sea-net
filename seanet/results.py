"""
seanet/results.py - where every model's results are stored, and how they are compared to MILLET.

What this file is for:
    Two jobs, both just pandas + csv:
      1. Remember what each model has already trained, and store its numbers, so a long sweep can
         be stopped and resumed without redoing work.
      2. Compare our UCR numbers to the MILLET paper's numbers, dataset by dataset, and say where
         we win / tie / lose on accuracy, loss and AOPCR.

One config file = ONE FOLDER, with a unique name
------------------------------------------------
A model's folder is named "<config file name>__<encoder>__<pooling>", e.g.
"seanet__sea_mstcn_sep__mil_additive" (see seanet/config.model_folder_name). The config file name is in front
so two configs that build the same encoder+pooling never share a folder. Everything a model produces
lives in its own folder, so two models can never mix their numbers up:

    results/SEA_NET/
      data_summary.csv           <- shared: facts about the DATA, not about any model
      model_comparison.csv       <- shared: the cross-model ranking (written by compare_models)
      figures/                   <- shared: the cross-model figure
      logs/                      <- shared: logs of commands that belong to no model (summary, report)

      seanet__sea_mstcn_sep__mil_additive/        <- ONE model
        results.csv                      <- one row per dataset (best accuracy kept, see save_result_row)
        done_train_dataset.txt           <- the "what is finished" list (the resume switch)
        comparison_vs_millet.csv         <- our numbers next to MILLET's, per dataset
        summary.csv / summary.md         <- the headline means (over the 85, and overall)
        logs/<command>_<date-time>.log   <- one log per run of THIS model
        figures/                         <- this model's figures
        interpretation/                  <- this model's explanation figures

      seanet_acp__sea_mstcn_sep__sea_adaptive_classwise/   <- another model, same layout
        ...

How resuming works (done_train_dataset.txt)
-------------------------------------------
That file is a plain list of dataset names, one per line. `python main.py train` skips any dataset
already in it. So:
  - delete the whole file      -> the model trains every dataset again from the start,
  - delete one name from it    -> only that dataset is trained again, and its row in results.csv is
                                  updated ONLY IF the new run beats the old accuracy (save_result_row
                                  keeps the better result; a worse re-run is logged to MLflow but the
                                  .csv is left unchanged).

Files it reads:
    - results/UCR/InceptionTime/{test_acc,test_loss,test_aopcr}.csv : the MILLET paper's published
      numbers (already in the repo). We compare against ConjunctiveInceptionTime (mean of its 5 reps).

Related files:
    - seanet/config.py  -> model_folder_name(cfg) builds the model id this module keys everything on.
    - main.py           -> calls result_exists() to skip finished datasets, save_result_row() after
                           each dataset, and build_comparison() / compare_models() to report.
    - seanet/report.py  -> draws the figures from the tables this module writes.

Note on this machine: some tool keeps re-aligning .csv files (padding columns with spaces), which
once corrupted a results file mid-run. Two defences stay in place:
  - the "what is done" list is a plain .txt file, which that tool leaves alone, and
  - results.csv is written atomically (to a temporary file, then renamed), so a half-written file
    can never replace a good one.
"""
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from seanet.config import is_millet, is_ours, split_model_id
from seanet.data import UCR_128_DATASETS, WEB_TRAFFIC, read_our_csv

# --------------------------------------------------------------------------------------
# 1. Paths - everything is keyed by the model id ("<config>__<encoder>__<pooling>")
# --------------------------------------------------------------------------------------
RESULTS_ROOT = os.path.join("results", "SEA_NET")     # everything SEA-Net writes lives under here

# The per-model file names. They are the same inside every model folder, so you always know where
# to look; the folder name is what tells the models apart.
RESULTS_FILE = "results.csv"                  # one row per dataset
DONE_FILE = "done_train_dataset.txt"          # the resume list (delete it to retrain everything)
COMPARISON_FILE = "comparison_vs_millet.csv"  # our numbers vs the paper's
SUMMARY_CSV_FILE = "summary.csv"              # the headline means
SUMMARY_MD_FILE = "summary.md"                # the same means as a markdown table

# The shared (not per-model) outputs.
MODEL_COMPARISON_CSV = os.path.join(RESULTS_ROOT, "model_comparison.csv")   # the cross-model ranking
SHARED_FIGURES_DIR = os.path.join(RESULTS_ROOT, "figures")                  # the cross-model figure
SHARED_LOGS_DIR = os.path.join(RESULTS_ROOT, "logs")                        # logs with no model


def model_dir(model_id: str) -> str:
    """The one folder that holds everything for a model, e.g. results/SEA_NET/seanet__sea_mstcn_sep__mil_additive."""
    return os.path.join(RESULTS_ROOT, model_id)


def results_csv(model_id: str) -> str:
    """That model's metrics table (one row per dataset)."""
    return os.path.join(model_dir(model_id), RESULTS_FILE)


def done_txt(model_id: str) -> str:
    """That model's 'what is finished' list. Delete it (or a line in it) to retrain."""
    return os.path.join(model_dir(model_id), DONE_FILE)


def comparison_csv(model_id: str) -> str:
    """That model's per-dataset comparison against the MILLET baseline."""
    return os.path.join(model_dir(model_id), COMPARISON_FILE)


def summary_paths(model_id: str) -> tuple:
    """That model's headline summary, as (csv path, markdown path)."""
    return (os.path.join(model_dir(model_id), SUMMARY_CSV_FILE),
            os.path.join(model_dir(model_id), SUMMARY_MD_FILE))


def figures_dir(model_id: str) -> str:
    """Where that model's figures are saved."""
    return os.path.join(model_dir(model_id), "figures")


def predictions_dir(model_id: str) -> str:
    """
    Where that model's per-series test predictions are saved (one .npz per dataset per seed).

    These are what scripts/ensemble_vote.py reads to build a majority vote between two models -
    results.csv only has summary numbers, and you cannot vote with an accuracy.
    """
    return os.path.join(model_dir(model_id), "predictions")


def curves_dir(model_id: str) -> str:
    """
    Where that model's training curves are saved (one small CSV per dataset per seed).

    results.csv only keeps the FINAL test numbers, so it cannot answer "did this model overfit?" or
    "at which epoch did early stopping fire?". The per-epoch loss lives in model.history, which used
    to go to MLflow only - and MLflow lives on whichever machine trained the model. Writing the same
    curve next to results.csv means the answer travels with the results when we copy them back.
    """
    return os.path.join(model_dir(model_id), "curves")


def logs_dir(model_id: str) -> str:
    """Where that model's run logs are saved (one dated file per run)."""
    return os.path.join(model_dir(model_id), "logs")


def interpretation_dir(model_id: str) -> str:
    """Where that model's explanation figures are saved."""
    return os.path.join(model_dir(model_id), "interpretation")


# The column order for results.csv, so the file is always laid out the same way.
RESULT_COLUMNS = [
    "dataset", "model", "encoder", "pooling", "seed", "device", "params", "model_size_mb",
    "n_train", "n_val", "n_test", "series_length", "n_classes", "lambda_entropy",
    "test_acc", "test_bal_acc", "test_auroc", "test_loss", "test_aopcr", "test_ndcg",
    "train_time_s", "run_datetime",
]

# --------------------------------------------------------------------------------------
# 2. The MILLET baseline (the paper's published numbers)
# --------------------------------------------------------------------------------------
MILLET_UCR_DIR = os.path.join("results", "UCR", "InceptionTime")
MILLET_WEBTRAFFIC_DIR = os.path.join("results", "WebTraffic", "InceptionTime")
BASELINE_MODEL = "ConjunctiveInceptionTime"   # the MILLET model we compare against (it has 5 reps)

# The three metrics we compare, and which direction is good. "lower_is_better" matters for loss:
# a smaller loss is a WIN, while a bigger accuracy / AOPCR is a win.
#   name  -> (the MILLET csv file, tie band, lower_is_better)
COMPARED_METRICS = {
    "acc":   ("test_acc.csv",   0.005, False),   # 0.5% of accuracy is not a real difference
    "loss":  ("test_loss.csv",  0.010, True),    # cross-entropy: lower wins
    "aopcr": ("test_aopcr.csv", 0.100, False),   # AOPCR is on a ~0..15 scale, so 0.1 is a small band
}


def millet_baseline(metric_csv: str, directory: str = MILLET_UCR_DIR) -> pd.Series:
    """
    Read one MILLET metric file and return, per dataset, the mean of the 5 Conjunctive reps.

    The paper reports 5 runs (reps 0..4). We average them for a fair single number. The
    "...Ensemble 0" column is left out because it is an ensemble of the 5, not a single model.

    These are MILLET's files, which we only ever READ - never write. We still read them tolerantly
    (skipinitialspace + stripping the header and the dataset names), because the CSV-aligning tool on
    this machine pads columns with spaces in any csv it finds, including these ones.

    metric_csv : file name inside `directory`, e.g. "test_acc.csv".
    directory : which baseline folder to read (UCR by default; WebTraffic has its own).
    returns : a Series indexed by dataset name, values = the mean baseline metric.
    """
    df = pd.read_csv(os.path.join(directory, metric_csv), skipinitialspace=True)
    df.columns = df.columns.str.strip()                       # clean any padded header names
    df["Dataset"] = df["Dataset"].str.strip()
    reps = [f"{BASELINE_MODEL} {i}" for i in range(5)]        # the 5 columns to average
    return df.set_index("Dataset")[reps].mean(axis=1)


def millet_datasets() -> List[str]:
    """
    The exact list of datasets MILLET publishes, in MILLET's own order (85 of them, alphabetical).

    This is the list we can compare against one-for-one. Everything else we train is extra.

    returns : a list of 85 dataset names.
    """
    return [str(name) for name in millet_baseline("test_acc.csv").index]


def sweep_order(include_webtraffic: bool = True) -> List[str]:
    """
    The order the sweep trains datasets in: MILLET's order first, then our extras.

    Concretely: WebTraffic (the only dataset with per-timestep ground truth, so the only one with
    NDCG), then the 85 datasets MILLET publishes IN MILLET'S ORDER, then the remaining ~43 UCR
    datasets alphabetically. Doing the 85 first means the full head-to-head comparison is ready
    early - you can stop a sweep after them and still have the complete MILLET table.

    include_webtraffic : put WebTraffic at the front (set False for a UCR-only sweep).
    returns : the list of dataset names to train, in order.
    """
    paper = millet_datasets()                                 # the 85, in the paper's order
    paper_set = set(paper)
    extra = sorted(name for name in UCR_128_DATASETS if name not in paper_set)   # the other ~43
    ordered = [name for name in paper if name in set(UCR_128_DATASETS)] + extra
    return ([WEB_TRAFFIC] if include_webtraffic else []) + ordered


# --------------------------------------------------------------------------------------
# 3. Remembering what is done + saving results
# --------------------------------------------------------------------------------------
def load_done(model_id: str) -> set:
    """
    Read a model's done_train_dataset.txt into a set of dataset names.

    model_id : the model id, e.g. "seanet__sea_mstcn_sep__mil_additive".
    returns : set of finished dataset names (empty set if the file does not exist yet).
    """
    path = done_txt(model_id)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}   # one name per line, ignore blanks


def done_key(name: str, seed: int = 0) -> str:
    """
    One line of done_train_dataset.txt: which (dataset, seed) is finished.

    Seed 0 keeps the plain dataset name, exactly as the file was written before seeds existed, so
    every done-file already on disk keeps working. Other seeds get "Name#1", "Name#2", ... - so
    running seed 1 does NOT skip the datasets seed 0 already finished.

    name : dataset name.  seed : the training seed.
    returns : the line to store / look for.
    """
    seed = int(seed)
    return name if seed == 0 else f"{name}#{seed}"


def mark_done(model_id: str, name: str, seed: int = 0) -> None:
    """
    Add a (dataset, seed) to a model's done_train_dataset.txt (only if it is not already there).

    model_id : the model id.  name : the finished dataset name.  seed : the training seed.
    returns : nothing.
    """
    path = done_txt(model_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    key = done_key(name, seed)
    if key not in load_done(model_id):                        # avoid writing the same line twice
        with open(path, "a") as f:
            f.write(key + "\n")


def result_exists(model_id: str, name: str, seed: int = 0) -> bool:
    """
    Say whether a dataset is already finished FOR THIS MODEL AT THIS SEED.

    We check done_train_dataset.txt (a plain-text file the CSV aligner cannot corrupt), not
    results.csv. Delete the file to retrain everything; delete one line to retrain that one run.

    model_id : the model id.  name : dataset name.  seed : the training seed.
    returns : True if that (dataset, seed) is in the model's done list.
    """
    return done_key(name, seed) in load_done(model_id)


def load_results(model_id: str, path: Optional[str] = None) -> pd.DataFrame:
    """
    Read a model's results.csv into a DataFrame.

    save_result_row keeps one row per (dataset, seed), and we de-duplicate on the same pair when
    reading (keeping the last row) so an older or hand-edited file can never double-count.

    Older files were written before seeds were tracked and have no "seed" column (or a blank one).
    Those rows are read as seed 0, so nothing that already exists on disk is lost.

    model_id : which model's results to read.
    path : read this exact file instead (overrides model_id); handy for tests.
    returns : a DataFrame of results (empty but correctly-columned if the file is missing).
    """
    path = path or results_csv(model_id)
    if not os.path.exists(path):
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = read_our_csv(path)                                   # tolerant read (handles aligned csv)
    if "dataset" not in df.columns:
        return df
    if "seed" not in df.columns:
        df["seed"] = 0                                        # file predates seed tracking
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").fillna(0).astype(int)
    return df.drop_duplicates(["dataset", "seed"], keep="last").reset_index(drop=True)


def mean_over_seeds(res: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a results table that has several seeds per dataset down to ONE row per dataset, by
    averaging the numbers.

    Why this exists: a paper reports the AVERAGE over repeats, not the best repeat. Every place that
    computes a mean over datasets (build_comparison, the leaderboard) calls this first, so running a
    model with 3 seeds gives the mean of the 3 - never 3 copies of the same dataset, and never just
    the luckiest one.

    Non-numeric columns (model, encoder, pooling, device...) are the same for every seed, so we keep
    the first. "n_seeds" is added so a table can say how many repeats a number is averaged over.

    res : a results frame from load_results().
    returns : one row per dataset (the same columns, plus n_seeds).
    """
    if res.empty or "dataset" not in res.columns:
        return res
    numeric = [c for c in res.columns
               if c not in ("dataset", "model", "encoder", "pooling", "device", "run_datetime")]
    res = res.copy()
    for c in numeric:
        res[c] = pd.to_numeric(res[c], errors="coerce")
    # "seed" itself must NOT be averaged - the mean of seeds 0,1,2 is 1, which means nothing. We keep
    # the first seed only so the column still exists, and report how many there were in n_seeds.
    agg = {c: "mean" if (c in numeric and c != "seed") else "first"
           for c in res.columns if c != "dataset"}
    out = res.groupby("dataset", as_index=False).agg(agg)
    counts = res.groupby("dataset")["seed"].nunique()
    out["n_seeds"] = out["dataset"].map(counts).astype(int)
    return out


def save_result_row(model_id: str, row: Dict, path: Optional[str] = None) -> bool:
    """
    Save one run's result into a model's results.csv, keyed by (dataset, seed).

    Rule: one row per dataset PER SEED, and a re-run of the same (dataset, seed) always replaces the
    old row. Different seeds live side by side, so `--seed 0/1/2` builds up three rows per dataset
    and the means become "average over 3 repeats".

    This used to keep only the HIGHEST-accuracy run per dataset. That was wrong for a paper: it
    turns every reported mean into a max-over-restarts number, so a model that happened to be re-run
    more often looks better than one that was run once, and repeats can never be measured because a
    worse seed is thrown away. Keeping every (dataset, seed) fixes both.

    The save is an "upsert": read the table, drop this (dataset, seed)'s old row, add the new one,
    sort it back into MILLET's order, and write the whole file out again.

    The write is atomic: we write a temporary file first and then rename it over the real one. A
    rename either happens completely or not at all, so an interrupted (or externally re-aligned)
    write can never leave a half-finished results.csv behind.

    model_id : which model's folder to write into.
    row : a results-row dict from score_model (a "run_datetime" is added if missing).
    path : write this exact file instead (overrides model_id).
    returns : True (the row is always saved now).
    """
    path = path or results_csv(model_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    row = dict(row)                                           # do not modify the caller's dict
    row.setdefault("run_datetime", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    row["seed"] = int(row.get("seed", 0) or 0)
    name, seed = row["dataset"], row["seed"]

    df = load_results(model_id, path=path)                    # what is already there
    new = pd.DataFrame([row])
    if len(df):                                               # drop only THIS (dataset, seed) row
        df = df[~((df["dataset"] == name) & (df["seed"] == seed))]
        df = pd.concat([df, new], ignore_index=True)
    else:
        df = new                                              # first row - concat would warn on empty
    df = df.reindex(columns=RESULT_COLUMNS)                   # fixed column order

    # keep the file in the same order the sweep runs in, so results.csv reads like the paper's table
    order = {n: i for i, n in enumerate(sweep_order())}
    df["_k"] = df["dataset"].map(lambda n: order.get(n, len(order)))
    df = df.sort_values(["_k", "dataset", "seed"]).drop(columns="_k").reset_index(drop=True)

    tmp = path + ".tmp"                                       # write, then rename (atomic)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

    mark_done(model_id, name, seed)                           # remember this seed is done
    return True


# --------------------------------------------------------------------------------------
# 4. Comparing one model's numbers to MILLET
# --------------------------------------------------------------------------------------
def _outcome(diff: float, band: float, lower_is_better: bool = False) -> str:
    """
    Turn a (ours - baseline) difference into "win" / "tie" / "loss".

    diff : ours minus baseline.
    band : how far from 0 still counts as a tie (a difference smaller than this is noise).
    lower_is_better : True for loss (being lower than MILLET is a win); False for accuracy / AOPCR.
    returns : "win", "tie", or "loss".
    """
    if pd.isna(diff):
        return "no_baseline"
    if abs(diff) <= band:
        return "tie"
    better = diff < 0 if lower_is_better else diff > 0
    return "win" if better else "loss"


def build_comparison(model_id: str, out: Optional[str] = None, verbose: bool = True) -> pd.DataFrame:
    """
    Join one model's UCR results to the MILLET baseline and write its comparison_vs_millet.csv.

    Every UCR dataset the model has a result for shows up, in MILLET's order:
      - the 85 the paper also reports -> millet_acc / millet_loss / millet_aopcr + win/tie/loss,
      - the ~43 the paper skipped     -> blank baseline and outcome "no_baseline".
    WebTraffic is left out on purpose (it is not a UCR dataset; `python main.py webtraffic` checks
    it against its own baseline, and it is the only one with NDCG).

    model_id : which model to compare, e.g. "seanet__sea_mstcn_sep__mil_additive".
    out : where to write the csv (defaults to that model's comparison_vs_millet.csv).
    verbose : if True, also print the win/tie/loss summary.
    returns : the comparison DataFrame (also saved to `out`).
    """
    out = out or comparison_csv(model_id)
    res = mean_over_seeds(load_results(model_id))             # several seeds -> their average
    ucr_set = set(UCR_128_DATASETS)
    res = res[res["dataset"].isin(ucr_set)].copy() if len(res) else res   # UCR only (drop WebTraffic)
    if res.empty:
        cmp = pd.DataFrame()
        if verbose:
            _print_comparison_summary(model_id, cmp, out)
        return cmp

    # the paper's numbers for each compared metric, keyed by dataset
    baselines = {metric: millet_baseline(csv_name)
                 for metric, (csv_name, _band, _low) in COMPARED_METRICS.items()}

    rows = []
    for _, r in res.iterrows():                               # go through each dataset we ran
        name = r["dataset"]
        entry = {"dataset": name}
        for metric, (_csv_name, band, lower_is_better) in COMPARED_METRICS.items():
            base = baselines[metric]
            ours = pd.to_numeric(r.get(f"test_{metric}"), errors="coerce")
            theirs = float(base[name]) if name in base.index else np.nan
            diff = ours - theirs                              # NaN when the paper has no number
            entry[f"ours_{metric}"] = round(float(ours), 4) if pd.notna(ours) else np.nan
            entry[f"millet_{metric}"] = round(theirs, 4) if pd.notna(theirs) else np.nan
            entry[f"{metric}_diff"] = round(float(diff), 4) if pd.notna(diff) else np.nan
            entry[f"{metric}_outcome"] = _outcome(diff, band, lower_is_better)
        entry["params"] = int(r["params"]) if pd.notna(r.get("params")) else np.nan
        entry["model_size_mb"] = r.get("model_size_mb")
        rows.append(entry)

    # sort so the paper's 85 come first (in the paper's order), then the rest alphabetically
    paper_order = {name: i for i, name in enumerate(millet_datasets())}
    cmp = pd.DataFrame(rows)
    cmp["_k"] = cmp["dataset"].map(lambda n: paper_order.get(n, len(paper_order)))
    cmp = cmp.sort_values(["_k", "dataset"]).drop(columns="_k").reset_index(drop=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmp.to_csv(out, index=False)

    if verbose:
        _print_comparison_summary(model_id, cmp, out)
    return cmp


def overlap_rows(cmp: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the datasets that MILLET also reports (the 85) - the ones we can compare fairly.

    cmp : a comparison frame from build_comparison.
    returns : the subset with a baseline (empty frame if there is none).
    """
    if cmp.empty or "acc_outcome" not in cmp.columns:
        return pd.DataFrame()
    return cmp[cmp["acc_outcome"] != "no_baseline"]


def summarise_model(model_id: str, cmp: Optional[pd.DataFrame] = None) -> Dict:
    """
    Build one model's headline numbers: the means over the 85 MILLET datasets (ours next to theirs),
    and the overall means over every dataset we trained.

    This is the table that answers "did we beat MILLET?". The two means are kept apart on purpose:
      - the "millet85_*" numbers are the fair head-to-head (same 85 datasets, ours vs theirs),
      - the "overall_*" numbers describe our model across everything we ran, which MILLET never
        reported, so there is nothing to compare them to.

    model_id : which model to summarise.
    cmp : its comparison frame (built if not given).
    returns : an ordered dict of metric name -> value.
    """
    res = mean_over_seeds(load_results(model_id))             # several seeds -> their average
    cmp = build_comparison(model_id, verbose=False) if cmp is None else cmp
    out: Dict = {"model": model_id}

    for col in ["test_acc", "test_loss", "test_aopcr", "params", "model_size_mb"]:
        if col in res.columns:
            res[col] = pd.to_numeric(res[col], errors="coerce")

    # --- the fair head-to-head: the 85 datasets MILLET published ---
    overlap = overlap_rows(cmp)
    out["millet85_datasets"] = int(len(overlap))
    if len(overlap):
        for metric in COMPARED_METRICS:                       # acc, loss, aopcr
            ours = overlap[f"ours_{metric}"].mean()
            theirs = overlap[f"millet_{metric}"].mean()
            out[f"mean_{metric}_ours"] = round(float(ours), 4)
            out[f"mean_{metric}_millet"] = round(float(theirs), 4)
            counts = overlap[f"{metric}_outcome"].value_counts()
            out[f"{metric}_win_tie_loss"] = (f"{counts.get('win', 0)}/{counts.get('tie', 0)}"
                                             f"/{counts.get('loss', 0)}")

    # --- our own overall picture: every UCR dataset we trained (no baseline exists for these) ---
    ucr = res[res["dataset"].isin(set(UCR_128_DATASETS))] if len(res) else res
    out["overall_ucr_datasets"] = int(len(ucr))
    if len(ucr):
        out["overall_mean_acc"] = round(float(ucr["test_acc"].mean()), 4)
        out["overall_mean_loss"] = round(float(ucr["test_loss"].mean()), 4)
        out["overall_mean_aopcr"] = round(float(ucr["test_aopcr"].mean()), 4)

    # --- WebTraffic: the only dataset with per-timestep ground truth, so the only NDCG ---
    web = res[res["dataset"] == WEB_TRAFFIC] if len(res) else res
    if len(web):
        out["webtraffic_acc"] = round(float(pd.to_numeric(web["test_acc"].iloc[0], errors="coerce")), 4)
        ndcg = pd.to_numeric(web["test_ndcg"].iloc[0], errors="coerce")
        if pd.notna(ndcg):
            out["webtraffic_ndcg"] = round(float(ndcg), 4)

    if len(res) and pd.notna(res["params"].iloc[0]):
        out["params"] = int(res["params"].iloc[0])
        out["model_size_mb"] = round(float(res["model_size_mb"].iloc[0]), 3)
    return out


def write_summary(model_id: str, perf: Optional[Dict] = None) -> tuple:
    """
    Write one model's headline numbers as summary.csv + summary.md inside its own folder.

    model_id : which model.
    perf : the dict from summarise_model (built if not given).
    returns : (csv path, md path).
    """
    perf = summarise_model(model_id) if perf is None else perf
    out_csv, out_md = summary_paths(model_id)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame([{"metric": k, "value": v} for k, v in perf.items()]).to_csv(out_csv, index=False)
    with open(out_md, "w") as f:
        f.write(f"# {model_id} - results summary\n\n| metric | value |\n|---|---|\n")
        for key, value in perf.items():
            f.write(f"| {key} | {value} |\n")
    return out_csv, out_md


def _print_comparison_summary(model_id: str, cmp: pd.DataFrame, out: str) -> None:
    """
    Print the win/tie/loss record and the means, over the datasets that have a MILLET baseline.

    model_id : the model being reported.
    cmp : the comparison DataFrame from build_comparison.
    out : the path the comparison was written to (just for the printed message).
    returns : nothing.
    """
    if cmp.empty:                                             # nothing run yet
        print(f"No UCR results for {model_id} yet - run "
              f"`python main.py train --model <config>` first (nothing to compare).")
        return
    overlap = overlap_rows(cmp)
    n_no_base = int(len(cmp) - len(overlap))
    print(f"=== {model_id} vs MILLET ConjunctiveInceptionTime (mean of 5 reps) ===")
    print(f"  UCR datasets evaluated: {len(cmp)}  "
          f"(with MILLET baseline: {len(overlap)}, without: {n_no_base})")
    if len(overlap):
        print(f"  {'metric':8s} {'ours':>9s} {'MILLET':>9s} {'win/tie/loss':>14s}   (mean over the "
              f"{len(overlap)} datasets MILLET published)")
        for metric, (_csv, _band, lower) in COMPARED_METRICS.items():
            counts = overlap[f"{metric}_outcome"].value_counts()
            wtl = f"{counts.get('win', 0)}/{counts.get('tie', 0)}/{counts.get('loss', 0)}"
            arrow = "(lower is better)" if lower else ""
            print(f"  {metric:8s} {overlap[f'ours_{metric}'].mean():>9.4f} "
                  f"{overlap[f'millet_{metric}'].mean():>9.4f} {wtl:>14s}   {arrow}")
    print(f"  wrote {out}")


# --------------------------------------------------------------------------------------
# 5. Comparing several models (the pooling heads) to each other
# --------------------------------------------------------------------------------------
def discover_models() -> List[str]:
    """
    List the models that have results, so they can be compared.

    A model is "there" if results/SEA_NET/<model_id>/results.csv exists. The shared folders
    (figures/, logs/) have no results.csv, so they are skipped automatically.

    returns : a sorted list of model ids (empty if nothing has been swept yet).
    """
    if not os.path.isdir(RESULTS_ROOT):
        return []
    return [name for name in sorted(os.listdir(RESULTS_ROOT))
            if os.path.exists(os.path.join(RESULTS_ROOT, name, RESULTS_FILE))]


def compare_models(models: Optional[List[str]] = None, out: str = MODEL_COMPARISON_CSV,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Build the head-to-head table ACROSS models: one row per model with its mean accuracy / loss /
    AOPCR over the 85 MILLET datasets, MILLET's own means for reference, and its win/tie/loss record.
    This is the "which encoder+pooling wins overall" table.

    It reuses summarise_model(...) per model (which also refreshes that model's own
    comparison_vs_millet.csv), so the per-model and cross-model numbers can never disagree.

    models : which models to include (default: every model with a results folder).
    out : where to write the table (default: results/SEA_NET/model_comparison.csv).
    verbose : if True, print the ranking.
    returns : the cross-model DataFrame (also saved to `out`), best mean accuracy first.
    """
    models = discover_models() if models is None else models
    rows = []
    for model_id in models:
        if load_results(model_id).empty:                      # nothing trained -> nothing to write
            continue
        cmp = build_comparison(model_id, verbose=False)       # also refreshes that model's csv
        perf = summarise_model(model_id, cmp=cmp)
        write_summary(model_id, perf)                         # keep each model's summary fresh too
        rows.append(perf)
    df = pd.DataFrame(rows)
    if not df.empty:
        sort_col = "mean_acc_ours" if "mean_acc_ours" in df.columns else "overall_mean_acc"
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False)
    if verbose:
        _print_model_comparison(df, out)
    return df


def _print_model_comparison(df: pd.DataFrame, out: str) -> None:
    """Print the cross-model ranking (best mean accuracy over the 85 MILLET datasets first)."""
    if df.empty:
        print("No model results yet - run `python main.py train --model <config>` first.")
        return
    if "mean_acc_ours" not in df.columns:
        print(f"{len(df)} model(s) have results, but none has finished a dataset MILLET published yet.")
        return
    # Only models that finished at least one MILLET-published dataset have head-to-head numbers. Models
    # we only ran on WebTraffic (the fast screen) have NaN there, so we leave them out of THIS table -
    # they are still saved in the CSV and drawn by `python main.py report`. (Without this filter the
    # print crashed trying to format a NaN win/tie/loss with the string format code.)
    ranked = df[df["mean_acc_ours"].notna()]
    n_skipped = len(df) - len(ranked)
    if ranked.empty:
        print(f"{len(df)} model(s) have results, but none has finished a dataset MILLET published yet "
              f"(run the full sweep: `python main.py train --model <config>`).")
        print(f"  wrote {out}")
        return
    millet_acc = ranked["mean_acc_millet"].dropna()
    note = f"   ({n_skipped} more have WebTraffic-only results, not shown here)" if n_skipped else ""
    print(f"Cross-model comparison over {len(ranked)} model(s), on the "
          f"{int(ranked['millet85_datasets'].max())} datasets MILLET published (best accuracy first):{note}")
    print(f"  {'model':32s} {'acc':>7s} {'loss':>7s} {'aopcr':>8s} {'acc W/T/L':>11s}")
    for _, r in ranked.iterrows():
        print(f"  {r['model']:32s} {r['mean_acc_ours']:>7.4f} {r['mean_loss_ours']:>7.4f} "
              f"{r['mean_aopcr_ours']:>8.3f} {str(r['acc_win_tie_loss']):>11s}")
    if len(millet_acc):                                       # the bar every model above must clear
        r = ranked.iloc[0]
        print(f"  {'MILLET (the baseline)':32s} {r['mean_acc_millet']:>7.4f} "
              f"{r['mean_loss_millet']:>7.4f} {r['mean_aopcr_millet']:>8.3f}")
    print(f"  wrote {out}")


# --------------------------------------------------------------------------------------
# 6. WebTraffic-only comparison (the fast screen: ONE dataset, with the paper baseline)
#
# WebTraffic is our headline dataset (it is the only one with per-timestep ground truth, so the only
# one with NDCG). This table ranks every model on WebTraffic alone, next to TWO baselines kept apart on
# purpose:
#   - "MILLET (paper)"  : the numbers the paper published (results/WebTraffic/InceptionTime/), and
#   - our own rerun of the same baseline model (millet/fcn/resnet in sv1), which usually scores a bit
#     lower - seeing the gap between the two is exactly what we want.
# --------------------------------------------------------------------------------------
WEBTRAFFIC_COMPARISON_CSV = os.path.join(RESULTS_ROOT, "webtraffic_comparison.csv")


def webtraffic_paper_baseline() -> Dict[str, float]:
    """
    The paper's published WebTraffic numbers (mean of MILLET's 5 Conjunctive reps).

    Read straight from results/WebTraffic/InceptionTime/ (we only ever READ these files). Kept SEPARATE
    from our own rerun of the same model, because our rerun usually scores a little lower.

    returns : {"acc":.., "loss":.., "aopcr":.., "ndcg":..}; a metric is skipped if its file is missing.
    """
    files = {"acc": "test_acc.csv", "loss": "test_loss.csv",
             "aopcr": "test_aopcr.csv", "ndcg": "test_ndcg.csv"}
    out: Dict[str, float] = {}
    for metric, fname in files.items():
        try:
            s = millet_baseline(fname, directory=MILLET_WEBTRAFFIC_DIR)
            out[metric] = round(float(s.get(WEB_TRAFFIC, s.iloc[0])), 4)
        except Exception:
            pass                                                 # file missing / unreadable -> just skip
    return out


def webtraffic_table(models: Optional[List[str]] = None) -> pd.DataFrame:
    """
    One row per model with its WebTraffic metrics (acc, loss, aopcr, ndcg, params), best accuracy first.

    models : which models to include (default: every model that has a WebTraffic result). Because it
             auto-discovers, the table (and its figure) grow by themselves as new models finish.
    returns : the table as a DataFrame (empty if nothing has a WebTraffic row yet).
    """
    models = discover_models() if models is None else models
    rows = []
    for model_id in models:
        res = mean_over_seeds(load_results(model_id))         # several seeds -> their average
        if res.empty:
            continue
        web = res[res["dataset"] == WEB_TRAFFIC]
        if web.empty:                                            # this model has no WebTraffic row yet
            continue
        r = web.iloc[0]
        num = lambda col: pd.to_numeric(r[col], errors="coerce") if col in web.columns else float("nan")
        rows.append({
            "model": model_id,
            "acc": num("test_acc"),
            "loss": num("test_loss"),
            "aopcr": num("test_aopcr"),
            "ndcg": num("test_ndcg"),
            "params": num("params"),
            "size_mb": num("model_size_mb"),   # weight-file size in MB (the "small model" goal)
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("acc", ascending=False).reset_index(drop=True)
    return df


def compare_webtraffic(models: Optional[List[str]] = None, out: str = WEBTRAFFIC_COMPARISON_CSV,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Build + save the WebTraffic-only comparison table, and print every model next to the paper baseline.

    out : where to write the CSV.
    verbose : print the ranking.
    returns : the table (also saved to `out`).
    """
    df = webtraffic_table(models)
    if df.empty:
        if verbose:
            print("No WebTraffic results yet - run e.g. `bash scripts/run_all.sh web`.")
        return df
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    if verbose:
        _print_webtraffic_comparison(df, out)
    return df


def _print_webtraffic_comparison(df: pd.DataFrame, out: str) -> None:
    """Print the WebTraffic ranking, then the paper's published MILLET number as the bar to beat."""
    paper = webtraffic_paper_baseline()
    print(f"WebTraffic comparison over {len(df)} model(s) (best accuracy first):")
    print(f"  {'model':44s} {'acc':>7s} {'loss':>7s} {'aopcr':>8s} {'ndcg':>7s} {'params':>9s}")
    for _, r in df.iterrows():
        ndcg = f"{r['ndcg']:>7.4f}" if pd.notna(r['ndcg']) else f"{'-':>7s}"
        params = f"{int(r['params']):>9d}" if pd.notna(r['params']) else f"{'-':>9s}"
        print(f"  {str(r['model']):44s} {r['acc']:>7.4f} {r['loss']:>7.4f} "
              f"{r['aopcr']:>8.3f} {ndcg} {params}")
    if paper:
        nd = f"{paper.get('ndcg'):>7.4f}" if 'ndcg' in paper else f"{'-':>7s}"
        print(f"  {'MILLET (paper, published)':44s} {paper.get('acc', float('nan')):>7.4f} "
              f"{paper.get('loss', float('nan')):>7.4f} {paper.get('aopcr', float('nan')):>8.3f} {nd}")
    print(f"  wrote {out}")


# --------------------------------------------------------------------------------------
# 7. The leaderboard: ONE table with every model, best WebTraffic accuracy first
#
# Why this table exists on top of the two above:
#   - webtraffic_comparison.csv ranks by WebTraffic but shows ONLY WebTraffic columns,
#   - model_comparison.csv has the UCR columns but ranks by UCR accuracy, so a model we only
#     screened on WebTraffic sinks to the bottom with blanks.
# We screen every model on WebTraffic first and only send the winners to the full 129-dataset
# sweep, so the question we actually ask is "who won the screen, and how did they then do on UCR?".
# The leaderboard answers exactly that: sorted by WebTraffic accuracy, UCR columns filled in for
# the models that went on to the full sweep and LEFT EMPTY for the ones that did not. Empty is
# information here - it means "screened only, not swept yet", not "missing data".
#
# It is REBUILT from scratch every time you run it, straight out of each model's own results.csv.
# So there is nothing to keep in sync by hand: a new model appears by itself, and a re-trained
# model's numbers are replaced by themselves.
# --------------------------------------------------------------------------------------
LEADERBOARD_CSV = os.path.join(RESULTS_ROOT, "leaderboard.csv")

# Old name -> new name. The two source tables use different words for the same thing
# ("mean_acc_ours", "overall_mean_acc", "acc"...), which is confusing when they sit side by side.
# Here every column gets a prefix that says WHERE the number comes from:
#   web_*     = the WebTraffic dataset alone (our fast screen; the only dataset with NDCG)
#   ucr85_*   = the 85 UCR datasets the MILLET paper published (the fair head-to-head)
#   ucr_all_* = every UCR dataset we trained (no paper baseline exists for these)
#   millet_*  = the paper's own means over the same 85, so the bar to beat is in the same row
_LEADERBOARD_RENAMES = {
    "acc": "web_acc", "loss": "web_loss", "aopcr": "web_aopcr", "ndcg": "web_ndcg",
    "millet85_datasets": "ucr85_n",
    "mean_acc_ours": "ucr85_acc", "mean_loss_ours": "ucr85_loss", "mean_aopcr_ours": "ucr85_aopcr",
    "acc_win_tie_loss": "ucr85_acc_wtl", "loss_win_tie_loss": "ucr85_loss_wtl",
    "aopcr_win_tie_loss": "ucr85_aopcr_wtl",
    "mean_acc_millet": "millet_acc", "mean_loss_millet": "millet_loss",
    "mean_aopcr_millet": "millet_aopcr",
    "overall_ucr_datasets": "ucr_all_n",
    "overall_mean_acc": "ucr_all_acc", "overall_mean_loss": "ucr_all_loss",
    "overall_mean_aopcr": "ucr_all_aopcr",
}

# The column order of the file. Reads left to right like the question we ask: who is it, how did it
# do on the screen, how big is it, then how did it do on the real benchmark, then the bar to beat.
LEADERBOARD_COLUMNS = [
    "rank", "model", "config", "encoder", "pooling", "origin",
    "web_acc", "web_loss", "web_aopcr", "web_ndcg",
    "params", "size_mb",
    "ucr85_n", "ucr85_acc", "ucr85_loss", "ucr85_aopcr",
    "ucr85_acc_wtl", "ucr85_loss_wtl", "ucr85_aopcr_wtl",
    "ucr_all_n", "ucr_all_acc", "ucr_all_loss", "ucr_all_aopcr",
    "millet_acc", "millet_loss", "millet_aopcr",
]


def origin_label(encoder: str, pooling: str) -> str:
    """
    Say in one word how much of a model is new: "millet", "ours" or "half-ours".

    Both halves carry their own origin tag ("sea_" = ours, "mil_" = MILLET's), so this just reads
    them. It is the column to sort by when writing the paper - it separates "the baseline we rerun",
    "our encoder with the paper's head" and "fully our model" without reading any name in detail.

    encoder : the encoder name out of the model id.
    pooling : the pooling name out of the model id.
    returns : "millet" | "ours" | "half-ours" (or "?" if a name carries no tag).
    """
    enc_ours, pool_ours = is_ours(encoder), is_ours(pooling)
    enc_mil, pool_mil = is_millet(encoder), is_millet(pooling)
    if enc_ours and pool_ours:
        return "ours"
    if enc_mil and pool_mil:
        return "millet"
    if (enc_ours and pool_mil) or (enc_mil and pool_ours):
        return "half-ours"
    return "?"                                                 # an untagged name (should not happen)


def build_leaderboard(models: Optional[List[str]] = None, out: str = LEADERBOARD_CSV,
                      refresh: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Build the one-file leaderboard: every model, best WebTraffic accuracy first, UCR columns
    filled in where the full sweep has finished and left empty where it has not.

    How it works (it does not compute anything new - it joins the two tables we already build):
      1. webtraffic_table() gives the WebTraffic row of every model,
      2. compare_models() gives the UCR head-to-head numbers of every model,
      3. we join them on the model name with an OUTER join, so a model that appears in only one of
         the two still gets a row (its other columns simply stay empty),
      4. sort by WebTraffic accuracy, highest first. A model with no WebTraffic result yet goes to
         the bottom (na_position="last") instead of being dropped.

    Why an OUTER join and not an inner one: an inner join would keep only models present in BOTH
    tables, which would silently hide every WebTraffic-only model - exactly the ones we are trying
    to screen. Empty cells are the honest answer.

    models : which models to include (default: every model that has a results.csv).
    out : where to write the table (default: results/SEA_NET/leaderboard.csv).
    refresh : True = recompute each model's UCR comparison first (slower, always correct).
              False = reuse the model_comparison.csv already on disk (fast, for a quick re-look).
    verbose : print the table after writing it.
    returns : the leaderboard as a DataFrame (also saved to `out`).
    """
    web = webtraffic_table(models)                              # model, acc, loss, aopcr, ndcg, params, size_mb

    # the UCR side. Refreshing also rewrites model_comparison.csv + each model's summary, so the
    # leaderboard can never disagree with the other tables.
    if refresh:
        full = compare_models(models, verbose=False)
    elif os.path.exists(MODEL_COMPARISON_CSV):
        full = read_our_csv(MODEL_COMPARISON_CSV)               # tolerant read (the csv-padding tool)
    else:
        full = pd.DataFrame()

    if web.empty and full.empty:
        if verbose:
            print("No results yet - run e.g. `bash scripts/run_all.sh web` first.")
        return pd.DataFrame()

    # params / size_mb are in BOTH tables and identical, so keep one copy (the WebTraffic one).
    if not full.empty:
        full = full.drop(columns=[c for c in ("params", "model_size_mb") if c in full.columns])

    # join the two sides on the model name. "outer" = keep every model from either side.
    if web.empty:
        df = full.copy()
    elif full.empty:
        df = web.copy()
    else:
        df = web.merge(full, on="model", how="outer")

    df = df.rename(columns=_LEADERBOARD_RENAMES)

    # Split the model id into its three parts and store them as their own columns. A model id is
    # "<config>__<encoder>__<pooling>", so this is exact (see seanet/config.split_model_id).
    # "origin" then says in one word how new the model is, which is the column you sort by when
    # writing the paper: is this MILLET's model, half ours, or fully ours?
    parts = df["model"].astype(str).apply(split_model_id)
    df["config"] = parts.str[0]
    df["encoder"] = parts.str[1]
    df["pooling"] = parts.str[2]
    df["origin"] = [origin_label(e, p) for e, p in zip(df["encoder"], df["pooling"])]

    # sort by the screen result, best first; models with no WebTraffic row sink to the bottom.
    if "web_acc" in df.columns:
        df = df.sort_values("web_acc", ascending=False, na_position="last")
    df = df.reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    df = _round_leaderboard(df)
    # keep our column order, and quietly ignore any column that does not exist yet
    df = df[[c for c in LEADERBOARD_COLUMNS if c in df.columns]]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    if verbose:
        _print_leaderboard(df, out)
    return df


def _round_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the numbers readable: 4 decimals for metrics, 3 for the file size, whole numbers for counts.

    df : the joined leaderboard frame.
    returns : the same frame with its numeric columns rounded (unknown columns are left alone).
    """
    four = ["web_acc", "web_loss", "web_aopcr", "web_ndcg",
            "ucr85_acc", "ucr85_loss", "ucr85_aopcr",
            "ucr_all_acc", "ucr_all_loss", "ucr_all_aopcr",
            "millet_acc", "millet_loss", "millet_aopcr"]
    for col in four:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
    if "size_mb" in df.columns:
        df["size_mb"] = pd.to_numeric(df["size_mb"], errors="coerce").round(3)
    # counts stay whole numbers. Int64 (capital I) is pandas' integer type that ALLOWS empty cells -
    # plain int cannot hold a blank, and a plain float would print "128.0" instead of "128".
    for col in ["params", "ucr85_n", "ucr_all_n"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def _print_leaderboard(df: pd.DataFrame, out: str) -> None:
    """
    Print the leaderboard as a plain text table, then the paper's WebTraffic number as the bar to beat.

    A dash means "not run yet" - it is never a zero.

    df : the finished leaderboard frame.
    out : the path it was written to (just for the message).
    returns : nothing.
    """
    def cell(value, width: int, decimals: int = 4) -> str:
        """One number as text, or a dash when it is empty."""
        if pd.isna(value):
            return f"{'-':>{width}s}"
        if decimals == 0:
            return f"{int(value):>{width}d}"
        return f"{float(value):>{width}.{decimals}f}"

    n_swept = int(df["ucr85_n"].notna().sum()) if "ucr85_n" in df.columns else 0
    print(f"Leaderboard: {len(df)} model(s), best WebTraffic accuracy first "
          f"({n_swept} of them also finished the full UCR sweep; '-' means not run yet).")
    print(f"  {'#':>3s} {'model':56s} {'web_acc':>8s} {'web_ndcg':>9s} {'params':>8s} "
          f"{'ucr85_n':>8s} {'ucr85_acc':>10s} {'ucr85_aopcr':>12s}")
    for _, r in df.iterrows():
        print(f"  {int(r['rank']):>3d} {str(r['model'])[:56]:56s} "
              f"{cell(r.get('web_acc'), 8)} {cell(r.get('web_ndcg'), 9)} "
              f"{cell(r.get('params'), 8, 0)} {cell(r.get('ucr85_n'), 8, 0)} "
              f"{cell(r.get('ucr85_acc'), 10)} {cell(r.get('ucr85_aopcr'), 12)}")
    # the bar to beat, on the same two datasets. Both numbers come from files, never hardcoded:
    # the WebTraffic ones from results/WebTraffic/InceptionTime/, the UCR ones from the millet_*
    # columns we just joined in (they are the paper's means, so the same on every row).
    paper = webtraffic_paper_baseline()
    p_acc = df["millet_acc"].dropna() if "millet_acc" in df.columns else pd.Series(dtype=float)
    p_aopcr = df["millet_aopcr"].dropna() if "millet_aopcr" in df.columns else pd.Series(dtype=float)
    if paper or len(p_acc):
        print(f"  {'':>3s} {'MILLET (paper, published)':56s} "
              f"{cell(paper.get('acc', float('nan')), 8)} {cell(paper.get('ndcg', float('nan')), 9)} "
              f"{'-':>8s} {'-':>8s} "
              f"{cell(p_acc.iloc[0] if len(p_acc) else float('nan'), 10)} "
              f"{cell(p_aopcr.iloc[0] if len(p_aopcr) else float('nan'), 12)}")
    print(f"  wrote {out}")
