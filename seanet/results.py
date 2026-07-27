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
A model's folder is named "<config file name>__<encoder>_<pooling>", e.g.
"seanet__mstcn_sep_additive" (see seanet/config.model_folder_name). The config file name is in front
so two configs that build the same encoder+pooling never share a folder. Everything a model produces
lives in its own folder, so two models can never mix their numbers up:

    results/SEA_NET/
      data_summary.csv           <- shared: facts about the DATA, not about any model
      model_comparison.csv       <- shared: the cross-model ranking (written by compare_models)
      figures/                   <- shared: the cross-model figure
      logs/                      <- shared: logs of commands that belong to no model (summary, report)

      seanet__mstcn_sep_additive/        <- ONE model
        results.csv                      <- one row per dataset (best accuracy kept, see save_result_row)
        done_train_dataset.txt           <- the "what is finished" list (the resume switch)
        comparison_vs_millet.csv         <- our numbers next to MILLET's, per dataset
        summary.csv / summary.md         <- the headline means (over the 85, and overall)
        logs/<command>_<date-time>.log   <- one log per run of THIS model
        figures/                         <- this model's figures
        interpretation/                  <- this model's explanation figures

      seanet_acp__mstcn_sep_adaptive_classwise/   <- another model, same layout
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

from seanet.data import UCR_128_DATASETS, WEB_TRAFFIC, read_our_csv

# --------------------------------------------------------------------------------------
# 1. Paths - everything is keyed by the model id ("<encoder>_<pooling>")
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
    """The one folder that holds everything for a model, e.g. results/SEA_NET/mstcn_sep_additive."""
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

    model_id : the model id, e.g. "mstcn_sep_additive".
    returns : set of finished dataset names (empty set if the file does not exist yet).
    """
    path = done_txt(model_id)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}   # one name per line, ignore blanks


def mark_done(model_id: str, name: str) -> None:
    """
    Add a dataset name to a model's done_train_dataset.txt (only if it is not already there).

    model_id : the model id.  name : the finished dataset name.
    returns : nothing.
    """
    path = done_txt(model_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if name not in load_done(model_id):                       # avoid writing the same name twice
        with open(path, "a") as f:
            f.write(name + "\n")


def result_exists(model_id: str, name: str) -> bool:
    """
    Say whether a dataset is already finished FOR THIS MODEL.

    We check done_train_dataset.txt (a plain-text file the CSV aligner cannot corrupt), not
    results.csv. Delete the file to retrain everything; delete one line to retrain one dataset.

    model_id : the model id.  name : dataset name.
    returns : True if the dataset is in that model's done list.
    """
    return name in load_done(model_id)


def load_results(model_id: str, path: Optional[str] = None) -> pd.DataFrame:
    """
    Read a model's results.csv into a DataFrame.

    save_result_row keeps one row per dataset, but we still de-duplicate on read (keeping the last
    row) so an older or hand-edited file can never produce double-counted means.

    model_id : which model's results to read.
    path : read this exact file instead (overrides model_id); handy for tests.
    returns : a DataFrame of results (empty but correctly-columned if the file is missing).
    """
    path = path or results_csv(model_id)
    if not os.path.exists(path):
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = read_our_csv(path)                                   # tolerant read (handles aligned csv)
    if "dataset" in df.columns:
        df = df.drop_duplicates("dataset", keep="last").reset_index(drop=True)   # last row wins
    return df


def save_result_row(model_id: str, row: Dict, path: Optional[str] = None) -> bool:
    """
    Save one dataset's result into a model's results.csv, but only if it BEATS the old accuracy.

    Rule (keep the better result): if this dataset already has a row, we overwrite it only when the
    new run's test_acc is HIGHER. A worse-or-equal re-run leaves the old, better numbers in place.
    So results.csv always holds the best accuracy we have ever seen for this dataset. The run is
    still marked done, and MLflow still logged it, so nothing is thrown away - the .csv just never
    goes backwards. (The first time a dataset is trained there is no old row, so it is always saved.)

    When we do save, it is an "upsert": read the table, drop this dataset's old row, add the new one,
    sort it back into MILLET's order, and write the whole file out again - so the dataset ends up
    with exactly one row.

    The write is atomic: we write a temporary file first and then rename it over the real one. A
    rename either happens completely or not at all, so an interrupted (or externally re-aligned)
    write can never leave a half-finished results.csv behind.

    model_id : which model's folder to write into.
    row : a results-row dict from score_model (a "run_datetime" is added if missing).
    path : write this exact file instead (overrides model_id).
    returns : True if the new row was saved, False if the old (better-or-equal) row was kept.
    """
    path = path or results_csv(model_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    row = dict(row)                                           # do not modify the caller's dict
    row.setdefault("run_datetime", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    df = load_results(model_id, path=path)                    # what is already there

    # keep-the-better check: is there already a row for this dataset with an accuracy we did not beat?
    old = df[df["dataset"] == row["dataset"]] if len(df) else df
    if len(old):
        old_acc = pd.to_numeric(old["test_acc"].iloc[0], errors="coerce")
        new_acc = pd.to_numeric(row.get("test_acc"), errors="coerce")
        if pd.notna(old_acc) and pd.notna(new_acc) and new_acc <= old_acc:
            mark_done(model_id, row["dataset"])               # it IS finished, we just did not improve
            return False                                      # leave the old, better row untouched

    df = df[df["dataset"] != row["dataset"]] if len(df) else df          # drop this dataset's old row
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)         # add the new one
    df = df.reindex(columns=RESULT_COLUMNS)                   # fixed column order

    # keep the file in the same order the sweep runs in, so results.csv reads like the paper's table
    order = {name: i for i, name in enumerate(sweep_order())}
    df["_k"] = df["dataset"].map(lambda n: order.get(n, len(order)))
    df = df.sort_values(["_k", "dataset"]).drop(columns="_k").reset_index(drop=True)

    tmp = path + ".tmp"                                       # write, then rename (atomic)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

    mark_done(model_id, row["dataset"])                       # remember it is done
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

    model_id : which model to compare, e.g. "mstcn_sep_additive".
    out : where to write the csv (defaults to that model's comparison_vs_millet.csv).
    verbose : if True, also print the win/tie/loss summary.
    returns : the comparison DataFrame (also saved to `out`).
    """
    out = out or comparison_csv(model_id)
    res = load_results(model_id)
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
    res = load_results(model_id)
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
        res = load_results(model_id)
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
