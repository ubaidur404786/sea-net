"""
seanet/results.py - saving results, resuming a sweep, and comparing to MILLET.

What this file is for:
    Three jobs, all just pandas + csv:
      1. Remember what has finished, PER MODEL AND PER SETTING, so a long run can stop and resume
         without redoing work - and so a different model (or the same model with new hyperparameters)
         is not wrongly skipped just because another model already ran that dataset.
      2. Store each finished (model, settings, dataset) run's numbers, stamped with the date/time.
      3. Compare our numbers to the MILLET paper's, dataset by dataset, and say where we win/tie/lose.

What counts as "already done" (the resume key):
    A run is identified by THREE things together:  model | settings | dataset
      - model    : the config name, e.g. "seanet", "seanet_acp", "millet".
      - settings : an 8-character fingerprint of the encoder / pooling / training blocks + seed +
                   preprocessing (see seanet/config.settings_fingerprint).
      - dataset  : e.g. "Coffee".
    So: change a hyperparameter -> the fingerprint changes -> that model retrains on every dataset
    (its old rows stay as history). Run a different model -> it trains on every dataset even though
    "seanet" already did them. Re-run the same model + settings -> finished datasets are skipped.

Files it writes (all under results/SEA_NET/):
    - results.csv               : one row per finished (model, settings, dataset) (append-only).
    - done.txt                  : plain-text list of finished "model|settings|dataset" keys.
    - best_results.csv          : ONE row per dataset - the best model for it, and when it ran.
    - comparison_vs_millet.csv  : our numbers next to MILLET's, with win/tie/loss columns.
    - runs/<datetime>_<model>_<command>/ : a per-invocation folder holding just that run's rows.
    - archive/                  : one-time backups taken before the schema migration (see below).

Files it reads:
    - results.csv / done.txt    : our own outputs.
    - results/UCR/InceptionTime/test_acc.csv and test_aopcr.csv : the MILLET paper's published
      numbers (already in the repo). We use the Conjunctive-InceptionTime model as the baseline.

Related files:
    - seanet/config.py  -> settings_fingerprint() builds the "settings" id used as part of the key.
    - main.py           -> calls result_exists() to skip finished runs and save_result_row() after each.
    - seanet/report.py  -> reads these CSVs to draw the figures and the summary table.

Note on this machine: some tool keeps re-aligning .csv files (padding columns with spaces), which
once corrupted results.csv mid-run and made the sweep restart from scratch. To be safe:
  - the "what is done" list is a plain .txt file (done.txt), which the tool leaves alone, and
  - results.csv is written append-only (never read-then-rewritten), so it can't be corrupted mid-write.
The ONLY exception is the one-time migration below, which backs the file up first.
"""
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from seanet.data import UCR_128_DATASETS, read_our_csv

# --------------------------------------------------------------------------------------
# Paths + column order
# --------------------------------------------------------------------------------------
RESULTS_DIR = os.path.join("results", "SEA_NET")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")            # our metrics, one row per run
COMPARISON_CSV = os.path.join(RESULTS_DIR, "comparison_vs_millet.csv")   # our numbers vs MILLET
BEST_CSV = os.path.join(RESULTS_DIR, "best_results.csv")          # best model per dataset (+ when)
DONE_TXT = os.path.join(RESULTS_DIR, "done.txt")                  # finished "model|settings|dataset" keys
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")                      # one folder per invocation (datetime)
ARCHIVE_DIR = os.path.join(RESULTS_DIR, "archive")                # pre-migration backups

# Where the MILLET paper's published numbers live (one column per model and repetition).
MILLET_UCR_DIR = os.path.join("results", "UCR", "InceptionTime")
BASELINE_MODEL = "ConjunctiveInceptionTime"   # the MILLET model we compare against (it has 5 reps)

# The column order for results.csv, so the file is always laid out the same way.
# "settings" and "run_at" are what make a row traceable: which recipe produced it, and when.
RESULT_COLUMNS = [
    "dataset", "model", "settings", "run_at", "seed", "device", "params", "model_size_mb",
    "n_train", "n_val", "n_test", "series_length", "n_classes", "lambda_entropy",
    "test_acc", "test_bal_acc", "test_auroc", "test_loss", "test_aopcr", "test_ndcg",
    "train_time_s",
]

# The settings tag given to the 129 rows that were trained before this file understood settings.
# They are kept as history and still show up in the reports, but because no live config can ever
# fingerprint to the word "legacy", they never satisfy the resume check -> everything retrains clean.
LEGACY_SETTINGS = "legacy"

# How close counts as a "tie" (anything smaller than this is not a real win or loss).
ACC_TIE_BAND = 0.005    # 0.5% for accuracy
AOPCR_TIE_BAND = 0.1    # AOPCR is on a ~0..15 scale, so 0.1 is a small band


def _now() -> str:
    """The timestamp written into every results row, e.g. '2026-07-15 14:30:12'."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------------------
# 0. One-time migration to the model+settings schema
#
# The old results.csv/done.txt had no idea which settings produced a row, and done.txt keyed on the
# dataset name alone (so a second model was skipped because SEA-Net had already run that dataset).
# This upgrades both, ONCE, keeping the originals in archive/ first. Old rows are tagged
# settings="legacy": they stay visible as history, but they no longer block a retrain.
# --------------------------------------------------------------------------------------
_MIGRATED = False


def _archive(path: str) -> Optional[str]:
    """Copy a file into archive/ with a timestamp, so a migration can never lose data."""
    if not os.path.exists(path):
        return None
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base, ext = os.path.splitext(os.path.basename(path))
    dest = os.path.join(ARCHIVE_DIR, f"{base}_pre-migration_{stamp}{ext}")
    shutil.copy2(path, dest)
    return dest


def ensure_migrated(verbose: bool = True) -> None:
    """
    Bring results.csv + done.txt up to the model+settings schema, once per process.

    Safe to call as often as you like: it checks whether the files already have the new shape and
    returns immediately if so. Both files are backed up into archive/ before anything is rewritten.

    verbose : print what was migrated (only ever prints on the one run that does the work).
    returns : nothing.
    """
    global _MIGRATED
    if _MIGRATED:
        return
    _MIGRATED = True

    # --- results.csv: add the "settings" / "run_at" columns to any pre-existing rows ---
    if os.path.exists(RESULTS_CSV):
        df = read_our_csv(RESULTS_CSV)
        if "settings" not in df.columns:
            backup = _archive(RESULTS_CSV)
            # the old rows all came from the default SEA-Net recipe; date them from the file itself
            # so the history keeps a truthful (if approximate) "when did this run" stamp.
            mtime = datetime.fromtimestamp(os.path.getmtime(RESULTS_CSV)).strftime("%Y-%m-%d %H:%M:%S")
            df["settings"] = LEGACY_SETTINGS
            df["run_at"] = mtime
            df.reindex(columns=RESULT_COLUMNS).to_csv(RESULTS_CSV, index=False)
            if verbose:
                print(f"  [results] migrated {len(df)} existing rows to the model+settings schema "
                      f"(tagged settings='{LEGACY_SETTINGS}'; backup -> {backup})")

    # --- done.txt: old lines are bare dataset names; the new ones are "model|settings|dataset" ---
    if os.path.exists(DONE_TXT):
        with open(DONE_TXT) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        legacy = [ln for ln in lines if "|" not in ln]
        if legacy:
            backup = _archive(DONE_TXT)
            keyed = [ln for ln in lines if "|" in ln]        # keep any already-new keys
            with open(DONE_TXT, "w") as f:
                for line in keyed:
                    f.write(line + "\n")
            if verbose:
                print(f"  [results] dropped {len(legacy)} un-keyed done.txt entries so every model "
                      f"retrains under its own settings id (backup -> {backup})")


# --------------------------------------------------------------------------------------
# 1. Remembering what is done (per model + settings + dataset) + saving results
# --------------------------------------------------------------------------------------
def done_key(model: str, settings: str, dataset: str) -> str:
    """
    Build the one string that identifies a finished run.

    model : the config name, e.g. "seanet".
    settings : the 8-char fingerprint from seanet.config.settings_fingerprint.
    dataset : e.g. "Coffee".
    returns : e.g. "seanet|9eb2ac03|Coffee".
    """
    return f"{model}|{settings}|{dataset}"


def load_done() -> set:
    """
    Read done.txt into a set of "model|settings|dataset" keys.

    returns : set of finished keys (empty set if done.txt does not exist yet).
    """
    ensure_migrated()
    if not os.path.exists(DONE_TXT):
        return set()
    with open(DONE_TXT) as f:
        return {line.strip() for line in f if line.strip()}   # one key per line, ignore blanks


def mark_done(model: str, settings: str, dataset: str) -> None:
    """
    Add a finished run's key to done.txt (only if it is not already there).

    model, settings, dataset : the three parts of the resume key.
    returns : nothing.
    """
    os.makedirs(os.path.dirname(DONE_TXT), exist_ok=True)
    key = done_key(model, settings, dataset)
    if key not in load_done():                                # avoid writing the same key twice
        with open(DONE_TXT, "a") as f:
            f.write(key + "\n")


def result_exists(dataset: str, model: Optional[str] = None, settings: Optional[str] = None) -> bool:
    """
    Say whether THIS model, with THESE settings, has already finished THIS dataset.

    We check done.txt (a plain-text file the CSV aligner cannot corrupt), not results.csv.

    dataset : dataset name.
    model : the config name, e.g. "seanet". If None, the answer is always False - an unidentified
            run has no key, so it can never match a finished one (we would rather retrain than
            wrongly skip).
    settings : the settings fingerprint. If None, same reasoning as model.
    returns : True only if the exact model|settings|dataset key is in done.txt.
    """
    if model is None or settings is None:
        return False
    return done_key(model, settings, dataset) in load_done()


def load_results(path: str = RESULTS_CSV) -> pd.DataFrame:
    """
    Read results.csv into a DataFrame.

    Because results.csv is append-only, the same run could appear more than once (if it was re-run),
    so we keep only the last row for each (dataset, model, settings) - the newest wins. Rows for
    different models or different settings are all kept side by side; that is the point of the file.

    path : the results csv file.
    returns : a DataFrame of results (empty but correctly-columned if the file is missing).
    """
    ensure_migrated()
    if not os.path.exists(path):
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = read_our_csv(path)                                   # tolerant read (handles aligned csv)
    keys = [c for c in ("dataset", "model", "settings") if c in df.columns]
    if keys:
        df = df.drop_duplicates(keys, keep="last").reset_index(drop=True)   # last row wins
    return df


def new_run_dir(command: str, model: str) -> str:
    """
    Make a fresh, datetime-stamped folder for ONE invocation's outputs.

    Every command that trains something gets its own folder, e.g.
        results/SEA_NET/runs/2026-07-15_14-30-12_seanet_train/
    so you can always point at "the run I did on Tuesday afternoon" instead of digging through the
    master results.csv. The master file still gets every row too; this is an extra, per-run copy.

    command : the main.py command that is running ("train", "run", "single", ...).
    model : the model config name, e.g. "seanet".
    returns : the created folder's path.
    """
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(RUNS_DIR, f"{stamp}_{model}_{command}")
    os.makedirs(path, exist_ok=True)
    return path


def save_result_row(row: Dict, path: str = RESULTS_CSV, run_dir: Optional[str] = None) -> None:
    """
    Append one finished run to results.csv, record its key in done.txt, and (optionally) copy the
    row into this invocation's own run folder.

    We append instead of rewriting the whole file, so the CSV-aligning tool cannot corrupt it while
    we write. If a run ends up with more than one row (from a re-run), load_results keeps the last.

    row : a results-row dict from score_model. Must carry "dataset"; "model" and "settings" are
          expected (they are the resume key). "run_at" is stamped here if the caller did not.
    path : the results csv file.
    run_dir : optional per-invocation folder (from new_run_dir) to also write the row into.
    returns : nothing.
    """
    ensure_migrated()
    row = dict(row)
    row.setdefault("run_at", _now())                          # stamp the row if it is not stamped yet
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([row]).reindex(columns=RESULT_COLUMNS)  # put columns in the fixed order
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)   # append (header if new)

    if run_dir is not None:                                   # this invocation's own copy
        run_csv = os.path.join(run_dir, "results.csv")
        df.to_csv(run_csv, mode="a", header=not os.path.exists(run_csv), index=False)

    model, settings = row.get("model"), row.get("settings")
    if model and settings:                                    # remember it is done (per model+settings)
        mark_done(model, settings, row["dataset"])


# --------------------------------------------------------------------------------------
# 2. best_results.csv - one row per dataset: which model won it, and when
# --------------------------------------------------------------------------------------
def build_best_results(results_path: str = RESULTS_CSV, out: str = BEST_CSV,
                       verbose: bool = False) -> pd.DataFrame:
    """
    Reduce results.csv to ONE row per dataset: the model that scored best on it.

    This is the "which model should I actually use, and when did I prove it" tracking table. It reads
    every model's rows and picks, per dataset, the highest test_acc (ties broken by the lower
    test_loss, i.e. the more confident model). The winning row keeps its model, settings and run_at,
    so a result is always traceable back to the recipe and the day that produced it.

    results_path : the results csv to read.
    out : where to write best_results.csv.
    verbose : print a one-line confirmation.
    returns : the best-per-dataset DataFrame (also saved to `out`).
    """
    res = load_results(results_path)
    if res.empty:
        return pd.DataFrame()

    res = res.copy()
    for col in ("test_acc", "test_loss"):                     # the sort keys must be numeric
        if col in res.columns:
            res[col] = pd.to_numeric(res[col], errors="coerce")

    # highest accuracy first; if two models tie on accuracy, the lower test_loss wins
    res = res.sort_values(["test_acc", "test_loss"], ascending=[False, True], na_position="last")
    best = res.drop_duplicates("dataset", keep="first").copy()

    n_models = res.groupby("dataset")["model"].nunique()      # how many models were compared per dataset
    best["models_compared"] = best["dataset"].map(n_models)
    best = best.rename(columns={"model": "best_model"})

    columns = ["dataset", "best_model", "settings", "run_at", "models_compared",
               "test_acc", "test_bal_acc", "test_auroc", "test_loss", "test_aopcr", "test_ndcg",
               "params", "model_size_mb", "train_time_s"]
    best = best.reindex(columns=[c for c in columns if c in best.columns])
    best = best.sort_values("dataset").reset_index(drop=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    best.to_csv(out, index=False)
    if verbose:
        wins = best["best_model"].value_counts().to_dict()
        print(f"  best_results: {len(best)} datasets | wins per model: {wins}")
        print(f"  wrote {out}")
    return best


# --------------------------------------------------------------------------------------
# 3. Comparing our numbers to MILLET
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


def _rows_to_compare(results_path: str, model: Optional[str]) -> pd.DataFrame:
    """
    Choose which of our rows go into the MILLET comparison: one row per dataset.

    model : a model config name -> compare exactly that model (an honest single-model table, which
            is what a paper reports). None -> compare the best model per dataset (from
            build_best_results), and the table then carries a "model" column saying which one won,
            so a mixed table can never be mistaken for a single model's numbers.
    returns : a DataFrame with a "model" column and one row per dataset.
    """
    if model is not None:
        res = load_results(results_path)
        res = res[res["model"] == model].copy()
        # a model may have rows under several settings ids (e.g. before and after tuning);
        # keep its newest row per dataset so the table reflects the current recipe.
        if "run_at" in res.columns:
            res = res.sort_values("run_at").drop_duplicates("dataset", keep="last")
        return res
    best = build_best_results(results_path, verbose=False)
    if best.empty:
        return best
    return best.rename(columns={"best_model": "model"})


def build_comparison(results_path: str = RESULTS_CSV, out: str = COMPARISON_CSV,
                     verbose: bool = True, model: Optional[str] = None) -> pd.DataFrame:
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
    model : compare only this model's rows (recommended for a paper table). None = best model per
            dataset; the "model" column then says which model each row came from.
    returns : the comparison DataFrame (also saved to `out`).
    """
    res = _rows_to_compare(results_path, model)
    if res.empty:
        if verbose:
            print("No results yet - run the sweep first (nothing to compare).")
        return pd.DataFrame()

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
            "model": r.get("model"),
            "settings": r.get("settings"),
            "run_at": r.get("run_at"),
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
        _print_comparison_summary(cmp, out, model)
    return cmp


def _print_comparison_summary(cmp: pd.DataFrame, out: str, model: Optional[str] = None) -> None:
    """
    Print a short win/tie/loss summary over the datasets that have a MILLET baseline.

    cmp : the comparison DataFrame from build_comparison.
    out : the path the comparison was written to (just for the printed message).
    model : the model filter that was used (None = best-per-dataset), so the print says what it is.
    returns : nothing.
    """
    if cmp.empty:                                             # nothing run yet
        print("No UCR results yet - run the sweep first (nothing to compare).")
        return
    which = f"model '{model}'" if model else "best model per dataset"
    overlap = cmp[cmp["acc_outcome"] != "no_baseline"]        # only the datasets with a baseline
    n_no_base = int((cmp["acc_outcome"] == "no_baseline").sum())
    print(f"Comparing: {which}")
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


def sweep_status(model: str, settings: str, datasets: List[str]) -> Dict:
    """
    Say how far a given model+settings has got through a list of datasets.

    Used by the sweep to print "resuming: 41 done, 88 to go" before it starts, so a long run is
    never a mystery.

    model, settings : the recipe being swept.
    datasets : the full list of dataset names the sweep intends to cover.
    returns : {"done": [...], "todo": [...]}.
    """
    done = load_done()
    finished = [d for d in datasets if done_key(model, settings, d) in done]
    todo = [d for d in datasets if done_key(model, settings, d) not in done]
    return {"done": finished, "todo": todo}
