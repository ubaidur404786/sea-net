"""
seanet/analysis/data.py - load every number the paper figures need, once.

Why one loading file
--------------------
The figures must never disagree with the tables, and two figures must never disagree with each
other. The safest way to guarantee that is to have ONE place that reads the result files and hands
back plain DataFrames. Every figure then draws from the same objects. Nothing in this package reads
a csv directly.

What it loads
-------------
    leaderboard()   : one row per model - WebTraffic metrics, UCR means, size (57 models today)
    matrix()        : the per-dataset table, model x dataset -> accuracy / loss / AOPCR
                      (only the models that finished the FULL 129-dataset sweep - 16 today)
    millet_matrix() : the same shape, for the MILLET paper's published numbers (85 datasets)
    profile()       : speed / FLOPs / memory per model (from scripts/profile_models.py; may be empty)
    paper_baseline(): MILLET's published WebTraffic numbers

An honesty rule
---------------
A model that was only screened on WebTraffic has NO UCR numbers. We never fill that in with a zero
or a guess - it stays missing, and every figure that needs it simply leaves that model out and says
so in its caption. `swept_models()` is the list of models that are allowed into those figures.
"""
import os
from typing import Dict, List, Optional

import pandas as pd

from seanet import results as R
from seanet.config import split_model_id
from seanet.data import UCR_128_DATASETS, WEB_TRAFFIC, read_our_csv

PROFILE_CSV = os.path.join(R.RESULTS_ROOT, "profile.csv")

# a model needs at least this many UCR datasets before it may enter the statistical figures.
# Below this the mean and the rank are too noisy to be worth printing.
MIN_DATASETS_FOR_STATS = 80


def leaderboard(refresh: bool = False) -> pd.DataFrame:
    """
    One row per model: WebTraffic metrics, UCR means, params, size, origin.

    refresh : True recomputes it from every model's results.csv (slower, always right);
              False reuses results/top_results/SEA_NET/leaderboard.csv if it exists.
    """
    if not refresh and os.path.exists(R.LEADERBOARD_CSV):
        df = read_our_csv(R.LEADERBOARD_CSV)
        numeric = [c for c in df.columns if c not in
                   ("model", "config", "encoder", "pooling", "origin",
                    "ucr85_acc_wtl", "ucr85_loss_wtl", "ucr85_aopcr_wtl")]
        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return _ensure_parts(df)
    return _ensure_parts(R.build_leaderboard(refresh=True, verbose=False))


def _ensure_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the config / encoder / pooling / origin columns exist, deriving them if they do not.

    Why this is needed: a leaderboard.csv written by an older version of the code does not have
    these columns, and asking the user to rebuild the file before any figure will draw is a bad
    trap. The columns are just the model id split on "__", so we can rebuild them here in one line
    each and the figures work whatever produced the file.
    """
    if df.empty or "model" not in df.columns:
        return df
    parts = df["model"].astype(str).apply(split_model_id)
    for name, index in (("config", 0), ("encoder", 1), ("pooling", 2)):
        if name not in df.columns or df[name].isna().all():
            df[name] = parts.str[index]
    if "origin" not in df.columns or df["origin"].isna().all():
        df["origin"] = [R.origin_label(e, p) for e, p in zip(df["encoder"], df["pooling"])]
    return df


def matrix(metric: str = "test_acc", models: Optional[List[str]] = None) -> pd.DataFrame:
    """
    The per-dataset table: one ROW per model, one COLUMN per dataset, one metric in the cells.

    This is the table every statistical figure is built on - ranks, significance tests, win matrices
    and the dataset x model heatmap all need "how did model M do on dataset D", not just the mean.

    metric : "test_acc" | "test_loss" | "test_aopcr" (any column of results.csv).
    models : which models to include (default: every model with a full sweep, see swept_models()).
    returns : a DataFrame indexed by model, columns = dataset names. Missing runs stay NaN.
    """
    models = swept_models() if models is None else models
    rows = {}
    for model_id in models:
        res = R.mean_over_seeds(R.load_results(model_id))     # several seeds -> their average
        if res.empty or metric not in res.columns:
            continue
        s = pd.to_numeric(res.set_index("dataset")[metric], errors="coerce")
        rows[model_id] = s[~s.index.duplicated(keep="last")]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T


def swept_models(min_datasets: int = MIN_DATASETS_FOR_STATS) -> List[str]:
    """
    The models that finished a full sweep, so they may enter the statistical figures.

    Everything else was only screened on WebTraffic. Those models are perfectly real - they just
    cannot be ranked over 85 datasets, so they belong in the screening figures instead.

    min_datasets : how many UCR datasets a model must have to qualify.
    returns : the qualifying model ids, sorted.
    """
    out = []
    for model_id in R.discover_models():
        res = R.load_results(model_id)
        if res.empty:
            continue
        # count DISTINCT datasets - with several seeds the same dataset appears more than once, and
        # 85 datasets x 3 seeds must not be mistaken for 255 datasets
        n_ucr = res.loc[res["dataset"].isin(set(UCR_128_DATASETS)), "dataset"].nunique()
        if n_ucr >= min_datasets:
            out.append(model_id)
    return sorted(out)


def millet_series(metric: str = "acc") -> pd.Series:
    """
    The MILLET paper's published per-dataset numbers (the mean of its 5 Conjunctive repeats).

    metric : "acc" | "loss" | "aopcr".
    returns : a Series indexed by dataset name (85 entries), or an empty Series if the file is gone.
    """
    fname = R.COMPARED_METRICS[metric][0]                     # e.g. "test_acc.csv"
    try:
        return R.millet_baseline(fname)
    except Exception:
        return pd.Series(dtype=float)


def common_datasets(mat: pd.DataFrame, with_millet: bool = True,
                    metric: str = "acc") -> List[str]:
    """
    The datasets EVERY model in the matrix has a result for (and MILLET too, if asked).

    Why this matters: a rank or a significance test is only valid when every model was scored on the
    same datasets. If model A skipped a hard dataset that model B ran, comparing their means rewards
    A for the skip. So we intersect first, and every caption reports how many datasets survived.

    mat : the model x dataset matrix.
    with_millet : also require a MILLET baseline for the dataset (needed for head-to-head figures).
    returns : the dataset names, in the matrix's column order.
    """
    if mat.empty:
        return []
    keep = mat.dropna(axis=1, how="any").columns
    keep = [d for d in keep if d != WEB_TRAFFIC]              # UCR only - WebTraffic is its own story
    if with_millet:
        base = millet_series(metric)
        keep = [d for d in keep if d in base.index]
    return list(keep)


BASELINE_ROW_NAME = "MILLET (published)"


def with_baseline_row(mat: pd.DataFrame, baseline: pd.Series,
                      name: str = BASELINE_ROW_NAME) -> pd.DataFrame:
    """
    Add the published MILLET numbers to the matrix as if it were one more model.

    Why: a ranking that leaves the baseline out can only say which of OUR models is best. Putting
    MILLET in as a row lets the same figure answer the question the paper actually asks - where does
    the published baseline sit among them? On a critical difference diagram this is the difference
    between "our models are ordered like this" and "these models are statistically inseparable from
    the published baseline", which is the claim worth making.

    mat : the model x dataset matrix.
    baseline : the published per-dataset scores.
    name : what to call the extra row.
    returns : the matrix with one extra row (unchanged if the baseline is empty).
    """
    if mat.empty or baseline.empty:
        return mat
    row = baseline.reindex(mat.columns)
    if row.notna().sum() < len(mat.columns):                 # baseline must cover the same datasets
        return mat
    out = mat.copy()
    out.loc[name] = row.astype(float)
    return out


def profile() -> pd.DataFrame:
    """
    The cost measurements from scripts/profile_models.py: speed, FLOPs, memory.

    Returns an EMPTY frame if the script has not been run yet. Every figure that needs it checks and
    skips itself with a printed note - the pipeline never crashes just because the profiling step was
    not done.
    """
    if not os.path.exists(PROFILE_CSV):
        return pd.DataFrame()
    df = read_our_csv(PROFILE_CSV)
    for col in ("params", "size_mb", "flops_m", "infer_ms", "throughput", "peak_mem_mb"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def paper_baseline() -> Dict[str, float]:
    """MILLET's published WebTraffic numbers: {"acc":.., "loss":.., "aopcr":.., "ndcg":..}."""
    return R.webtraffic_paper_baseline()


def with_profile(lb: pd.DataFrame) -> pd.DataFrame:
    """
    Join the cost measurements onto the leaderboard, so one frame has quality AND cost.

    That is what the efficiency figures need: accuracy on one axis, FLOPs or milliseconds on the
    other. If profiling has not been run the leaderboard comes back unchanged, just without those
    columns, and the efficiency figures skip themselves.
    """
    prof = profile()
    if prof.empty or "model" not in prof.columns:
        return lb
    cost = prof[[c for c in ("model", "flops_m", "infer_ms", "throughput", "peak_mem_mb")
                 if c in prof.columns]]
    return lb.merge(cost, on="model", how="left")


def ours_mask(lb: pd.DataFrame) -> pd.Series:
    """True for the models we are proposing, False for our rerun of the paper's own backbones."""
    if "origin" in lb.columns:
        return lb["origin"] != "millet"
    return pd.Series(True, index=lb.index)
