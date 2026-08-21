"""
seanet/analysis/tables.py - the comparison tables, written next to the figures.

Every table is written twice from the SAME DataFrame:

    .csv : the raw numbers, for a reader who wants to re-analyse them or open them in a spreadsheet
    .md  : markdown, so the table is readable on GitHub without running anything

Writing both from one frame is the point - they cannot drift apart.

(LaTeX output was dropped in seanetv7 together with the report. If you ever need a .tex table
again, `pandas.DataFrame.to_latex()` on the .csv does the job in one line.)
"""
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from seanet.analysis import data as PD
from seanet.analysis import stats as PS
from seanet.analysis import style as S

def tables_dir() -> str:
    """Where the tables go: <analysis root>/tables. A function, not a constant, so that
    set_analysis_root() at start-up is respected (a constant would freeze the old path)."""
    return os.path.join(S.ANALYSIS_ROOT, "tables")

# how each column should be printed and what to call it in the paper
COLUMN_TITLES = {
    "model": "Model", "config": "Config", "encoder": "Encoder", "pooling": "Pooling",
    "origin": "Origin", "rank": "#",
    "web_acc": "Acc.", "web_loss": "Loss", "web_aopcr": "AOPCR", "web_ndcg": "NDCG",
    "ucr85_acc": "UCR-85 Acc.", "ucr85_loss": "UCR-85 Loss", "ucr85_aopcr": "UCR-85 AOPCR",
    "ucr85_acc_wtl": "W/T/L", "ucr85_n": "#datasets",
    "params": "Params", "size_mb": "Size (MB)", "flops_m": "FLOPs (M)",
    "infer_ms": "Latency (ms)", "peak_mem_mb": "Mem (MB)", "throughput": "Series/s",
    "avg_rank": "Avg. rank", "win": "Win", "tie": "Tie", "loss": "Loss",
    "mean_gap": "Mean gap",
}

# how many decimals each column gets (anything not listed is left as it is)
DECIMALS = {
    "web_acc": 3, "web_loss": 3, "web_aopcr": 2, "web_ndcg": 3,
    "ucr85_acc": 4, "ucr85_loss": 4, "ucr85_aopcr": 3,
    "size_mb": 2, "flops_m": 1, "infer_ms": 3, "peak_mem_mb": 1, "avg_rank": 2, "mean_gap": 4,
}


def _format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a frame printable: round the numbers, thousands-separate the parameter counts, and turn
    every missing value into an em dash.

    An em dash rather than "0" or "nan" matters: a missing UCR number means "this model was only
    screened", and a reader must never mistake that for a score of zero.
    """
    out = df.copy()
    for col in out.columns:
        if col == "params" and col in out:
            out[col] = out[col].map(lambda v: f"{int(v):,}" if pd.notna(v) else "--")
        elif col in DECIMALS:
            nd = DECIMALS[col]
            out[col] = out[col].map(lambda v, nd=nd: f"{v:.{nd}f}" if pd.notna(v) else "--")
    return out.fillna("--")


def _bold_best(df: pd.DataFrame, column: str, higher_better: bool = True) -> pd.DataFrame:
    """
    Mark the best value in one column with Markdown bold (**0.955**), so the winner jumps out.

    df : the already-formatted table.
    column : which column to look at.
    higher_better : True for accuracy/AOPCR/NDCG, False for loss/params/FLOPs.
    returns : a copy of df with that column's best cell wrapped in **.
    """
    if column not in df.columns:
        return df
    values = pd.to_numeric(df[column], errors="coerce")
    if values.notna().sum() == 0:
        return df
    best = values.max() if higher_better else values.min()
    out = df.copy()
    out[column] = [f"**{v}**" if pd.notna(raw) and raw == best else v
                   for v, raw in zip(df[column], values)]
    return out


def write_table(df: pd.DataFrame, name: str, caption: str, label=None,
                columns=None, bold=None, directory: str = None) -> Dict:
    """
    Write one table as CSV (the real numbers) and Markdown (readable at a glance).

    df : the data.
    name : file name without extension.
    caption : one full sentence describing the table; it becomes the Markdown heading.
    label : an id for the table; defaults to "tab:<name>".
    columns : which columns to keep, in order (default: all of them).
    bold : {column: higher_is_better} - which columns get their best value bolded in the Markdown.
    directory : where to write.
    returns : the metadata entry for this table.
    """
    directory = directory or tables_dir()
    os.makedirs(directory, exist_ok=True)
    keep = [c for c in (columns or list(df.columns)) if c in df.columns]
    raw = df[keep].copy()

    # the CSV keeps the real numbers - never the formatted strings
    csv_path = os.path.join(directory, f"{name}.csv")
    raw.to_csv(csv_path, index=False)

    pretty = _format(raw)
    for column, higher in (bold or {}).items():
        pretty = _bold_best(pretty, column, higher)
    headers = [COLUMN_TITLES.get(c, c.replace("_", " ")) for c in keep]

    # --- Markdown ---
    md = ["| " + " | ".join(headers) + " |",
          "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in pretty.iterrows():
        md.append("| " + " | ".join(str(v) for v in row[keep]) + " |")
    md_path = os.path.join(directory, f"{name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"### {caption}\n\n" + "\n".join(md) + "\n")

    return {"name": name, "caption": caption, "label": label or f"tab:{name}",
            "rows": len(raw), "files": {"csv": csv_path.replace(os.sep, "/"),
                                        "md": md_path.replace(os.sep, "/")}}


# --------------------------------------------------------------------------------------
# The tables the paper needs
# --------------------------------------------------------------------------------------
def main_results_table(lb: pd.DataFrame, k: int = 10) -> Dict:
    """
    THE main results table: the best k models, quality and cost side by side.

    This is the table a reader checks first, so it holds everything needed to judge a model in one
    row - what it is, how well it does, and what it costs.
    """
    top = lb.dropna(subset=["web_acc"]).sort_values("web_acc", ascending=False).head(k)
    columns = [c for c in ["model", "origin", "web_acc", "web_aopcr", "web_ndcg", "web_loss",
                           "ucr85_acc", "ucr85_acc_wtl", "params", "size_mb", "flops_m", "infer_ms"]
               if c in top.columns]
    show = top[columns].copy()
    show["model"] = show["model"].map(lambda m: S.shorten(m, 26))
    return write_table(
        show, "table1_main_results", columns=columns,
        bold={"web_acc": True, "web_aopcr": True, "web_ndcg": True, "web_loss": False,
              "ucr85_acc": True, "params": False},
        caption=(f"Main results: the {len(show)} best models by WebTraffic accuracy. "
                 f"UCR-85 columns are the mean over the 85 datasets the MILLET paper reports, and "
                 f"W/T/L is the per-dataset win/tie/loss record against it; a dash means the model "
                 f"was screened on WebTraffic only. Best value per column in bold."),
        label="tab:main_results")


def leaderboard_table(lb: pd.DataFrame) -> Dict:
    """The complete leaderboard - every model, for the appendix."""
    columns = [c for c in ["rank", "model", "encoder", "pooling", "origin", "web_acc", "web_aopcr",
                           "web_ndcg", "web_loss", "ucr85_n", "ucr85_acc", "params", "size_mb"]
               if c in lb.columns]
    show = lb[columns].copy()
    # This table is a LOOKUP table: a reader who meets a model name in the report has to be able
    # to find its row here. The abbreviated display label is NOT good enough for that, because
    # shorten() cuts the middle out and two different models can end up with the same string --
    # seanet_gated_mschan and seanet_inputgate_mschan both became "sea channels...ea topk conj".
    # The config identifier is unique, is what the YAML files and the commands use, and is what
    # Appendix C refers to, so print that instead.
    if "config" in lb.columns:
        show["model"] = lb["config"].astype(str)
    else:
        show["model"] = show["model"].map(lambda m: S.shorten(m, 26))
    return write_table(
        show, "table_appendix_full_leaderboard", columns=columns,
        caption=(f"Complete leaderboard: all {len(show)} evaluated models, ranked by WebTraffic "
                 f"accuracy. The Model column is the configuration identifier under "
                 f"configs/models/, so any model named in the report can be looked up here. "
                 f"Encoder and pooling names are prefixed sea\\_ for components introduced "
                 f"in this work and mil\\_ for components reused unchanged from MILLET. A dash marks "
                 f"a model screened on WebTraffic only."),
        label="tab:full_leaderboard")


def ranking_table(mat: pd.DataFrame, baseline: pd.Series, metric_name: str = "accuracy") -> Optional[Dict]:
    """
    The statistical summary: average rank plus the win/tie/loss record, in one table.

    These two belong together - the rank says who is best overall, the record says how that was
    earned - and a reviewer will want both numbers without flipping between two figures.
    """
    if mat.empty:
        return None
    ranks = PS.average_ranks(mat, higher_better=True).rename("avg_rank")
    wtl = PS.win_tie_loss(mat, baseline, higher_better=True).set_index("model")
    df = pd.concat([ranks, wtl[["win", "tie", "loss", "mean_gap"]]], axis=1).reset_index()
    df = df.rename(columns={"index": "model"}).sort_values("avg_rank")
    df["model"] = df["model"].map(lambda m: S.shorten(m, 26))
    return write_table(
        df, f"table_ranking_{metric_name}",
        columns=["model", "avg_rank", "win", "tie", "loss", "mean_gap"],
        bold={"avg_rank": False, "win": True, "mean_gap": True},
        caption=(f"Statistical summary over {mat.shape[1]} datasets for the {len(df)} fully-swept "
                 f"models: average {metric_name} rank (1 is best), the per-dataset win/tie/loss "
                 f"record against the published MILLET results (ties are differences within 0.005), "
                 f"and the mean gap. Best value per column in bold."),
        label=f"tab:ranking_{metric_name}")


def efficiency_table(lb: pd.DataFrame) -> Optional[Dict]:
    """Cost only: parameters, size, FLOPs, latency, memory. Skipped if profiling was never run."""
    cost = [c for c in ["params", "size_mb", "flops_m", "infer_ms", "throughput", "peak_mem_mb"]
            if c in lb.columns and lb[c].notna().any()]
    if len(cost) < 2:
        return None
    show = lb.dropna(subset=["web_acc"]).sort_values("web_acc", ascending=False).head(15)
    columns = ["model", "web_acc"] + cost
    show = show[[c for c in columns if c in show.columns]].copy()
    show["model"] = show["model"].map(lambda m: S.shorten(m, 26))
    return write_table(
        show, "table_efficiency", columns=[c for c in columns if c in show.columns],
        bold={"params": False, "flops_m": False, "infer_ms": False, "web_acc": True},
        caption=("Computational cost of the 15 most accurate models. FLOPs, latency and peak memory "
                 "are measured on a fixed dummy input so that architectures are compared on equal "
                 "footing; see the profiling script for the exact shape. Best value per column in "
                 "bold."),
        label="tab:efficiency")


def per_dataset_table(mat: pd.DataFrame, baseline: pd.Series, top_models: int = 5) -> Optional[Dict]:
    """
    Per-dataset accuracy for the best few models next to MILLET - the appendix's biggest table.

    Only the top models are included: a table of 16 models x 85 datasets is unreadable on a page, and
    the complete matrix is available as CSV for anyone who wants it.
    """
    if mat.empty:
        return None
    order = list(PS.average_ranks(mat, higher_better=True).index)[:top_models]
    sub = mat.loc[order].T                                    # datasets down the page, models across
    sub.columns = [S.shorten(m, 18) for m in order]
    if not baseline.empty:
        sub.insert(0, "MILLET", baseline.reindex(sub.index))
    sub = sub.reset_index().rename(columns={"index": "dataset", "dataset": "dataset"})
    return write_table(
        sub, "table_appendix_per_dataset",
        caption=(f"Per-dataset accuracy of the {len(order)} best-ranked models against the published "
                 f"MILLET results, over {len(sub)} datasets. The complete matrix for all models is "
                 f"provided as CSV alongside this table."),
        label="tab:per_dataset")


def generate_tables(lb: pd.DataFrame, mat: pd.DataFrame, baseline: pd.Series) -> List[Dict]:
    """Write every table. Any table whose data is missing is skipped, never faked."""
    entries = [main_results_table(lb), leaderboard_table(lb)]
    for maybe in (ranking_table(mat, baseline), efficiency_table(lb),
                  per_dataset_table(mat, baseline)):
        if maybe:
            entries.append(maybe)
    return entries
