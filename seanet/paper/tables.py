"""
seanet/paper/tables.py - the tables, in the three formats a paper actually needs.

Every table is written three times from the SAME DataFrame:

    .tex : booktabs LaTeX, ready to \\input{} straight into Overleaf
    .csv : the raw numbers, for a reader who wants to re-analyse them
    .md  : markdown, so the table is readable on GitHub without compiling anything

Writing all three from one frame is the point - they cannot drift apart.

A note on the LaTeX style: we use booktabs (\\toprule, \\midrule, \\bottomrule) and NO vertical rules.
That is the standard in every ML venue: vertical lines and double horizontal lines make a table look
like a spreadsheet, and every LaTeX style guide advises against them. Add \\usepackage{booktabs} to
the Overleaf preamble.
"""
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from seanet.paper import data as PD
from seanet.paper import stats as PS
from seanet.paper import style as S

TABLES_DIR = os.path.join(S.PAPER_ROOT, "tables")

# how each column should be printed and what to call it in the paper
COLUMN_TITLES = {
    "model": "Model", "config": "Config", "encoder": "Encoder", "pooling": "Pooling",
    "origin": "Origin", "rank": "\\#",
    "web_acc": "Acc.", "web_loss": "Loss", "web_aopcr": "AOPCR", "web_ndcg": "NDCG",
    "ucr85_acc": "UCR-85 Acc.", "ucr85_loss": "UCR-85 Loss", "ucr85_aopcr": "UCR-85 AOPCR",
    "ucr85_acc_wtl": "W/T/L", "ucr85_n": "\\#DS",
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
    Bold the winning cell in a column - the convention every ML paper follows.

    It is done on the FORMATTED strings, after rounding, so the bolded value is exactly the value the
    reader sees. Bolding the raw number and rounding afterwards can bold a cell that ties on screen.
    """
    if column not in df.columns:
        return df
    values = pd.to_numeric(df[column], errors="coerce")
    if values.notna().sum() == 0:
        return df
    best = values.max() if higher_better else values.min()
    out = df.copy()
    out[column] = [f"\\textbf{{{v}}}" if pd.notna(raw) and raw == best else v
                   for v, raw in zip(out[column], values)]
    return out


def _tex_cell(value) -> str:
    """
    Make one table cell safe to compile.

    Two things in our data are ordinary text to Python but commands to LaTeX:

        _   model names are full of them (sea_mstcn_sep) and LaTeX reads "_" as
            "start a subscript", which stops the build with "Missing $ inserted"
        ...  shorten() cuts long names in the middle with a real "…" character,
            which the report's font does not have

    So we spell both the LaTeX way. Cells we built ourselves already hold a real
    command (the bolded best value), so those are left untouched.
    """
    text = str(value)
    if "\\textbf{" in text:                  # already LaTeX, do not touch it
        return text
    for ch in ("&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(ch, "\\" + ch)
    return text.replace("…", "\\dots{}")


def write_table(df: pd.DataFrame, name: str, caption: str, label: Optional[str] = None,
                columns: Optional[List[str]] = None, bold: Optional[Dict[str, bool]] = None,
                directory: str = TABLES_DIR) -> Dict:
    """
    Write one table as LaTeX, CSV and Markdown.

    df : the data.
    name : file name without extension.
    caption : the LaTeX caption, written as a full sentence.
    label : LaTeX label (defaults to "tab:<name>").
    columns : which columns to keep, in order (default: all of them).
    bold : {column: higher_is_better} - which columns get their best value bolded.
    returns : the metadata entry for this table.
    """
    os.makedirs(directory, exist_ok=True)
    keep = [c for c in (columns or list(df.columns)) if c in df.columns]
    raw = df[keep].copy()

    # the CSV keeps the real numbers - never the formatted strings
    csv_path = os.path.join(directory, f"{name}.csv")
    raw.to_csv(csv_path, index=False)

    pretty = _format(raw)
    for column, higher in (bold or {}).items():
        pretty = _bold_best(pretty, column, higher)
    # Known columns have a hand-written title (already LaTeX-safe). Anything else is a raw
    # column name - in the per-dataset table those are model names, so drop the underscores
    # and spell the "…" that shorten() may have put in the middle.
    headers = [COLUMN_TITLES.get(c, c.replace("_", " ")).replace("…", "\\dots{}") for c in keep]

    # --- LaTeX (booktabs, no vertical rules) ---
    align = "l" + "r" * (len(keep) - 1)                      # names left, numbers right
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             f"\\caption{{{caption}}}", f"\\label{{{label or f'tab:{name}'}}}",
             f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(headers) + " \\\\", "\\midrule"]
    for _, row in pretty.iterrows():
        cells = [_tex_cell(v) for v in row[keep]]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex_path = os.path.join(directory, f"{name}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # --- Markdown ---
    md = ["| " + " | ".join(h.replace("\\#", "#") for h in headers) + " |",
          "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in pretty.iterrows():
        md.append("| " + " | ".join(str(v).replace("\\textbf{", "**").replace("}", "**")
                                    if "textbf" in str(v) else str(v)
                                    for v in row[keep]) + " |")
    md_path = os.path.join(directory, f"{name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"### {caption}\n\n" + "\n".join(md) + "\n")

    return {"name": name, "caption": caption, "label": label or f"tab:{name}",
            "rows": len(raw), "files": {"tex": tex_path.replace(os.sep, "/"),
                                        "csv": csv_path.replace(os.sep, "/"),
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
