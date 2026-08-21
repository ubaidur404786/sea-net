"""
seanet.analysis - turn all the results into the comparison figures and tables.

    python main.py analyse

It reads every model's results/SEA_NET/<model>/results.csv (through the leaderboard) and writes
results/analysis/ - the cross-model comparisons that answer the questions the project exists to
answer:

    01_leaderboard/  who is strongest overall, in non-overlapping accuracy bands
    02_ablation/     what each ENCODER and each POOLING head contributes (the encoder x pooling grid)
    03_detail/       every model, every dataset
    04_webtraffic/   our headline dataset on its own (accuracy, AOPCR, NDCG)
    05_statistics/   average ranks, Wilcoxon-Holm significance, win/tie/loss vs the MILLET baseline
    tables/          the same numbers as .csv (exact) and .md (readable on GitHub)
    INDEX.md         one page listing every figure and the question it answers

The design rule
---------------
ONE FIGURE ANSWERS ONE QUESTION. Every figure carries its question in the metadata, and a figure
that would answer a question already answered elsewhere is not drawn. That is why this produces
about thirty figures rather than the few hundred that "every metric x every grouping" would give.

The modules
-----------
    style.py         : the house style - fonts, palette, sizes, and the save helper
    data.py          : loads every number once, so figures and tables can never disagree
    stats.py         : ranks, Wilcoxon-Holm significance, Pareto fronts, win counts
    figures.py       : accuracy bands, top-k, WebTraffic, efficiency, ablation
    figures_stats.py : critical-difference diagram, ranks, heatmaps, distributions
    tables.py        : the CSV + Markdown tables
    model_figures.py : the PER-MODEL figures (one model vs the MILLET baseline), used by
                       `python main.py report`

Two things you must run first for the full set
----------------------------------------------
    python main.py leaderboard          rebuilds results/SEA_NET/leaderboard.csv
    python scripts/profile_models.py    measures FLOPs / latency / memory (the efficiency figures
                                        are skipped, not faked, when profile.csv is missing)
"""
import os
from typing import Dict, List

from seanet.analysis import data as PD
from seanet.analysis import figures as F
from seanet.analysis import figures_stats as FS
from seanet.analysis import stats as PS
from seanet.analysis import style as S
from seanet.analysis import tables as T

# which top-k sizes get their own figures. 5 is the headline (main paper); 3 and 10 go to the
# appendix. We deliberately do NOT draw top-1 and top-2: a "top-1 accuracy" chart is a single bar
# and answers nothing that the main results table does not already say.
TOP_K_SIZES = [3, 5, 10]
HEADLINE_K = 5


def generate(refresh: bool = False, verbose: bool = True) -> Dict:
    """
    Build every figure and table, then the LaTeX glue.

    refresh : True rebuilds the leaderboard from every model's results.csv first (slower, always
              current); False reuses results/SEA_NET/leaderboard.csv.
    verbose : print progress and a summary.
    returns : {"figures": [...], "tables": [...], "written": {...}}.
    """
    S.apply_style()
    S.reset_manifest()

    lb = PD.leaderboard(refresh=refresh)
    if lb.empty:
        print("No results yet. Train some models first, then run `python main.py leaderboard`.")
        return {"figures": [], "tables": [], "written": {}}
    lb = PD.with_profile(lb)                                 # adds FLOPs / latency / memory if measured
    # one label per model, computed from EVERY model at once so collisions are resolved the same
    # way in every figure (see style.build_label_map).
    S.build_label_map(list(lb["model"]))

    have_cost = any(c in lb.columns and lb[c].notna().any() for c in ("flops_m", "infer_ms"))
    if verbose:
        print(f"Leaderboard: {len(lb)} models")
        if not have_cost:
            print("  note: no profile.csv found, so the FLOPs / latency / memory figures are")
            print("        skipped. Run `python scripts/profile_models.py` to add them.")

    # the per-dataset matrix, for everything statistical
    mat_all = PD.matrix("test_acc")
    baseline = PD.millet_series("acc")
    shared = PD.common_datasets(mat_all, with_millet=True, metric="acc")
    mat = mat_all[shared] if shared else mat_all
    if verbose:
        print(f"Per-dataset matrix: {mat.shape[0]} fully-swept models x {mat.shape[1]} shared datasets")
        if not PS.HAVE_SCIPY:
            print("  note: SciPy is missing, so the critical-difference bars are omitted.")

    # ---- 1. benchmark bands: figures 1-3 ----
    F.benchmark_bands(lb, "web_acc")

    # ---- 2. WebTraffic, our headline dataset ----
    F.webtraffic_metrics(lb, top_n=10)

    # ---- 3. efficiency and Pareto fronts ----
    F.efficiency_figures(lb)

    # ---- 4. ablation: what each part contributes ----
    F.ablation_figures(lb)

    # ---- 5. top-k analysis ----
    for k in TOP_K_SIZES:
        F.topk_multimetric(lb, k, section="main" if k == HEADLINE_K else "appendix")
        if k == HEADLINE_K:                                  # per-metric only for the headline k,
            F.topk_per_metric(lb, k, section="appendix")     # otherwise it is the same story 3 times

    # ---- 6. the statistical section ----
    # The ranking figures include MILLET itself as one more row, so they answer "where does the
    # published baseline sit among these models?" and not merely "how do our models order?".
    # The win/tie/loss and improvement figures compare AGAINST the baseline, so they must not
    # contain it - a model cannot win against itself.
    ranked = PD.with_baseline_row(mat, baseline)
    if not mat.empty and len(mat) >= 2:
        FS.critical_difference(ranked, higher_better=True, metric_name="accuracy")
        FS.average_rank(ranked, higher_better=True, metric_name="accuracy")
        FS.win_tie_loss(mat, baseline, metric_name="accuracy", section="main")
        FS.improvement_over_baseline(mat, baseline, metric_name="accuracy")
        FS.pairwise_win_matrix(ranked, higher_better=True, metric_name="accuracy")
        FS.dataset_model_heatmap(mat, baseline, metric_name="accuracy")
        FS.accuracy_distribution(ranked, metric_name="accuracy")
    FS.metric_correlation(lb)

    # ---- 7. the tables, then one index page listing everything ----
    figures = S.manifest()
    tbls = T.generate_tables(lb, mat, baseline)
    index = write_index(figures, tbls)

    if verbose:
        _print_summary(figures, tbls, index)
    return {"figures": figures, "tables": tbls, "index": index}


# the order the groups are listed in, most important first
SECTION_ORDER = ["main", "web", "ablation", "stats", "appendix"]


def write_index(figures: List[Dict], tables: List[Dict]) -> str:
    """
    Write results/analysis/INDEX.md - one page that says what every figure and table is.

    Without it, results/analysis/ is thirty PNGs with cryptic names. With it, you can read down
    the page, find the question you have, and open the one file that answers it.

    figures : the figure metadata from style.manifest().
    tables : the table metadata from tables.generate_tables().
    returns : the path written.
    """
    by_section: Dict[str, List[Dict]] = {}
    for entry in figures:
        by_section.setdefault(entry["section"], []).append(entry)

    lines = ["# Analysis output", "",
             "Everything here is generated by `python main.py analyse`. Do not edit it by hand -",
             "the next run overwrites it. The numbers come from `results/SEA_NET/*/results.csv`",
             "via `results/SEA_NET/leaderboard.csv`.", "",
             "## Figures", ""]
    for section in SECTION_ORDER:
        items = by_section.get(section)
        if not items:
            continue
        lines.append(f"### {S.SECTIONS.get(section, section)}")
        lines.append("")
        lines.append("| figure | the question it answers |")
        lines.append("|---|---|")
        for entry in items:
            rel = entry["files"].get("png", entry["name"])
            rel = rel.split("results/analysis/")[-1]
            lines.append(f"| [`{entry['name']}`]({rel}) | {entry['question']} |")
        lines.append("")

    lines += ["## Tables", "",
              "Each table is written twice: `.csv` (exact numbers) and `.md` (readable here).", "",
              "| table | rows | what it shows |", "|---|---|---|"]
    for table in tables:
        lines.append(f"| [`{table['name']}`](tables/{table['name']}.md) | {table['rows']} | {table['caption']} |")
    lines.append("")

    path = os.path.join(S.ANALYSIS_ROOT, "INDEX.md")
    os.makedirs(S.ANALYSIS_ROOT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def _print_summary(figures: List[Dict], tbls: List[Dict], index: str) -> None:
    """Print what was produced, grouped by section."""
    by_section: Dict[str, List[Dict]] = {}
    for entry in figures:
        by_section.setdefault(entry["section"], []).append(entry)

    print(f"\n{len(figures)} figures written to {S.ANALYSIS_ROOT}/ (PNG, 300 dpi)")
    for section in SECTION_ORDER:
        items = by_section.get(section)
        if not items:
            continue
        print(f"\n  {S.SECTIONS.get(section, section)}/  ({len(items)})")
        for entry in items:
            print(f"    {entry['name']:52s} {entry['question'][:60]}")

    print(f"\n{len(tbls)} tables written to {S.ANALYSIS_ROOT}/tables/ (.csv + .md)")
    for table in tbls:
        print(f"    {table['name']:52s} {table['rows']} rows")

    print(f"\nStart with {index}")
