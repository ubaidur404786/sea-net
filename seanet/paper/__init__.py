"""
seanet/paper - the publication figure pipeline.

One command builds everything a submission needs:

    python main.py paper

It writes results/paper_figures/ with the figures split into paper sections, the tables in three
formats, and the LaTeX glue to drop them into Overleaf.

The design rule
---------------
ONE FIGURE ANSWERS ONE QUESTION. Every figure carries its question in the metadata, and figures that
would answer a question already answered elsewhere are not drawn. That is why this produces around
thirty figures rather than the few hundred that "every metric x every grouping" would give.

The modules
-----------
    style.py         : the house style - fonts, palette, sizes, and the save-in-3-formats helper
    data.py          : loads every number once, so figures and tables can never disagree
    stats.py         : ranks, Wilcoxon-Holm significance, Pareto fronts, win counts
    figures.py       : benchmark bands, top-k, WebTraffic, efficiency, ablation
    figures_stats.py : critical difference diagram, ranks, heatmaps, distributions
    tables.py        : LaTeX / CSV / Markdown tables
    latex.py         : figures.json, figures.tex, tables.tex, README.md

What is NOT here, and why
-------------------------
Some figures a reviewer might ask for cannot be drawn from what this project has measured, and are
left out rather than faked:

    - error bars / variance / confidence intervals : every model was trained ONCE (seed 0). Real
      error bars need the sweep repeated with several seeds.
    - training and validation curves : the per-epoch history only went to MLflow, and the database
      from the Grid5000 runs was never copied back.
    - confusion matrices : never stored per dataset.

FLOPs, inference time and peak memory are NOT in this list - they are measured by
`python scripts/profile_models.py`, which must be run once before the efficiency figures appear.
"""
from typing import Dict, List

from seanet.paper import data as PD
from seanet.paper import figures as F
from seanet.paper import figures_stats as FS
from seanet.paper import latex as L
from seanet.paper import stats as PS
from seanet.paper import style as S
from seanet.paper import tables as T

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

    # ---- 7. tables, then the LaTeX glue ----
    figures = S.manifest()
    tbls = T.generate_tables(lb, mat, baseline)
    written = L.write_latex_snippets(figures, tbls)

    if verbose:
        _print_summary(figures, tbls, written)
    return {"figures": figures, "tables": tbls, "written": written}


def _print_summary(figures: List[Dict], tbls: List[Dict], written: Dict) -> None:
    """Print what was produced, grouped by paper section."""
    by_section: Dict[str, List[Dict]] = {}
    for entry in figures:
        by_section.setdefault(entry["section"], []).append(entry)

    print(f"\n{len(figures)} figures written to {S.PAPER_ROOT}/ (PDF + SVG + 600dpi PNG each)")
    for section in L.SECTION_ORDER:
        items = by_section.get(section)
        if not items:
            continue
        print(f"\n  {S.SECTIONS.get(section, section)}/  ({len(items)})")
        for entry in items:
            print(f"    {entry['name']:52s} {entry['question'][:60]}")

    print(f"\n{len(tbls)} tables written to {S.PAPER_ROOT}/tables/ (.tex + .csv + .md)")
    for table in tbls:
        print(f"    {table['name']:52s} {table['rows']} rows")

    print("\nLaTeX glue:")
    for what, path in written.items():
        print(f"    {what:14s} {path}")
    print(f"\nStart with {S.PAPER_ROOT}/README.md")
