# PROJECT_STATE.md — read this first, every new session

Purpose of this file: so a new Claude session knows where we are **without
exploring the repo**. Exploring costs a lot of tokens. Everything below is
already checked and true as of **2026-08-10**.

If something here disagrees with the actual files, the files win — but tell me,
so we fix this file.

---

## 1. What the project is (one paragraph)

SEA-Net is a time-series classification model that is also **interpretable**.
It follows the MILLET idea: instead of only predicting a label, the model also
says *which time steps* made it choose that label. We build our own encoders
(`sea_mstcn_*`) and our own pooling heads (`sea_*_conjunctive`, etc.) and
compare them against MILLET and the usual baselines (FCN, ResNet,
InceptionTime). The goal is to beat MILLET Conjunctive on **accuracy** and on
**AOPCR** (the interpretability score) at the same time.

Datasets: **WebTraffic** (our main one) + the **UCR 2018** archive
(85-dataset subset and the full 128).

---

## 2. WHERE WE ARE RIGHT NOW  ← the important part

**Current phase: writing the report. Not running new experiments.**

- Branch: `report4` (created from `report3`).
- **72 model variants** trained and ranked in `results/SEA_NET/leaderboard.csv`.
- **The report was fully restructured and rewritten on 2026-08-13.** New reading
  order: Problem → Motivation → State of the art → Contributions → Background →
  MILLET baseline → Proposed methods → Methodology → Results → Discussion →
  Conclusion → References → Appendices. The internship context, objectives,
  constraints and skills now live in **Appendix A**, after the references, so
  they no longer interrupt the scientific story.
- **The PDF is 34 pages** (was 40), builds with **0 errors, 0 undefined
  references/citations, 0 overfull boxes**, no placeholders, no guide boxes.
  Title + abstract are on page 1 and the table of contents is exactly page 2.
- Page count came down through structure and formatting, not by deleting
  results: wider text block (6.4 in instead of ICLR's 5.5 in), tighter float and
  list spacing, deduplicated prose, and removal of two redundant tables.

### Two code changes that need `python main.py paper` to take effect

These were made in the plotting/table code, so the report will only show them
after the figures and tables are regenerated:

1. `seanet/paper/figures_stats.py` — `_labels()` no longer truncates to 20
   characters (the two MILLET rows both collapsed to `mil_incep..._mil_conj`),
   and the win/tie/loss figure now prefixes each bar with a **competition rank**,
   so two models with the same number of wins visibly share a rank and are marked
   with `=`. Appendix B.6 of the report already describes this.
2. `seanet/paper/tables.py` — the appendix leaderboard's Model column now prints
   the **configuration identifier** (`seanet_gated_mean_topk`) instead of an
   abbreviated label. Two different models used to shorten to the same string, so
   the table could not be used for lookup. Appendix B.1 already describes this.

Still open in the generated figures: `topk5_multimetric` (Figure 8) has value
labels that collide with the dashed reference lines.

### Float placement rule learned the hard way (2026-08-12)

**Never use `[H]` on a tall float.** `[H]` (from the `float` package) means
"stay exactly here even if there is no room", so it silently runs off the
bottom of the page and the overflow is *lost* — no error, no warning. It ate
the whole encoder x pooling grid figure and half of Table 10. Every float in
`A1_`/`A2_` is now `[htbp]`, and the full leaderboard has a `\clearpage` before
it so it gets a page to itself.
- Two blue `\missing{}` flags are still live on purpose: the `millet_paper` UCR
  row (8.2) and the ensemble numbers (Appendix A).

So: if I ask for something, assume it is about **writing/figures/tables/page
count**, not about training. Do not suggest new training runs unless I ask.

### The last long job is DONE (2026-08-12)

`sv1/millet_paper` (MILLET's own 1500-epoch recipe on our harness, for objective
O2) has finished, and `ensemble_vote.csv` has been rebuilt. **Nothing is left
running.** The result closes O2:

| | WebTraffic acc | UCR-85 acc | mean rank | W/T/L vs published |
|---|---|---|---|---|
| MILLET, **our** recipe | 0.887 | 0.8274 | 12.60 | 13/25/46 |
| MILLET, **their** recipe (`millet_paper`) | 0.920 | **0.8434** | **9.41** | 26/32/26 |
| MILLET, published | — | 0.8445 | — | — |

Same architecture, same code, only the recipe differs. 0.8434 against a
published 0.8445 is a reproduction, so **the gap was always the training budget,
not our re-implementation.** Its AOPCR of 13.27 (against 2.57 under our recipe)
is also the direct proof that AOPCR is unnormalised.

`millet_paper` has `ucr85_n = 84`, same as `millet`, so both are ranked over the
84 datasets shared by the **28** fully-swept models (was 27 — every mean rank in
Table 7 moved by about +0.7 when this row joined the pool; nothing regressed).

---

## 3. The report — the one folder that matters

**`ICLR_2025_Report/Seanet/` is the real report. Do not touch it unless I ask.**

I am actively writing in it. Never delete, move, rename, or "clean" anything
inside it, and never delete anything elsewhere that the report might use.

```
ICLR_2025_Report/
  Seanet/                  <- THE REAL REPORT (I am writing here)
    main.tex               <- the only file that gets compiled
    preamble.tex           <- packages, colours, macros
    refs.bib               <- citations
    sections/              <- 14 .tex files, one per section (edit these)
    figures/               <- my own images go here (still empty)
    *.sty, *.bst           <- official ICLR 2025 style files
  Template/                <- untouched original ICLR template, reference only
```

Section files, in report order (restructured 2026-08-13). The old files are kept
in `sections/_superseded/` — nothing was deleted, they are just no longer
`\input` by `main.tex`.

| file | section | pages |
|---|---|---|
| `00_abstract.tex` | Abstract (2 paragraphs, page 1 with the title) | — |
| `01_introduction.tex` | 1 Introduction: problem, motivation, state of the art, contributions | 3 |
| `02_background.tex` | 2 Background: notation table, MIL, MILLET + its limits L1–L3, AOPCR/NDCG | 4 |
| `03_methods.tex` | 3 Proposed Methods: encoder, block, 7 encoder variants, 7 pooling heads, loss | 9 |
| `04_methodology.tex` | 4 Experimental Methodology | 2 |
| `05_results.tex` | 5 Results | 2 |
| `06_discussion.tex` | 6 Discussion | 3 |
| `07_conclusion.tex` | 7 Conclusion + Future work | 2 |
| `A1_internship.tex` | App. A internship: team, mission, objectives table, constraints, skills | 2 |
| `A2_additional_results.tex` | App. B additional results | 6 |
| `A3_reproducibility.tex` | App. C reproducibility | 3 |

**Total 34** including the title page, the one-page contents and 2 pages of
references. Page budget: **20–30 including appendix** (examiner rule, AIEDA400),
so it is still about 4 pages over. Getting to 30 now means dropping content, and
that is my decision, not a guess: the candidates, cheapest first, are the 72-row
leaderboard table (B.1, one full page), the encoder × pooling grid figure (B.3),
the ensemble / Optuna / Transformer paragraphs (B.5), and the per-dataset table
(B.4).

### Figures — 13, all real, no placeholders left

| where | figure | source |
|---|---|---|
| 1 | teaser | generated |
| 2 | the MIL view | `figures/fig02_mil_view.png` (mine) |
| 3 | the whole model | `figures/fig04_seanet_architecture.pdf` (mine) |
| 3 | attach points A–D | `figures/fig_var_attach.tex` — TikZ |
| 3 | one figure per encoder variant | `figures/fig_var_{bottleneck,gated,inputgate,spiketrend,recon,mschan,pyramid}.tex` |
| 3 | pooling template + one panel per head | `figures/fig_pool_template.tex`, `fig_pool_aggregators.tex` |
| 5 | top-5 quality vs cost | generated |
| B | ablation grid, win/tie/loss | generated |

The old combined diagrams (`fig_variants_tikz.tex`, `fig_pooling_tikz.tex`) and
`fig_bottleneck_tikz.tex` are kept but **unused** — they were split into the
small per-variant files above so each diagram is readable on its own. Do not
delete them. `fig_block_tikz.tex` is still switchable
(`\showblockfigtrue/false` in `preamble.tex`) and is currently **off** to save a
page; nothing cites it, so turning it back on breaks nothing. The
`appendix_dataset_model_heatmap_accuracy` figure was removed from the report on
2026-08-13 (it did not add enough over the win/tie/loss figure); the image file
is still in `results/paper_figures/`.

The block figure is **switchable**: `\showblockfigfalse` in `preamble.tex`
removes it, and no paragraph cites it, so nothing breaks. Keep it uncited.

The full writing guide, macro list, figure list and submission checklist are in
`ICLR_2025_Report/Seanet/README.md` — read that file when working on the report.

### Build the report

```powershell
cd "D:\Documents\Intership TSC\Projects\SEA_NET\ICLR_2025_Report\Seanet"
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

Four passes, because LaTeX only learns page and citation numbers on the first
pass. `??` or `[?]` in the PDF just means "build again".

**When Ctrl+S does not refresh `main.pdf`, read
`ICLR_2025_Report/Seanet/LATEX_GUIDE.md`** — it has the checklist, how to read a
LaTeX error, and the "root file" trap (VS Code used to compile
`ICLR_2025_Report/Template/` by mistake, so `main.pdf` never changed). Every
file in `sections/` now starts with `% !TEX root = ../main.tex`; keep that line
on any new section file.

### Switches in `preamble.tex`

- `\showguidestrue` → grey "Writing guide" boxes are visible (drafting mode, now)
- `\showguidesfalse` → hides them all (before submitting)
- `\showblockfigtrue` / `\showblockfigfalse` → shows or hides the optional TikZ
  block diagram in section 5.3

Placeholders that must all be gone before submission: `\todo{...}` (red),
`\num{...}`, `\figph{...}`, `\tabph{...}`, `\tbd`, `\ms`.

---

## 4. Where the numbers and figures come from

The report pulls from `results/`. **Nothing in `results/` should be deleted
without checking here first.**

| path | what it is | used by the report? |
|---|---|---|
| `results/SEA_NET/leaderboard.csv` | all 66 models ranked by WebTraffic acc | **yes** — main table |
| `results/SEA_NET/<model>/results.csv` | per-dataset numbers for one model | **yes** — feeds leaderboard + all paper figures |
| `results/SEA_NET/<model>/comparison_vs_millet.csv` | that model vs MILLET | yes |
| `results/SEA_NET/<model>/summary.csv`, `summary.md` | short per-model summary | yes |
| `results/paper_figures/` | 148 files, the report's real figures (PDF+PNG+SVG) | **yes** |
| `results/paper_figures/tables/` | generated LaTeX tables to `\input` | **yes** |
| `results/SEA_NET/teaser/` | page-1 teaser image | **yes** — cited in `01_introduction.tex` |
| `results/SEA_NET/<model>/curves/*.csv` | **NEW** per-epoch loss curve, one file per dataset per seed | yes — training-behaviour figure |
| `results/SEA_NET/<model>/interpretation/` | per-sample explanation figures (`main.py interpret`) | yes |
| `results/SEA_NET/<model>/predictions/*.npz` | raw outputs, 3 models only | ensemble voting — keep |
| `results/SEA_NET/ensemble_vote.csv` | ensemble voting result | yes |
| `results/SEA_NET/profile.csv` | params / FLOPs / memory / latency | yes — efficiency table |
| `results/SEA_NET/data_summary.csv` | dataset statistics | yes — setup section |

**Cleanup done 2026-08-06 (~138 MB freed).** All of it is still recoverable from
git history, because the files were committed before being deleted. Removed:

- `results/SEA_NET/*/figures/` — per-run preview PNGs, superseded by `results/paper_figures/`
- `results/SEA_NET/*/logs/` — per-run training logs
- `logs/` and `OAR.*.stdout` at the root — raw Grid5000 console dumps, ~108 MB.
  `logs/` is recreated automatically: `scripts/run_all.sh` and `scripts/launch.sh`
  both run `mkdir -p logs` first. Both are now in `.gitignore`.

**Kept on purpose — do not confuse these with the deleted ones:**

- `results/SEA_NET/figures/` (top level, 13 PNGs) — cross-model comparison
  figures: `winner_dashboard`, `model_comparison`, WebTraffic accuracy tier bands
- `results/SEA_NET/logs/` (top level, 56 logs) — the log of every `main.py`
  command run (`paper`, `report`, `teaser`, `results`, `web-compare`, `leaderboard`)

The difference is the folder depth: `results/SEA_NET/<model>/figures/` was junk,
`results/SEA_NET/figures/` is a keeper.

### Using a generated figure or table in the report

`preamble.tex` adds the project root to the LaTeX image search path, so figures
are referenced by their project path with **no file extension** (LaTeX then
picks the sharp `.pdf`):

```latex
\includegraphics[width=\linewidth]{results/paper_figures/01_main_figures/pareto_web_acc_vs_params}
\input{../../results/paper_figures/tables/table1_main_results}
```

(`../../` on `\input` because the report sits two folders below the root.)

Available generated tables: `table1_main_results`,
`table_appendix_full_leaderboard`, `table_ranking_accuracy`,
`table_efficiency`, `table_appendix_per_dataset`.

---

## 5. Current headline results (from `leaderboard.csv`)

Ranked by WebTraffic accuracy. `web_aopcr` is the interpretability score
(higher is better).

| rank | config | encoder | pooling | web_acc | web_aopcr | web_ndcg | params |
|---|---|---|---|---|---|---|---|
| 1 | `seanet_gated_mean_topk` | `sea_mstcn_sep_gated` | `sea_topk_conjunctive` | 0.9547 | 2.225 | 0.750 | 61,740 |
| 2 | `seanet_conjunctive` | `sea_mstcn_sep` | `mil_conjunctive` | 0.9540 | 1.502 | 0.698 | 269,083 |
| 3 | `seanet_gated_max_topk` | `sea_mstcn_sep_gated` | `sea_topk_conjunctive` | 0.9520 | 2.268 | 0.719 | 61,740 |
| 4 | `seanet_topk_nofocus` | `sea_mstcn_sep_bottleneck` | `sea_topk_conjunctive` | 0.9500 | 2.303 | 0.765 | 41,324 |
| 5 | `seanet_spiketrend_topk` | `sea_mstcn_sep_spiketrend` | `sea_topk_conjunctive` | 0.9500 | 2.316 | 0.756 | 67,020 |

**Seeds matter here.** Four models have 3 seeds on WebTraffic, and two of them
also have 3 seeds over the whole UCR archive:

| model | acc | AOPCR | NDCG | params |
|---|---|---|---|---|
| `seanet_gated_mean_topk` | 0.955 ± 0.003 | 2.225 ± 0.131 | 0.750 ± 0.022 | 61,740 |
| `seanet_bottleneck_topk` | 0.947 ± 0.016 | 2.621 ± 0.214 | **0.773** ± 0.013 | **41,324** |
| `seanet_inputgate_adaptive` | 0.905 ± 0.030 | 2.651 ± 0.437 | 0.732 ± 0.043 | 58,102 |
| `millet` re-trained | 0.887 ± 0.010 | 2.569 ± 0.884 | 0.677 ± 0.016 | 423,707 |

**The story for the report:** `seanet_bottleneck_topk` beats the re-trained
MILLET baseline on accuracy, AOPCR *and* NDCG at once, with **90 % fewer
parameters** (41 K vs 424 K) and 11× fewer FLOPs. `seanet_gated_mean_topk` has
the best accuracy of the sweep and is the most stable model trained.

**Two honest counterweights the report states plainly:**
1. On the UCR archive MILLET is still ahead — 0.8274 (our recipe, rank 12.60 of
   28) and 0.8434 (their recipe, rank 9.41) against our ~15.3–15.5.
2. AOPCR is unnormalised. Proof: the *same* architecture scores AOPCR 2.57
   under our recipe and **13.27** under MILLET's (`sv1/millet_paper`).

Read the real CSV before quoting any number not in the tables above.

---

## 6. Repo map (only what matters)

```
main.py                  <- ONE entry point, all commands (44 KB)
configs/                 <- YAML: models/ (sv1..sv7), datasets, main.yaml
                            sv7/ = the ablation copies added 2026-08-10:
                            top_frac 0.05/0.25/0.5/1.0 + lambda_entropy 0
seanet/                  <- our code (encoders, pooling heads, paper figures)
millet/                  <- upstream MILLET, kept diffable — avoid editing
model/                   <- saved weights (git-ignored)
data/                    <- UCR + WebTraffic (871 MB, git-ignored)
results/                 <- all outputs, see section 4
ICLR_2025_Report/        <- THE REPORT, see section 3
Latex/                   <- old presentation leftovers + archi_level1-3.png
report/                  <- OLD abandoned draft, ignored by git — not the report
workflow.md              <- my long working notes (git-ignored, 53 KB)
GRID5K_CMD_HELP.md       <- Grid5000 command cheatsheet (30 KB)
MLFLOW_GUIDE.md          <- MLflow how-to
```

**Careful:** `report/` is NOT the report. It is an abandoned earlier draft
(12 KB total, different section names). The real one is
`ICLR_2025_Report/Seanet/` (99 KB). Never confuse the two.

---

## 7. Commands (`python main.py <cmd>`)

Read-only / cheap — safe to suggest while writing the report:

| command | what it does |
|---|---|
| `leaderboard` | rebuild `leaderboard.csv` from every model's `results.csv` |
| `paper` | rebuild every paper figure **and** the LaTeX tables |
| `results` | build the comparison vs MILLET |
| `web-compare` | WebTraffic comparison table + tier figures |
| `summary` | dataset statistics |
| `params` | SEA-Net vs baseline parameter counts |
| `report` | every figure + summary table under `results/SEA_NET/` |
| `teaser` | the page-1 teaser figure |

Expensive (training — never run these for me, see rule 3 in `CLAUDE.md`):
`train`, `single`, `webtraffic`, `run`, `interpret`, `optuna`.

Most likely command while drafting:

```powershell
python main.py paper     # refresh all report figures + tables
```

---

## 8. Two environments — never mix them up

- **Local (this machine)**: Windows, VS Code. Writing code, writing the report,
  small smoke tests only.
- **Grid5000 (Lille / Sophia)**: VS Code over SSH. All real GPU training.

The *code* is the same in both. The *environment* is not: Python version, conda
env name, package versions, GPU/CUDA, paths, and how a job starts (local
terminal vs `oarsub`) all differ **between the two sites as well**. Never guess
server settings from the local machine — ask me for the current server details
first and wait for me to paste them.

See `GRID5K_CMD_HELP.md` for the per-site commands.

---

## 9. Rules that bite most often

Full list is in `CLAUDE.md`. The three I care about most:

1. **Never run training or long scripts.** Give me the command, I run it.
2. **Give me git commands to copy-paste.** Do not run git for me.
3. **Explain like a teacher, in simple English.** Student-style comments.

---

## 10. Keeping this file honest

Update this file whenever:

- the phase changes (draft finished → back to experiments),
- a new branch is started,
- results are added or deleted,
- the leaderboard top-5 changes.

One line in the right place beats a long paragraph. Stale info here is worse
than no info, because it gets trusted.
