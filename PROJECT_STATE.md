# PROJECT_STATE.md — read this first, every new session

Purpose of this file: so a new Claude session knows where we are **without
exploring the repo**. Exploring costs a lot of tokens. Everything below is
already checked and true as of **2026-08-06**.

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

- Branch: `report2` (created from `report1` on 2026-08-06).
- The experiments are **done for now**. 66 model variants trained, all ranked in
  `results/SEA_NET/leaderboard.csv`.
- The job now is to produce a **complete first draft** of the report.
- Only after the draft is finished do we go back to new experiments.

So: if I ask for something, assume it is about **writing/figures/tables**, not
about training. Do not suggest new training runs unless I ask.

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

Section files, in report order:

| file | section | state |
|---|---|---|
| `00_abstract.tex` | Abstract | write last |
| `01_introduction.tex` | Introduction | has the teaser figure wired up |
| `02_context.tex` | Context | short |
| `03_objectives.tex` | What I had to do | |
| `04_background.tex` | Background (MIL, AOPCR) | |
| `05_architecture.tex` | SEA-Net architecture | needs the main diagram |
| `06_pooling.tex` | Pooling heads | biggest section (16 KB) |
| `07_setup.tex` | Experimental setup | |
| `08_results.tex` | Results | tables to fill from CSVs |
| `09_discussion.tex` | Discussion | |
| `10_skills.tex` | Skills (internship requirement) | |
| `11_conclusion.tex` | Conclusion | |
| `A1_additional_results.tex` | Appendix A | |
| `A2_reproducibility.tex` | Appendix B | |

Page budget: **20–30 pages including appendix** (examiner rule, AIEDA400).
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

### Two drafting switches in `preamble.tex`

- `\showguidestrue` → grey "Writing guide" boxes are visible (drafting mode, now)
- `\showguidesfalse` → hides them all (before submitting)

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
| `results/SEA_NET/<model>/predictions/*.npz` | raw outputs, 3 models only | ensemble voting — keep |
| `results/SEA_NET/ensemble_vote.csv` | ensemble voting result | yes |
| `results/SEA_NET/profile.csv` | params / FLOPs / memory / latency | yes — efficiency table |
| `results/SEA_NET/data_summary.csv` | dataset statistics | yes — setup section |

Deleted on 2026-08-06 as junk (recoverable from git history if ever needed):
`results/SEA_NET/*/figures/` and `results/SEA_NET/*/logs/` — per-run preview
PNGs and training logs, superseded by `results/paper_figures/`.

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

| rank | config | encoder | pooling | web_acc | web_aopcr | params |
|---|---|---|---|---|---|---|
| 1 | `seanet_gated_mean_topk` | `sea_mstcn_sep_gated` | `sea_topk_conjunctive` | 0.954 | 2.293 | 61,740 |
| 2 | `seanet_conjunctive` | `sea_mstcn_sep` | `mil_conjunctive` | 0.954 | 1.502 | 269,083 |
| 3 | `seanet_gated_max_topk` | `sea_mstcn_sep_gated` | `sea_topk_conjunctive` | — | — | — |
| 4 | `seanet_spiketrend_topk` | `sea_mstcn_sep_spiketrend` | `sea_topk_conjunctive` | — | — | — |
| 5 | `seanet_inputgate_mschan` | `sea_multiscale_channels` | `sea_topk_conjunctive` | — | — | — |

**The story for the report:** rank 1 matches rank 2 on accuracy (0.954) but has
a much better AOPCR (2.29 vs 1.50) using **4.4× fewer parameters** (62 K vs
269 K). That is the headline: more interpretable *and* far cheaper, at equal
accuracy. MILLET's own reference numbers are in the last three leaderboard
columns (`millet_acc` 0.8445, `millet_loss` 1.2241, `millet_aopcr` 4.5532).

Read the real CSV before quoting any other number — do not trust this table for
values marked `—`.

---

## 6. Repo map (only what matters)

```
main.py                  <- ONE entry point, all commands (44 KB)
configs/                 <- YAML: models/, datasets, main.yaml
seanet/                  <- our code (encoders, pooling heads, paper figures)
millet/                  <- upstream MILLET, kept diffable — avoid editing
model/                   <- saved weights (git-ignored)
data/                    <- UCR + WebTraffic (871 MB, git-ignored)
results/                 <- all outputs, see section 4
ICLR_2025_Report/        <- THE REPORT, see section 3
Latex/                   <- old presentation leftovers + archi_level1-3.png
report/                  <- OLD abandoned draft, ignored by git — not the report
logs/, OAR.*.stdout      <- raw Grid5000 job dumps, ~108 MB, not used by anything
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
