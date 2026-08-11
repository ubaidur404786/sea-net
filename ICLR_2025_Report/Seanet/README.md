# SEA-Net — final internship report

This folder is **the real report**. It is written, it builds clean, and it is the only
report folder that matters.

- `ICLR_2025_Report/Seanet/` ← **this folder, the real one**
- `ICLR_2025_Report/Template/` — the blank ICLR 2025 sample, reference only, never compile it
- `report/` at the project root — an abandoned older draft, not the report

Status right now: **all sections written, 37 pages, 0 errors, 0 undefined references or
citations.** The one job left is the page count — the examiner limit is 20–30 pages
including the appendix. See section 7 below for where to cut.

```
ICLR_2025_Report/Seanet/
  main.tex                 <- the ONLY file you compile
  preamble.tex             <- packages, colours, macros, the two on/off switches
  refs.bib                 <- the papers cited
  sections/                <- 14 .tex files, one per section (edit these)
  figures/                 <- my own diagrams
      fig02_mil_view.png            the MIL view (Section 4.2)
      fig04_seanet_architecture.pdf the whole model (Section 5.2)
      fig_block_tikz.tex            one encoder block, drawn in TikZ (Section 5.3)
  LATEX_GUIDE.md           <- read this when the build misbehaves
  iclr2025_conference.sty / .bst    the official ICLR 2025 style
  fancyhdr.sty  natbib.sty  math_commands.tex
```

---

## 1. Build it

From **this folder**, in PowerShell:

```powershell
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

Or open any file in the folder and press **Ctrl+Alt+B** (LaTeX Workshop).

Four passes are needed because LaTeX only learns page numbers and citation numbers on the
first pass. **If you see `??` or `[?]` in the PDF, just build again.**

**If Ctrl+S does not refresh `main.pdf`, read `LATEX_GUIDE.md`.** It has the checklist, how
to read a LaTeX error, and the "root file" trap. Every file in `sections/` starts with
`% !TEX root = ../main.tex` — keep that line on any new section file.

---

## 2. What each section contains

Sections 00–03 are in my own words and should not be rewritten; only factual errors get
fixed there. Sections 04–A2 were drafted from the real CSVs and code.

| file | section | what it does |
|---|---|---|
| `00_abstract.tex` | Abstract | the whole story in one paragraph |
| `01_introduction.tex` | Introduction | the problem + the teaser figure |
| `02_context.tex` | Context | lab, supervisor, the microcontroller motivation |
| `03_objectives.tex` | What I had to do | objectives O1–O6 |
| `04_background.tex` | Background | TSC, MIL, the MILLET framework, AOPCR + NDCG |
| `05_architecture.tex` | The SEA-Net encoder | design goals, the block, the variants |
| `06_pooling.tex` | Pooling heads | my seven MIL heads |
| `07_setup.tex` | Experimental setup | datasets, baselines, the pipeline I built |
| `08_results.tex` | Results | WebTraffic table + UCR table, nothing else |
| `09_discussion.tex` | Discussion | what worked, what did not, limitations |
| `10_skills.tex` | Skills | scientific / technical / professional |
| `11_conclusion.tex` | Conclusion | + future work |
| `A1_additional_results.tex` | Appendix A | **every extra result lives here** |
| `A2_reproducibility.tex` | Appendix B | hyperparameters, commands, environment |

**Rule for Section 8:** if removing something would break an argument, it stays in Section 8;
everything else goes to Appendix A. That is why model cost and the ablation grid sit in
`A1_additional_results.tex` and Section 8 only points at them in one line each.

---

## 3. The figures

All eight are real — there are no grey `FIGURE PLACEHOLDER` boxes left.

| # | figure | where | source |
|---|---|---|---|
| 1 | teaser: label vs label + explanation | `01_introduction` | generated |
| 2 | the MIL view (bag / instance / pooling) | `04_background` | `figures/fig02_mil_view.png` |
| 3 | the whole SEA-Net model | `05_architecture` | `figures/fig04_seanet_architecture.pdf` |
| 4 | one encoder block + receptive field | `05_architecture` | `figures/fig_block_tikz.tex` (TikZ) |
| 5 | top-5 models, quality vs cost | `08_results` | generated |
| 6 | encoder × pooling ablation grid | `A1` | generated |
| 7 | win / tie / loss per dataset | `A1` | generated |
| 8 | dataset × model accuracy heatmap | `A1` | generated |

**Figure 4 is optional on purpose.** It is drawn in TikZ, wrapped in
`\ifshowblockfig ... \fi`, and **no paragraph cites it**, so hiding it cannot leave a broken
reference. To remove it, change one line in `preamble.tex`:

```latex
\showblockfigtrue    ->    \showblockfigfalse
```

Figures produced by the code are used with their **project path** and **no file extension**,
because `preamble.tex` adds the project root to the image search path:

```latex
\includegraphics[width=\linewidth]{results/paper_figures/01_main_figures/topk5_multimetric}
```

Leaving the extension off lets LaTeX pick the `.pdf`, which stays sharp at any zoom.
Regenerate every figure and table with `python main.py paper` at the project root.

---

## 4. The helper macros

All defined in `preamble.tex`.

| macro | what it does |
|---|---|
| `\seanet`, `\millet`, `\webtraffic`, `\ucr` | names, always spelled the same way |
| `\bag`, `\emb`, `\instpred`, `\bagpred`, `\interp`, `\nsteps`, `\nclz`, `\dmodel` | the maths symbols |
| `\begin{contribution}...\end{contribution}` | "My contribution." box — **keep in the final PDF** |
| `\begin{priorwork}...\end{priorwork}` | "Existing work (reused)." box — **keep** |
| `\missing{...}` | blue `[CHECK: ...]` flag — something to decide, delete before submitting |
| `\todo{...}` | red `[TODO: ...]` — **none left, keep it that way** |
| `\num{...}`, `\tbd`, `\ms`, `\figph{...}`, `\tabph{...}` | drafting placeholders — **all gone** |

### The two switches in `preamble.tex`

```latex
\showguidestrue    ->    \showguidesfalse     % hides every grey "Writing guide" box
\showblockfigtrue  ->    \showblockfigfalse   % hides Figure 4 (see section 3)
```

The contribution boxes are **not** affected by either switch — they stay, because they are
the part the examiner needs to see.

---

## 5. Tables

Most tables are written by hand from the results CSVs, with a `%` comment right under each
one saying exactly which file and column every number came from. **Do not change a number
without changing that comment.**

Two tables are generated and pulled in with `\input`:

```latex
\input{../../results/paper_figures/tables/table_appendix_full_leaderboard}
```

(the `../../` is because this report sits two folders below the project root).

Available generated tables: `table1_main_results`, `table_appendix_full_leaderboard`,
`table_ranking_accuracy`, `table_efficiency`, `table_appendix_per_dataset`.
Rebuild them all with `python main.py paper`.

---

## 6. Where the numbers come from

Every number in the report traces back to `results/SEA_NET/`:

- `leaderboard.csv` — all 72 models ranked by WebTraffic accuracy
- `<model>/results.csv` — one row per dataset per seed
- `profile.csv` — parameters, size, prediction time
- `results/paper_figures/` — every generated figure and table

Rebuild in this order, because each step reads the one before it:

```powershell
python main.py leaderboard
python main.py results
python main.py report
python main.py paper
```

---

## 7. Page count — the one job left

**Now: 37 pages. Target: 20–30 including the appendix** (examiner rule, AIEDA400).

Cut in this order, cheapest first:

1. `\showblockfigfalse` — removes Figure 4, about half a page, breaks nothing.
2. Appendix A — the biggest single target. Extra experiments and the additional figures go
   first; the full leaderboard and the ablations are the ones worth keeping.
3. `04_background` — the MILLET recap can be shortened further with a pointer to the paper.
4. The per-dataset and per-seed tables in Appendix A can become a sentence each.

Check the count after every cut, and never cut something Section 8 or 9 cites — the build
will tell you with an undefined reference, but only if you read the log.

---

## 8. Final checklist before uploading to Moodle (AIEDA400)

- [ ] no red `[TODO: ...]` in the PDF
- [ ] no blue `[CHECK: ...]` left (two remain today: the `millet_paper` UCR row in Section 8.2,
      and the ensemble numbers in Appendix A)
- [ ] no grey `FIGURE PLACEHOLDER` / `TABLE PLACEHOLDER` boxes
- [ ] `\showguidesfalse` set in `preamble.tex`
- [ ] every figure and table is cited from the text — **except Figure 4, which is
      deliberately not cited so it can be switched off**
- [ ] no `??` and no `[?]` in the PDF
- [ ] page count between 20 and 30
- [ ] PDF smaller than 50 MB
- [ ] "I" for my own work, "we" for the team's — check the whole document once
- [ ] each contribution box says clearly what is mine
- [ ] spell check run (LTeX extension in VS Code)
