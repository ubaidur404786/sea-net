# SEA-Net — final internship report (LaTeX template)

This folder holds the **template** of the final internship report. Nothing here is a
finished text yet: every section has a title, a grey **Writing guide** box telling you
what to write, guiding questions, and red `[TODO: ...]` markers where your own words go.

Fill the sections in your own words first. Ask me to polish the language afterwards.

```
ICLR_2025_Report/Seanet/
  main.tex                 <- the only file you compile
  preamble.tex             <- packages, colours, macros (\todo, \figph, \ms, maths)
  refs.bib                 <- the papers you cite
  sections/                <- one .tex file per section (edit these)
  figures/                 <- put your own images here (empty for now)
  iclr2025_conference.sty  <- the official ICLR 2025 style (copied from ../Template)
  iclr2025_conference.bst
  fancyhdr.sty  natbib.sty  math_commands.tex
```

---

## 1. Build it

From this folder, in PowerShell:

```powershell
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

Or open `main.tex` in VS Code and press **Ctrl+Alt+B** (LaTeX Workshop).

Four passes are needed because LaTeX only learns the page numbers and citation numbers
on the first pass. **If you see `??` or `[?]` in the PDF, just build again.**

---

## 2. Writing order (recommended)

Write the sections in the order that is easiest, not in the order they are printed:

| order | file | why this order |
|---|---|---|
| 1 | `03_objectives.tex` | you already know what you were asked to do |
| 2 | `05_architecture.tex`, `06_pooling.tex` | your own work, still fresh |
| 3 | `07_setup.tex` | facts, quick to write |
| 4 | `08_results.tex` | fill the tables from the result files |
| 5 | `09_discussion.tex` | needs the tables to exist first |
| 6 | `04_background.tex` | now you know exactly what the reader needs |
| 7 | `02_context.tex`, `10_skills.tex` | short and easy |
| 8 | `01_introduction.tex`, `11_conclusion.tex` | write these once the story is fixed |
| 9 | `00_abstract.tex` | always last |

---

## 3. Page budget (examiner rule: 20–30 pages **including** the appendix)

| section | file | target |
|---|---|---|
| title + abstract + contents | `main.tex`, `00_abstract` | 2 |
| 1. Introduction | `01_introduction` | 1.5–2 |
| 2. Context | `02_context` | ≤ 1 |
| 3. What I had to do | `03_objectives` | 1.5–2 |
| 4. Background | `04_background` | 3–3.5 |
| 5. Architecture | `05_architecture` | 3.5–4 |
| 6. Pooling heads | `06_pooling` | 4–4.5 |
| 7. Experimental setup | `07_setup` | 2.5–3 |
| 8. Results | `08_results` | 4–4.5 |
| 9. Discussion | `09_discussion` | 2–2.5 |
| 10. Skills | `10_skills` | 1–1.5 |
| 11. Conclusion | `11_conclusion` | 1 |
| References | `refs.bib` | 1.5 |
| Appendix A + B | `A1_`, `A2_` | 4–6 |

That lands at about **28 pages**. If you go over, cut the appendix first, then
`04_background`.

**Measured right now** (empty template, nothing written yet):

| version | pages |
|---|---|
| with the writing guides shown (`\showguidestrue`) | 33 |
| with the writing guides hidden (`\showguidesfalse`) | 25 |

So the skeleton alone already fills the space. As you replace a `[TODO: ...]` by a
real paragraph the page count grows, and as you replace a grey `FIGURE PLACEHOLDER`
by a real figure it usually shrinks. Check the count again when the report is half
written, and cut early if it drifts above 30.

---

## 4. The helper macros

All defined in `preamble.tex`.

| macro | what it does |
|---|---|
| `\todo{...}` | red inline note — **delete all of these before submitting** |
| `\note{...}` | blue note to yourself |
| `\num{web_acc}` | a number you still have to read off a results file |
| `\tbd` | an empty table cell (`--`) |
| `\ms` | an empty "mean ± std" table cell |
| `\figph{description}` | grey box standing in for a figure you have not drawn yet |
| `\tabph{description}` | same, for a table |
| `\begin{guide}...\end{guide}` | the grey **Writing guide** box (drafting only) |
| `\begin{contribution}...\end{contribution}` | "My contribution." box — **keep in the final PDF** |
| `\begin{priorwork}...\end{priorwork}` | "Existing work (reused)." box — **keep** |
| `\seanet`, `\millet`, `\webtraffic`, `\ucr` | names, always spelled the same way |
| `\bag`, `\emb`, `\instpred`, `\bagpred`, `\interp`, `\nsteps`, `\nclz` | the maths symbols |

**Before submitting**, in `preamble.tex` change

```latex
\showguidestrue    ->    \showguidesfalse
```

That hides every writing-guide box in one go. The contribution boxes stay, because they
are the part the examiner needs to see.

---

## 5. Figures

`figures/` is empty on purpose. Every figure in the report is a `\figph{...}` grey box
describing what to draw. When a picture is ready:

```latex
% before
\figph{Overall SEA-Net architecture: ...}

% after
\includegraphics[width=\linewidth]{figures/seanet_architecture}
```

Figures already produced by the code can be used with their project path, because
`preamble.tex` adds the project root to the search path:

```latex
\includegraphics[width=\linewidth]{results/paper_figures/01_main_figures/pareto_web_acc_vs_params}
```

Leave the file extension off — LaTeX then picks the `.pdf`, which stays sharp at any zoom.

The figures the report is waiting for:

1. Figure — teaser: label vs label + explanation (`01_introduction`)
2. Figure — the MIL view: bag / instance / pooling (`04_background`)
3. Figure — how AOPCR is computed (`04_background`)
4. Figure — **overall SEA-Net architecture** (`05_architecture`) ← the new diagram
5. Figure — multi-scale feature extraction, one block (`05_architecture`)
6. Figure — **proposed pooling architecture**, all heads in one picture (`06_pooling`)
7. Figure — training pipeline (`07_setup`)
8. Figure — result comparison bars (`08_results`)
9. Figure — critical-difference diagram (`08_results`)
10. Figure — qualitative visualisation (`08_results`)
11. Figure — accuracy vs cost, Pareto (`08_results`)
12. Figure — training curves (`08_results`)
13. Figure — encoder × pooling ablation grid (`08_results`)
14. Figures — extra ones in the appendix (`A1_additional_results`)

---

## 6. Tables

Every table is already there as a skeleton with `\tbd` / `\ms` cells. Two ways to fill one:

- **by hand**, reading the value from `results/SEA_NET/...csv`, or
- **automatically**, replacing the whole skeleton with the generated table:

```latex
\input{../../results/paper_figures/tables/table1_main_results}
```

(the `../../` is because this report sits two folders below the project root).

Available generated tables: `table1_main_results`, `table_appendix_full_leaderboard`,
`table_ranking_accuracy`, `table_efficiency`, `table_appendix_per_dataset`.
Regenerate them all with `python main.py paper` at the project root.

---

## 7. Final checklist before uploading to Moodle (AIEDA400)

- [ ] no red `[TODO: ...]` left anywhere in the PDF
- [ ] no `\num{...}` placeholders left
- [ ] no grey `FIGURE PLACEHOLDER` / `TABLE PLACEHOLDER` boxes left
- [ ] `\showguidesfalse` set in `preamble.tex`
- [ ] every figure and table is referenced from the text (`\Cref{fig:...}`)
- [ ] no `??` and no `[?]` in the PDF
- [ ] page count is between 20 and 30
- [ ] PDF smaller than 50 MB
- [ ] "I" for my own work, "we" for the team's work — check the whole document once
- [ ] each contribution box says clearly what is mine
- [ ] spell check run (LTeX extension in VS Code)
