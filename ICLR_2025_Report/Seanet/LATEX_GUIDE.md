# LaTeX survival guide — SEA-Net report

My own notes for when Ctrl+S stops updating `main.pdf`. Read the first section,
it fixes the problem 9 times out of 10.

---

## 0. The one thing to understand: the "root file"

LaTeX does not compile `sections/03_objectives.tex`. That file has no
`\documentclass` and no `\begin{document}` — on its own it is not a document,
it is a **piece** of one. The only real document here is:

```
ICLR_2025_Report/Seanet/main.tex
```

`main.tex` pulls every piece in with `\input{sections/03_objectives}`. So when I
save a section file, LaTeX Workshop has to answer one question first:

> "which `main.tex` does this little file belong to?"

That answer is called the **root file**. If it picks the wrong root, it happily
compiles *something* — just not my report. No error, no warning. The PDF simply
never changes. That is exactly the bug I hit on 2026-08-07.

### How LaTeX Workshop picks the root (in order)

1. A magic comment on the **first line** of the file I am editing:
   `% !TEX root = ../main.tex`  ← strongest, always wins
2. The file I am editing, if it contains `\documentclass`
3. A workspace-wide search for any `.tex` with `\documentclass`, filtered by
   `latex-workshop.latex.search.rootFiles.include` / `.exclude`
4. Whatever it picked last time

Step 3 was the problem: this repo has **two** files with `\documentclass`:

| file | what it is |
|---|---|
| `ICLR_2025_Report/Seanet/main.tex` | the real report |
| `ICLR_2025_Report/Template/iclr2025_conference.tex` | the blank ICLR sample |

It kept choosing the Template one, so every Ctrl+S rebuilt
`Template/iclr2025_conference.pdf` and my `main.pdf` stayed frozen.

### The fix that is now in place (two layers, so it cannot come back)

**Layer 1 — a magic comment on line 1 of every piece file.** Already added to
all 14 files in `sections/`, plus `preamble.tex` and `math_commands.tex`:

```latex
% !TEX root = ../main.tex     <- in sections/*.tex  (one folder up)
% !TEX root = main.tex        <- in preamble.tex    (same folder)
```

**Rule for me: any new `.tex` file I create in `sections/` must start with that
line.** It is one line and it removes all guessing.

**Layer 2 — tell the extension where to search**, in `.vscode/settings.json`:

```jsonc
"latex-workshop.latex.search.rootFiles.include": [
  "ICLR_2025_Report/Seanet/main.tex"
],
"latex-workshop.latex.search.rootFiles.exclude": [
  "ICLR_2025_Report/Template/**",
  "report/**"
],
```

### How to check which root is active, right now

Look at the **VS Code status bar, bottom left**. LaTeX Workshop shows the root
file name there. If it does not say `main.tex`, that is my bug.

Or: `Ctrl+Shift+P` → **LaTeX Workshop: Show compilation log**, and read the very
first lines — they name the file being compiled.

---

## 1. Quick checklist when the PDF does not update

Go down this list in order. Stop at the first one that is wrong.

1. **Did the file actually save?** A white dot on the tab means unsaved. `Ctrl+S`.
2. **Is the root file correct?** Status bar, bottom left. Must be `main.tex`.
   If not → `Ctrl+Shift+P` → **LaTeX Workshop: Set LaTeX root file**.
3. **Is the file actually `\input` by `main.tex`?** Open `main.tex` and look. A
   file that is not listed there is simply not part of the report — LaTeX will
   never read it, no matter how much I edit it.
4. **Did the build fail?** `Ctrl+Shift+P` → **LaTeX Workshop: Show compilation
   log**. Search for a line starting with `!`. That is the real error. When
   pdflatex fails it keeps the **old** PDF on disk, which looks exactly like
   "nothing happened".
5. **Is a build still running?** The status bar shows a spinning icon. A full
   4-pass build of this report takes ~20–40 s. Wait for it.
6. **Is the PDF viewer stuck?** Close the PDF tab and reopen it with
   `Ctrl+Alt+V` (View LaTeX PDF).
7. **Stale helper files.** Rare, but if page numbers / references go crazy,
   delete `main.aux`, `main.toc`, `main.out` and build again (see §3).
8. **Locked PDF.** If I ever opened `main.pdf` in Adobe Reader, Windows locks
   the file and pdflatex cannot overwrite it. The log then says
   `I can't write on file main.pdf`. Close Adobe.

---

## 2. Building by hand (when VS Code is being unhelpful)

From `ICLR_2025_Report/Seanet/`:

```powershell
cd "D:\Documents\Intership TSC\Projects\SEA_NET\ICLR_2025_Report\Seanet"
pdflatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex
bibtex   main
pdflatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex
pdflatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex
```

A good run ends with:

```
Output written on main.pdf (33 pages, 613769 bytes).
```

If that line is missing, the build failed — read upwards for the `!`.

### Why four passes?

LaTeX reads the file **once, top to bottom**, like a person reading a book for
the first time. On pass 1 it cannot know that Figure 3 will land on page 14, or
what number `\citep{early2024millet}` should get, because it has not read that
far yet. So:

- **pass 1** — writes down everything it learned into `main.aux`
  (labels, page numbers, which citation keys were used)
- **bibtex** — reads `main.aux`, looks those keys up in `refs.bib`, formats them
  with `iclr2025_conference.bst`, and writes `main.bbl`
- **pass 2** — pulls `main.bbl` into the document; the bibliography now exists,
  which shifts every page number after it
- **pass 3** — with the page numbers settled, every `\ref` and `\cite` finally
  points to the right place

This is exactly what Overleaf's green button does behind the scenes.

**Shortcut:** if I only changed wording and touched no `\cite`, `\ref` or
`\label`, one `pdflatex` pass is enough. That is the second recipe in
`settings.json` ("pdflatex only, fast").

---

## 3. The helper files, in plain words

| file | who writes it | what it holds |
|---|---|---|
| `main.aux` | pdflatex | labels, page numbers, citation keys — the notes for the next pass |
| `main.bbl` | bibtex | the finished, formatted bibliography |
| `main.blg` | bibtex | bibtex's own log (missing keys show up here) |
| `main.log` | pdflatex | the full build log — **the file to read when something breaks** |
| `main.toc` | pdflatex | the table of contents |
| `main.out` | hyperref | the PDF bookmarks sidebar |
| `main.synctex.gz` | pdflatex | the map for Ctrl+click jumping between source and PDF |
| `main.pdf` | pdflatex | the report |

All of them are hidden from the VS Code explorer (`files.exclude`) and ignored
by git. **Never edit them by hand.** Deleting them is always safe — they are
rebuilt in seconds.

Nuclear reset, if references or the table of contents go strange:

```powershell
cd "D:\Documents\Intership TSC\Projects\SEA_NET\ICLR_2025_Report\Seanet"
Remove-Item main.aux, main.bbl, main.blg, main.log, main.out, main.toc, main.synctex.gz -ErrorAction SilentlyContinue
```

Then run the four-pass build from §2.

---

## 4. Reading a LaTeX error

The log is long and mostly noise. Only lines starting with `!` matter.

```
./sections/03_objectives.tex:74: Undefined control sequence.
l.74 ...we got 89\% accuracy to \milet
                                       {} of the published number
```

Read it as: **file : line : what went wrong**, then the line is shown cut in two
at the exact spot where LaTeX gave up. Here `\milet` is a typo for `\millet`.

The `-file-line-error` flag in our settings is what makes the `file:line:` part
appear, so errors are clickable in the VS Code Problems panel.

### The errors I actually hit

| message | what it really means |
|---|---|
| `Undefined control sequence` | typo in a command, or the package that defines it is missing from `preamble.tex` |
| `Missing $ inserted` | a maths-only character (`_`, `^`, `\alpha`) used in normal text |
| `Missing } inserted` | unbalanced braces — usually one `}` too few or too many earlier |
| `File 'xyz.png' not found` | wrong path in `\includegraphics`, see §6 |
| `Citation 'key' undefined` | the key is not in `refs.bib`, **or** bibtex has not been run yet |
| `Reference 'fig:x' undefined` | `\label{fig:x}` is missing, or only one pass has run |
| `There were undefined references` | just run the build again — nearly always harmless |
| `Emergency stop` / `texput.log` appears | pdflatex was started with **no file name**. Something ran `pdflatex` on nothing. Check the root file. |

**A special character rule:** `% $ & # _ { } ~ ^ \` are reserved. To print them
literally write `\%`, `\$`, `\&`, `\#`, `\_`, `\{`, `\}`.
This bites me most with percentages: `89\%` is correct, `89%` silently comments
out the rest of the line — the text just vanishes from the PDF with no error at
all. If a sentence disappears, look for an unescaped `%`.

---

## 5. Our own commands (defined in `preamble.tex`)

Names, so the spelling never drifts:

| command | prints |
|---|---|
| `\seanet` | SEA-Net |
| `\millet` | MILLET |
| `\webtraffic` | WebTraffic |
| `\ucr` | UCR |

Drafting helpers (these are meant to be **removed before submitting**):

| command | what it does |
|---|---|
| `\todo{...}` | red `[TODO: ...]` in the text |
| `\note{...}` | blue `[...]`, a reminder to myself |
| `\num{web_acc}` | red `<web_acc>`, a number I still have to read off a results file |
| `\tbd` | a red `--` in a table cell |
| `\ms` | a red `--±--` for a mean±std cell |
| `\figph[0.8]{description}` | grey "FIGURE PLACEHOLDER" box |
| `\tabph{description}` | grey "TABLE PLACEHOLDER" box |

Boxes:

| environment | meaning |
|---|---|
| `guide` | grey "Writing guide" box — **switch off with `\showguidesfalse`** in `preamble.tex` before the final PDF |
| `contribution` | left bar, "My contribution" — **stays** in the final PDF |
| `priorwork` | left bar, "Existing work (reused)" — **stays** |

Before submitting, search the whole `sections/` folder for `\todo`, `\num` and
`\begin{guide}` and clear them out.

---

## 6. Figures and paths

`preamble.tex` sets:

```latex
\graphicspath{{./}{figures/}{../../}}
```

Three places are searched, in order: next to `main.tex`, then the local
`figures/` folder, then **the project root** (two folders up). That last one is
why a figure made by the code can be written with its project path exactly as
`python main.py paper` produces it:

```latex
\includegraphics[width=\linewidth]{results/paper_figures/01_main_figures/fig1_teaser}
```

Rules that save time:

- **Use forward slashes `/`**, never Windows `\`. A backslash starts a command.
- **Leave the extension off.** `{...fig1_teaser}` lets LaTeX choose `.pdf` over
  `.png` when both exist (pdf is sharper, it is vector).
- **No spaces in file names.** LaTeX handles them badly.
- Set the size with `[width=\linewidth]`, not by resizing the image file.

---

## 7. Citations

Keys live in `refs.bib`. Two forms:

```latex
... as shown by MILLET \citep{early2024millet}.   % -> (Early et al., 2024)
\citet{early2024millet} showed that ...           % -> Early et al. (2024)
```

`\citep` = parenthetical, at the end of a sentence.
`\citet` = textual, when the authors are the subject of my sentence.

A new citation needs **the full 4-pass build** (§2), because bibtex has to run.
One pdflatex pass alone will leave `[?]` in the PDF.

If a citation shows as `[?]`: check `main.blg` — bibtex writes
`Warning--I didn't find a database entry for "somekey"` there when a key is
missing from `refs.bib`.

---

## 8. Cross-references

```latex
\label{sec:objectives}      % put it right after \section{...}
\Cref{sec:objectives}       % prints "Section 3"  (capital C at sentence start)
\cref{fig:timeline}         % prints "figure 2"
```

`cleveref` writes the word "Section"/"Figure"/"Table" for me, so I never type it
by hand and the two can never disagree.

**Important ordering rule inside `preamble.tex`:** `hyperref` must be loaded
before `cleveref`, and both must come last. Changing that order breaks links in
strange ways.

Naming convention I follow: `sec:`, `fig:`, `tab:`, `eq:` prefixes. A `\label`
must come **after** the `\caption` inside a figure or table, otherwise it points
at the wrong number.

---

## 9. Ctrl+click between source and PDF (SyncTeX)

- **source → PDF**: `Ctrl+Alt+J` jumps the preview to where my cursor is
- **PDF → source**: `Ctrl+click` anywhere in the PDF jumps the editor there

This needs `-synctex=1` (already in our settings) and a fresh
`main.synctex.gz`. If it jumps to the wrong place, the synctex file is stale —
just rebuild.

---

## 10. Useful commands, all in one place

| what I want | how |
|---|---|
| Build now | `Ctrl+Alt+B` |
| Open the PDF preview | `Ctrl+Alt+V` |
| Jump PDF to cursor | `Ctrl+Alt+J` |
| Read the build log | `Ctrl+Shift+P` → *LaTeX Workshop: Show compilation log* |
| Change the root file | `Ctrl+Shift+P` → *LaTeX Workshop: Set LaTeX root file* |
| Kill a stuck build | `Ctrl+Shift+P` → *LaTeX Workshop: Terminate current compilation* |
| Reload the whole window | `Ctrl+Shift+P` → *Developer: Reload Window* |

Reloading the window is the right move after editing `.vscode/settings.json` —
LaTeX Workshop caches the root file and only re-reads settings on reload.

---

## 11. Things specific to this project — do not break them

- `ICLR_2025_Report/Seanet/` is the **real** report. `report/` at the repo root
  is an abandoned old draft, and `ICLR_2025_Report/Template/` is the blank ICLR
  sample kept for reference. Never edit those two thinking they are the report.
- `main.pdf` is git-ignored on purpose. It is rebuilt from source; committing it
  would only create merge conflicts.
- Numbers and figures come from `results/paper_figures/` and
  `results/SEA_NET/*/results.csv`, refreshed with `python main.py paper`. Never
  retype a number by hand from memory — copy it from the CSV.
