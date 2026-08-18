# PAPER_PLAN.md — step-by-step plan for the ICLR 2027 paper

Read this file and `PAPER_ISSUES.md` before working on the paper.

- **Folder:** `ICLR_2025_Report/Paper_ICLR2027/` — the paper lives here, nowhere else
- **Branch:** `iclr1`
- **Never touch:** `ICLR_2025_Report/Seanet/` — that is the internship report
- **Rules:** you run all `git` and all `python` commands; I only write files and
  give you the commands to copy-paste

---

## Part 1 — The story

> **2026-08-17 — the story changed. Draft 2 supersedes Draft 1.**
> Draft 1 was written as a *study paper* ("Selection, Not Scale"). You read it
> and said the central idea was not represented: the real research problem is
> **classification + interpretation on tiny devices**, reached by *MILLET
> baseline → cheaper encoder via a TCN bottleneck → better pooling*. Draft 2 is
> written to that. Draft 1 is archived, unchanged, in `sections/_draft1/`.
> Everything from "Why the report cannot simply be cut down" to the end of the
> C1–C4 table below is **Draft 1's reasoning, kept for the record only.**

### Draft 2 — the thesis, in one sentence

> \MILLET gives a time series classifier both a prediction and a per-time-step
> explanation, but its cost is set by an InceptionTime backbone that is one to
> two orders of magnitude above microcontroller budgets. Replacing that backbone
> with a bottlenecked multi-scale separable dilated encoder, and the pooling
> with per-class Top-$k$ selection, keeps both capabilities at $10.3\times$
> fewer parameters and $11.1\times$ fewer FLOPs — for about two accuracy points
> on the UCR archive.

### Draft 2 — the five research questions the paper answers

| | question | where | verdict |
|---|---|---|---|
| RQ1 | Is classification accuracy retained? | §5.1, §5.3 | Yes on WebTraffic (+6.0 pts); **no** on UCR-85 (−1.8 vs matched, −3.4 vs published) |
| RQ2 | Is the interpretation retained? | §5.1 | Improved: NDCG@n 0.772 vs 0.677; AOPCR tied |
| RQ3 | How much cost is removed? | §5.2 | Params 10.3×, FLOPs 11.1×; latency only 1.6×; **peak memory worse** |
| RQ4 | What does the bottleneck contribute? | §5.4 | −28 % params at no measurable accuracy cost |
| RQ5 | What does Top-$k$ contribute? | §5.4 | +0.044/+0.059 NDCG@n (3–4 seed sd); accuracy effect not claimed |

### Draft 2 — what is NOT claimed

- No on-device deployment. No microcontroller was used. int8 numbers are
  arithmetic, labelled as such.
- No novelty for separable convolutions, bottlenecks, or $k$-max MIL selection
  (issue I23) — only for their combination and its measurement.
- No state of the art on UCR.
- No comparison against MILLET's published AOPCR (§5.5 says it is not
  comparable, so quoting it would contradict our own finding).

---

### Draft 1's reasoning (archived — do not act on this section)

#### Why the report cannot simply be cut down to 9 pages

The report is written as *"here is SEA-Net, a new architecture"*. As an ICLR
paper that framing loses, for one reason: an architecture paper is judged on
whether the architecture wins, and the honest headline is a win on **one
synthetic dataset that the models were selected on** (issue I7), with a much
smaller effect on the 85 real UCR datasets. A reviewer finds that in ten
minutes and the paper is done.

So the paper is not an architecture paper. It is an **empirical study of what
actually controls accuracy and explanation quality** in inherently
interpretable time series classification — and the 72-model sweep, which is
merely supporting material in the report, becomes the main evidence.

### The thesis, in one sentence

> In MIL-based inherently interpretable time series classification, explanation
> quality is controlled by **selection inside the aggregator**, accuracy is
> controlled far more by **training budget and capacity** than by the backbone
> design — and the metric the field uses to report faithfulness (AOPCR) is not
> comparable across papers at all.

### The four claims, and the evidence for each

| # | Claim | Evidence we already have | Strength |
|---|---|---|---|
| **C1** | **Selection, not capacity, drives explanation quality.** Averaging only the top-κ evidence values makes the interpretation exactly zero where the model looked at nothing. | The κ ablation is a *controlled* experiment because κ = 1.0 **is** the class-wise baseline exactly. NDCG falls monotonically 0.777 → 0.722 as κ → 1. Second independent knob, the attention-entropy penalty, points the same way (0.777 → 0.765 when switched off). Cost is about one accuracy point. | strong |
| **C2** | **Faithful interpretability does not need a big backbone; the UCR gap is capacity, not architecture.** | 41 K params beat the 424 K baseline on accuracy *and* AOPCR *and* NDCG at once on the only dataset with ground truth. On UCR-85 the **wide** SEA-Net encoder outranks the baseline (11.53 vs 12.60), the **narrow** ones do not (14.0–15.8). That locates the effect. | strong (issue I2) |
| **C3** | **AOPCR cannot be compared across training configurations, so cross-paper AOPCR comparisons in this literature are meaningless.** | Same architecture, same code, same data: 2.569 vs **13.268**, a 5.2× swing from the recipe alone. | strong and original (issue I3) |
| **C4** | **"Capacity-safe" is not "optimisation-safe."** A head that provably reduces to the baseline still fails, because nothing in the loss pushes it back there. | Five of seven heads provably contain the baseline. Dual-stream is still the worst head tested (0.744 mean, worst pairing 0.556). Mechanism is identifiable: λ starts at 0.05 and never returns. | good, honest negative result (issue I11) |

**C3 is the most original thing in this work.** It is a criticism of the
evaluation practice of an ICLR 2024 paper, backed by a controlled measurement.
Nothing else here is as hard to argue with.

### Working title (to be revisited in Phase 6)

> **Selection, Not Scale: What Controls Accuracy and Faithfulness in Inherently
> Interpretable Time Series Classification**

### What we deliberately do NOT claim

- Not "state of the art on UCR" — we are not.
- Not "SEA-Net is the best architecture" — the sweep does not support it.
- Not any AOPCR comparison with a published number — see C3, that would
  contradict our own finding.

---

## Part 2 — The phases

Each phase ends with something you can look at. Nothing starts before the
phase before it is signed off.

### Phase 0 — Set up (do this first)

- [ ] **0.1** You create the branch:
      ```powershell
      git checkout -b iclr1
      ```
- [ ] **0.2** I create `Paper_ICLR2027/` and copy in the official ICLR 2027
      style files (`iclr2027_conference.sty`, `.bst`, `fancyhdr.sty`,
      `natbib.sty`, `math_commands.tex`) from `ICLR_2025_Report/iclr2027/`.
      The style files are copied, never edited — *"tweaking the style files may
      be grounds for rejection."*
- [ ] **0.3** These two files (`PAPER_PLAN.md`, `PAPER_ISSUES.md`) exist. ✅
- [ ] **0.4** You commit the setup.

### Phase 1 — Agree the story ← **you sign off here**

- [ ] **1.1** You read Part 1 above and either accept the thesis and the four
      claims, or tell me what to change.
- [ ] **1.2** Freeze the claim list. After this point a new claim needs a new
      entry in `PAPER_ISSUES.md`, so we never quietly grow the paper.

### Phase 2 — Lock every number before writing prose

Writing first and checking later is how wrong numbers get into a paper. So the
numbers come first.

- [ ] **2.1** I build `NUMBERS.md` in this folder: one row per number the paper
      will print, with the exact file and column it came from.
- [ ] **2.2** Re-check the three that matter most, because the paper stands on
      them: the κ ablation (all seed 0, issue I5), the UCR mean ranks (issue
      I2), and the two AOPCR values (issue I3).
- [ ] **2.3** Any number I cannot trace to a file does **not** go in the paper.

### Phase 3 — Skeleton and page budget

- [ ] **3.1** I write `main.tex` + `preamble.tex`, anonymous, 9-page layout,
      one `\input` per section — same working style as the report.
- [ ] **3.2** Section files with the page budget written into each one as a
      comment:

      | # | Section | Pages | Carries |
      |---|---|---|---|
      | 1 | Introduction + contributions | 1.25 | the gap, C1–C4 |
      | 2 | Background: MIL, conjunctive pooling, AOPCR / NDCG | 1.00 | notation |
      | 3 | The design space we sweep (encoders + 7 heads as 2 slots) | 1.25 | method |
      | 4 | Experimental protocol | 0.50 | + selection caveat (I7) |
      | 5 | C1 — selection drives faithfulness | 1.25 | κ table, entropy table |
      | 6 | C2 — faithfulness is cheap; the UCR gap is capacity | 1.50 | main table, UCR table, CD figure |
      | 7 | C3 — AOPCR is not comparable across configurations | 0.75 | the 2.57 / 13.27 table |
      | 8 | C4 — capacity-safe is not optimisation-safe | 0.75 | head ablation |
      | 9 | Related work | 0.40 | |
      | 10 | Limitations and conclusion | 0.40 | |
      | | **Total** | **9.05** | trim in Phase 6 |

      Then, not counted toward the limit: reproducibility statement, **AI use
      statement (required, issue I10)**, references, appendices.

- [ ] **3.3** `refs.bib` — copy from the report, drop anything uncited, and add
      what a study paper needs: a recent TSC bake-off, and work on evaluating
      explanation faithfulness (to support C3).

### Phase 4 — Write the paper, claim by claim

Order matters: the evidence sections are written **before** the introduction,
because you cannot promise in the introduction what the evidence does not
deliver.

- [ ] **4.1** Section 4, protocol (shortest, fixes the vocabulary)
- [ ] **4.2** Section 6, C2 — the main table
- [ ] **4.3** Section 5, C1 — the κ story
- [ ] **4.4** Section 7, C3 — AOPCR
- [ ] **4.5** Section 8, C4 — the negative result
- [ ] **4.6** Section 3, the design space
- [ ] **4.7** Section 2, background
- [ ] **4.8** Section 1, introduction — written last, promises only what 5–8 deliver
- [ ] **4.9** Sections 9–10, related work and conclusion
- [ ] **4.10** Abstract — written after everything else
- [ ] **4.11** Reproducibility statement + AI use statement

### Phase 5 — Figures and tables

- [ ] **5.1** Main body gets at most **4 floats** — pages are the scarce
      resource. Current shortlist:
      1. Teaser: same series, three models, prediction + explanation (exists)
      2. The κ curve, NDCG and accuracy against κ (**new figure needed**)
      3. ~~Critical-difference diagram on UCR-85~~ (dropped 2026-08-17, I29)
      4. ~~Quality against cost (`pareto_web_acc_vs_params`)~~ (dropped
         2026-08-18, I30)
- [ ] **5.2** The κ figure does not exist yet. Either I add a small plotting
      function to `seanet/paper/` and you run `python main.py paper`, or we
      keep the κ table only and save the page. Decide in Phase 5.
- [ ] **5.3** Everything else goes to the appendix: the full 72-model
      leaderboard, the encoder × head grid, per-dataset UCR, cost table, seeds.

### Phase 6 — Compliance pass ← nothing is finished before this

- [ ] **6.1** **Anonymity sweep** (issue I1): no GitHub URL, no names, no
      company, no "internship", no Grid'5000 site, and check the PDF metadata.
- [ ] **6.2** **Page count** — main text ≤ 9 pages, measured on the built PDF.
- [ ] **6.3** AI use statement present (issue I10).
- [ ] **6.4** Reproducibility statement present, pointing at the anonymous repo.
- [ ] **6.5** Appendices after the references, clearly marked.
- [ ] **6.6** No `\todo`, no placeholder macros, no `??`, no `[?]`.
- [ ] **6.7** Every number re-checked against `NUMBERS.md`.
- [ ] **6.8** Every issue in `PAPER_ISSUES.md` is either **DONE** or explicitly
      accepted as a stated limitation.
- [ ] **6.9** Build clean: 0 errors, 0 undefined references, 0 overfull boxes.

---

## Part 3 — Decisions taken (2026-08-17)

| question | decision | what it forces |
|---|---|---|
| Framing | **Study paper** — the thesis in Part 1 | SEA-Net is the instrument, the sweep is the evidence; the introduction must promise findings, not an architecture |
| Compute | **None left, existing results only** | C1 stays single-seed (issue I19); no new dataset for NDCG (issue I8); both become stated limitations, not hidden ones |
| Intent | **Real ICLR 2027 submission** | Full anonymity is mandatory (I1) and the anonymous repository must be built (I18). Abstract deadline 18 Sep 2026, paper 25 Sep 2026, both AOE |

Because there is no compute, the paper's honesty is its defence. Every claim
that a run would have strengthened is written as what it is, with the seed
spread printed next to it.

---

## Progress log

| date | phase | what happened |
|---|---|---|
| 2026-08-17 | 0 | Folder, plan and issue list created; 16 issues found; numbers verified against `leaderboard.csv` and `table_ranking_accuracy.csv` |
| 2026-08-17 | 0 | Style files copied in. Three decisions taken (study paper / no compute / real submission). `profile.csv` checked, which found issues I17 and I19 |
| 2026-08-17 | 2 | `NUMBERS.md` written — every number the paper prints, traced to its source file and column. κ seed-0 values verified directly against `results.csv` |
| 2026-08-17 | 3–4 | **First draft complete and building clean.** `main.tex`, `preamble.tex`, `refs.bib`, 12 section files. 0 errors, 0 undefined references, 0 undefined citations, 0 overfull boxes. 17 pages total |
| 2026-08-17 | — | Two gaps found in the draft itself: **12 pages of main text against a 9-page limit** (I20) and **no figures yet** (I21). Both must be solved together, since figures cost about a page |
| 2026-08-17 | — | **Story changed on your instruction.** Draft 1 archived to `sections/_draft1/`; Draft 2 written to the tiny-device framing (see Part 1) |
| 2026-08-17 | novelty | Literature check run before writing: separable conv, squeeze/expand bottleneck, $k$-max MIL selection and the MCU budget framing all already exist. Six new references added; the paper now states plainly that no component is novel in isolation (issue I23) |
| 2026-08-17 | 3–5 | **Draft 2 complete.** New sections `02_related`, `03_method`, `04_setup`, `05_results`, `06_discussion`, `07_conclusion`; abstract, introduction, statements and appendix rewritten. New TikZ architecture figure in `figures/fig_arch.tex`. 72-row leaderboard generated from the CSV rather than typed |
| 2026-08-17 | 6 | **Compliance pass passed.** Main text ends exactly at page 9 (measured, not estimated); 17 pages total; 0 errors, 0 undefined references, 0 undefined citations, 0 overfull boxes, no `??`. Anonymity sweep clean including PDF metadata (`/Author` empty). AI use statement present. Statements and appendices after the references |
| 2026-08-17 | 6 | **Critical-difference diagram removed on your instruction** — it was confusing to read. Figure float and its paragraph cut from the appendix, and the "critical-difference analysis" promise in §4 Metrics reduced to mean rank. UCR evidence is now `tab:app_ucr` (mean rank + win/tie/loss) plus one honest sentence about non-significant middle-of-field differences. Rebuild: 16 pages, main text still ends exactly at page 9, 0 errors / 0 undefined / 0 overfull. Issue I15 reopened by decision, I25 moot (see I29) |
| 2026-08-18 | 6 | **Pareto figure removed and the whole paper reworded in plain English, on your instruction.** Appendix C.3 and `fig:app_pareto` deleted (nothing referenced the label). "sweep"/"swept" and "caveat" are gone from the paper, along with surrogate, artefact, verdict, auditable, nominally, monotone, screened, noise floor, lever, instrument, characterise, unnormalised and others. Section headings are now plain ICLR ones in template order — the RQ names moved out of the headings and into the first line of each results subsection. No claim, number, hedge or citation changed; `check_numbers.py` still passes. Rebuild: 16 pages, main text still ends exactly at page 9, 0 errors / 0 undefined / 0 overfull. Issue I24 now moot (see I30) |
| 2026-08-18 | 6 | **Final read-through with the pages actually rendered.** Four fixes: (1) the main text was 9 pages **plus two lines** — the conclusion's last sentence sat on page 10, over the ICLR limit; cut two repeated sentences and it now ends on page 9 (I32). (2) The teaser was floating to page 3, after the whole introduction; moved up to the top of page 2 where it can do its job. (3) Title was breaking with "TCNs" alone on a line; rebalanced to three lines. (4) The teaser caption claimed the model is "more selective", which the picture does not clearly show — replaced with what a reader can check in the image (0.65 against 0.19 on the true class, strongest evidence inside the marked window). `seanet/paper/teaser.py` fixed for names, fonts and the colour-bar collision, but the image still needs one re-run from you (I31) |
