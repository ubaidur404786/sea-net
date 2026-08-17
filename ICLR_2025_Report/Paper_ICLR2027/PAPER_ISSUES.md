# PAPER_ISSUES.md — running list of things to fix in the ICLR paper

This file is the memory of the paper. Every time I find something that is
wrong, risky, or not yet checked, it gets one entry here. Nothing is ever
deleted — an entry is only moved to **DONE** with the date, so we can always
see what was decided and why.

- **Folder:** `ICLR_2025_Report/Paper_ICLR2027/` (the paper)
- **Branch:** `iclr1`
- **Do not touch:** `ICLR_2025_Report/Seanet/` (the internship report)
- **Target:** ICLR 2027, main text **9 pages**, double blind

Severity: **BLOCKER** = paper is rejected without fixing · **HIGH** = a
reviewer will attack it · **MED** = weakens the paper · **LOW** = polish.

---

## OPEN

### I1 — Author identity is printed in the report — BLOCKER
The internship report prints `https://github.com/ubaidur404786/sea-net` in the
abstract and again in Appendix B.6. ICLR 2027 is double blind: *"Any paper
where author identity is revealed in either the main text or the supplementary
material will be desk rejected."* A GitHub username is author identity.

**Fix:** the paper never prints that URL. Use an anonymous placeholder
(`https://anonymous.4open.science/r/seanet-XXXX`) and create the real anonymous
mirror before submitting. Also sweep for: the supervisor's name, the host
company, "internship", Grid'5000 site names (Lille / Sophia identify the lab),
and the PDF metadata.

---

### I2 — The UCR result is understated in the report, and this matters a lot — HIGH
Section 5.2 of the report lists only the three **narrow** (41–62 K) models and
concludes *"the re-trained MILLET baseline is ahead of all three"*. That is
true for those three rows, but it is not the whole picture. From
`results/paper_figures/tables/table_ranking_accuracy.csv` (mean accuracy rank
over the 84 shared datasets, lower is better):

| model | mean rank |
|---|---|
| MILLET, **their** 1500-epoch recipe | 9.41 |
| **SEA-Net wide encoder + our class-wise conjunctive** | **11.53** |
| SEA-Net wide encoder + MILLET additive head | 11.98 |
| SEA-Net wide encoder + MILLET conjunctive head | 12.43 |
| **MILLET, our identical configuration** | **12.60** |
| SEA-Net narrow models (41–62 K) | 14.0 – 15.8 |

So under one identical configuration the SEA-Net encoder **outranks** the
MILLET baseline on UCR-85 (11.53 vs 12.60). The gap only appears when the model
is shrunk to 41–62 K. That reframes the UCR result from *"our architecture is
worse"* to *"this is a capacity effect, and we measured where it starts"*.

**Fix:** the paper reports both the wide and the narrow rows in the same table
and says plainly which effect is which. Do not repeat the report's wording.

---

### I3 — AOPCR is not comparable across training configurations — HIGH (this is a contribution)
Same architecture, same code, same metric implementation, same dataset, only
the training budget differs:

| run | WebTraffic AOPCR |
|---|---|
| `millet` (our 400-epoch configuration) | 2.569 |
| `millet_paper` (their 1500-epoch recipe) | 13.268 |

A **5.2× swing** from the recipe alone, on a metric that is supposed to measure
explanation quality. AOPCR is built from raw confidence differences, so its
scale follows the scale of the logits and it has no fixed range.

**Fix:** this is a headline finding of the paper, not a footnote. Every AOPCR
number we print must be inside one configuration, and we must never place it
next to MILLET's published 4.55. Add a short reporting protocol.

---

### I4 — Seed counts are uneven: 4 models have 3 seeds, 68 have 1 — HIGH
ICLR reviewers will hit this hard. Seed spread is not small: 0.003 for
`seanet_gated_mean_topk` up to 0.030 for `seanet_inputgate_adaptive`, and AOPCR
is noisier still (±0.884 for the MILLET baseline).

**Fix:** mark every single-seed number in the tables, never claim a difference
smaller than the seed spread, and state the rule explicitly once. If there is
any compute left, more seeds on the κ ablation would be the highest-value run.

---

### I5 — The κ ablation must stay all-seed-0 — MED
`seanet_bottleneck_topk` appears twice with different values, and mixing them
would silently break the ablation:
- leaderboard (3-seed mean): 0.9473 / AOPCR 2.621 / NDCG 0.7725
- seed 0 only (the κ = 0.1 row of the ablation): 0.938 / 2.778 / 0.777

The other four κ rows (`k005`, `k025`, `k050`, `k100`) are seed 0 only.

**Fix:** the κ table uses the **seed-0** value in all five rows, and the caption
says so. Never copy 0.9473 into that table.

---

### I6 — The UCR dataset count is not the same for every model — MED
`ucr85_n` in `leaderboard.csv`: 85 for most SEA-Net models, **84** for `millet`
and `millet_paper`, and **6** for `seanet_bottleneck_adaptive` (an unfinished
run).

**Fix:** the mean rank is already computed over the 84 shared datasets, so use
the rank for the comparison. Print `n` in the table. Never quote the UCR number
of `seanet_bottleneck_adaptive` — 6 datasets is not a result.

---

### I7 — The headline models were selected on the dataset they are headlined on — HIGH
WebTraffic was the screening set: every encoder and head was tried there first
and only the strongest went on to UCR. So the WebTraffic result is partly a
selection effect.

**Fix:** say this in the protocol section, before the results, not buried in
limitations. It is what makes the UCR table (I2) the honest counterweight.

---

### I8 — Explanation correctness is measured on one synthetic dataset — HIGH
NDCG@n needs per-time-step ground truth, and only WebTraffic has it. WebTraffic
is synthetic, so its important regions are generated, not observed.

**Fix:** state it once, clearly. Consider whether a second dataset with known
ground-truth regions can be constructed — that would be the single biggest
strengthening move available, but it needs new training runs.

---

### I9 — 9-page limit, strictly enforced — BLOCKER
*"Papers with main text beyond the page limit will be desk-rejected."* The
report is 29 pages. Roughly two thirds of it cannot appear in the main text.

**Fix:** budget pages per section up front (see `PAPER_PLAN.md`, Phase 3) and
check the page count after every writing session, not at the end.

---

### I10 — The AI use statement is required — BLOCKER
ICLR 2027 requires an AI use statement. It does not count toward the page
limit. Missing it is a compliance failure.

**Fix:** write it honestly, near the reproducibility statement.

---

### I11 — Two heads failed badly and this must be framed as a finding — MED
Dual-stream conjunctive pooling is the worst head tested (mean 0.744 over 4
pairings; worst pairing 0.556) even though it *provably* contains the baseline
at λ → 0. Per-class gated attention sits mid-table (0.918) and is never the
best partner for any encoder.

**Fix:** present this as the "capacity-safe is not optimisation-safe" claim with
the mechanism named (no term in the loss pushes λ back to the safe branch), not
as an apology. Reviewers reward a well-diagnosed negative result and punish a
hidden one.

---

### I12 — SEA-Net trains 2–3× slower than the baseline — MED
From `Table 12` of the report: 117.3 ± 14.0 s vs 50.4 ± 9.5 s to convergence.
Prediction time is the other way round (0.131 ms vs 0.203 ms).

**Fix:** report both. Hiding the training cost in a paper that sells efficiency
is the kind of thing a reviewer finds and then distrusts everything else.

---

### I13 — `topk5_multimetric` figure has colliding labels — LOW
Known from `PROJECT_STATE.md`: the value labels collide with the dashed
reference lines.

**Fix:** either fix `seanet/paper/figures_stats.py` and re-run
`python main.py paper`, or use `pareto_web_acc_vs_params` instead.

---

### I14 — Some rows are "half ours" — MED
`origin = half-ours` in the leaderboard means our encoder with a MILLET pooling
head (e.g. `seanet_conjunctive`, `seanet` + additive). Two of them are in the
UCR top rows of I2.

**Fix:** the tables must label which half is which. Claiming a MILLET head as
our contribution would be a real integrity problem.

---

### I15 — No statistical test in the report — MED
`demsar2006statistical` is cited but no Wilcoxon / Nemenyi test is reported.
The critical-difference figure exists
(`results/paper_figures/01_main_figures/fig4_critical_difference_accuracy.pdf`)
but the report does not use it.

**Fix:** put the critical-difference diagram in the paper and say which test
produced it. ICLR expects this for a multi-dataset comparison.

---

### I16 — Cost numbers need a re-check before submission — LOW
`results/SEA_NET/profile.csv` was measured on 2026-08-09. Sizes and latencies in
the paper must match whatever is in that file at submission time.

**Fix:** re-read the CSV during the final compliance pass.

---

### I17 — "Smaller" is true for parameters, but NOT for latency or peak memory — HIGH
From `results/SEA_NET/profile.csv` (batch 32, length 1008, 10 classes):

| model | params | size MB | FLOPs M | infer ms | peak mem MB |
|---|---|---|---|---|---|
| SEA-Net bottleneck + Top-k (small) | 41,324 | 1.43 | 76.7 | 0.131 | **179.0** |
| SEA-Net wide + class-wise (the UCR winner) | 269,164 | 3.54 | 523.7 | **0.359** | 123.7 |
| MILLET (InceptionTime + conjunctive) | 423,707 | 4.11 | 847.6 | 0.203 | 112.3 |

Two things the report does not say:
1. The **small** model has the **highest peak memory** of the three (179 MB vs
   112 MB). Fewer parameters did not mean a smaller activation footprint.
2. The **wide** model — the one that wins on UCR-85 (issue I2) — is the
   **slowest per prediction** (0.359 ms vs 0.203 ms), despite having fewer
   parameters than the baseline.

**Fix:** the paper claims a reduction in **parameters, model size and FLOPs**,
and states the latency and peak-memory numbers as they are. Claiming "smaller
and faster" without qualification is not supported, and a reviewer who opens
the profile CSV will find it. This also weakens the microcontroller argument,
so that argument must be phrased as *"parameter and model-size budget"*, not
*"it will run on an MCU"* — nothing was ever deployed (report limitation 6).

---

### I18 — The anonymous code repository does not exist yet — BLOCKER (real submission)
You confirmed this is a real submission, so the reproducibility statement has
to point somewhere a reviewer can actually reach, and it must not be your
GitHub account (issue I1).

**Fix:** before submitting, build an anonymised mirror (e.g.
`anonymous.4open.science`) with: no author names, no institution, no
`CLAUDE.md` / `PROJECT_STATE.md` / `workflow.md` (they contain personal notes),
no Grid'5000 site names in scripts, and the `NOTICE` file preserved — the code
builds on Amazon's MILTimeSeriesClassification under Apache 2.0 and that
attribution is a legal requirement, not an optional courtesy.

---

### I19 — Claim C1 rests on single-seed runs and no compute is left — HIGH
You confirmed there is no GPU budget, so the κ ablation stays at one seed per
point. The seed spread on this exact model is ±0.016 accuracy, which is larger
than the 0.938 → 0.940 step between κ = 0.10 and κ = 0.25.

Also, the trend is **not** monotone at the low end, and the report does not say
so: NDCG@n goes 0.715 (κ=0.05) → **0.777** (0.10) → 0.733 (0.25) → 0.732
(0.50) → 0.722 (1.00). And AOPCR does not track κ at all: 2.29 → 2.78 → 2.65 →
**3.11** → 2.44.

**Fix:** state the shape honestly — an optimum near κ = 0.1, with both too much
and too little selection hurting. Lean on NDCG@n (normalised, ground truth) and
say plainly that AOPCR does not track κ, which is itself consistent with claim
C3. Mark every κ row as single seed and never claim the 0.938 vs 0.940 step.

---

### I20 — First draft is 12 pages of main text, limit is 9 — BLOCKER
Built 2026-08-17: 17 pages total. Main text (Sections 1–11) runs to page 12,
statements 12–13, references 13, appendix 14–17. **About 3 pages over**, and
that is before any figure is added (issue I21), which will add roughly another
page.

Where the space is now: §1 intro 1.6 pp, §2 background 1.3, §3 design space
1.6, §4 protocol 0.8, §5 selection 1.7, §6 cheap 2.4, §7 AOPCR 1.3, §8 safety
1.2, §9–11 1.1.

**Fix — cut list in priority order (about 3.5 pages):**
1. Move Table 7 (head ablation) to the appendix; keep only the dual-stream row
   inline in §8. **≈0.6 pp**
2. Compress §2: the MIL paragraph and the histopathology sentence can lose half
   their length; \Cref{eq:conj_gate} and \Cref{eq:conj_bag} can merge. **≈0.5 pp**
3. Compress §3's encoder prose — the variants are already in the appendix, so
   the main text needs two sentences, not two paragraphs. **≈0.5 pp**
4. Trim Table 2 (\webtraffic{}) from 8 rows to 6, dropping the two weakest
   baselines to a sentence. **≈0.3 pp**
5. Tighten §6, which is the longest section, by merging §6.4 into §6.3.
   **≈0.5 pp**
6. Shorten §9 related work to three paragraphs. **≈0.3 pp**
7. Merge §10 limitations into §11 conclusion. **≈0.3 pp**

Do the cuts in that order and re-measure after each one — not all at the end.

---

### I21 — The draft has no figures at all — HIGH
Everything is currently a table. A 9-page ICLR paper with zero figures reads as
unfinished, and the two strongest findings are both visual by nature.

**Fix — three floats, in priority order:**
1. **The teaser** (exists): same series, three models, prediction and
   explanation, at `results/SEA_NET/teaser/2026-08-06_15-02-34/`. Sells the
   whole paper on page 1.
2. **The κ curve** (does not exist): \ndcg{} and accuracy against κ, from
   \Cref{tab:kappa}. It makes the optimum visible instead of asking the reader
   to find it in five rows. Needs a small plotting function and a
   `python main.py paper` run — **you** would run that.
3. **The critical-difference diagram** (exists,
   `results/paper_figures/01_main_figures/fig4_critical_difference_accuracy.pdf`),
   which also closes issue I15.

Adding these costs about a page, so I20 and I21 must be solved together.

---

### I22 — One claim in Related Work has no citation — MED
§9 says the AOPCR concern is "adjacent to a wider one about perturbation-based
faithfulness measures". That sentence needs a real reference and I did not add
one rather than invent a plausible-looking entry.

**Fix:** find an actual paper on the pitfalls of perturbation-based
faithfulness evaluation, verify it exists, and cite it. If none is found, delete
the sentence — an uncited gesture at a literature is worse than not making it.

---

## DONE

*(nothing yet — entries move here with a date and one line on what was decided)*
