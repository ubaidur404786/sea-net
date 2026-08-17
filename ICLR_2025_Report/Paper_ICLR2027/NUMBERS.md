# NUMBERS.md — every number the paper prints, and where it came from

Rule: **if a number is not in this file, it does not go in the paper.**
Checked on 2026-08-17 against the files named in each section.

Short names for the sources:
- `LB` = `results/SEA_NET/leaderboard.csv`
- `RK` = `results/paper_figures/tables/table_ranking_accuracy.csv`
- `PF` = `results/SEA_NET/profile.csv`
- `RC(model)` = `results/SEA_NET/<model_dir>/results.csv`

---

## 1. WebTraffic, main comparison — from `LB`

Columns used: `web_acc`, `web_aopcr`, `web_ndcg`, `params`.
"S" = number of seeds. A 1-seed row must be marked as such in the paper (I4).

| model | encoder + head | acc | AOPCR | NDCG@n | params | S |
|---|---|---|---|---|---|---|
| `seanet_gated_mean_topk` | SEA-Net gated + Top-k | 0.9547 | 2.225 | 0.7496 | 61,740 | 3 |
| `seanet_conjunctive` | SEA-Net wide + **MILLET** conjunctive | 0.9540 | 1.5019 | 0.6983 | 269,083 | 1 |
| `seanet_bottleneck_topk` | SEA-Net bottleneck + Top-k | 0.9473 | 2.6214 | 0.7725 | 41,324 | 3 |
| `seanet_classwise` | SEA-Net wide + class-wise conj. | 0.9460 | 1.7219 | 0.6860 | 269,164 | 1 |
| `seanet_inputgate_adaptive` | SEA-Net input-gated + adaptive | 0.9053 | 2.6508 | 0.7321 | 58,102 | 3 |
| `millet` | InceptionTime + conjunctive | 0.8873 | 2.5690 | 0.6769 | 423,707 | 3 |
| `resnet` | ResNet + conjunctive | 0.7720 | 2.9522 | 0.5540 | 506,331 | 1 |
| `fcn` | FCN + conjunctive | 0.7420 | 3.8256 | 0.5327 | 267,035 | 1 |

⚠ `seanet_conjunctive` and `seanet` (+ additive) have `origin = half-ours`:
our encoder, MILLET's head. Must be labelled (issue I14).

## 2. Per-seed spread — from `RC(seanet_bottleneck_topk)` etc.

Verified directly for `seanet_bottleneck_topk` (WebTraffic rows):
seed 0 = 0.938 / loss 0.2992 / AOPCR 2.7784 / NDCG 0.7768;
seed 1 = 0.938 / 0.3107 / 2.7079 / 0.7829;
seed 2 = 0.966 / 0.2358 / 2.3779 / 0.7578.
Mean 0.9473, AOPCR mean 2.6214 ± 0.214 — matches `LB`. ✅

| model | seed 0 | seed 1 | seed 2 | acc mean ± sd | AOPCR sd | NDCG sd |
|---|---|---|---|---|---|---|
| `seanet_gated_mean_topk` | 0.954 | 0.958 | 0.952 | 0.955 ± **0.003** | ±0.131 | ±0.022 |
| `seanet_bottleneck_topk` | 0.938 | 0.938 | 0.966 | 0.947 ± **0.016** | ±0.214 | ±0.013 |
| `seanet_inputgate_adaptive` | 0.938 | 0.880 | 0.898 | 0.905 ± **0.030** | ±0.437 | ±0.043 |
| `millet` | 0.894 | 0.876 | 0.892 | 0.887 ± 0.010 | ±**0.884** | ±0.016 |

**The rule this table creates:** no difference smaller than the relevant sd is
reported as a result. In particular 0.955 vs 0.947 is *not* decisive (sd 0.016),
and AOPCR 2.651 vs 2.621 is noise.

## 3. UCR-85 — accuracy from `LB`, mean rank from `RK`

Mean rank is over the **84 datasets shared by the 28 fully-swept models**;
lower is better. W/T/L is against MILLET's *published* per-dataset accuracies.

| model | UCR-85 acc | n | mean rank | W/T/L |
|---|---|---|---|---|
| MILLET published \citep{early2024millet} | 0.8445 | 85 | — | — |
| `millet_paper` (their 1500-epoch recipe, our code) | 0.8434 | 84 | **9.41** | 26/32/26 |
| `seanet` (SEA-Net wide + MILLET additive) | 0.8298 | 85 | 11.98 | 23/19/42 |
| **`seanet_classwise` (SEA-Net wide + our class-wise)** | **0.8292** | 85 | **11.53** | 26/19/39 |
| `seanet_conjunctive` (SEA-Net wide + MILLET conj.) | 0.8238 | 85 | 12.43 | 22/20/42 |
| **`millet` (our identical configuration)** | **0.8274** | 84 | **12.60** | 13/25/46 |
| `resnet` | 0.8146 | 85 | 13.54 | 25/11/49 |
| `fcn` | 0.8141 | 85 | 14.48 | 20/12/52 |
| `seanet_inputgate_adaptive` (58 K) | 0.8153 | 85 | 15.47 | 20/15/50 |
| `seanet_gated_mean_topk` (62 K) | 0.8083 | 85 | 15.29 | 19/13/53 |
| `seanet_bottleneck_topk` (41 K) | 0.8097 | 85 | 15.40 | 20/15/50 |

**Two effect sizes the paper is built on:**
- best architecture change, same budget: 12.60 → 11.53 = **1.07 rank**
- same architecture, longer budget: 12.60 → 9.41 = **3.19 rank**

⚠ Never quote `seanet_bottleneck_adaptive` UCR numbers — `ucr85_n = 6` (I6).

## 4. Top-k fraction κ — ALL SEED 0 (issue I5)

`seanet_topk_k005/k025/k050/k100` from `LB`; κ = 0.10 row is the **seed-0** row
of `RC(seanet_bottleneck_topk)`, verified above.

| κ | accuracy | loss | AOPCR | NDCG@n |
|---|---|---|---|---|
| 0.05 | 0.888 | 0.4013 | 2.2883 | 0.7150 |
| **0.10** (default) | 0.938 | 0.2992 | 2.7784 | **0.7768** |
| 0.25 | 0.940 | 0.2773 | 2.6479 | 0.7332 |
| 0.50 | 0.882 | 0.4184 | **3.1143** | 0.7321 |
| 1.00 (= class-wise conjunctive, exactly) | 0.908 | 0.3946 | 2.4355 | 0.7220 |

Shape, stated honestly (issue I19): NDCG@n peaks at κ = 0.10 and falls to 0.722
at κ = 1.00, but also falls to 0.715 at κ = 0.05 — there is an **optimum**, not
a monotone gain. AOPCR does **not** track κ (peak at κ = 0.50), which is itself
evidence for claim C3. All rows single seed.

## 5. Attention entropy penalty — seed 0

`seanet_topk_nofocus` (λ = 0) from `LB`, against the seed-0 row above.

| λ_focus | accuracy | loss | AOPCR | NDCG@n |
|---|---|---|---|---|
| 0.01 (on) | 0.938 | 0.2992 | **2.7784** | **0.7768** |
| 0.00 (off) | **0.950** | **0.2626** | 2.3030 | 0.7653 |

Same encoder, same head, same κ, same seed. One matched pair only.

## 6. AOPCR is configuration-dependent — claim C3, from `LB`

`millet` and `millet_paper` are the **same architecture, same code, same metric
implementation, same data**; only the training recipe differs.

| | WebTraffic acc | WebTraffic AOPCR | UCR-85 acc | UCR-85 AOPCR |
|---|---|---|---|---|
| `millet` (400 epochs, our configuration) | 0.8873 | **2.569** | 0.8274 | **0.8119** |
| `millet_paper` (1500 epochs, their recipe) | 0.9200 | **13.2684** | 0.8434 | **3.9264** |
| ratio | — | **5.17×** | — | **4.84×** |

MILLET's own published UCR AOPCR is 4.5532 (`millet_aopcr` column) — the paper
must never compare it with any of our AOPCR values.

## 7. Head ablation, mean over every encoder it was paired with — from `LB`

Grouped by the `pooling` column. Excludes the five `sv7` ablation configs and
`millet_paper` (verified by hand: Top-k n = 16 sums to 14.470/16 = 0.9044 ✅;
class-wise n = 10 sums to 9.210/10 = 0.9210 ✅).

| head | mean acc | mean AOPCR | mean NDCG@n | n |
|---|---|---|---|---|
| Class-wise conjunctive | 0.921 | 2.322 | 0.692 | 10 |
| Per-class gated attention | 0.918 | 1.869 | 0.733 | 4 |
| Adaptive class-wise | 0.912 | 2.210 | 0.732 | 9 |
| Softmax conjunctive | 0.907 | 1.690 | 0.712 | 9 |
| Top-k conjunctive | 0.904 | 2.267 | 0.712 | 16 |
| Attention-max | 0.838 | 1.758 | 0.645 | 8 |
| **Dual-stream conjunctive** | **0.744** | 2.007 | 0.636 | 4 |

Dual-stream's four pairings: 0.860, 0.824, 0.736, **0.556**.

## 8. Encoder ablation, mean over every head — from `LB`

| encoder | mean acc | mean AOPCR | mean NDCG@n | n |
|---|---|---|---|---|
| multi-scale channels (wrapper) | 0.937 | 2.385 | 0.747 | 4 |
| reconstruction-residual | 0.924 | 2.315 | 0.719 | 5 |
| input-gated | 0.904 | 2.345 | 0.718 | 5 |
| gated | 0.902 | 2.056 | 0.710 | 19 |
| base | 0.898 | 1.933 | 0.694 | 12 |
| bottleneck | 0.884 | 2.209 | 0.694 | 8 |
| spike/trend | 0.858 | 1.814 | 0.677 | 7 |
| multi-scale pyramid (wrapper) | 0.771 | 1.394 | 0.604 | 3 |

Spread of the head choice (0.744–0.921, **0.177**) and of the encoder choice
(0.771–0.937, **0.166**) are comparable — that is the evidence that the head is
a first-class design axis, not a detail.

## 9. Cost — from `PF` (batch 32, length 1008, 10 classes, one GPU)

| model | params | size MB | FLOPs M | infer ms | peak mem MB |
|---|---|---|---|---|---|
| `seanet_bottleneck_topk` | 41,324 | 1.4279 | 76.673 | 0.1305 | **179.015** |
| `seanet_gated_mean_topk` | 61,740 | 1.5025 | 109.711 | 0.1279 | 64.871 |
| `seanet_classwise` (wide) | 269,164 | 3.5389 | 523.741 | **0.3591** | 123.735 |
| `millet` | 423,707 | 4.1098 | 847.639 | 0.2026 | 112.284 |
| `resnet` | 506,331 | 4.4156 | 1013.242 | 0.1389 | 76.524 |
| `fcn` | 267,035 | 3.4737 | 535.208 | 0.0680 | 132.153 |

Ratios `millet` ÷ `seanet_bottleneck_topk`: params **10.25×**, FLOPs **11.06×**,
file size **2.88×**. Latency only 1.55× better, and peak memory is **worse**
(issue I17) — say so.

Training time to convergence (mean ± sd over 3 WebTraffic seeds, from `RC`):
41 K model 117.3 ± 14.0 s, 62 K model 91.3 ± 25.9 s, `millet` 50.4 ± 9.5 s.
SEA-Net trains **2–3× slower** (issue I12).

## 10. Relation to published MILLET — appendix only

| how obtained | WebTraffic acc | UCR-85 acc |
|---|---|---|
| Published, mean of 5 seeds | 0.924 | 0.8445 |
| Published, 5-model ensemble | 0.940 | — |
| Their released weights, evaluated on our pipeline | 0.922 | — |
| Re-trained here under **their** hyperparameters | 0.920 | 0.8434 |
| Re-trained here under **our** configuration | 0.887 | 0.8274 |

0.8434 against a published 0.8445 = the re-implementation is correct; the
remaining gap is the **training budget**, not the code.

## 11. Datasets and protocol

- WebTraffic: 1,000 series (500 train / 500 test), length 1,008, 10 classes,
  per-time-step ground truth **yes** (synthetic generator records it).
- UCR-85: 40–16,637 series, length 24–2,709, 2–60 classes, ground truth **no**.
- UCR full archive: 128 datasets, length 15–2,844.
- Training, identical for every model: Adam, lr 1.25e-3, weight decay 1.2e-4,
  max 400 epochs, patience 60, batch min(⌊n_train/10⌋, 16) floor 2, stratified
  80/20 validation when n_train ≥ 100, label smoothing 0.13, λ_focus 0.01
  (0 for the MILLET baseline).
- Encoder used in all headline models: width d = 64, 4 blocks. Wide variant:
  d = 128, 6 blocks. Kernels {5, 11, 23}, dilation capped at 16, dropout 0.2.
- Attention width 8, Top-k fraction κ = 0.1, τ₀ = 1.0, β₀ = 0.3,
  λ₀ = 0.5 (attention-max) / 0.05 (dual-stream).
- Sweep size: **72 encoder × head combinations**, up to 129 datasets each.

## 12. Numbers deliberately NOT used

| number | why not |
|---|---|
| MILLET's published AOPCR 4.5532 | our own claim C3 says it is not comparable |
| `seanet_bottleneck_adaptive` UCR results | only 6 datasets finished (I6) |
| Any "runs on a microcontroller" claim | nothing was ever deployed (I17) |
| MLPerf Tiny 38.6 K as a pass/fail threshold | it is an order of magnitude, not a spec |
