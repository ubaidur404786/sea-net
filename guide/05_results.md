# 05 - Results: what gets saved, where, and how to read it

## The layout

```text
results/
├── SEA_NET/                                 everything our models produce
│   ├── leaderboard.csv                        every model, one row, best WebTraffic acc first
│   ├── model_comparison.csv                   the cross-model ranking over the 85 MILLET datasets
│   ├── webtraffic_comparison.csv              the WebTraffic-only table
│   ├── data_summary.csv                       facts about the DATA (not about any model)
│   ├── profile.csv                            params / FLOPs / latency / memory per model
│   ├── ensemble_vote.csv                      the ensembling + multi-seed study
│   ├── figures/                               cross-model figures (winner dashboard, tiers)
│   ├── logs/                                  logs of commands that belong to no model
│   │
│   └── <model_id>/                          ONE FOLDER PER MODEL
│       ├── results.csv                        one row per (dataset, seed)   <- the source of truth
│       ├── done_train_dataset.txt             the resume list
│       ├── comparison_vs_millet.csv           this model next to MILLET, dataset by dataset
│       ├── summary.csv / summary.md           its headline means
│       ├── history/                           PER-EPOCH TRAINING RECORD  (see below)
│       ├── predictions/                       per-series test probabilities (.npz), for ensembling
│       ├── interpretation/                    per-sample explanation figures
│       ├── figures/                           this model's own figures
│       └── logs/                              one dated log per run of this model
│
├── analysis/                                the CROSS-MODEL comparison output
│   ├── INDEX.md                               start here: every figure + the question it answers
│   ├── 01_leaderboard/                        who is strongest overall
│   ├── 02_ablation/                           what each encoder / pooling head contributes
│   ├── 03_detail/                             every model, every dataset
│   ├── 04_webtraffic/                         our headline dataset on its own
│   ├── 05_statistics/                         ranks, significance, win/tie/loss
│   └── tables/                                the same numbers as .csv (exact) and .md (readable)
│
├── UCR/InceptionTime/                       the MILLET paper's PUBLISHED numbers (85 datasets)
└── WebTraffic/InceptionTime/                the MILLET paper's PUBLISHED WebTraffic numbers
```

`<model_id>` is `<config name>__<encoder>__<pooling>`, e.g.
`seanet_bottleneck_topk__sea_mstcn_sep_bottleneck__sea_topk_conjunctive`. The config name is in
front so two configs that build the same encoder+pooling never share a folder.

`results/UCR/` and `results/WebTraffic/` are **not ours** - they are the numbers the MILLET paper
published, kept so we can compare against them. Never overwrite them.

---

## `results.csv` - the one row that everything is built from

One row per (model, dataset, seed):

| column | meaning |
|---|---|
| `dataset`, `model`, `encoder`, `pooling`, `seed`, `device` | what was run |
| `params`, `model_size_mb` | how big the network is |
| `n_train`, `n_val`, `n_test`, `series_length`, `n_classes` | the data it saw |
| `lambda_entropy` | the attention focus penalty used |
| `test_acc`, `test_bal_acc`, `test_auroc`, `test_loss` | did it predict correctly? |
| `test_aopcr` | did it explain correctly? (higher is better) |
| `test_ndcg` | did it point at the RIGHT timesteps? (WebTraffic only; empty elsewhere) |
| `train_time_s` | wall-clock training time |
| `epochs_run`, `best_epoch` | how long it trained, and which epoch's weights were kept |
| `overfit_gap` | `val_loss - train_loss` at the best epoch; empty when there was no validation split |
| `run_datetime` | when |

FLOPs, latency and peak memory are **not** here - they are measured once per model, without
training, by `python scripts/profile_models.py`, into `results/top_results/SEA_NET/profile.csv`. The
leaderboard and the analysis figures join the two.

### An empty cell means "not run", never zero

The leaderboard's UCR columns are blank for models that were only screened on WebTraffic. Do not
read a blank as a 0.

---

## `history/` - the training curves

New in seanetv7. One CSV and two PNGs per (dataset, seed):

```text
<model_id>/history/Coffee__seed0.csv        epoch, train_loss, train_acc, val_loss, val_acc,
                                            monitored, best_epoch
<model_id>/history/Coffee__seed0_loss.png   training vs validation LOSS, best epoch marked
<model_id>/history/Coffee__seed0_acc.png    training vs validation ACCURACY
```

The folder name carries the model (and therefore its encoder and pooling), the file name carries
the dataset and the seed - so every figure says exactly which experiment produced it.

**How to read the loss curve.** The training curve should fall. If the validation curve stops
following it and turns back up while the training curve keeps falling, the model is memorising
the training set - that is overfitting. The dashed line is the epoch whose weights were actually
kept, which is where early stopping decided things stopped improving. The `overfit_gap` column in
`results.csv` is the same story as one number.

**When there is no validation curve**, the dataset had fewer than `min_train_for_val` training
series, so the whole training file was used for training and early stopping watched the training
loss instead. The figure says so in the legend rather than drawing an empty line.

---

## The comparison tables

| command | writes | answers |
|---|---|---|
| `python main.py results --model M` | `<model_id>/comparison_vs_millet.csv`, `summary.csv/.md` | how does this ONE model do against MILLET, dataset by dataset? |
| `python main.py results` | the above for every model, plus `model_comparison.csv` | which encoder+pooling wins overall? |
| `python main.py leaderboard` | `leaderboard.csv` | one row per model, ranked |
| `python main.py web-compare` | `webtraffic_comparison.csv` + tier figures | the WebTraffic screen |
| `python main.py analyse` | `results/top_results/analysis/` | everything cross-model: encoder comparison, pooling comparison, Pareto fronts, ranks, significance |

A win/tie/loss uses a **tie band** so a meaningless difference is not called a win: 0.005 on
accuracy, 0.010 on loss, 0.100 on AOPCR (`results.py: COMPARED_METRICS`).

---

## The metrics, in one line each

| metric | question | direction |
|---|---|---|
| accuracy | how often is the predicted class right? | higher |
| balanced accuracy | the same, but each class counts equally (matters on unbalanced data) | higher |
| AUROC | how well are the class probabilities ordered? | higher |
| loss | cross-entropy on the test set | lower |
| **AOPCR** | delete the timesteps the model says matter; how far does the true-class score fall? | higher |
| **NDCG@n** | of the timesteps that really are important, how many did it rank at the top? | higher |
| params / FLOPs / latency / memory | what does it cost to run? | lower |

**AOPCR is not normalised.** Its size depends on how confident the model is, so only compare AOPCR
between models trained with the *same* recipe. The same architecture in this repo scores 2.57
under our recipe and 13.27 under MILLET's own longer recipe - that is the training budget talking,
not the interpretability.

**NDCG only exists for WebTraffic**, because it is the only dataset with per-timestep ground truth.
Everywhere else the column is empty, and that is correct, not a bug.

---

## What is in git and what is not

| kept in git | not in git |
|---|---|
| every `results.csv`, the leaderboard, the comparison tables, `summary.*` | `data/` (870 MB, own licence) |
| `history/` (a few KB per run, and it is the evidence a run behaved) | `model/` and `*.pth` (regenerated by training) |
| `interpretation/` and `figures/` | `mlflow.db`, `mlruns/`, `mlartifacts/` (per machine; copy with scp) |
| real training logs | `predictions/*.npz` (~30 MB, regenerable; only ensembling reads them) |
| `results/top_results/analysis/` (small now that it is PNG-only) | smoke-run outputs, raw Grid5000 console dumps |

See `.gitignore` - its section headers explain each rule.

---

Next: [06 - Adding an encoder](06_adding_an_encoder.md)
