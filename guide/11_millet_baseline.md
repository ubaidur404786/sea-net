# 11 - The MILLET baseline

MILLET (*Multiple Instance Learning for Locally Explainable Time series classification*) is the
paper this project builds on, and the model we have to beat. Its code lives in `millet/`, and this
guide is about the boundary between it and ours.

## The rule: `millet/` is not edited

`millet/` is upstream code, kept as close to byte-identical with the original repository as
possible. That is what lets anyone diff this repo against theirs and see **exactly** what we
changed - which is the difference between "we re-implemented MILLET" and "we ran MILLET".

When we need different behaviour, we **subclass in `seanet/`**. The clearest example:
15 UCR datasets ship with NaNs, so `seanet/data.py` defines `AdjustedUCRDataset(UCRDataset)` to
read the archive's fixed copies. `millet/data/ucr_2018_dataset.py` is untouched.

```text
millet/
├── data/
│   ├── mil_tsc_dataset.py        the base class: what a "bag" is + z-normalisation
│   ├── ucr_2018_dataset.py       reading UCR
│   ├── web_traffic_dataset.py    reading WebTraffic (the only per-timestep ground truth)
│   └── web_traffic_generation.py the WebTraffic generator
├── model/
│   ├── millet_model.py           the train/evaluate harness we subclass as SeaNetModel
│   ├── pooling.py                5 MIL pooling heads
│   └── backbone/                 InceptionTime, FCN, ResNet
├── interpretability_metrics.py   AOPCR and NDCG
├── util.py
└── notebooks/                    the upstream example notebooks
```

## The three places we use it

| what | where | why |
|---|---|---|
| **baseline components** | its backbones and pooling heads, registered in our registries under the `mil_` prefix | so the baseline is selected by config exactly like our own models, through the same interface |
| **the dataset base class** | `MILTSCDataset` | it defines what a bag is and does the z-normalisation, identically for us and for them |
| **the metrics** | `interpretability_metrics.calculate_aopcr`, `MILLETModel.evaluate_interpretability` | so our AOPCR and NDCG are computed the same way as the published numbers, and are comparable |

### The naming rule that makes it visible

Every encoder and every pooling head carries its origin in its name:

* `mil_*` = MILLET's, reused unchanged - `mil_inceptiontime`, `mil_fcn`, `mil_resnet`,
  `mil_conjunctive`, `mil_additive`, `mil_attention`, `mil_instance`, `mil_gap`
* `sea_*` = ours - `sea_mstcn_sep_bottleneck`, `sea_topk_conjunctive`, ...

And a results folder is `<config>__<encoder>__<pooling>`, so
`millet__mil_inceptiontime__mil_conjunctive` says at a glance that nothing in it is ours, while
`seanet_bottleneck_topk__sea_mstcn_sep_bottleneck__sea_topk_conjunctive` says the opposite.
`seanet/config.py: is_ours()` / `is_millet()` use that prefix, and the analysis figures colour the
bars by it.

---

## Running the baseline

Exactly like any other model - same pipeline, same commands:

```bash
python main.py single WebTraffic --model baselines/millet --smoke
python main.py train  --model baselines/millet --env grid5000
python main.py train  --model baselines/millet_paper --env grid5000
```

| config | what it is |
|---|---|
| `baselines/millet` | MILLET's architecture trained with **our** recipe. The fair comparison: same budget, same loss, only the architecture differs. |
| `baselines/millet_paper` | MILLET's architecture trained with **their** recipe (1500 epochs). Reproduces their published numbers. |
| `baselines/fcn`, `baselines/resnet` | the classic backbones with Conjunctive pooling |
| `baselines/conventional` | global average pooling - a normal classifier with no per-timestep explanation |
| `baselines/transformer` | a placeholder, marked `implemented: false`; building it raises a clear error rather than training something wrong |

Note `lambda_entropy: 0.0` in the baseline configs. The attention-focus penalty is a SEA-Net idea,
not part of MILLET, so adding it to the baseline would make the comparison unfair.

## The published numbers

`results/UCR/InceptionTime/` and `results/WebTraffic/InceptionTime/` hold the numbers the MILLET
paper published, for the 85 datasets they report. They are **not ours** - do not overwrite them.
`seanet/results.py: millet_baseline()` reads them, and every `comparison_vs_millet.csv` is our
model against `ConjunctiveInceptionTime` (the mean of its 5 repetitions).

---

## One result worth knowing before you compare AOPCR

The same MILLET architecture, in this repo, scores:

| recipe | WebTraffic acc | UCR-85 acc | AOPCR |
|---|---|---|---|
| ours (400 epochs) | 0.887 | 0.8274 | 2.57 |
| theirs (1500 epochs) | 0.920 | 0.8434 | **13.27** |
| published | — | 0.8445 | — |

0.8434 against a published 0.8445 is a reproduction, so the accuracy gap was the **training
budget**, not the re-implementation. And the AOPCR moving from 2.57 to 13.27 on the *same
architecture* is the direct proof that **AOPCR is unnormalised**: it scales with how confident the
model is. Only compare AOPCR between models trained with the same recipe.

---

## The upstream notebooks

`millet/notebooks/` holds the original examples (Amazon, Apache-2.0): the MILLET pipeline, a UCR
dataset walk-through, and the WebTraffic generator. They are kept with `millet/` because they are
upstream material, not ours.

---

Next: [12 - Grid5000](12_grid5000.md)
