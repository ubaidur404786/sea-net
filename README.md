# SEA-Net

**Small, interpretable time-series classification.**

SEA-Net does not only predict a label - it also says *which timesteps* made it choose that label.
It follows the MILLET idea (Multiple Instance Learning for time-series classification): one series
is a "bag", each timestep is an "instance", and the model produces a per-timestep importance map
alongside its prediction.

The question the project asks: **can a much smaller model beat MILLET on accuracy AND on
interpretability at the same time?**

Datasets: **WebTraffic** (the headline set - the only one with per-timestep ground truth) and the
**UCR 2018 archive** (128 datasets, 85 of them with published MILLET numbers to compare against).

---

## Quick start

```bash
conda create -n seanet python=3.10 -y && conda activate seanet
pip install torch                     # pick the right build for your machine: pytorch.org
pip install -r requirements.txt

python main.py models                                                  # what can I run?
python main.py single Coffee --model seanet_bottleneck_topk --smoke    # 3-epoch flow check
```

Datasets are not in git - see [guide/02_datasets.md](guide/02_datasets.md).

---

## The architecture

```text
series (B, 1, T)
    -> ENCODER            (B, d, T)     seanet/models/encoders.py
    -> MIL POOLING HEAD                 seanet/models/pooling.py
         |-> bag_logits     (B, n_clz)    the prediction
         |-> interpretation (B, n_clz, T) the explanation  -> AOPCR, NDCG
         `-> attn           (B, T, 1)     the attention gate
```

**The encoder and the MIL pooling head are independently switchable.** Each is looked up by name
in its own registry, so any encoder pairs with any head through configuration alone:

```yaml
encoder: {type: sea_mstcn_sep_bottleneck}   +  pooling: {type: sea_topk_conjunctive}
encoder: {type: mil_inceptiontime}          +  pooling: {type: sea_topk_conjunctive}
encoder: {type: sea_mstcn_sep_bottleneck}   +  pooling: {type: mil_conjunctive}
```

The classification head lives *inside* the pooling head, on purpose: a MIL head must score every
timestep before aggregating, and that per-timestep score **is** the interpretation map.

---

## The pipeline

```text
configs  ->  data  ->  preprocessing  ->  encoder  ->  MIL pooling  ->  training
                                                                           |
                              results / leaderboard / analysis  <-  evaluation  ->  MLflow
```

One module per step, and `main.py` is the only file that knows the order:

| step | file |
|---|---|
| configuration | `seanet/config.py` |
| data | `seanet/data.py` |
| preprocessing + split | `seanet/preprocessing.py` |
| encoder | `seanet/models/encoders.py` |
| MIL pooling | `seanet/models/pooling.py` |
| build the model | `seanet/models/build.py` |
| training | `seanet/training.py` |
| evaluation | `seanet/evaluation.py` |
| metrics (accuracy / AOPCR / NDCG) | `seanet/metrics.py` |
| results + leaderboard | `seanet/results.py` |
| comparison figures + tables | `seanet/analysis/` |
| MLflow | `seanet/tracking.py` |
| Optuna (optional) | `seanet/optuna_search.py` |

---

## Repository layout

```text
main.py           the ONE entry point
configs/          main.yaml, environments/ (local, grid5000), models/ (baselines, seanet, ablations)
seanet/           our code
millet/           the MILLET baseline - upstream code, kept unchanged so it stays diffable
scripts/          profiling, ensembling, config generation, Grid5000 launchers
data/             the datasets (not in git)
results/          all outputs
guide/            the documentation
```

---

## Commands

```bash
# cheap - read what already exists
python main.py models | summary | params | results | leaderboard | analyse | report | web-compare

# expensive - these train (add --smoke for a 3-epoch check)
python main.py single NAME --model M
python main.py train --model M --env grid5000
python main.py webtraffic | run | interpret | optuna --model M
```

`python main.py -h` for all of them.

---

## Documentation

Everything is in **[`guide/`](guide/README.md)**:

| | |
|---|---|
| [01 Setup](guide/01_setup.md) | [08 Adding a dataset](guide/08_adding_a_dataset.md) |
| [02 Datasets](guide/02_datasets.md) | [09 MLflow](guide/09_mlflow.md) |
| [03 Running experiments](guide/03_running_experiments.md) | [10 Optuna](guide/10_optuna.md) |
| [04 Configuration](guide/04_configuration.md) | [11 The MILLET baseline](guide/11_millet_baseline.md) |
| [05 Results](guide/05_results.md) | [12 Grid5000](guide/12_grid5000.md) |
| [06 Adding an encoder](guide/06_adding_an_encoder.md) | [13 Debugging](guide/13_debugging.md) |
| [07 Adding a MIL pooling method](guide/07_adding_a_pooling_method.md) | [14 Git](guide/14_git.md) |

---

## Results

72 model configurations have been trained and ranked. The current table is
`results/SEA_NET/leaderboard.csv`; the comparison figures and tables are in `results/analysis/`
(start with `results/analysis/INDEX.md`).

Top of the WebTraffic leaderboard:

| rank | config | encoder | pooling | acc | AOPCR | NDCG | params |
|---|---|---|---|---|---|---|---|
| 1 | `seanet_gated_mean_topk` | `sea_mstcn_sep_gated` | `sea_topk_conjunctive` | 0.9547 | 2.225 | 0.750 | 61,740 |
| 2 | `seanet_conjunctive` | `sea_mstcn_sep` | `mil_conjunctive` | 0.9540 | 1.502 | 0.698 | 269,083 |
| 3 | `seanet_gated_max_topk` | `sea_mstcn_sep_gated` | `sea_topk_conjunctive` | 0.9520 | 2.268 | 0.719 | 61,740 |
| 4 | `seanet_topk_nofocus` | `sea_mstcn_sep_bottleneck` | `sea_topk_conjunctive` | 0.9500 | 2.303 | 0.765 | 41,324 |
| 5 | `seanet_spiketrend_topk` | `sea_mstcn_sep_spiketrend` | `sea_topk_conjunctive` | 0.9500 | 2.316 | 0.756 | 67,020 |

Over three seeds, `seanet_bottleneck_topk` beats the re-trained MILLET baseline on accuracy, AOPCR
**and** NDCG at once with **90% fewer parameters** (41 K vs 424 K) and 11x fewer FLOPs.

Two things the numbers above do **not** say, stated plainly:

1. On the full UCR archive MILLET is still ahead - 0.8274 with our recipe, 0.8434 with theirs,
   against our ~15.3-15.5 mean rank.
2. **AOPCR is unnormalised.** The same architecture scores 2.57 under our recipe and 13.27 under
   MILLET's longer one. Only compare AOPCR between models trained the same way.

---

## Licence

Apache 2.0 - see `LICENSE` and `NOTICE`. `millet/` is the MILLET authors' code (Amazon,
Apache-2.0), included and kept unmodified.
