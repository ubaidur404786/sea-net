# 04 - Configuration

The rule: **nothing about an experiment is written in the Python.** If you want to change a
dataset, an encoder, a pooling head, a learning rate or where MLflow writes, you change a config
value or pass a flag.

## The four layers

Each one can override the one above it:

```text
1. configs/main.yaml               WHAT to run    (model, dataset, seed, output paths, defaults)
2. configs/environments/<env>.yaml WHERE it runs  (device, MLflow store)
3. configs/models/**/<model>.yaml  the MODEL      (encoder, pooling, training recipe, optuna)
4. --flags on the command line     the last word  (--model, --dataset, --seed, --env, --smoke)
```

`seanet/config.py: load_config()` merges them, in that order, into one object you read with dots:

```python
cfg.model                                 # "seanet/seanet_bottleneck_topk"
cfg.seed                                  # 0
cfg.env                                   # "local"
cfg.model_config.encoder.type             # "sea_mstcn_sep_bottleneck"
cfg.model_config.pooling.type             # "sea_topk_conjunctive"
cfg.model_config.training.learning_rate   # 0.00125
```

To see every resolved value for a run, use the `run` command - it prints the whole thing before
training and logs it to MLflow as the run's params:

```bash
python main.py run --model seanet_bottleneck_topk --smoke
```

---

## The folders

```text
configs/
├── main.yaml
├── environments/
│   ├── local.yaml           device: auto,  MLflow: sqlite:///mlflow.db, weights not saved
│   └── grid5000.yaml        device: cuda,  MLflow: sqlite:///mlflow.db, weights saved
└── models/
    ├── baselines/           millet, millet_paper, fcn, resnet, conventional, transformer
    ├── seanet/              our encoder x pooling combinations  (61 files)
    └── ablations/           one-knob-at-a-time studies          (5 files)
```

The three model folders say what a config **is**, not when it was written. List them all:

```bash
python main.py models
```

`--model` accepts either the full name (`seanet/seanet_bottleneck_topk`) or just the file name
(`seanet_bottleneck_topk`). A name that matches nothing, or matches two files, is a loud error -
a typo must never quietly train the wrong model.

---

## What a model config looks like

```yaml
name: seanet_bottleneck_topk       # goes into the results folder name and results.csv
use_params: default                # default | optuna_best | auto   (see guide 10)

encoder:                           # <- swap this freely
  type: sea_mstcn_sep_bottleneck
  d: 64
  n_blocks: 4
  dropout: 0.2
  max_dilation: 16
  kernels: [5, 11, 23]
  bottleneck_ratio: 4

pooling:                           # <- and swap this freely, independently
  type: sea_topk_conjunctive
  d_attn: 8
  dropout: 0.2
  positional_encoding: true
  top_frac: 0.1

training:
  n_epochs: 400
  patience: 60
  max_batch: 16
  learning_rate: 0.00125
  weight_decay: 0.00012
  label_smoothing: 0.13
  lambda_entropy: 0.01             # the attention "focus" penalty; 0 turns it off
  min_train_for_val: 100           # below this, no validation split (see guide 03)
  val_frac: 0.2
  optimizer: adam

# ===== AUTO-FILLED RESULTS (managed by main.py - do not edit below this line) =====
records:
  default:      { metrics: {...} }   # written by `main.py run`
  optuna_best:  { params: {...}, metrics: {...} }   # written by `main.py optuna`
```

The `records` block at the bottom is **managed by the code**. Everything above the marker line is
yours; the code rewrites only what is below it, so your comments are never lost.

### Encoder and pooling are independent

That is the whole design. Any encoder can be paired with any pooling head, because each is looked
up by string in its own registry:

```yaml
encoder: {type: sea_mstcn_sep_bottleneck}   +  pooling: {type: sea_topk_conjunctive}
encoder: {type: mil_inceptiontime}          +  pooling: {type: sea_topk_conjunctive}
encoder: {type: sea_mstcn_sep_bottleneck}   +  pooling: {type: mil_conjunctive}
```

All three are valid configs and need no code change. `scripts/make_cross_configs.py` writes the
full cross product for you (and never overwrites an existing file).

---

## Environment configs

The only settings that differ between a laptop and a GPU node:

```yaml
# configs/environments/grid5000.yaml
env: grid5000
device: cuda
mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  log_model_weights: true
```

Selected by, in order: `--env grid5000`, then the `SEANET_ENV` variable, then `local`.

There is deliberately **no** second training script for the cluster. `scripts/run_all.sh` just
calls `python main.py train --env grid5000`, so the training and evaluation code cannot drift
apart between the two machines.

---

## Where things live - the quick answer

| you want to change | edit |
|---|---|
| which dataset | `main.yaml: run.dataset`, or `--dataset` / the positional name |
| which model | `main.yaml: model`, or `--model` |
| the encoder | the model file's `encoder.type` (+ its parameters) |
| the MIL pooling | the model file's `pooling.type` (+ its parameters) |
| learning rate, batch size, epochs | the model file's `training:` block |
| the seed | `main.yaml: seed`, or `--seed` |
| where results go | `main.yaml: output.results_dir` / `output.analysis_dir` |
| MLflow on/off, experiment name | `main.yaml: mlflow` |
| the MLflow store | `configs/environments/<env>.yaml`, or `$MLFLOW_TRACKING_URI` |
| Optuna trials / sampler | `main.yaml: optuna` (defaults) or the model file's `optuna:` block |
| the device | `configs/environments/<env>.yaml: device` |

---

Next: [05 - Results](05_results.md)
