# 09 - MLflow

MLflow is the project's automatic lab notebook. Every real training run and every Optuna trial is
recorded, and you compare them in a web page instead of grepping through logs.

All the MLflow code is in **one file**, `seanet/tracking.py`. No training script talks to MLflow
directly - `seanet/evaluation.py` is its only caller for training runs, and
`seanet/optuna_search.py` for trials. If logging is off, not installed, or a single call fails,
everything quietly no-ops and training carries on.

---

## What gets logged for every run

**Params (the inputs)** - the full resolved config, flattened: model, dataset, seed, encoder type
and all its settings, pooling type and all its settings, learning rate, weight decay, dropout,
label smoothing, `lambda_entropy`, batch cap, epochs, patience, validation policy.

**Metrics (the results)**

| | |
|---|---|
| final | `train_acc`, `train_loss`, `val_acc`, `val_loss`, `test_acc`, `test_bal_acc`, `test_auroc`, `test_loss`, `test_aopcr`, `test_ndcg` |
| cost | `params`, `model_size_mb`, `train_time_s` |
| training behaviour | `epochs_run`, `best_epoch`, `overfit_gap` |
| per epoch (a curve) | `epoch_train_loss`, `epoch_train_acc`, `epoch_val_loss`, `epoch_val_acc` |

The four `epoch_*` series draw as line charts in the UI, so you can see train vs validation loss
and accuracy diverge without opening any file.

**Tags (what you filter on)** - `env` (local / grid5000), `host`, `model_id`, `config`, `encoder`,
`pooling`, `dataset`, `seed`, `date`, `command`.

**The model** - registered as a versioned model named after the `model_id`, so every run of the
same model is listed together in the "Models" tab. The weights are saved as an artifact when
`log_model_weights` is true (it is on `grid5000`, off on `local` - a laptop run is a check, not a
keeper).

Smoke runs are **never** logged. The web page only ever holds real results.

---

## One experiment, on purpose

Everything goes into a single experiment (`SEA-Net`), not one per model per day. That is what
makes the thing you actually want possible: a laptop run and a Grid5000 run of the same model,
side by side in one table. Use the tags to narrow down:

```text
tags.model_id = 'seanet_bottleneck_topk__sea_mstcn_sep_bottleneck__sea_topk_conjunctive'
tags.env = 'grid5000'
tags.encoder = 'sea_mstcn_sep_bottleneck'
tags.dataset = 'WebTraffic' and metrics.test_acc > 0.9
```

If you really want the old grouping back, set `mlflow.split_by_model: true` in
`configs/main.yaml`.

---

## Where the runs are stored

`seanet/tracking.py: tracking_uri()` decides, in this order:

1. **the `MLFLOW_TRACKING_URI` environment variable** - wins over everything
2. `mlflow.tracking_uri` from `configs/environments/<env>.yaml`
3. `sqlite:///mlflow.db` in the repo

That order is the whole design: the config gives each machine a sensible default, and one
environment variable overrides both machines at once when you want them to share.

> A SQLite file, not the old `./mlruns` folder: recent MLflow dropped the folder store, and a
> database is what lets us register versioned models.

### Browse locally

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# then open http://127.0.0.1:5000
```

---

## Getting local and Grid5000 into ONE view

There are two honest options. Pick one.

### Option A - a shared tracking server (best, if you have somewhere to put it)

Run an MLflow server somewhere both machines can reach, then point both at it:

```bash
# on the host
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlartifacts \
              --host 0.0.0.0 --port 5000

# on your laptop, and on the Grid5000 frontend/node, before running anything
export MLFLOW_TRACKING_URI=http://<host>:5000
```

Nothing else changes - the pipeline reads the variable and both environments log to the same
place. Runs are already tagged with `env` and `host`, so you can always tell them apart.

**The catch on Grid5000:** compute nodes usually have **no internet**. This only works with a
server reachable from inside the cluster, or via the frontend. Check before relying on it.

### Option B - copy the database back (always works)

Each machine writes its own `mlflow.db`. After a sweep:

```bash
scp <site>:~/SEA_NET/mlflow.db  mlflow_grid5000.db
mlflow ui --backend-store-uri sqlite:///mlflow_grid5000.db --port 5001
```

Now you have two UIs on two ports. To get one combined view, use the MLflow export/import tooling
(`pip install mlflow-export-import`) to pull the cluster's runs into your local store:

```bash
export-experiment  --experiment SEA-Net --output-dir /tmp/exp \
                   --tracking-uri sqlite:///mlflow_grid5000.db
import-experiment  --experiment SEA-Net --input-dir  /tmp/exp \
                   --tracking-uri sqlite:///mlflow.db
```

The `env` and `host` tags keep the two sets distinguishable after the merge.

`mlflow.db`, `mlruns/` and `mlartifacts/` are all git-ignored - a database in git bloats the
history and conflicts on every merge.

---

## Turning it off

```yaml
# configs/main.yaml
mlflow:
  enabled: false
```

Or just do not install MLflow. Training prints a one-line note and carries on; `results.csv`,
`history/` and every figure are unaffected. Nothing in the pipeline depends on MLflow being there.

---

Next: [10 - Optuna](10_optuna.md)
