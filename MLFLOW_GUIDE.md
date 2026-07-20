# Viewing results locally (MLflow + logs)

A short, practical guide: what the pipeline records in MLflow, how to open the MLflow web page on
your laptop, and how to copy results back from Grid5000. No prior MLflow knowledge needed.

---

## 1. What MLflow is used for here

MLflow is an automatic "lab notebook": it records each **run** (its inputs + the numbers they
produced) and gives you a web page to compare runs and models.

In this project MLflow records **every training / evaluation run and every Optuna trial**:

| Command | What gets recorded |
|---|---|
| `python main.py run` / `single` / `webtraffic` / `train` | one run per trained model: its hyperparameters (inputs), its **train / validation / test accuracy + loss**, AOPCR, NDCG, the per-epoch loss curve, and the trained model saved as a **versioned model** |
| `python main.py optuna` | one run per trial: the sampled hyperparameters + the validation loss (sort by it to pick the best) |

So you can open the web page and answer questions like *"which model / dataset / hyperparameters gave
the best test accuracy?"* or *"how do train, validation and test accuracy compare (is it overfitting)?"*
by comparing runs and versioned models side by side.

Turn it on/off in [`configs/main.yaml`](configs/main.yaml) under the `mlflow:` block. `--smoke` runs are
**not** logged (they are throwaway 3-epoch checks, so they never clutter what you compare).

> **Storage note.** Recent MLflow versions dropped the old "`./mlruns` folder" store, so we store the
> runs in a small **SQLite database file, `mlflow.db`**, and the saved model weights under
> `mlartifacts/`. Both are git-ignored (each machine keeps its own) and the single `mlflow.db` copies
> cleanly from Grid5000 to your laptop. Every `mlflow` command below takes `--backend-store-uri
> sqlite:///mlflow.db`.
>
> **torch note.** The trained weights are saved with `mlflow.pytorch.log_state_dict` (a plain
> `torch.save`) rather than the full pytorch "flavor", because the flavor needs `torch>=2.1` and we
> pin `torch==2.0.1`. The versioned model, its params and its metrics are recorded either way.

---

## 2. Where the pipeline saves things

Everything lands under the repo, so it is easy to copy around. All paths are relative to the project
root (`SEA_NET/`, the folder that contains `main.py`):

| What | Where | Made by |
|---|---|---|
| MLflow runs + trials + versioned models + metrics | `mlflow.db` (SQLite file) | every training command + `optuna` |
| Saved model weights (per versioned model) | `mlartifacts/` | training commands (when `log_model_weights: true`) |
| Best hyperparameters found | `configs/models/<model>.yaml` (the `records.optuna_best` block) | `optuna` |
| Run logs (a dated copy of the terminal output) | `results/SEA_NET/<model_id>/logs/<command>_<date-time>.log` | **every** command |
| Metrics table (accuracy, loss, AOPCR, NDCG, ...) | `results/SEA_NET/<model_id>/results.csv` | `single`, `train`, `run` |
| Comparison vs MILLET + headline means | `results/SEA_NET/<model_id>/comparison_vs_millet.csv`, `summary.*` | `results`, `report` |
| Figures | `results/SEA_NET/<model_id>/figures/` | `report` |
| Cross-model ranking | `results/SEA_NET/model_comparison.csv` + `figures/model_comparison.png` | `results`, `report` |
| Interpretability figures | `results/SEA_NET/<model_id>/interpretation/WebTraffic/<date-time>/` | `interpret` |

`<model_id>` is `<encoder>_<pooling>`, e.g. `mstcn_sep_additive` for `--model seanet`. Every model
keeps its own copy of all of the above, so two models never mix their numbers up.

Every command writes a dated log file, so **smoke, train, optuna and the rest all leave a permanent
record** of exactly what was printed — useful for the long sweeps you run on Grid5000.

> Saving weights for the full 129-dataset sweep uses some disk. To log only params + metrics (no
> weight files) set `log_model_weights: false` in the `mlflow:` block.

---

## 3. Open the MLflow web page on your laptop

After any training run or Optuna search (or after copying `mlflow.db` from Grid5000 — see section 4):

```bash
cd SEA_NET                       # the folder that contains mlflow.db
mlflow ui --backend-store-uri sqlite:///mlflow.db    # starts a small local web server
```

Then open **http://127.0.0.1:5000** in your browser and:

**To compare trained models (run / single / train):**
1. Click the **`SEA-Net`** experiment on the left — you see one row per run.
2. Add columns for `metrics.train_acc`, `metrics.val_acc`, `metrics.test_acc`, `metrics.test_loss`,
   `metrics.test_aopcr`, and click a column header to **sort**.
3. Tick two or more rows and press **Compare** to see them side by side (train vs val vs test tells
   you at a glance whether a model is overfitting).
4. The **Models** tab lists every trained model as a versioned entry (name = the model, e.g. `seanet`),
   each carrying its params and metrics — this is the "compare all of them" view.

**To pick the best Optuna hyperparameters:**
1. In the runs table, click the **`val_loss`** column header to **sort ascending**.
2. The **top row is the best trial** — click it to see the exact hyperparameters. The same winning
   values are also written into `configs/models/<model>.yaml` itself, in the `records.optuna_best`
   block at the bottom, so you don't have to copy them by hand. The `use_params` setting at the top
   of that file decides whether a run trains with them (`default` / `optuna_best` / `auto`).

---

## 4. See Grid5000 results on your laptop

You run the heavy training/searches on Grid5000, then browse them locally. The trick: **copy the file
back, then run `mlflow ui` on your laptop.**

**Step 1 — on Grid5000**, train or search inside the project folder (wherever you cloned `sea-net.git`,
e.g. `~/sea-net`):

```bash
python main.py train        # the full sweep; each dataset -> one run + one versioned model
# or
python main.py optuna       # a hyperparameter search (needs optuna.enabled: true in the model yaml)
```

**Step 2 — on your laptop**, copy the results back with `scp` (or `rsync`). Replace
`user@access.grid5000.fr` and the path with your own:

```bash
# from your laptop, inside your local SEA_NET/ folder:

# the MLflow database (runs + trials + versioned models + metrics) -> needed for the web page
scp user@access.grid5000.fr:~/sea-net/mlflow.db ./

# (optional) the saved model weights, if you want to reload models later
scp -r user@access.grid5000.fr:~/sea-net/mlartifacts ./

# (optional) the tuned hyperparameters (they live inside the model config now)
scp user@access.grid5000.fr:~/sea-net/configs/models/seanet.yaml ./configs/models/

# (optional) one model's whole results folder: results.csv, done list, logs, figures, summary
scp -r user@access.grid5000.fr:~/sea-net/results/SEA_NET/mstcn_sep_additive ./results/SEA_NET/
```

`rsync` is better for repeat copies (it only sends what changed):

```bash
rsync -avz user@access.grid5000.fr:~/sea-net/mlflow.db ./
rsync -avz user@access.grid5000.fr:~/sea-net/mlartifacts/ ./mlartifacts/
rsync -avz user@access.grid5000.fr:~/sea-net/results/ ./results/
```

**Step 3 — on your laptop**, browse them:

```bash
cd SEA_NET
mlflow ui --backend-store-uri sqlite:///mlflow.db    # open http://127.0.0.1:5000
```

That's it — the runs and models you produced on the cluster now show up in your local MLflow web page,
and the copied `seanet.yaml` (with its `records.optuna_best` block) makes your next local
`python main.py run` able to use the tuned hyperparameters.

> Tip: if you copy only `mlflow.db` (not `mlartifacts/`), you can still see every run, its params and
> all its metrics, and compare the versioned models — you just can't reload the saved weights.
