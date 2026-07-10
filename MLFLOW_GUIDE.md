# Viewing results locally (MLflow + logs)

A short, practical guide: where the pipeline saves things, how to open the MLflow web page on your
laptop, and how to copy results back from Grid5000. No prior MLflow knowledge needed.

---

## 1. What MLflow is used for here

MLflow is an automatic "lab notebook": it records each **run** (its inputs + the numbers they
produced) and gives you a web page to compare runs.

In this project MLflow does **exactly one job**: during an **Optuna hyperparameter search**
(`python main.py optuna`), every trial is recorded as one MLflow run — its hyperparameters (inputs)
and its validation loss (result). Afterwards you open the web page, **sort the trials by validation
loss, and the top row is the best set of hyperparameters**.

MLflow is **not** used for ordinary training (`python main.py run`, `single`, `train`). Those already
save their numbers to `results/SEA_NET/results.csv` and their full terminal output to
`results/SEA_NET/logs/` (see below), which is all you need for a single run.

Turn it on/off in [`configs/main.yaml`](configs/main.yaml) under the `mlflow:` block.

> **Storage note.** Recent MLflow versions dropped the old "`./mlruns` folder" store, so we store the
> runs in a small **SQLite database file, `mlflow.db`**, in the repo. It's one file, which also makes
> it trivial to copy from Grid5000 to your laptop. Every `mlflow` command below therefore takes
> `--backend-store-uri sqlite:///mlflow.db`.

---

## 2. Where the pipeline saves things

Everything lands under the repo, so it is easy to copy around. All paths are relative to the project
root (`SEA_NET/`, the folder that contains `main.py`):

| What | Where | Made by |
|---|---|---|
| MLflow trials (params + val_loss) | `mlflow.db` (SQLite file) | `optuna` |
| Best hyperparameters found | `configs/models/<model>.best.yaml` | `optuna` |
| Run logs (a dated copy of the terminal output) | `results/SEA_NET/logs/<command>_<date-time>.log` | **every** command |
| Metrics table (accuracy, AOPCR, NDCG, ...) | `results/SEA_NET/results.csv` | `single`, `train`, `run` |
| Paper-ready figures + summary tables | `results/SEA_NET/figures/`, `results/SEA_NET/summary.*` | `report` |
| Interpretability figures | `results/SEA_NET/interpretation/<dataset>/<date-time>/` | `interpret` |

Every command writes a dated log file, so **smoke, train, optuna and the rest all leave a permanent
record** of exactly what was printed — useful for the long sweeps you run on Grid5000.

---

## 3. Open the MLflow web page on your laptop

After an Optuna search (or after copying `mlflow.db` from Grid5000 — see section 4):

```bash
cd SEA_NET                       # the folder that contains mlflow.db
mlflow ui --backend-store-uri sqlite:///mlflow.db    # starts a small local web server
```

Then open **http://127.0.0.1:5000** in your browser and:

1. Click the **`SEA-Net`** experiment on the left.
2. You see one row per trial. Click the **`val_loss`** column header to **sort ascending**.
3. The **top row is the best trial** — click it to see the exact hyperparameters.
4. Tick two or more rows and press **Compare** to see them side by side (or use the
   parallel-coordinates plot to see which values lead to low loss).

The same winning values are also written to `configs/models/<model>.best.yaml`, which every future
run loads automatically — so you don't have to copy them by hand.

---

## 4. See Grid5000 results on your laptop

You run the heavy searches on Grid5000, then browse them locally. The trick: **copy the file back,
then run `mlflow ui` on your laptop**. Because each trial only logs small values (hyperparameters +
one number, no big files), the single `mlflow.db` copies cleanly between machines.

**Step 1 — on Grid5000**, run the search (inside the project folder — on the cluster that is wherever
you cloned `sea-net.git`, e.g. `~/sea-net`):

```bash
# turn optuna on (optuna.enabled: true in configs/models/seanet.yaml), then:
python main.py optuna                 # writes mlflow.db, seanet.best.yaml, and a log file
```

**Step 2 — on your laptop**, copy the results back with `scp` (or `rsync`). Replace
`user@access.grid5000.fr` and the path with your own:

```bash
# from your laptop, inside your local SEA_NET/ folder:

# the MLflow trials database (needed for the web page)
scp user@access.grid5000.fr:~/sea-net/mlflow.db ./

# the best hyperparameters it found
scp user@access.grid5000.fr:~/sea-net/configs/models/seanet.best.yaml ./configs/models/

# (optional) the run logs and the metrics table
scp -r user@access.grid5000.fr:~/sea-net/results/SEA_NET/logs ./results/SEA_NET/
scp user@access.grid5000.fr:~/sea-net/results/SEA_NET/results.csv ./results/SEA_NET/
```

`rsync` is better for repeat copies (it only sends what changed):

```bash
rsync -avz user@access.grid5000.fr:~/sea-net/mlflow.db ./
rsync -avz user@access.grid5000.fr:~/sea-net/results/ ./results/
```

**Step 3 — on your laptop**, browse them:

```bash
cd SEA_NET
mlflow ui --backend-store-uri sqlite:///mlflow.db    # open http://127.0.0.1:5000
#   -> SEA-Net experiment -> sort by val_loss -> top row is the best trial
```

That's it — the trials you ran on the cluster now show up in your local MLflow web page, and
`seanet.best.yaml` makes your next local `python main.py run` use the tuned hyperparameters.

> Tip: you don't strictly need MLflow to *use* the result — `seanet.best.yaml` alone is enough for
> training. MLflow is there so you can *see and compare* all the trials and understand why a set won.
