# 03 - Running experiments

Everything runs through `main.py`. `python main.py -h` lists the commands; `python main.py single -h`
lists the flags of one.

---

## The two kinds of command

**Cheap** - they only read results that already exist. Safe to run any time:

```bash
python main.py models                 # every model config you can pass to --model
python main.py summary Coffee         # dataset stats
python main.py params                 # SEA-Net vs baseline parameter counts
python main.py results                # each model's comparison table vs MILLET
python main.py leaderboard            # one table, every model, best WebTraffic accuracy first
python main.py analyse                # every cross-model comparison figure + table
python main.py report                 # the per-model figures
python main.py web-compare            # WebTraffic-only comparison + accuracy tiers
```

**Expensive** - these really train:

```bash
python main.py single Coffee --model seanet_bottleneck_topk
python main.py train  --model seanet_bottleneck_topk
python main.py webtraffic --model seanet_bottleneck_topk
python main.py run    --model seanet_bottleneck_topk
python main.py interpret --model seanet_bottleneck_topk
python main.py optuna --model seanet_bottleneck_topk
```

Add `--smoke` to any of them: 3 epochs, nothing saved, nothing logged to MLflow. **Always smoke
first**, especially before starting something that will run for a day.

---

## `single` vs `train` - the only difference is how many datasets

| | `single NAME` | `train` |
|---|---|---|
| datasets | one, the one you name | every one: WebTraffic + 128 UCR |
| time | seconds to minutes | hours to days |
| writes | one row | one row per dataset |
| resumable | n/a | yes - Ctrl+C and start again |

Same config, same training code. `train` is `single` repeated down the whole list, plus the
remembering. Try it small first:

```bash
python main.py train --model seanet_bottleneck_topk --limit 5      # first 5 datasets only
python main.py train --model seanet_bottleneck_topk --only Coffee Beef GunPoint
python main.py train --model seanet_bottleneck_topk --no-webtraffic
```

---

## The flags you will use

| flag | what it does |
|---|---|
| `--model M` | which config. The folder is optional: `seanet_bottleneck_topk` or `seanet/seanet_bottleneck_topk` |
| `--env NAME` | which `configs/environments/NAME.yaml`. Default `local`, or `$SEANET_ENV` |
| `--seed N` | a NEW seed ADDS a repeat row; it never overwrites seed 0 |
| `--smoke` | 3 epochs, nothing saved |
| `--dataset NAME` | override the dataset (`run` / `interpret` / `optuna`; `single` takes it positionally) |
| `--config PATH` | a different `main.yaml` |

---

## What one run does, step by step

This is `seanet/training.py` -> `seanet/evaluation.py`, in order:

1. seed python, numpy and torch (`utils.set_seed`)
2. load the train and test split (`data.load_dataset`)
3. hold out a validation split if the training set has >= `min_train_for_val` series
   (`preprocessing.prepare_splits`) - otherwise train on all of it and early-stop on training loss
4. build `encoder -> pooling head` from the config (`models.build_model_from_config`)
5. for every epoch:
   - train over the batches, accumulating **train loss** and **train accuracy** as it goes
   - one pass over the validation set -> **val loss** and **val accuracy**
   - keep the weights if the monitored loss improved; stop after `patience` epochs without
6. reload the best epoch's weights
7. save the history: `history/<dataset>__seed<N>.csv` plus a loss curve and an accuracy curve PNG
8. score on the test set: accuracy, balanced accuracy, AUROC, loss, AOPCR, and NDCG on WebTraffic
9. save the per-series test probabilities (`predictions/`), for ensembling later
10. write one row into `results.csv` and log the whole run to MLflow

The four per-epoch numbers cost nothing extra: the training loop already has those logits, and the
validation pass was needed for early stopping anyway.

---

## Resuming a sweep

Each model has `results/SEA_NET/<model_id>/done_train_dataset.txt` - a plain list of what it
finished. `train` skips anything in it, so Ctrl+C and restart is always safe.

To retrain:

* delete the whole file -> every dataset again
* delete one line -> just that dataset
* set `run.re_train: true` in `configs/main.yaml` -> train again and REPLACE the row

Without `re_train`, a re-run only overwrites the old row if it beats the old accuracy
(`save_result_row` keeps the better result), so `results.csv` always holds the best numbers seen.

---

## Reproducing a result exactly

Everything that decides an outcome is written down:

| what | where |
|---|---|
| the seed | `configs/main.yaml` (`seed:`) or `--seed`, and the `seed` column of `results.csv` |
| encoder + pooling + every hyperparameter | the model YAML, and the `encoder`/`pooling` columns of `results.csv` |
| the device and environment | the `device` column, and the `env` / `host` tags in MLflow |
| the exact resolved config | printed in full by `python main.py run`, and logged to MLflow as params |
| the training curve | `results/SEA_NET/<model_id>/history/<dataset>__seed<N>.csv` |
| the per-series predictions | `results/SEA_NET/<model_id>/predictions/<dataset>__seed<N>.npz` |
| the console output | `results/SEA_NET/<model_id>/logs/<command>_<date-time>.log` |

So to repeat a row of `results.csv`:

```bash
python main.py single <dataset> --model <config> --seed <seed>
```

`utils.set_seed` seeds `random`, `numpy` and `torch` together. GPU kernels still introduce tiny
non-determinism, so expect the last decimal to move, not the result.

---

## Several seeds

```bash
for s in 0 1 2; do python main.py single WebTraffic --model seanet_bottleneck_topk --seed $s; done
```

Each seed adds its own row. `results.py` averages them (`mean_over_seeds`), and
`scripts/ensemble_vote.py` reports mean +- std and runs the paired Wilcoxon test against MILLET.

---

## After the experiments

```bash
python main.py leaderboard                # one row per model
python scripts/profile_models.py          # FLOPs / latency / memory (once)
python main.py analyse                    # every comparison figure + table -> results/analysis/
```

Read `results/analysis/INDEX.md` - it lists every figure and the question it answers.

---

Next: [04 - Configuration](04_configuration.md)
