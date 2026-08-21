# 10 - Optuna (hyperparameter search)

Optuna is **optional**. The normal training pipeline never touches it, and it is not a required
dependency. If you never run `python main.py optuna`, `seanet/optuna_search.py` is never imported.

## The important design point

Optuna has **no training code of its own**. One trial calls
`seanet.training.fit_model()` - the exact same function `python main.py single` calls - with a
copy of the config in which the sampled values have been substituted:

```text
Optuna  ->  the ONE training pipeline  ->  evaluation  ->  MLflow
```

So a searched model and a normally-trained model are trained identically, and a change to the
training loop applies to both automatically. There is no second copy to keep in sync.

---

## Run it

```bash
pip install optuna

python main.py optuna --model seanet_bottleneck_topk --smoke     # 2 trials x 3 epochs, plumbing check
python main.py optuna --model seanet_bottleneck_topk             # the real search
python main.py optuna --model seanet_bottleneck_topk --dataset Coffee
```

A search needs `optuna.enabled: true` in that model's config. If it is not there, the command says
so and stops rather than doing something surprising.

---

## What one trial does

1. Optuna suggests a value for each hyperparameter in the search space.
2. Those values are written into a **copy** of the config (the file on disk is not touched).
3. `fit_model()` trains once with that config.
4. After every epoch the trial reports its loss; the **pruner** kills a trial that is clearly worse
   than the others, so a bad combination does not waste a full run.
5. Train and validation accuracy AND loss are measured and printed.
6. The trial is logged to MLflow: the suggested params, plus `train_acc`, `train_loss`, `val_acc`,
   `val_loss`, and the model size.
7. The trial returns the **validation loss**, which is what Optuna minimises.

Optuna minimises validation loss, not test accuracy - deliberately. Searching on the test set is
how you fool yourself into a number you cannot reproduce.

---

## The search space is config, not code

Each model file's `optuna:` block. The dotted path is set straight into that model's config, so
you can tune the training recipe, the **encoder** and the **pooling head** in one search:

```yaml
optuna:
  enabled: true
  n_trials: 30
  timeout: null                 # seconds; null = no limit
  sampler: tpe                  # tpe (learns from past trials) | random
  pruner: median                # median (kill the below-average trials) | none
  save_best_parameters: true

  search_space:
    training:
      learning_rate:   {type: float, low: 0.0003, high: 0.005, log: true}
      weight_decay:    {type: float, low: 0.00001, high: 0.0005, log: true}
      label_smoothing: {type: float, low: 0.0, high: 0.2}
      lambda_entropy:  {type: float, low: 0.0, high: 0.05}
      max_batch:       {type: categorical, choices: [8, 16, 32]}
    encoder:
      d:            {type: categorical, choices: [64, 96, 128]}
      n_blocks:     {type: int, low: 3, high: 8}
      dropout:      {type: float, low: 0.1, high: 0.4}
      max_dilation: {type: categorical, choices: [8, 16, 32]}
    pooling:
      d_attn:  {type: categorical, choices: [4, 8, 16, 32]}
      dropout: {type: float, low: 0.1, high: 0.4}
```

Leaf types: `{type: float, low, high, log}`, `{type: int, low, high}`,
`{type: categorical, choices: [...]}`.

`n_trials`, `sampler`, `pruner` and `timeout` fall back to the project-wide defaults in
`configs/main.yaml` when the model block leaves them out.

**`log: true` matters for a learning rate.** Sampling uniformly between 0.0003 and 0.005 spends
most trials in the top half of the range; log-uniform spends them evenly across the orders of
magnitude, which is how learning rates actually behave.

---

## What happens at the end

The winner is retrained **once** and scored on the test set, so its numbers are directly
comparable with the default recipe's. Then both the winning params and their metrics are written
back into the *same* model YAML, under `records.optuna_best`:

```yaml
records:
  default:
    metrics: {test_acc: 0.938, test_loss: 0.2992, ...}
  optuna_best:
    params:  {training: {learning_rate: 0.0021, ...}, encoder: {d: 96, ...}}
    metrics: {test_acc: 0.951, val_loss: 0.19, n_trials: 30, ...}
```

One file per model, not a second `<model>.best.yaml`. Only the block below the
`# ===== AUTO-FILLED RESULTS =====` marker is rewritten, so your comments above it survive.

## Using the winner

The `use_params` key at the top of the model file:

| value | what it trains with |
|---|---|
| `default` | the hand-written recipe (the safe default) |
| `optuna_best` | the params Optuna found |
| `auto` | whichever of the two recorded higher `test_acc` |

`python main.py run` prints which one it picked and how the two compare, so a run can never
silently use a recipe you did not expect.

Smoke searches record nothing and log nothing.

---

Next: [11 - The MILLET baseline](11_millet_baseline.md)
