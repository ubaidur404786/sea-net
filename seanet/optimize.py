"""
seanet/optimize.py - Optuna hyperparameter search (kept as close to the official tutorial as possible).

The official Optuna pattern (https://optuna.org) is only three steps:

    def objective(trial):            # 1) describe ONE experiment
        x = trial.suggest_float("x", -10, 10)   # ask Optuna for a value to try
        return (x - 2) ** 2          #    return the number to minimise (here: a made-up score)

    study = optuna.create_study(direction="minimize")   # 2) make the study (it runs the search)
    study.optimize(objective, n_trials=100)             # 3) run it; study.best_params has the winner

This file is exactly that, with two practical additions:
    - the hyperparameters and their ranges are read from the config (the `optuna` block in the model
      file) instead of being written in code - so tuning a new encoder/pooling head is just a config
      edit, no Python change; and
    - each finished trial is recorded to MLflow (its hyperparameters + its validation loss), so you
      can open the MLflow web page afterwards, sort by validation loss, and pick the best set. See
      seanet/tracking.py and MLFLOW_GUIDE.md.

Our objective: one trial = train the model once with the sampled values and return its best VALIDATION
LOSS (lower is better). When the search finishes, we retrain the best config once, score it on the test
set, and save the winning params + their metrics (test_acc / test_loss / test_auroc / val_loss) into the
SAME model file, under records.optuna_best (NOT a separate <model>.best.yaml - fewer files). You then
compare it with the default recipe's recorded metrics and pick which to use via the model file's
`use_params` setting (default / optuna_best / auto). See seanet/config.py (record_metrics, load_config).

Related files:
    - configs/models/<model>.yaml -> the "optuna" block (settings + search_space) AND the records block.
    - seanet/train.py  -> fit_model() (one trial = one fit), safe_evaluate(), _train_kwargs_from_config().
    - seanet/config.py -> record_metrics() saves the best into the model file; load_config() applies it.
    - seanet/tracking.py -> the MLflow helpers used to record each trial.
    - main.py ("optuna" command) -> calls run_optuna().
"""
import copy
from datetime import datetime
from typing import Dict, Iterator, Tuple

import torch

from seanet import tracking
from seanet.config import record_metrics
from seanet.model import num_params
from seanet.train import fit_model, _train_kwargs_from_config, get_device, safe_evaluate


# --------------------------------------------------------------------------------------
# Reading the search space from the config (grouped like the config: training.learning_rate, ...)
# --------------------------------------------------------------------------------------
def _iter_search_space(node, prefix: str = "") -> Iterator[Tuple[str, object]]:
    """
    Walk the (possibly nested) search_space and yield (dotted_path, spec) for each hyperparameter.

    A "leaf" is a hyperparameter if it has a "type" (float/int/categorical); anything else is a group
    to look inside. Example yield: ("training.learning_rate", <spec with low/high/log>).

    node : a search_space config node (SimpleNamespace).
    prefix : used by the recursion; leave empty when calling.
    yields : (dotted path into the model config, the leaf spec).
    """
    for key, value in vars(node).items():
        path = f"{prefix}{key}"
        if hasattr(value, "type"):                           # a hyperparameter (has type/low/high/...)
            yield path, value
        else:                                                # a group -> go one level deeper
            yield from _iter_search_space(value, prefix=path + ".")


def _suggest(trial, name: str, spec):
    """
    Ask Optuna for one value, using the spec's type and range.

    This is the config-driven version of the tutorial's `trial.suggest_float("x", -10, 10)`:
      float       -> trial.suggest_float(name, low, high, log=?)     e.g. learning_rate in [3e-4, 5e-3]
      int         -> trial.suggest_int(name, low, high, log=?)       e.g. n_blocks in [4, 8]
      categorical -> trial.suggest_categorical(name, choices)        e.g. optimizer in ["adam","sgd"]

    trial : the Optuna trial.
    name : the dotted path (also used as the parameter's name in Optuna/MLflow).
    spec : the leaf spec (has .type and its range).
    returns : the sampled value.
    """
    kind = spec.type
    if kind == "float":
        return trial.suggest_float(name, float(spec.low), float(spec.high), log=bool(getattr(spec, "log", False)))
    if kind == "int":
        return trial.suggest_int(name, int(spec.low), int(spec.high), log=bool(getattr(spec, "log", False)))
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec.choices))
    raise ValueError(f"Unknown search-space type {kind!r} for {name!r} (use float/int/categorical).")


def _set_nested(namespace, path: str, value) -> None:
    """
    Set a value deep inside a config namespace, e.g. path "training.learning_rate".

    namespace : the config object to change (always a copy - we never touch the original).
    path : dotted path.
    value : the value to set.
    """
    parts = path.split(".")
    obj = namespace
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _format_params(params: Dict) -> str:
    """Pretty one-line string of a params dict, e.g. 'training.learning_rate=0.0021, encoder.dropout=0.27'."""
    bits = []
    for key, value in params.items():
        bits.append(f"{key}={value:.5g}" if isinstance(value, float) else f"{key}={value}")
    return ", ".join(bits)


# --------------------------------------------------------------------------------------
# Saving the winners
# --------------------------------------------------------------------------------------
def _nested_from_flat(flat: Dict) -> Dict:
    """
    Turn {"training.learning_rate": 0.002, "encoder.dropout": 0.3} into nested dicts for YAML.

    flat : dict with dotted keys (Optuna's best_params).
    returns : nested dict, e.g. {"training": {"learning_rate": 0.002}, "encoder": {"dropout": 0.3}}.
    """
    out: Dict = {}
    for key, value in flat.items():
        parts = key.split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = float(value) if isinstance(value, float) else value
    return out


def _build_sampler(kind: str, seed: int):
    """
    Build the Optuna SAMPLER named in the config (it decides which values to try next).

    tpe    = Tree-structured Parzen Estimator: learns from past trials to pick promising values (smart).
    random = just picks values at random (a simple baseline).
    """
    import optuna
    if (kind or "tpe") == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed)             # default: TPE (smart search)


def _build_pruner(kind: str):
    """
    Build the Optuna PRUNER named in the config (it stops hopeless trials early to save time).

    median = stop a trial if, partway through, its loss is worse than the median of past trials.
    none   = never stop early (let every trial finish).
    """
    import optuna
    if (kind or "median") in ("none", "nop"):
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner()                     # default: median


# --------------------------------------------------------------------------------------
# The search itself (the three official steps: objective -> create_study -> optimize)
# --------------------------------------------------------------------------------------
def run_optuna(cfg, dataset=None, device=None, smoke: bool = False, verbose: bool = True):
    """
    Run an Optuna hyperparameter search using the model config's "optuna" block.

    cfg : a loaded config (reads cfg.model_config.optuna, cfg.seed and cfg.mlflow).
    dataset : dataset to search on; defaults to WebTraffic (the spec's default).
    device : where to train; if None, get_device() picks one.
    smoke : if True, a tiny search (2 trials x 3 epochs) to check the plumbing; nothing is saved or logged.
    verbose : print progress and the result.
    returns : the finished Optuna study, or None if Optuna is disabled / not installed.
    """
    ocfg = getattr(cfg.model_config, "optuna", None)
    if ocfg is None or not getattr(ocfg, "enabled", False):
        print(f"Optuna is disabled for model {cfg.model!r}. Set 'optuna.enabled: true' in "
              f"configs/models/{cfg.model}.yaml to run a search.")
        return None
    try:
        import optuna
    except ImportError:
        print("Optuna is not installed. Install it with:  pip install optuna")
        return None

    optuna.logging.set_verbosity(optuna.logging.WARNING)      # quiet Optuna's own per-trial logging (we print our own)
    if device is None:
        device = get_device()
    dataset = dataset or "WebTraffic"                         # spec: WebTraffic by default
    n_trials = 2 if smoke else int(getattr(ocfg, "n_trials", 20))
    timeout = getattr(ocfg, "timeout", None)
    seed = int(getattr(cfg, "seed", 0))
    space = list(_iter_search_space(ocfg.search_space))       # [(dotted_path, spec), ...] read from the config

    # MLflow records each trial so you can compare them later and pick the best. Skipped for smoke
    # (a plumbing check) so the MLflow web page only ever holds real searches.
    mlf = None if smoke else tracking.start_experiment(cfg, model=cfg.model)

    print(f"=== optuna: model={cfg.model} dataset={dataset} device={device} "
          f"mode={'smoke' if smoke else 'full'} ===")
    print(f"trials={n_trials} sampler={getattr(ocfg, 'sampler', 'tpe')} "
          f"pruner={getattr(ocfg, 'pruner', 'median')} | tuning: {[p for p, _ in space]}")
    print("objective: minimise validation loss (lower is better)\n")

    # ---- step 1: the objective. One call = one trial = train once and return the validation loss. ----
    def objective(trial):
        # (a) SUGGEST: ask Optuna for a value for each hyperparameter, and drop them into a config copy
        trial_cfg = copy.deepcopy(cfg)
        suggested = {}
        for path, spec in space:
            value = _suggest(trial, path, spec)
            _set_nested(trial_cfg.model_config, path, value)
            suggested[path] = value
        print(f"  trial {trial.number:>2}: {_format_params(suggested)}")   # show this trial's picks clearly

        # (b) TRAIN: one trial = one normal training run with those values (reusing fit_model)
        kwargs = _train_kwargs_from_config(trial_cfg, smoke=smoke)

        def epoch_callback(epoch, loss):          # lets the pruner stop a hopeless trial early
            trial.report(loss, step=epoch)        # tell Optuna the loss so far this trial
            if trial.should_prune():              # Optuna: "worse than the others" -> abandon this trial
                raise optuna.TrialPruned()

        model, _, _, _, _ = fit_model(dataset, device=device, verbose=False,
                                      epoch_callback=epoch_callback, **kwargs)

        # (c) SCORE: the trial's result is its best validation loss (this is what Optuna minimises)
        val_loss = min(model.history) if model.history else float("inf")
        n_model_params = num_params(model.net)                # network size for this config (logged to MLflow)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"           -> val_loss={val_loss:.4f}")

        # (d) RECORD in MLflow: the hyperparameters (inputs) + the val_loss (result) for this trial
        with tracking.trial_run(mlf, run_name=f"{cfg.model}_{dataset}_trial{trial.number}",
                                tags={"dataset": dataset, "model": cfg.model, "kind": "optuna-trial"}):
            tracking.log_params(mlf, suggested)
            tracking.log_params(mlf, {"model_params": n_model_params})
            tracking.log_metric(mlf, "val_loss", val_loss)

        return val_loss

    # ---- step 2: create the study (the object that runs the search and remembers every trial) ----
    study = optuna.create_study(
        direction="minimize",                                 # we want the SMALLEST validation loss
        sampler=_build_sampler(getattr(ocfg, "sampler", "tpe"), seed),
        pruner=_build_pruner(getattr(ocfg, "pruner", "median")),
    )

    # ---- step 3: run the search (calls objective n_trials times, or until timeout) ----
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    # ---- the result ----
    try:
        best_value, best_params = study.best_value, study.best_params
        best_number = study.best_trial.number
    except ValueError:                                          # no trial finished (all pruned/failed)
        print("\nNo trial completed successfully - nothing to save.")
        return study
    print(f"\nbest validation loss: {best_value:.4f}  (trial {best_number})")
    print("best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key} = {value}")

    if smoke:
        print("\n  (smoke = 2 trials x 3 epochs; best params NOT saved, MLflow not used)")
        return study
    if not getattr(ocfg, "save_best_parameters", True):
        return study

    # Retrain the best config ONCE and score it on the TEST set, so its metrics (test_acc / test_loss /
    # test_auroc) are directly comparable with the default recipe's recorded metrics. Then save the
    # params + these metrics into the SAME model yaml under records.optuna_best (no separate .best.yaml).
    print(f"\n  retraining the best config on {dataset} to record its test metrics ...", flush=True)
    best_cfg = copy.deepcopy(cfg)
    for path, value in best_params.items():
        _set_nested(best_cfg.model_config, path, value)
    model, _, _, test_ds, _ = fit_model(dataset, device=device, verbose=verbose,
                                        **_train_kwargs_from_config(best_cfg, smoke=False))
    cls = safe_evaluate(model, test_ds)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    metrics = {
        "test_acc": round(float(cls["acc"]), 4),
        "test_loss": round(float(cls["loss"]), 4),
        "test_auroc": round(float(cls["auroc"]), 4),
        "val_loss": round(float(best_value), 4),
        "dataset": dataset,
        "n_trials": len(study.trials),
        "recorded": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    saved_path = record_metrics(cfg.model, "optuna_best", metrics, params=_nested_from_flat(best_params))
    print(f"  saved best params + metrics -> {saved_path}  (records.optuna_best)")
    print(f"  best on {dataset}: test_acc={metrics['test_acc']} test_loss={metrics['test_loss']} "
          f"test_auroc={metrics['test_auroc']}  (val_loss={metrics['val_loss']})")
    print(f"  set `use_params: auto` (or optuna_best) in configs/models/{cfg.model}.yaml to train with these.")
    if mlf is not None:
        print("  compare all trials in the MLflow web page:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    return study
