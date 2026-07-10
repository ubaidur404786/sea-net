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

Our objective: one trial = train SEA-Net once with the sampled values and return its best VALIDATION
LOSS (lower is better). When the search finishes, the best values are written to
configs/models/<model>.best.yaml, which every future run loads on top of the model config - so the
whole project starts using the tuned values, with no code change (delete that file to go back).

Related files:
    - configs/models/<model>.yaml -> the "optuna" block (settings + search_space).
    - seanet/train.py  -> fit_model() (one trial = one fit) and _train_kwargs_from_config().
    - seanet/config.py -> load_config() merges <model>.best.yaml on top, so tuned values are used.
    - seanet/tracking.py -> the MLflow helpers used to record each trial.
    - main.py ("optuna" command) -> calls run_optuna().
"""
import copy
import os
from datetime import datetime
from typing import Dict, Iterator, Tuple

import torch
import yaml

from seanet import tracking
from seanet.config import MODELS_DIR
from seanet.model import num_params
from seanet.train import fit_model, _train_kwargs_from_config, get_device


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


def _save_best(model_name: str, study, n_params_tuned: int, verbose: bool = True) -> str:
    """
    Write Optuna's best hyperparameters (plus a short summary of the search) to <model>.best.yaml.

    That file is loaded automatically on top of the model config by load_config, so future runs use
    the tuned values. It is machine-generated - safe to delete to go back to the documented defaults.

    model_name : the model whose params were tuned.
    study : the finished Optuna study (for best_params, best_value, and the trial count).
    n_params_tuned : how many hyperparameters were searched.
    verbose : print where it was saved.
    returns : the path written.
    """
    nested = _nested_from_flat(study.best_params)
    path = os.path.join(MODELS_DIR, f"{model_name}.best.yaml")
    stamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    # the header doubles as a record of the search ("optuna related info"): when, how many trials,
    # how many hyperparameters, and the best score reached.
    header = (
        f"# {model_name}.best.yaml - AUTO-GENERATED by 'python main.py optuna' on {stamp}.\n"
        f"# Optuna searched {len(study.trials)} trials over {n_params_tuned} hyperparameter(s) and kept\n"
        f"# the best set below (lowest validation loss = {study.best_value:.4f}). load_config() merges\n"
        f"# this on top of {model_name}.yaml, so every run now uses these values. Delete this file to\n"
        f"# go back to the documented defaults in {model_name}.yaml.\n\n"
    )
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(nested, f, default_flow_style=False, sort_keys=False)
    if verbose:
        print(f"  best parameters saved -> {path}")
    return path


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
    mlf = None if smoke else tracking.start_experiment(cfg)

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
    print(f"\nbest validation loss: {study.best_value:.4f}  (trial {study.best_trial.number})")
    print("best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key} = {value}")

    if smoke:
        print("\n  (smoke = 2 trials x 3 epochs; best params NOT saved, MLflow not used)")
    elif getattr(ocfg, "save_best_parameters", True):
        _save_best(cfg.model, study, n_params_tuned=len(space), verbose=verbose)
        print("  future runs will now use these values automatically.")
        if mlf is not None:
            print("  compare all trials in the MLflow web page:  "
                  "mlflow ui --backend-store-uri sqlite:///mlflow.db")
    return study
