"""
seanet/tracking.py - MLflow, explained simply.

What MLflow is:
    MLflow is an automatic "lab notebook". It records each run (a set of inputs + the numbers they
    produced) into a local database, and gives you a web page to compare all your runs and models.

What we use it for HERE (everything now, not just Optuna):
    Every time we train and evaluate a model - `python main.py run / single / webtraffic / train` -
    AND every Optuna trial is recorded. For each run we save:
      - the INPUTS  : the hyperparameters (learning rate, dropout, ...) as "params",
      - the RESULTS : train / validation / test accuracy and loss, plus AOPCR and NDCG, as "metrics",
      - the MODEL   : the trained network is saved as a *versioned model* (a "LoggedModel"), so you
                      can open the MLflow "Models" tab and compare every version side by side.
    Afterwards you open the web page and compare: which model, which dataset, which hyperparameters
    gave the best test accuracy / lowest loss. That is the whole point.

This mirrors the official MLflow 3.x "LoggedModel" flow (adapted so it also works on torch 2.0.1 -
the pytorch model "flavor" needs torch>=2.1, so we register the model with create_external_model and
save the raw weights separately):
    with mlflow.start_run():
        logged = mlflow.create_external_model(name="seanet", params={...})   # a versioned model entity
        logged = mlflow.get_logged_model(logged.model_id)                    # read it back
        mlflow.log_metrics(metrics={...}, model_id=logged.model_id)          # link metrics to it
        mlflow.pytorch.log_state_dict(net.state_dict(), "weights")           # (optional) save the weights
    log_run() below wraps exactly that so the training code stays short.

The individual MLflow calls, each with a one-line example:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # where to store runs (here: a local db file)
    mlflow.set_experiment("SEA-Net")      # group runs under a named experiment (like a folder/tab)
    mlflow.start_run(run_name="seanet_Coffee")  # begin ONE run; use a `with` block so it auto-closes
    mlflow.log_param("learning_rate", lr) # save one INPUT value for this run (a hyperparameter)
    mlflow.log_metric("test_acc", 0.93)   # save one RESULT number for this run
    mlflow.log_metrics({"test_acc": 0.93, "test_loss": 0.21})   # save several results at once
    mlflow.create_external_model(name="seanet", params={...})   # register a versioned model to compare
    mlflow.get_logged_model(model_id)     # read a versioned model back (its params + metrics)
    mlflow.pytorch.log_state_dict(net.state_dict(), "weights")  # (optional) save the trained weights
    mlflow.set_tags({"model": "seanet", "dataset": "Coffee"})   # attach searchable labels

Everything here is safe: if MLflow is turned off in the config, or is not installed, or a single log
call fails, every function quietly does nothing and training carries on unaffected.

Where the runs are stored:
    A small local SQLite database file, `mlflow.db`, in the repo (no server needed). We use a database
    file (not the old "./mlruns folder") because recent MLflow versions dropped the folder store, and
    because a database is what lets us save *versioned* models (the Model Registry). The saved model
    weights go under `mlartifacts/`. Both are git-ignored (regenerated per machine). Browse with:
        mlflow ui --backend-store-uri sqlite:///mlflow.db
    See MLFLOW_GUIDE.md for how to open the web page and how to copy runs back from Grid5000.

Related files:
    - configs/main.yaml   -> the "mlflow" block (enabled, experiment_name, tracking_uri, log_model_weights).
    - seanet/train.py     -> score_model() logs each training/eval run via log_run() below.
    - seanet/optimize.py  -> logs each Optuna trial via trial_run() / log_params() / log_metric().
"""
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Optional


def _import_mlflow():
    """Import mlflow if it is installed, else return None (so callers can quietly do nothing)."""
    try:
        import mlflow                                           # imported here so the project runs without it
        return mlflow
    except ImportError:
        return None


def _as_float(value):
    """Turn a value into a float for logging, or None if it is not a real number (NaN / not numeric)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:                                       # NaN check (NaN != NaN is True)
        return None
    return number


def is_enabled(cfg) -> bool:
    """
    Say whether MLflow tracking is switched on in the config (the `mlflow.enabled` flag).

    cfg : a loaded config.
    returns : True if cfg.mlflow.enabled is true.
    """
    m = getattr(cfg, "mlflow", None)
    return bool(m is not None and getattr(m, "enabled", False))


def experiment_name(cfg, model: Optional[str] = None, when: Optional[str] = None) -> str:
    """
    Build the experiment name a run is filed under: "<base>_<model>_<date>".

    Grouping by model AND day is what keeps the MLflow page readable once several models have each
    swept 129 datasets: instead of one endless "SEA-Net" list, you get e.g.

        SEA-Net_seanet_2026-07-15
        SEA-Net_seanet_acp_2026-07-15
        SEA-Net_seanet_2026-07-16          <- yesterday's sweep stays separate

    so "the seanet sweep I ran on the 15th" is one click, and re-running the same model on another
    day never mixes the two.

    cfg : a loaded config (reads cfg.mlflow.experiment_name as the base).
    model : the model config name, e.g. "seanet". None -> the base name is used unchanged.
    when : the date stamp; None -> today. Passed in only by the tests.
    returns : the experiment name string.
    """
    base = getattr(getattr(cfg, "mlflow", None), "experiment_name", "SEA-Net") or "SEA-Net"
    if not model:
        return base
    date = when or datetime.now().strftime("%Y-%m-%d")
    return f"{base}_{model}_{date}"


def start_experiment(cfg, model: Optional[str] = None):
    """
    Point MLflow at the right store + experiment, ONCE, before you start logging runs.

    Call this once at the start of a command. It returns the mlflow module (to pass to log_run /
    trial_run below) or None if MLflow is off in the config or not installed - in which case all the
    logging becomes a no-op automatically.

    cfg : a loaded config (reads cfg.mlflow.tracking_uri and cfg.mlflow.experiment_name).
    model : the model being run, e.g. "seanet". Runs are filed under "<base>_<model>_<date>" so each
            model's daily sweep is its own group in the MLflow UI (see experiment_name above).
    returns : the mlflow module, or None.
    """
    if not is_enabled(cfg):
        return None
    mlflow = _import_mlflow()
    if mlflow is None:                                          # enabled but not installed -> warn once, no-op
        print("  [mlflow] enabled in config but not installed -> skipping (pip install 'mlflow>=3.1')")
        return None
    # a local SQLite file by default (recent MLflow dropped the old ./mlruns folder store); blank in
    # the config falls back to the same default so tracking always works out of the box.
    uri = getattr(cfg.mlflow, "tracking_uri", "") or "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(uri)
    experiment = experiment_name(cfg, model)
    mlflow.set_experiment(experiment)                          # all runs get grouped under this name
    print(f"  [mlflow] logging runs to experiment '{experiment}' at {uri}")
    print(f"  [mlflow] browse later with:  mlflow ui --backend-store-uri {uri}")
    return mlflow


def log_run(mlflow, *, run_name: str, params: Dict, metrics: Dict, tags: Optional[Dict] = None,
            history=None, model=None, model_params: Optional[Dict] = None,
            logged_model_name: Optional[str] = None, verbose: bool = False) -> Optional[str]:
    """
    Record ONE training/evaluation run: its inputs, its results, and (optionally) the model weights.

    This is the high-level helper the training code calls. It follows the official MLflow 3.x
    "LoggedModel" flow (see the module docstring): open a run, log the hyperparameters, register a
    versioned model entity, log the metrics linked to that model, and optionally save the weights.

    Example:
        log_run(mlflow, run_name="seanet_Coffee",
                params={"learning_rate": 0.00125, "dropout": 0.2},          # the inputs
                metrics={"train_acc": 0.98, "val_acc": 0.92, "test_acc": 0.90, "test_loss": 0.21},
                model=net, logged_model_name="seanet")                      # the versioned model

    mlflow : the module returned by start_experiment (or None -> this whole function is a no-op).
    run_name : a readable name for the run (shows up in the runs table).
    params : the INPUTS to log (hyperparameters / the resolved config).
    metrics : the RESULTS to log (train/val/test accuracy + loss, AOPCR, NDCG, ...). None / NaN are skipped.
    tags : optional searchable labels (model, dataset, command, ...).
    history : optional list of per-epoch losses -> logged as an "epoch_loss" curve you can plot.
    model : optional torch nn.Module whose weights are saved as an artifact (None = don't save weights).
    model_params : optional small dict describing the model (size, name) attached to the versioned model.
    logged_model_name : the name to register the versioned model under (all runs with the same name are
                        listed together in the MLflow "Models" tab, so you can compare them).
    verbose : print a short confirmation line.
    returns : the versioned model's id (a string), or None if MLflow is off.
    """
    if mlflow is None:                                         # MLflow off / not installed -> do nothing
        return None
    with mlflow.start_run(run_name=run_name):                 # the `with` block ends the run for us
        if tags:
            mlflow.set_tags(tags)

        # --- log the INPUTS (one param per hyperparameter -> one column in the runs table) ---
        for key, value in (params or {}).items():
            mlflow.log_param(key, value)                      # mlflow stores each as a string

        # --- register a versioned MODEL entity so every run shows up in the "Models" tab ---
        # create_external_model records the model's identity + a short description WITHOUT serialising
        # the weights, so it works on any torch version (the pytorch "flavor" would need torch>=2.1).
        logged = mlflow.create_external_model(
            name=logged_model_name or "model",
            params={k: str(v) for k, v in (model_params or {}).items()},   # small model description
        )
        model_id = logged.model_id
        if verbose:
            back = mlflow.get_logged_model(model_id)          # read it back, like the official example
            print(f"    [mlflow] versioned model '{back.name}' (id={model_id}) params={back.params}")

        # --- log the RESULTS, linked to the versioned model (so the Models tab shows each one's scores) ---
        clean = {k: _as_float(v) for k, v in (metrics or {}).items()}
        clean = {k: v for k, v in clean.items() if v is not None}   # drop None / NaN (e.g. NDCG on UCR)
        if clean:
            mlflow.log_metrics(metrics=clean, model_id=model_id)

        # --- (optional) save the actual trained weights as an artifact you can reload later ---
        if model is not None:
            try:
                import mlflow.pytorch                          # log_state_dict just torch.saves the weights
                mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path="weights")
            except Exception as e:                             # never let weight-saving fail the run
                print(f"    [mlflow] could not save weights (metrics still logged): {type(e).__name__}: {e}")

        # --- log the training curve (one loss value per epoch -> a line chart in the UI) ---
        if history:
            for step, loss in enumerate(history):
                value = _as_float(loss)
                if value is not None:
                    mlflow.log_metric("epoch_loss", value, step=step)   # step = epoch number
    return model_id


# --------------------------------------------------------------------------------------
# Optuna-trial logging (used by seanet/optimize.py). A trial logs only params + val_loss
# (no model file), because a search runs dozens of trials and we only need to compare their scores.
# --------------------------------------------------------------------------------------
@contextmanager
def trial_run(mlflow, run_name: str, tags: Optional[Dict] = None):
    """
    Open ONE MLflow run for a single Optuna trial (or a do-nothing context if mlflow is None).

    Use it like:
        with trial_run(mlflow, "seanet_WebTraffic_trial3") as run:
            if run is not None:
                log_params(mlflow, {...})
                log_metric(mlflow, "val_loss", 0.12)

    mlflow : the module returned by start_experiment (or None -> no-op).
    run_name : a readable name for this run (shows up in the MLflow table).
    tags : optional searchable labels (dataset, model, ...).
    yields : the mlflow module if a run is active, else None.
    """
    if mlflow is None:
        yield None
        return
    with mlflow.start_run(run_name=run_name):                  # the `with` block ends the run for us
        if tags:
            mlflow.set_tags(tags)
        yield mlflow


def log_params(mlflow, params: Dict) -> None:
    """
    Save a whole dict of INPUT values (e.g. a trial's hyperparameters) to the current run.

    mlflow : the mlflow module (or None -> no-op).
    params : e.g. {"training.learning_rate": 0.002, "encoder.dropout": 0.3}.
    """
    if mlflow is None:
        return
    for key, value in params.items():
        mlflow.log_param(key, value)                           # each becomes a column in the runs table


def log_metric(mlflow, name: str, value: float, step: Optional[int] = None) -> None:
    """
    Save one RESULT number (e.g. the validation loss) to the current run.

    mlflow : the mlflow module (or None -> no-op).
    name : the metric name, e.g. "val_loss".
    value : the number to store.
    step : optional step index (used when logging a value once per epoch to draw a curve).
    """
    if mlflow is None:
        return
    if step is None:
        mlflow.log_metric(name, float(value))
    else:
        mlflow.log_metric(name, float(value), step=step)
