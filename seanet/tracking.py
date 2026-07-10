"""
seanet/tracking.py - MLflow, explained simply.

What MLflow is:
    MLflow records each "run" (a set of inputs + the numbers they produced) into a local folder, so
    you can later open a web page and compare all your runs. Think of it as an automatic lab
    notebook.

What we use it for HERE (only one thing):
    To record every Optuna trial. Each trial tries a set of hyperparameters and gets a validation
    loss. We log the hyperparameters (the inputs) and the validation loss (the result) as one MLflow
    run. Afterwards you open the MLflow web page, sort the trials by validation loss, and instantly
    SEE which hyperparameters were the best. That is the whole point: MLflow helps you pick the best
    parameters from an Optuna search.

    We do NOT use MLflow for ordinary training (`python main.py run`). Ordinary runs already save
    their numbers to results.csv and their text output to results/SEA_NET/logs/. MLflow is reserved
    for the hyperparameter search, where comparing dozens of runs is what it is good at.

The MLflow calls used below, each with a one-line example:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # where to store the runs (here: a local db file)
    mlflow.set_experiment("SEA-Net")      # group runs under a named experiment (like a folder/tab)
    mlflow.start_run(run_name="trial 3")  # begin ONE run; use inside a `with` block so it auto-closes
    mlflow.log_param("learning_rate", lr) # save one INPUT value for this run (a hyperparameter)
    mlflow.log_metric("val_loss", 0.12)   # save one RESULT number for this run
    mlflow.set_tags({"model": "seanet"})  # attach searchable labels to the run

Everything here is safe: if MLflow is turned off in the config, or is not installed, every function
quietly does nothing and the search still runs.

Where the runs are stored:
    A small local SQLite database file, `mlflow.db`, in the repo (no server needed). We use a database
    file because recent MLflow versions dropped the old "./mlruns folder" store. It is one file, so it
    copies cleanly from Grid5000 to your laptop. Browse it with:
        mlflow ui --backend-store-uri sqlite:///mlflow.db
    See MLFLOW_GUIDE.md for how to open the web page and how to copy runs back from Grid5000.

Related files:
    - configs/main.yaml   -> the "mlflow" block (enabled, experiment_name, tracking_uri).
    - seanet/optimize.py  -> the only caller: logs each Optuna trial with the helpers below.
"""
from contextlib import contextmanager
from typing import Dict, Optional


def _import_mlflow():
    """Import mlflow if it is installed, else return None (so callers can quietly do nothing)."""
    try:
        import mlflow                                           # imported here so the project runs without it
        return mlflow
    except ImportError:
        return None


def is_enabled(cfg) -> bool:
    """
    Say whether MLflow tracking is switched on in the config (the `mlflow.enabled` flag).

    cfg : a loaded config.
    returns : True if cfg.mlflow.enabled is true.
    """
    m = getattr(cfg, "mlflow", None)
    return bool(m is not None and getattr(m, "enabled", False))


def start_experiment(cfg):
    """
    Point MLflow at the right store + experiment, ONCE, before a search.

    Call this before the Optuna loop. It returns the mlflow module (to pass to trial_run/log_* below)
    or None if MLflow is off in the config or not installed - in which case all the logging becomes a
    no-op automatically.

    cfg : a loaded config (reads cfg.mlflow.tracking_uri and cfg.mlflow.experiment_name).
    returns : the mlflow module, or None.
    """
    if not is_enabled(cfg):
        return None
    mlflow = _import_mlflow()
    if mlflow is None:                                          # enabled but not installed -> warn once, no-op
        print("  [mlflow] enabled in config but not installed -> skipping (pip install mlflow)")
        return None
    # a local SQLite file by default (recent MLflow dropped the old ./mlruns folder store); blank in
    # the config falls back to the same default so tracking always works out of the box.
    uri = getattr(cfg.mlflow, "tracking_uri", "") or "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(uri)
    experiment = getattr(cfg.mlflow, "experiment_name", "SEA-Net")
    mlflow.set_experiment(experiment)                          # all trials get grouped under this name
    print(f"  [mlflow] logging trials to experiment '{experiment}' at {uri}")
    print(f"  [mlflow] browse later with:  mlflow ui --backend-store-uri {uri}")
    return mlflow


@contextmanager
def trial_run(mlflow, run_name: str, tags: Optional[Dict] = None):
    """
    Open ONE MLflow run for a single trial (or a do-nothing context if mlflow is None).

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
    Save a whole dict of INPUT values (the trial's hyperparameters) to the current run.

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
