"""
seanet/evaluation.py - scoring a TRAINED model and turning the answer into one results row.

What this file is for:
    Training stops here. seanet/training.py fits a model; this file measures it on the test set,
    packs every number into one flat dictionary (a "row"), and hands that row to MLflow and to
    seanet/results.py.

    Keeping this apart from training matters in practice: if a run fails, the traceback tells you
    immediately whether it broke while LEARNING (training.py) or while MEASURING (this file), and
    the two do not have to be read together.

The row this file produces is the unit of everything downstream:
    one row = one (model, dataset, seed). results.csv is a pile of these rows; the leaderboard,
    every comparison table and every figure is built by grouping them.

    dataset, model, encoder, pooling, seed, device,
    params, model_size_mb, n_train, n_val, n_test, series_length, n_classes,
    test_acc, test_bal_acc, test_auroc, test_loss, test_aopcr, test_ndcg, train_time_s

Related files:
    - seanet/metrics.py   -> the metrics themselves (accuracy / AOPCR / NDCG).
    - seanet/training.py  -> produces the trained model this file scores.
    - seanet/tracking.py  -> the one MLflow interface; this file is its only caller for training runs.
    - seanet/results.py   -> writes the row to results/SEA_NET/<model_id>/results.csv.
"""
import os
from typing import Dict, Optional

import numpy as np
import torch

from seanet import metrics as M
from seanet import tracking
from seanet.models import num_params, state_dict_size_mb


# --------------------------------------------------------------------------------------
# Predictions (kept so models can be ensembled later without retraining them)
# --------------------------------------------------------------------------------------
def predict_proba(model, dataset):
    """
    Get the model's class PROBABILITIES for every series in a dataset.

    The model outputs "bag_logits" - raw scores, one per class. softmax turns a row of scores into
    probabilities that add up to 1. We need probabilities (not just the predicted class) so that two
    models can be ensembled by AVERAGING their confidence, which is usually better than only counting
    votes: a model that is 95% sure should outweigh one that is barely 40% sure.

    model : a trained model.  dataset : the set to predict (normally the test set).
    returns : (probs, y) - probs is (n_series, n_classes), y is the true label of each series.
    """
    logits, targets = [], []
    with torch.no_grad():                                        # no gradients needed, just predicting
        for batch in dataset.create_dataloader(batch_size=16):
            logits.append(model(batch["bags"])["bag_logits"].cpu())
            targets.append(batch["targets"])
    logits = torch.cat(logits)
    y = torch.cat(targets).long()
    return torch.softmax(logits, dim=1).numpy(), y.numpy()


def save_predictions(model, name: str, test_ds, seed: int, pred_dir: str) -> str:
    """
    Save this run's test-set probabilities to a small .npz file, so models can be ensembled LATER
    without retraining them.

    Why we need this: results.csv only stores summary numbers (accuracy, AOPCR...). You cannot build
    a majority vote out of two accuracies - voting needs to know what each model predicted for each
    individual series. One file per (dataset, seed) keeps that, and it is tiny: a 500-series,
    10-class test set is about 20 KB.

    "npz" is numpy's own zip format: several named arrays in one compressed file.

    model : the trained model.  name : dataset name.  test_ds : the test set.
    seed : the training seed (part of the filename, so seeds do not overwrite each other).
    pred_dir : the folder to write into (created if missing).
    returns : the path written.
    """
    os.makedirs(pred_dir, exist_ok=True)
    probs, y = predict_proba(model, test_ds)
    path = os.path.join(pred_dir, f"{name}__seed{int(seed)}.npz")
    np.savez_compressed(path, probs=probs.astype(np.float32), y=y.astype(np.int16))
    return path


# --------------------------------------------------------------------------------------
# The main entry point: score a trained model -> one results row
# --------------------------------------------------------------------------------------
def score_model(model, name: str, train_ds, val_ds, test_ds, device: torch.device, seed: int,
                lambda_entropy: float, train_time_s: float, verbose: bool = False,
                mlf=None, mlf_params: Optional[Dict] = None, mlf_tags: Optional[Dict] = None,
                logged_model_name: Optional[str] = None, log_model_weights: bool = True,
                pred_dir: Optional[str] = None, history_dir: Optional[str] = None,
                deploy_dir: Optional[str] = None) -> Dict:
    """
    Score a trained model on the test set, pack the results into one flat row, and (optionally)
    record the whole run in MLflow so every model can be compared later.

    It does NOT free the model; the caller decides when to do that.

    model : the trained SeaNetModel.
    name : dataset name.
    train_ds, val_ds, test_ds : the datasets (for the size fields + train/val metrics; val_ds may be None).
    device, seed, lambda_entropy, train_time_s : metadata for the row.
    verbose : if True, print the scoring stage line.
    mlf : the mlflow handle from tracking.start_experiment (None -> no MLflow logging happens).
    mlf_params : INPUTS to log to MLflow (usually the resolved config); if None a summary is logged.
    mlf_tags : extra searchable labels for the MLflow run (e.g. {"command": "run"}).
    logged_model_name : name for the versioned model in the MLflow "Models" tab (default: model.name).
    log_model_weights : also save the trained network's weights as an artifact.
    pred_dir : if given, save the per-series test probabilities there (for ensembling later).
    history_dir : if given, save the per-epoch training history + its curve figures there.
    deploy_dir : if given, save the complete deployment bundle there - the weights AND the config
                 that built them, plus TorchScript/ONNX. See seanet/deployment.py.
    returns : one flat results-row dict (same shape written to results.csv).
    """
    from seanet.training import save_history                     # imported here to avoid a cycle

    if verbose:
        print(f"    stage 3/3: scoring on {len(test_ds)} test series (accuracy + AOPCR"
              f"{' + NDCG' if name == 'WebTraffic' else ''}) ...", flush=True)
    cls = M.classification_metrics(model, test_ds)               # accuracy / bal_acc / auroc / loss
    aopcr, ndcg = M.interpretability_metrics(model, test_ds)     # AOPCR always, NDCG on WebTraffic only

    if pred_dir:                                                 # keep the per-series predictions so
        try:                                                     # models can be ensembled later
            save_predictions(model, name, test_ds, seed, pred_dir)
        except Exception as e:                                   # never fail a run over a side file
            print(f"    (could not save predictions for {name}: {type(e).__name__}: {e})", flush=True)
    if history_dir:                                              # keep the per-epoch curves so the
        try:                                                     # training behaviour can be inspected
            save_history(model, name, seed, history_dir)
        except Exception as e:                                   # never fail a run over a side file
            print(f"    (could not save history for {name}: {type(e).__name__}: {e})", flush=True)

    row = {
        "dataset": name,
        "model": model.name,
        "seed": seed,
        "device": str(device),
        "params": num_params(model.net),
        "model_size_mb": round(state_dict_size_mb(model.net), 4),
        "n_train": len(train_ds),
        "n_val": len(val_ds) if val_ds is not None else 0,
        "n_test": len(test_ds),
        "series_length": int(len(test_ds.get_bag(0))),
        "n_classes": int(test_ds.n_clz),
        "lambda_entropy": lambda_entropy,
        "test_acc": round(cls["acc"], 4),
        "test_bal_acc": round(cls["bal_acc"], 4),
        "test_auroc": round(cls["auroc"], 4),
        "test_loss": round(cls["loss"], 4),
        "test_aopcr": round(aopcr, 4),
        "test_ndcg": round(ndcg, 4) if ndcg is not None else None,   # None for UCR
        "train_time_s": round(train_time_s, 2),
        "epochs_run": int(len(getattr(model, "history", []) or [])),
        "best_epoch": int(getattr(model, "best_epoch", 0)),
        "overfit_gap": _overfit_gap(model),
    }

    if deploy_dir:                                               # keep the whole model, not just
        try:                                                     # its score, so it can be deployed
            from seanet.deployment import save_bundle
            path = save_bundle(
                model, name, seed, deploy_dir,
                model_cfg=getattr(model, "model_cfg", None),
                model_id=os.path.basename(os.path.dirname(deploy_dir)),
                metrics={k: row[k] for k in ("test_acc", "test_loss", "test_aopcr", "test_ndcg",
                                             "params", "model_size_mb") if k in row},
                series_length=row["series_length"], n_in=1,
            )
            if verbose:
                print(f"    deployment bundle -> {path}", flush=True)
        except Exception as e:                                   # never fail a run over a side file
            print(f"    (could not save the deployment bundle for {name}: "
                  f"{type(e).__name__}: {e})", flush=True)

    # Record this run in MLflow (if it is switched on). Wrapped in try/except so a logging problem
    # can never fail a training run - especially important during the long 129-dataset sweep.
    if mlf is not None:
        try:
            log_run_to_mlflow(mlf, model, name, row, train_ds, val_ds,
                              mlf_params=mlf_params, mlf_tags=mlf_tags,
                              logged_model_name=logged_model_name or model.name,
                              log_model_weights=log_model_weights, verbose=verbose)
        except Exception as e:                                   # keep training going no matter what
            print(f"    [mlflow] logging failed (training is unaffected): {type(e).__name__}: {e}")
    return row


def _overfit_gap(model) -> Optional[float]:
    """
    A single number that says "how much did this run overfit?".

    We take the epoch the best weights came from and subtract the training loss from the validation
    loss at that epoch. A big positive gap means the model fitted the training data far better than
    the validation data - the textbook definition of overfitting. It is None when the dataset was
    too small for a validation split (there is nothing to compare against).

    model : the trained SeaNetModel (its .history holds the per-epoch record).
    returns : val_loss - train_loss at the best epoch, or None.
    """
    history = getattr(model, "history", None)
    if not history:
        return None
    best = int(getattr(model, "best_epoch", 1)) - 1              # best_epoch is 1-based
    best = min(max(best, 0), len(history) - 1)
    point = history[best]
    if point.get("val_loss") is None:
        return None
    return round(float(point["val_loss"]) - float(point["train_loss"]), 4)


# --------------------------------------------------------------------------------------
# MLflow: build the metrics/params/tags for one training run and hand them to tracking.log_run.
# This is the ONLY place a training run talks to MLflow, so the logging logic is never duplicated.
# --------------------------------------------------------------------------------------
def log_run_to_mlflow(mlf, model, name: str, row: Dict, train_ds, val_ds, *,
                      mlf_params: Optional[Dict] = None, mlf_tags: Optional[Dict] = None,
                      logged_model_name: Optional[str] = None,
                      log_model_weights: bool = True, verbose: bool = False) -> None:
    """
    Send one finished training run to MLflow: its inputs, its results, its curves and its weights.

    The test metrics are already in `row`; here we add the TRAIN and VALIDATION accuracy + loss
    measured on the final weights, so the web page shows train / val / test side by side and an
    overfitting gap is visible at a glance.

    mlf : the mlflow handle. model : the trained SeaNetModel. name : dataset name. row : the results row.
    train_ds, val_ds : datasets to score for the train/val metrics (val_ds may be None).
    mlf_params : INPUTS to log (the resolved config); if None a short summary of this run is used.
    mlf_tags : extra labels. logged_model_name : versioned-model name. log_model_weights : save weights too.
    verbose : print a one-line confirmation.
    """
    # RESULTS: test numbers come from `row`; train/val are measured here (accuracy + loss).
    tr = M.classification_metrics(model, train_ds)
    metrics = {
        "train_acc": tr["acc"], "train_loss": tr["loss"],
        "test_acc": row["test_acc"], "test_bal_acc": row["test_bal_acc"],
        "test_auroc": row["test_auroc"], "test_loss": row["test_loss"],
        "test_aopcr": row["test_aopcr"], "train_time_s": row["train_time_s"],
        "params": row["params"], "model_size_mb": row["model_size_mb"],
        "epochs_run": row["epochs_run"], "best_epoch": row["best_epoch"],
    }
    if val_ds is not None:                                       # tiny datasets train without a val split
        va = M.classification_metrics(model, val_ds)
        metrics["val_acc"] = va["acc"]
        metrics["val_loss"] = va["loss"]
    if row["test_ndcg"] is not None:                            # WebTraffic only
        metrics["test_ndcg"] = row["test_ndcg"]
    if row["overfit_gap"] is not None:
        metrics["overfit_gap"] = row["overfit_gap"]

    # INPUTS: prefer the resolved config passed by the caller; otherwise log a short summary.
    params = dict(mlf_params) if mlf_params else {
        "model": model.name, "dataset": name, "seed": row["seed"],
        "lambda_entropy": row["lambda_entropy"], "n_train": row["n_train"],
        "n_val": row["n_val"], "n_test": row["n_test"], "n_classes": row["n_classes"],
        "series_length": row["series_length"],
    }
    tags = {"dataset": name, "model": model.name}
    if mlf_tags:
        tags.update(mlf_tags)

    tracking.log_run(
        mlf, run_name=f"{model.name}_{name}", params=params, metrics=metrics, tags=tags,
        history=getattr(model, "history", None),
        model=model.net if log_model_weights else None,        # save the network as a versioned model
        model_params={"n_model_params": row["params"], "model_size_mb": row["model_size_mb"],
                      "name": model.name},
        logged_model_name=logged_model_name, verbose=verbose,
    )
    if verbose:
        line = f"train_acc={metrics['train_acc']:.4f}"
        if "val_acc" in metrics:
            line += f" val_acc={metrics['val_acc']:.4f}"
        line += f" test_acc={metrics['test_acc']:.4f} test_loss={metrics['test_loss']:.4f}"
        print(f"    [mlflow] logged run '{model.name}_{name}'  ({line})")
