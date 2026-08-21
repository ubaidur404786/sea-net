"""
seanet/train.py - training one dataset and scoring it.

What this file is for:
    This file trains SEA-Net on a dataset and measures how good it is. The main function is
    train_one(name): it loads the data, trains the model, evaluates it, and returns one row of
    numbers (accuracy, AOPCR, NDCG, size, time). The sweep in main.py calls train_one once per
    dataset.

Input:
    A dataset name.
Output:
    A dict ("row") with the metrics and some metadata for that dataset. main.py saves this row to
    results.csv.

Related files:
    - seanet/data.py     -> load_dataset() gives us the train/test data.
    - seanet/model.py    -> make_sea_net() gives us the network.
    - millet/model/...    -> we reuse MILLETModel's evaluate() and evaluate_interpretability(), so
      the metrics are computed exactly like the MILLET paper. We subclass MILLETModel to add our
      training recipe (SeaNetModel below).
    - main.py            -> calls train_one() in the sweep, and get_device() to pick cuda/cpu.

The training recipe (same for every dataset):
    loss = label-smoothed cross-entropy + a small "focus" penalty on the attention.
    Adam optimiser, early stopping. If the dataset has enough training series we hold out 20% as
    a validation set and stop on validation loss; if it is tiny we train on everything and stop on
    training loss (a 5-series validation set would just be noise).
"""
import copy
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch import nn

from millet.data.mil_tsc_dataset import MILTSCDataset
from millet.model.millet_model import MILLETModel
from millet.util import cross_entropy_criterion, custom_tqdm

from seanet import data as D
from seanet import tracking
from seanet.model import make_sea_net, build_model_from_config, num_params, state_dict_size_mb
from seanet.preprocessing import apply_instance_representation

# --------------------------------------------------------------------------------------
# The fixed training settings (same recipe for every dataset).
# --------------------------------------------------------------------------------------
N_EPOCHS = 400             # max epochs (early stopping usually stops us sooner)
PATIENCE = 60              # stop if the monitored loss has not improved for this many epochs
MAX_BATCH = 16             # batch size = min(len(train)//10, MAX_BATCH), at least 2
LEARNING_RATE = 1.25e-3
WEIGHT_DECAY = 1.2e-4
LABEL_SMOOTHING = 0.13     # softens the labels a bit, helps avoid over-confident wrong answers
LAMBDA_ENTROPY = 0.01      # how strong the attention "focus" penalty is (0 turns it off)
MIN_TRAIN_FOR_VAL = 100    # need at least this many train series to bother holding out a val set
VAL_FRAC = 0.2             # size of the validation split (20%)
SEED = 0


# --------------------------------------------------------------------------------------
# Device picker (MILLET's own one crashes on Windows, so I wrote this)
# --------------------------------------------------------------------------------------
def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Pick the best torch device that actually exists here.

    MILLET's get_gpu_device_for_os() raises on Windows and on CPU-only machines; this never does.

    prefer_gpu : if True, use a GPU when one is available.
    returns : a torch.device, one of cuda / mps / cpu.
    """
    if prefer_gpu:
        if torch.cuda.is_available():                # NVIDIA GPU
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)   # Apple GPU
        if mps is not None and mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")                       # fall back to CPU


# --------------------------------------------------------------------------------------
# The "focus the attention" penalty (needs no labels, so it works on every dataset)
# --------------------------------------------------------------------------------------
def attention_entropy(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Average entropy of the attention over time. Low entropy means the model puts its attention on
    a few timesteps (what we want for a sharp explanation), high entropy means it is spread out.
    Adding this to the loss nudges the model to focus.

    attn : attention weights from the pooling head, shape (B, T, 1).
    eps : tiny number so log(0) does not blow up.
    returns : a single number (mean entropy over the batch).
    """
    a = attn.squeeze(-1)                                     # (B, T, 1) -> (B, T)
    entropy_per_bag = -(a * torch.log(a + eps)).sum(dim=1)   # entropy of each series' attention
    return entropy_per_bag.mean()                            # average over the batch


# --------------------------------------------------------------------------------------
# Loss function with label smoothing
# --------------------------------------------------------------------------------------
def make_label_smoothing_criterion(smoothing: float = LABEL_SMOOTHING) -> Callable:
    """
    Build a cross-entropy loss with label smoothing.

    smoothing : label smoothing amount (0 = plain cross-entropy).
    returns : a function criterion(predictions, targets) -> loss.
    """
    loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def criterion(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return loss_fn(predictions, targets.long())   # targets must be int class indices

    return criterion


# --------------------------------------------------------------------------------------
# Train / validation split (keeps the class balance)
# --------------------------------------------------------------------------------------
def _subset(dataset: MILTSCDataset, idx: List[int]) -> MILTSCDataset:
    """
    Make a small "view" of a dataset that only contains the chosen rows.

    It shallow-copies the dataset object and swaps in the filtered series/labels, so all the
    original methods (dataloader, indexing, normalisation) keep working. We reuse the class
    instead of writing a new one.

    dataset : the full dataset.
    idx : the row indices to keep.
    returns : a dataset containing only those rows.
    """
    idx = [int(i) for i in idx]
    sub = copy.copy(dataset)
    sub.ts_collection = [dataset.ts_collection[i] for i in idx]   # keep only these series
    sub.targets = dataset.targets[idx]                           # and their labels
    if hasattr(dataset, "_metadata"):                            # WebTraffic also has per-series metadata
        sub._metadata = [dataset._metadata[i] for i in idx]
    sub.n_clz = dataset.n_clz                                    # keep the original class count
    return sub


def split_train_val(
    dataset: MILTSCDataset, val_frac: float = VAL_FRAC, seed: int = SEED
) -> Tuple[MILTSCDataset, MILTSCDataset]:
    """
    Split a dataset into (train, validation), keeping each class's proportion the same in both.

    dataset : the dataset to split.
    val_frac : fraction to put in validation (e.g. 0.2 = 20%).
    seed : random seed, so the split is the same every run.
    returns : (train_subset, val_subset).
    """
    g = torch.Generator().manual_seed(seed)
    targets = torch.as_tensor(dataset.targets)
    train_idx, val_idx = [], []
    for c in torch.unique(targets):                              # do the split class by class (stratified)
        c_idx = (targets == c).nonzero(as_tuple=True)[0]         # all rows of this class
        c_idx = c_idx[torch.randperm(len(c_idx), generator=g)]   # shuffle them
        n_val = int(len(c_idx) * val_frac)
        if n_val == 0 and len(c_idx) > 1 and val_frac > 0:       # rare class -> still give val 1 row
            n_val = 1
        val_idx.append(c_idx[:n_val])
        train_idx.append(c_idx[n_val:])
    return _subset(dataset, torch.cat(train_idx)), _subset(dataset, torch.cat(val_idx))


# --------------------------------------------------------------------------------------
# The SEA-Net training model (MILLETModel + our loss + our early stopping)
# --------------------------------------------------------------------------------------
class SeaNetModel(MILLETModel):
    """
    A MILLETModel with my training recipe.

    Two things are added on top of MILLETModel:
      - the training loss is the class loss plus a small attention-entropy penalty, and
      - fit() early-stops on validation loss (or training loss if there is no validation set).
    Everything else (evaluate, evaluate_interpretability, forward) is inherited from MILLETModel.
    """

    def __init__(self, name: str, device: torch.device, n_classes: int, net: nn.Module,
                 lambda_entropy: float = LAMBDA_ENTROPY):
        """
        name : a label for the model (used in the results row).
        device : where to run (cuda/cpu).
        n_classes : number of classes.
        net : the network (from make_sea_net).
        lambda_entropy : strength of the attention penalty.
        """
        super().__init__(name, device, n_classes, net)
        self.lambda_entropy = lambda_entropy

    def dataset_loss(self, dataset: MILTSCDataset, criterion: Callable) -> float:
        """
        Compute just the classification loss over a whole dataset (no accuracy/AUROC). This is the
        number we watch for early stopping.

        dataset : the dataset to measure on (train or validation).
        criterion : the loss function.
        returns : the average loss as a plain float.
        """
        logits, targets = [], []
        with torch.no_grad():                                    # no gradients, this is just measuring
            for batch in dataset.create_dataloader(batch_size=16):
                logits.append(self(batch["bags"])["bag_logits"].cpu())
                targets.append(batch["targets"])
        return criterion(torch.cat(logits), torch.cat(targets)).item()

    def fit(  # type: ignore[override]
        self,
        train_dataset: MILTSCDataset,
        val_dataset: Optional[MILTSCDataset] = None,
        n_epochs: int = N_EPOCHS,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        smoothing: float = LABEL_SMOOTHING,
        patience: int = PATIENCE,
        batch_size: int = MAX_BATCH,
        verbose: bool = False,
        epoch_callback: Optional[Callable] = None,
    ) -> None:
        """
        Train the network with early stopping, and keep the best weights.

        train_dataset : data to train on.
        val_dataset : data to watch for early stopping; if None we watch the training loss instead.
        n_epochs : maximum number of epochs.
        learning_rate, weight_decay : Adam optimiser settings.
        smoothing : label smoothing amount.
        patience : stop after this many epochs with no improvement.
        batch_size : training batch size.
        verbose : if True, show a live progress bar.
        epoch_callback : optional function called each epoch as callback(epoch, monitor_loss). Optuna
            uses it to report progress and prune bad trials (it raises to stop training early).
        returns : nothing; the best weights are stored in self.net.
        """
        loader = train_dataset.create_dataloader(shuffle=True, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = make_label_smoothing_criterion(smoothing)
        monitor_ds = val_dataset if val_dataset is not None else train_dataset   # what early stopping watches

        best_net = None                                          # best weights seen so far
        best_loss = np.inf
        epochs_no_improve = 0
        self.history = []                                        # per-epoch monitored loss (for MLflow curves)
        # progress bar over epochs; leave=False so it vanishes when the dataset is done
        epoch_bar = custom_tqdm(range(n_epochs), desc="    training", disable=not verbose, leave=False)
        for epoch in epoch_bar:
            self.net.train()                                     # training mode (dropout/batchnorm on)
            for batch in loader:                                 # one pass over the training data
                targets = batch["targets"].to(self.device)
                optimizer.zero_grad()
                out = self(batch["bags"])
                loss = criterion(out["bag_logits"], targets)     # classification loss
                # plus the attention focus penalty (only for pooling heads that return "attn";
                # gap/instance pooling do not, so we just skip it for them)
                if self.lambda_entropy > 0 and "attn" in out:
                    loss = loss + self.lambda_entropy * attention_entropy(out["attn"])
                loss.backward()                                  # backprop
                optimizer.step()                                 # update the weights

            self.net.eval()                                      # eval mode to measure the loss
            monitor_loss = self.dataset_loss(monitor_ds, criterion)
            if monitor_loss != monitor_loss:                     # NaN check (NaN != NaN is True)
                if verbose:
                    print("    training: NaN loss -> stopping (keeping best weights so far)")
                break
            self.history.append(float(monitor_loss))             # record the curve (one point per epoch)
            if epoch_callback is not None:                       # let Optuna report/prune this trial
                epoch_callback(epoch, float(monitor_loss))
            if monitor_loss < best_loss:                         # improved -> remember these weights
                best_loss = monitor_loss
                best_net = copy.deepcopy(self.net)
                epochs_no_improve = 0
            else:                                                # no improvement -> count towards patience
                epochs_no_improve += 1
            if verbose:
                epoch_bar.set_postfix_str(f"loss={monitor_loss:.4f} best={best_loss:.4f} patience={epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:                    # waited long enough -> stop
                break
        if best_net is not None:                                 # load the best weights back
            self.net = best_net


def make_model(n_clz: int, device: torch.device, lambda_entropy: float = LAMBDA_ENTROPY,
               model_cfg=None, n_in: int = 1) -> SeaNetModel:
    """
    Build a trainable model for a dataset.

    If no model config is given we build the default SEA-Net (exactly as before, so nothing
    regresses). If a model config is given, we build whatever it describes (SEA-Net, the MILLET
    baseline, ...) through the config-driven factory in seanet/model.py.

    n_clz : number of classes.
    device : where to run.
    lambda_entropy : strength of the attention penalty.
    model_cfg : optional model config (the "model_config" section from a loaded config); None = SEA-Net.
    n_in : number of input channels (1 for single-point instances; the window size when windowing).
    returns : a SeaNetModel ready to .fit().
    """
    if model_cfg is None:
        net = make_sea_net(n_clz, n_in=n_in)                 # default path (only n_in can change)
        name = "SEA-Net"
    else:
        net = build_model_from_config(model_cfg, n_clz, n_in=n_in)   # config-driven path
        name = getattr(model_cfg, "name", "model")
    return SeaNetModel(name, device, n_clz, net, lambda_entropy=lambda_entropy)


def safe_evaluate(model: MILLETModel, dataset: MILTSCDataset) -> Dict:
    """
    Run MILLET's evaluate(), but do not let a failing AUROC kill the dataset.

    MILLET's evaluate() computes roc_auc_score, which raises when a test split does not contain
    every class (it happens on a few UCR datasets). Instead of losing that whole dataset over one
    secondary metric, we fall back to computing accuracy / balanced accuracy / loss and set AUROC
    to NaN.

    model : the trained model.
    dataset : the test set to score.
    returns : a dict with acc, bal_acc, auroc, loss.
    """
    try:
        return model.evaluate(dataset)                           # normal path (all metrics)
    except Exception:                                            # AUROC (or similar) failed -> fallback
        logits, targets = [], []
        with torch.no_grad():
            for batch in dataset.create_dataloader(batch_size=16):
                logits.append(model(batch["bags"])["bag_logits"].cpu())
                targets.append(batch["targets"])
        logits = torch.cat(logits)
        targets = torch.cat(targets)
        preds = torch.argmax(logits, dim=1)                      # predicted class = highest logit
        return {
            "acc": accuracy_score(targets.long(), preds),
            "bal_acc": balanced_accuracy_score(targets.long(), preds),
            "auroc": float("nan"),                               # could not compute it
            "loss": cross_entropy_criterion(logits, targets).item(),
        }


# --------------------------------------------------------------------------------------
# Load + build + fit a model (the shared first half). train_one scores it; the interpret
# command keeps the live model to draw explanations from. Written once so there is no duplication.
# --------------------------------------------------------------------------------------
def fit_model(
    name: str,
    device: Optional[torch.device] = None,
    seed: int = SEED,
    n_epochs: int = N_EPOCHS,
    patience: int = PATIENCE,
    lambda_entropy: float = LAMBDA_ENTROPY,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING,
    max_batch: int = MAX_BATCH,
    min_train_for_val: int = MIN_TRAIN_FOR_VAL,
    val_frac: float = VAL_FRAC,
    instance_type: str = "single_point",
    window_size: int = 5,
    window_stride: int = 5,
    model_cfg=None,
    verbose: bool = False,
    epoch_callback: Optional[Callable] = None,
) -> Tuple[SeaNetModel, MILTSCDataset, Optional[MILTSCDataset], MILTSCDataset, float]:
    """
    Load a dataset, apply the instance representation, build the model, and train it.

    This is the first half of train_one, pulled out so it can be reused (train_one scores the model;
    the interpret command draws figures from it). The arguments and defaults are identical to
    train_one's, so behaviour is unchanged.

    name : dataset name.
    device : where to run; if None, get_device() picks one.
    (all other arguments) : see train_one - same meaning and defaults.
    returns : (model, train_ds, val_ds, test_ds, train_time_s).
    """
    if device is None:
        device = get_device()
    torch.manual_seed(seed)                                      # make the run reproducible

    # --- stage 1: load the data ---
    if verbose:
        print("    stage 1/3: loading data ...", flush=True)
    train_full = D.load_dataset(name, "train")
    test_ds = D.load_dataset(name, "test")

    # re-shape into the chosen instance representation (single_point = unchanged; sliding_window =
    # cut each series into windows). This is the only preprocessing step; normalisation lives inside.
    train_full = apply_instance_representation(train_full, instance_type, window_size, window_stride)
    test_ds = apply_instance_representation(test_ds, instance_type, window_size, window_stride)

    # number of input channels the model needs: 1 for single timesteps, or the window size when
    # windowing (each bag is (n_instances, d_instance), so d_instance is the channel count).
    n_in = int(train_full.get_bag(0).shape[1])

    # decide whether to hold out a validation set (only if there is enough training data)
    if len(train_full) >= min_train_for_val:
        train_ds, val_ds = split_train_val(train_full, val_frac=val_frac, seed=seed)
    else:
        train_ds, val_ds = train_full, None

    batch_size = max(min(len(train_ds) // 10, max_batch), 2)     # small batch for small datasets
# Actual model create here 
    model = make_model(train_full.n_clz, device, lambda_entropy=lambda_entropy, model_cfg=model_cfg, n_in=n_in)

    # --- stage 2: train ---
    if verbose:
        val_note = f"val={len(val_ds)}" if val_ds is not None else "no val (train-loss early stop)"
        print(f"    stage 2/3: training on {len(train_ds)} series ({val_note}, "
              f"C={train_full.n_clz}, T={len(train_full.get_bag(0))}, batch={batch_size}) ...", flush=True)
    t0 = time.perf_counter()
    model.fit(train_ds, val_ds, n_epochs=n_epochs, learning_rate=learning_rate,
              weight_decay=weight_decay, smoothing=label_smoothing, patience=patience,
              batch_size=batch_size, verbose=verbose, epoch_callback=epoch_callback)
    train_time_s = time.perf_counter() - t0
    return model, train_ds, val_ds, test_ds, train_time_s


# --------------------------------------------------------------------------------------
# Train + evaluate ONE dataset -> one results row (the sweep calls this per dataset)
# --------------------------------------------------------------------------------------
def train_one(
    name: str,
    device: Optional[torch.device] = None,
    seed: int = SEED,
    n_epochs: int = N_EPOCHS,
    patience: int = PATIENCE,
    lambda_entropy: float = LAMBDA_ENTROPY,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING,
    max_batch: int = MAX_BATCH,
    min_train_for_val: int = MIN_TRAIN_FOR_VAL,
    val_frac: float = VAL_FRAC,
    instance_type: str = "single_point",
    window_size: int = 5,
    window_stride: int = 5,
    model_cfg=None,
    verbose: bool = False,
    mlf=None,
    mlf_params: Optional[Dict] = None,
    mlf_tags: Optional[Dict] = None,
    logged_model_name: Optional[str] = None,
    log_model_weights: bool = True,
    pred_dir: Optional[str] = None,
    curve_dir: Optional[str] = None,
) -> Dict:
    """
    Train a model on one dataset and score it on the test set. This is the single path used for
    WebTraffic and every UCR dataset.

    Every hyperparameter is an argument with a default equal to the original hardcoded constant,
    so calling train_one(name) behaves exactly as before. The config-driven path
    (train_one_from_config) just fills these arguments from the YAML files instead.

    Validation policy: if the training set has at least min_train_for_val series we hold out a
    val_frac slice for validation and early-stop on it; otherwise we train on all of it and
    early-stop on the training loss.

    name : dataset name.
    device : where to run; if None, get_device() picks one.
    seed : random seed for reproducibility.
    n_epochs, patience : training length / early stopping (small values are used for --smoke).
    lambda_entropy : strength of the attention penalty.
    learning_rate, weight_decay : Adam optimiser settings.
    label_smoothing : label smoothing amount.
    max_batch : upper bound on the batch size (actual batch = clamp(n_train // 10, 2, max_batch)).
    min_train_for_val : need at least this many train series to hold out a validation set.
    val_frac : size of the validation split.
    instance_type : "single_point" (one instance = one timestep, the default) or "sliding_window"
                    (one instance = a window of the series).
    window_size, window_stride : the window length and step (only used for sliding_window).
    model_cfg : optional model config; None = the default SEA-Net (unchanged behaviour).
    verbose : if True, print the 3 stages and show the training bar.
    mlf : the mlflow handle from tracking.start_experiment (None -> no MLflow logging).
    mlf_params : INPUTS to log to MLflow; if None, a summary of this run's recipe is built and logged.
    mlf_tags : extra MLflow labels (e.g. {"command": "single"}).
    logged_model_name : name to give the versioned model in the MLflow "Models" tab.
    log_model_weights : also save the trained network's weights as an artifact.
    returns : one flat results-row dict (metrics + footprint + metadata).
    """
    if device is None:
        device = get_device()

    # stages 1-2 (load -> instance representation -> build -> train) are shared with the interpret
    # command, so they live in fit_model. train_one just scores the fitted model below.
    model, train_ds, val_ds, test_ds, train_time_s = fit_model(
        name, device=device, seed=seed, n_epochs=n_epochs, patience=patience,
        lambda_entropy=lambda_entropy, learning_rate=learning_rate, weight_decay=weight_decay,
        label_smoothing=label_smoothing, max_batch=max_batch, min_train_for_val=min_train_for_val,
        val_frac=val_frac, instance_type=instance_type, window_size=window_size,
        window_stride=window_stride, model_cfg=model_cfg, verbose=verbose,
    )
    # if MLflow is on but the caller did not spell out the inputs, log this run's actual recipe
    # (learning rate, weight decay, ...) so the web page shows what produced each result.
    if mlf is not None and mlf_params is None:
        mlf_params = {
            "model": model.name, "dataset": name, "seed": seed,
            "n_epochs": n_epochs, "patience": patience, "max_batch": max_batch,
            "learning_rate": learning_rate, "weight_decay": weight_decay,
            "label_smoothing": label_smoothing, "lambda_entropy": lambda_entropy,
            "min_train_for_val": min_train_for_val, "val_frac": val_frac,
            "instance_type": instance_type, "window_size": window_size, "window_stride": window_stride,
        }
    row = score_model(model, name, train_ds, val_ds, test_ds, device, seed, lambda_entropy,
                      train_time_s, verbose=verbose, mlf=mlf, mlf_params=mlf_params,
                      mlf_tags=mlf_tags, logged_model_name=logged_model_name,
                      log_model_weights=log_model_weights, pred_dir=pred_dir, curve_dir=curve_dir)
    del model                                                   # free the model
    if device.type == "cuda":                                  # free GPU memory before the next dataset
        torch.cuda.empty_cache()
    return row


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


def save_curve(model, name: str, seed: int, curve_dir: str, has_val: bool) -> Optional[str]:
    """
    Save this run's per-epoch loss curve to a small CSV, so we can plot it later.

    Why we need this: results.csv only has the FINAL test numbers. It cannot say when early stopping
    fired or whether the loss was still going down when we stopped. fit() already records that curve
    in model.history (one loss per epoch), but until now it only went to MLflow - which stays on the
    machine that trained the model. A tiny CSV next to results.csv travels with the results instead.

    What the "loss" column is: the loss early stopping was watching. That is the VALIDATION loss when
    the dataset was big enough for a validation split (>= min_train_for_val series), and the TRAINING
    loss otherwise - the same rule fit() uses. The monitored column says which one it was, so a plot
    can never mislabel the curve.

    model : the trained SeaNetModel (its .history holds the curve).
    name : dataset name.  seed : the training seed (part of the filename, so seeds do not clash).
    curve_dir : the folder to write into (created if missing).
    has_val : True when this run had a validation split (so the curve is the validation loss).
    returns : the path written, or None when there is no curve to write.
    """
    history = getattr(model, "history", None)
    if not history:                                              # no epochs ran (or fit was skipped)
        return None
    os.makedirs(curve_dir, exist_ok=True)
    monitored = "val_loss" if has_val else "train_loss"
    frame = pd.DataFrame({
        "epoch": range(1, len(history) + 1),
        "loss": history,
        "monitored": monitored,
    })
    # the best epoch is where the monitored loss was lowest - that is the point fit() kept the
    # weights from, so it is also the epoch to mark on a training-curve figure.
    frame["best_epoch"] = int(np.argmin(history)) + 1
    path = os.path.join(curve_dir, f"{name}__seed{int(seed)}.csv")
    frame.to_csv(path, index=False)
    return path


def score_model(model, name: str, train_ds, val_ds, test_ds, device: torch.device, seed: int,
                lambda_entropy: float, train_time_s: float, verbose: bool = False,
                mlf=None, mlf_params: Optional[Dict] = None, mlf_tags: Optional[Dict] = None,
                logged_model_name: Optional[str] = None, log_model_weights: bool = True,
                pred_dir: Optional[str] = None, curve_dir: Optional[str] = None) -> Dict:
    """
    Score a trained model on the test set, pack the results into one flat row, and (optionally) record
    the whole run in MLflow so every model can be compared later.

    This is stage 3 of train_one, pulled out so the "run" command can reuse it with the live model.
    It does NOT free the model; the caller decides when to do that.

    model : the trained SeaNetModel.
    name : dataset name.
    train_ds, val_ds, test_ds : the datasets (for the size fields + the train/val metrics; val_ds may be None).
    device, seed, lambda_entropy, train_time_s : metadata for the row.
    verbose : if True, print the scoring stage line.
    mlf : the mlflow handle from tracking.start_experiment (None -> no MLflow logging happens).
    mlf_params : INPUTS to log to MLflow (usually the resolved config); if None a summary is logged.
    mlf_tags : extra searchable labels for the MLflow run (e.g. {"command": "run"}).
    logged_model_name : name to give the versioned model in the MLflow "Models" tab (default: the model's name).
    log_model_weights : also save the trained network's weights as an artifact (else log only params+metrics).
    returns : one flat results-row dict (same shape written to results.csv).
    """
    # safe_evaluate gives accuracy/AUROC/loss (AUROC may be NaN). evaluate_interpretability gives
    # AOPCR (always) and NDCG (WebTraffic only). AOPCR is the slow part because it re-runs the model.
    if verbose:
        print(f"    stage 3/3: scoring on {len(test_ds)} test series (accuracy + AOPCR"
              f"{' + NDCG' if name == D.WEB_TRAFFIC else ''}) ...", flush=True)
    cls = safe_evaluate(model, test_ds)
    aopcr, ndcg = model.evaluate_interpretability(test_ds)
    if pred_dir:                                                 # keep the per-series predictions so
        try:                                                     # models can be ensembled later
            save_predictions(model, name, test_ds, seed, pred_dir)
        except Exception as e:                                   # never fail a run over a side file
            print(f"    (could not save predictions for {name}: {type(e).__name__}: {e})", flush=True)
    if curve_dir:                                                # keep the per-epoch loss curve so the
        try:                                                     # training-behaviour figure can be drawn
            save_curve(model, name, seed, curve_dir, has_val=val_ds is not None)
        except Exception as e:                                   # never fail a run over a side file
            print(f"    (could not save curve for {name}: {type(e).__name__}: {e})", flush=True)
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
        "test_aopcr": round(float(aopcr), 4),
        "test_ndcg": round(float(ndcg), 4) if ndcg is not None else None,   # None for UCR
        "train_time_s": round(train_time_s, 2),
    }

    # Record this run in MLflow (if it is switched on). Wrapped in try/except so a logging problem
    # can never fail a training run - especially important during the long sweep in "train".
    if mlf is not None:
        try:
            _log_run_to_mlflow(mlf, model, name, row, train_ds, val_ds,
                               mlf_params=mlf_params, mlf_tags=mlf_tags,
                               logged_model_name=logged_model_name or model.name,
                               log_model_weights=log_model_weights, verbose=verbose)
        except Exception as e:                                   # keep training going no matter what
            print(f"    [mlflow] logging failed (training is unaffected): {type(e).__name__}: {e}")
    return row


def _log_run_to_mlflow(mlf, model, name: str, row: Dict, train_ds, val_ds, *,
                       mlf_params: Optional[Dict] = None, mlf_tags: Optional[Dict] = None,
                       logged_model_name: Optional[str] = None,
                       log_model_weights: bool = True, verbose: bool = False) -> None:
    """
    Build the metrics/params/tags for one training run and hand them to tracking.log_run.

    The test metrics are already in `row`; here we add the TRAIN and VALIDATION accuracy + loss (one
    extra evaluation pass each, only done when MLflow is on) so the web page shows train/val/test side
    by side - exactly the "show performance by train, val, test accuracies with loss" we want.

    mlf : the mlflow handle. model : the trained SeaNetModel. name : dataset name. row : the results row.
    train_ds, val_ds : datasets to score for the train/val metrics (val_ds may be None).
    mlf_params : INPUTS to log (the resolved config); if None a short summary of this run is used.
    mlf_tags : extra labels. logged_model_name : versioned-model name. log_model_weights : save weights too.
    verbose : print a one-line confirmation.
    """
    # RESULTS: test numbers come from `row`; train/val are measured here (accuracy + loss).
    tr = safe_evaluate(model, train_ds)
    metrics = {
        "train_acc": tr["acc"], "train_loss": tr["loss"],
        "test_acc": row["test_acc"], "test_bal_acc": row["test_bal_acc"],
        "test_auroc": row["test_auroc"], "test_loss": row["test_loss"],
        "test_aopcr": row["test_aopcr"], "train_time_s": row["train_time_s"],
    }
    if val_ds is not None:                                       # tiny datasets train without a val split
        va = safe_evaluate(model, val_ds)
        metrics["val_acc"] = va["acc"]
        metrics["val_loss"] = va["loss"]
    if row["test_ndcg"] is not None:                            # WebTraffic only
        metrics["test_ndcg"] = row["test_ndcg"]

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


# --------------------------------------------------------------------------------------
# Config-driven entry points (used by "python main.py run" and "python main.py interpret")
# --------------------------------------------------------------------------------------
def _train_kwargs_from_config(cfg, smoke: bool = False) -> Dict:
    """
    Turn a loaded config into the keyword arguments that train_one / fit_model expect.

    Both config-driven wrappers below read the config the same way, so that reading lives here once
    (no duplication). It pulls the training block from the model file and the preprocessing block
    from main.yaml.

    cfg : a loaded config (from load_config).
    smoke : if True, force a quick 3-epoch run (for testing, not a real result).
    returns : a dict of keyword arguments for train_one / fit_model.
    """
    t = cfg.model_config.training                               # the "training" block of the model file
    n_epochs, patience = (3, 3) if smoke else (t.n_epochs, t.patience)
    # the "preprocessing" block of main.yaml (optional -> default to single_point)
    p = getattr(cfg, "preprocessing", None)
    instance_type = getattr(p, "instance_type", "single_point") if p is not None else "single_point"
    window_size = getattr(p, "window_size", 5) if p is not None else 5
    window_stride = getattr(p, "window_stride", 5) if p is not None else 5
    return dict(
        seed=cfg.seed,
        n_epochs=n_epochs,
        patience=patience,
        lambda_entropy=t.lambda_entropy,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        label_smoothing=t.label_smoothing,
        max_batch=t.max_batch,
        min_train_for_val=t.min_train_for_val,
        val_frac=t.val_frac,
        instance_type=instance_type,
        window_size=window_size,
        window_stride=window_stride,
        model_cfg=cfg.model_config,
    )


def train_one_from_config(name: str, cfg, device: Optional[torch.device] = None,
                          smoke: bool = False, verbose: bool = False, **kwargs) -> Dict:
    """
    Train + evaluate one dataset using values read from a config object (see seanet/config.py).

    A thin wrapper that fills train_one's arguments from the config, so there is still only one
    training code path.

    name : dataset name to run.
    cfg : a loaded config (from load_config).
    device : where to run; if None, get_device() picks one.
    smoke : if True, force a quick 3-epoch run (for testing, not a real result).
    verbose : if True, print the 3 stages and the training bar.
    kwargs : passed straight to train_one - the MLflow arguments (mlf, mlf_params, mlf_tags,
             logged_model_name, log_model_weights), so a config-driven sweep records its runs
             exactly like the older hardcoded path did.
    returns : the same flat results-row dict as train_one.
    """
    return train_one(name, device=device, verbose=verbose,
                     **_train_kwargs_from_config(cfg, smoke), **kwargs)


def fit_model_from_config(name: str, cfg, device: Optional[torch.device] = None,
                          smoke: bool = False, verbose: bool = False):
    """
    Load + build + train a model using config values, and return the LIVE model (not scored).

    Used by the interpret command, which needs the fitted model to draw explanations from. It shares
    the same config reading and the same fit_model path as training, so nothing is duplicated.

    name : dataset name to run.
    cfg : a loaded config (from load_config).
    device : where to run; if None, get_device() picks one.
    smoke : if True, force a quick 3-epoch run.
    verbose : if True, print the stages and the training bar.
    returns : (model, train_ds, val_ds, test_ds, train_time_s) - same as fit_model.
    """
    return fit_model(name, device=device, verbose=verbose, **_train_kwargs_from_config(cfg, smoke))
