"""
seanet/training.py - the training component: fit ONE model on ONE dataset.

What this file is for:
    This is the only training loop in the project. Every command that trains (single / train /
    webtraffic / run / interpret) and every Optuna trial goes through it, so there is exactly one
    place where "how we train" is defined and nothing can drift out of sync.

    It STOPS at the trained model. Measuring it is seanet/evaluation.py's job.

The flow inside train_one(), in order:
    1. seed everything                              seanet/utils.set_seed
    2. load the dataset (train + test)              seanet/data.load_dataset
    3. hold out a validation split if big enough    seanet/preprocessing.prepare_splits
    4. build the model from the config              seanet/models.build_model_from_config
    5. fit it, recording a per-epoch history        SeaNetModel.fit (below)
    6. score it on the test set                     seanet/evaluation.score_model

The training recipe (identical for every dataset, set in the model's YAML):
    loss = label-smoothed cross-entropy + a small "focus" penalty on the attention.
    Adam optimiser, early stopping on validation loss - or on TRAINING loss when the dataset is too
    small to spare a validation set (a 5-series validation set is noise, not a signal).

The per-epoch history (new in seanetv7):
    fit() records, for every epoch:
        train_loss, train_acc   - accumulated from the batches as they are trained on, so they
                                  cost NOTHING extra (the loop already has those logits),
        val_loss,   val_acc     - one pass over the validation set (which early stopping needed
                                  anyway), or None when there is no validation split.
    save_history() writes that to a CSV and draws the loss and accuracy curves. This is what lets
    you see WHEN early stopping fired and whether the model was overfitting.

Related files:
    - seanet/data.py          -> load_dataset()
    - seanet/preprocessing.py -> prepare_splits()
    - seanet/models/          -> the encoder + pooling head this file trains
    - seanet/evaluation.py    -> scores the trained model into one results row
    - seanet/utils.py         -> get_device() / set_seed()
"""
import copy
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from millet.data.mil_tsc_dataset import MILTSCDataset
from millet.model.millet_model import MILLETModel
from millet.util import custom_tqdm

from seanet import data as D
from seanet import metrics as M
from seanet.models import make_sea_net, build_model_from_config
from seanet.preprocessing import prepare_splits
from seanet.utils import get_device, set_seed

# --------------------------------------------------------------------------------------
# Default training settings.
#
# These are FALLBACKS only. Every real run reads its numbers from the model's YAML file
# (configs/models/.../<model>.yaml, the "training:" block), so nothing here is hard-coded into an
# experiment. They exist so that make_model()/fit_model() can still be called directly from a
# notebook or a test without writing a config first.
# --------------------------------------------------------------------------------------
N_EPOCHS = 400             # max epochs (early stopping usually stops us sooner)
PATIENCE = 60              # stop if the monitored loss has not improved for this many epochs
MAX_BATCH = 16             # batch size = clamp(len(train)//10, 2, MAX_BATCH)
LEARNING_RATE = 1.25e-3
WEIGHT_DECAY = 1.2e-4
LABEL_SMOOTHING = 0.13     # softens the labels a bit, helps avoid over-confident wrong answers
LAMBDA_ENTROPY = 0.01      # how strong the attention "focus" penalty is (0 turns it off)
MIN_TRAIN_FOR_VAL = 100    # need at least this many train series to bother holding out a val set
VAL_FRAC = 0.2             # size of the validation split (20%)
SEED = 0


# --------------------------------------------------------------------------------------
# The loss: label-smoothed cross-entropy + the "focus the attention" penalty
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
# The trainable model: MILLETModel + our loss + our early stopping + our history
# --------------------------------------------------------------------------------------
class SeaNetModel(MILLETModel):
    """
    A MILLETModel with our training recipe.

    Three things are added on top of MILLETModel:
      - the training loss is the class loss plus a small attention-entropy penalty,
      - fit() early-stops on validation loss (or training loss if there is no validation set),
      - fit() records a per-epoch history (train/val loss and accuracy) in self.history.
    Everything else (evaluate, evaluate_interpretability, forward) is inherited from MILLETModel,
    so our metrics are computed exactly the way the MILLET paper computes them.
    """

    def __init__(self, name: str, device: torch.device, n_classes: int, net: nn.Module,
                 lambda_entropy: float = LAMBDA_ENTROPY):
        """
        name : a label for the model (used in the results row).
        device : where to run (cuda/cpu).
        n_classes : number of classes.
        net : the network (encoder + pooling head).
        lambda_entropy : strength of the attention penalty.
        """
        super().__init__(name, device, n_classes, net)
        self.lambda_entropy = lambda_entropy
        self.history: List[Dict] = []      # one dict per epoch, filled by fit()
        self.best_epoch: int = 0           # the epoch the kept weights came from (1-based)

    def evaluate_loss_and_acc(self, dataset: MILTSCDataset, criterion: Callable) -> Tuple[float, float]:
        """
        One pass over a dataset: its average classification loss AND its accuracy.

        This is what early stopping watches. We compute both in the same pass because the logits
        are already there - getting the accuracy costs nothing extra.

        dataset : the dataset to measure on (train or validation).
        criterion : the loss function.
        returns : (loss, accuracy).
        """
        logits, targets = [], []
        with torch.no_grad():                                    # no gradients, this is just measuring
            for batch in dataset.create_dataloader(batch_size=16):
                logits.append(self(batch["bags"])["bag_logits"].cpu())
                targets.append(batch["targets"])
        logits = torch.cat(logits)
        targets = torch.cat(targets)
        return float(criterion(logits, targets).item()), M.batch_accuracy(logits, targets)

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
        Train the network with early stopping, keep the best weights, and record the history.

        What happens in ONE epoch:
            for every batch:  forward -> loss -> backward -> update the weights
                              (and, for free, accumulate the training loss + accuracy)
            then:             one pass over the validation set -> val loss + val accuracy
                              (or over the training set, when there is no validation split)
            then:             is this the best monitored loss so far? keep these weights.
                              no improvement for `patience` epochs? stop.

        train_dataset : data to train on.
        val_dataset : data to watch for early stopping; if None we watch the training loss instead.
        n_epochs : maximum number of epochs.
        learning_rate, weight_decay : Adam optimiser settings.
        smoothing : label smoothing amount.
        patience : stop after this many epochs with no improvement.
        batch_size : training batch size.
        verbose : if True, show a live progress bar.
        epoch_callback : optional function called each epoch as callback(epoch, monitor_loss).
            Optuna uses it to report progress and prune bad trials (it raises to stop training).
        returns : nothing; the best weights are stored in self.net and the curve in self.history.
        """
        loader = train_dataset.create_dataloader(shuffle=True, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = make_label_smoothing_criterion(smoothing)

        best_net = None                                          # best weights seen so far
        best_loss = np.inf
        epochs_no_improve = 0
        self.history = []
        self.best_epoch = 0
        # progress bar over epochs; leave=False so it vanishes when the dataset is done
        epoch_bar = custom_tqdm(range(n_epochs), desc="    training", disable=not verbose, leave=False)
        for epoch in epoch_bar:
            # ---- the training pass ----------------------------------------------------------
            self.net.train()                                     # training mode (dropout/batchnorm on)
            run_loss, run_acc, n_batches = 0.0, 0.0, 0
            for batch in loader:                                 # one pass over the training data
                targets = batch["targets"].to(self.device)
                optimizer.zero_grad()
                out = self(batch["bags"])
                loss = criterion(out["bag_logits"], targets)     # classification loss
                # plus the attention focus penalty (only for pooling heads that return "attn";
                # gap/instance pooling do not, so we just skip it for them)
                total = loss
                if self.lambda_entropy > 0 and "attn" in out:
                    total = loss + self.lambda_entropy * attention_entropy(out["attn"])
                total.backward()                                 # backprop
                optimizer.step()                                 # update the weights
                # record the training numbers from what we already computed - no extra pass
                run_loss += float(loss.item())
                run_acc += M.batch_accuracy(out["bag_logits"].detach().cpu(), targets.detach().cpu())
                n_batches += 1
            train_loss = run_loss / max(n_batches, 1)
            train_acc = run_acc / max(n_batches, 1)

            # ---- the validation pass --------------------------------------------------------
            self.net.eval()                                      # eval mode to measure honestly
            if val_dataset is not None:
                val_loss, val_acc = self.evaluate_loss_and_acc(val_dataset, criterion)
                monitor_loss = val_loss                          # early stopping watches validation
            else:                                                # dataset too small for a val split
                val_loss, val_acc = None, None
                monitor_loss, _ = self.evaluate_loss_and_acc(train_dataset, criterion)

            if monitor_loss != monitor_loss:                     # NaN check (NaN != NaN is True)
                if verbose:
                    print("    training: NaN loss -> stopping (keeping best weights so far)")
                break
            self.history.append({                                # one row per epoch
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": None if val_loss is None else round(val_loss, 6),
                "val_acc": None if val_acc is None else round(val_acc, 6),
                "monitored": "val_loss" if val_dataset is not None else "train_loss",
            })
            if epoch_callback is not None:                       # let Optuna report/prune this trial
                epoch_callback(epoch, float(monitor_loss))

            # ---- early stopping bookkeeping -------------------------------------------------
            if monitor_loss < best_loss:                         # improved -> remember these weights
                best_loss = monitor_loss
                best_net = copy.deepcopy(self.net)
                self.best_epoch = epoch + 1
                epochs_no_improve = 0
            else:                                                # no improvement -> count to patience
                epochs_no_improve += 1
            if verbose:
                shown = f"loss={monitor_loss:.4f} best={best_loss:.4f}"
                if val_acc is not None:
                    shown += f" val_acc={val_acc:.3f}"
                epoch_bar.set_postfix_str(f"{shown} patience={epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:                    # waited long enough -> stop
                break
        if best_net is not None:                                 # load the best weights back
            self.net = best_net


def make_model(n_clz: int, device: torch.device, lambda_entropy: float = LAMBDA_ENTROPY,
               model_cfg=None, n_in: int = 1) -> SeaNetModel:
    """
    Build a trainable model for a dataset.

    If no model config is given we build the default SEA-Net. If a model config is given, we build
    whatever it describes (any encoder x any pooling head, SEA-Net's or MILLET's) through the
    config-driven factory in seanet/models/build.py.

    n_clz : number of classes.
    device : where to run.
    lambda_entropy : strength of the attention penalty.
    model_cfg : optional model config (the "model_config" section); None = the default SEA-Net.
    n_in : number of input channels (always 1: one instance is one timestep).
    returns : a SeaNetModel ready to .fit().
    """
    if model_cfg is None:
        net = make_sea_net(n_clz, n_in=n_in)                 # default path
        name = "SEA-Net"
    else:
        net = build_model_from_config(model_cfg, n_clz, n_in=n_in)   # config-driven path
        name = getattr(model_cfg, "name", "model")
    model = SeaNetModel(name, device, n_clz, net, lambda_entropy=lambda_entropy)
    # keep the config that built this network. seanet/deployment.py saves it next to the weights,
    # which is what makes a trained model rebuildable months later on another machine.
    model.model_cfg = model_cfg
    return model


# --------------------------------------------------------------------------------------
# The per-epoch history: save it as a CSV and draw the four curves
# --------------------------------------------------------------------------------------
def save_history(model, name: str, seed: int, history_dir: str) -> Optional[str]:
    """
    Save this run's per-epoch history to a CSV and draw its curves.

    Why we need it: results.csv only has the FINAL test numbers. It cannot say when early stopping
    fired, whether the loss was still falling when we stopped, or whether the model was overfitting.
    fit() already records all of that; this writes it next to the results so it travels with them.

    What is written, into <history_dir>/:
        <dataset>__seed<N>.csv        epoch, train_loss, train_acc, val_loss, val_acc, monitored
        <dataset>__seed<N>_loss.png   training vs validation LOSS, with the best epoch marked
        <dataset>__seed<N>_acc.png    training vs validation ACCURACY

    model : the trained SeaNetModel (its .history holds the curve).
    name : dataset name.  seed : the training seed (part of the filename, so seeds do not clash).
    history_dir : the folder to write into (created if missing).
    returns : the CSV path, or None when there is no history to write.
    """
    history = getattr(model, "history", None)
    if not history:                                              # no epochs ran (or fit was skipped)
        return None
    os.makedirs(history_dir, exist_ok=True)
    frame = pd.DataFrame(history)
    frame["best_epoch"] = int(getattr(model, "best_epoch", 0))
    path = os.path.join(history_dir, f"{name}__seed{int(seed)}.csv")
    frame.to_csv(path, index=False)
    try:
        plot_history(frame, name, seed, history_dir)
    except Exception as e:                                       # a figure must never fail a run
        print(f"    (could not draw history curves for {name}: {type(e).__name__}: {e})", flush=True)
    return path


def plot_history(frame: pd.DataFrame, name: str, seed: int, out_dir: str) -> List[str]:
    """
    Draw the loss curve and the accuracy curve for one run.

    Reading them: the training curve should fall. If the validation curve stops following it and
    turns back up while the training curve keeps falling, the model is memorising the training set
    - that is overfitting, and the dashed line marks the epoch whose weights we actually kept.

    frame : the per-epoch history (from save_history).
    name : dataset name.  seed : training seed.  out_dir : where to write the PNGs.
    returns : the list of paths written.
    """
    import matplotlib
    matplotlib.use("Agg")                                        # no window needed, we only save files
    import matplotlib.pyplot as plt

    best = int(frame["best_epoch"].iloc[0]) if "best_epoch" in frame else 0
    written = []
    for kind, ylabel in (("loss", "loss"), ("acc", "accuracy")):
        train_col, val_col = f"train_{kind}", f"val_{kind}"
        if train_col not in frame:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(frame["epoch"], frame[train_col], label=f"train {ylabel}", linewidth=1.6)
        if val_col in frame and frame[val_col].notna().any():
            ax.plot(frame["epoch"], frame[val_col], label=f"validation {ylabel}", linewidth=1.6)
        else:                                                    # say so instead of drawing nothing
            ax.plot([], [], " ", label="no validation split (dataset too small)")
        if best:
            ax.axvline(best, color="grey", linestyle="--", linewidth=1,
                       label=f"best epoch ({best}) - weights kept from here")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{name} - seed {seed} - training {ylabel}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{name}__seed{int(seed)}_{kind}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


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
    model_cfg=None,
    verbose: bool = False,
    epoch_callback: Optional[Callable] = None,
) -> Tuple[SeaNetModel, MILTSCDataset, Optional[MILTSCDataset], MILTSCDataset, float]:
    """
    Load a dataset, split it, build the model, and train it. (Stages 1 and 2 of a run.)

    Pulled out of train_one so the "interpret" command can reuse it and keep the live model.

    name : dataset name.
    device : where to run; if None, get_device() picks one.
    (all other arguments) : see train_one - same meaning and defaults.
    returns : (model, train_ds, val_ds, test_ds, train_time_s).
    """
    if device is None:
        device = get_device()
    set_seed(seed)                                               # make the run reproducible

    # --- stage 1: load the data and split it ---
    if verbose:
        print("    stage 1/3: loading data ...", flush=True)
    train_full = D.load_dataset(name, "train")
    test_ds = D.load_dataset(name, "test")
    train_ds, val_ds = prepare_splits(train_full, min_train_for_val=min_train_for_val,
                                      val_frac=val_frac, seed=seed)

    # one instance = one timestep, so the network always has a single input channel
    n_in = int(train_full.get_bag(0).shape[1])
    batch_size = max(min(len(train_ds) // 10, max_batch), 2)     # small batch for small datasets

    # --- stage 2: build and train ---
    model = make_model(train_full.n_clz, device, lambda_entropy=lambda_entropy,
                       model_cfg=model_cfg, n_in=n_in)
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
    model_cfg=None,
    verbose: bool = False,
    mlf=None,
    mlf_params: Optional[Dict] = None,
    mlf_tags: Optional[Dict] = None,
    logged_model_name: Optional[str] = None,
    log_model_weights: bool = True,
    pred_dir: Optional[str] = None,
    history_dir: Optional[str] = None,
    deploy_dir: Optional[str] = None,
) -> Dict:
    """
    Train a model on one dataset and score it on the test set. This is the single path used for
    WebTraffic and for every UCR dataset.

    Every hyperparameter is an argument with a sensible default, and the config-driven wrapper
    (train_one_from_config) fills those arguments from the YAML files. Nothing about an experiment
    is hard-coded here.

    Validation policy: if the training set has at least min_train_for_val series we hold out a
    val_frac slice and early-stop on it; otherwise we train on all of it and early-stop on the
    training loss.

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
    model_cfg : optional model config; None = the default SEA-Net.
    verbose : if True, print the 3 stages and show the training bar.
    mlf : the mlflow handle from tracking.start_experiment (None -> no MLflow logging).
    mlf_params : INPUTS to log to MLflow; if None, a summary of this run's recipe is built and logged.
    mlf_tags : extra MLflow labels (e.g. {"command": "single"}).
    logged_model_name : name to give the versioned model in the MLflow "Models" tab.
    log_model_weights : also save the trained network's weights as an artifact.
    pred_dir : if given, save the per-series test probabilities there.
    history_dir : if given, save the per-epoch history CSV + curve figures there.
    deploy_dir : if given, save the complete deployment bundle there (weights + config + ONNX).
    returns : one flat results-row dict (metrics + footprint + metadata).
    """
    from seanet.evaluation import score_model                    # imported here to avoid a cycle

    if device is None:
        device = get_device()

    model, train_ds, val_ds, test_ds, train_time_s = fit_model(
        name, device=device, seed=seed, n_epochs=n_epochs, patience=patience,
        lambda_entropy=lambda_entropy, learning_rate=learning_rate, weight_decay=weight_decay,
        label_smoothing=label_smoothing, max_batch=max_batch, min_train_for_val=min_train_for_val,
        val_frac=val_frac, model_cfg=model_cfg, verbose=verbose,
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
        }
    row = score_model(model, name, train_ds, val_ds, test_ds, device, seed, lambda_entropy,
                      train_time_s, verbose=verbose, mlf=mlf, mlf_params=mlf_params,
                      mlf_tags=mlf_tags, logged_model_name=logged_model_name,
                      log_model_weights=log_model_weights, pred_dir=pred_dir,
                      history_dir=history_dir, deploy_dir=deploy_dir)
    del model                                                    # free the model
    if device.type == "cuda":                                    # free GPU memory before the next dataset
        torch.cuda.empty_cache()
    return row


# --------------------------------------------------------------------------------------
# Config-driven entry points (what main.py actually calls)
# --------------------------------------------------------------------------------------
def _train_kwargs_from_config(cfg, smoke: bool = False) -> Dict:
    """
    Turn a loaded config into the keyword arguments that train_one / fit_model expect.

    Both config-driven wrappers below read the config the same way, so that reading lives here once
    (no duplication). Everything comes from the model file's "training:" block; only the seed and
    the smoke flag come from main.yaml / the command line.

    cfg : a loaded config (from load_config).
    smoke : if True, force a quick 3-epoch run (for testing, not a real result).
    returns : a dict of keyword arguments for train_one / fit_model.
    """
    t = cfg.model_config.training                               # the "training" block of the model file
    n_epochs, patience = (3, 3) if smoke else (t.n_epochs, t.patience)
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
        model_cfg=cfg.model_config,
    )


def train_one_from_config(name: str, cfg, device: Optional[torch.device] = None,
                          smoke: bool = False, verbose: bool = True, **extra) -> Dict:
    """
    Train + score one dataset, with every setting read from the config. Used by every main.py
    command that trains.

    name : dataset name.
    cfg : a loaded config (from load_config).
    device : where to run; None -> get_device().
    smoke : quick 3-epoch check (nothing is saved by the caller).
    verbose : print the stages.
    extra : passed through to train_one (mlf, mlf_params, pred_dir, history_dir, ...).
    returns : one results-row dict.
    """
    return train_one(name, device=device, verbose=verbose,
                     **_train_kwargs_from_config(cfg, smoke), **extra)


def fit_model_from_config(name: str, cfg, device: Optional[torch.device] = None,
                          smoke: bool = False, verbose: bool = True, **extra):
    """
    Load + build + train one dataset from the config, and hand back the LIVE model (not a row).
    Used by the "interpret" command, which needs the model itself to draw explanations.

    name : dataset name.
    cfg : a loaded config.
    device : where to run; None -> get_device().
    smoke : quick 3-epoch check.
    verbose : print the stages.
    extra : passed through to fit_model (e.g. epoch_callback for Optuna).
    returns : (model, train_ds, val_ds, test_ds, train_time_s).
    """
    kwargs = _train_kwargs_from_config(cfg, smoke)
    return fit_model(name, device=device, verbose=verbose, **kwargs, **extra)
