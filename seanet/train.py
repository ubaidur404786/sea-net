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
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch import nn

from millet.data.mil_tsc_dataset import MILTSCDataset
from millet.model.millet_model import MILLETModel
from millet.util import cross_entropy_criterion, custom_tqdm

from seanet import data as D
from seanet.model import make_sea_net, num_params, state_dict_size_mb

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
        returns : nothing; the best weights are stored in self.net.
        """
        loader = train_dataset.create_dataloader(shuffle=True, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = make_label_smoothing_criterion(smoothing)
        monitor_ds = val_dataset if val_dataset is not None else train_dataset   # what early stopping watches

        best_net = None                                          # best weights seen so far
        best_loss = np.inf
        epochs_no_improve = 0
        # progress bar over epochs; leave=False so it vanishes when the dataset is done
        epoch_bar = custom_tqdm(range(n_epochs), desc="    training", disable=not verbose, leave=False)
        for epoch in epoch_bar:
            self.net.train()                                     # training mode (dropout/batchnorm on)
            for batch in loader:                                 # one pass over the training data
                targets = batch["targets"].to(self.device)
                optimizer.zero_grad()
                out = self(batch["bags"])
                loss = criterion(out["bag_logits"], targets)     # classification loss
                if self.lambda_entropy > 0:                      # plus the attention focus penalty
                    loss = loss + self.lambda_entropy * attention_entropy(out["attn"])
                loss.backward()                                  # backprop
                optimizer.step()                                 # update the weights

            self.net.eval()                                      # eval mode to measure the loss
            monitor_loss = self.dataset_loss(monitor_ds, criterion)
            if monitor_loss != monitor_loss:                     # NaN check (NaN != NaN is True)
                if verbose:
                    print("    training: NaN loss -> stopping (keeping best weights so far)")
                break
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


def make_model(n_clz: int, device: torch.device, lambda_entropy: float = LAMBDA_ENTROPY) -> SeaNetModel:
    """
    Build a trainable SEA-Net model for a dataset.

    n_clz : number of classes.
    device : where to run.
    lambda_entropy : strength of the attention penalty.
    returns : a SeaNetModel ready to .fit().
    """
    return SeaNetModel("SEA-Net", device, n_clz, make_sea_net(n_clz), lambda_entropy=lambda_entropy)


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
# Train + evaluate ONE dataset -> one results row (the sweep calls this per dataset)
# --------------------------------------------------------------------------------------
def train_one(
    name: str,
    device: Optional[torch.device] = None,
    seed: int = SEED,
    n_epochs: int = N_EPOCHS,
    patience: int = PATIENCE,
    lambda_entropy: float = LAMBDA_ENTROPY,
    verbose: bool = False,
) -> Dict:
    """
    Train SEA-Net on one dataset and score it on the test set. This is the single path used for
    WebTraffic and every UCR dataset.

    Validation policy: if the training set has at least MIN_TRAIN_FOR_VAL series we hold out 20%
    for validation and early-stop on it; otherwise we train on all of it and early-stop on the
    training loss.

    name : dataset name.
    device : where to run; if None, get_device() picks one.
    seed : random seed for reproducibility.
    n_epochs, patience : training length / early stopping (small values are used for --smoke).
    lambda_entropy : strength of the attention penalty.
    verbose : if True, print the 3 stages and show the training bar.
    returns : one flat results-row dict (metrics + footprint + metadata).
    """
    if device is None:
        device = get_device()
    torch.manual_seed(seed)                                      # make the run reproducible

    # --- stage 1: load the data ---
    if verbose:
        print("    stage 1/3: loading data ...", flush=True)
    train_full = D.load_dataset(name, "train")
    test_ds = D.load_dataset(name, "test")

    # decide whether to hold out a validation set (only if there is enough training data)
    if len(train_full) >= MIN_TRAIN_FOR_VAL:
        train_ds, val_ds = split_train_val(train_full, val_frac=VAL_FRAC, seed=seed)
    else:
        train_ds, val_ds = train_full, None

    batch_size = max(min(len(train_ds) // 10, MAX_BATCH), 2)     # small batch for small datasets
    model = make_model(train_full.n_clz, device, lambda_entropy=lambda_entropy)

    # --- stage 2: train ---
    if verbose:
        val_note = f"val={len(val_ds)}" if val_ds is not None else "no val (train-loss early stop)"
        print(f"    stage 2/3: training on {len(train_ds)} series ({val_note}, "
              f"C={train_full.n_clz}, T={len(train_full.get_bag(0))}, batch={batch_size}) ...", flush=True)
    t0 = time.perf_counter()
    model.fit(train_ds, val_ds, n_epochs=n_epochs, patience=patience, batch_size=batch_size, verbose=verbose)
    train_time_s = time.perf_counter() - t0

    # --- stage 3: score on the test set ---
    # safe_evaluate gives accuracy/AUROC/loss (AUROC may be NaN). evaluate_interpretability gives
    # AOPCR (always) and NDCG (WebTraffic only). AOPCR is the slow part because it re-runs the model.
    if verbose:
        print(f"    stage 3/3: scoring on {len(test_ds)} test series (accuracy + AOPCR"
              f"{' + NDCG' if name == D.WEB_TRAFFIC else ''}) ...", flush=True)
    cls = safe_evaluate(model, test_ds)
    aopcr, ndcg = model.evaluate_interpretability(test_ds)

    # pack everything into one flat row (this is what gets written to results.csv)
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
    del model                                                   # free the model
    if device.type == "cuda":                                  # free GPU memory before the next dataset
        torch.cuda.empty_cache()
    return row
