"""
seanet/preprocessing.py - everything that happens to the data BETWEEN loading it and training on it.

What this file is for:
    seanet/data.py only LOADS a dataset from disk. This file does the two preparation steps that
    come next, and nothing else:

      1. normalisation  - each series is z-normalised (mean 0, std 1). MILLET's dataset classes
                          already do this inside apply_bag_transform(), so here we only expose a
                          tiny helper that says whether it is on, plus the maths itself for anyone
                          who wants to normalise an array by hand.
      2. train/val split - cut the training file into a real training part and a validation part,
                          keeping each class's proportion the same in both (a "stratified" split).

    In Multiple Instance Learning a time series is a "bag" and each timestep is an "instance".
    That is the ONLY instance representation this project uses: one instance = one timestep. The
    old sliding-window mode (one instance = a window of W timesteps) was removed in seanetv7,
    because every result in results/ was produced with the per-timestep pipeline and the window
    mode could not produce NDCG (windows have no per-timestep ground truth to compare against).

How a bag is shaped:
    one series of length T  ->  bag of shape (T, 1)  ->  the model sees (batch, 1, T).

Related files:
    - seanet/data.py      -> load_dataset() gives us the dataset we prepare here.
    - seanet/training.py  -> calls split_train_val() before it builds the model.
    - configs/main.yaml   -> the "preprocessing" block (normalise, val_frac, min_train_for_val).
"""
import copy
from typing import List, Tuple

import torch

from millet.data.mil_tsc_dataset import MILTSCDataset


# --------------------------------------------------------------------------------------
# 1. Normalisation
# --------------------------------------------------------------------------------------
def z_normalise(series: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Z-normalise one series: subtract its mean, divide by its standard deviation.

    Why we do it: UCR datasets come in wildly different units (some series live around 0.001,
    some around 5000). Without normalising, the network would spend its capacity learning the
    scale instead of the shape. After it, every series has mean 0 and std 1, so only the SHAPE
    is left for the model to learn.

    series : one series, any shape.
    eps : tiny number so a flat series (std = 0) does not divide by zero.
    returns : the normalised series, same shape.
    """
    mean = series.mean()
    std = series.std()
    return (series - mean) / (std + eps)


def normalisation_is_on(dataset: MILTSCDataset) -> bool:
    """
    Say whether this dataset normalises its bags when you index it.

    MILLET's dataset classes carry an `apply_transform` flag and do the z-normalisation inside
    apply_bag_transform(). We do not repeat that work here; this helper just lets the pipeline
    print/log which setting was used, so a run can be reproduced.

    dataset : a loaded dataset.
    returns : True when the dataset normalises each bag as it is read.
    """
    return bool(getattr(dataset, "apply_transform", True))


# --------------------------------------------------------------------------------------
# 2. Train / validation split (stratified: each class keeps its proportion)
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
    dataset: MILTSCDataset, val_frac: float = 0.2, seed: int = 0
) -> Tuple[MILTSCDataset, MILTSCDataset]:
    """
    Split a dataset into (train, validation), keeping each class's proportion the same in both.

    "Stratified" means we split class by class instead of shuffling everything together. On a
    small dataset a plain random split can easily put every example of a rare class in one side,
    which makes the validation loss meaningless.

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


def prepare_splits(train_full: MILTSCDataset, min_train_for_val: int = 100,
                   val_frac: float = 0.2, seed: int = 0):
    """
    Decide whether this dataset is big enough to hold out a validation set, and split it if so.

    The rule: a validation set of 5 series is pure noise, so on tiny datasets we train on all of
    the training file and let early stopping watch the TRAINING loss instead. This is the one
    place that rule is written down.

    train_full : the whole training split, straight from load_dataset(name, "train").
    min_train_for_val : need at least this many series before holding a validation set out.
    val_frac : how much to hold out when we do.
    seed : random seed for the split.
    returns : (train_ds, val_ds) - val_ds is None when the dataset was too small.
    """
    if len(train_full) >= min_train_for_val:
        return split_train_val(train_full, val_frac=val_frac, seed=seed)
    return train_full, None
