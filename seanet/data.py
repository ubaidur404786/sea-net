"""
seanet/data.py - everything about loading a dataset and describing it.

What this file is for:
    This is the only place we load data. Whether it is the synthetic WebTraffic dataset or any of
    the 128 UCR datasets, we always load it by name through one function, load_dataset(name, split).
    The rest of the project (model, train, results) never reads a data file directly; it just asks
    this file for a dataset object.

Input:
    A dataset name (like "Coffee" or "WebTraffic") and a split ("train" or "test").
    The actual files live under data/ (data/UCR/... and data/WebTraffic/...).

Output:
    - load_dataset -> a MILLET dataset object you can loop over (already z-normalised).
    - summarise_dataset / write_summary_row -> one row of info per dataset in
      results/SEA_NET/data_summary.csv (sizes, series length, number of classes, class balance).

Related files:
    - It reuses three classes from the original MILLET code (millet/data/...): UCRDataset,
      WebTrafficDataset, and the base MILTSCDataset (which does the z-normalisation).
    - seanet/training.py calls load_dataset to get the train/test sets.
    - main.py ("summary" command) calls summarise_dataset + write_summary_row.

The one thing we change vs MILLET:
    15 UCR datasets have missing values or different-length series, so their raw files contain
    NaN, which would ruin normalisation and give NaN loss. The UCR archive ships fixed copies of
    those 15 in a special folder. For those names we read the fixed copy instead (see
    AdjustedUCRDataset). We do NOT edit the millet code to do this; we subclass it here.
"""
import os
from typing import Dict, List, Tuple

import pandas as pd
import torch

from millet.data.mil_tsc_dataset import MILTSCDataset
from millet.data.ucr_2018_dataset import UCRDataset
from millet.data.web_traffic_dataset import WebTrafficDataset


def read_our_csv(path: str) -> pd.DataFrame:
    """
    Read one of OUR csv files (data_summary.csv / results.csv) in a way that survives column
    alignment.

    Why this exists: this machine has some tool that "prettifies" csv files by padding the
    columns with spaces, so the header 'dataset' turns into 'dataset      ' and a normal
    pd.read_csv(path)["dataset"] then throws KeyError. This actually broke a full run once.
    Setting skipinitialspace=True and stripping the header + text cells fixes it. Our csv files
    never put a comma inside a value, so extra spaces can only pad a cell, never split a row.

    path : the csv file to read.
    returns : a DataFrame with clean (stripped) column names and text values.
    """
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()             # drop padding spaces from the header
    for col in df.columns:
        if df[col].dtype == object:                 # text columns: also strip each value
            df[col] = df[col].str.strip()
    return df


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# Name of the special folder inside data/UCR that holds the fixed copies of the 15 problem
# datasets. This name is fixed by the UCR archive itself, we do not choose it.
ADJUSTED_SUBDIR = "Missing_value_and_variable_length_datasets_adjusted"

# The 15 datasets we must read from the fixed folder instead of their normal folder. The list is
# copied straight from the archive's own README in that folder (4 with missing values, 11 with
# variable length). None of these 15 are in the MILLET paper's 85 datasets, which is why the
# paper could skip them. We still run them, they just have no MILLET number to compare against.
UCR_ADJUSTED_DATASETS = frozenset({
    # 1-4: had missing values (fixed by interpolation)
    "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend", "MelbournePedestrian",
    # 5-15: had variable length (fixed by padding to the longest series)
    "AllGestureWiimoteX", "AllGestureWiimoteY", "AllGestureWiimoteZ",
    "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3",
    "GesturePebbleZ1", "GesturePebbleZ2",
    "PickupGestureWiimoteZ", "PLAID", "ShakeGestureWiimoteZ",
})

# The 128 official UCR dataset names, sorted. I did not type these from memory: I generated the
# list by scanning the downloaded archive (every folder in data/UCR that has a <name>_TRAIN.tsv,
# skipping the adjusted folder) and checked the count is 128. It is kept here as a plain list so
# it is easy to read, and discover_ucr_datasets() re-checks it against the disk at runtime.
UCR_128_DATASETS: List[str] = [
    "ACSF1", "Adiac", "AllGestureWiimoteX", "AllGestureWiimoteY",
    "AllGestureWiimoteZ", "ArrowHead", "BME", "Beef",
    "BeetleFly", "BirdChicken", "CBF", "Car",
    "Chinatown", "ChlorineConcentration", "CinCECGTorso", "Coffee",
    "Computers", "CricketX", "CricketY", "CricketZ",
    "Crop", "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW", "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend",
    "ECG200", "ECG5000", "ECGFiveDays", "EOGHorizontalSignal",
    "EOGVerticalSignal", "Earthquakes", "ElectricDevices", "EthanolLevel",
    "FaceAll", "FaceFour", "FacesUCR", "FiftyWords",
    "Fish", "FordA", "FordB", "FreezerRegularTrain",
    "FreezerSmallTrain", "Fungi", "GestureMidAirD1", "GestureMidAirD2",
    "GestureMidAirD3", "GesturePebbleZ1", "GesturePebbleZ2", "GunPoint",
    "GunPointAgeSpan", "GunPointMaleVersusFemale", "GunPointOldVersusYoung", "Ham",
    "HandOutlines", "Haptics", "Herring", "HouseTwenty",
    "InlineSkate", "InsectEPGRegularTrain", "InsectEPGSmallTrain", "InsectWingbeatSound",
    "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Lightning7",
    "Mallat", "Meat", "MedicalImages", "MelbournePedestrian",
    "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MixedShapesRegularTrain",
    "MixedShapesSmallTrain", "MoteStrain", "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2",
    "OSULeaf", "OliveOil", "PLAID", "PhalangesOutlinesCorrect",
    "Phoneme", "PickupGestureWiimoteZ", "PigAirwayPressure", "PigArtPressure",
    "PigCVP", "Plane", "PowerCons", "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices", "Rock",
    "ScreenType", "SemgHandGenderCh2", "SemgHandMovementCh2", "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ", "ShapeletSim", "ShapesAll", "SmallKitchenAppliances",
    "SmoothSubspace", "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves",
    "Strawberry", "SwedishLeaf", "Symbols", "SyntheticControl",
    "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG",
    "TwoPatterns", "UMD", "UWaveGestureLibraryAll", "UWaveGestureLibraryX",
    "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Wafer", "Wine",
    "WordSynonyms", "Worms", "WormsTwoClass", "Yoga",
]
assert len(UCR_128_DATASETS) == 128, f"expected 128 UCR names, got {len(UCR_128_DATASETS)}"

# The synthetic dataset that ships with the repo. We always run it first because it is the only
# dataset with per-timestep labels, so it is the only one where the NDCG@n metric is defined.
WEB_TRAFFIC = "WebTraffic"

# Where we save the per-dataset summary table.
DATA_SUMMARY_CSV = os.path.join("results", "SEA_NET", "data_summary.csv")


# --------------------------------------------------------------------------------------
# Working-directory helper
# --------------------------------------------------------------------------------------
def chdir_to_repo_root(marker: str = os.path.join("data", "WebTraffic")) -> str:
    """
    Move into the repo root folder.

    The MILLET loaders build relative paths like "data/UCR/..." and "data/WebTraffic/...", so they
    only work if the current folder is the repo root. This walks up from wherever we started until
    it finds a folder that contains `marker`, and does os.chdir into it. main.py calls this once at
    the start so it does not matter where you launched python from.

    marker : a path that only exists at the repo root (default: the WebTraffic data folder).
    returns : the absolute path of the repo root we moved into.
    """
    cur = os.path.abspath(os.getcwd())
    while True:
        if os.path.exists(os.path.join(cur, marker)):   # found it -> go there and stop
            os.chdir(cur)
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:                                # hit the drive root without finding it
            raise FileNotFoundError(
                f"Could not find repo root containing {marker!r} above {os.getcwd()!r}. "
                "Run from inside the SEA-Net repo."
            )
        cur = parent                                     # go one folder up and try again


# --------------------------------------------------------------------------------------
# Path resolution (including the fixed-folder override for the 15 datasets)
# --------------------------------------------------------------------------------------
def is_adjusted(name: str) -> bool:
    """
    Say whether a UCR dataset must be read from the fixed folder.

    name : a UCR dataset name.
    returns : True if it is one of the 15 problem datasets, else False.
    """
    return name in UCR_ADJUSTED_DATASETS


def ucr_tsv_path(name: str, split: str, data_root: str = os.path.join("data", "UCR")) -> str:
    """
    Build the path to a UCR .tsv file, sending the 15 problem datasets to the fixed folder and
    everything else to its normal folder.

    name : UCR dataset name.
    split : "train" or "test" (the file name uses the upper-case version).
    data_root : the root of the UCR archive on disk.
    returns : the full path to "<...>/<name>_<SPLIT>.tsv".
    """
    file_name = "{:s}_{:s}.tsv".format(name, split.upper())
    if is_adjusted(name):                                        # one of the 15 -> fixed folder
        return os.path.join(data_root, ADJUSTED_SUBDIR, name, file_name)
    return os.path.join(data_root, name, file_name)             # normal dataset -> normal folder


def discover_ucr_datasets(data_root: str = os.path.join("data", "UCR")) -> List[str]:
    """
    Look at the archive on disk and check it matches our hard-coded list of 128 names.

    This is a safety check: it fails loudly if the folder is missing datasets or has extra ones,
    so we notice a bad/incomplete download instead of silently skipping data.

    data_root : the root of the UCR archive on disk.
    returns : the sorted list of dataset names actually found on disk.
    """
    if not os.path.isdir(data_root):                             # archive not downloaded yet
        raise FileNotFoundError(
            f"{data_root!r} not found. Provision the UCR archive first "
            "(copy or download it into data/UCR)."
        )
    # a real dataset = a folder (not the adjusted one) that has a <name>_TRAIN.tsv inside it
    found = sorted(
        d for d in os.listdir(data_root)
        if d != ADJUSTED_SUBDIR
        and os.path.isdir(os.path.join(data_root, d))
        and os.path.exists(os.path.join(data_root, d, f"{d}_TRAIN.tsv"))
    )
    on_disk, canonical = set(found), set(UCR_128_DATASETS)
    if on_disk != canonical:                                     # disk and our list disagree
        missing = sorted(canonical - on_disk)
        extra = sorted(on_disk - canonical)
        raise AssertionError(
            "Archive on disk does not match the canonical 128-name list.\n"
            f"  missing from disk: {missing}\n"
            f"  unexpected on disk: {extra}"
        )
    return found


# --------------------------------------------------------------------------------------
# The fixed-folder dataset class (the only place we change MILLET's behaviour)
# --------------------------------------------------------------------------------------
def _load_ucr_frame_and_targets(csv_path: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Read a UCR .tsv from any path and turn it into (list of series tensors, targets tensor).

    This copies MILLET's own parsing (millet/data/ucr_2018_dataset.py) line for line. The only
    reason it exists is that MILLET hard-codes the file path with no way to change it, so I could
    not reuse its method directly. If MILLET's parsing ever changes, this must be updated to match.
    It is used only by AdjustedUCRDataset (the 15 datasets); the other 113 use MILLET directly.

    csv_path : the already-resolved path to the .tsv file.
    returns : (list of (T,1) float tensors, an int tensor of class labels), same as MILLET returns.
    """
    df = pd.read_csv(csv_path, sep="\t", header=None)
    ts_collection = []
    targets = []
    for row_idx in range(len(df)):                       # one row = one time series
        ts_pd = df.iloc[row_idx, 1:]                     # columns 1.. are the series values
        target = df.iloc[row_idx, 0]                     # column 0 is the class label
        ts_tensor = torch.as_tensor(ts_pd.to_numpy(), dtype=torch.float)
        ts_tensor = ts_tensor.unsqueeze(1)               # shape (T,) -> (T, 1) (1 channel)
        ts_collection.append(ts_tensor)
        targets.append(target)
    targets_tensor = torch.as_tensor(targets, dtype=torch.int)
    # Re-index the labels to 0..c-1 (MILLET's exact logic: handle the {-1,1} case and the
    # "labels start at 1" case).
    unique_targets = sorted(torch.unique(targets_tensor))
    if -1 in unique_targets:                             # binary labels are {-1, 1} -> make -1 be 0
        if 0 in unique_targets:
            raise ValueError
        targets_tensor[targets_tensor == -1] = 0
    else:                                                # labels like {1,2,3} -> shift down to {0,1,2}
        targets_tensor -= torch.min(targets_tensor)
    return ts_collection, targets_tensor


class AdjustedUCRDataset(UCRDataset):
    """
    A UCRDataset that reads from the fixed folder instead of the raw folder.

    We only override the one method that reads the file, so it points at
    data/UCR/<ADJUSTED_SUBDIR>/<name>/... . Everything else (normalisation, the bag/target
    format, the dataloader) is inherited from MILLET's UCRDataset unchanged.
    """

    def get_time_series_collection_and_targets(self, split: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Load this dataset's series + labels from the fixed folder.

        split : "train" or "test".
        returns : (list of (T,1) tensors, int tensor of labels).
        """
        csv_path = ucr_tsv_path(self.dataset_name, split)   # resolves to the fixed folder for these names
        return _load_ucr_frame_and_targets(csv_path)


# --------------------------------------------------------------------------------------
# The single entry point: load_dataset(name, split)
# --------------------------------------------------------------------------------------
def load_dataset(name: str, split: str) -> MILTSCDataset:
    """
    Load any dataset by name. This is the one function the whole project uses to get data.

    - "WebTraffic"                 -> WebTrafficDataset (has per-timestep labels -> NDCG@n).
    - one of the 15 problem names  -> AdjustedUCRDataset (reads the fixed copy).
    - any other UCR name           -> MILLET's UCRDataset, unchanged.

    name : dataset name, e.g. "WebTraffic", "Coffee", "PLAID".
    split : "train" or "test".
    returns : a dataset object; z-normalisation is applied automatically when you index it.
    """
    if name == WEB_TRAFFIC:
        return WebTrafficDataset(split)
    if is_adjusted(name):
        return AdjustedUCRDataset(name, split)
    return UCRDataset(name, split)


# --------------------------------------------------------------------------------------
# Raw-frame reader (used by the sanity check and the summary)
# --------------------------------------------------------------------------------------
def read_raw_frame(name: str, split: str) -> pd.DataFrame:
    """
    Read the raw file for a dataset as a plain table where column 0 is the label and columns 1..
    are the series values.

    UCR files are tab-separated with no header; WebTraffic is comma-separated WITH a header. We
    turn both into the same "column 0 = label" shape so the sanity check below works for either.
    The 15 problem datasets are read from the fixed folder here too.

    name : dataset name.
    split : "train" or "test".
    returns : a header-less DataFrame (column 0 = label, columns 1.. = series).
    """
    if name == WEB_TRAFFIC:
        df = pd.read_csv(os.path.join("data", "WebTraffic", f"{name}_{split.upper()}.csv"))
        df.columns = range(df.shape[1])                 # throw away header names -> 0,1,2,... like UCR
        return df
    return pd.read_csv(ucr_tsv_path(name, split), sep="\t", header=None)


# --------------------------------------------------------------------------------------
# Label/column sanity check
# --------------------------------------------------------------------------------------
def assert_label_column_convention(name: str, split: str) -> Dict:
    """
    Check that a dataset file really is "column 0 = class label, rest = a time series", and raise
    a clear error if it is not. This catches a corrupted or transposed file before we train on it.

    The rules (kept loose on purpose, this is a guard not a statistics test):
      - column 0 has few unique values and they are all whole numbers -> looks like a label.
      - columns 1.. have many more unique values -> looks like a real (roughly continuous) series.

    name : dataset name.
    split : "train" or "test".
    returns : a small dict of the measured numbers (also used by the summary).
    raises ValueError : if the file does not look like label + series.
    """
    df = read_raw_frame(name, split)
    if df.shape[1] < 2:                                          # need at least a label + one value
        raise ValueError(f"[{name}/{split}] expected >=2 columns (label + series), got shape {df.shape}")

    label_col = df.iloc[:, 0]                                    # the class column
    series_vals = df.iloc[:, 1:].to_numpy().ravel()             # all the series numbers, flattened

    n_label_unique = int(pd.unique(label_col.dropna()).size)
    # allow float labels like 1.0 / -1.0 that UCR uses -> "integer-valued" means it equals its round
    labels_are_integer = bool((label_col.dropna() == label_col.dropna().round()).all())
    # count how many different series values there are (ignoring NaN)
    series_series = pd.Series(series_vals)
    n_series_unique = int(pd.unique(series_series.dropna()).size)

    # now the actual checks
    if not labels_are_integer:                                  # column 0 has decimals -> not a label
        raise ValueError(
            f"[{name}/{split}] column 0 is not integer-valued -> does not look like a class "
            f"label column (sample: {label_col.head().tolist()})"
        )
    if n_label_unique < 2:                                       # only one class -> unusable
        raise ValueError(
            f"[{name}/{split}] column 0 has <2 unique values ({n_label_unique}) -> not a usable "
            "class label column"
        )
    if not (n_series_unique > n_label_unique and n_series_unique >= 10):   # series too flat -> suspicious
        raise ValueError(
            f"[{name}/{split}] series columns have too few unique values "
            f"(n_series_unique={n_series_unique}) relative to the label column "
            f"(n_label_unique={n_label_unique}) -> column 0 may not be the label, or the file "
            "is malformed/transposed"
        )

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "series_length": int(df.shape[1] - 1),
        "n_label_unique": n_label_unique,
        "n_series_unique": n_series_unique,
        "raw_nan_count": int(pd.isna(series_vals).sum()),       # should be 0 after the fixed-folder routing
    }


# --------------------------------------------------------------------------------------
# Per-dataset summary row
# --------------------------------------------------------------------------------------
def summarise_dataset(name: str) -> Dict:
    """
    Build one summary row for a dataset: how many train/test series, series length, number of
    classes, class balance, and whether it used the fixed folder. It also runs the sanity check
    on both splits, so a broken dataset is caught here.

    Class balance is computed from the loaded labels (after re-indexing to 0..c-1), so it matches
    exactly what the model will see.

    name : dataset name.
    returns : a flat dict, ready to be written as one row of data_summary.csv.
    """
    # 1) sanity-check both files first (raises loudly if a file is malformed)
    train_check = assert_label_column_convention(name, "train")
    test_check = assert_label_column_convention(name, "test")

    # 2) load through the same path the model uses (so the fixed folder is applied for the 15)
    ds_train = load_dataset(name, "train")
    ds_test = load_dataset(name, "test")

    series_length = int(len(ds_train.get_bag(0)))               # all series in a dataset share T
    n_classes = int(ds_train.n_clz)

    # 3) count how many train series fall in each class -> class balance
    targets = torch.as_tensor(ds_train.targets).long()
    counts = torch.bincount(targets, minlength=n_classes).tolist()
    n_train = int(len(ds_train))
    class_fracs = [c / n_train for c in counts] if n_train else []
    min_frac = float(min(class_fracs)) if class_fracs else 0.0
    max_frac = float(max(class_fracs)) if class_fracs else 0.0
    imbalance_ratio = float(max(counts) / max(min(counts), 1)) if counts else 0.0   # biggest / smallest class

    return {
        "dataset": name,
        "source": "WebTraffic" if name == WEB_TRAFFIC else "UCR",
        "used_adjusted_folder": is_adjusted(name),
        "n_train": n_train,
        "n_test": int(len(ds_test)),
        "series_length": series_length,
        "n_classes": n_classes,
        # store the counts space-separated (not as a JSON list) so this cell has no commas that a
        # csv reader could split on; read it back with [int(x) for x in field.split()]
        "train_class_counts": " ".join(str(c) for c in counts),
        "min_class_frac": round(min_frac, 4),
        "max_class_frac": round(max_frac, 4),
        "imbalance_ratio": round(imbalance_ratio, 2),
        # proof that both files were clean (0 NaN even for the 15 fixed datasets -> the whole point)
        "train_raw_nan": train_check["raw_nan_count"],
        "test_raw_nan": test_check["raw_nan_count"],
        "sanity_ok": True,
    }


def write_summary_row(row: Dict, path: str = DATA_SUMMARY_CSV) -> None:
    """
    Write one dataset's summary row into data_summary.csv (one row per dataset).

    If a row for this dataset is already there it is replaced (so re-running refreshes it instead
    of duplicating), otherwise it is added.

    row : a summary dict from summarise_dataset.
    path : the csv file to write (default results/SEA_NET/data_summary.csv).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = pd.DataFrame([row])
    if os.path.exists(path):
        existing = read_our_csv(path)
        existing = existing[existing["dataset"] != row["dataset"]]   # drop the old row for this dataset
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
    combined.to_csv(path, index=False)


def summary_row_exists(name: str, path: str = DATA_SUMMARY_CSV) -> bool:
    """
    Say whether data_summary.csv already has a row for this dataset (used to skip re-summarising).

    name : dataset name.
    path : the csv file to check.
    returns : True if a row for `name` is already present.
    """
    if not os.path.exists(path):
        return False
    existing = read_our_csv(path)
    return name in set(existing.get("dataset", pd.Series(dtype=str)).tolist())
