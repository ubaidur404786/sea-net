"""
seanet/utils.py - the small shared helpers every other component may use.

Three things live here, and nothing else:
    1. get_device()   - pick cuda / mps / cpu, without crashing on Windows or a CPU-only machine.
    2. set_seed()     - one call that seeds python, numpy and torch, so a run is reproducible.
    3. start_logging()- save a copy of everything printed to a dated log file (the "Tee" below).

This module deliberately imports nothing from the rest of seanet except results.py (for the log
folder paths), so it can never take part in a circular import.

--- 3. LOGGING ---------------------------------------------------------------

What the logging part is for:
    Every command (train, single, run, interpret, optuna, ...) prints its progress and results to
    the terminal. This little helper ALSO writes that same text to a file - so after a run you have
    a permanent record of what happened (handy for long sweeps and for runs on Grid5000).

    It works like the Unix "tee" command: text goes to the screen AND to the file at the same time.

Where the log goes:
    A log belongs to the MODEL it was run for, so it is saved next to that model's results:

        results/SEA_NET/seanet__sea_mstcn_sep__mil_additive/logs/train_2026-07-17_14-02-11.log

    Every run gets its own file, named with the command plus the date and time, so nothing is ever
    overwritten - run the same model ten times and you keep all ten records, in order.

    Commands that belong to no particular model (summary, report) go to the shared folder
    results/SEA_NET/logs/ instead.

    Smoke runs are throwaway pipeline checks (usually on your laptop), so their logs go into a
    logs/smoke/ subfolder that .gitignore excludes. Real training logs ARE committed, so the
    Grid5000 training record travels with the repo.

How to use it (see main.py):
    start_logging("train", model_id="seanet__sea_mstcn_sep__mil_additive")     # -> <model>/logs/train_<date-time>.log
    start_logging("report")                                   # -> results/SEA_NET/logs/report_<date-time>.log
    start_logging("run", model_id=..., smoke=True)            # -> <model>/logs/smoke/run_<date-time>.log

The one class here (Tee) just forwards each write() to two places. Nothing clever.

Related files:
    - seanet/results.py -> logs_dir(model_id) / SHARED_LOGS_DIR (the folders used here).
    - main.py           -> calls start_logging() once, right before running the chosen command.
"""
import os
import random
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import torch

from seanet import results as R   # the module, not the constants - see log_dir_for


# --------------------------------------------------------------------------------------
# 1. Which device do we train on?
# --------------------------------------------------------------------------------------
def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Pick the best torch device that actually exists here.

    MILLET's own get_gpu_device_for_os() raises on Windows and on CPU-only machines; this never
    does. That matters because the SAME code runs on a Windows laptop and on a Grid5000 GPU node.

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


def resolve_device(setting: str = "auto") -> torch.device:
    """
    Turn the config's `device:` string into a real torch device.

    setting : "auto" (pick the best one available), or an explicit "cpu" / "cuda" / "mps".
    returns : a torch.device.
    """
    if not setting or str(setting).lower() == "auto":
        return get_device()
    return torch.device(str(setting))


# --------------------------------------------------------------------------------------
# 2. Reproducibility: seed everything that can draw a random number
# --------------------------------------------------------------------------------------
def set_seed(seed: int = 0) -> int:
    """
    Seed every random number generator this project touches, in one call.

    Three different libraries draw random numbers here and each has its own seed:
      - `random`  : python's own shuffling,
      - `numpy`   : the data indices,
      - `torch`   : the weight initialisation, dropout and the dataloader shuffle.
    Seeding only torch (what the old code did) left the other two free, so two runs with the
    "same seed" could still differ. This fixes all three.

    seed : the seed to use.
    returns : the seed, so a caller can log it.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


class Tee:
    """
    Send every write to two streams at once (e.g. the real terminal AND a log file).

    Example:
        sys.stdout = Tee(sys.stdout, open("run.log", "w"))   # now print() goes to both
    """

    def __init__(self, stream, logfile):
        self.stream = stream          # the original stream (the real terminal)
        self.logfile = logfile        # the open log file we also write to

    def write(self, text):
        self.stream.write(text)       # show it on screen
        self.logfile.write(text)      # and keep a copy in the file
        self.logfile.flush()          # flush so the file is up to date even if the run crashes

    def flush(self):
        self.stream.flush()
        self.logfile.flush()


def log_dir_for(model_id: Optional[str] = None, smoke: bool = False) -> str:
    """
    Work out which folder a log file belongs in.

    model_id : the model this run is for (e.g. "seanet__sea_mstcn_sep__mil_additive"); None for commands that belong
               to no model (summary, report), which use the shared results/SEA_NET/logs/ folder.
    smoke : if True, use the logs/smoke/ subfolder (git-ignored - throwaway checks are not kept).
    returns : the folder path.
    """
    # read R.SHARED_LOGS_DIR through the module, never `from ... import SHARED_LOGS_DIR`: main.py
    # may repoint the results root at start-up, and a name imported at import time would be stale.
    base = R.logs_dir(model_id) if model_id else R.SHARED_LOGS_DIR
    return os.path.join(base, "smoke") if smoke else base


def start_logging(command: str, model_id: Optional[str] = None, smoke: bool = False) -> str:
    """
    Begin saving all terminal output to <log folder>/<command>_<date-time>.log.

    We replace sys.stdout with a Tee, so every print() from here on is written to both the screen
    and the file. tqdm progress bars stay on the terminal only (they write to stderr, which we leave
    alone), so the log file stays clean and readable.

    command : the command name, used in the file name (e.g. "train", "optuna").
    model_id : the model this run is for; None puts the log in the shared folder (see log_dir_for).
    smoke : if True, write into the logs/smoke/ subfolder instead.
    returns : the path of the log file that was opened.
    """
    folder = log_dir_for(model_id, smoke)
    os.makedirs(folder, exist_ok=True)                         # make the logs folder if needed
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")       # date-time, so each run has its own file
    path = os.path.join(folder, f"{command}_{stamp}.log")
    logfile = open(path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, logfile)                      # from now on, print() also writes to the file
    print(f"[log] command '{command}' started at {stamp}"
          + (f" for model '{model_id}'" if model_id else ""))
    print(f"[log] a copy of everything below is being saved to: {path}\n")
    return path
