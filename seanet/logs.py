"""
seanet/logs.py - save a copy of everything printed to a dated log file.

What this file is for:
    Every command (smoke, single, train, run, interpret, optuna, ...) prints its progress and
    results to the terminal. This little helper ALSO writes that same text to a file under
    results/SEA_NET/logs/, named with the command and the date-time - so after a run you have a
    permanent record of what happened (handy for long sweeps and for runs on Grid5000).

    It works like the Unix "tee" command: text goes to the screen AND to the file at the same time.

How to use it (see main.py):
    start_logging("train")     # from now on, every print() also lands in the log file
    ... run the command ...

The one class here (Tee) just forwards each write() to two places. Nothing clever.

Related files:
    - main.py -> calls start_logging(command) once, right before running the chosen command.
"""
import os
import sys
from datetime import datetime

# where the log files go (relative to the repo root, which main.py has already cd'd into)
LOGS_DIR = os.path.join("results", "SEA_NET", "logs")


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


def start_logging(command: str) -> str:
    """
    Begin saving all terminal output to results/SEA_NET/logs/<command>_<date-time>.log.

    We replace sys.stdout with a Tee, so every print() from here on is written to both the screen
    and the file. tqdm progress bars stay on the terminal only (they write to stderr, which we leave
    alone), so the log file stays clean and readable.

    command : the command name, used in the file name (e.g. "train", "optuna").
    returns : the path of the log file that was opened.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)                       # make the logs folder if needed
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")       # date-time, so each run has its own file
    path = os.path.join(LOGS_DIR, f"{command}_{stamp}.log")
    logfile = open(path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, logfile)                      # from now on, print() also writes to the file
    print(f"[log] command '{command}' started at {stamp}")
    print(f"[log] a copy of everything below is being saved to: {path}\n")
    return path
