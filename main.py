"""
main.py - the one place you run everything from.

What this file is for:
    This is the command-line entry point for the whole project. You do not import the seanet
    modules yourself; you run "python main.py <command>" and this file calls the right functions.
    It parses the command, sets up the working directory, and prints the results.

Commands (run "python main.py -h" to see them):
    python main.py summary [NAME|--all]   look at the data (shapes + a summary row per dataset)
    python main.py params                 print how much smaller SEA-Net is than the baseline
    python main.py webtraffic [--smoke]   train on WebTraffic and compare to MILLET
    python main.py single NAME [--smoke]  train + evaluate one dataset
    python main.py train [options]        the full sweep: WebTraffic + all 128 UCR datasets
    python main.py results                rebuild + print the comparison table vs MILLET

Related files:
    - seanet/data.py    -> loading, summaries (used by "summary").
    - seanet/model.py   -> the model + size helpers (used by "params").
    - seanet/train.py   -> train_one() + get_device() (used by "webtraffic", "single", "train").
    - seanet/results.py -> saving results + the comparison (used by "train", "single", "results").
    - analysis.ipynb    -> the notebook that turns the saved csv files into figures.

The training commands (webtraffic / single / train without --smoke) really train models, so you
run them yourself. Everything is resumable: re-running "train" skips datasets that are already
finished (it checks done.txt), so it is safe to stop with Ctrl+C and start again.
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so "import seanet" works

# Silence one harmless warning to keep the log clean. pandas' to_numpy() gives a read-only array,
# and torch.as_tensor wraps it without copying, so PyTorch warns it is "not writable". We never
# write to those tensors (normalisation makes a new one), so it is safe to ignore just this message.
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

import torch

from seanet import data as D
from seanet.model import make_sea_net, make_baseline, num_params, state_dict_size_mb
from seanet.train import train_one, get_device
from seanet.results import result_exists, save_result_row, build_comparison, millet_baseline, RESULTS_CSV


# ---------------------------------------------------------------------------
# small print helpers
# ---------------------------------------------------------------------------
def summarise_and_print(name):
    """
    Summarise one dataset (this also runs the sanity check), save the row, and print a one-liner.

    name : dataset name.
    returns : nothing.
    """
    row = D.summarise_dataset(name)                          # build the summary (raises if the file is bad)
    D.write_summary_row(row)                                 # save it to data_summary.csv
    print(f"  {name:28s} src={row['source']:10s} adj={str(row['used_adjusted_folder']):5s} "
          f"train/test={row['n_train']}/{row['n_test']:<5d} T={row['series_length']:<5d} "
          f"C={row['n_classes']:<3d} imbalance={row['imbalance_ratio']:.2f} "
          f"raw_nan(tr/te)={row['train_raw_nan']}/{row['test_raw_nan']}")


def print_shapes(name):
    """
    Print the tensor shapes for one dataset, so you can see what the model receives.

    name : dataset name.
    returns : nothing.
    """
    ds = D.load_dataset(name, "train")
    bag = ds[0]["bag"]                                       # one normalised series, shape (T, 1)
    batch = next(iter(ds.create_dataloader(batch_size=4)))   # one batch of 4 series
    stacked = torch.stack(batch["bags"])                     # (B, T, 1)
    has_inst = batch.get("instance_targets") is not None     # only WebTraffic has per-timestep labels
    print(f"  [{name}] bag (T,1)={tuple(bag.shape)}  stack (B,T,1)={tuple(stacked.shape)}  "
          f"model (B,1,T)={tuple(stacked.transpose(1, 2).shape)}  instance_labels={has_inst}")


def print_row(row):
    """
    Print the important fields of a results row.

    row : a results-row dict from train_one.
    returns : nothing.
    """
    for k in ["params", "model_size_mb", "n_train", "n_val", "n_test", "series_length",
              "n_classes", "test_acc", "test_aopcr", "test_ndcg", "test_loss", "train_time_s"]:
        print(f"  {k:16s}: {row[k]}")


def webtraffic_millet(metric_csv):
    """
    Get MILLET's WebTraffic baseline for one metric (the mean of its 5 Conjunctive reps).

    metric_csv : file name under results/WebTraffic/InceptionTime/, e.g. "test_acc.csv".
    returns : the mean baseline value as a float.
    """
    import pandas as pd
    df = pd.read_csv(f"results/WebTraffic/InceptionTime/{metric_csv}")
    df.columns = df.columns.str.strip()
    df["Dataset"] = df["Dataset"].str.strip()
    reps = [f"ConjunctiveInceptionTime {i}" for i in range(5)]
    return float(df.set_index("Dataset").loc["WebTraffic", reps].mean())


# ---------------------------------------------------------------------------
# subcommands (one function per "python main.py <command>")
# ---------------------------------------------------------------------------
def cmd_summary(args):
    """
    "summary" command: look at the data.

    With --all it summarises every dataset; with a name it summarises that one; with nothing it
    runs a small demo on WebTraffic + Coffee (shapes + summary rows).

    args : parsed command-line arguments (args.all, args.dataset).
    returns : nothing.
    """
    if args.all:
        D.discover_ucr_datasets()                            # check the archive matches the 128-name list
        for name in [D.WEB_TRAFFIC] + D.UCR_128_DATASETS:
            if D.summary_row_exists(name):                   # already summarised -> skip
                print(f"  {name:28s} already summarised -> skip")
                continue
            try:
                summarise_and_print(name)
            except Exception as e:                           # one bad dataset never stops the loop
                print(f"  {name:28s} FAILED: {type(e).__name__}: {e}")
    elif args.dataset:                                        # a single named dataset
        summarise_and_print(args.dataset)
    else:                                                     # the default demo
        print("=== tensor shapes ===")
        for name in [D.WEB_TRAFFIC, "Coffee"]:
            print_shapes(name)
        print("=== summary rows ===")
        for name in [D.WEB_TRAFFIC, "Coffee"]:
            summarise_and_print(name)
    print(f"\ndata_summary.csv -> {D.DATA_SUMMARY_CSV}")


def cmd_params(args):
    """
    "params" command: print SEA-Net's size next to the MILLET baseline for a few class counts.

    args : parsed arguments (unused).
    returns : nothing.
    """
    header = f"{'model':30s} {'n_clz':>6s} {'params':>12s} {'size (MB)':>11s}"
    print(header)
    print("-" * len(header))
    # counts depend only slightly on the number of classes, so we show a small, a medium and a large
    for label, n_clz in [("Coffee", 2), ("WebTraffic", 10), ("Crop", 24)]:
        sea, base = make_sea_net(n_clz), make_baseline(n_clz)
        sp, bp = num_params(sea), num_params(base)            # parameter counts
        sm, bm = state_dict_size_mb(sea), state_dict_size_mb(base)   # sizes in MB
        print(f"{'SEA-Net':30s} {n_clz:6d} {sp:12,d} {sm:11.3f}   (for {label})")
        print(f"{'InceptionTime+Conjunctive':30s} {n_clz:6d} {bp:12,d} {bm:11.3f}   (for {label})")
        print(f"{'  -> SEA-Net / baseline':30s} {'':6s} {sp / bp:11.2%} {sm / bm:10.2%}\n")


def cmd_webtraffic(args):
    """
    "webtraffic" command: train on WebTraffic and compare to MILLET (this is our main sanity check,
    because WebTraffic is the only dataset with NDCG).

    args : parsed arguments (args.smoke).
    returns : nothing.
    """
    device = get_device()
    print(f"device: {device}  mode: {'smoke' if args.smoke else 'full'}")
    kw = dict(n_epochs=3, patience=3) if args.smoke else {}   # smoke = 3 epochs only
    row = train_one("WebTraffic", device=device, verbose=True, **kw)
    print("\n=== SEA-Net on WebTraffic ===")
    print_row(row)
    # print SEA-Net next to MILLET for the 3 metrics
    print("\n=== vs MILLET ConjunctiveInceptionTime (mean of 5 reps) ===")
    print(f"  {'metric':8s} {'SEA-Net':>10s} {'MILLET':>10s}")
    print(f"  {'acc':8s} {row['test_acc']:>10.4f} {webtraffic_millet('test_acc.csv'):>10.4f}")
    print(f"  {'AOPCR':8s} {row['test_aopcr']:>10.4f} {webtraffic_millet('test_aopcr.csv'):>10.4f}")
    print(f"  {'NDCG@n':8s} {str(row['test_ndcg']):>10s} {webtraffic_millet('test_ndcg.csv'):>10.4f}")
    if args.smoke:                                            # smoke numbers are not real results
        print("\n  (smoke = 3 epochs; correctness only, NOT a result)")
    else:
        save_result_row(row)                                 # save the real result
        print(f"\n  saved -> {RESULTS_CSV}")


def cmd_single(args):
    """
    "single" command: train + evaluate one dataset and save its result.

    args : parsed arguments (args.dataset, args.smoke).
    returns : nothing.
    """
    device = get_device()
    name = args.dataset
    if result_exists(name) and not args.smoke:               # already finished -> do not redo it
        print(f"{name} already done -> skip (delete its row in {RESULTS_CSV} to redo)")
        return
    kw = dict(n_epochs=3, patience=3) if args.smoke else {}
    row = train_one(name, device=device, verbose=True, **kw)
    print(f"\n=== SEA-Net on {name} ===")
    print_row(row)
    base = millet_baseline("test_acc.csv")                   # show the MILLET accuracy if this is one of the 85
    if name in base.index:
        print(f"\n  MILLET acc baseline for {name}: {base[name]:.4f}  (ours {row['test_acc']:.4f})")
    if not args.smoke:
        save_result_row(row)
        print(f"\n  saved -> {RESULTS_CSV}")
    else:
        print("\n  (smoke = 3 epochs; correctness only, NOT a result - not saved)")


def cmd_train(args):
    """
    "train" command: the full sweep. Trains WebTraffic first, then all 128 UCR datasets, saving
    each result. It is resumable (skips finished datasets) and fault-tolerant (a failure on one
    dataset is logged and the loop keeps going).

    args : parsed arguments (args.only, args.limit, args.no_webtraffic, args.smoke).
    returns : nothing.
    """
    device = get_device()
    # decide which datasets to run
    if args.only:
        names = args.only                                    # explicit list from the user
    else:
        D.discover_ucr_datasets()                            # check the archive first
        ucr = D.UCR_128_DATASETS[: args.limit] if args.limit is not None else D.UCR_128_DATASETS
        names = ([] if args.no_webtraffic else [D.WEB_TRAFFIC]) + ucr   # WebTraffic first, then UCR

    total = len(names)
    print(f"device: {device} | mode: {'smoke' if args.smoke else 'full'} | datasets: {total}", flush=True)
    kw = dict(n_epochs=3, patience=3) if args.smoke else {}
    done = skipped = failed = 0                              # running tally for the summary line
    for i, name in enumerate(names, 1):
        tag = f"[{i:>3}/{total}]"                             # e.g. "[ 22/129]"
        if result_exists(name):                              # already finished -> skip
            print(f"{tag} {name:28s} already done -> skip", flush=True)
            skipped += 1
            continue
        try:
            if not D.summary_row_exists(name):               # make sure its data_summary row exists
                D.write_summary_row(D.summarise_dataset(name))
            print(f"{tag} {name}", flush=True)               # header; the stage lines below belong to it
            row = train_one(name, device=device, verbose=True, **kw)   # trains + scores (prints stages)
            save_result_row(row)                             # save the metrics + mark it done
            ndcg = "n/a" if row["test_ndcg"] is None else f"{row['test_ndcg']:.3f}"
            print(f"{tag} {name:28s} DONE  acc={row['test_acc']:.4f} AOPCR={row['test_aopcr']:7.3f} "
                  f"NDCG={ndcg:>5s} C={row['n_classes']:>3d} T={row['series_length']:>5d} "
                  f"({row['train_time_s']:.0f}s)  [{done + 1} trained / {failed} failed so far]", flush=True)
            done += 1
        except KeyboardInterrupt:                            # Ctrl+C -> stop cleanly (progress is saved)
            print("\nInterrupted - progress is saved, re-run to resume.", flush=True)
            raise
        except Exception as e:                               # any other error -> log it and continue
            print(f"{tag} {name:28s} FAILED: {type(e).__name__}: {e}", flush=True)
            failed += 1

    print(f"\nDone: {done} trained, {skipped} skipped, {failed} failed. Results -> {RESULTS_CSV}", flush=True)
    print("\n=== Comparison vs MILLET ===", flush=True)
    build_comparison(verbose=True)                           # print the win/tie/loss summary at the end


def cmd_results(args):
    """
    "results" command: rebuild and print the comparison-vs-MILLET table from whatever is finished.

    args : parsed arguments (unused).
    returns : nothing.
    """
    build_comparison(verbose=True)


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def main():
    """
    Parse the command line and run the chosen subcommand.

    returns : nothing.
    """
    D.chdir_to_repo_root()   # move to the repo root so the "data/..." paths work from anywhere
    parser = argparse.ArgumentParser(prog="main.py", description="SEA-Net: run every part of the pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)   # each command is its own sub-parser

    # summary
    p = sub.add_parser("summary", help="data summary (one dataset, --all, or the demo)")
    p.add_argument("dataset", nargs="?", help="dataset name (omit for the WebTraffic+Coffee demo)")
    p.add_argument("--all", action="store_true", help="summarise WebTraffic + all 128 UCR")
    p.set_defaults(func=cmd_summary)

    # params
    p = sub.add_parser("params", help="SEA-Net vs baseline parameter counts")
    p.set_defaults(func=cmd_params)

    # webtraffic
    p = sub.add_parser("webtraffic", help="train on WebTraffic + sanity-check vs MILLET")
    p.add_argument("--smoke", action="store_true", help="quick check (3 epochs)")
    p.set_defaults(func=cmd_webtraffic)

    # single
    p = sub.add_parser("single", help="train + evaluate one dataset, save its result")
    p.add_argument("dataset", help="dataset name, e.g. Coffee")
    p.add_argument("--smoke", action="store_true", help="quick check (3 epochs)")
    p.set_defaults(func=cmd_single)

    # train (the full sweep)
    p = sub.add_parser("train", help="full sweep: WebTraffic + all 128 UCR (resumable)")
    p.add_argument("--only", nargs="+", metavar="NAME", help="only these datasets")
    p.add_argument("--limit", type=int, help="only the first N UCR datasets")
    p.add_argument("--no-webtraffic", action="store_true", help="UCR only")
    p.add_argument("--smoke", action="store_true", help="quick check (3 epochs each)")
    p.set_defaults(func=cmd_train)

    # results
    p = sub.add_parser("results", help="build + print the comparison vs MILLET")
    p.set_defaults(func=cmd_results)

    args = parser.parse_args()
    args.func(args)          # call the function set by set_defaults for the chosen command


if __name__ == "__main__":
    main()
