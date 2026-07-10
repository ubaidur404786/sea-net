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
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so "import seanet" works

# Silence one harmless warning to keep the log clean. pandas' to_numpy() gives a read-only array,
# and torch.as_tensor wraps it without copying, so PyTorch warns it is "not writable". We never
# write to those tensors (normalisation makes a new one), so it is safe to ignore just this message.
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

import torch

from seanet import data as D
from seanet import tracking
from seanet.config import load_config, to_flat_dict
from seanet.logs import start_logging
from seanet.model import make_sea_net, make_baseline, num_params, state_dict_size_mb
from seanet.train import (train_one, train_one_from_config, fit_model_from_config, score_model,
                          get_device)
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


def _mlflow_for_default_run(smoke, config_path=os.path.join("configs", "main.yaml")):
    """
    Start MLflow for the commands that do not build a full config themselves (webtraffic/single/train).

    Those commands train the default SEA-Net, so they do not go through load_config. This reads the
    mlflow block from configs/main.yaml and starts the experiment, so they still record every trained
    model (its params + train/val/test metrics + the versioned model) - the same as "run".

    smoke : if True, MLflow is skipped (smoke runs are throwaway 3-epoch checks; we do not want them
            polluting the runs/models you compare).
    config_path : where to read the mlflow settings from.
    returns : (mlf_handle_or_None, log_model_weights_flag).
    """
    if smoke:
        return None, True
    try:
        cfg = load_config(config_path)
    except Exception as e:                                       # never let a config problem stop training
        print(f"  [mlflow] could not read {config_path} ({e}); MLflow logging is off for this run.")
        return None, True
    mlf = tracking.start_experiment(cfg)
    log_weights = getattr(getattr(cfg, "mlflow", None), "log_model_weights", True)
    return mlf, log_weights


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
    mlf, log_weights = _mlflow_for_default_run(args.smoke)     # record the run in MLflow (skipped for smoke)
    kw = dict(n_epochs=3, patience=3) if args.smoke else {}   # smoke = 3 epochs only
    row = train_one("WebTraffic", device=device, verbose=True, mlf=mlf,
                    mlf_tags={"command": "webtraffic"}, logged_model_name="seanet",
                    log_model_weights=log_weights, **kw)
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
    mlf, log_weights = _mlflow_for_default_run(args.smoke)   # record the run in MLflow (skipped for smoke)
    kw = dict(n_epochs=3, patience=3) if args.smoke else {}
    row = train_one(name, device=device, verbose=True, mlf=mlf,
                    mlf_tags={"command": "single"}, logged_model_name="seanet",
                    log_model_weights=log_weights, **kw)
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
    mlf, log_weights = _mlflow_for_default_run(args.smoke)   # record every dataset's run in MLflow (skipped for smoke)
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
            row = train_one(name, device=device, verbose=True, mlf=mlf,   # trains + scores (prints stages)
                            mlf_tags={"command": "train"}, logged_model_name="seanet",
                            log_model_weights=log_weights, **kw)
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


def cmd_run(args):
    """
    "run" command: the new config-driven entry point. It reads configs/main.yaml (plus the model
    file it points at), then trains + evaluates the chosen dataset with the chosen model, using
    only values from the config. Command-line flags (--model, --dataset, --smoke) override the
    file so you can try things quickly without editing YAML.

    args : parsed arguments (args.config, args.model, args.dataset, args.smoke).
    returns : nothing.
    """
    overrides = {"model": args.model} if args.model else None   # --model swaps the model file
    cfg = load_config(args.config, overrides=overrides)

    # let the command line override a couple of run settings
    dataset = args.dataset or cfg.run.dataset
    smoke = args.smoke or getattr(cfg.run, "smoke", False)
    device = get_device() if cfg.device == "auto" else torch.device(cfg.device)

    # show exactly what config is driving this run (reproducibility + a quick sanity check)
    print(f"=== run: model={cfg.model} dataset={dataset} device={device} "
          f"mode={'smoke' if smoke else 'full'} ===")
    print("resolved config:")
    for key, value in to_flat_dict(cfg).items():
        print(f"  {key} = {value}")

    if cfg.run.mode != "single":                                # only "single" is wired up so far
        raise SystemExit(f"run mode {cfg.run.mode!r} is not supported yet (use mode: single).")

    if result_exists(dataset) and not smoke:                    # already finished -> do not redo it
        print(f"\n{dataset} already done -> skip (delete its row in {RESULTS_CSV} to redo)")
        return

    # MLflow records this run so it can be compared with every other model/dataset later (skipped for
    # smoke, which is a throwaway 3-epoch check). This run's full terminal output is also saved to
    # results/SEA_NET/logs/ by start_logging in main().
    mlf = None if smoke else tracking.start_experiment(cfg)
    log_weights = getattr(getattr(cfg, "mlflow", None), "log_model_weights", True)

    # train + score. fit_model + score_model are the same code train_one uses, so nothing is
    # duplicated. score_model also logs to MLflow when mlf is not None (the whole resolved config is
    # logged as the run's inputs, so the web page shows exactly what produced each result).
    model, train_ds, val_ds, test_ds, train_time_s = fit_model_from_config(
        dataset, cfg, device=device, smoke=smoke, verbose=True)
    lambda_entropy = cfg.model_config.training.lambda_entropy
    row = score_model(model, dataset, train_ds, val_ds, test_ds, device, cfg.seed,
                      lambda_entropy, train_time_s, verbose=True, mlf=mlf,
                      mlf_params=to_flat_dict(cfg), mlf_tags={"command": "run", "mode": cfg.run.mode},
                      logged_model_name=cfg.model, log_model_weights=log_weights)
    print(f"\n=== {cfg.model} on {dataset} ===")
    print_row(row)

    if smoke:
        print("\n  (smoke = 3 epochs; correctness only, NOT a result - not saved)")
    else:
        save_result_row(row)
        print(f"\n  saved -> {RESULTS_CSV}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def cmd_interpret(args):
    """
    "interpret" command: train a model and draw MILLET-style per-sample explanation figures.

    It reads the "interpretability" block of configs/main.yaml (which dataset, how many figures per
    class, how many per test sample, where to save), trains a model on that dataset through the same
    config-driven path as "run", then saves one figure per selected test series. --dataset / --model
    / --smoke override the config for a quick look.

    args : parsed arguments (args.config, args.model, args.dataset, args.smoke).
    returns : nothing.
    """
    from seanet.interpretability import generate_interpretations   # imported here so matplotlib only loads for this command

    overrides = {"model": args.model} if args.model else None
    cfg = load_config(args.config, overrides=overrides)
    icfg = getattr(cfg, "interpretability", None)

    # figure out the settings (command line overrides the config, config overrides the defaults)
    dataset = args.dataset or (getattr(icfg, "dataset", None) if icfg else None) or cfg.run.dataset
    figures_per_class = getattr(icfg, "figures_per_class", 2) if icfg else 2
    figures_per_test = getattr(icfg, "figures_per_test", 4) if icfg else 4
    base_dir = getattr(icfg, "output_dir", os.path.join("results", "SEA_NET", "interpretation")) if icfg \
        else os.path.join("results", "SEA_NET", "interpretation")
    device = get_device() if cfg.device == "auto" else torch.device(cfg.device)
    smoke = args.smoke

    print(f"=== interpret: model={cfg.model} dataset={dataset} device={device} "
          f"mode={'smoke' if smoke else 'full'} ===")

    # train a model to explain (same fit path as training), then draw the figures from it
    model, _, _, test_ds, _ = fit_model_from_config(dataset, cfg, device=device, smoke=smoke, verbose=True)
    print("    drawing figures ...", flush=True)

    if smoke:
        # smoke = a quick check: draw ONE preview figure into a fixed file, so repeated smoke runs
        # do not pile up dozens of throwaway figures. Real (training) runs are the ones we keep.
        out_dir = os.path.join(base_dir, dataset, "smoke")
        paths = generate_interpretations(model, test_ds, out_dir, dataset_name=dataset,
                                         limit=1, fixed_name="smoke_preview.png")
    else:
        # real run: save all figures into a fresh folder stamped with the date and time, so every
        # training run's interpretation results are kept separately and are easy to find later.
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(base_dir, dataset, stamp)
        paths = generate_interpretations(model, test_ds, out_dir, figures_per_class, figures_per_test, dataset)

    print(f"\nsaved {len(paths)} figure(s) to {out_dir}/")
    for p in paths[:8]:
        print("  ", p)
    if len(paths) > 8:
        print(f"   ... and {len(paths) - 8} more")

    if smoke:
        print("\n  (smoke = 3 epochs + 1 preview figure only; run without --smoke to save the full,\n"
              "   date-stamped set from a properly trained model)")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def cmd_optuna(args):
    """
    "optuna" command: run an Optuna hyperparameter search for the chosen model.

    It reads the "optuna" block from the model config (configs/models/<model>.yaml), trains many
    trials on the dataset (WebTraffic by default), and - unless smoke - saves the best values to
    configs/models/<model>.best.yaml so future runs use them automatically.

    args : parsed arguments (args.config, args.model, args.dataset, args.smoke).
    returns : nothing.
    """
    from seanet.optimize import run_optuna

    overrides = {"model": args.model} if args.model else None
    cfg = load_config(args.config, overrides=overrides)
    device = get_device() if cfg.device == "auto" else torch.device(cfg.device)
    run_optuna(cfg, dataset=args.dataset, device=device, smoke=args.smoke, verbose=True)


def cmd_results(args):
    """
    "results" command: rebuild and print the comparison-vs-MILLET table from whatever is finished.

    args : parsed arguments (unused).
    returns : nothing.
    """
    build_comparison(verbose=True)


def cmd_report(args):
    """
    "report" command: build all the paper-ready outputs from the finished results - the summary
    table, the SEA-Net vs MILLET comparison, and every figure - and save them under results/SEA_NET/.

    This is the same code the analysis notebook uses (seanet/report.py), so the figures are drawn in
    one place. Run it any time; it uses whatever has finished so far.

    args : parsed arguments (unused).
    returns : nothing.
    """
    from seanet.report import generate_report
    generate_report(verbose=True)


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

    # run (config-driven entry point)
    p = sub.add_parser("run", help="config-driven run: read configs/main.yaml and train + evaluate")
    p.add_argument("--config", default=os.path.join("configs", "main.yaml"), help="path to main.yaml")
    p.add_argument("--model", help="override the model in the config (e.g. seanet, millet)")
    p.add_argument("--dataset", help="override the dataset in the config (e.g. Coffee)")
    p.add_argument("--smoke", action="store_true", help="quick check (3 epochs), not saved")
    p.set_defaults(func=cmd_run)

    # interpret (per-sample explanation figures)
    p = sub.add_parser("interpret", help="train a model + draw per-sample interpretability figures")
    p.add_argument("--config", default=os.path.join("configs", "main.yaml"), help="path to main.yaml")
    p.add_argument("--model", help="override the model in the config (e.g. seanet, millet)")
    p.add_argument("--dataset", help="override the dataset to explain (e.g. WebTraffic)")
    p.add_argument("--smoke", action="store_true", help="quick check (3 epochs)")
    p.set_defaults(func=cmd_interpret)

    # optuna (hyperparameter search)
    p = sub.add_parser("optuna", help="run an Optuna hyperparameter search (reads the model's optuna block)")
    p.add_argument("--config", default=os.path.join("configs", "main.yaml"), help="path to main.yaml")
    p.add_argument("--model", help="which model to tune (e.g. seanet)")
    p.add_argument("--dataset", help="dataset to search on (default: WebTraffic)")
    p.add_argument("--smoke", action="store_true", help="quick check: 2 trials x 3 epochs, not saved")
    p.set_defaults(func=cmd_optuna)

    # results
    p = sub.add_parser("results", help="build + print the comparison vs MILLET")
    p.set_defaults(func=cmd_results)

    # report (all paper-ready tables + figures)
    p = sub.add_parser("report", help="generate the summary table + all figures under results/SEA_NET/")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    # save a timestamped copy of everything printed to results/SEA_NET/logs/<command>_<date-time>.log,
    # so every command (smoke, single, train, run, interpret, optuna, ...) leaves a permanent record.
    log_path = start_logging(args.command)
    try:
        args.func(args)      # call the function set by set_defaults for the chosen command
    except Exception:        # make sure a crash's traceback is also written to the log file
        import traceback
        traceback.print_exc(file=sys.stdout)   # stdout is teed to the log; stderr is not
        raise
    finally:
        print(f"\n[log] finished; full output saved to {log_path}")


if __name__ == "__main__":
    main()
