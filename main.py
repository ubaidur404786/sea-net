"""
main.py - the one place you run everything from.

What this file is for:
    This is the command-line entry point for the whole project. You do not import the seanet
    modules yourself; you run "python main.py <command>" and this file calls the right functions.
    It parses the command, sets up the working directory, and prints the results.

Commands (run "python main.py -h" to see them):
    python main.py summary [NAME|--all]   look at the data (shapes + a summary row per dataset)
    python main.py params                 print how much smaller SEA-Net is than the baseline
    python main.py webtraffic [--model M] train on WebTraffic and compare to MILLET
    python main.py single NAME [--model M] train + evaluate one dataset
    python main.py train [--model M]      the full sweep: WebTraffic + all 128 UCR, for ONE model
    python main.py run                    config-driven single run (reads configs/main.yaml)
    python main.py benchmark              compare pooling heads on the same encoder
    python main.py interpret              per-sample explanation figures
    python main.py optuna [--model M]     hyperparameter search
    python main.py results [--model M]    rebuild best_results.csv + the comparison vs MILLET
    python main.py report                 all figures + summary + refresh the README results section

Related files:
    - seanet/data.py    -> loading, summaries (used by "summary").
    - seanet/model.py   -> the model + size helpers (used by "params").
    - seanet/train.py   -> train_one_from_config() + get_device() (the one training path).
    - seanet/results.py -> saving results + resume + the comparison (used by "train", "single", "results").
    - seanet/report.py  -> the figures, the summary table, and the README results section.
    - analysis.ipynb    -> a thin notebook that calls seanet/report.py.

The training commands (webtraffic / single / train / run without --smoke) really train models, so
you run them yourself.

How resuming works: a finished run is remembered as "model|settings|dataset", where "settings" is a
short fingerprint of the encoder/pooling/training values (see seanet/config.settings_fingerprint).
So it is safe to stop "train" with Ctrl+C and start again - it picks up where it left off. Every
model sweeps the whole archive on its own (running "seanet" does not mark the datasets done for
"seanet_acp"), and changing any hyperparameter changes the fingerprint, which retrains that model
across all datasets instead of silently mixing old and new numbers into one table.
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
from seanet.config import (load_config, to_flat_dict, param_choice_message, record_metrics,
                           settings_fingerprint)
from seanet.logs import start_logging
from seanet.model import make_sea_net, make_baseline, num_params, state_dict_size_mb
from seanet.train import (train_one_from_config, fit_model_from_config, score_model, get_device)
from seanet.results import (result_exists, save_result_row, build_comparison, build_best_results,
                            millet_baseline, new_run_dir, sweep_status, RESULTS_CSV, BEST_CSV)


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


def _load_run_config(args, command):
    """
    Work out everything a training command needs from the config + the command line, in one place.

    Every training command (run / single / webtraffic / train) resolves its settings the same way,
    so they all share this: the config file, the --model / --dataset / --smoke overrides, the device,
    and - importantly - the settings fingerprint that identifies this exact recipe. The fingerprint
    is what makes a sweep resumable per model: see seanet/results.py.

    args : the parsed command-line arguments.
    command : the main.py command name (used for the MLflow tag + the run folder name).
    returns : (cfg, device, smoke, fingerprint).
    """
    config_path = getattr(args, "config", None) or os.path.join("configs", "main.yaml")
    overrides = {"model": args.model} if getattr(args, "model", None) else None
    cfg = load_config(config_path, overrides=overrides)
    device = get_device() if cfg.device == "auto" else torch.device(cfg.device)
    smoke = bool(getattr(args, "smoke", False)) or bool(getattr(cfg.run, "smoke", False))
    fingerprint = settings_fingerprint(cfg)
    return cfg, device, smoke, fingerprint


def _train_and_save(name, cfg, device, smoke, fingerprint, command, run_dir, mlf, log_weights,
                    verbose=True):
    """
    Train one dataset through the config path, stamp the row with its recipe, and save it.

    This is the single place a results row is produced and written, so every command records the
    same fields: which model, which settings fingerprint, and (via save_result_row) when it ran.

    name : dataset name.
    cfg : the loaded config. device : where to train. smoke : quick 3-epoch check (never saved).
    fingerprint : the settings id from _load_run_config.
    command : the main.py command (an MLflow tag, so runs can be filtered by how they were started).
    run_dir : this invocation's datetime folder (from new_run_dir), or None.
    mlf : the mlflow handle (None = no logging). log_weights : also save the trained weights.
    verbose : print the training stages.
    returns : the results row dict.
    """
    row = train_one_from_config(
        name, cfg, device=device, smoke=smoke, verbose=verbose,
        mlf=mlf, mlf_params=to_flat_dict(cfg),
        mlf_tags={"command": command, "settings": fingerprint},
        logged_model_name=cfg.model, log_model_weights=log_weights,
    )
    row["settings"] = fingerprint                            # the recipe that produced this row
    if not smoke:                                            # smoke runs are throwaway, never saved
        save_result_row(row, run_dir=run_dir)
    return row


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

    args : parsed arguments (args.model, args.config, args.smoke).
    returns : nothing.
    """
    cfg, device, smoke, fingerprint = _load_run_config(args, "webtraffic")
    print(f"=== webtraffic: model={cfg.model} settings={fingerprint} device={device} "
          f"mode={'smoke' if smoke else 'full'} ===")
    print(param_choice_message(cfg))
    run_dir = None if smoke else new_run_dir("webtraffic", cfg.model)
    mlf = None if smoke else tracking.start_experiment(cfg, model=cfg.model)
    log_weights = getattr(getattr(cfg, "mlflow", None), "log_model_weights", True)
    row = _train_and_save("WebTraffic", cfg, device, smoke, fingerprint, "webtraffic", run_dir,
                          mlf, log_weights, verbose=True)
    print(f"\n=== {cfg.model} on WebTraffic ===")
    print_row(row)
    # print our model next to MILLET for the 3 metrics
    print("\n=== vs MILLET ConjunctiveInceptionTime (mean of 5 reps) ===")
    print(f"  {'metric':8s} {cfg.model:>10s} {'MILLET':>10s}")
    print(f"  {'acc':8s} {row['test_acc']:>10.4f} {webtraffic_millet('test_acc.csv'):>10.4f}")
    print(f"  {'AOPCR':8s} {row['test_aopcr']:>10.4f} {webtraffic_millet('test_aopcr.csv'):>10.4f}")
    print(f"  {'NDCG@n':8s} {str(row['test_ndcg']):>10s} {webtraffic_millet('test_ndcg.csv'):>10.4f}")
    if smoke:                                                 # smoke numbers are not real results
        print("\n  (smoke = 3 epochs; correctness only, NOT a result - not saved)")
    else:
        print(f"\n  saved -> {RESULTS_CSV}  (run folder: {run_dir})")


def cmd_single(args):
    """
    "single" command: train + evaluate one dataset and save its result.

    args : parsed arguments (args.dataset, args.model, args.config, args.smoke).
    returns : nothing.
    """
    cfg, device, smoke, fingerprint = _load_run_config(args, "single")
    name = args.dataset
    print(f"=== single: model={cfg.model} dataset={name} settings={fingerprint} device={device} "
          f"mode={'smoke' if smoke else 'full'} ===")
    print(param_choice_message(cfg))
    if result_exists(name, cfg.model, fingerprint) and not smoke:   # this model+settings already did it
        print(f"{name} already done for {cfg.model}|{fingerprint} -> skip "
              f"(delete its line in results/SEA_NET/done.txt to redo)")
        return
    run_dir = None if smoke else new_run_dir("single", cfg.model)
    mlf = None if smoke else tracking.start_experiment(cfg, model=cfg.model)
    log_weights = getattr(getattr(cfg, "mlflow", None), "log_model_weights", True)
    row = _train_and_save(name, cfg, device, smoke, fingerprint, "single", run_dir,
                          mlf, log_weights, verbose=True)
    print(f"\n=== {cfg.model} on {name} ===")
    print_row(row)
    base = millet_baseline("test_acc.csv")                   # show the MILLET accuracy if this is one of the 85
    if name in base.index:
        print(f"\n  MILLET acc baseline for {name}: {base[name]:.4f}  (ours {row['test_acc']:.4f})")
    if smoke:
        print("\n  (smoke = 3 epochs; correctness only, NOT a result - not saved)")
    else:
        print(f"\n  saved -> {RESULTS_CSV}  (run folder: {run_dir})")


def cmd_train(args):
    """
    "train" command: the full sweep for ONE model. Trains WebTraffic first, then all 128 UCR
    datasets, saving each result. It is resumable and fault-tolerant (a failure on one dataset is
    logged and the loop keeps going).

    Which model it sweeps comes from the config (--model overrides it), so every model can be swept
    over the whole archive: `python main.py train --model seanet_acp` trains that model on all 129,
    even if `seanet` already did them. Resuming is per model AND per settings: if you change a
    hyperparameter, the fingerprint changes and this model retrains everything under the new id;
    if you do not, it picks up exactly where it stopped.

    args : parsed arguments (args.model, args.config, args.only, args.limit, args.no_webtraffic,
           args.smoke).
    returns : nothing.
    """
    cfg, device, smoke, fingerprint = _load_run_config(args, "train")
    model_name = cfg.model

    # decide which datasets to run
    if args.only:
        names = args.only                                    # explicit list from the user
    else:
        D.discover_ucr_datasets()                            # check the archive first
        ucr = D.UCR_128_DATASETS[: args.limit] if args.limit is not None else D.UCR_128_DATASETS
        names = ([] if args.no_webtraffic else [D.WEB_TRAFFIC]) + ucr   # WebTraffic first, then UCR

    total = len(names)
    print(f"=== train: model={model_name} settings={fingerprint} device={device} "
          f"mode={'smoke' if smoke else 'full'} datasets={total} ===", flush=True)
    print(param_choice_message(cfg), flush=True)

    # say up front how much of this sweep is already finished, so a long resume is never a mystery
    status = sweep_status(model_name, fingerprint, names)
    if status["done"]:
        print(f"  resuming: {len(status['done'])} already done for {model_name}|{fingerprint}, "
              f"{len(status['todo'])} to go", flush=True)
    else:
        print(f"  fresh sweep for {model_name}|{fingerprint}: nothing done yet for these settings",
              flush=True)

    run_dir = None if smoke else new_run_dir("train", model_name)   # this invocation's own folder
    if run_dir:
        print(f"  this run's rows -> {run_dir}/results.csv", flush=True)
    mlf = None if smoke else tracking.start_experiment(cfg, model=model_name)
    log_weights = getattr(getattr(cfg, "mlflow", None), "log_model_weights", True)

    done = skipped = failed = 0                              # running tally for the summary line
    for i, name in enumerate(names, 1):
        tag = f"[{i:>3}/{total}]"                             # e.g. "[ 22/129]"
        if result_exists(name, model_name, fingerprint):     # this model+settings already did it
            print(f"{tag} {name:28s} already done -> skip", flush=True)
            skipped += 1
            continue
        try:
            if not D.summary_row_exists(name):               # make sure its data_summary row exists
                D.write_summary_row(D.summarise_dataset(name))
            print(f"{tag} {name}", flush=True)               # header; the stage lines below belong to it
            row = _train_and_save(name, cfg, device, smoke, fingerprint, "train", run_dir,
                                  mlf, log_weights, verbose=True)
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
    if not smoke:
        build_best_results(verbose=True)                     # refresh "which model won which dataset"
    print(f"\n=== Comparison vs MILLET (model {model_name}) ===", flush=True)
    build_comparison(verbose=True, model=model_name)         # this model's own win/tie/loss summary


def cmd_run(args):
    """
    "run" command: the new config-driven entry point. It reads configs/main.yaml (plus the model
    file it points at), then trains + evaluates the chosen dataset with the chosen model, using
    only values from the config. Command-line flags (--model, --dataset, --smoke) override the
    file so you can try things quickly without editing YAML.

    args : parsed arguments (args.config, args.model, args.dataset, args.smoke).
    returns : nothing.
    """
    cfg, device, smoke, fingerprint = _load_run_config(args, "run")
    dataset = args.dataset or cfg.run.dataset               # --dataset overrides the config

    # show exactly what config is driving this run (reproducibility + a quick sanity check)
    print(f"=== run: model={cfg.model} dataset={dataset} settings={fingerprint} device={device} "
          f"mode={'smoke' if smoke else 'full'} ===")
    print("resolved config:")
    for key, value in to_flat_dict(cfg).items():
        print(f"  {key} = {value}")

    # say which recipe (default vs Optuna-best) this run uses, and how they compare (see use_params)
    msg = param_choice_message(cfg)
    if msg:
        print(msg)

    if cfg.run.mode != "single":                                # only "single" is wired up so far
        raise SystemExit(f"run mode {cfg.run.mode!r} is not supported yet (use mode: single).")

    if result_exists(dataset, cfg.model, fingerprint) and not smoke:   # this model+settings did it
        print(f"\n{dataset} already done for {cfg.model}|{fingerprint} -> skip "
              f"(delete its line in results/SEA_NET/done.txt to redo)")
        return

    # MLflow records this run so it can be compared with every other model/dataset later (skipped for
    # smoke, which is a throwaway 3-epoch check). This run's full terminal output is also saved to
    # results/SEA_NET/logs/ by start_logging in main().
    run_dir = None if smoke else new_run_dir("run", cfg.model)
    mlf = None if smoke else tracking.start_experiment(cfg, model=cfg.model)
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
    row["settings"] = fingerprint                               # the recipe that produced this row
    print(f"\n=== {cfg.model} on {dataset} ===")
    print_row(row)

    if smoke:
        print("\n  (smoke = 3 epochs; correctness only, NOT a result - not saved)")
    else:
        save_result_row(row, run_dir=run_dir)
        print(f"\n  saved -> {RESULTS_CSV}  (run folder: {run_dir})")
        # save THIS default recipe's metrics into the model file, so they can be compared with Optuna's
        # best later. Only when we actually trained the DEFAULT recipe (not an auto-picked optuna-best).
        choice = getattr(cfg, "_param_choice", None)
        if choice is None or getattr(choice, "used", "default") == "default":
            record_metrics(cfg.model, "default", {
                "test_acc": row["test_acc"], "test_loss": row["test_loss"],
                "test_auroc": row["test_auroc"], "dataset": dataset,
                "recorded": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            print(f"  recorded default metrics -> configs/models/{cfg.model}.yaml (records.default)")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def cmd_benchmark(args):
    """
    "benchmark" command: train the same encoder with several pooling heads on several datasets and
    print a ranking, so you can see which pooling wins (and by how much).

    It swaps ONLY the pooling head on the base model config (default seanet), so the encoder and
    training recipe are identical across variants - a fair A/B. Results go to their OWN file
    results/SEA_NET/pooling_benchmark.csv; it never touches results.csv / done.txt or the MILLET
    comparison. Each run is also logged to MLflow when enabled. --datasets / --pooling / --model
    override the defaults; --smoke does a quick 3-epoch check (not saved).

    args : parsed arguments (args.config, args.model, args.datasets, args.pooling, args.smoke).
    returns : nothing.
    """
    from seanet.benchmark import run_benchmark
    run_benchmark(datasets=args.datasets, poolings=args.pooling, base_model=args.model,
                  config_path=args.config, smoke=args.smoke, verbose=True)


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
    "results" command: rebuild the tracking tables from whatever is finished.

    Builds two things:
      - best_results.csv         : one row per dataset - which model won it, with which settings and
                                   on which date (the "what should I actually use" table), and
      - comparison_vs_millet.csv : our numbers vs the paper's. With --model it is a single model's
                                   honest table; without, it uses the best model per dataset and
                                   carries a "model" column saying which one that was.

    args : parsed arguments (args.model).
    returns : nothing.
    """
    best = build_best_results(verbose=True)
    if not best.empty:
        print(f"\n  best_results -> {BEST_CSV}")
    print()
    build_comparison(verbose=True, model=args.model)


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
    p.add_argument("--config", default=os.path.join("configs", "main.yaml"), help="path to main.yaml")
    p.add_argument("--model", help="override the model in the config (e.g. seanet, millet)")
    p.add_argument("--smoke", action="store_true", help="quick check (3 epochs)")
    p.set_defaults(func=cmd_webtraffic)

    # single
    p = sub.add_parser("single", help="train + evaluate one dataset, save its result")
    p.add_argument("dataset", help="dataset name, e.g. Coffee")
    p.add_argument("--config", default=os.path.join("configs", "main.yaml"), help="path to main.yaml")
    p.add_argument("--model", help="override the model in the config (e.g. seanet, millet)")
    p.add_argument("--smoke", action="store_true", help="quick check (3 epochs)")
    p.set_defaults(func=cmd_single)

    # train (the full sweep, for ONE model)
    p = sub.add_parser("train", help="full sweep for one model: WebTraffic + all 128 UCR (resumable)")
    p.add_argument("--config", default=os.path.join("configs", "main.yaml"), help="path to main.yaml")
    p.add_argument("--model", help="which model to sweep (e.g. seanet, seanet_acp); default: the config's")
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

    # benchmark (compare pooling heads on the same encoder)
    p = sub.add_parser("benchmark", help="compare pooling heads (same encoder) on several datasets + rank them")
    p.add_argument("--config", default=os.path.join("configs", "main.yaml"), help="path to main.yaml")
    p.add_argument("--model", default="seanet",
                   help="base model config whose pooling head is swapped (default: seanet)")
    p.add_argument("--datasets", nargs="+", metavar="NAME",
                   help="datasets to run (default: a small, fast set incl. WebTraffic)")
    p.add_argument("--pooling", nargs="+", metavar="TYPE",
                   help="pooling heads to compare (default: the 2 baselines + our 3 new heads)")
    p.add_argument("--smoke", action="store_true", help="quick check (3 epochs each), not saved")
    p.set_defaults(func=cmd_benchmark)

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
    p = sub.add_parser("results", help="build + print the comparison vs MILLET + best_results.csv")
    p.add_argument("--model", help="compare only this model (default: the best model per dataset)")
    p.set_defaults(func=cmd_results)

    # report (all paper-ready tables + figures)
    p = sub.add_parser("report", help="generate the summary table + all figures under results/SEA_NET/")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    # save a timestamped copy of everything printed to results/SEA_NET/logs/<command>_<date-time>.log,
    # so every command (smoke, single, train, run, interpret, optuna, ...) leaves a permanent record.
    # Smoke runs go to logs/smoke/ (git-ignored) so only real training logs are committed.
    log_path = start_logging(args.command, smoke=getattr(args, "smoke", False))
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
