"""
main.py - the one place you run everything from.

What this file is for:
    This is the command-line entry point for the whole project. You do not import the seanet
    modules yourself; you run "python main.py <command>" and this file calls the right functions.
    It parses the command, sets up the working directory, and prints the results.

Commands (run "python main.py -h" to see them):
    python main.py summary [NAME|--all]     look at the data (shapes + a summary row per dataset)
    python main.py params                   print how much smaller SEA-Net is than the baseline
    python main.py train [--model M]        the full sweep for ONE model (resumable)
    python main.py single NAME [--model M]  train + evaluate one dataset
    python main.py webtraffic [--model M]   train on WebTraffic and compare to MILLET
    python main.py run [--model M]          config-driven single run (reads configs/main.yaml)
    python main.py interpret [--model M]    per-sample explanation figures (WebTraffic)
    python main.py optuna [--model M]       hyperparameter search
    python main.py results [--model M]      rebuild the comparison vs MILLET (all models, or one)
    python main.py report                   every figure + summary table

What "--model" means:
    It is the name of a config file under configs/models/, WITHOUT the .yaml - e.g. "--model seanet"
    reads configs/models/seanet.yaml. That file says which encoder and which pooling head to use.

One config file = one results folder, with a UNIQUE name:
    A model's folder is named "<config file name>__<encoder>_<pooling>". So configs/models/seanet.yaml
    (encoder mstcn_sep + pooling additive) writes everything into
    results/SEA_NET/seanet__mstcn_sep_additive/ - its results.csv, its done_train_dataset.txt, its
    logs, its figures, its interpretation figures. The config file name is in front so two configs
    that build the same encoder+pooling (e.g. seanet_slim vs seanet_classwise) never share a folder.
    Nothing is shared, so you can sweep several models and compare them fairly afterwards with
    "python main.py results" / "python main.py report".

How resuming works:
    Each model has its own results/SEA_NET/<model>/done_train_dataset.txt listing the datasets it
    has finished. "train" skips anything in that list, so it is safe to stop with Ctrl+C and start
    again. To retrain: delete the whole file (all datasets), delete one name from it (just that
    dataset), or set run.re_train: true in configs/main.yaml. A retrained dataset only OVERWRITES
    its old row in results.csv if the new run beats the old accuracy (save_result_row keeps the
    better result), so the table always holds the best numbers we have seen.

Related files:
    - seanet/data.py    -> loading, summaries (used by "summary").
    - seanet/model.py   -> the model + size helpers (used by "params").
    - seanet/train.py   -> train_one_from_config() + get_device() (the one training path).
    - seanet/results.py -> saving results + resume + the comparison (used by "train"/"results").
    - seanet/report.py  -> the figures and the summary tables.
    - analysis.ipynb    -> a thin notebook that calls seanet/report.py.

The training commands (train / single / webtraffic / run without --smoke) really train models, so
you run them yourself.
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
from seanet.config import load_config, to_flat_dict, param_choice_message, record_metrics, model_folder_name
from seanet.logs import start_logging
from seanet.model import make_sea_net, make_baseline, num_params, state_dict_size_mb
from seanet.train import train_one_from_config, fit_model_from_config, score_model, get_device
from seanet.results import (result_exists, save_result_row, build_comparison, compare_models,
                            millet_baseline, sweep_order, summarise_model, write_summary,
                            results_csv, done_txt, interpretation_dir, model_dir,
                            MILLET_WEBTRAFFIC_DIR)

# The commands that train (or tune) a model. They all resolve a config + a model id up front, so
# their log file can be saved inside that model's own folder. Everything else logs to the shared folder.
MODEL_COMMANDS = {"train", "single", "webtraffic", "run", "interpret", "optuna"}


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

    row : a results-row dict from train_one_from_config.
    returns : nothing.
    """
    for k in ["params", "model_size_mb", "n_train", "n_val", "n_test", "series_length",
              "n_classes", "test_acc", "test_loss", "test_aopcr", "test_ndcg", "train_time_s"]:
        print(f"  {k:16s}: {row[k]}")


def _resolve_model(args):
    """
    Load the config a command will run with, and work out the model id it writes under.

    Every model-specific command resolves its model exactly the same way, so that reading lives here
    once: the --config file, the --model override, and the "<config>__<encoder>_<pooling>" id that
    names the results folder. main() calls this BEFORE start_logging, so the log lands right.

    args : the parsed command-line arguments.
    returns : (cfg, model_id).
    """
    config_path = getattr(args, "config", None) or os.path.join("configs", "main.yaml")
    overrides = {"model": args.model} if getattr(args, "model", None) else None
    cfg = load_config(config_path, overrides=overrides)
    return cfg, model_folder_name(cfg)


def _run_context(args):
    """
    Unpack what a training command needs: the config, the model id, the device and the smoke flag.

    The config and model id were already resolved by main() (so logging could start in the right
    folder); this just reads them back off args and adds the device + smoke decision.

    args : the parsed command-line arguments.
    returns : (cfg, model_id, device, smoke).
    """
    cfg, model_id = args._cfg, args._model_id
    device = get_device() if cfg.device == "auto" else torch.device(cfg.device)
    smoke = bool(getattr(args, "smoke", False)) or bool(getattr(cfg.run, "smoke", False))
    return cfg, model_id, device, smoke


def _skip_if_done(cfg, model_id, name, smoke):
    """
    Decide whether to skip a dataset that is already finished for this model, and print why.

    Normally we skip a dataset that is already in the model's done list (so a run is not wasted redoing
    it). But main.yaml has a "run.re_train" switch: when it is true we train again anyway, and
    save_result_row REPLACES the old row - so you can re-check a model without hand-editing the done
    file. Smoke runs are never saved, so they always train and never count as "done".

    cfg : the loaded config.  model_id : the model folder id.  name : dataset name.
    smoke : True = a throwaway check (never skips).
    returns : True if the caller should skip this dataset (and has already printed the reason).
    """
    if smoke or not result_exists(model_id, name):              # nothing done yet -> just train
        return False
    re_train = bool(getattr(cfg.run, "re_train", False))        # the main.yaml switch
    if re_train:                                                 # train again, and say so
        print(f"\n{name} already done for {model_id}, but run.re_train is true -> training again "
              f"(its row in results.csv will be replaced).")
        return False
    print(f"\n{name} already done for {model_id} -> skip\n"
          f"  (set 're_train: true' in configs/main.yaml to train it again, or delete the line\n"
          f"   '{name}' from {done_txt(model_id)}.)")
    return True


def _start_mlflow(cfg, model_id, smoke):
    """
    Start MLflow for a run (unless it is a smoke check), and say whether to save model weights.

    Smoke runs are throwaway 3-epoch checks, so they are never tracked - the MLflow page only ever
    holds real results.

    cfg : the loaded config.  model_id : the model id (used as the versioned-model name).
    smoke : True = do not track.
    returns : (mlflow handle or None, log_model_weights flag).
    """
    mlf = None if smoke else tracking.start_experiment(cfg, model=model_id)
    log_weights = getattr(getattr(cfg, "mlflow", None), "log_model_weights", True)
    return mlf, log_weights


def _train_and_save(name, cfg, model_id, device, smoke, command, mlf, log_weights, verbose=True):
    """
    Train one dataset through the config path, stamp the row with its encoder/pooling, and save it.

    This is the single place a results row is produced and written, so every command records the
    same fields in the same model folder.

    name : dataset name.
    cfg : the loaded config. model_id : which results folder to write into.
    device : where to train. smoke : quick 3-epoch check (never saved).
    command : the main.py command (an MLflow tag, so runs can be filtered by how they were started).
    mlf : the mlflow handle (None = no logging). log_weights : also save the trained weights.
    verbose : print the training stages.
    returns : the results row dict.
    """
    row = train_one_from_config(
        name, cfg, device=device, smoke=smoke, verbose=verbose,
        mlf=mlf, mlf_params=to_flat_dict(cfg),
        mlf_tags={"command": command, "model_id": model_id},
        logged_model_name=model_id, log_model_weights=log_weights,
    )
    # the two things that define this model, written into every row so results.csv is self-describing
    row["encoder"] = cfg.model_config.encoder.type
    row["pooling"] = cfg.model_config.pooling.type
    if not smoke:                                            # smoke runs are throwaway, never saved
        saved = save_result_row(model_id, row)
        if not saved:                                       # keep-the-better rule: old row was higher
            print(f"  kept the previous result for {name}: this run's test_acc "
                  f"{row['test_acc']:.4f} did not beat it, so results.csv was left unchanged.")
    return row


def _smoke_note():
    """The one-line reminder printed after any --smoke run (so it is worded the same everywhere)."""
    return "\n  (smoke = 3 epochs; correctness only, NOT a result - nothing was saved)"


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


def cmd_train(args):
    """
    "train" command: the full sweep for ONE model.

    It trains, in MILLET's order, WebTraffic then the 85 datasets MILLET published then the rest of
    UCR (see seanet.results.sweep_order - the 85 come first so the head-to-head comparison is ready
    early). It is resumable (skips datasets already in this model's done_train_dataset.txt) and
    fault-tolerant (a failure on one dataset is logged and the loop keeps going).

    args : parsed arguments (args.model, args.only, args.limit, args.no_webtraffic, args.smoke).
    returns : nothing.
    """
    cfg, model_id, device, smoke = _run_context(args)

    # decide which datasets to run, and in what order
    if args.only:
        names = args.only                                    # explicit list from the user
    else:
        D.discover_ucr_datasets()                            # check the archive first
        names = sweep_order(include_webtraffic=not args.no_webtraffic)
        if args.limit is not None:
            names = names[: args.limit]                      # only the first N of the sweep order

    total = len(names)
    print(f"=== train: model={model_id} (config {cfg.model}) device={device} "
          f"mode={'smoke' if smoke else 'full'} datasets={total} ===")
    print(param_choice_message(cfg))
    print(f"  results  -> {results_csv(model_id)}")
    print(f"  resume   -> {done_txt(model_id)}  (delete it to retrain everything; "
          f"delete one line to retrain that dataset)\n")

    mlf, log_weights = _start_mlflow(cfg, model_id, smoke)
    done = skipped = failed = 0                              # running tally for the summary line
    for i, name in enumerate(names, 1):
        tag = f"[{i:>3}/{total}]"                             # e.g. "[ 22/129]"
        if result_exists(model_id, name) and not smoke:      # already finished -> skip
            print(f"{tag} {name:28s} already done -> skip", flush=True)
            skipped += 1
            continue
        try:
            if not D.summary_row_exists(name):               # make sure its data_summary row exists
                D.write_summary_row(D.summarise_dataset(name))
            print(f"{tag} {name}", flush=True)               # header; the stage lines below belong to it
            row = _train_and_save(name, cfg, model_id, device, smoke, "train", mlf, log_weights)
            ndcg = "n/a" if row["test_ndcg"] is None else f"{row['test_ndcg']:.3f}"
            print(f"{tag} {name:28s} DONE  acc={row['test_acc']:.4f} loss={row['test_loss']:.4f} "
                  f"AOPCR={row['test_aopcr']:7.3f} NDCG={ndcg:>5s} C={row['n_classes']:>3d} "
                  f"T={row['series_length']:>5d} ({row['train_time_s']:.0f}s)  "
                  f"[{done + 1} trained / {failed} failed so far]", flush=True)
            done += 1
        except KeyboardInterrupt:                            # Ctrl+C -> stop cleanly (progress is saved)
            print("\nInterrupted - progress is saved, re-run the same command to resume.", flush=True)
            raise
        except Exception as e:                               # any other error -> log it and continue
            print(f"{tag} {name:28s} FAILED: {type(e).__name__}: {e}", flush=True)
            failed += 1

    print(f"\nDone: {done} trained, {skipped} skipped, {failed} failed.", flush=True)
    if smoke:
        print(_smoke_note())
        return
    print(f"Results -> {results_csv(model_id)}\n", flush=True)
    build_comparison(model_id, verbose=True)                 # the win/tie/loss summary vs MILLET
    write_summary(model_id)                                  # refresh this model's summary.csv/.md
    print(f"\nNext: `python main.py report` to draw the figures for {model_id}.")


def cmd_single(args):
    """
    "single" command: train + evaluate one dataset and save its result.

    args : parsed arguments (args.dataset, args.model, args.config, args.smoke).
    returns : nothing.
    """
    cfg, model_id, device, smoke = _run_context(args)
    name = args.dataset
    if _skip_if_done(cfg, model_id, name, smoke):            # already finished (and re_train is off)
        return

    print(f"=== single: model={model_id} (config {cfg.model}) dataset={name} device={device} "
          f"mode={'smoke' if smoke else 'full'} ===")
    print(param_choice_message(cfg))
    mlf, log_weights = _start_mlflow(cfg, model_id, smoke)
    row = _train_and_save(name, cfg, model_id, device, smoke, "single", mlf, log_weights)

    print(f"\n=== {model_id} on {name} ===")
    print_row(row)
    base = millet_baseline("test_acc.csv")                   # show MILLET's accuracy if this is one of the 85
    if name in base.index:
        print(f"\n  MILLET acc baseline for {name}: {base[name]:.4f}  (ours {row['test_acc']:.4f})")
    print(_smoke_note() if smoke else f"\n  saved -> {results_csv(model_id)}")


def cmd_webtraffic(args):
    """
    "webtraffic" command: train on WebTraffic and compare to MILLET.

    This is our main sanity check, because WebTraffic is the only dataset with per-timestep ground
    truth - so it is the only one where NDCG (did we point at the RIGHT timesteps?) can be measured.

    args : parsed arguments (args.model, args.config, args.smoke).
    returns : nothing.
    """
    cfg, model_id, device, smoke = _run_context(args)
    print(f"=== webtraffic: model={model_id} (config {cfg.model}) device={device} "
          f"mode={'smoke' if smoke else 'full'} ===")
    print(param_choice_message(cfg))
    mlf, log_weights = _start_mlflow(cfg, model_id, smoke)
    row = _train_and_save(D.WEB_TRAFFIC, cfg, model_id, device, smoke, "webtraffic", mlf, log_weights)

    print(f"\n=== {model_id} on WebTraffic ===")
    print_row(row)
    # print our model next to MILLET for the 4 metrics (WebTraffic has its own baseline folder)
    print("\n=== vs MILLET ConjunctiveInceptionTime (mean of 5 reps) ===")
    print(f"  {'metric':8s} {'ours':>10s} {'MILLET':>10s}")
    for label, metric_csv, ours in [("acc", "test_acc.csv", row["test_acc"]),
                                    ("loss", "test_loss.csv", row["test_loss"]),
                                    ("AOPCR", "test_aopcr.csv", row["test_aopcr"]),
                                    ("NDCG@n", "test_ndcg.csv", row["test_ndcg"])]:
        theirs = float(millet_baseline(metric_csv, directory=MILLET_WEBTRAFFIC_DIR)["WebTraffic"])
        ours_txt = "n/a" if ours is None else f"{ours:.4f}"
        print(f"  {label:8s} {ours_txt:>10s} {theirs:>10.4f}")
    print(_smoke_note() if smoke else f"\n  saved -> {results_csv(model_id)}")


def cmd_run(args):
    """
    "run" command: the config-driven entry point. It reads configs/main.yaml (plus the model file it
    points at), then trains + evaluates the chosen dataset with the chosen model, using only values
    from the config. Command-line flags (--model, --dataset, --smoke) override the file so you can
    try things quickly without editing YAML.

    args : parsed arguments (args.config, args.model, args.dataset, args.smoke).
    returns : nothing.
    """
    cfg, model_id, device, smoke = _run_context(args)
    dataset = args.dataset or cfg.run.dataset               # --dataset overrides the config

    # show exactly what config is driving this run (reproducibility + a quick sanity check)
    print(f"=== run: model={model_id} (config {cfg.model}) dataset={dataset} device={device} "
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

    if _skip_if_done(cfg, model_id, dataset, smoke):            # already finished (and re_train is off)
        return

    mlf, log_weights = _start_mlflow(cfg, model_id, smoke)
    row = _train_and_save(dataset, cfg, model_id, device, smoke, "run", mlf, log_weights)
    print(f"\n=== {model_id} on {dataset} ===")
    print_row(row)

    if smoke:
        print(_smoke_note())
        return
    print(f"\n  saved -> {results_csv(model_id)}")
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


def cmd_interpret(args):
    """
    "interpret" command: train a model and draw MILLET-style per-sample explanation figures.

    It draws WebTraffic by default, and for a reason: WebTraffic is the only dataset that ships
    per-timestep ground truth, so it is the only one where the figure can shade the truly important
    region and let you CHECK that the model highlighted the right points. It draws one figure for
    each of the first few classes, using a test series the model predicted CORRECTLY.

    The figures go into this model's own folder:
        results/SEA_NET/<model_id>/interpretation/<dataset>/<date-time>/

    args : parsed arguments (args.config, args.model, args.dataset, args.smoke).
    returns : nothing.
    """
    from seanet.interpretability import generate_interpretations   # imported here so matplotlib only loads for this command

    cfg, model_id, device, smoke = _run_context(args)
    icfg = getattr(cfg, "interpretability", None)

    # figure out the settings (command line overrides the config, config overrides the defaults)
    dataset = args.dataset or (getattr(icfg, "dataset", None) if icfg else None) or D.WEB_TRAFFIC
    n_classes = getattr(icfg, "n_classes", 3) if icfg else 3
    per_class = getattr(icfg, "per_class", 1) if icfg else 1

    print(f"=== interpret: model={model_id} (config {cfg.model}) dataset={dataset} device={device} "
          f"mode={'smoke' if smoke else 'full'} ===")
    if dataset != D.WEB_TRAFFIC:
        # be honest about what the picture can and cannot show on a UCR dataset
        print(f"  NOTE: {dataset} has no per-timestep ground truth, so the figures cannot show the\n"
              f"        true important region and you cannot check the highlights against it.\n"
              f"        WebTraffic is the dataset to trust here.")

    # train a model to explain (same fit path as training), then draw the figures from it
    model, _, _, test_ds, _ = fit_model_from_config(dataset, cfg, device=device, smoke=smoke, verbose=True)
    print("    drawing figures ...", flush=True)

    base_dir = os.path.join(interpretation_dir(model_id), dataset)
    if smoke:
        # smoke = a quick check: draw ONE preview figure into a fixed file, so repeated smoke runs
        # do not pile up dozens of throwaway figures. Real (training) runs are the ones we keep.
        out_dir = os.path.join(base_dir, "smoke")
        paths = generate_interpretations(model, test_ds, out_dir, dataset_name=dataset,
                                         limit=1, fixed_name="smoke_preview.png")
    else:
        # real run: save into a fresh folder stamped with the date and time, so every run's
        # interpretation results are kept separately and are easy to find later.
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(base_dir, stamp)
        paths = generate_interpretations(model, test_ds, out_dir, n_classes=n_classes,
                                         per_class=per_class, dataset_name=dataset)

    print(f"\nsaved {len(paths)} figure(s) to {out_dir}/")
    for p in paths:
        print("  ", p)
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
    trials on the dataset (WebTraffic by default), and - unless smoke - records the best values in
    that same model file, under its "records" block.

    args : parsed arguments (args.config, args.model, args.dataset, args.smoke).
    returns : nothing.
    """
    from seanet.optimize import run_optuna

    cfg, model_id, device, smoke = _run_context(args)
    print(f"=== optuna: model={model_id} (config {cfg.model}) device={device} ===")
    run_optuna(cfg, dataset=args.dataset, device=device, smoke=smoke, verbose=True)


def cmd_results(args):
    """
    "results" command: rebuild the comparison tables from whatever has finished.

    With --model it reports that one model: its comparison_vs_millet.csv plus the means over the 85
    datasets MILLET published. Without --model it does that for EVERY model that has results and
    also writes the cross-model ranking, results/SEA_NET/model_comparison.csv - the "which
    encoder+pooling wins" table.

    args : parsed arguments (args.model, args.config).
    returns : nothing.
    """
    if args.model:                                            # one model
        cfg, model_id = _resolve_model(args)
        if not os.path.exists(results_csv(model_id)):         # never trained -> do not make an empty folder
            print(f"{model_id} (config {cfg.model}) has no results yet.\n"
                  f"  Train it first: python main.py train --model {cfg.model}")
            return
        build_comparison(model_id, verbose=True)
        perf = summarise_model(model_id)
        write_summary(model_id, perf)
        print(f"\n=== {model_id} summary ===")
        for key, value in perf.items():
            print(f"  {key:24s}: {value}")
        print(f"\n  wrote {model_dir(model_id)}/summary.csv and summary.md")
        return
    compare_models(verbose=True)                              # every model + the cross-model ranking


def cmd_report(args):
    """
    "report" command: build every figure and summary table from the finished results.

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
def _add_model_flags(p, with_dataset=False):
    """
    Add the flags every model-specific command shares, so they are worded the same everywhere.

    p : the sub-parser to add them to.
    with_dataset : also add --dataset.
    returns : nothing.
    """
    p.add_argument("--config", default=os.path.join("configs", "main.yaml"), help="path to main.yaml")
    p.add_argument("--model", help="model config under configs/models/ without .yaml "
                                   "(e.g. seanet, seanet_acp, millet)")
    if with_dataset:
        p.add_argument("--dataset", help="override the dataset (e.g. Coffee)")
    p.add_argument("--smoke", action="store_true", help="quick check (3 epochs), not saved")


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

    # train (the full sweep, for one model)
    p = sub.add_parser("train", help="full sweep for ONE model: WebTraffic + all 128 UCR (resumable)")
    _add_model_flags(p)
    p.add_argument("--only", nargs="+", metavar="NAME", help="only these datasets")
    p.add_argument("--limit", type=int, help="only the first N datasets of the sweep order")
    p.add_argument("--no-webtraffic", action="store_true", help="UCR only")
    p.set_defaults(func=cmd_train)

    # single
    p = sub.add_parser("single", help="train + evaluate one dataset, save its result")
    p.add_argument("dataset", help="dataset name, e.g. Coffee")
    _add_model_flags(p)
    p.set_defaults(func=cmd_single)

    # webtraffic
    p = sub.add_parser("webtraffic", help="train on WebTraffic + sanity-check vs MILLET")
    _add_model_flags(p)
    p.set_defaults(func=cmd_webtraffic)

    # run (config-driven entry point)
    p = sub.add_parser("run", help="config-driven run: read configs/main.yaml and train + evaluate")
    _add_model_flags(p, with_dataset=True)
    p.set_defaults(func=cmd_run)

    # interpret (per-sample explanation figures)
    p = sub.add_parser("interpret", help="train a model + draw per-sample explanation figures (WebTraffic)")
    _add_model_flags(p, with_dataset=True)
    p.set_defaults(func=cmd_interpret)

    # optuna (hyperparameter search)
    p = sub.add_parser("optuna", help="run an Optuna hyperparameter search (reads the model's optuna block)")
    _add_model_flags(p, with_dataset=True)
    p.set_defaults(func=cmd_optuna)

    # results
    p = sub.add_parser("results", help="build the comparison vs MILLET (all models, or one with --model)")
    p.add_argument("--config", default=os.path.join("configs", "main.yaml"), help="path to main.yaml")
    p.add_argument("--model", help="report only this model config (default: every model with results)")
    p.set_defaults(func=cmd_results)

    # report (every figure + summary table)
    p = sub.add_parser("report", help="generate every figure + summary table under results/SEA_NET/")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()

    # Model-specific commands resolve their config FIRST, so their log file can be written inside
    # that model's own folder (results/SEA_NET/<model_id>/logs/). Everything else logs to the
    # shared results/SEA_NET/logs/ folder.
    model_id = None
    if args.command in MODEL_COMMANDS:
        args._cfg, args._model_id = _resolve_model(args)
        model_id = args._model_id

    # save a timestamped copy of everything printed, so every run leaves a permanent record.
    # Smoke runs go to logs/smoke/ (git-ignored) so only real training logs are committed.
    log_path = start_logging(args.command, model_id=model_id, smoke=getattr(args, "smoke", False))
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
