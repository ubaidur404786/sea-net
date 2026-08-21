"""
main.py - the one entry point. Read this file to understand the whole project.

    python main.py -h                list every command
    python main.py single -h         list the flags of ONE command

THE PIPELINE, and the file that owns each step
----------------------------------------------
    configuration    configs/               ->  seanet/config.py
    data             data/                  ->  seanet/data.py
    preprocessing    normalise + split      ->  seanet/preprocessing.py
    encoder          (B,1,T) -> (B,d,T)     ->  seanet/models/encoders.py
    MIL pooling      (B,d,T) -> logits+map  ->  seanet/models/pooling.py
    build            encoder + pooling      ->  seanet/models/build.py
    training         the one training loop  ->  seanet/training.py
    evaluation       score -> one row       ->  seanet/evaluation.py
    metrics          acc / AOPCR / NDCG     ->  seanet/metrics.py
    results          save rows, leaderboard ->  seanet/results.py
    analysis         comparison figures     ->  seanet/analysis/
    tracking         MLflow                 ->  seanet/tracking.py
    optuna           OPTIONAL search        ->  seanet/optuna_search.py

This file only ORCHESTRATES: it parses the command line, loads the config, and calls those
components in order. No model maths, no training loop, no plotting lives here.

THE COMMANDS
------------
Cheap - they only read results that already exist, so they are safe to run any time:
    models                        list every model config you can pass to --model
    summary [NAME] [--all]        dataset stats: length, classes, train/test sizes
    params                        parameter counts: SEA-Net vs the baselines
    results [--model M]           rebuild each model's comparison table vs MILLET
    leaderboard [--fast]          one table of every model, best WebTraffic accuracy first
    analyse [--refresh]           all the comparison figures + tables -> results/analysis/
    report                        the per-model figures -> results/SEA_NET/<model>/figures/
    web-compare                   WebTraffic-only comparison + accuracy tiers + the winner

Expensive - these really train. Add --smoke to test the flow in 3 epochs instead:
    single NAME [--model M]       train + evaluate ONE dataset and save its row
    train [--model M]             the same model on EVERY dataset (WebTraffic + 128 UCR), resumable
    webtraffic [--model M]        train on WebTraffic and sanity-check against MILLET
    run [--model M]               whatever configs/main.yaml says
    interpret [--model M]         train, then draw the per-sample explanation figures
    optuna [--model M]            hyperparameter search (optional; same training pipeline)

"single" vs "train" - the ONLY difference is how many datasets:
    single NAME   one dataset, one row, seconds to minutes. Use this while testing.
    train         every dataset, one row each, hours to days. It remembers what it finished,
                  so Ctrl+C and restart later is safe. --limit 5 tries the first 5 only.

SHARED FLAGS (train / single / webtraffic / run / interpret / optuna)
    --model M      which config under configs/models/. The folder is optional:
                   "seanet/seanet_bottleneck_topk" and "seanet_bottleneck_topk" both work.
    --env NAME     which environment file (configs/environments/NAME.yaml). Default: local,
                   or whatever SEANET_ENV says. Use --env grid5000 on the cluster.
    --config PATH  path to main.yaml                      (default: configs/main.yaml)
    --seed N       training seed. A new seed ADDS a repeat row; it never overwrites seed 0.
    --smoke        3 epochs, nothing saved - use this when testing or debugging.
    --dataset NAME override the dataset (run / interpret / optuna; "single" takes it positionally)

EXAMPLES
    python main.py models                                          what can I run?
    python main.py summary Coffee                                  look at one dataset
    python main.py single Coffee --model seanet_bottleneck_topk --smoke      3-epoch flow check
    python main.py single Coffee --model seanet_bottleneck_topk              the real run
    python main.py train --model baselines/millet --env grid5000             a full sweep
    python main.py analyse                                         rebuild every comparison figure

WHERE THE OUTPUT GOES
    results/SEA_NET/<model_id>/    one folder per model: results.csv, history/, predictions/,
                                   figures/, interpretation/, logs/, done_train_dataset.txt
    results/analysis/              the cross-model comparison figures and tables + INDEX.md
    mlflow.db                      every run, comparable in the MLflow web page

    <model_id> is "<config name>__<encoder>__<pooling>", so two configs that happen to build the
    same encoder+pooling never share a folder and can never mix their numbers up.

RESUMING
    Each model has results/SEA_NET/<model_id>/done_train_dataset.txt listing what it finished.
    "train" skips those. To retrain: delete the file (everything), delete one line (that dataset),
    or set run.re_train: true in configs/main.yaml.

Full documentation is in guide/ - start with guide/README.md.
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
                           model_folder_name, available_models)
from seanet.models import make_sea_net, make_baseline, num_params, state_dict_size_mb
from seanet import results as R
from seanet.results import (result_exists, save_result_row, build_comparison, compare_models,
                            millet_baseline, sweep_order, summarise_model, write_summary,
                            results_csv, done_txt, interpretation_dir, model_dir,
                            predictions_dir, history_dir, MILLET_WEBTRAFFIC_DIR)
from seanet.training import train_one_from_config, fit_model_from_config
from seanet.utils import resolve_device, start_logging

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
    row = D.summarise_dataset(name)
    D.write_summary_row(row)
    print(f"  {row['dataset']:28s} T={row['series_length']:>5d} C={row['n_classes']:>3d} "
          f"train={row['n_train']:>5d} test={row['n_test']:>5d}")


def print_shapes(name):
    """
    Print the tensor shapes of one dataset's train split (a quick "is the data as I expect?" check).

    name : dataset name.
    returns : nothing.
    """
    ds = D.load_dataset(name, "train")
    item = ds[0]
    print(f"  {name:28s} n_bags={len(ds)} bag={tuple(item['bags'].shape)} "
          f"target={item['targets'].item()} classes={ds.n_clz}")


def print_row(row):
    """
    Print one results row as aligned "key : value" lines.

    row : the dict returned by the training path.
    returns : nothing.
    """
    for key, value in row.items():
        if value is None:
            shown = "n/a"
        elif isinstance(value, float):
            shown = f"{value:.4f}"
        else:
            shown = str(value)
        print(f"  {key:16s}: {shown}")


# ---------------------------------------------------------------------------
# shared plumbing: resolve the config, decide what to skip, start MLflow, train + save
# ---------------------------------------------------------------------------
def _resolve_model(args):
    """
    Load the config a command will run with, and work out the model id it writes under.

    Every model-specific command resolves its model exactly the same way, so that reading lives here
    once: the --config file, the --env environment, the --model override, and the
    "<config>__<encoder>__<pooling>" id that names the results folder. main() calls this BEFORE
    start_logging, so the log lands in the right folder.

    args : the parsed command-line arguments.
    returns : (cfg, model_id).
    """
    config_path = getattr(args, "config", None) or os.path.join("configs", "main.yaml")
    overrides = {}
    if getattr(args, "model", None):
        overrides["model"] = args.model
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = int(args.seed)                   # --seed beats main.yaml's seed
    cfg = load_config(config_path, overrides=overrides or None, env=getattr(args, "env", None))
    _apply_output_paths(cfg)                                 # honour output.results_dir / analysis_dir
    return cfg, model_folder_name(cfg)


def _apply_output_paths(cfg) -> None:
    """
    Point the results and analysis modules at the folders the config names.

    Without this, `output.results_dir` in configs/main.yaml would be decoration - the code would
    keep writing to its hard-coded default. Called once, before anything reads or writes.

    cfg : a loaded config.
    returns : nothing.
    """
    out = getattr(cfg, "output", None)
    if out is None:
        return
    if getattr(out, "results_dir", None):
        R.set_results_root(out.results_dir)
    if getattr(out, "analysis_dir", None):
        from seanet.analysis import style as analysis_style   # matplotlib loads only if needed
        analysis_style.set_analysis_root(out.analysis_dir)


def _run_context(args):
    """
    Unpack what a training command needs: the config, the model id, the device and the smoke flag.

    The config and model id were already resolved by main() (so logging could start in the right
    folder); this just reads them back off args and adds the device + smoke decision.

    args : the parsed command-line arguments.
    returns : (cfg, model_id, device, smoke).
    """
    cfg, model_id = args._cfg, args._model_id
    device = resolve_device(getattr(cfg, "device", "auto"))
    smoke = bool(getattr(args, "smoke", False)) or bool(getattr(cfg.run, "smoke", False))
    return cfg, model_id, device, smoke


def _skip_if_done(cfg, model_id, name, smoke):
    """
    Decide whether to skip a dataset that is already finished for this model, and print why.

    Normally we skip a dataset already in the model's done list, so a run is not wasted redoing it.
    main.yaml's "run.re_train" switch overrides that: when it is true we train again anyway and
    save_result_row REPLACES the old row. Smoke runs are never saved, so they always train.

    cfg : the loaded config.  model_id : the model folder id.  name : dataset name.
    smoke : True = a throwaway check (never skips).
    returns : True if the caller should skip this dataset (and has already printed the reason).
    """
    if smoke or not result_exists(model_id, name, cfg.seed):    # nothing done yet -> just train
        return False
    if bool(getattr(cfg.run, "re_train", False)):               # train again, and say so
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
        mlf_tags=tracking.run_tags(cfg, model_id=model_id, command=command),
        logged_model_name=model_id, log_model_weights=log_weights,
        # keep the per-series test predictions (smoke runs are throwaway, so not those). They are
        # tiny and they are the only way to build an ensemble vote later without retraining.
        pred_dir=None if smoke else predictions_dir(model_id),
        # keep the per-epoch history + its two curve figures. It costs nothing extra to record
        # (fit() already builds it) and it is the only way to see the training behaviour after the
        # results have been copied off the training machine.
        history_dir=None if smoke else history_dir(model_id),
    )
    # the two things that define this model, written into every row so results.csv is self-describing
    row["encoder"] = cfg.model_config.encoder.type
    row["pooling"] = cfg.model_config.pooling.type
    if not smoke:                                            # smoke runs are throwaway, never saved
        save_result_row(model_id, row)                       # one row per (dataset, seed)
    return row


def _smoke_note():
    """The one-line reminder printed after any --smoke run (so it is worded the same everywhere)."""
    return "\n  (smoke = 3 epochs; correctness only, NOT a result - nothing was saved)"


# ---------------------------------------------------------------------------
# subcommands (one function per "python main.py <command>")
# ---------------------------------------------------------------------------
def cmd_models(args):
    """
    "models" command: list every model config, grouped by folder.

    args : parsed arguments (unused).
    returns : nothing.
    """
    names = available_models()
    groups = {}
    for name in names:
        folder = name.split("/")[0] if "/" in name else "(root)"
        groups.setdefault(folder, []).append(name)
    what = {
        "baselines": "MILLET and the classic baselines - the models we compare against",
        "seanet": "our own encoder x pooling combinations",
        "ablations": "one-knob-at-a-time studies",
    }
    print(f"{len(names)} model configs under configs/models/\n")
    for folder in sorted(groups):
        print(f"{folder}/   {what.get(folder, '')}")
        for name in groups[folder]:
            print(f"    {name}")
        print()
    print("Pass any of these to --model. The folder is optional when the file name is unique:")
    print("    python main.py single Coffee --model seanet_bottleneck_topk --smoke")


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
    "train" command: train ONE model on every dataset, one after another.

    It trains, in MILLET's order, WebTraffic then the 85 datasets MILLET published then the rest of
    UCR (the order comes from seanet.results.sweep_order() - the 85 come first so the head-to-head
    comparison is ready early). It is resumable (skips datasets already in this model's
    done_train_dataset.txt) and fault-tolerant (a failure on one dataset is logged, the loop goes on).

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
            names = names[: args.limit]                      # only the first N of the standard order

    total = len(names)
    print(f"=== train: model={model_id} (config {cfg.model}) env={getattr(cfg, 'env', 'local')} "
          f"device={device} mode={'smoke' if smoke else 'full'} datasets={total} ===")
    print(param_choice_message(cfg))
    print(f"  results  -> {results_csv(model_id)}")
    print(f"  resume   -> {done_txt(model_id)}  (delete it to retrain everything; "
          f"delete one line to retrain that dataset)\n")

    mlf, log_weights = _start_mlflow(cfg, model_id, smoke)
    done = skipped = failed = 0                              # running tally for the summary line
    for i, name in enumerate(names, 1):
        tag = f"[{i:>3}/{total}]"                             # e.g. "[ 22/129]"
        if result_exists(model_id, name, cfg.seed) and not smoke:   # already finished -> skip
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
    print(f"\nNext: `python main.py analyse` to rebuild the comparison figures.")


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

    print(f"=== single: model={model_id} (config {cfg.model}) dataset={name} "
          f"env={getattr(cfg, 'env', 'local')} device={device} "
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
    "run" command: train + evaluate whatever configs/main.yaml says, and print the resolved config.

    Same training path as "single"; the difference is that this one prints every resolved setting
    first, which is what you want when you are checking reproducibility. Command-line flags
    (--model, --dataset, --env, --smoke) still override the file.

    args : parsed arguments (args.config, args.model, args.dataset, args.smoke).
    returns : nothing.
    """
    cfg, model_id, device, smoke = _run_context(args)
    dataset = args.dataset or cfg.run.dataset               # --dataset overrides the config

    # show exactly what config is driving this run (reproducibility + a quick sanity check)
    print(f"=== run: model={model_id} (config {cfg.model}) dataset={dataset} "
          f"env={getattr(cfg, 'env', 'local')} device={device} "
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
        print(f"  recorded default metrics -> the records.default block of {cfg.model}.yaml")


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
    from seanet.interpretability import generate_interpretations   # matplotlib loads only for this command

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

    Optuna is OPTIONAL and has no training code of its own: every trial calls the same
    seanet.training pipeline that "single" calls, with the sampled values substituted into a copy
    of the config. It reads the "optuna" block from the model config, trains many trials on the
    dataset (WebTraffic by default), and - unless --smoke - records the best values back into that
    same model file under its "records" block.

    args : parsed arguments (args.config, args.model, args.dataset, args.smoke).
    returns : nothing.
    """
    from seanet.optuna_search import run_optuna

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


def cmd_leaderboard(args):
    """
    "leaderboard" command: ONE table with every model, best WebTraffic accuracy first.

    This is the "which model do we keep?" table. WebTraffic is our fast screen (every model runs it),
    so ranking by it puts every model on the same page. The UCR columns are filled in for the models
    that went on to all 129 datasets and left EMPTY for the ones that were only screened -
    empty means "not run yet", never zero.

    It rebuilds from each model's own results.csv every time, so a new model shows up by itself and a
    re-trained model's numbers replace themselves. Writes results/SEA_NET/leaderboard.csv.

    args : parsed arguments (args.fast).
    returns : nothing.
    """
    from seanet.results import build_leaderboard
    build_leaderboard(refresh=not args.fast, verbose=True)


def cmd_analyse(args):
    """
    "analyse" command: build every CROSS-MODEL comparison figure and table.

    This is where the questions the project exists to answer get answered, from the results that
    are already saved (nothing is retrained). It writes results/analysis/:

      01_leaderboard/  who is strongest overall, in non-overlapping accuracy bands
      02_ablation/     what each ENCODER and each POOLING head contributes (the full grid)
      03_detail/       every model, every dataset
      04_webtraffic/   our headline dataset on its own (accuracy, AOPCR, NDCG)
      05_statistics/   average ranks, significance, win/tie/loss against MILLET
      tables/          the same numbers as .csv (exact) and .md (readable on GitHub)
      INDEX.md         one page listing every figure and the question it answers

    The FLOPs / latency / memory figures need `python scripts/profile_models.py` to have been run
    once; without it they are skipped rather than faked.

    args : parsed arguments (args.refresh).
    returns : nothing.
    """
    from seanet.analysis import generate
    generate(refresh=args.refresh, verbose=True)


def cmd_report(args):
    """
    "report" command: build the PER-MODEL figures (one model against the MILLET baseline).

    Use "analyse" to compare models with each other; use this to look at one model in detail.
    Every figure goes under results/SEA_NET/<model_id>/figures/.

    args : parsed arguments (unused).
    returns : nothing.
    """
    from seanet.analysis.model_figures import generate_report
    generate_report(verbose=True)


def cmd_webcompare(args):
    """
    "web-compare" command: WebTraffic-ONLY comparison of every model.

    It ranks all models on WebTraffic (accuracy, loss, AOPCR, NDCG, params) next to two baselines -
    the MILLET PAPER number and our own rerun of the paper baselines - and draws three kinds of
    figure: one per metric, the accuracy-TIER figures (models grouped so each figure holds only a
    few bars), and a winner dashboard. Use it after the WebTraffic screen.

    args : parsed arguments (unused).
    returns : nothing.
    """
    from seanet.results import compare_webtraffic
    from seanet.analysis.model_figures import (plot_webtraffic_comparison, plot_webtraffic_tiers,
                                               plot_winner_dashboard)
    compare_webtraffic(verbose=True)
    figs = (plot_webtraffic_comparison() + plot_webtraffic_tiers() + plot_winner_dashboard())
    for p in figs:
        print(f"  wrote {p}")


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
    p.add_argument("--model", help="model config under configs/models/ without .yaml. The folder is "
                                   "optional: 'seanet_bottleneck_topk' or 'seanet/seanet_bottleneck_topk'. "
                                   "List them with `python main.py models`.")
    p.add_argument("--env", help="environment config under configs/environments/ (local | grid5000). "
                                 "Default: $SEANET_ENV, else local.")
    if with_dataset:
        p.add_argument("--dataset", help="override the dataset (e.g. Coffee)")
    p.add_argument("--seed", type=int, help="training seed (default: the seed in main.yaml, 0). "
                                            "Results are stored per seed, so --seed 1 adds a repeat "
                                            "instead of overwriting seed 0.")
    p.add_argument("--smoke", action="store_true", help="quick check (3 epochs), not saved")


def main():
    """
    Parse the command line and run the chosen subcommand.

    returns : nothing.
    """
    D.chdir_to_repo_root()   # move to the repo root so the "data/..." paths work from anywhere
    parser = argparse.ArgumentParser(prog="main.py", description="SEA-Net: run every part of the pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)   # each command is its own sub-parser

    # --- cheap commands: they only read what already exists ---------------------------------
    p = sub.add_parser("models", help="list every model config you can pass to --model")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("summary", help="data summary (one dataset, --all, or the demo)")
    p.add_argument("dataset", nargs="?", help="dataset name (omit for the WebTraffic+Coffee demo)")
    p.add_argument("--all", action="store_true", help="summarise WebTraffic + all 128 UCR")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("params", help="SEA-Net vs baseline parameter counts")
    p.set_defaults(func=cmd_params)

    p = sub.add_parser("results", help="build the comparison vs MILLET (all models, or one with --model)")
    p.add_argument("--config", default=os.path.join("configs", "main.yaml"), help="path to main.yaml")
    p.add_argument("--env", help="environment config (local | grid5000)")
    p.add_argument("--model", help="report only this model config (default: every model with results)")
    p.set_defaults(func=cmd_results)

    p = sub.add_parser("leaderboard", help="one table of every model ranked by WebTraffic accuracy "
                                           "(UCR columns empty for models only screened)")
    p.add_argument("--fast", action="store_true",
                   help="reuse model_comparison.csv instead of recomputing every model's UCR comparison")
    p.set_defaults(func=cmd_leaderboard)

    p = sub.add_parser("analyse", help="all the cross-model comparison figures + tables "
                                       "-> results/analysis/")
    p.add_argument("--refresh", action="store_true",
                   help="recompute the leaderboard first (slower; default reuses the saved one)")
    p.set_defaults(func=cmd_analyse)

    p = sub.add_parser("report", help="the per-model figures (one model vs MILLET) "
                                      "-> results/SEA_NET/<model>/figures/")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("web-compare", help="WebTraffic comparison of all models vs MILLET: "
                                           "table + accuracy-tier figures + winner")
    p.set_defaults(func=cmd_webcompare)

    # --- expensive commands: these train --------------------------------------------------
    p = sub.add_parser("single", help="train + evaluate one dataset, save its result")
    p.add_argument("dataset", help="dataset name, e.g. Coffee")
    _add_model_flags(p)
    p.set_defaults(func=cmd_single)

    p = sub.add_parser("train", help="train ONE model on EVERY dataset: WebTraffic + 128 UCR (resumable)")
    _add_model_flags(p)
    p.add_argument("--only", nargs="+", metavar="NAME", help="only these datasets")
    p.add_argument("--limit", type=int, help="only the first N datasets of the standard order")
    p.add_argument("--no-webtraffic", action="store_true", help="UCR only")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("webtraffic", help="train on WebTraffic + sanity-check vs MILLET")
    _add_model_flags(p)
    p.set_defaults(func=cmd_webtraffic)

    p = sub.add_parser("run", help="config-driven run: read configs/main.yaml, print every resolved "
                                   "setting, then train + evaluate")
    _add_model_flags(p, with_dataset=True)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("interpret", help="train a model + draw per-sample explanation figures (WebTraffic)")
    _add_model_flags(p, with_dataset=True)
    p.set_defaults(func=cmd_interpret)

    p = sub.add_parser("optuna", help="OPTIONAL hyperparameter search; every trial uses the same "
                                      "training pipeline (reads the model's optuna block)")
    _add_model_flags(p, with_dataset=True)
    p.set_defaults(func=cmd_optuna)

    args = parser.parse_args()

    # Model-specific commands resolve their config FIRST, so their log file can be written inside
    # that model's own folder (results/SEA_NET/<model_id>/logs/). Everything else logs to the
    # shared results/SEA_NET/logs/ folder.
    model_id = None
    if args.command in MODEL_COMMANDS:
        args._cfg, args._model_id = _resolve_model(args)
        model_id = args._model_id
    else:
        # the read-only commands still need output.results_dir / analysis_dir to be honoured
        try:
            _apply_output_paths(load_config(getattr(args, "config", None)
                                            or os.path.join("configs", "main.yaml"),
                                            env=getattr(args, "env", None)))
        except Exception as e:                               # a broken config must not hide the real command
            print(f"[config] could not read the output paths ({type(e).__name__}: {e}) - using the defaults")

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
