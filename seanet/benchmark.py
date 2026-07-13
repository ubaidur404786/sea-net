"""
seanet/benchmark.py - compare pooling heads head-to-head on the same encoder + datasets.

What this file is for:
    We added new MIL pooling heads (classwise_conjunctive, softmax_conjunctive, adaptive_classwise) to
    try to beat MILLET's Conjunctive / Additive. This module trains the SAME encoder with EACH pooling
    head on the SAME datasets, so the ONLY thing that changes between runs is the pooling - a fair A/B.
    It collects one row per (dataset, pooling) and prints a ranking (mean test accuracy / AOPCR / NDCG
    per pooling head) so you can see which one wins, and by how much.

    Fairness: for every variant it reloads the base model config (default seanet.yaml) and swaps only
    `pooling.type`, so the encoder and the whole training recipe stay identical.

    Safety: it writes to its OWN file, results/SEA_NET/pooling_benchmark.csv, and does NOT touch
    results.csv / done.txt. So it never marks a dataset "done" (which would make the main sweep skip it)
    and never affects the SEA-Net-vs-MILLET comparison. Each run is still logged to MLflow (if enabled
    in the config), tagged command=benchmark + the pooling name, so you can also compare in the web UI.

Related files:
    - seanet/config.py  -> load_config (we reload the base model per run and swap pooling.type).
    - seanet/train.py   -> fit_model_from_config + score_model (the exact same path as "python main.py run").
    - seanet/pooling.py -> POOLING_REGISTRY (the set of valid pooling names we validate against).
    - main.py ("benchmark" command) -> parses the CLI and calls run_benchmark().
"""
import os
from typing import Dict, List, Optional

import pandas as pd
import torch

from seanet import tracking
from seanet.config import load_config, to_flat_dict
from seanet.pooling import POOLING_REGISTRY
from seanet.train import fit_model_from_config, score_model, get_device

# our own results file (separate from results.csv so the benchmark never disturbs the main sweep)
BENCHMARK_CSV = os.path.join("results", "SEA_NET", "pooling_benchmark.csv")

# the pooling heads compared by default: 2 MILLET baselines + our 3 new heads
DEFAULT_POOLINGS: List[str] = [
    "additive",               # MILLET (SEA-Net's usual head)
    "conjunctive",            # MILLET's proposed baseline
    "classwise_conjunctive",  # ours: one gate per class
    "softmax_conjunctive",    # ours: attention normalised over time
    "adaptive_classwise",     # ours: per-class gate + learnable mean<->max  (recommended)
]

# a small, fast default dataset set. WebTraffic is first because it is the only one with NDCG.
DEFAULT_DATASETS: List[str] = ["WebTraffic", "Coffee", "ECG200", "GunPoint", "ItalyPowerDemand"]

# the column order for pooling_benchmark.csv (pooling right after dataset so the file reads well)
BENCH_COLUMNS: List[str] = [
    "dataset", "pooling", "model", "params", "model_size_mb",
    "n_train", "n_val", "n_test", "series_length", "n_classes",
    "test_acc", "test_bal_acc", "test_auroc", "test_loss", "test_aopcr", "test_ndcg",
    "train_time_s",
]


def _append_benchmark_row(row: Dict, path: str = BENCHMARK_CSV) -> None:
    """
    Append one (dataset, pooling) result to pooling_benchmark.csv.

    Append-only (like results.csv), so a crash mid-benchmark cannot corrupt the file and the finished
    runs are already saved. Re-running a pair just adds a newer row; _load_benchmark keeps the last one.

    row : a results-row dict from score_model, with an extra "pooling" key added.
    path : the benchmark csv file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([row]).reindex(columns=BENCH_COLUMNS)         # fixed column order
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)   # header only for a new file


def _print_ranking(df: pd.DataFrame) -> None:
    """
    Print two tables: the mean metric per pooling head (the ranking), and the best head per dataset.

    df : the benchmark results collected this run (one row per dataset x pooling).
    """
    if df.empty:
        print("\nNo benchmark results (did every run fail?).")
        return
    df = df.copy()
    for col in ["test_acc", "test_aopcr", "test_ndcg"]:            # make sure these are numeric for mean()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ranking: average each metric over the datasets, per pooling head, best accuracy first
    agg = df.groupby("pooling").agg(
        n=("dataset", "count"),
        mean_acc=("test_acc", "mean"),
        mean_aopcr=("test_aopcr", "mean"),
        mean_ndcg=("test_ndcg", "mean"),                          # NaN if no NDCG dataset was run
    ).sort_values("mean_acc", ascending=False)

    print("\n=== pooling ranking (mean over datasets; higher acc / AOPCR / NDCG is better) ===")
    print(f"  {'pooling':24s} {'n':>3s} {'mean_acc':>9s} {'mean_aopcr':>11s} {'mean_ndcg':>10s}")
    print("  " + "-" * 60)
    for pooling, r in agg.iterrows():
        ndcg = "n/a" if pd.isna(r["mean_ndcg"]) else f"{r['mean_ndcg']:.4f}"
        print(f"  {pooling:24s} {int(r['n']):>3d} {r['mean_acc']:>9.4f} "
              f"{r['mean_aopcr']:>11.3f} {ndcg:>10s}")

    # per-dataset winner (which head got the best accuracy on each dataset)
    print("\n=== best pooling per dataset (by test_acc) ===")
    for dataset, g in df.groupby("dataset"):
        best = g.loc[g["test_acc"].idxmax()]
        print(f"  {dataset:16s} -> {best['pooling']:24s} acc={best['test_acc']:.4f}")
    print(f"\n  full table -> {BENCHMARK_CSV}")


def run_benchmark(datasets: Optional[List[str]] = None, poolings: Optional[List[str]] = None,
                  base_model: str = "seanet", config_path: str = os.path.join("configs", "main.yaml"),
                  device: Optional[torch.device] = None, smoke: bool = False,
                  verbose: bool = True) -> pd.DataFrame:
    """
    Train the base model's encoder with each pooling head on each dataset, and rank the heads.

    datasets : datasets to run (default: a small, fast set incl. WebTraffic for NDCG).
    poolings : pooling head names to compare (default: the 2 baselines + our 3 new heads).
    base_model : the model config whose pooling head is swapped (default "seanet"); its encoder +
                 training recipe are used for every variant, so only the pooling changes.
    config_path : path to main.yaml.
    device : where to train; if None, taken from the config (auto -> cuda/mps/cpu).
    smoke : if True, a quick 3-epoch check per run - NOT saved to the csv (just checks the plumbing).
    verbose : print per-run progress and the ranking tables.
    returns : a DataFrame with one row per (dataset, pooling).
    """
    datasets = datasets or DEFAULT_DATASETS
    poolings = poolings or DEFAULT_POOLINGS

    # fail early on a typo instead of failing once per dataset deep inside the loop
    unknown = [p for p in poolings if p not in POOLING_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown pooling type(s) {unknown}. Registered: {sorted(POOLING_REGISTRY)}")

    # load the base config once for the device + MLflow settings (each run reloads its own copy below)
    base_cfg = load_config(config_path, overrides={"model": base_model})
    if device is None:
        device = get_device() if base_cfg.device == "auto" else torch.device(base_cfg.device)
    # MLflow records every run so they are comparable in the web UI too (off for smoke - a plumbing check)
    mlf = None if smoke else tracking.start_experiment(base_cfg)
    log_weights = getattr(getattr(base_cfg, "mlflow", None), "log_model_weights", True)

    total = len(datasets) * len(poolings)
    print(f"=== pooling benchmark: {len(poolings)} heads x {len(datasets)} datasets = {total} runs "
          f"| encoder={base_model} device={device} mode={'smoke' if smoke else 'full'} ===")
    print(f"  poolings: {poolings}")
    print(f"  datasets: {datasets}")

    rows: List[Dict] = []
    i = 0
    for dataset in datasets:
        for pooling in poolings:
            i += 1
            tag = f"[{i:>2}/{total}]"
            try:
                # reload a fresh config and swap ONLY the pooling head (keeps encoder + recipe identical)
                cfg = load_config(config_path, overrides={"model": base_model})
                cfg.model_config.pooling.type = pooling
                cfg.model_config.name = f"{base_model}-{pooling}"     # labels the row + the MLflow model
                print(f"\n{tag} {dataset}  pooling={pooling}", flush=True)

                # exact same train + score path as "python main.py run"
                model, train_ds, val_ds, test_ds, train_time_s = fit_model_from_config(
                    dataset, cfg, device=device, smoke=smoke, verbose=verbose)
                lambda_entropy = cfg.model_config.training.lambda_entropy
                row = score_model(model, dataset, train_ds, val_ds, test_ds, device, cfg.seed,
                                  lambda_entropy, train_time_s, verbose=verbose, mlf=mlf,
                                  mlf_params=to_flat_dict(cfg),
                                  mlf_tags={"command": "benchmark", "pooling": pooling},
                                  logged_model_name=cfg.model_config.name, log_model_weights=log_weights)
                row["pooling"] = pooling                              # the extra column for this file
                rows.append(row)
                if not smoke:                                         # smoke is throwaway -> do not save
                    _append_benchmark_row(row)

                ndcg = "n/a" if row["test_ndcg"] is None else f"{row['test_ndcg']:.3f}"
                print(f"{tag} {dataset:16s} {pooling:22s} DONE  acc={row['test_acc']:.4f} "
                      f"AOPCR={row['test_aopcr']:7.3f} NDCG={ndcg:>5s} ({row['train_time_s']:.0f}s)",
                      flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except KeyboardInterrupt:                                 # Ctrl+C -> keep what finished
                print("\nInterrupted - results so far are saved; ranking below.", flush=True)
                break
            except Exception as e:                                    # one bad run never stops the rest
                print(f"{tag} {dataset:16s} {pooling:22s} FAILED: {type(e).__name__}: {e}", flush=True)
        else:
            continue          # inner loop finished normally -> next dataset
        break                 # inner loop hit KeyboardInterrupt (break) -> stop the outer loop too

    df = pd.DataFrame(rows)
    if verbose:
        _print_ranking(df)
    if smoke:
        print("\n  (smoke = 3 epochs each; correctness only, NOT saved to the csv)")
    return df
