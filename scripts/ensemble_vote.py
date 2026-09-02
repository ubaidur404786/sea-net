"""
scripts/ensemble_vote.py - compare two (or more) models across seeds, and try ensembling them.

WHAT THIS ANSWERS
    1. "How much does a result move when I only change the random seed?"
       -> mean +- std over seeds, per model. Without this number no accuracy comparison means
          anything, because we already measured the SAME config landing on 0.924 and 0.894.
    2. "Is my model actually different from MILLET, or is the gap noise?"
       -> a paired Wilcoxon signed-rank test over the 85 datasets.
    3. "Do two models help each other if I combine them?"
       -> hard voting (count votes) and soft voting (average the probabilities).

WHY VOTING NEEDS EXTRA FILES
    results.csv only stores summary numbers. You cannot build a vote out of two accuracies - a vote
    needs to know what each model predicted for each individual series. Training now writes those
    predictions to <model folder>/predictions/<dataset>__seed<k>.npz. Any model trained BEFORE that
    change has no prediction files, so it can still be compared on accuracy but cannot be ensembled;
    the script says so instead of guessing.

WHAT "PAIRED WILCOXON" MEANS (the statistics part, in plain words)
    We have the same 85 datasets for both models, so the results come in PAIRS (our accuracy and
    their accuracy on the same dataset). The test looks at the 85 differences and asks: "if the two
    models were really equally good, how likely is a pattern of differences this lopsided?" That
    answer is the p-value. Small p (< 0.05) = the difference is unlikely to be luck. We use Wilcoxon
    rather than a t-test because accuracy differences across very different datasets are not
    bell-shaped, and Wilcoxon only assumes the differences are symmetric around their middle - this
    is also the test the MILLET paper itself uses.

HOW TO RUN IT
    python scripts/ensemble_vote.py --models seanet/seanet_bottleneck_topk baselines/millet
    python scripts/ensemble_vote.py --models seanet/seanet_bottleneck_topk seanet/seanet_inputgate_adaptive \
                                    --baseline baselines/millet
    python scripts/ensemble_vote.py --models A B --datasets all      # all 128 UCR, not just the 85

Related files:
    seanet/training.py    -> save_predictions() writes the .npz files this reads.
    seanet/results.py  -> load_results / mean_over_seeds / millet_datasets.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seanet import data as D                                    # noqa: E402
from seanet import results as R                                 # noqa: E402
from seanet.config import load_config, model_folder_name        # noqa: E402


# --------------------------------------------------------------------------------------
# 1. Turning a config name into the results folder it writes to
# --------------------------------------------------------------------------------------
def short(model_id: str) -> str:
    """The readable half of a model id ('seanet_x__enc__pool' -> 'seanet_x'), for table labels."""
    return model_id.split("__")[0]


def resolve_model_id(name: str) -> str:
    """
    Turn what the user typed into the results folder id.

    Two forms are accepted, so you can use whichever you have to hand:
      - a config name  ("seanet/seanet_bottleneck_topk")  -> loaded, then turned into its full id
      - a full id already ("seanet_bottleneck_topk__sea_mstcn_sep_bottleneck__sea_topk_conjunctive")

    name : what the user passed to --models.
    returns : the model id (the folder name under results/SEA_NET/).
    """
    if "__" in name:                                             # already a full id
        return name
    cfg = load_config(os.path.join("configs", "main.yaml"), overrides={"model": name})
    return model_folder_name(cfg)


# --------------------------------------------------------------------------------------
# 2. Reading the saved predictions
# --------------------------------------------------------------------------------------
def load_predictions(model_id: str) -> Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]]:
    """
    Read every prediction file a model has saved.

    model_id : the results folder id.
    returns : {(dataset, seed) -> (probs (n, C), y (n,))}. Empty if the model has no files yet.
    """
    out: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
    folder = R.predictions_dir(model_id)
    if not os.path.isdir(folder):
        return out
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".npz") or "__seed" not in fname:
            continue
        stem = fname[: -len(".npz")]
        dataset, seed_part = stem.rsplit("__seed", 1)
        try:
            seed = int(seed_part)
        except ValueError:                                       # a stray file - skip it quietly
            continue
        with np.load(os.path.join(folder, fname)) as z:
            out[(dataset, seed)] = (z["probs"], z["y"])
    return out


# --------------------------------------------------------------------------------------
# 3. Per-model accuracy across seeds (works with or without prediction files)
# --------------------------------------------------------------------------------------
def seed_table(model_id: str, datasets: List[str]) -> pd.DataFrame:
    """
    Build a (dataset x seed) table of test accuracy straight from the model's results.csv.

    This does NOT need prediction files, so it works for every model already trained.

    model_id : the results folder id.  datasets : which datasets to keep.
    returns : a DataFrame indexed by dataset, one column per seed (NaN where a seed is missing).
    """
    res = R.load_results(model_id)
    if res.empty:
        return pd.DataFrame()
    res = res[res["dataset"].isin(datasets)].copy()
    if res.empty:
        return pd.DataFrame()
    res["test_acc"] = pd.to_numeric(res["test_acc"], errors="coerce")
    return res.pivot_table(index="dataset", columns="seed", values="test_acc", aggfunc="mean")


def describe_seeds(table: pd.DataFrame) -> Dict[str, float]:
    """
    Summarise a (dataset x seed) accuracy table into the numbers a paper reports.

    "mean_acc" is the headline: average first over seeds (so each dataset counts once, no matter how
    many times it was run), then over datasets. "seed_std" is the average spread BETWEEN seeds on the
    same dataset - this is the noise floor, the number that tells you whether a 0.01 gap is real.

    table : from seed_table().
    returns : a dict of summary numbers (NaN-safe; seed_std is NaN with only one seed).
    """
    if table.empty:
        return {"n_datasets": 0, "n_seeds": 0, "mean_acc": np.nan, "seed_std": np.nan}
    per_dataset = table.mean(axis=1)                             # average over seeds, per dataset
    spread = table.std(axis=1, ddof=1) if table.shape[1] > 1 else pd.Series(dtype=float)
    return {
        "n_datasets": int(per_dataset.notna().sum()),
        "n_seeds": int(table.shape[1]),
        "mean_acc": float(per_dataset.mean()),
        "seed_std": float(spread.mean()) if len(spread) else np.nan,
    }


# --------------------------------------------------------------------------------------
# 4. The ensembles
# --------------------------------------------------------------------------------------
def soft_vote(prob_list: List[np.ndarray]) -> np.ndarray:
    """
    Soft voting: average the probability rows, then take the highest.

    A model that is 95% sure pulls the average further than one that is 40% sure, so confident
    models get more say. This usually beats hard voting, and with only 2 models it also avoids ties.

    prob_list : one (n, C) probability array per member.
    returns : the predicted class per series, shape (n,).
    """
    return np.mean(np.stack(prob_list, axis=0), axis=0).argmax(axis=1)


def hard_vote(prob_list: List[np.ndarray]) -> np.ndarray:
    """
    Hard voting: each member picks one class, and the class with the most votes wins.

    Ties are broken by the summed probability, which is what a human would do - if one model says
    "class 2" and the other says "class 5", believe whichever was more confident.

    prob_list : one (n, C) probability array per member.
    returns : the predicted class per series, shape (n,).
    """
    n, n_clz = prob_list[0].shape
    votes = np.zeros((n, n_clz), dtype=np.float64)
    for p in prob_list:
        votes[np.arange(n), p.argmax(axis=1)] += 1.0
    total = np.mean(np.stack(prob_list, axis=0), axis=0)         # tiny tie-break, never changes a
    return (votes + 1e-6 * total).argmax(axis=1)                 # clear majority


def ensemble_table(model_ids: List[str], datasets: List[str]) -> pd.DataFrame:
    """
    For every dataset where ALL members have predictions, score each member and both ensembles.

    Members are combined ACROSS SEEDS too: every (model, seed) file for a dataset becomes one voter.
    So two models at three seeds is a six-member ensemble, which is the normal way to report this.

    model_ids : the members.  datasets : which datasets to try.
    returns : one row per dataset (accuracy of each member + soft vote + hard vote); empty if no
              dataset has predictions from every member.
    """
    preds = {m: load_predictions(m) for m in model_ids}
    rows = []
    for name in datasets:
        per_model: Dict[str, List[np.ndarray]] = {}
        truth: Optional[np.ndarray] = None
        ok = True
        for m in model_ids:
            got = [(s, pr, y) for (ds, s), (pr, y) in preds[m].items() if ds == name]
            if not got:                                          # this member never ran this dataset
                ok = False
                break
            per_model[m] = [pr for _s, pr, _y in got]
            y0 = got[0][2]
            if truth is None:
                truth = y0
            elif len(truth) != len(y0) or not np.array_equal(truth, y0):
                ok = False                                       # different test order -> unsafe
                break
        if not ok or truth is None:
            continue

        row = {"dataset": name}
        members: List[np.ndarray] = []
        for m in model_ids:
            avg = np.mean(np.stack(per_model[m], axis=0), axis=0)     # this model, averaged over seeds
            row["acc__" + short(m)] = float((avg.argmax(axis=1) == truth).mean())
            members.extend(per_model[m])
        row["acc__soft_vote"] = float((soft_vote(members) == truth).mean())
        row["acc__hard_vote"] = float((hard_vote(members) == truth).mean())
        row["n_members"] = len(members)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 5. The paired test against a baseline
# --------------------------------------------------------------------------------------
def paired_test(ours: pd.Series, theirs: pd.Series) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test on the datasets both models have.

    ours / theirs : accuracy per dataset, indexed by dataset name.
    returns : a dict with n, the two means, win/tie/loss counts and the p-value (NaN if scipy is
              missing or there are too few pairs).
    """
    joined = pd.concat([ours.rename("ours"), theirs.rename("theirs")], axis=1).dropna()
    out = {
        "n": int(len(joined)),
        "mean_ours": float(joined["ours"].mean()) if len(joined) else np.nan,
        "mean_theirs": float(joined["theirs"].mean()) if len(joined) else np.nan,
        "wins": int((joined["ours"] > joined["theirs"]).sum()),
        "ties": int((joined["ours"] == joined["theirs"]).sum()),
        "losses": int((joined["ours"] < joined["theirs"]).sum()),
        "p_value": np.nan,
    }
    if len(joined) < 6:                                          # too few pairs to say anything
        return out
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("  (scipy is not installed, so no p-value: pip install scipy)")
        return out
    diff = joined["ours"] - joined["theirs"]
    if not np.any(diff != 0):                                    # identical everywhere
        return out
    out["p_value"] = float(wilcoxon(joined["ours"], joined["theirs"]).pvalue)
    return out


# --------------------------------------------------------------------------------------
# 6. Printing
# --------------------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="Compare models across seeds and try ensembling them.")
    p.add_argument("--models", nargs="+", required=True,
                   help="two or more model configs (seanet/seanet_bottleneck_topk) or full ids")
    p.add_argument("--baseline", default="baselines/millet",
                   help="model to run the paired test against (default: baselines/millet, our MILLET rerun)")
    p.add_argument("--datasets", choices=["millet85", "all"], default="millet85",
                   help="millet85 = the 85 datasets MILLET published (default); all = all 128 UCR")
    p.add_argument("--out", default=os.path.join("results", "top_results", "SEA_NET", "ensemble_vote.csv"),
                   help="where to write the per-dataset ensemble table")
    args = p.parse_args()

    D.chdir_to_repo_root()
    names = R.millet_datasets() if args.datasets == "millet85" else list(R.UCR_128_DATASETS)
    names = list(names)
    model_ids = [resolve_model_id(m) for m in args.models]
    baseline_id = resolve_model_id(args.baseline) if args.baseline else None

    print(f"=== comparing {len(model_ids)} models over {len(names)} datasets "
          f"({args.datasets}) ===\n")

    # --- part 1: seeds ------------------------------------------------------------------
    print("1. ACCURACY ACROSS SEEDS  (mean over datasets; seed_std = average spread between seeds")
    print("   on the SAME dataset, i.e. your noise floor - ignore any gap smaller than this)\n")
    print(f"   {'model':<34} {'datasets':>8} {'seeds':>6} {'mean acc':>9} {'seed std':>9}")
    tables: Dict[str, pd.DataFrame] = {}
    for m in model_ids + ([baseline_id] if baseline_id and baseline_id not in model_ids else []):
        t = seed_table(m, names)
        tables[m] = t
        d = describe_seeds(t)
        std = "n/a" if np.isnan(d["seed_std"]) else f"{d['seed_std']:.4f}"
        mean = "n/a" if np.isnan(d["mean_acc"]) else f"{d['mean_acc']:.4f}"
        print(f"   {short(m):<34} {d['n_datasets']:>8} {d['n_seeds']:>6} {mean:>9} {std:>9}")
    if all(t.shape[1] <= 1 for t in tables.values() if not t.empty):
        print("\n   NOTE: every model has only ONE seed, so seed_std cannot be measured yet.")
        print("   Re-run with --seed 1 and --seed 2 to get error bars.")

    # --- part 2: paired test vs the baseline --------------------------------------------
    if baseline_id and not tables.get(baseline_id, pd.DataFrame()).empty:
        print(f"\n2. PAIRED TEST vs {short(baseline_id)}  (Wilcoxon signed-rank over the same datasets)\n")
        base_acc = tables[baseline_id].mean(axis=1)
        print(f"   {'model':<34} {'n':>4} {'ours':>7} {'base':>7} {'W/T/L':>12} {'p-value':>9}")
        for m in model_ids:
            if m == baseline_id or tables[m].empty:
                continue
            st = paired_test(tables[m].mean(axis=1), base_acc)
            pv = "n/a" if np.isnan(st["p_value"]) else f"{st['p_value']:.4f}"
            wtl = f"{st['wins']}/{st['ties']}/{st['losses']}"
            print(f"   {short(m):<34} {st['n']:>4} {st['mean_ours']:>7.4f} "
                  f"{st['mean_theirs']:>7.4f} {wtl:>12} {pv:>9}")
        print("\n   p >= 0.05 means the difference is NOT statistically significant - which, when we")
        print("   are behind, is the honest way to say 'we match the baseline'.")

    # --- part 3: the ensemble -----------------------------------------------------------
    print("\n3. ENSEMBLE  (needs saved predictions; models trained before that change have none)\n")
    ens = ensemble_table(model_ids, names)
    if ens.empty:
        have = {short(m): len(load_predictions(m)) for m in model_ids}
        print("   No dataset has predictions from every member yet.")
        print(f"   prediction files found: {have}")
        print("   Re-train the members (predictions are saved automatically now), e.g.:")
        for m in args.models:
            print(f"     python main.py train --model {m}")
        return 0

    acc_cols = [c for c in ens.columns if c.startswith("acc__")]
    print(f"   datasets with a full set of predictions: {len(ens)}")
    print(f"   members per dataset: {int(ens['n_members'].iloc[0])}\n")
    print(f"   {'what':<34} {'mean acc':>9}")
    for c in acc_cols:
        print(f"   {c[len('acc__'):]:<34} {ens[c].mean():>9.4f}")

    best_single = max((c for c in acc_cols if "vote" not in c), key=lambda c: ens[c].mean())
    for vote in ["acc__soft_vote", "acc__hard_vote"]:
        gain = ens[vote].mean() - ens[best_single].mean()
        verdict = "HELPS" if gain > 0 else "does not help"
        print(f"\n   {vote[len('acc__'):]} vs best single ({best_single[len('acc__'):]}): "
              f"{gain:+.4f}  -> {verdict}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ens.to_csv(args.out, index=False)
    print(f"\n   per-dataset table -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
