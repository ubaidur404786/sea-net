"""
seanet/analysis/stats.py - the statistics behind the "is this difference real?" figures.

Why a paper needs this
----------------------
"Our mean accuracy is 0.8298 and theirs is 0.8254" proves nothing on its own. With 85 datasets, a
gap that small can easily be luck. Reviewers at ICLR / NeurIPS / ICML expect a proper comparison of
multiple classifiers over multiple datasets, and in time-series classification that means the
Demsar (2006) procedure, as used by Ismail Fawaz et al. in the deep-learning TSC reviews:

    1. RANK the models on every dataset separately (1 = best on that dataset).
    2. Average each model's rank over all datasets. Average rank is robust: a model cannot win by
       being wildly good on one easy dataset, the way a mean accuracy can.
    3. Test every PAIR with the Wilcoxon signed-rank test - a test that only looks at which model
       won on each dataset and by how much in RANK terms, so it makes no assumption that the scores
       are normally distributed (they are not).
    4. Correct for multiple comparisons with HOLM's step-down method. With 16 models there are 120
       pairs, so at p<0.05 you would expect ~6 "significant" results from pure chance. Holm fixes
       that by making the threshold stricter for the smallest p-values.
    5. Draw the CRITICAL DIFFERENCE DIAGRAM: models on a rank axis, and a thick bar joining any group
       that is NOT significantly different. If two models share a bar, the paper cannot claim one
       beats the other.

What needs SciPy
----------------
Only the Wilcoxon test itself. SciPy is a normal part of any ML environment, but if it is missing
this module says so and the pipeline draws the rank figures without the significance bars, instead
of crashing.
"""
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except ImportError:                                          # the rank figures still work without it
    HAVE_SCIPY = False


def rank_matrix(mat: pd.DataFrame, higher_better: bool = True) -> pd.DataFrame:
    """
    Turn scores into ranks, one dataset at a time.

    On each dataset the best model gets rank 1, the next 2, and so on. Models that tie share the
    average of the ranks they cover (the standard "average" tie rule) - so two models tied for first
    both get 1.5, not 1 and 2, which would invent a winner that does not exist.

    mat : the model x dataset score matrix.
    higher_better : True for accuracy / AOPCR, False for loss.
    returns : the same shape, holding ranks instead of scores.
    """
    # rank down each COLUMN (a column is one dataset, so we rank the models on it)
    return mat.rank(axis=0, ascending=not higher_better, method="average")


def average_ranks(mat: pd.DataFrame, higher_better: bool = True) -> pd.Series:
    """
    Each model's mean rank over all datasets, best (lowest) first.

    mat : the model x dataset matrix (already reduced to the datasets every model shares).
    higher_better : which direction of the score is good.
    returns : a Series model -> average rank, sorted ascending (rank 1 is the best possible).
    """
    return rank_matrix(mat, higher_better).mean(axis=1).sort_values()


def wilcoxon_holm(mat: pd.DataFrame, higher_better: bool = True,
                  alpha: float = 0.05) -> pd.DataFrame:
    """
    Test every pair of models and correct the p-values with Holm's method.

    The Wilcoxon signed-rank test asks: "over these datasets, is one model consistently better than
    the other?" It works on the per-dataset DIFFERENCES, ranks them by size, and checks whether the
    positive differences clearly outweigh the negative ones. It is the right test here because it
    does not assume the accuracies follow a normal distribution.

    Holm's correction: sort the p-values smallest first; the i-th of m tests must beat alpha/(m-i)
    instead of alpha. Once one test fails, everything after it fails too (a "step-down" procedure).
    This keeps the chance of ANY false positive at alpha across the whole family of comparisons.

    mat : the model x dataset matrix.
    higher_better : which direction is good (only affects the reported winner, not the p-value).
    alpha : the significance level for the whole family of tests.
    returns : one row per pair with p_value, p_holm and whether the pair differs significantly.
              An empty frame if SciPy is missing or there are fewer than two models.
    """
    if not HAVE_SCIPY or len(mat) < 2:
        return pd.DataFrame()

    models = list(mat.index)
    rows = []
    for a, b in combinations(models, 2):
        x, y = mat.loc[a], mat.loc[b]
        both = x.notna() & y.notna()
        if both.sum() < 5:                                   # too few datasets to say anything
            continue
        xa, yb = x[both].to_numpy(float), y[both].to_numpy(float)
        if np.allclose(xa, yb):                              # identical scores -> no difference
            p = 1.0
        else:
            try:
                # zero_method="pratt" keeps the datasets where the two models tied, which is the
                # recommended handling; the test is two-sided because we do not assume a direction.
                p = float(wilcoxon(xa, yb, zero_method="pratt", alternative="two-sided").pvalue)
            except ValueError:
                p = 1.0
        mean_a, mean_b = float(np.mean(xa)), float(np.mean(yb))
        better = a if ((mean_a > mean_b) == higher_better) else b
        rows.append({"model_a": a, "model_b": b, "n_datasets": int(both.sum()),
                     "mean_a": mean_a, "mean_b": mean_b, "better": better, "p_value": p})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    m = len(df)
    # Holm step-down: threshold alpha/(m-i) for the i-th smallest p-value, and once a test fails
    # every later test fails too. We store the ADJUSTED p (p * (m-i), capped at 1 and kept
    # non-decreasing) so the table can be read directly against alpha.
    adjusted, running_max = [], 0.0
    for i, p in enumerate(df["p_value"]):
        adj = min(1.0, p * (m - i))
        running_max = max(running_max, adj)                  # keep it non-decreasing down the list
        adjusted.append(running_max)
    df["p_holm"] = adjusted
    df["significant"] = df["p_holm"] < alpha
    return df


def cliques(order: List[str], pairs: pd.DataFrame, alpha: float = 0.05) -> List[Tuple[int, int]]:
    """
    Find the groups of models that are NOT significantly different - the bars of a CD diagram.

    Reading a CD diagram: models sit on a rank axis, best on one side. A thick horizontal bar joins a
    run of models that the test could not separate. If two models are under the same bar, the paper
    must not claim one is better.

    How we build them: walk the models in rank order and find every maximal run in which no pair is
    significantly different. Runs fully contained in a longer run are dropped, so each bar is as long
    as it can be and no bar is redundant.

    order : the model ids sorted by average rank (best first).
    pairs : the output of wilcoxon_holm.
    alpha : the significance level.
    returns : (start index, end index) pairs into `order`, one per bar.
    """
    if pairs.empty or len(order) < 2:
        return []

    # a quick lookup: is this pair significantly different?
    sig = {}
    for _, r in pairs.iterrows():
        sig[(r["model_a"], r["model_b"])] = bool(r["significant"])
        sig[(r["model_b"], r["model_a"])] = bool(r["significant"])

    def differs(a: str, b: str) -> bool:
        return sig.get((a, b), False)

    found = []
    for i in range(len(order)):
        j = i
        # push j as far right as we can while every pair inside stays non-significant
        while j + 1 < len(order) and all(not differs(order[k], order[j + 1]) for k in range(i, j + 1)):
            j += 1
        if j > i:                                            # a bar needs at least two models
            found.append((i, j))

    # drop any run that sits fully inside another one
    maximal = [(i, j) for (i, j) in found
               if not any(a <= i and j <= b and (a, b) != (i, j) for (a, b) in found)]
    return sorted(set(maximal))


def pairwise_wins(mat: pd.DataFrame, higher_better: bool = True,
                  tie_band: float = 0.0) -> pd.DataFrame:
    """
    How many datasets each model wins against each other model, head to head.

    The result is a square table: the cell (row A, column B) is the number of datasets where A beat
    B. It is the data behind the win-matrix heatmap, and it answers "who beats whom" directly rather
    than through the means.

    mat : the model x dataset matrix.
    higher_better : which direction of the score is a win.
    tie_band : differences smaller than this count as a tie, not a win (0 = exact comparison).
    returns : a square DataFrame of win counts (the diagonal is 0).
    """
    models = list(mat.index)
    out = pd.DataFrame(0, index=models, columns=models, dtype=int)
    for a in models:
        for b in models:
            if a == b:
                continue
            x, y = mat.loc[a], mat.loc[b]
            both = x.notna() & y.notna()
            diff = (x[both] - y[both]) if higher_better else (y[both] - x[both])
            out.loc[a, b] = int((diff > tie_band).sum())
    return out


def win_tie_loss(mat: pd.DataFrame, baseline: pd.Series, higher_better: bool = True,
                 tie_band: float = 0.005) -> pd.DataFrame:
    """
    Each model's win / tie / loss record against ONE reference (the MILLET paper's numbers).

    tie_band exists because two accuracies that differ by 0.001 on a 100-sample test set are the same
    result in practice - calling that a "win" would overstate the finding. The band is the same
    0.005 the project already uses elsewhere, so this figure agrees with the existing tables.

    mat : the model x dataset matrix.
    baseline : the reference score per dataset (MILLET's published numbers).
    higher_better : which direction is a win.
    tie_band : differences within this count as a tie.
    returns : one row per model with win / tie / loss counts and the mean gap.
    """
    rows = []
    for model_id in mat.index:
        ours = mat.loc[model_id]
        shared = [d for d in ours.index if d in baseline.index and pd.notna(ours[d])]
        if not shared:
            continue
        diff = ours[shared].astype(float) - baseline[shared].astype(float)
        if not higher_better:
            diff = -diff                                     # for loss, smaller is a win
        rows.append({
            "model": model_id,
            "win": int((diff > tie_band).sum()),
            "tie": int((diff.abs() <= tie_band).sum()),
            "loss": int((diff < -tie_band).sum()),
            "n_datasets": len(shared),
            "mean_gap": float(diff.mean()),
        })
    return pd.DataFrame(rows)


def pareto_front(df: pd.DataFrame, x: str, y: str,
                 x_lower_better: bool = True, y_higher_better: bool = True) -> pd.Series:
    """
    Mark the Pareto-optimal rows: the ones nothing else beats on BOTH axes at once.

    This is the honest way to argue "smaller AND just as good". A model is Pareto-optimal if no other
    model is at least as cheap and at least as accurate, with one of the two strictly better. Those
    models form the front; everything behind the front is strictly worse than something, so there is
    no reason to choose it.

    df : the frame holding both columns.
    x / y : the two column names (e.g. "params" and "web_acc").
    x_lower_better : True when a smaller x is better (cost: params, FLOPs, ms, MB).
    y_higher_better : True when a bigger y is better (quality: accuracy, AOPCR).
    returns : a boolean Series, True for the Pareto-optimal rows.
    """
    keep = df[[x, y]].notna().all(axis=1)
    optimal = pd.Series(False, index=df.index)
    for i in df.index[keep]:
        xi, yi = df.loc[i, x], df.loc[i, y]
        dominated = False
        for j in df.index[keep]:
            if i == j:
                continue
            xj, yj = df.loc[j, x], df.loc[j, y]
            # is j at least as good as i on BOTH axes?
            as_cheap = (xj <= xi) if x_lower_better else (xj >= xi)
            as_good = (yj >= yi) if y_higher_better else (yj <= yi)
            # ... and strictly better on at least one?
            cheaper = (xj < xi) if x_lower_better else (xj > xi)
            better = (yj > yi) if y_higher_better else (yj < yi)
            if as_cheap and as_good and (cheaper or better):
                dominated = True
                break
        optimal[i] = not dominated
    return optimal


def metric_correlation(lb: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    How strongly the metrics agree with each other, as Spearman correlations.

    Spearman (rank) correlation rather than Pearson: we care whether the metrics ORDER the models the
    same way, not whether they move in a straight line together. A high accuracy-vs-NDCG correlation
    would mean "the accurate models are also the interpretable ones"; a near-zero one means the two
    goals are independent, which is itself a result worth showing.

    lb : the leaderboard.
    columns : which metric columns to correlate.
    returns : the square correlation matrix (NaN-safe; pairs with too little overlap come back NaN).
    """
    present = [c for c in columns if c in lb.columns]
    return lb[present].corr(method="spearman", min_periods=5)


def improvement_over(mat: pd.DataFrame, baseline: pd.Series) -> pd.Series:
    """
    Each model's average improvement over a reference, per dataset.

    Reported as the plain difference (ours minus theirs) averaged over the shared datasets, so the
    number reads in the metric's own units - "+0.004 accuracy on average" is directly quotable.
    """
    out: Dict[str, float] = {}
    for model_id in mat.index:
        ours = mat.loc[model_id]
        shared = [d for d in ours.index if d in baseline.index and pd.notna(ours[d])]
        if shared:
            out[model_id] = float((ours[shared].astype(float) - baseline[shared].astype(float)).mean())
    return pd.Series(out).sort_values(ascending=False)
