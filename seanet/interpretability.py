"""
seanet/interpretability.py - drawing "why did the model say that?" figures.

What this file is for:
    A big point of MILLET/SEA-Net is that the model does not only predict a class - it also says how
    important each instance (one timestep) was to that prediction. This file turns that into
    pictures, one per test sample, similar to the figures in the MILLET paper.

    Each figure has seven panels (see make_sample_figure for the exact layout):
      1. the series, each point COLOURED by its importance to the predicted class (red = supports,
         blue = argues against); the true important region is shaded green when we know it;
      2. the class probabilities from the pooling head, with the true and predicted class marked
         and their probability values written on the bars;
      3. the AOPCR "deletion" curves - remove the model's top points first (blue) vs random (orange)
         and watch the predicted-class score fall; the gap between them IS the AOPCR;
      4. the instance-importance curve for the TRUE class;
      5. the instance-importance curve for the PREDICTED class (same y-scale as panel 4, so the two
         are directly comparable - handy when the model is wrong and you want to see what it keyed on);
      6. histograms of those contribution values (true vs predicted);
      7. a plain-English guidance box explaining the colours and why the AOPCR is high or low.

Which samples we draw, and why (see select_correct_samples):
    Only samples the model predicted CORRECTLY, for the first three classes - three figures per run.
    A correct prediction shows what good evidence looks like, which is the thing worth comparing
    between models, and three small figures keep every model's folder the same easy-to-scan size.

Which dataset we draw, and why (see main.py's "interpret" command):
    WebTraffic only. It is the only dataset that ships per-timestep ground truth, so it is the only
    one where we can shade the truly important region in green and CHECK whether the model's
    highlighted points are the right ones. On a UCR dataset the same picture would be unverifiable -
    pretty, but no evidence of anything - so we do not draw it.

Where the numbers come from (the model output dict):
    - bag_logits      (n_clz,)     -> softmax -> class probabilities (panel 2, guidance).
    - interpretation  (n_clz, T)   -> the true-class and predicted-class rows -> importance for
      panels 1, 4, 5, 6.
    - AOPCR curves come from millet.interpretability_metrics.calculate_aopcr (panel 3).

Related files:
    - seanet/training.py  -> fit_model_from_config() gives us the trained model to explain.
    - main.py ("interpret" command) -> trains a model then calls generate_interpretations().
    - configs/main.yaml -> the "interpretability" block picks the dataset / how many classes.
"""
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")                                        # no screen needed; we only save PNG files
import matplotlib.pyplot as plt                              # noqa: E402  (must come after use("Agg"))
import numpy as np
import torch

from millet.interpretability_metrics import calculate_aopcr

# AOPCR settings for the per-sample curve (same defaults MILLET uses, so the value matches the metric)
_AOPCR_STOP = 0.5      # remove up to 50% of the instances
_AOPCR_STEP = 0.05     # in 5% chunks
_AOPCR_N_RANDOM = 3    # average over 3 random orderings for the "random" baseline


def _model_explanation(model, item: Dict) -> Dict:
    """
    Run the model on one series and pull out everything the figure needs (including the AOPCR curves).

    model : a trained SeaNetModel.
    item : one dataset item, i.e. dataset[idx] (has "bag", "target", maybe "instance_targets").
    returns : a dict with the series values, per-instance importance, class probabilities, the
              predicted/true class, the AOPCR deletion curves, and (if present) the ground-truth mask.
    """
    bag = item["bag"]                                        # (n_instances, d_instance)
    true_clz = int(item["target"])

    model.net.eval()
    with torch.no_grad():
        out = model(bag)                                     # unbatched -> values are per-sample
        # AOPCR deletion curves for THIS series: morf = remove model's top points first; rand = random
        aopcr_t, morf_pc, rand_pc = calculate_aopcr(
            model, [bag], verbose=False,
            stop=_AOPCR_STOP, step=_AOPCR_STEP, n_random=_AOPCR_N_RANDOM,
        )
    logits = out["bag_logits"].detach().cpu()                # (n_clz,)
    probs = torch.softmax(logits, dim=0).numpy()             # class probabilities from the pooling head
    pred_clz = int(torch.argmax(logits))
    interpretation = out["interpretation"].detach().cpu().numpy()   # (n_clz, n_instances)
    importance_pre = interpretation[pred_clz]                    # importance for the PREDICTED class
    importance_true = interpretation[true_clz]                   # importance for the TRUE class

    morf_curve = morf_pc[0].cpu().numpy()                    # predicted-class score change as we delete top points
    rand_curve = rand_pc[0].cpu().numpy()                    # ... as we delete random points
    removed_frac = np.linspace(0, 1 - _AOPCR_STOP, len(morf_curve)) * 100   # x-axis: % of instances removed
    aopcr = float(aopcr_t[0])

    # a single value to draw per instance (one timestep = one instance, so normally just its value)
    bag_np = bag.detach().cpu().numpy()
    if bag_np.shape[1] == 1:
        series = bag_np[:, 0]                                # one value per timestep
        xlabel = "timestep"
    else:
        series = bag_np.mean(axis=1)                         # multi-channel bag: average the channels
        xlabel = "window index"

    # ground-truth important instances, if the dataset has them (WebTraffic only)
    gt_mask = None
    inst_targets = item.get("instance_targets")
    if inst_targets is not None:
        mask = np.asarray(inst_targets).astype(bool)
        # skip the "all important" case (WebTraffic labels the whole series for class 0) - it tells us nothing
        if mask.shape[0] == series.shape[0] and mask.any() and not mask.all():
            gt_mask = mask
    
    return {
        "series": series, "importance_pre": importance_pre, "importance_true": importance_true, "probs": probs,
        "pred_clz": pred_clz, "true_clz": true_clz, "xlabel": xlabel, "gt_mask": gt_mask,
        "morf_curve": morf_curve, "rand_curve": rand_curve, "removed_frac": removed_frac, "aopcr": aopcr,
    }


def _aopcr_reason(aopcr: float) -> str:
    """
    Turn an AOPCR value into a short plain-English reason for why it is high or low.

    aopcr : the AOPCR value for this sample.
    returns : a one/two-line explanation.
    """
    if aopcr >= 1.0:
        return ("HIGH: the blue line drops much faster than random, so the model leans on a\n"
                "few key points - removing them hurts a lot.")
    if aopcr <= 0.0:
        return ("NEGATIVE: the blue line drops no faster than random, so the model's 'top'\n"
                "points are not really the ones driving the prediction.")
    return ("LOW: the blue line drops only a little faster than random, so the model spreads\n"
            "its evidence out - removing its top points does not hurt much.")


def make_sample_figure(model, dataset, idx: int, out_dir: str, dataset_name: str,
                       filename: Optional[str] = None) -> str:
    """
    Draw and save the seven-panel explanation figure for one test series.

    model : a trained SeaNetModel.
    dataset : the test dataset.
    idx : which series to explain.
    out_dir : folder to save the PNG in.
    dataset_name : dataset name (used in the title and file name).
    filename : optional fixed file name (used by the smoke preview so it overwrites instead of piling up).
    returns : the path of the saved PNG.
    """
    e = _model_explanation(model, dataset[idx])
    series, importance_pre, importance_true, probs = e["series"], e["importance_pre"], e["importance_true"], e["probs"]
    pred_clz, true_clz = e["pred_clz"], e["true_clz"]
    p_pred, p_true = float(probs[pred_clz]), float(probs[true_clz])
    x = np.arange(len(series))

    # 3 rows x 3 cols; the bottom row (shorter) is one wide guidance strip.
    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.7])
    correct = "CORRECT" if pred_clz == true_clz else "WRONG"
    fig.suptitle(f"{dataset_name}  #{idx}    true class = {true_clz} (p={p_true:.2f})    "
                 f"predicted class = {pred_clz} (p={p_pred:.2f})    [{correct}]", fontsize=13)

    # symmetric colour range around 0 for panel 1 (importance of the predicted class)
    vmax = float(np.abs(importance_pre).max()) or 1e-8
    # ONE shared symmetric y-range for panels 4 & 5, so the true-class and predicted-class
    # importance curves can be compared directly (same scale).
    vmax_contrib = float(max(np.abs(importance_true).max(), np.abs(importance_pre).max())) or 1e-8

    # panel 1: the series, points coloured by importance to the predicted class (+ true region shaded)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, series, color="lightgray", linewidth=1, zorder=1)          # the raw shape, faint
    sc = ax.scatter(x, series, c=importance_pre, cmap="coolwarm", vmin=-vmax, vmax=vmax, s=14, zorder=2)
    if e["gt_mask"] is not None:                              # shade the ground-truth important region
        ax.fill_between(x, series.min(), series.max(), where=e["gt_mask"],
                        color="green", alpha=0.15, label="true important region")
        ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(sc, ax=ax, label="importance  (red = supports pred, blue = against)")
    ax.set_title(f"1. series, coloured by importance (pred class {pred_clz})")
    ax.set_xlabel(e["xlabel"])

    # panel 2: class probabilities from the pooling head, with the values written on the bars
    ax = fig.add_subplot(gs[0, 1])
    n_clz = len(probs)
    colors = ["tab:blue"] * n_clz
    colors[pred_clz] = "tab:red"                             # predicted class in red
    bars = ax.bar(np.arange(n_clz), probs, color=colors)
    ax.axvline(true_clz, color="green", linestyle="--", linewidth=1.6, label=f"true class ({true_clz})")
    for i, b in enumerate(bars):                             # write each probability above its bar
        if probs[i] > 0.02:
            ax.text(i, probs[i] + 0.01, f"{probs[i]:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_title(f"2. class probabilities (softmax of pooling)\npred {pred_clz}: {p_pred:.2f}   true {true_clz}: {p_true:.2f}")
    ax.set_xlabel("class")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # panel 3: AOPCR deletion curves - model's most-important-first vs random
    ax = fig.add_subplot(gs[0, 2])
    rf = e["removed_frac"]
    ax.plot(rf, e["morf_curve"], color="tab:blue", marker="o", markersize=3, label="delete model's top first")
    ax.plot(rf, e["rand_curve"], color="tab:orange", marker="s", markersize=3, label="delete random")
    ax.fill_between(rf, e["morf_curve"], e["rand_curve"], color="tab:blue", alpha=0.12)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"3. AOPCR = gap between the curves = {e['aopcr']:.2f}")
    ax.set_xlabel("% of instances removed")
    ax.set_ylabel("change in predicted-class score")
    ax.legend(fontsize=8, loc="lower left")

    # panel 4: instance importance for the TRUE class
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, importance_true, color="tab:green", linewidth=1)
    ax.fill_between(x, 0, importance_true, color="tab:green", alpha=0.2)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylim(-1.05 * vmax_contrib, 1.05 * vmax_contrib)   # shared scale with panel 5
    ax.set_title(f"4. instance importance - TRUE class {true_clz}")
    ax.set_xlabel(e["xlabel"])
    ax.set_ylabel("contribution")

    # panel 5: instance importance for the PREDICTED class (same y-scale as panel 4)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x, importance_pre, color="tab:purple", linewidth=1)
    ax.fill_between(x, 0, importance_pre, color="tab:purple", alpha=0.2)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylim(-1.05 * vmax_contrib, 1.05 * vmax_contrib)   # shared scale with panel 4
    ax.set_title(f"5. instance importance - PRED class {pred_clz}")
    ax.set_xlabel(e["xlabel"])
    ax.set_ylabel("contribution")

    # panel 6: histogram of the contribution values (true vs predicted, overlaid)
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(importance_true, bins=30, color="tab:green", alpha=0.5, label=f"true ({true_clz})")
    ax.hist(importance_pre, bins=30, color="tab:purple", alpha=0.5, label=f"pred ({pred_clz})")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("6. histogram of contributions")
    ax.set_xlabel("contribution value")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)

    # panel 7: plain-English guidance across the whole bottom row (two columns of text)
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    guidance_left = (
        "How to read this figure\n"
        "------------------------------------------------\n"
        "Panel 1 - each point's COLOUR is its importance to the PREDICTED class:\n"
        "   red = supports the prediction,  blue = argues against it\n"
        "   grey line = the raw series shape\n"
        "   green band = the TRUE important region (shown only when known)\n\n"
        "Panel 2 - probabilities from the pooling head (softmax):\n"
        f"   predicted class {pred_clz}: p = {p_pred:.2f} (red bar)\n"
        f"   true class      {true_clz}: p = {p_true:.2f} (green line)"
    )
    guidance_right = (
        "Panels 4 & 5 - instance importance, SAME y-scale so we can compare:\n"
        f"   panel 4 = contribution to the TRUE class {true_clz} (green)\n"
        f"   panel 5 = contribution to the PRED class {pred_clz} (purple)\n"
        "   if the model is WRONG, compare the two to see what it keyed on.\n"
        "Panel 6 - histograms of those contributions (true vs pred).\n\n"
        "Panel 3 - AOPCR (why it is high or low):\n"
        f"   AOPCR = {e['aopcr']:.2f}\n"
        f"   {_aopcr_reason(e['aopcr'])}"
    )
    ax.text(0.0, 1.0, guidance_left, ha="left", va="top", fontsize=9, family="monospace",
            transform=ax.transAxes)
    ax.text(0.52, 1.0, guidance_right, ha="left", va="top", fontsize=9, family="monospace",
            transform=ax.transAxes)

    fig.tight_layout(rect=(0, 0, 1, 0.96))                   # leave room for the suptitle
    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        filename = f"{dataset_name}_sample{idx:04d}_true{true_clz}_pred{pred_clz}.png"
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)                                           # free the figure's memory
    return path


def _predicted_class(model, dataset, idx: int) -> int:
    """
    Ask the model which class it predicts for one test series.

    We need this to pick only the samples the model got RIGHT (see select_correct_samples). It is a
    single forward pass: the pooling head's bag_logits is a score per class, and the highest score
    is the prediction.

    model : a trained SeaNetModel.  dataset : the test dataset.  idx : which series.
    returns : the predicted class index.
    """
    bag = dataset[idx]["bag"]
    with torch.no_grad():
        out = model([bag])                                   # the model takes a LIST of bags (MIL style)
    return int(torch.argmax(out["bag_logits"][0]))


def select_correct_samples(model, dataset, n_classes: int = 3, per_class: int = 1) -> List[int]:
    """
    Pick the test series to explain: ones the model predicted CORRECTLY, for the first few classes.

    Why only correct ones: a figure is meant to show what good evidence looks like - which points
    the model leaned on when it got the answer right. A wrong prediction explains a mistake, which
    is interesting but is not what we want to compare between models.

    Why only the first few classes: three figures per run is enough to eyeball a model, and it keeps
    every model's interpretation folder the same small size, so they are easy to compare side by side.

    model : a trained SeaNetModel.
    dataset : the test dataset.
    n_classes : how many classes to cover, counting from class 0 (default 3).
    per_class : how many correctly-predicted examples to take from each of those classes.
    returns : a list of dataset indices (may be shorter than n_classes * per_class if the model got
              every sample of some class wrong - we draw what we can rather than failing).
    """
    targets = torch.as_tensor(dataset.targets)
    chosen: List[int] = []
    for clz in range(min(n_classes, dataset.n_clz)):         # classes 0, 1, 2 (or fewer if binary)
        found = 0
        for idx in (targets == clz).nonzero(as_tuple=True)[0].tolist():
            if found >= per_class:                           # got enough examples of this class
                break
            if _predicted_class(model, dataset, int(idx)) == clz:   # keep it only if the model was right
                chosen.append(int(idx))
                found += 1
        if found == 0:
            print(f"    note: no correctly-predicted test sample for class {clz} - skipping it.")
    return chosen


def generate_interpretations(model, dataset, out_dir: str, n_classes: int = 3, per_class: int = 1,
                             dataset_name: str = "dataset", limit: Optional[int] = None,
                             fixed_name: Optional[str] = None) -> List[str]:
    """
    Draw explanation figures for the correctly-predicted samples of the first few classes.

    model : a trained SeaNetModel.
    dataset : the test dataset.
    out_dir : folder to save the figures in.
    n_classes : how many classes to cover, from class 0 (default 3).
    per_class : how many correctly-predicted examples per class (default 1 -> 3 figures in total).
    dataset_name : dataset name (used in titles and file names).
    limit : if given, only draw this many figures (used by the smoke preview to draw just one).
    fixed_name : if given, save under this single file name (overwrites) - used by the smoke preview
                 so repeated smoke runs do not pile up files.
    returns : the list of saved figure paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    idxs = select_correct_samples(model, dataset, n_classes=n_classes, per_class=per_class)
    if limit is not None:
        idxs = idxs[:limit]
    return [make_sample_figure(model, dataset, idx, out_dir, dataset_name, filename=fixed_name)
            for idx in idxs]
