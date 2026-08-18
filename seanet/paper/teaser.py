"""
seanet/paper/teaser.py - the "why this work exists" figure for page 1.

What this figure shows
----------------------
The SAME WebTraffic series, sent to several models, one model per row. For each model
the row shows two things, MILLET-Figure-1 style:

    left  panel : the class-probability bars (the prediction), predicted class in red,
                  the true class marked - this is the "what class?" part.
    right panel : the series itself, each timestep COLOURED by how important it was to the
                  predicted class (red = supports, blue = argues against). The injected
                  anomaly is shaded green. This is the "where is the evidence?" part.

A "conventional" (Global Average Pooling) model has no per-timestep map, so its right panel
is drawn plain and labelled "prediction only" - that is the whole point of phase 1.

Each row also prints the model's PARAMETER COUNT (and accuracy / AOPCR when we know them),
so the reader sees the headline story at a glance: our small model gives the same (or a
better) answer AND a sharper explanation with far fewer parameters.

Which models: chosen at run time (see main.py's "teaser" command, --models). The default is
the three phases:
    sv1/conventional          - black box, prediction only
    sv1/millet                - MILLET baseline, self-explaining but large
    sv4/seanet_bottleneck_topk - SEA-Net (ours), tiny and self-explaining

Why WebTraffic: it is the only dataset with per-timestep ground truth, so it is the only one
where we can shade the true anomaly and CHECK the highlights. Same reason as the interpret
command.

Why we train here: no model weights are saved on disk, so each selected model is trained
first (exactly like the interpret command does), then explained. Use --smoke for a quick,
throwaway preview; run without it for the real figure.

Related files:
    - seanet/train.py           -> fit_model_from_config() gives us each trained model.
    - seanet/interpretability.py -> _predicted_class() (used to find a shared correct sample).
    - configs/models/sv1/conventional.yaml -> the black-box model this figure uses.
"""
import os
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd
import torch

from seanet import data as D
from seanet.config import load_config, model_folder_name
from seanet.model import num_params, state_dict_size_mb
from seanet.train import fit_model_from_config, get_device
from seanet.interpretability import _predicted_class

matplotlib.use("Agg")                                   # no screen needed; we only save files
import matplotlib.pyplot as plt                         # noqa: E402  (must come after use("Agg"))

# the default line-up: the three phases of the story (black box -> MILLET -> SEA-Net).
DEFAULT_MODELS = ["sv1/conventional", "sv1/millet", "sv4/seanet_bottleneck_topk"]

# the leaderboard holds each model's OFFICIAL WebTraffic numbers (so the teaser prints the
# same accuracy / AOPCR as the rest of the paper, not a one-off number from this quick run).
_LEADERBOARD = os.path.join("results", "SEA_NET", "leaderboard.csv")

# Pretty names for the figure. The config name (millet, seanet_bottleneck_topk) is what the
# code calls the model, but a reader of the paper should not have to decode underscores --
# the figure must use the same names the tables use.
DISPLAY_NAMES = {
    "conventional":           "Plain classifier (no explanation)",
    "millet":                 "MILLET",
    "seanet_bottleneck_topk": "SEA-Net bottleneck + Top-k",
    "seanet_gated_mean_topk": "SEA-Net gated + Top-k",
    "seanet_classwise":       "SEA-Net wide + class-wise",
}


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _fmt_params(n: int) -> str:
    """Human-friendly parameter count: 41324 -> '41K', 423707 -> '424K', 1_200_000 -> '1.2M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{round(n / 1_000)}K"
    return str(n)


def _leaderboard_row(config_name: str) -> Optional[Dict]:
    """
    Look up a model's official WebTraffic accuracy + AOPCR by its config name (e.g. 'millet').

    Returns None if there is no leaderboard yet or this model is not in it (e.g. the brand-new
    conventional model that has not been swept) - the figure then just leaves those numbers off.
    """
    if not os.path.exists(_LEADERBOARD):
        return None
    df = pd.read_csv(_LEADERBOARD, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]             # the header has padding spaces
    if "config" not in df.columns:
        return None
    df["config"] = df["config"].astype(str).str.strip()
    hit = df[df["config"] == config_name]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {"web_acc": row.get("web_acc"), "web_aopcr": row.get("web_aopcr")}


def _gt_localized(item: Dict, series_len: int) -> Optional[np.ndarray]:
    """
    Get the ground-truth "important region" mask for one series, but only if it is LOCALISED.

    WebTraffic marks the injected anomaly per timestep. We skip the "all important" case (class 0
    labels the whole series), because a green band over everything tells the reader nothing.
    Returns a boolean mask, or None when there is no usable localised region.
    """
    inst = item.get("instance_targets")
    if inst is None:
        return None
    mask = np.asarray(inst).astype(bool)
    if mask.shape[0] == series_len and mask.any() and not mask.all():
        return mask
    return None


def _explain(model, item: Dict) -> Dict:
    """
    One forward pass -> everything the teaser draws for a row (NO AOPCR, so it stays fast).

    This is the light version of interpretability._model_explanation: we only need the class
    probabilities and the per-timestep importance for the predicted class, not the AOPCR deletion
    curves, so we skip that expensive part.
    """
    bag = item["bag"]                                        # (n_instances, d_instance)
    true_clz = int(item["target"])
    model.net.eval()
    with torch.no_grad():
        out = model(bag)                                     # unbatched -> per-sample outputs
    logits = out["bag_logits"].detach().cpu()                # (n_clz,)
    probs = torch.softmax(logits, dim=0).numpy()
    pred_clz = int(torch.argmax(logits))
    interpretation = out["interpretation"].detach().cpu().numpy()   # (n_clz, n_instances)
    importance = interpretation[pred_clz]                    # importance for the PREDICTED class

    bag_np = bag.detach().cpu().numpy()
    series = bag_np[:, 0] if bag_np.shape[1] == 1 else bag_np.mean(axis=1)
    return {
        "series": series, "probs": probs, "pred": pred_clz, "true": true_clz,
        "importance": importance, "gt": _gt_localized(item, len(series)),
    }


# --------------------------------------------------------------------------------------
# step 1: train (or re-train) each selected model and record what the row needs
# --------------------------------------------------------------------------------------
def _prepare_models(model_names: List[str], dataset: str, device, smoke: bool, seed: int,
                    config_path: str, verbose: bool) -> List[Dict]:
    """
    For each model name: load its config, train it, and collect the model + its facts (params,
    size, whether it can explain, official accuracy/AOPCR). Returns one dict per model, in order.
    """
    entries: List[Dict] = []
    for i, name in enumerate(model_names):
        cfg = load_config(config_path, overrides={"model": name, "seed": seed})
        model_id = model_folder_name(cfg)
        pooling_type = cfg.model_config.pooling.type
        config_base = name.split("/")[-1]                    # 'sv1/millet' -> 'millet'

        print(f"\n=== teaser [{i + 1}/{len(model_names)}]: training {name} "
              f"(pooling={pooling_type}) ===", flush=True)
        model, _, _, test_ds, _ = fit_model_from_config(
            dataset, cfg, device=device, smoke=smoke, verbose=verbose)

        official = _leaderboard_row(config_base) if not smoke else None
        entries.append({
            "name": name,
            "model_id": model_id,                            # the results-folder id (for reference)
            "config_base": config_base,
            # the label shown on the row: the paper name if we have one, else the config name
            "display": DISPLAY_NAMES.get(config_base, config_base),
            "model": model,
            "test_ds": test_ds,
            "params": num_params(model.net),
            "size_mb": state_dict_size_mb(model.net),
            # GAP is the only "black box": it has no attention gate, so its map is not meaningful.
            "show_heatmap": pooling_type != "mil_gap",
            "acc": (float(official["web_acc"]) if official and pd.notna(official["web_acc"]) else None),
            "aopcr": (float(official["web_aopcr"]) if official and pd.notna(official["web_aopcr"]) else None),
            "hero": i == len(model_names) - 1,               # the last model is treated as "ours"
        })
    return entries


# --------------------------------------------------------------------------------------
# step 2: find ONE series that every model gets RIGHT (true positive) and whose anomaly is localised
# --------------------------------------------------------------------------------------
def _pick_shared_sample(entries: List[Dict], target_class: Optional[int], verbose: bool) -> int:
    """
    Choose the single series the teaser draws. We want the SAME input for every model, and we want
    it to be a TRUE POSITIVE for all of them (so the picture shows good evidence, not a mistake).
    We also prefer a series whose true anomaly region is localised, so the green band is meaningful.

    Search order: try every test series; keep the first one that (a) is in target_class if one was
    given, (b) has a localised ground-truth region, and (c) is predicted correctly by EVERY model.
    If nothing passes (c)+(b) we relax the localised requirement, and finally fall back to index 0.
    """
    ref_ds = entries[0]["test_ds"]
    targets = np.asarray(ref_ds.targets)
    order = np.arange(len(targets))

    def all_correct(idx: int) -> bool:
        return all(_predicted_class(e["model"], e["test_ds"], idx) == int(targets[idx])
                   for e in entries)

    # pass 1: correct for all models AND a localised anomaly (the ideal teaser sample)
    for idx in order:
        true = int(targets[idx])
        if target_class is not None and true != target_class:
            continue
        if _gt_localized(ref_ds[int(idx)], ref_ds[int(idx)]["bag"].shape[0]) is None:
            continue
        if all_correct(int(idx)):
            if verbose:
                print(f"  chosen sample: #{idx} (true class {true}, localised anomaly, all models correct)")
            return int(idx)

    # pass 2: correct for all models, even if the anomaly is not localised
    for idx in order:
        true = int(targets[idx])
        if target_class is not None and true != target_class:
            continue
        if all_correct(int(idx)):
            if verbose:
                print(f"  chosen sample: #{idx} (true class {true}, all models correct; anomaly not localised)")
            return int(idx)

    print("  WARNING: no series is a true positive for every model - drawing sample #0 instead.")
    return 0


# --------------------------------------------------------------------------------------
# step 3: draw the figure
# --------------------------------------------------------------------------------------
def _draw(entries: List[Dict], idx: int, dataset: str, out_dir: str, smoke: bool) -> str:
    """Draw the stacked teaser (one row per model) and save it as PDF + PNG. Returns the PDF path."""
    explanations = [_explain(e["model"], e["test_ds"][idx]) for e in entries]
    true_clz = explanations[0]["true"]

    n = len(entries)
    # The paper prints this figure about 5.5 inches wide, so a 11-inch figure gets shrunk to
    # half size and every label becomes unreadable. Drawing it near its final width instead
    # means the fonts below survive the shrink.
    fig = plt.figure(figsize=(9.0, 2.1 * n + 0.9))
    gs = fig.add_gridspec(n, 2, width_ratios=[1.0, 3.0], hspace=0.70, wspace=0.22)

    scatter_handle = None                                    # kept for the shared colour bar
    right_axes = []                                          # the series panels, for the colour bar
    for i, (e, ex) in enumerate(zip(entries, explanations)):
        series, probs = ex["series"], ex["probs"]
        pred, true = ex["pred"], ex["true"]
        x = np.arange(len(series))

        # ---- left panel: the prediction (class-probability bars) ----
        axL = fig.add_subplot(gs[i, 0])
        n_clz = len(probs)
        colors = ["#B7B7B7"] * n_clz
        colors[pred] = "#C0392B"                             # predicted class in red
        axL.bar(np.arange(n_clz), probs, color=colors)
        axL.axvline(true, color="#2E8B57", linestyle="--", linewidth=1.6)   # true class marker
        for c in range(n_clz):
            if probs[c] > 0.03:
                axL.text(c, probs[c] + 0.02, f"{probs[c]:.2f}", ha="center", va="bottom", fontsize=8)
        axL.set_ylim(0, 1.12)
        axL.set_xlabel("class", fontsize=9.5)
        axL.set_ylabel("prob.", fontsize=9.5)
        axL.tick_params(labelsize=8)
        if i == 0:
            axL.set_title("prediction", fontsize=10.5)

        # ---- right panel: the explanation (series coloured by importance) ----
        axR = fig.add_subplot(gs[i, 1])
        right_axes.append(axR)
        axR.plot(x, series, color="#CFCFCF", linewidth=1.0, zorder=1)        # the raw shape, faint
        if ex["gt"] is not None:                             # shade the true anomaly region (any row)
            axR.fill_between(x, series.min(), series.max(), where=ex["gt"],
                             color="#2E8B57", alpha=0.15, zorder=0)
        if e["show_heatmap"]:
            vmax = float(np.abs(ex["importance"]).max()) or 1e-8
            sc = axR.scatter(x, series, c=ex["importance"], cmap="coolwarm",
                             vmin=-vmax, vmax=vmax, s=12, zorder=2)
            scatter_handle = sc
        else:
            # the black box: predicts, but has no per-timestep story to tell
            axR.text(0.5, 0.5, "conventional model: prediction only\n(no per-timestep explanation)",
                     transform=axR.transAxes, ha="center", va="center", fontsize=9,
                     color="#666666", style="italic")
        axR.set_xlabel("timestep", fontsize=9.5)
        axR.tick_params(labelsize=8)

        # ---- the row banner: model name + params (+ accuracy / AOPCR when known) ----
        bits = [f"params {_fmt_params(e['params'])}"]
        if e["acc"] is not None:
            bits.append(f"acc {e['acc']:.3f}")
        if e["aopcr"] is not None and e["show_heatmap"]:
            bits.append(f"AOPCR {e['aopcr']:.2f}")
        if smoke:
            bits.append("(smoke)")
        tag = " (ours)" if e["hero"] else ""
        # Tight separators on purpose: a long banner runs past the right edge of the panel and
        # collides with the colour bar, which is what the first version of this figure did.
        banner = f"{e['display']}{tag}  |  " + "  ·  ".join(bits)
        axR.set_title(banner, loc="left", fontsize=10,
                      fontweight="bold", color="#1F6FB2" if e["hero"] else "black")

    # one shared colour bar for the importance maps (only if at least one model drew a map)
    if scatter_handle is not None:
        # pad keeps a clear gap between the panels and the bar, so a long row banner has
        # somewhere to end without running into the bar's label.
        cbar = fig.colorbar(scatter_handle, ax=right_axes, fraction=0.025, pad=0.045)
        cbar.set_label("importance to predicted class\n(red = supports, blue = against)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    # the headline: how much smaller "ours" is than the biggest model in the picture
    params = [e["params"] for e in entries]
    hero = next((e for e in entries if e["hero"]), entries[-1])
    ratio = max(params) / hero["params"] if hero["params"] else 1.0
    fig.suptitle(f"{dataset}  #{idx}, true class {true_clz}: the same series through {n} models "
                 f"—  ours uses {ratio:.0f}× fewer parameters",
                 fontsize=12, y=0.995)

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"teaser_{dataset}_idx{idx:04d}")
    pdf_path = stem + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


# --------------------------------------------------------------------------------------
# the one public entry point
# --------------------------------------------------------------------------------------
def make_teaser(model_names: Optional[List[str]] = None, dataset: str = D.WEB_TRAFFIC,
                out_dir: Optional[str] = None, device=None, smoke: bool = False,
                sample_idx: Optional[int] = None, target_class: Optional[int] = None,
                seed: int = 0, config_path: str = os.path.join("configs", "main.yaml"),
                verbose: bool = True) -> str:
    """
    Build the teaser figure: train the selected models, pick one shared true-positive series, draw it.

    model_names : list of config names to compare (e.g. ["sv1/millet", "sv4/seanet_bottleneck_topk"]);
                  None -> DEFAULT_MODELS (the three phases). Each becomes one row, top to bottom, and
                  the LAST one is highlighted as "ours".
    dataset : which dataset to draw (WebTraffic - the only one with per-timestep ground truth).
    out_dir : where to save; None -> results/SEA_NET/teaser/<date-time> (or .../smoke for smoke runs).
    device : torch device; None -> auto (cuda/mps/cpu).
    smoke : quick 3-epoch training for a throwaway preview (accuracy labels are hidden).
    sample_idx : force a specific test series; None -> search for a good shared true positive.
    target_class : only consider series of this class when searching (None -> any localised-anomaly class).
    seed : training seed (same for every model, so they share the exact same test split).
    config_path : path to main.yaml.
    verbose : print the training stages.
    returns : the saved PDF path.
    """
    model_names = model_names or DEFAULT_MODELS
    device = device or get_device()
    if out_dir is None:
        stamp = "smoke" if smoke else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join("results", "SEA_NET", "teaser", stamp)

    if dataset != D.WEB_TRAFFIC:
        print(f"  NOTE: {dataset} has no per-timestep ground truth, so the anomaly cannot be shaded "
              f"and the highlights cannot be checked. WebTraffic is the dataset to trust here.")

    entries = _prepare_models(model_names, dataset, device, smoke, seed, config_path, verbose)
    idx = sample_idx if sample_idx is not None else _pick_shared_sample(entries, target_class, verbose)
    path = _draw(entries, idx, dataset, out_dir, smoke)

    print(f"\nsaved teaser to {path}")
    print(f"  (PNG next to it: {path[:-4]}.png)")
    if smoke:
        print("\n  (smoke = 3 epochs per model, throwaway preview only - run without --smoke for the "
              "real figure)")

    # free the models
    for e in entries:
        del e["model"]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return path
