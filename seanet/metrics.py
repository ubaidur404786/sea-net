"""
seanet/metrics.py - every number we score a model with, in one place.

What this file is for:
    The pipeline measures two very different things, and this file is the only place that knows
    how each one is computed:

    A) IS THE PREDICTION RIGHT?  (classification quality)
         accuracy      - share of test series whose predicted class is the true one.
         bal_acc       - "balanced accuracy": the accuracy of each class averaged. On an
                         unbalanced dataset plain accuracy can be high just by always answering
                         the majority class; balanced accuracy cannot.
         auroc         - area under the ROC curve, from the class probabilities. It can fail on a
                         few UCR datasets whose test split is missing a class, so we return NaN
                         there instead of losing the whole dataset (see classification_metrics).
         loss          - cross-entropy on the test set (lower is better).

    B) IS THE EXPLANATION RIGHT?  (interpretability quality - the point of MILLET and SEA-Net)
         AOPCR   - "Area Over the Perturbation Curve at Random". Take the timesteps the model says
                   matter most, delete them one by one, and watch the true-class score fall. A big
                   fall means the model really was leaning on those timesteps. Compared against
                   deleting timesteps at random, hence "at Random". HIGHER IS BETTER.
                   Careful: AOPCR is NOT normalised, so its size depends on how confident the model
                   is. Only compare AOPCR between models trained with the SAME recipe.
         NDCG@n  - needs per-timestep GROUND TRUTH, so it only exists for WebTraffic (the only
                   dataset that ships it). It asks: of the n timesteps that really are important,
                   how many did the model rank at the top? Between 0 and 1, higher is better.

    C) HOW EXPENSIVE IS THE MODEL?
         parameter count and state-dict size come from seanet/models/build.py; FLOPs, latency and
         peak memory are measured separately by scripts/profile_models.py.

    The maths for AOPCR/NDCG is MILLET's (millet/interpretability_metrics.py and
    MILLETModel.evaluate_interpretability), reused unchanged so our numbers are comparable with
    the published ones. This file is the thin, documented wrapper around it.

Related files:
    - seanet/evaluation.py -> calls these on the test set and packs the answers into a results row.
    - millet/interpretability_metrics.py -> the AOPCR / NDCG implementation itself.
"""
from typing import Dict, Optional, Tuple

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from millet.util import cross_entropy_criterion


def classification_metrics(model, dataset) -> Dict:
    """
    Score a trained model's PREDICTIONS on a dataset: accuracy, balanced accuracy, AUROC, loss.

    We first try MILLET's own evaluate() so our numbers match the paper's exactly. That call
    computes roc_auc_score, which raises when a test split does not contain every class (it
    happens on a few UCR datasets). Losing a whole dataset over one secondary metric would be
    silly, so we fall back to computing the other three by hand and set AUROC to NaN.

    model : the trained model (a SeaNetModel / MILLETModel).
    dataset : the split to score (train, validation or test).
    returns : {"acc", "bal_acc", "auroc", "loss"}.
    """
    try:
        return model.evaluate(dataset)                           # normal path (all four metrics)
    except Exception:                                            # AUROC (or similar) failed -> fallback
        logits, targets = [], []
        with torch.no_grad():
            for batch in dataset.create_dataloader(batch_size=16):
                logits.append(model(batch["bags"])["bag_logits"].cpu())
                targets.append(batch["targets"])
        logits = torch.cat(logits)
        targets = torch.cat(targets)
        preds = torch.argmax(logits, dim=1)                      # predicted class = highest logit
        return {
            "acc": accuracy_score(targets.long(), preds),
            "bal_acc": balanced_accuracy_score(targets.long(), preds),
            "auroc": float("nan"),                               # could not compute it
            "loss": cross_entropy_criterion(logits, targets).item(),
        }


def interpretability_metrics(model, dataset) -> Tuple[float, Optional[float]]:
    """
    Score a trained model's EXPLANATIONS on a dataset: AOPCR and (WebTraffic only) NDCG@n.

    This is the slow part of scoring, because AOPCR re-runs the model many times with timesteps
    removed. On a long series that is far more forward passes than the accuracy needs.

    model : the trained model.
    dataset : the split to score (normally the test set).
    returns : (aopcr, ndcg). ndcg is None on any dataset without per-timestep ground truth.
    """
    aopcr, ndcg = model.evaluate_interpretability(dataset)
    return float(aopcr), (float(ndcg) if ndcg is not None else None)


def batch_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Accuracy of one batch, straight from the logits the training step already computed.

    Used to build the per-epoch training accuracy for free: the training loop already has these
    logits, so measuring the training accuracy costs no extra forward pass.

    logits : (B, n_clz) class scores.  targets : (B,) true class indices.
    returns : share of correct predictions in this batch, between 0 and 1.
    """
    preds = torch.argmax(logits, dim=1)
    return float((preds == targets.long()).float().mean().item())
