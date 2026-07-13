"""
seanet/pooling.py - the pooling module (the MIL pooling heads).

What this file is for:
    A pooling head is the second half of a model. It takes the encoder's per-timestep features and
    turns them into (a) one class prediction for the whole series and (b) an "importance per
    timestep" explanation. This file keeps a small registry (name -> pooling head) so the model can
    pick a pooling head by name from the config, exactly like the encoders in seanet/features.py.

Input / output contract (the same for EVERY pooling head, so they are interchangeable):
    Input  : per-timestep features from the encoder, shape (B, d_in, T), plus optional positions.
    Output : a dict with at least
               "bag_logits"     (B, n_clz)     -> the class scores,
               "interpretation" (B, n_clz, T)  -> importance of each timestep (used by AOPCR/NDCG).
    The attention-based heads (additive / conjunctive / attention) also return "attn" (B, T, 1),
    which the training loop uses for the optional focus penalty. The others do not, and the training
    loop simply skips the penalty when "attn" is absent, so every pooling head still trains.

Five pooling heads come from millet/model/pooling.py (reused unchanged):
    - "additive"    : MILAdditivePooling   (SEA-Net uses this)
    - "conjunctive" : MILConjunctivePooling (the MILLET baseline uses this)
    - "attention"   : MILAttentionPooling
    - "instance"    : MILInstancePooling
    - "gap"         : GlobalAveragePooling

Three more are NEW SEA-Net heads defined at the bottom of THIS file. Each upgrades one weak point of
Conjunctive pooling, and each is a strict generalisation of it (so a well-trained model can never do
worse than the Conjunctive baseline - only better):
    - "classwise_conjunctive" : one attention gate PER CLASS (sharper class-specific explanations)
    - "softmax_conjunctive"   : attention normalised OVER TIME (a real distribution + learnable temperature)
    - "adaptive_classwise"    : per-class gate + a learnable mean<->max aggregator   <- recommended default
See their class docstrings for the simple formulas (in the notation of the MILLET paper).

Related files:
    - seanet/model.py    -> build_model_from_config() calls build_pooling() from here.
    - seanet/features.py -> the first half of a model (the encoder).
    - configs/models/<model>.yaml -> the "pooling" block picks the type and its settings.
"""
import math
from typing import Dict, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from millet.model import pooling as millet_pooling

# --------------------------------------------------------------------------------------
# The pooling registry: name -> (pooling class, does_it_have_an_attention_MLP?).
# The "has_attn" flag tells us whether to pass d_attn (only the attention-based heads take it).
# --------------------------------------------------------------------------------------
POOLING_REGISTRY: Dict[str, Tuple[Type[nn.Module], bool]] = {
    "additive":    (millet_pooling.MILAdditivePooling, True),
    "conjunctive": (millet_pooling.MILConjunctivePooling, True),
    "attention":   (millet_pooling.MILAttentionPooling, True),
    "instance":    (millet_pooling.MILInstancePooling, False),
    "gap":         (millet_pooling.GlobalAveragePooling, False),
}


def register_pooling(name: str, pooling_cls: Type[nn.Module], has_attn: bool = False) -> None:
    """
    Add a pooling head to the registry so it can be picked by config.

    A new pooling head only needs to follow the input/output contract at the top of this file; then
    one call to register_pooling makes it usable from any model config - no other file changes.

    name : the string used in the config (pooling.type).
    pooling_cls : the pooling class, built as pooling_cls(d_in, n_clz, dropout=, apply_positional_encoding=).
    has_attn : True if the class also takes a d_attn argument (an attention-MLP width).
    returns : nothing.
    """
    POOLING_REGISTRY[name] = (pooling_cls, has_attn)


def build_pooling(pooling_cfg, d_in: int, n_clz: int) -> nn.Module:
    """
    Build a pooling head from its config block, using the registry.

    pooling_cfg : the "pooling" section of a model config (.type, .dropout, .positional_encoding,
                  and .d_attn for the attention-based heads).
    d_in : the feature width coming out of the encoder (the pooling input size).
    n_clz : number of classes.
    returns : a pooling module that maps (B, d_in, T) -> the output dict.
    raises ValueError : if the pooling type is not registered.
    """
    kind = pooling_cfg.type
    if kind not in POOLING_REGISTRY:
        raise ValueError(f"Unknown pooling type {kind!r}. Registered: {sorted(POOLING_REGISTRY)}")
    pooling_cls, has_attn = POOLING_REGISTRY[kind]
    kwargs = dict(
        dropout=pooling_cfg.dropout,
        apply_positional_encoding=pooling_cfg.positional_encoding,
    )
    if has_attn and hasattr(pooling_cfg, "d_attn"):          # only the attention heads accept d_attn
        kwargs["d_attn"] = pooling_cfg.d_attn
    return pooling_cls(d_in, n_clz, **kwargs)


# ======================================================================================
# NEW SEA-Net pooling heads (added on top of MILLET's five).
#
# All three keep the MILLET recipe - an attention gate + a per-time-point classifier - but each
# upgrades ONE weak point of Conjunctive pooling. Reminder of the baseline (paper notation):
#
#   Conjunctive:  a_j = sigmoid(scalar)     # ONE gate, shared by all classes
#                 p_j^k = classifier(z_j)_k # per-time-point class logit
#                 Y^k   = (1/T) sum_j a_j * p_j^k     # a plain mean over time
#
# where z_j in R^d is the encoder embedding at time step j (d = d_in), k indexes the C classes,
# and Y^k is the whole-series logit we classify with. The three weak points and our fixes:
#
#   weak point                       ->  new head
#   the gate a_j is class-agnostic   ->  "classwise_conjunctive"  (one gate per class)
#   the gate is unnormalised /T      ->  "softmax_conjunctive"    (softmax over time + temperature)
#   the aggregator is a fixed mean   ->  "adaptive_classwise"     (learnable mean<->max)   [recommended]
#
# Each is a STRICT generalisation of Conjunctive, so a well-trained model can only match or beat it.
#
# Univariate or multivariate: the encoder already maps c input channels -> d embeddings, so these
# heads see d-dim embeddings no matter how many channels the series has - nothing here changes.
# Different series lengths: every head aggregates over time with mean / softmax / soft-argmax, all of
# which are length-invariant, so each dataset's own length just works (no fixed-length assumptions).
# ======================================================================================
class _AttnPoolingBase(millet_pooling.MILPooling):
    """
    Shared plumbing for our attention-based heads.

    It reuses MILLET's positional encoding + dropout (set up by MILPooling.__init__) and does the
    (B, d, T) -> (B, T, d) preamble that every MILLET pooling head starts with, so each subclass only
    has to write its own aggregation in forward().
    """

    def _prepare(self, instance_embeddings: torch.Tensor, pos) -> torch.Tensor:
        """(B, d, T) -> (B, T, d), add positional encoding, then dropout - exactly like MILLET."""
        x = instance_embeddings.transpose(1, 2)              # (B, d, T) -> (B, T, d)
        if self.apply_positional_encoding:
            x = self.positional_encoding(x, pos)             # inject temporal order (Req. 3)
        if self.dropout_p > 0:
            x = self.dropout(x)
        return x


class ClasswiseConjunctivePooling(_AttnPoolingBase):
    """
    Conjunctive pooling with ONE attention gate per class (instead of one shared scalar gate).

    Why: in Conjunctive the gate a_j is the same for every class, so it cannot say "this point
    supports class A but argues against class B". Giving each class its own gate fixes that and gives
    sharper, class-specific explanations.

        a_j^k = sigmoid( (W_A z_j)_k )        # a gate for EACH class k   (W_A maps d -> n_clz)
        p_j^k = classifier(z_j)_k
        Y^k   = (1/T) sum_j  a_j^k * p_j^k    # mean over time, per class

    If the per-class gates all collapse to one value this is exactly Conjunctive; if they -> 1 it is
    Instance pooling. So it only ever adds capacity on top of the baseline.
    """

    def __init__(self, d_in: int, n_clz: int, d_attn: int = 8, dropout: float = 0.1,
                 apply_positional_encoding: bool = True):
        super().__init__(d_in, n_clz, dropout=dropout, apply_positional_encoding=apply_positional_encoding)
        # attention head ends in n_clz outputs (one gate per class), not 1
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, n_clz),
            nn.Sigmoid(),
        )
        self.instance_classifier = nn.Linear(d_in, n_clz)

    def forward(self, instance_embeddings: torch.Tensor, pos=None) -> Dict[str, torch.Tensor]:
        x = self._prepare(instance_embeddings, pos)          # (B, T, d)
        a = self.attention_head(x)                           # (B, T, C) per-class gates in [0,1]
        p = self.instance_classifier(x)                      # (B, T, C) per-time-point class logits
        g = a * p                                            # (B, T, C) gated evidence  a_j^k * p_j^k
        bag_logits = g.mean(dim=1)                           # (B, C) mean over time
        return {
            "bag_logits": bag_logits,
            "interpretation": g.transpose(1, 2),             # (B, C, T) class-specific importance
            "instance_logits": p.transpose(1, 2),            # (B, C, T) before gating
            "attn": a.mean(dim=2, keepdim=True),             # (B, T, 1) mean gate, for the focus penalty
        }


class SoftmaxConjunctivePooling(_AttnPoolingBase):
    """
    Conjunctive pooling where the attention is a real probability distribution OVER TIME (softmax),
    with a learnable temperature, instead of an unnormalised sigmoid gate divided by T.

        s_j^k     = (W_A z_j)_k                    # a raw score per class (no sigmoid)
        alpha_j^k = softmax_j( s_j^k / tau )       # a distribution over time: sum_j alpha_j^k = 1
        p_j^k     = classifier(z_j)_k
        Y^k       = sum_j  alpha_j^k * p_j^k       # a TRUE weighted average (length-invariant)

    tau (temperature) is learned: large tau -> near-uniform weights (~ mean pooling, the safe
    baseline), small tau -> peaky weights (focus on a few decisive points). Because the weights sum to
    1 over time, long and short series are put on the same scale.
    """

    def __init__(self, d_in: int, n_clz: int, d_attn: int = 8, dropout: float = 0.1,
                 apply_positional_encoding: bool = True, temperature: float = 1.0):
        super().__init__(d_in, n_clz, dropout=dropout, apply_positional_encoding=apply_positional_encoding)
        self.attention_scorer = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, n_clz),                        # raw scores per class (softmax done below)
        )
        self.instance_classifier = nn.Linear(d_in, n_clz)
        # temperature > 0, learned. Stored as log(tau) so it can never turn negative during training.
        self.log_temperature = nn.Parameter(torch.tensor(float(math.log(temperature))))

    def forward(self, instance_embeddings: torch.Tensor, pos=None) -> Dict[str, torch.Tensor]:
        x = self._prepare(instance_embeddings, pos)          # (B, T, d)
        s = self.attention_scorer(x)                         # (B, T, C) raw scores
        tau = self.log_temperature.exp().clamp(min=1e-3)     # keep it strictly > 0
        alpha = torch.softmax(s / tau, dim=1)                # (B, T, C) softmax OVER TIME (dim=1)
        p = self.instance_classifier(x)                      # (B, T, C)
        contrib = alpha * p                                  # (B, T, C)
        bag_logits = contrib.sum(dim=1)                      # (B, C) weighted sum (alpha sums to 1 over time)
        return {
            "bag_logits": bag_logits,
            "interpretation": contrib.transpose(1, 2),       # (B, C, T)
            "instance_logits": p.transpose(1, 2),
            "attn": alpha.mean(dim=2, keepdim=True),         # (B, T, 1) mean distribution, for focus penalty
        }


class AdaptiveClasswisePooling(_AttnPoolingBase):
    """
    Recommended head: per-class gates (like classwise_conjunctive) PLUS a learnable mean<->max
    aggregator, so each class learns how "peaky" its evidence is.

        a_j^k = sigmoid( (W_A z_j)_k )              # per-class gate
        p_j^k = classifier(z_j)_k
        g_j^k = a_j^k * p_j^k                        # gated per-time-point evidence
        w_j^k = softmax_j( beta_k * g_j^k )         # soft-argmax weights over time (beta_k >= 0)
        Y^k   = sum_j  w_j^k * g_j^k                 # beta->0 : mean (= Conjunctive)
                                                     # beta->inf: max  (one decisive point)

    beta_k is a learnable per-class sharpness (beta = softplus(theta), so it stays >= 0). This is the
    numerically-stable "soft-argmax" form of a smooth maximum: Y^k is always a weighted average of the
    g values, so it stays in their range (no overflow), and it slides between averaging the whole
    series (good for shape datasets) and trusting a single key region (good for "needle in a haystack"
    datasets) - chosen automatically, per class, per dataset.

    It strictly generalises GAP, Instance, Conjunctive AND max pooling, so it is very hard for it to do
    worse than the baseline. beta starts small (init_beta) so training begins near safe mean pooling.
    """

    def __init__(self, d_in: int, n_clz: int, d_attn: int = 8, dropout: float = 0.1,
                 apply_positional_encoding: bool = True, init_beta: float = 0.3):
        super().__init__(d_in, n_clz, dropout=dropout, apply_positional_encoding=apply_positional_encoding)
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, n_clz),
            nn.Sigmoid(),
        )
        self.instance_classifier = nn.Linear(d_in, n_clz)
        # per-class sharpness beta_k = softplus(theta_k) >= 0. Pick theta so beta starts ~ init_beta
        # (near mean pooling); training sharpens it only where that helps. (inverse of softplus below)
        theta0 = math.log(math.expm1(init_beta)) if init_beta > 0 else -5.0
        self.beta_param = nn.Parameter(torch.full((n_clz,), float(theta0)))

    def forward(self, instance_embeddings: torch.Tensor, pos=None) -> Dict[str, torch.Tensor]:
        x = self._prepare(instance_embeddings, pos)          # (B, T, d)
        a = self.attention_head(x)                           # (B, T, C) per-class gates
        p = self.instance_classifier(x)                      # (B, T, C)
        g = a * p                                            # (B, T, C) gated evidence
        beta = F.softplus(self.beta_param).view(1, 1, -1)    # (1, 1, C) >= 0
        w = torch.softmax(beta * g, dim=1)                   # (B, T, C) soft-argmax weights over time
        bag_logits = (w * g).sum(dim=1)                      # (B, C): mean<->max controlled by beta
        return {
            "bag_logits": bag_logits,
            "interpretation": g.transpose(1, 2),             # (B, C, T) same meaning as Conjunctive
            "instance_logits": p.transpose(1, 2),
            "attn": a.mean(dim=2, keepdim=True),             # (B, T, 1) for the focus penalty
        }


# Register the three new heads so a config can pick them by name (encoder + pooling are swappable
# exactly like MILLET's own). has_attn=True: each one accepts a d_attn argument from the config.
register_pooling("classwise_conjunctive", ClasswiseConjunctivePooling, has_attn=True)
register_pooling("softmax_conjunctive", SoftmaxConjunctivePooling, has_attn=True)
register_pooling("adaptive_classwise", AdaptiveClasswisePooling, has_attn=True)
