"""
seanet/models/pooling.py - the pooling module (the MIL pooling heads).

What this file is for:
    A pooling head is the second half of a model. It takes the encoder's per-timestep features and
    turns them into (a) one class prediction for the whole series and (b) an "importance per
    timestep" explanation. This file keeps a small registry (name -> pooling head) so the model can
    pick a pooling head by name from the config, exactly like the encoders in seanet/models/encoders.py.

Input / output contract (the same for EVERY pooling head, so they are interchangeable):
    Input  : per-timestep features from the encoder, shape (B, d_in, T), plus optional positions.
    Output : a dict with at least
               "bag_logits"     (B, n_clz)     -> the class scores,
               "interpretation" (B, n_clz, T)  -> importance of each timestep (used by AOPCR/NDCG).
    The attention-based heads (additive / conjunctive / attention) also return "attn" (B, T, 1),
    which the training loop uses for the optional focus penalty. The others do not, and the training
    loop simply skips the penalty when "attn" is absent, so every pooling head still trains.

Naming rule: every registered name says WHO it came from, so a results folder or a figure label can
be read without opening any code.
    "mil_..." = MILLET (reused from millet/, code untouched)
    "sea_..." = OURS   (written for this project, defined at the bottom of this file)

MILLET - five heads reused unchanged from millet/model/pooling.py:
    - "mil_additive"    : MILAdditivePooling    (the original SEA-Net uses this)
    - "mil_conjunctive" : MILConjunctivePooling (the MILLET baseline uses this)
    - "mil_attention"   : MILAttentionPooling
    - "mil_instance"    : MILInstancePooling
    - "mil_gap"         : GlobalAveragePooling

OURS - defined at the bottom of THIS file. Each upgrades one weak point of Conjunctive pooling, and
each is a strict generalisation of it (so a well-trained model can never do worse than the
Conjunctive baseline - only better):
    - "sea_classwise_conjunctive" : one attention gate PER CLASS (sharper class-specific explanations)
    - "sea_softmax_conjunctive"   : attention normalised OVER TIME (a real distribution + learnable temperature)
    - "sea_adaptive_classwise"    : per-class gate + a learnable mean<->max aggregator  <- recommended default
    - "sea_topk_conjunctive"      : average only the top-k most-supporting timesteps per class
    - "sea_attention_max"         : learnable per-class blend of mean-over-time and max-over-time

OURS too, but the IDEA was taken from two cancer / pathology MIL papers on whole-slide images
(there a bag = one slide and an instance = one image patch, which is exactly our bag = one series
and instance = one timestep). We changed the architecture to fit our setting, so the code is ours;
the papers are credited in the class docstrings and belong in the references:
    - "sea_gated_attention"        : gated attention, from the MS-DA-MIL line of work (Ilse et al. 2018
                                     gated attention). We made it PER CLASS, which the original is not.
    - "sea_dualstream_conjunctive" : the two-stream idea from DSMIL (Li et al. 2021). We wrapped it as a
                                     strict generalisation of classwise Conjunctive (learnable blend
                                     starting at lambda ~ 0), and dropped their contrastive pretraining.
                                     Aimed straight at our AOPCR gap (it concentrates the evidence).
See their class docstrings for the simple formulas (in the notation of the MILLET paper).

Related files:
    - seanet/models/build.py    -> build_model_from_config() calls build_pooling() from here.
    - seanet/models/encoders.py -> the first half of a model (the encoder).
    - configs/models/<model>.yaml -> the "pooling" block picks the type and its settings.
"""
import inspect
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
#
# Naming rule: the name says WHO the head came from, so a results folder or a figure label can be
# read without opening any code.
#   "mil_..." = MILLET's own head, reused unchanged from millet/model/pooling.py
#   "sea_..." = OURS, written for this project (in this file)
# --------------------------------------------------------------------------------------
POOLING_REGISTRY: Dict[str, Tuple[Type[nn.Module], bool]] = {
    "mil_additive":    (millet_pooling.MILAdditivePooling, True),
    "mil_conjunctive": (millet_pooling.MILConjunctivePooling, True),
    "mil_attention":   (millet_pooling.MILAttentionPooling, True),
    "mil_instance":    (millet_pooling.MILInstancePooling, False),
    "mil_gap":         (millet_pooling.GlobalAveragePooling, False),
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

    pooling_cfg : the "pooling" section of a model config. Always has .type, .dropout,
                  .positional_encoding; may also name head-specific settings - .d_attn (attention
                  heads), .temperature (softmax_conjunctive), .init_beta (adaptive_classwise).
    d_in : the feature width coming out of the encoder (the pooling input size).
    n_clz : number of classes.
    returns : a pooling module that maps (B, d_in, T) -> the output dict.
    raises ValueError : if the pooling type is not registered.
    """
    kind = pooling_cfg.type
    if kind not in POOLING_REGISTRY:
        raise ValueError(f"Unknown pooling type {kind!r}. Registered: {sorted(POOLING_REGISTRY)}")
    pooling_cls, _has_attn = POOLING_REGISTRY[kind]
    kwargs = dict(
        dropout=pooling_cfg.dropout,
        apply_positional_encoding=pooling_cfg.positional_encoding,
    )
    # Optional, head-specific settings. Each is passed ONLY if (a) the config names it AND (b) this
    # head's constructor actually accepts it - so d_attn goes only to attention heads, temperature only
    # to softmax_conjunctive, and init_beta only to adaptive_classwise. Adding a knob to a new head
    # later needs no change here: just name it in the config and as a constructor argument.
    accepted = inspect.signature(pooling_cls).parameters
    for opt in ("d_attn", "temperature", "init_beta", "top_frac", "init_lam"):
        if opt in accepted and hasattr(pooling_cfg, opt):
            kwargs[opt] = getattr(pooling_cfg, opt)
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


class TopKConjunctivePooling(_AttnPoolingBase):
    """
    Classwise Conjunctive, but the whole-series logit averages ONLY the top-k most-supporting
    timesteps per class (instead of all T of them).

        a_j^k = sigmoid( (W_A z_j)_k )        # per-class gate
        p_j^k = classifier(z_j)_k
        g_j^k = a_j^k * p_j^k                 # gated evidence per timestep
        Y^k   = mean of the k LARGEST g_j^k over time j   ,   k = ceil(top_frac * T)  (>= 1)

    Why: on long series most timesteps are just background. Averaging over ALL of them dilutes a few
    strong evidence points. Keeping only the top-k concentrates the decision on the points that matter,
    which suits spike-like evidence AND should raise AOPCR (deleting the key points now really hurts the
    score). top_frac = 1.0 recovers ordinary classwise Conjunctive, so it only ever adds capacity.
    """

    def __init__(self, d_in: int, n_clz: int, d_attn: int = 8, dropout: float = 0.1,
                 apply_positional_encoding: bool = True, top_frac: float = 0.1):
        super().__init__(d_in, n_clz, dropout=dropout, apply_positional_encoding=apply_positional_encoding)
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, n_clz),
            nn.Sigmoid(),
        )
        self.instance_classifier = nn.Linear(d_in, n_clz)
        self.top_frac = float(top_frac)                      # fraction of timesteps to keep (0<..<=1)

    def forward(self, instance_embeddings: torch.Tensor, pos=None) -> Dict[str, torch.Tensor]:
        x = self._prepare(instance_embeddings, pos)          # (B, T, d)
        a = self.attention_head(x)                           # (B, T, C) per-class gates
        p = self.instance_classifier(x)                      # (B, T, C) per-time-point class logits
        g = a * p                                            # (B, T, C) gated evidence
        T = g.shape[1]
        k = max(1, int(math.ceil(self.top_frac * T)))        # how many top timesteps to keep (>=1)
        topk = torch.topk(g, k=k, dim=1).values              # (B, k, C) the k largest per class over time
        bag_logits = topk.mean(dim=1)                        # (B, C) mean of just those top-k
        return {
            "bag_logits": bag_logits,
            "interpretation": g.transpose(1, 2),             # (B, C, T) class-specific importance
            "instance_logits": p.transpose(1, 2),
            "attn": a.mean(dim=2, keepdim=True),             # (B, T, 1) for the focus penalty
        }


class AttentionMaxPooling(_AttnPoolingBase):
    """
    Classwise Conjunctive with a learnable blend of MEAN-over-time and MAX-over-time, per class.

        g_j^k  = sigmoid( (W_A z_j)_k ) * classifier(z_j)_k   # gated evidence per timestep
        mean^k = mean_j g_j^k          # spread-out evidence (trends)
        max^k  = max_j  g_j^k          # single strongest point (spikes)
        lam_k  = sigmoid(theta_k)      # per-class blend in [0,1], learned
        Y^k    = (1 - lam_k) * mean^k + lam_k * max^k

    lam_k -> 0 gives ordinary classwise Conjunctive (mean); lam_k -> 1 gives max pooling. Each class
    learns whether its evidence is a trend or a spike. This is a simpler, "hard-max" cousin of
    adaptive_classwise (which uses a smooth soft-argmax); kept separate so we can compare the two.
    """

    def __init__(self, d_in: int, n_clz: int, d_attn: int = 8, dropout: float = 0.1,
                 apply_positional_encoding: bool = True, init_lam: float = 0.5):
        super().__init__(d_in, n_clz, dropout=dropout, apply_positional_encoding=apply_positional_encoding)
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, n_clz),
            nn.Sigmoid(),
        )
        self.instance_classifier = nn.Linear(d_in, n_clz)
        # per-class blend lam_k = sigmoid(theta_k); pick theta0 so lam starts at init_lam
        init_lam = min(max(float(init_lam), 1e-3), 1.0 - 1e-3)
        theta0 = math.log(init_lam / (1.0 - init_lam))       # inverse of sigmoid (the logit)
        self.lam_param = nn.Parameter(torch.full((n_clz,), float(theta0)))

    def forward(self, instance_embeddings: torch.Tensor, pos=None) -> Dict[str, torch.Tensor]:
        x = self._prepare(instance_embeddings, pos)          # (B, T, d)
        a = self.attention_head(x)                           # (B, T, C) per-class gates
        p = self.instance_classifier(x)                      # (B, T, C)
        g = a * p                                            # (B, T, C) gated evidence
        mean_g = g.mean(dim=1)                               # (B, C) mean over time
        max_g = g.max(dim=1).values                          # (B, C) max over time
        lam = torch.sigmoid(self.lam_param).view(1, -1)      # (1, C) per-class blend in [0,1]
        bag_logits = (1.0 - lam) * mean_g + lam * max_g      # (B, C)
        return {
            "bag_logits": bag_logits,
            "interpretation": g.transpose(1, 2),             # (B, C, T)
            "instance_logits": p.transpose(1, 2),
            "attn": a.mean(dim=2, keepdim=True),             # (B, T, 1) for the focus penalty
        }


# ======================================================================================
# Heads ported from the cancer / pathology MIL papers we read.
#
# In those papers a WSI (whole-slide image) is a BAG and each 224x224 patch is an INSTANCE. That is the
# same shape as our problem: a time series is a bag and each timestep is an instance. So their MIL
# aggregators (the "pooling" step that turns per-instance features into one bag prediction + an
# importance map) drop straight onto our (B, d, T) -> bag_logits + interpretation contract.
# ======================================================================================
class GatedAttentionPooling(_AttnPoolingBase):
    """
    Gated-attention MIL pooling. From Ilse et al. 2018 ("Attention-based Deep MIL"), which is the same
    attention head used in paper 1 (MS-DA-MIL). We make it PER-CLASS so it fits our interpretation.

    Plain attention scores each timestep with one tanh MLP. "Gated" attention multiplies the tanh branch
    by a sigmoid branch, so the network can PASS or SUPPRESS each attention feature on its own - a richer,
    more selective gate than tanh alone.

        A_j        = tanh(V z_j)  *  sigmoid(U z_j)     # (per timestep) gated attention features, * = elementwise
        s_j^k      = (W A_j)_k                          # a raw attention score per class   (W: d_attn -> C)
        alpha_j^k  = softmax_j( s_j^k )                 # attention as a distribution OVER TIME (sums to 1)
        p_j^k      = classifier(z_j)_k                  # per-timestep class logit (same as Conjunctive)
        Y^k        = sum_j  alpha_j^k * p_j^k           # weighted average over time (length-invariant)

    Layers: two Linear(d_in, d_attn) (the V and U branches), one Linear(d_attn, n_clz) (the scorer W),
    and one Linear(d_in, n_clz) (the per-timestep classifier). Like softmax_conjunctive the weights sum
    to 1 over time (so long and short series are on the same scale); the extra sigmoid gate is the only
    new part. interpretation = alpha * p (per class, per timestep).
    """

    def __init__(self, d_in: int, n_clz: int, d_attn: int = 8, dropout: float = 0.1,
                 apply_positional_encoding: bool = True):
        super().__init__(d_in, n_clz, dropout=dropout, apply_positional_encoding=apply_positional_encoding)
        self.attn_tanh = nn.Linear(d_in, d_attn)             # the V branch
        self.attn_gate = nn.Linear(d_in, d_attn)             # the U branch (the sigmoid gate)
        self.attn_score = nn.Linear(d_attn, n_clz)           # W: turns gated features into per-class scores
        self.instance_classifier = nn.Linear(d_in, n_clz)

    def forward(self, instance_embeddings: torch.Tensor, pos=None) -> Dict[str, torch.Tensor]:
        x = self._prepare(instance_embeddings, pos)          # (B, T, d)
        A = torch.tanh(self.attn_tanh(x)) * torch.sigmoid(self.attn_gate(x))   # (B, T, d_attn) gated features
        s = self.attn_score(A)                               # (B, T, C) raw scores per class
        alpha = torch.softmax(s, dim=1)                      # (B, T, C) softmax OVER TIME (dim=1)
        p = self.instance_classifier(x)                      # (B, T, C) per-timestep class logits
        contrib = alpha * p                                  # (B, T, C)
        bag_logits = contrib.sum(dim=1)                      # (B, C) weighted sum (alpha sums to 1 over time)
        return {
            "bag_logits": bag_logits,
            "interpretation": contrib.transpose(1, 2),       # (B, C, T) class-specific importance
            "instance_logits": p.transpose(1, 2),            # (B, C, T) before the attention weighting
            "attn": alpha.mean(dim=2, keepdim=True),         # (B, T, 1) mean distribution, for the focus penalty
        }


class DualStreamConjunctivePooling(_AttnPoolingBase):
    """
    DSMIL dual-stream pooling (Li et al. 2021), wrapped so it is a STRICT GENERALISATION of classwise
    Conjunctive (blend starts near Conjunctive, so it can only match or beat the baseline).

    The idea: run two streams and blend them.

    Stream A - the safe baseline (= classwise Conjunctive, a plain mean over time):
        a_j^k = sigmoid( (W_A z_j)_k )       # per-class gate
        p_j^k = classifier(z_j)_k            # per-timestep class logit
        g_j^k = a_j^k * p_j^k                 # gated evidence per timestep
        Ymean^k = (1/T) sum_j  g_j^k          # mean over time

    Stream B - DSMIL "critical-instance" focusing (this is the new, AOPCR-boosting part):
        m_k    = argmax_j g_j^k               # the single most-supporting timestep for class k (the "critical" one)
        q_j    = W_q z_j                       # a small query vector per timestep (W_q: d -> d_attn)
        U_j^k  = softmax_j( <q_j , q_{m_k}> )  # attention = how SIMILAR each timestep is to the critical one
        Yds^k  = sum_j  U_j^k * p_j^k          # class evidence re-weighted toward the critical region

    Blend (learnable, starts ~Conjunctive):
        lam  = sigmoid(theta)  in [0,1]        # one number, learned
        Y^k  = (1 - lam) * Ymean^k  +  lam * Yds^k

    Why it helps us: Stream B pulls the decision onto the critical timestep AND the timesteps that look
    like it, so the evidence stays concentrated. Deleting those points then really hurts the prediction
    -> higher AOPCR. In DSMIL's Figure 1 this also cleans the decision boundary for "needle in a
    haystack" bags. lam -> 0 recovers Conjunctive exactly, so it is never worse than the baseline.
    The interpretation returned is the SAME blend the logit uses, so the heat map matches the decision.

    Layers: the Conjunctive attention MLP + classifier (as usual), one extra Linear(d_in, d_attn) for
    the query, and one scalar blend parameter. Cheap: the attention is measured only against the single
    critical timestep (not every-timestep-to-every-timestep), so there is no T x T cost.
    """

    def __init__(self, d_in: int, n_clz: int, d_attn: int = 8, dropout: float = 0.1,
                 apply_positional_encoding: bool = True, init_lam: float = 0.05):
        super().__init__(d_in, n_clz, dropout=dropout, apply_positional_encoding=apply_positional_encoding)
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, n_clz),
            nn.Sigmoid(),
        )
        self.instance_classifier = nn.Linear(d_in, n_clz)
        self.query = nn.Linear(d_in, d_attn)                 # W_q: small query per timestep (d -> d_attn)
        # blend lam = sigmoid(theta); start small (init_lam) so training begins at ~Conjunctive (safe).
        init_lam = min(max(float(init_lam), 1e-3), 1.0 - 1e-3)
        theta0 = math.log(init_lam / (1.0 - init_lam))       # inverse of sigmoid (the logit)
        self.lam_param = nn.Parameter(torch.tensor(float(theta0)))

    def forward(self, instance_embeddings: torch.Tensor, pos=None) -> Dict[str, torch.Tensor]:
        x = self._prepare(instance_embeddings, pos)          # (B, T, d)
        a = self.attention_head(x)                           # (B, T, C) per-class gates
        p = self.instance_classifier(x)                      # (B, T, C) per-timestep class logits
        g = a * p                                            # (B, T, C) gated evidence
        mean_logit = g.mean(dim=1)                           # (B, C) Stream A: plain mean over time

        # ----- Stream B: focus on the critical timestep of each class -----
        q = self.query(x)                                    # (B, T, dq) one small query per timestep
        T = g.shape[1]
        m = g.argmax(dim=1)                                  # (B, C) index of the critical timestep per class
        # pick the query at each class's critical timestep. one-hot over time, then matrix-multiply:
        onehot = F.one_hot(m, num_classes=T).to(q.dtype)     # (B, C, T) a 1 at the critical timestep
        q_crit = torch.bmm(onehot, q)                        # (B, C, dq) = the critical query for each class
        # similarity of every timestep's query to the critical query, per class (dot product over dq):
        sim = (q.unsqueeze(2) * q_crit.unsqueeze(1)).sum(dim=-1)   # (B, T, C)  <q_j, q_{m_k}>
        U = torch.softmax(sim, dim=1)                        # (B, T, C) attention OVER TIME (sums to 1)
        ds_logit = (U * p).sum(dim=1)                        # (B, C) Stream B: evidence near the critical point

        lam = torch.sigmoid(self.lam_param)                  # scalar in [0,1], learned
        bag_logits = (1.0 - lam) * mean_logit + lam * ds_logit    # (B, C) blended
        interp = (1.0 - lam) * g + lam * (U * p)             # (B, T, C) SAME blend the logit uses
        return {
            "bag_logits": bag_logits,
            "interpretation": interp.transpose(1, 2),        # (B, C, T) class-specific importance
            "instance_logits": p.transpose(1, 2),
            "attn": a.mean(dim=2, keepdim=True),             # (B, T, 1) for the focus penalty
        }


# Register the new heads so a config can pick them by name (encoder + pooling are swappable
# exactly like MILLET's own). has_attn=True: each one accepts a d_attn argument from the config.
register_pooling("sea_classwise_conjunctive", ClasswiseConjunctivePooling, has_attn=True)
register_pooling("sea_softmax_conjunctive", SoftmaxConjunctivePooling, has_attn=True)
register_pooling("sea_adaptive_classwise", AdaptiveClasswisePooling, has_attn=True)
register_pooling("sea_topk_conjunctive", TopKConjunctivePooling, has_attn=True)
register_pooling("sea_attention_max", AttentionMaxPooling, has_attn=True)
register_pooling("sea_gated_attention", GatedAttentionPooling, has_attn=True)
register_pooling("sea_dualstream_conjunctive", DualStreamConjunctivePooling, has_attn=True)
