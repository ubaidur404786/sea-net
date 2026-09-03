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
    - "sea_simple_voting"         : EVERY timestep votes once for its best class; count the votes
    - "sea_topk_voting"           : the same idea, but only the top-k timesteps are allowed to vote
                                    (both are separate heads, so topk_conjunctive stays untouched)
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
    already = {"type", "dropout", "positional_encoding"}     # handled above (or not a constructor arg)
    for opt, value in vars(pooling_cfg).items():
        if opt not in already and opt in accepted:
            kwargs[opt] = value
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
        beta = F.softplus(self.beta_param).view(1, 1, -1)    # (1, 1, C) >= 0 it wil give always positive values , just add the dim  here view(1, 1, -1) so then it will be eay to multiply because the original was beta.shape=(2,) and g.shape=(2, 100, 2) so we need to add the dim to beta to be able to multiply it with g
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
        q_crit = torch.bmm(onehot, q)                        # (B, C, dq) = the critical query for each class,batch matrix multiplication
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


class SimpleVotingPooling(_AttnPoolingBase):
    """
    The simplest voting head there is: EVERY timestep votes once for its strongest class, we count
    the votes, and the counts become the class scores.

        a_j^k = sigmoid( (W_A z_j)_k )        # per-class gate      (same as Conjunctive)
        p_j^k = classifier(z_j)_k             # per-timestep logit  (same as Conjunctive)
        g_j^k = a_j^k * p_j^k                 # gated evidence
        1. every timestep votes for its best class:   argmax_k g_j^k
        2. count the votes:                           n = [15, 10, 5]
        3. bag_logits = log(n / T)

    No top-k, no temperature, no threshold. Every timestep has an equal say, which is the plain
    reading of "let the evidence vote". sea_topk_voting is the version where only the strongest
    timesteps are allowed to vote; this one is the baseline that says everybody votes.

    ------------------------------------------------------------------------------------------
    THE ONE LINE THAT IS NOT OBVIOUS - and why it has to be there
    ------------------------------------------------------------------------------------------
    Written in the most natural way, this head CANNOT BE TRAINED:

        votes = F.one_hot(g.argmax(dim=-1), C).float()      # <- looks right, but...
        counts = votes.sum(dim=1)
        bag_logits = torch.log(counts)
        loss.backward()   ->  RuntimeError: element 0 of tensors does not require grad

    argmax throws away every number and keeps only "which class won", and one_hot of an integer is
    just a constant. So `counts` is no longer connected to the network at all: PyTorch sees a
    constant, there is nothing to differentiate, and backward() raises. Even if it did not raise,
    every weight would get a gradient of exactly zero and the model would never learn anything.

    The standard fix is the STRAIGHT-THROUGH ESTIMATOR, and it is one line:

        votes = one_vote - share.detach() + share

    Read it as two halves. In the FORWARD pass the two `share` terms cancel (`- share + share`), so
    `votes` really is the hard 1/0 vote - the counts you get out are exactly the [15, 10, 5] you
    wanted. In the BACKWARD pass `one_vote` and `share.detach()` are constants and vanish, so the
    gradient flows through the surviving `share`, which is smooth. Forward is hard, backward is
    soft.

    Be honest about the cost: the gradient no longer matches the function the forward pass computed,
    so it is a BIASED gradient. It trains, but it is an approximation. On WebTraffic the top-k
    version of this idea could not fit the training data at all (train_acc stuck at 0.757); see
    guide/15_voting_pooling.md before reading too much into a result from this head.

    ------------------------------------------------------------------------------------------
    WHY log(n / T) AND NOT softmax(n)
    ------------------------------------------------------------------------------------------
    The training loss is nn.CrossEntropyLoss, which runs its own log_softmax inside. So bag_logits
    must be RAW LOGITS, never probabilities. n / T is the share of the vote each class won, which
    already sums to 1, and log turns that share back into logits, because softmax(log q) == q. The
    loss then sees exactly the vote proportion.

    Dividing by T does NOT change training, only readability: log(n / T) = log(n) - log(T), and
    log(T) is the same number for every class in a row, so softmax cannot see it. Plain log(n)
    trains identically. What you must NOT do is hand over the raw counts: the loss would then
    compute softmax(n), which is a far more extreme distribution AND grows with the length of the
    series, so a 100-step and a 1000-step dataset would behave completely differently.
    """

    def __init__(self, d_in: int, n_clz: int, d_attn: int = 8, dropout: float = 0.1,
                 apply_positional_encoding: bool = True):
        super().__init__(d_in, n_clz, dropout=dropout, apply_positional_encoding=apply_positional_encoding)
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, n_clz),
            nn.Sigmoid(),
        )
        self.instance_classifier = nn.Linear(d_in, n_clz)
        self.last_vote_counts = None                         # kept so a figure can show the votes

    def forward(self, instance_embeddings: torch.Tensor, pos=None) -> Dict[str, torch.Tensor]:
        x = self._prepare(instance_embeddings, pos)          # (B, T, d)
        a = self.attention_head(x)                           # (B, T, C) per-class gates
        p = self.instance_classifier(x)                      # (B, T, C) per-time-point class logits
        g = a * p                                            # (B, T, C) gated evidence
        T, C = g.shape[1], g.shape[2]

        share = torch.softmax(g, dim=-1)                     # (B, T, C) smooth, only for the gradient
        one_vote = F.one_hot(g.argmax(dim=-1), num_classes=C).to(g.dtype)   # (B, T, C) the real vote
        votes = one_vote - share.detach() + share            # forward = hard vote, backward = smooth

        counts = votes.sum(dim=1)                            # (B, C) the [15, 10, 5] vector
        self.last_vote_counts = counts.detach()
        bag_logits = torch.log(counts / T + 1e-8)            # (B, C) raw logits for CrossEntropyLoss
        return {
            "bag_logits": bag_logits,
            "interpretation": g.transpose(1, 2),             # (B, C, T) same meaning as Conjunctive
            "instance_logits": p.transpose(1, 2),
            "attn": a.mean(dim=2, keepdim=True),             # (B, T, 1) for the focus penalty
        }


class TopKVotingPooling(_AttnPoolingBase):
    """
    Top-k VOTING: the k most-supporting timesteps each cast one vote, and the votes decide the class.

    This is a SEPARATE head from sea_topk_conjunctive on purpose. That one averages the evidence and
    is our best model; this one replaces the average with a vote. Keeping them apart means the good
    head stays simple and this experiment cannot change its behaviour by accident.

    The idea: take the k best timesteps, look at the class scores at each one, and let each timestep
    vote for the class it supports most. With k = 30 and 3 classes the counts might be

        class 1 wins at 15 points
        class 2 wins at 10 points     ->   [15, 10, 5]
        class 3 wins at  5 points

    and the series is called class 1. The steps in the code below:

        a_j^k = sigmoid( (W_A z_j)_k )        # per-class gate      (same as Conjunctive)
        p_j^k = classifier(z_j)_k             # per-timestep logit  (same as Conjunctive)
        g_j^k = a_j^k * p_j^k                 # gated evidence
        1. score each timestep by its best class,  s_j = max_k g_j^k
        2. keep the k timesteps with the biggest s_j        (k = ceil(top_frac * T), >= 1)
        3. every kept timestep casts ONE vote:  v_j = softmax(g_j / temperature)
        4. add the votes up:  n^k = sum_j v_j^k             <- the [15, 10, 5] vector
        5. bag_logits = scale * log(n / number_of_voters)

    WHY THE k TIMESTEPS ARE SHARED BY ALL CLASSES. sea_topk_conjunctive picks a DIFFERENT top-k for
    each class. A vote between classes only means something if every class is judged at the SAME time
    points, so here we pick one shared set (step 1) and then everyone votes on it.

    WHY log AND NOT softmax (the easy thing to get wrong). The training loss is nn.CrossEntropyLoss,
    which does its own log_softmax inside. So "bag_logits" must be RAW LOGITS, never probabilities.
    The vote share n / n_voters already IS a probability (it sums to 1), and the logits that give
    back exactly that probability are its logarithm, because softmax(log q) == q. Handing the raw
    counts over instead would make the loss compute softmax(counts), which is a different and much
    more extreme distribution - and its size would depend on k, i.e. on the length of the series.
    `scale` is one learned number (starts at exactly 1) that lets the model sharpen or flatten the
    vote when training needs it.

    WHY THE VOTES ARE SOFT BY DEFAULT. A "hard" vote uses argmax, and argmax has NO gradient - it is
    flat everywhere. Every gradient that reaches the encoder comes through this head, so a hard vote
    would send exactly zero signal back and the model would never learn. A softmax vote still gives
    each timestep exactly one vote (the counts still add up to the number of voters); it just lets a
    timestep split that vote when it is unsure. A small `temperature` makes it sharper.
    Set hard=True for the strict version - see the note on `hard` below.

    Settings:
        top_frac : fraction of timesteps that get to vote (0.1 = the best 10 %).
        temperature : < 1 makes each vote sharper (closer to "all of it to the winner").
        hard : True = a real 1/0 vote in the forward pass, with the soft vote's gradient in the
               backward pass (a "straight-through estimator"). It trains, but the gradient no longer
               matches what the forward pass computed, so it is a BIASED gradient. On WebTraffic it
               could not even fit the training set - see guide/15_voting_pooling.md.
        confidence_threshold : 0.0 = off. Above 0, a timestep whose best two classes are nearly tied
               is not allowed to vote (an undecided point should not get a full say). The gap is
               measured on the per-timestep class PROBABILITIES, softmax over classes of g, so it
               always lies in [0, 1] and the same number means the same thing on every dataset. On
               the raw g scale it would mean nothing, because g is an unnormalised logit.
    """

    def __init__(self, d_in: int, n_clz: int, d_attn: int = 8, dropout: float = 0.1,
                 apply_positional_encoding: bool = True, top_frac: float = 0.1,
                 temperature: float = 0.5, hard: bool = False,
                 confidence_threshold: float = 0.0):
        super().__init__(d_in, n_clz, dropout=dropout, apply_positional_encoding=apply_positional_encoding)
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0 (got {temperature}).")
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, n_clz),
            nn.Sigmoid(),
        )
        self.instance_classifier = nn.Linear(d_in, n_clz)
        self.top_frac = float(top_frac)
        self.temperature = float(temperature)
        self.hard = bool(hard)
        self.confidence_threshold = float(confidence_threshold)
        # one learned number. softplus(0.5413) = 1, so training starts at "logits ARE the log share".
        self.scale = nn.Parameter(torch.tensor(0.5413))
        self.last_vote_counts = None                         # kept so a figure can show the votes

    def forward(self, instance_embeddings: torch.Tensor, pos=None) -> Dict[str, torch.Tensor]:
        x = self._prepare(instance_embeddings, pos)          # (B, T, d)
        a = self.attention_head(x)                           # (B, T, C) per-class gates
        p = self.instance_classifier(x)                      # (B, T, C) per-time-point class logits
        g = a * p                                            # (B, T, C) gated evidence
        T, C = g.shape[1], g.shape[2]
        k = min(max(1, int(math.ceil(self.top_frac * T))), T)   # how many timesteps may vote

        # --- who is allowed to vote? (all of them unless a threshold is set) ---
        allowed = torch.ones_like(g[..., :1])                # (B, T, 1) 1 = may vote
        if self.confidence_threshold > 0.0 and C >= 2:
            q = torch.softmax(g, dim=-1)                     # over CLASSES -> a 0..1 score
            top2 = torch.topk(q, k=2, dim=-1).values         # (B, T, 2) best and second best
            margin = top2[..., 0] - top2[..., 1]             # (B, T) how clear the winner is
            allowed = (margin >= self.confidence_threshold).to(g.dtype).unsqueeze(-1)

        # --- step 1 + 2: pick the k best timesteps, ambiguous ones last ---
        score = g.max(dim=2).values                          # (B, T) strength of each point's best class
        if self.confidence_threshold > 0.0:
            score = score.masked_fill(allowed.squeeze(-1) == 0, float("-inf")) # remove the last dim allowed.squeeze(-1)
        idx = torch.topk(score, k=k, dim=1).indices          # (B, k) ONE shared set of timesteps
        g_top = g.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))     # (B, k, C)
        w = allowed.gather(1, idx.unsqueeze(-1))             # (B, k, 1) may this chosen point vote?
        # If the threshold silenced EVERY chosen point of a series, let them all vote after all.
        # Without this the next line divides by zero, the loss becomes NaN and the run dies.
        w = torch.where(w.sum(dim=1, keepdim=True) > 0, w, torch.ones_like(w))

        # --- step 3 + 4: each kept point casts one vote, and we add them up ---
        votes = torch.softmax(g_top / self.temperature, dim=-1)      # (B, k, C)
        if self.hard:
            # forward = a real 1/0 vote, backward = the soft vote's gradient (straight-through)
            one_hot = F.one_hot(votes.argmax(dim=-1), num_classes=C).to(votes.dtype)
            votes = one_hot.detach() - votes.detach() + votes
        counts = (votes * w).sum(dim=1)                      # (B, C) the [15, 10, 5] vector
        self.last_vote_counts = counts.detach()

        # --- step 5: turn the vote share into logits CrossEntropyLoss can use ---
        share = counts / w.sum(dim=1).clamp_min(1.0)         # (B, C) sums to 1
        bag_logits = F.softplus(self.scale) * torch.log(share.clamp_min(1e-8))
        return {
            "bag_logits": bag_logits,
            "interpretation": g.transpose(1, 2),             # (B, C, T) same meaning as Conjunctive
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



register_pooling("sea_simple_voting", SimpleVotingPooling, has_attn=True)
register_pooling("sea_topk_voting", TopKVotingPooling, has_attn=True)
