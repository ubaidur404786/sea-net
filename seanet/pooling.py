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

The pooling heads registered here all come from millet/model/pooling.py (reused unchanged):
    - "additive"    : MILAdditivePooling   (SEA-Net uses this)
    - "conjunctive" : MILConjunctivePooling (the MILLET baseline uses this)
    - "attention"   : MILAttentionPooling
    - "instance"    : MILInstancePooling
    - "gap"         : GlobalAveragePooling

Related files:
    - seanet/model.py    -> build_model_from_config() calls build_pooling() from here.
    - seanet/features.py -> the first half of a model (the encoder).
    - configs/models/<model>.yaml -> the "pooling" block picks the type and its settings.
"""
from typing import Dict, Tuple, Type

from torch import nn

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
