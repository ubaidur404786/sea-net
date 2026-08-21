"""
seanet/models/build.py - wiring an encoder and a pooling head into a full model.

What this file is for:
    A model here is always the same two-part shape: an encoder (from models/encoders.py) followed
    by a MIL pooling head (from models/pooling.py). This file holds the small glue that joins them
    (EncoderPoolNet), the two ready-made builders (SEA-Net and the MILLET baseline), and the
    config-driven builder that picks the encoder + pooling head by name.

    The actual encoders live in models/encoders.py and the pooling heads in models/pooling.py.
    This file just decides which of them to use and connects them. That split is what makes the
    encoder and the pooling head INDEPENDENTLY SWITCHABLE: to add a new one you edit those files,
    never this one.

Input / output of a model:
    Input  : a batch of series, shape (B, 1, T)   [B = batch, 1 = one channel, T = length].
    Output : a dict with
               "bag_logits"     (B, n_clz)     -> the class scores,
               "interpretation" (B, n_clz, T)  -> importance of each timestep (used by AOPCR/NDCG),
               "attn"           (B, T, 1)       -> the attention gate (attention-based heads only).

Related files:
    - seanet/models/encoders.py -> the encoders (build_encoder + the registry).
    - seanet/models/pooling.py  -> the pooling heads (build_pooling + the registry).
    - seanet/training.py        -> wraps a model in a trainable object and fits it.
    - main.py                   -> "params" prints sizes; "run"/"single"/"train" build from config.
"""
import io
from typing import Optional

import torch
from torch import nn

from millet.model import backbone
from millet.model import pooling as millet_pooling

# The encoder pieces are re-exported so `from seanet.models import MSTCNSepEncoder` works.
# build_encoder / build_pooling are the registry lookups the config-driven builder below uses.
from seanet.models.encoders import (  # noqa: F401  (re-exported on purpose)
    SEA_D,
    SEA_DROPOUT,
    SEA_N_BLOCKS,
    SEA_MAX_DILATION,
    SEA_KERNELS,
    MultiScaleSepBlock,
    MSTCNSepEncoder,
    build_encoder,
)
from seanet.models.pooling import build_pooling


# --------------------------------------------------------------------------------------
# Generic wiring: encoder + pooling head. Both SEA-Net and the baseline are built with this, so
# they are wired up exactly the same way.
# --------------------------------------------------------------------------------------
class EncoderPoolNet(nn.Module):
    """
    A MILLET-style network = a feature extractor (encoder) followed by a MIL pooling head.

    This is just glue: it runs the encoder, then the pooling, and returns the pooling's dict.
    """

    def __init__(self, feature_extractor: nn.Module, pool: nn.Module):
        """
        feature_extractor : the encoder module, maps (B, C, T) -> (B, d, T).
        pool : the pooling head, maps (B, d, T) -> the output dict.
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.pool = pool

    def forward(self, bags: torch.Tensor, pos: Optional[torch.Tensor] = None) -> dict:
        """
        Run one batch through the network.

        bags : input series, shape (B, n_channels, T).
        pos : optional timestep positions. AOPCR deletes timesteps and re-runs the model on the
              shorter series, so it passes the original positions through to the pooling head.
        returns : the pooling head's output dict (bag_logits, interpretation, attn, ...).
        """
        timestep_embeddings = self.feature_extractor(bags)   # (B, C, T) -> (B, d, T)
        return self.pool(timestep_embeddings, pos=pos)       # (B, d, T) -> output dict


# --------------------------------------------------------------------------------------
# Builders (the two ready-made networks we compare)
# --------------------------------------------------------------------------------------
def make_sea_net(n_clz: int, dropout: float = SEA_DROPOUT, n_in: int = 1) -> EncoderPoolNet:
    """
    Build the SEA-Net network (our encoder + MILLET's Additive pooling).

    n_clz : number of classes for this dataset.
    dropout : dropout probability (defaults to the fixed SEA_DROPOUT).
    n_in : number of input channels (always 1 here - one instance is one timestep).
    returns : an EncoderPoolNet ready to be wrapped for training (see seanet/training.py).
    """
    encoder = MSTCNSepEncoder(n_in=n_in, d=SEA_D, dropout=dropout)
    pool = millet_pooling.MILAdditivePooling(encoder.d_out, n_clz, dropout=dropout, apply_positional_encoding=True)
    return EncoderPoolNet(encoder, pool)


def make_baseline(n_clz: int) -> EncoderPoolNet:
    """
    Build MILLET's baseline network (InceptionTime encoder + Conjunctive pooling). This is what
    SEA-Net is compared against; both parts come straight from the millet code.

    n_clz : number of classes for this dataset.
    returns : an EncoderPoolNet using the MILLET baseline components.
    """
    return EncoderPoolNet(
        backbone.InceptionTimeFeatureExtractor(1),
        millet_pooling.MILConjunctivePooling(128, n_clz, dropout=0.1, apply_positional_encoding=True),
    )


# --------------------------------------------------------------------------------------
# Config-driven builder (used by the "python main.py run" path)
#
# It reads a model config (from configs/models/<model>.yaml) and builds "encoder -> pooling head"
# by name, using the registries in models/encoders.py and models/pooling.py. When the yaml holds
# the same numbers as the SEA_* constants, this produces exactly the same network as make_sea_net.
# --------------------------------------------------------------------------------------
def build_model_from_config(model_cfg, n_clz: int, n_in: int = 1) -> EncoderPoolNet:
    """
    Build a full network (encoder + pooling head) from a model config.

    model_cfg : a model config (the "model_config" section, from configs/models/<model>.yaml).
    n_clz : number of classes for this dataset.
    n_in : number of input channels (1 here).
    returns : an EncoderPoolNet ready to be wrapped for training.
    raises NotImplementedError : if the config is a placeholder marked implemented: false.
    """
    if getattr(model_cfg, "implemented", True) is False:     # e.g. the transformer placeholder
        raise NotImplementedError(
            f"Model {getattr(model_cfg, 'name', '?')!r} is a placeholder and not implemented yet."
        )
    encoder = build_encoder(model_cfg.encoder, n_in=n_in)    # from seanet/models/encoders.py
    pool = build_pooling(model_cfg.pooling, encoder.d_out, n_clz)   # from seanet/models/pooling.py
    return EncoderPoolNet(encoder, pool)


# --------------------------------------------------------------------------------------
# Size helpers (used to show SEA-Net is smaller than the baseline)
# --------------------------------------------------------------------------------------
def num_params(net: nn.Module) -> int:
    """
    Count the trainable parameters (weights) in a network.

    net : any nn.Module.
    returns : total number of parameters.
    """
    return sum(p.numel() for p in net.parameters())


def state_dict_size_mb(net: nn.Module) -> float:
    """
    Measure how big the saved weights are in megabytes (what torch.save would write to disk).

    net : any nn.Module.
    returns : size in MB.
    """
    buffer = io.BytesIO()                                # save into memory instead of a real file
    torch.save(net.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 * 1024)     # bytes -> MB
