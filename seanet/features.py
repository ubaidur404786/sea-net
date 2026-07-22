"""
seanet/features.py - the feature-extraction module (the encoders).

What this file is for:
    A feature extractor (encoder) is the first half of a model: it turns a raw series into a
    feature vector for every timestep. This file is the ONE place encoders live, and it keeps a
    small "registry" (a name -> builder table) so the model can pick an encoder by name from the
    config. Adding a new encoder is just: write the class, register it, done - no other file changes.

Input / output contract (the same for EVERY encoder, so they are interchangeable):
    Input  : a batch of series, shape (B, n_in, T).
    Output : per-timestep features, shape (B, d, T)   (T is unchanged).
    Every encoder also exposes an attribute `.d_out` = d, so the pooling head knows the width.

The encoders registered here today:
    - "mstcn_sep"     : SEA-Net's own multi-scale depthwise-separable TCN (defined in this file).
    - "inceptiontime" : MILLET's InceptionTime backbone (reused from millet/).
    - "fcn"           : MILLET's FCN backbone (reused from millet/).
    - "resnet"        : MILLET's ResNet backbone (reused from millet/).

Related files:
    - seanet/model.py   -> build_model_from_config() calls build_encoder() from here, then pairs the
      encoder with a pooling head (seanet/pooling.py).
    - seanet/pooling.py -> the second half of a model (turns (B, d, T) into a prediction).
    - configs/models/<model>.yaml -> the "encoder" block picks the type and its settings.
"""
from typing import Callable, Dict, Tuple

import torch
from torch import nn

from millet.model import backbone

# --------------------------------------------------------------------------------------
# Fixed SEA-Net encoder settings (same for every dataset). These are the values that worked best
# in earlier experiments, so they are the defaults here. The config file repeats them so results
# do not change; keeping them here too means MSTCNSepEncoder still works if you build it directly.
# --------------------------------------------------------------------------------------
SEA_D = 128               # number of channels the encoder works with (also the pooling input size)
SEA_DROPOUT = 0.2         # dropout used in the encoder blocks
SEA_N_BLOCKS = 6          # how many multi-scale blocks are stacked
SEA_MAX_DILATION = 16     # the dilation is not allowed to grow past this (keeps the view local)
SEA_KERNELS: Tuple[int, ...] = (5, 11, 23)   # the three kernel sizes used in every block


# --------------------------------------------------------------------------------------
# The SEA-Net encoder (moved here from model.py; the code is unchanged)
# --------------------------------------------------------------------------------------
class MultiScaleSepBlock(nn.Module):
    """
    One residual block of the encoder.

    It runs two identical "units" and then adds the input back (a residual connection). Each unit:
        [ depthwise conv k=5  +  depthwise conv k=11  +  depthwise conv k=23 ]   (the three summed)
        -> pointwise 1x1 conv (mixes channels) -> BatchNorm -> ReLU -> Dropout
    All convs use the block's dilation and zero "same" padding, so the length T does not change.
    "Depthwise" (groups == channels) means each channel gets its own filter, which is cheap.
    """

    def __init__(self, d: int, dilation: int, dropout: float, kernels: Tuple[int, ...] = SEA_KERNELS):
        """
        d : number of channels.
        dilation : spacing between filter taps (bigger = wider view of time).
        dropout : dropout probability.
        kernels : the kernel sizes to run in parallel (default 5, 11, 23).
        """
        super().__init__()
        self.units = nn.ModuleList()
        for _ in range(2):                                   # two units per block
            # one depthwise conv per kernel size; padding keeps the length the same
            branches = nn.ModuleList([
                nn.Conv1d(d, d, k, padding=(k - 1) * dilation // 2, dilation=dilation, groups=d)
                for k in kernels
            ])
            self.units.append(nn.ModuleDict({
                "branches": branches,
                "pointwise": nn.Conv1d(d, d, kernel_size=1),   # 1x1 conv that mixes the channels
                "bn": nn.BatchNorm1d(d),
            }))
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, d, T) input.
        returns : (B, d, T) output, same shape (residual block).
        """
        y = x
        for unit in self.units:                              # run the two units in sequence
            multi = sum(branch(y) for branch in unit["branches"])   # add the 3 kernel outputs together
            y = self.drop(self.act(unit["bn"](unit["pointwise"](multi))))   # mix + normalise + relu + dropout
        return self.act(x + y)                               # add the input back (residual), then relu


class MSTCNSepEncoder(nn.Module):
    """
    The full SEA-Net encoder: a stem conv, then a stack of MultiScaleSepBlocks whose dilation grows
    1, 2, 4, ... but is capped at max_dilation. Output length T is the same as the input.
    It exposes .d_out (= d) so the pooling head knows what size to expect.
    """

    def __init__(self, n_in: int = 1, d: int = SEA_D, n_blocks: int = SEA_N_BLOCKS,
                 dropout: float = SEA_DROPOUT, max_dilation: int = SEA_MAX_DILATION,
                 kernels: Tuple[int, ...] = SEA_KERNELS):
        """
        n_in : number of input channels (1 for these datasets).
        d : channel width used inside the encoder.
        n_blocks : how many blocks to stack.
        dropout : dropout probability passed to each block.
        max_dilation : cap on the dilation.
        kernels : the kernel sizes used in every block.
        """
        super().__init__()
        self.d_out = d
        self.stem = nn.Conv1d(n_in, d, kernel_size=7, padding=3)   # lift 1 channel up to d channels
        # block i uses dilation min(2**i, max_dilation): 1, 2, 4, 8, 16, 16 for 6 blocks
        self.blocks = nn.Sequential(*[
            MultiScaleSepBlock(d, dilation=min(2 ** i, max_dilation), dropout=dropout, kernels=kernels)
            for i in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, n_in, T) input series.
        returns : (B, d, T) per-timestep features.
        """
        return self.blocks(self.stem(x))


# --------------------------------------------------------------------------------------
# Self-gating "summary" version of the encoder.
#
# Idea (from our own brainstorm): after the encoder makes per-timestep features H (B, d, T), build ONE
# summary vector s (B, d) for the whole series, turn it into a per-channel gate in [0, 1], and multiply
# every timestep by that gate. So the whole-series summary decides "which feature channels matter" and
# turns the useful ones up and the useless ones down. This is cheap (one Linear(d, d)) and it does NOT
# collapse time, so the output is still (B, d, T) and every MIL pooling head still works unchanged.
# --------------------------------------------------------------------------------------
class SummaryGate(nn.Module):
    """
    Pool the features over time into one summary vector, make a per-channel gate from it, and use that
    gate to re-weight every timestep.

        s      = summarise_over_time(H)        # (B, d): one number per channel for the whole series
        gate   = sigmoid(W s)                  # (B, d): a weight in [0,1] for each channel
        H_out  = H * gate                       # (B, d, T): scale every timestep by its channel gate
        H_out  = LayerNorm(H_out)               # keep the values in a stable range after scaling

    summary : how to squeeze time into one vector -
        "max"  -> take the strongest value each channel reaches anywhere (good for spikes / needle-in-haystack)
        "mean" -> the average level of each channel (good for smooth trends)
        "last" -> just the last timestep. NOTE: our conv uses symmetric padding (not causal), so the last
                  point is only a LOCAL view, not a true whole-series summary - "max"/"mean" are the
                  robust choices. "last" is here only so we can compare all three like you asked.
    """

    def __init__(self, d: int, summary: str = "max"):
        """
        d : number of channels (same as the encoder width).
        summary : "max" | "mean" | "last" - how to pool over time.
        """
        super().__init__()
        self.summary = summary
        self.to_gate = nn.Linear(d, d)        # turns the summary vector into a per-channel gate
        self.norm = nn.LayerNorm(d)           # normalise channels after gating (stops values drifting)

    def _summarise(self, h: torch.Tensor) -> torch.Tensor:
        """h : (B, d, T) -> (B, d) one summary value per channel."""
        if self.summary == "mean":
            return h.mean(dim=2)
        if self.summary == "last":
            return h[:, :, -1]
        return h.max(dim=2).values            # default "max": strongest value each channel reaches

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h : (B, d, T) -> (B, d, T) gated features (same shape, so pooling is unaffected)."""
        s = self._summarise(h)                # (B, d) whole-series summary
        gate = torch.sigmoid(self.to_gate(s)) # (B, d) per-channel weight in [0,1]
        h = h * gate.unsqueeze(-1)            # (B, d, T) re-weight every timestep by its channel gate
        # LayerNorm needs channels last: (B, d, T) -> (B, T, d) -> norm -> back to (B, d, T)
        return self.norm(h.transpose(1, 2)).transpose(1, 2)


class MSTCNSepGatedEncoder(nn.Module):
    """
    The slim MSTCN-separable encoder followed by one SummaryGate. Output is still (B, d, T), and it
    exposes .d_out = d, so it drops into the registry exactly like the plain encoder.
    """

    def __init__(self, n_in: int = 1, d: int = SEA_D, n_blocks: int = SEA_N_BLOCKS,
                 dropout: float = SEA_DROPOUT, max_dilation: int = SEA_MAX_DILATION,
                 kernels: Tuple[int, ...] = SEA_KERNELS, summary: str = "max"):
        """Same arguments as MSTCNSepEncoder, plus `summary` for the gate (see SummaryGate)."""
        super().__init__()
        self.d_out = d
        # reuse the existing encoder as the backbone, then add the gate on top
        self.backbone = MSTCNSepEncoder(n_in, d, n_blocks, dropout, max_dilation, kernels)
        self.gate = SummaryGate(d, summary=summary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, n_in, T) -> (B, d, T) gated per-timestep features."""
        return self.gate(self.backbone(x))


# --------------------------------------------------------------------------------------
# The encoder registry (name -> builder). This is what makes encoders swappable by config.
# --------------------------------------------------------------------------------------
# A "builder" is a function that takes (encoder_cfg, n_in) and returns an encoder module with .d_out.
ENCODER_REGISTRY: Dict[str, Callable] = {}


def register_encoder(name: str) -> Callable:
    """
    Decorator that adds an encoder builder to the registry under `name`.

    Use it like:
        @register_encoder("my_encoder")
        def _build_my_encoder(cfg, n_in):
            enc = MyEncoder(...)
            enc.d_out = <output width>
            return enc

    name : the string used in the config (encoder.type).
    returns : the decorator that stores the builder and returns it unchanged.
    """
    def decorator(builder: Callable) -> Callable:
        ENCODER_REGISTRY[name] = builder
        return builder
    return decorator


@register_encoder("mstcn_sep")
def _build_mstcn_sep(cfg, n_in: int) -> nn.Module:
    """Build the SEA-Net encoder from its config block."""
    return MSTCNSepEncoder(
        n_in=n_in,
        d=cfg.d,
        n_blocks=cfg.n_blocks,
        dropout=cfg.dropout,
        max_dilation=cfg.max_dilation,
        kernels=tuple(cfg.kernels),
    )


@register_encoder("mstcn_sep_gated")
def _build_mstcn_sep_gated(cfg, n_in: int) -> nn.Module:
    """Build the self-gating SEA-Net encoder. Same config as mstcn_sep plus an optional `summary`."""
    return MSTCNSepGatedEncoder(
        n_in=n_in,
        d=cfg.d,
        n_blocks=cfg.n_blocks,
        dropout=cfg.dropout,
        max_dilation=cfg.max_dilation,
        kernels=tuple(cfg.kernels),
        summary=getattr(cfg, "summary", "max"),   # default to "max" if the config does not name it
    )


@register_encoder("inceptiontime")
def _build_inceptiontime(cfg, n_in: int) -> nn.Module:
    """Build MILLET's InceptionTime backbone. Output width = 4 * out_channels (default 128)."""
    out_channels = getattr(cfg, "out_channels", 32)
    enc = backbone.InceptionTimeFeatureExtractor(n_in, out_channels=out_channels)
    enc.d_out = out_channels * 4                              # this backbone concatenates 4 branches
    return enc


@register_encoder("fcn")
def _build_fcn(cfg, n_in: int) -> nn.Module:
    """Build MILLET's FCN backbone (fixed 128-channel output)."""
    enc = backbone.FCNFeatureExtractor(n_in)
    enc.d_out = 128
    return enc


@register_encoder("resnet")
def _build_resnet(cfg, n_in: int) -> nn.Module:
    """Build MILLET's ResNet backbone (fixed 128-channel output)."""
    enc = backbone.ResNetFeatureExtractor(n_in)
    enc.d_out = 128
    return enc


def build_encoder(encoder_cfg, n_in: int = 1) -> nn.Module:
    """
    Build an encoder from its config block, using the registry.

    encoder_cfg : the "encoder" section of a model config (must have .type).
    n_in : number of input channels (1 here).
    returns : an encoder module that maps (B, n_in, T) -> (B, d_out, T) and has a .d_out attribute.
    raises ValueError : if the encoder type is not registered, or the builder forgot to set .d_out.
    """
    kind = encoder_cfg.type
    if kind not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder type {kind!r}. Registered: {sorted(ENCODER_REGISTRY)}")
    encoder = ENCODER_REGISTRY[kind](encoder_cfg, n_in)
    if not hasattr(encoder, "d_out"):                        # the pooling head needs this
        raise ValueError(f"Encoder {kind!r} did not set .d_out (needed by the pooling head).")
    return encoder
