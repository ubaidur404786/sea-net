"""
seanet/models/encoders.py - the feature-extraction module (the encoders).

What this file is for:
    A feature extractor (encoder) is the first half of a model: it turns a raw series into a
    feature vector for every timestep. This file is the ONE place encoders live, and it keeps a
    small "registry" (a name -> builder table) so the model can pick an encoder by name from the
    config. Adding a new encoder is just: write the class, register it, done - no other file changes.

Input / output contract (the same for EVERY encoder, so they are interchangeable):
    Input  : a batch of series, shape (B, n_in, T).
    Output : per-timestep features, shape (B, d, T)   (T is unchanged).
    Every encoder also exposes an attribute `.d_out` = d, so the pooling head knows the width.

Naming rule: every registered name says WHO it came from, so a results folder or a figure label can
be read without opening any code.
    "sea_..." = OURS   (written for this project, defined in this file)
    "mil_..." = MILLET (reused from millet/, code untouched)

The encoders registered here today:
    OURS - the SEA-Net family (all multi-scale depthwise-separable TCNs, defined in this file):
    - "sea_mstcn_sep"            : the plain encoder (the original SEA-Net).
    - "sea_mstcn_sep_gated"      : plus one summary gate on top.
    - "sea_mstcn_sep_spiketrend" : two branches (smooth trend + high-pass spike), mixed and gated.
    - "sea_mstcn_sep_bottleneck" : the smallest one (the d->d 1x1 conv is factorised through d->r->d).
    - "sea_mstcn_sep_inputgate"  : a gate read straight off the raw input.
    - "sea_mstcn_sep_recon"      : adds features made from the reconstruction residual.

    OURS - multi-scale WRAPPERS. These do not replace an encoder, they wrap one: the encoder
    being wrapped is nested under "base" in the config, so they work on top of ANY encoder above.
    - "sea_multiscale_channels"  : adds rolling max/min/std channels to the input (length T unchanged).
    - "sea_multiscale_pyramid"   : runs the SAME encoder on the series downsampled by 1, 2, 4, 8 and
                                   fuses the results back to length T (attention / mean / max / concat).

    MILLET - the paper's own backbones, reused unchanged from millet/:
    - "mil_inceptiontime" : InceptionTime (the paper's baseline encoder).
    - "mil_fcn"           : FCN.
    - "mil_resnet"        : ResNet.

Related files:
    - seanet/models/build.py   -> build_model_from_config() calls build_encoder() from here, then pairs the
      encoder with a pooling head (seanet/models/pooling.py).
    - seanet/models/pooling.py -> the second half of a model (turns (B, d, T) into a prediction).
    - configs/models/<model>.yaml -> the "encoder" block picks the type and its settings.
"""
from typing import Callable, Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

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
        H_out  = H * (1 + gate)                 # (B, d, T): RESIDUAL gate - keep H, amplify good channels

    Why residual (H * (1 + gate)) and not a plain gate or a LayerNorm: a plain multiply or a per-timestep
    LayerNorm forces every timestep to a similar magnitude, which erased the "this timestep matters more"
    signal and destroyed our AOPCR / NDCG interpretability. Keeping H and only amplifying (factor in [1,2])
    preserves the per-timestep structure the MIL explanations rely on, while still letting the summary turn
    useful channels up.

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
        # residual gate: multiplier is (1 + gate) in [1,2], so we KEEP the original features and only
        # amplify the useful channels. This preserves per-timestep importance (needed for AOPCR/NDCG).
        return h * (1.0 + gate.unsqueeze(-1)) # (B, d, T)


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
# Spike/trend dual-branch encoder.
#
# Idea (from our brainstorm): class evidence in a time series comes in two very different shapes -
#   - a TREND: a slow, smooth pattern over many timesteps (rising / falling / flat).
#   - a SPIKE: a sharp jump at just one or two timesteps ("needle in a haystack").
# One normal conv stack is biased toward smooth patterns, so it can miss tiny spikes. So we run TWO
# small branches side by side: the usual mstcn_sep for trends, and a cheap "spike" branch that first
# removes the smooth part of the signal so only the sudden changes are left. We glue the two together
# with a 1x1 conv, then re-use the same SummaryGate. Output stays (B, d, T), so pooling is unchanged.
# --------------------------------------------------------------------------------------
class SpikeBranch(nn.Module):
    """
    A tiny branch that highlights sharp, local changes (spikes).

    How it works: first take a local average of the signal (a smooth version of it). Subtract that from
    the raw signal - what is left is the "high-pass" part: near zero on smooth stretches, but large right
    where the signal jumps. Then a small conv turns that spike signal into d_spike feature channels.

        smooth   = local_average(x)            # a blurred copy of the series
        highpass = x - smooth                   # ~0 on smooth parts, big at sudden jumps (the spikes)
        out      = ReLU(BN(Conv1d(highpass)))   # small features that describe the spikes
    """

    def __init__(self, n_in: int = 1, d_spike: int = 16, smooth_kernel: int = 5):
        """
        n_in : input channels (1 here).
        d_spike : how many spike feature channels to make (small on purpose).
        smooth_kernel : window size of the local average (odd, so the length stays the same).
        """
        super().__init__()
        # AvgPool1d with stride 1 = a moving average; padding keeps the length T the same.
        self.smooth = nn.AvgPool1d(kernel_size=smooth_kernel, stride=1, padding=smooth_kernel // 2)
        self.conv = nn.Sequential(
            nn.Conv1d(n_in, d_spike, kernel_size=3, padding=1),   # small k=3 conv: looks at a tiny window
            nn.BatchNorm1d(d_spike),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, n_in, T) -> (B, d_spike, T) spike features."""
        highpass = x - self.smooth(x)         # remove the smooth part, keep the sudden changes
        return self.conv(highpass)


class MSTCNSepSpikeTrendEncoder(nn.Module):
    """
    Two branches (trend + spike), mixed and then self-gated. Output is (B, d, T) with .d_out = d, so it
    slots into the registry like every other encoder.

        trend  = mstcn_sep(x)                  # (B, d, T)        smooth / long-range features
        spike  = SpikeBranch(x)                # (B, d_spike, T)  sharp / local features
        h      = Conv1d_1x1( concat[trend, spike] )   # (B, d, T)  mix the two back to d channels
        out    = SummaryGate(h)                # (B, d, T)        same self-gating as before
    """

    def __init__(self, n_in: int = 1, d: int = SEA_D, n_blocks: int = SEA_N_BLOCKS,
                 dropout: float = SEA_DROPOUT, max_dilation: int = SEA_MAX_DILATION,
                 kernels: Tuple[int, ...] = SEA_KERNELS, summary: str = "last",
                 d_spike: int = 16, smooth_kernel: int = 5):
        """Same arguments as the gated encoder, plus d_spike / smooth_kernel for the spike branch."""
        super().__init__()
        self.d_out = d
        self.trend = MSTCNSepEncoder(n_in, d, n_blocks, dropout, max_dilation, kernels)
        self.spike = SpikeBranch(n_in, d_spike, smooth_kernel)
        self.mix = nn.Conv1d(d + d_spike, d, kernel_size=1)   # 1x1 conv: combine both branches back to d
        self.gate = SummaryGate(d, summary=summary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, n_in, T) -> (B, d, T) gated features from both branches."""
        t = self.trend(x)                     # (B, d, T) trend features
        s = self.spike(x)                     # (B, d_spike, T) spike features
        h = torch.cat([t, s], dim=1)          # (B, d + d_spike, T) stack the two branches
        h = self.mix(h)                       # (B, d, T) mix them back to d channels
        return self.gate(h)                   # (B, d, T) self-gate, same as before


# --------------------------------------------------------------------------------------
# Bottleneck version of the encoder (a SMALLER model).
#
# Idea: almost all the parameters in a block live in the pointwise 1x1 conv Conv1d(d, d), which costs
# d*d weights. We shrink it by FACTORISING that one conv into two cheaper 1x1 convs through a narrow
# middle of width r:
#     Conv1d(d, r)  ->  Conv1d(r, d)      where r = d // bottleneck_ratio   (r << d)
# cost 2*d*r instead of d*d. With ratio 4 that is about half the block params, for a small accuracy
# cost. Everything else (depthwise multi-scale convs, residual, unchanged length T) stays the same.
# --------------------------------------------------------------------------------------
class MultiScaleSepBottleneckBlock(nn.Module):
    """Same as MultiScaleSepBlock, but the d->d pointwise conv is split into d->r->d (cheaper)."""

    def __init__(self, d: int, dilation: int, dropout: float, kernels: Tuple[int, ...] = SEA_KERNELS,
                 bottleneck_ratio: int = 4):
        super().__init__()
        r = max(1, d // bottleneck_ratio)                    # width of the narrow middle (r << d)
        self.units = nn.ModuleList()
        for _ in range(2):
            branches = nn.ModuleList([
                nn.Conv1d(d, d, k, padding=(k - 1) * dilation // 2, dilation=dilation, groups=d)
                for k in kernels
            ])
            self.units.append(nn.ModuleDict({
                "branches": branches,
                # the factorised pointwise: d -> r -> d, replaces the expensive d -> d 1x1 conv
                "reduce": nn.Conv1d(d, r, kernel_size=1),
                "expand": nn.Conv1d(r, d, kernel_size=1),
                "bn": nn.BatchNorm1d(d),
            }))
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for unit in self.units:
            multi = sum(branch(y) for branch in unit["branches"])   # add the 3 kernel outputs
            mixed = unit["expand"](unit["reduce"](multi))           # d -> r -> d instead of d -> d
            y = self.drop(self.act(unit["bn"](mixed)))
        return self.act(x + y)                               # residual, then relu


class MSTCNSepBottleneckEncoder(nn.Module):
    """The SEA-Net encoder but with bottleneck blocks, so it has far fewer parameters. .d_out = d."""

    def __init__(self, n_in: int = 1, d: int = SEA_D, n_blocks: int = SEA_N_BLOCKS,
                 dropout: float = SEA_DROPOUT, max_dilation: int = SEA_MAX_DILATION,
                 kernels: Tuple[int, ...] = SEA_KERNELS, bottleneck_ratio: int = 4):
        super().__init__()
        self.d_out = d
        self.stem = nn.Conv1d(n_in, d, kernel_size=7, padding=3)
        self.blocks = nn.Sequential(*[
            MultiScaleSepBottleneckBlock(d, dilation=min(2 ** i, max_dilation), dropout=dropout,
                                         kernels=kernels, bottleneck_ratio=bottleneck_ratio)
            for i in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


# --------------------------------------------------------------------------------------
# Input-gate encoder (the multiplicative "input x stem" skip idea).
#
# Idea (from our brainstorm): let the RAW input decide, at every timestep, which feature channels
# matter. A small conv reads x and builds a per-timestep, per-channel gate in [0,1]; we multiply the
# encoder features by (1 + gate). Unlike SummaryGate (ONE gate for the whole series), this gate can
# change along time, so a sharp event in the input can switch channels on right where it happens.
# The residual form (1 + gate, so a factor in [1,2]) keeps the original features, so per-timestep
# importance (AOPCR / NDCG) still survives.
# --------------------------------------------------------------------------------------
class InputGate(nn.Module):
    """Build a per-timestep, per-channel gate from the raw input and use it to re-weight features."""

    def __init__(self, n_in: int, d: int, kernel_size: int = 7):
        super().__init__()
        # a small conv reads the raw input and outputs one gate value per channel per timestep
        self.to_gate = nn.Conv1d(n_in, d, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.to_gate(x))                # (B, d, T) in [0,1], built from raw input
        return h * (1.0 + gate)                              # residual multiplicative skip


class MSTCNSepInputGateEncoder(nn.Module):
    """The slim encoder, then re-weighted by a gate built from the RAW input. .d_out = d."""

    def __init__(self, n_in: int = 1, d: int = SEA_D, n_blocks: int = SEA_N_BLOCKS,
                 dropout: float = SEA_DROPOUT, max_dilation: int = SEA_MAX_DILATION,
                 kernels: Tuple[int, ...] = SEA_KERNELS, gate_kernel: int = 7):
        super().__init__()
        self.d_out = d
        self.backbone = MSTCNSepEncoder(n_in, d, n_blocks, dropout, max_dilation, kernels)
        self.input_gate = InputGate(n_in, d, kernel_size=gate_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)                                 # (B, d, T) encoder features
        return self.input_gate(x, h)                         # gate built from the raw input x


# --------------------------------------------------------------------------------------
# Reconstruction-residual encoder (the "unmatched area" idea).
#
# Idea (from our brainstorm): the encoder learns the NORMAL / smooth shape of a series. If we try to
# rebuild the input from those features and subtract it, what is LEFT (the residual) is the part the
# smooth model could not explain - the surprising bits that often carry the class evidence. We turn
# that residual into a few extra feature channels and mix them back in.
# NOTE: there is NO extra reconstruction loss for now, so the "recon" conv is just learned end-to-end
# (this is really a "learned residual" feature). We can add a real reconstruction loss later if it helps.
# --------------------------------------------------------------------------------------
class MSTCNSepReconEncoder(nn.Module):
    """Trend features + a residual (input - rebuilt input) branch, mixed back to d channels. .d_out = d."""

    def __init__(self, n_in: int = 1, d: int = SEA_D, n_blocks: int = SEA_N_BLOCKS,
                 dropout: float = SEA_DROPOUT, max_dilation: int = SEA_MAX_DILATION,
                 kernels: Tuple[int, ...] = SEA_KERNELS, d_res: int = 16):
        super().__init__()
        self.d_out = d
        self.trend = MSTCNSepEncoder(n_in, d, n_blocks, dropout, max_dilation, kernels)
        self.recon = nn.Conv1d(d, n_in, kernel_size=1)       # rebuild the input from the trend features
        self.res_conv = nn.Sequential(
            nn.Conv1d(n_in, d_res, kernel_size=3, padding=1), # small features from the residual
            nn.BatchNorm1d(d_res),
            nn.ReLU(),
        )
        self.mix = nn.Conv1d(d + d_res, d, kernel_size=1)    # combine trend + residual back to d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trend(x)                                    # (B, d, T) smooth / trend features
        recon = self.recon(h)                                # (B, n_in, T) rebuilt input
        residual = x - recon                                 # (B, n_in, T) the "unmatched area"
        r = self.res_conv(residual)                          # (B, d_res, T) features from the surprise
        return self.mix(torch.cat([h, r], dim=1))            # (B, d, T)


# --------------------------------------------------------------------------------------
# Multi-scale INPUT CHANNELS (our "Technique 3").
#
# Idea: instead of feeding the encoder ONE channel (the raw series), feed it a few extra channels
# that describe the same timestep at BIGGER windows: the rolling max, min and std over a window of
# 3, 7, 15, 31 points around it. The length T never changes, so every timestep still lines up with
# its original position - AOPCR and NDCG keep working exactly as before.
#
# Why max / min / std and NOT just the rolling mean: a rolling mean is a LINEAR operation, and the
# stem conv is linear too, so a mean channel is something the encoder could mostly learn by itself.
# A max, a min or a std is NON-linear - a convolution can never compute them, no matter how it is
# trained. So those channels are real new information. (This is the same trick that made MultiRocket
# work: add cheap non-linear summaries instead of more layers.)
#
# "mean" and "range" are available too (KNOWN_STATS below). A rolling MEAN is linear, so it is the
# weakest of the five - keep it only if an experiment says it helps. A rolling RANGE (max - min) is
# non-linear and free, because max and min are already computed.
#
# Cost: only the STEM gets wider (Conv1d(1, d, 7) -> Conv1d(C, d, 7)). With d=64 and C=13 that is
# about +5k parameters and almost no extra compute. It is by far the cheapest thing we can try.
# --------------------------------------------------------------------------------------
class MultiScaleChannels(nn.Module):
    """
    Turn a series (B, n_in, T) into (B, n_in * n_out, T) by adding rolling-statistic channels.

    For every window size k we can add the rolling max / min / std / mean / range at every timestep.
    Windows must be ODD so the window is centred on its timestep (an even window would sit half a
    step to one side and slowly shift all the features - a silent way to lose NDCG).

    We pad with "replicate" (copy the edge value outwards) instead of zeros. Zero padding would make
    the first and last few timesteps look artificially like the series mean, which invents evidence
    at exactly the places MIL attention likes to look.
    """

    # the statistics we know how to compute (the config can pick any subset of these)
    KNOWN_STATS = ("mean", "max", "min", "std", "range")

    def __init__(self, windows: Tuple[int, ...] = (3, 7, 15, 31),
                 stats: Tuple[str, ...] = ("max", "min", "std"),
                 add_diff_sign: bool = True):
        """
        windows : the rolling window sizes, all ODD (3, 7, 15, 31 = radius 1, 3, 7, 15).
        stats : which statistics to compute in each window (see KNOWN_STATS).
        add_diff_sign : also add one channel of sign(x[t] - x[t-1]) -> -1 down / 0 flat / +1 up.
                        The difference itself is skipped on purpose: a conv can already learn it with
                        weights [-1, +1]. The SIGN is non-linear, so a conv cannot, which is what
                        makes it worth a channel.
        """
        super().__init__()
        for k in windows:
            if k % 2 == 0 or k < 1:
                raise ValueError(f"window {k} must be an ODD positive number so it stays centred "
                                 f"on its timestep.")
        # keep the order the config gave, but drop repeats - asking for "max" twice would only
        # build two identical channels and waste parameters in the stem.
        kept, seen = [], set()
        for s in stats:
            if s not in self.KNOWN_STATS:
                raise ValueError(f"Unknown stat {s!r}. Options: {self.KNOWN_STATS}")
            if s not in seen:
                seen.add(s)
                kept.append(s)
        self.windows = tuple(windows)
        self.stats = tuple(kept)
        self.add_diff_sign = bool(add_diff_sign)
        # how many channels we make PER input channel: the original + every (window, stat) pair + sign
        self.n_out = 1 + len(self.windows) * len(self.stats) + (1 if self.add_diff_sign else 0)

    @staticmethod
    def _pad(x: torch.Tensor, k: int) -> torch.Tensor:
        """Copy the edge values outwards by k//2 on both sides, so pooling keeps the length T."""
        return F.pad(x, (k // 2, k // 2), mode="replicate")

    def _roll_mean(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Rolling average over a window of k, stride 1 -> same length T."""
        return F.avg_pool1d(self._pad(x, k), kernel_size=k, stride=1)

    def _roll_max(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Rolling maximum over a window of k."""
        return F.max_pool1d(self._pad(x, k), kernel_size=k, stride=1)

    def _roll_min(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Rolling minimum = minus the rolling max of the flipped signal."""
        return -self._roll_max(-x, k)

    def _roll_std(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Rolling standard deviation, using the usual shortcut  std = sqrt(mean(x^2) - mean(x)^2).
        The clamp keeps the value inside the square root positive: without it, tiny floating-point
        error on a flat stretch can make it a very small NEGATIVE number and sqrt gives NaN.
        """
        m = self._roll_mean(x, k)
        m2 = self._roll_mean(x * x, k)
        return torch.sqrt(torch.clamp(m2 - m * m, min=1e-8))

    def _window_channels(self, x: torch.Tensor, k: int) -> list:
        """
        Every statistic we were asked for, inside ONE window of size k, in the config's order.

        The rolling max and min are computed at most once each and then reused, because "range" is
        just max - min. So asking for [max, min, range] costs the same as asking for [max, min].
        """
        need_max = ("max" in self.stats) or ("range" in self.stats)
        need_min = ("min" in self.stats) or ("range" in self.stats)
        roll_max = self._roll_max(x, k) if need_max else None
        roll_min = self._roll_min(x, k) if need_min else None
        out = []
        for stat in self.stats:
            if stat == "mean":
                out.append(self._roll_mean(x, k))
            elif stat == "max":
                out.append(roll_max)
            elif stat == "min":
                out.append(roll_min)
            elif stat == "std":
                out.append(self._roll_std(x, k))
            else:                                            # "range" = how far the signal swings
                out.append(roll_max - roll_min)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, n_in, T) -> (B, n_in * n_out, T). The length T is unchanged."""
        channels = [x]                                       # channel 0 is always the raw series
        for k in self.windows:                               # order stays window-major, then stat
            channels.extend(self._window_channels(x, k))
        if self.add_diff_sign:
            d = torch.zeros_like(x)
            d[:, :, 1:] = torch.sign(x[:, :, 1:] - x[:, :, :-1])  # diff=current - previous  # first step has no "previous", stays 0 , check difference from previous values 
# +1 increase / going up , -1 decrease / going down, 0 flat 
            channels.append(d)
        return torch.cat(channels, dim=1)


class MultiScaleChannelEncoder(nn.Module):
    """
    A WRAPPER: build the extra channels, then run any existing encoder on top of them.

    It does not replace an encoder - it wraps one. So `sea_mstcn_sep_bottleneck` (or gated, or
    inputgate, ...) stays exactly as it is, and the only thing that changes is that its stem now
    reads C channels instead of 1. Output is still (B, d, T) with .d_out = d, so every pooling head
    works unchanged.
    """

    def __init__(self, base_cfg, n_in: int = 1, windows: Tuple[int, ...] = (3, 7, 15, 31),
                 stats: Tuple[str, ...] = ("max", "min", "std"), add_diff_sign: bool = True,
                 norm_channels: bool = True):
        """
        base_cfg : the nested "base" block of the config - the encoder to wrap.
        n_in : channels coming IN from the dataset (1 normally).
        windows / stats / add_diff_sign : passed to MultiScaleChannels.
        norm_channels : normalise the channels before the encoder sees them. Keep this ON.

        WHY norm_channels MATTERS (this is what made the first multi-scale runs lose accuracy):
            the raw series is z-normalised, so it sits around 0 with spread 1. The channels we build
            from it do NOT. A rolling max is pushed up, a rolling min is pushed down, a std is always
            positive and small, and the sign channel only ever takes -1, 0 or +1. Feeding 13 channels
            with completely different centres and spreads into one conv makes training unfair: the
            big channels dominate the gradient and the small ones are effectively ignored.
            One BatchNorm puts every channel on the same footing first. It costs 2 numbers per
            channel (28 params at the default 14 channels), which is nothing.
        """
        super().__init__()
        self.channels = MultiScaleChannels(windows, stats, add_diff_sign)
        n_channels = n_in * self.channels.n_out
        # put every derived channel on the same scale before the encoder sees them
        self.norm = nn.BatchNorm1d(n_channels) if norm_channels else None
        # the wrapped encoder must be built with the WIDER input, so its stem matches
        self.base = build_encoder(base_cfg, n_in=n_channels)
        self.d_out = self.base.d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, n_in, T) -> (B, d, T)."""
        h = self.channels(x)                                 # (B, n_in * n_out, T)
        if self.norm is not None:
            h = self.norm(h)
        return self.base(h)


# --------------------------------------------------------------------------------------
# Multi-scale PYRAMID (our "Technique 2a" - the closest thing to the cancer paper, done properly).
#
# Idea: the pathology paper looks at the same tissue at 10x and 20x magnification. The 1-D version
# of "lower magnification" is NOT a bigger window - it is DOWNSAMPLING. So we average the series down
# by 2, 4, 8, run the SAME encoder on each shrunk copy, stretch the features back to length T, and
# mix them together.
#
# Why downsampling and not just bigger windows: our encoder already sees ~667 timesteps around every
# point (kernels 5/11/23 at dilations 1,2,4,8), so a bigger window shows it nothing new. But averaging
# CHANGES THE SIGNAL: it cancels noise and leaves only the slow shape, so the very same filters now
# respond to something different. That is genuinely new information.
#
# One encoder is SHARED by every scale, so the parameter count barely moves - which is the whole
# point of this project. Compute is about 2x (1 + 1/2 + 1/4 + 1/8), not 5x.
# --------------------------------------------------------------------------------------
class MultiScalePyramid(nn.Module):
    """
    Run one shared encoder at several downsampling factors, then fuse the features back into (B, d, T).

    fusion options:
        "attention" : learn a weight per scale PER TIMESTEP (softmax over scales, so the weights add
                      up to 1). Cheapest expressive option, and the weights can be PLOTTED - they say
                      "at this timestep the model trusted scale 4", which is the 1-D version of
                      Figure 6 in the cancer paper.
        "mean"      : plain average over scales. No parameters, but it dilutes: if only one scale
                      holds the evidence, its signal is divided by the number of scales.
        "max"       : strongest scale wins per channel. No parameters, but it can never COMBINE two
                      scales, and gradient only reaches one scale at a time.
        "concat"    : stack all scales and mix with a 1x1 conv. Most expressive, but costs
                      S * d * d parameters - about +20k at d=64 with 5 scales, which is half of the
                      bottleneck model. Kept for the ablation, not as the default.

    A softmax over the SCALE axis is safe for our metrics: it never normalises across TIME, so the
    "this timestep matters more" signal that AOPCR and NDCG depend on survives untouched. (This is the
    same lesson SummaryGate above records: normalising across time destroyed our interpretability.)

    ================== READ THIS BEFORE PICKING A `base` ENCODER ==================
    THE BASE MUST BE SHALLOW. Use a SMALL receptive field and let the SCALES provide the range.

    Our first attempt ignored this and lost 15 accuracy points. It wrapped the normal 4-block encoder,
    whose receptive field is about 667 timesteps. On WebTraffic (T = 1008) that gives:

        scale 1 -> length 1008, receptive field / length = 0.66   healthy
        scale 2 -> length  504, ratio 1.3                          degenerate
        scale 4 -> length  252, ratio 2.6                          degenerate
        scale 8 -> length  126, ratio 5.3                          degenerate

    Once the receptive field is BIGGER than the sequence, every output position sees the whole shrunk
    series, so the features go almost constant along time - and at scale 8 the dilation-8 block pads by
    88 on each side of a 126-long input, so there is more zero padding than signal. Three of the four
    scales were feeding the fusion smeared mush, which is why AOPCR fell from 2.78 to 1.03.

    The rule: pick the base so that  receptive_field <= length / max(scales).  A 2-block base with
    max_dilation 1 has a receptive field of about 95, so at scale 8 it still covers 95 * 8 = 760
    ORIGINAL timesteps - the same reach as the deep encoder, but built from four cheap local views
    instead of one deep stack. That is how an image pyramid works, and it is the whole point.

    scripts/check_multiscale.py measures the receptive field for you and flags degenerate scales.
    ==============================================================================
    """

    # F.interpolate looks at how many dimensions the tensor has. Ours is 3-D, (B, d, T), so only
    # these four modes are legal. "bilinear" / "bicubic" are 2-D image modes and "trilinear" is 3-D
    # volume - PyTorch raises on them here, so we refuse them ourselves with a clearer message.
    #   "linear"        : straight line between the two nearest feature values (the default, smooth).
    #   "nearest"       : copy the closest value - NO blending, so a feature is never invented.
    #   "nearest-exact" : the same idea with the off-by-half-a-pixel bug fixed (prefer it to nearest).
    #   "area"          : average pooling; when stretching UP it behaves like nearest.
    KNOWN_INTERPOLATIONS = ("linear", "nearest", "nearest-exact", "area")
    # only the "linear" family accepts align_corners; passing it to the others is a TypeError
    ALIGN_CORNERS_MODES = ("linear",)

    def __init__(self, base_cfg, n_in: int = 1, scales: Tuple[int, ...] = (1, 2, 4, 8),
                 fusion: str = "attention", d_attn: int = 8, per_scale_bn: bool = True,
                 interpolation: str = "linear", align_corners: bool = False):
        """
        base_cfg : the nested "base" block of the config - the encoder to run at every scale.
        n_in : input channels.
        scales : the downsampling factors. 1 = the original series, 2 = averaged in pairs, etc.
        fusion : "attention" | "mean" | "max" | "concat".
        d_attn : hidden width of the little scale-attention network (only used by "attention").
        per_scale_bn : give each scale its own BatchNorm. A shrunk series has different statistics
                       from the raw one, and one shared BatchNorm would be dragged to a bad middle.
                       Costs only 2*d numbers per scale, so it is nearly free.
        interpolation : how a shrunk scale is stretched back to length T (see KNOWN_INTERPOLATIONS).
                        "linear" is the old behaviour. "nearest" / "nearest-exact" invent nothing -
                        they only repeat values - which is the honest choice if we do not want a
                        made-up value sitting between two real timesteps.
        align_corners : only used by "linear". False keeps the old behaviour.
        """
        super().__init__()
        if fusion not in ("attention", "mean", "max", "concat"):
            raise ValueError(f"Unknown fusion {fusion!r} (options: attention, mean, max, concat).")
        if interpolation not in self.KNOWN_INTERPOLATIONS:
            raise ValueError(f"Unknown interpolation {interpolation!r}. On a 3-D (B, d, T) tensor "
                             f"only these work: {self.KNOWN_INTERPOLATIONS}.")
        self.scales = tuple(scales)
        self.fusion = fusion
        self.interpolation = str(interpolation)
        self.align_corners = bool(align_corners)
        self.base = build_encoder(base_cfg, n_in=n_in)        # ONE encoder, reused at every scale
        d = self.base.d_out
        self.d_out = d
        # one BatchNorm per scale (or none)
        self.norms = nn.ModuleList([nn.BatchNorm1d(d) for _ in self.scales]) if per_scale_bn else None
        if fusion == "attention":
            # gives one score per timestep per scale; softmax over scales turns them into weights
            self.score = nn.Sequential(
                nn.Conv1d(d, d_attn, kernel_size=1),
                nn.Tanh(),
                nn.Conv1d(d_attn, 1, kernel_size=1),
            )
        elif fusion == "concat":
            self.mix = nn.Conv1d(d * len(self.scales), d, kernel_size=1)
        # the last batch's scale weights, saved so a figure can show which scale was used where
        self.last_scale_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, n_in, T) -> (B, d, T). T is the SAME as the input length."""
        length = x.shape[-1]
        feats = []
        for i, scale in enumerate(self.scales):
            # min(scale, length) protects very short UCR series: a window bigger than the whole
            # series would crash the pooling. ceil_mode keeps the last partial window instead of
            # throwing those timesteps away.
            step = min(scale, length)
            xs = x if step == 1 else F.avg_pool1d(x, kernel_size=step, stride=step, ceil_mode=True)
            h = self.base(xs)                                # (B, d, T/scale) - SHARED weights
            if self.norms is not None:
                h = self.norms[i](h)
            if h.shape[-1] != length:                        # stretch back so every scale lines up
                # align_corners is only a valid argument for the "linear" family; for "nearest",
                # "nearest-exact" and "area" PyTorch raises if we pass it at all.
                extra = ({"align_corners": self.align_corners}
                         if self.interpolation in self.ALIGN_CORNERS_MODES else {})
                h = F.interpolate(h, size=length, mode=self.interpolation, **extra)
            feats.append(h)

        if self.fusion == "concat":
            return self.mix(torch.cat(feats, dim=1))         # (B, d, T)

        stacked = torch.stack(feats, dim=1)                  # (B, S, d, T)
        if self.fusion == "mean":
            return stacked.mean(dim=1)
        if self.fusion == "max":
            return stacked.max(dim=1).values

        # "attention": one weight per (scale, timestep), summing to 1 over the scales
        scores = torch.stack([self.score(h) for h in feats], dim=1)   # (B, S, 1, T)
        weights = torch.softmax(scores, dim=1)                        # (B, S, 1, T)
        self.last_scale_weights = weights.detach()                    # keep a copy for plotting
        return (stacked * weights).sum(dim=1)                         # (B, d, T)


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


@register_encoder("sea_mstcn_sep")
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


@register_encoder("sea_mstcn_sep_gated")
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


@register_encoder("sea_mstcn_sep_spiketrend")
def _build_mstcn_sep_spiketrend(cfg, n_in: int) -> nn.Module:
    """Build the spike/trend dual-branch encoder. Same config as mstcn_sep_gated plus d_spike/smooth_kernel."""
    return MSTCNSepSpikeTrendEncoder(
        n_in=n_in,
        d=cfg.d,
        n_blocks=cfg.n_blocks,
        dropout=cfg.dropout,
        max_dilation=cfg.max_dilation,
        kernels=tuple(cfg.kernels),
        summary=getattr(cfg, "summary", "last"),      # last was the best gate summary on WebTraffic
        d_spike=getattr(cfg, "d_spike", 16),          # size of the spike branch (small on purpose)
        smooth_kernel=getattr(cfg, "smooth_kernel", 5),   # window of the local average in the spike branch
    )


@register_encoder("sea_mstcn_sep_bottleneck")
def _build_mstcn_sep_bottleneck(cfg, n_in: int) -> nn.Module:
    """Build the bottleneck (smaller) encoder. Same config as mstcn_sep plus optional bottleneck_ratio."""
    return MSTCNSepBottleneckEncoder(
        n_in=n_in,
        d=cfg.d,
        n_blocks=cfg.n_blocks,
        dropout=cfg.dropout,
        max_dilation=cfg.max_dilation,
        kernels=tuple(cfg.kernels),
        bottleneck_ratio=getattr(cfg, "bottleneck_ratio", 4),   # r = d // ratio (bigger ratio = smaller)
    )


@register_encoder("sea_mstcn_sep_inputgate")
def _build_mstcn_sep_inputgate(cfg, n_in: int) -> nn.Module:
    """Build the input-gate encoder. Same config as mstcn_sep plus optional gate_kernel."""
    return MSTCNSepInputGateEncoder(
        n_in=n_in,
        d=cfg.d,
        n_blocks=cfg.n_blocks,
        dropout=cfg.dropout,
        max_dilation=cfg.max_dilation,
        kernels=tuple(cfg.kernels),
        gate_kernel=getattr(cfg, "gate_kernel", 7),   # window of the conv that reads the raw input
    )


@register_encoder("sea_mstcn_sep_recon")
def _build_mstcn_sep_recon(cfg, n_in: int) -> nn.Module:
    """Build the reconstruction-residual encoder. Same config as mstcn_sep plus optional d_res."""
    return MSTCNSepReconEncoder(
        n_in=n_in,
        d=cfg.d,
        n_blocks=cfg.n_blocks,
        dropout=cfg.dropout,
        max_dilation=cfg.max_dilation,
        kernels=tuple(cfg.kernels),
        d_res=getattr(cfg, "d_res", 16),   # how many feature channels to make from the residual
    )


@register_encoder("sea_multiscale_channels")
def _build_multiscale_channels(cfg, n_in: int) -> nn.Module:
    """
    Build the multi-scale CHANNEL wrapper (Technique 3).

    The config nests the encoder being wrapped under "base", e.g.

        encoder:
          type: sea_multiscale_channels
          windows: [3, 7, 15, 31]
          stats: [mean, max, min, std, range]   # any subset of MultiScaleChannels.KNOWN_STATS,
                                                # in any order; repeats are dropped
          add_diff_sign: true
          base:
            type: sea_mstcn_sep_bottleneck
            d: 64
            ...
    """
    return MultiScaleChannelEncoder(
        base_cfg=cfg.base,
        n_in=n_in,
        windows=tuple(getattr(cfg, "windows", (3, 7, 15, 31))),
        stats=tuple(getattr(cfg, "stats", ("max", "min", "std"))),
        add_diff_sign=getattr(cfg, "add_diff_sign", True),
        norm_channels=getattr(cfg, "norm_channels", True),   # keep ON: the channels have different scales
    )


@register_encoder("sea_multiscale_pyramid")
def _build_multiscale_pyramid(cfg, n_in: int) -> nn.Module:
    """
    Build the multi-scale PYRAMID wrapper (Technique 2a).

    Like the channel wrapper, the encoder being wrapped is nested under "base". `base` can itself be
    another wrapper, so "pyramid on top of channels" is just one config nested inside another.

        encoder:
          type: sea_multiscale_pyramid
          scales: [1, 2, 4, 8]
          fusion: attention         # attention | mean | max | concat
          interpolation: linear     # linear | nearest | nearest-exact | area  (3-D tensors only)
          align_corners: false      # read only when interpolation is "linear"
          base:
            ...
    """
    return MultiScalePyramid(
        base_cfg=cfg.base,
        n_in=n_in,
        scales=tuple(getattr(cfg, "scales", (1, 2, 4, 8))),
        fusion=getattr(cfg, "fusion", "attention"),
        d_attn=getattr(cfg, "d_attn", 8),
        per_scale_bn=getattr(cfg, "per_scale_bn", True),
        # how the shrunk scales are stretched back to length T. "linear" = the old behaviour.
        interpolation=getattr(cfg, "interpolation", "linear"),
        align_corners=getattr(cfg, "align_corners", False),   # only read when interpolation=linear
    )


@register_encoder("mil_inceptiontime")
def _build_inceptiontime(cfg, n_in: int) -> nn.Module:
    """Build MILLET's InceptionTime backbone. Output width = 4 * out_channels (default 128)."""
    out_channels = getattr(cfg, "out_channels", 32)
    enc = backbone.InceptionTimeFeatureExtractor(n_in, out_channels=out_channels)
    enc.d_out = out_channels * 4                              # this backbone concatenates 4 branches
    return enc


@register_encoder("mil_fcn")
def _build_fcn(cfg, n_in: int) -> nn.Module:
    """Build MILLET's FCN backbone (fixed 128-channel output)."""
    enc = backbone.FCNFeatureExtractor(n_in)
    enc.d_out = 128
    return enc


@register_encoder("mil_resnet")
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
