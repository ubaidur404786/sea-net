"""
seanet/model.py - the SEA-Net network (our model).

What this file is for:
    This is where the model is defined. SEA-Net is a small network made of two parts:
      1. an encoder that I wrote (a multi-scale, depthwise-separable TCN), and
      2. MILLET's Additive pooling head, reused as-is.
    The encoder turns each timestep of a series into a feature vector; the pooling head turns
    those per-timestep features into one class prediction plus an "importance per timestep"
    explanation.

Input / output of the network:
    Input  : a batch of series, shape (B, 1, T)   [B = batch, 1 = one channel, T = length].
    Output : a dict with
               "bag_logits"     (B, n_clz)     -> the class scores,
               "interpretation" (B, n_clz, T)  -> importance of each timestep (used by AOPCR/NDCG),
               "attn"           (B, T, 1)       -> the attention gate (used by the entropy penalty).

Related files:
    - Uses MILLET's InceptionTime backbone and pooling from millet/model/ (imported below).
    - seanet/train.py wraps make_sea_net() in a trainable model and fits it.
    - main.py ("params" command) calls make_sea_net / make_baseline + the size helpers to print
      how much smaller SEA-Net is than MILLET's InceptionTime+Conjunctive baseline.

Why the encoder is built the way it is (short version):
    - depthwise-separable convs  -> far fewer weights, which is what makes SEA-Net small.
    - three kernel sizes (5/11/23) added together -> the model sees narrow spikes AND broad
      shapes at the same time.
    - dilation capped at 16       -> each timestep only affects a local region, which keeps the
      "importance per timestep" honest (deleting a timestep really removes its effect).
    - zero padding                -> the series length T stays the same through the encoder, and
      very short inputs do not crash (AOPCR feeds the model shortened series).

The architecture is fixed. The only thing that changes per dataset is n_clz (number of classes).
"""
import io
from typing import Optional, Tuple

import torch
from torch import nn

from millet.model import backbone
from millet.model import pooling as millet_pooling

# --------------------------------------------------------------------------------------
# Fixed SEA-Net settings (same for every dataset). These are the values that worked best in my
# earlier experiments, so they are baked in here instead of being passed around.
# --------------------------------------------------------------------------------------
SEA_D = 128               # number of channels the encoder works with (also the pooling input size)
SEA_DROPOUT = 0.2         # dropout used in the encoder blocks and the pooling head
SEA_N_BLOCKS = 6          # how many multi-scale blocks are stacked
SEA_MAX_DILATION = 16     # the dilation is not allowed to grow past this (keeps the view local)
SEA_KERNELS: Tuple[int, ...] = (5, 11, 23)   # the three kernel sizes used in every block


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
# The SEA-Net encoder
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
    The full encoder: a stem conv, then a stack of MultiScaleSepBlocks whose dilation grows
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
# Builders (make the two networks we care about)
# --------------------------------------------------------------------------------------
def make_sea_net(n_clz: int, dropout: float = SEA_DROPOUT) -> EncoderPoolNet:
    """
    Build the SEA-Net network (my encoder + MILLET's Additive pooling).

    n_clz : number of classes for this dataset.
    dropout : dropout probability (defaults to the fixed SEA_DROPOUT).
    returns : an EncoderPoolNet ready to be wrapped for training (see seanet/train.py).
    """
    encoder = MSTCNSepEncoder(n_in=1, d=SEA_D, dropout=dropout)
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
