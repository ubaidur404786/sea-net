"""
scripts/check_multiscale.py - a quick safety check for the multi-scale encoders.

WHY THIS SCRIPT EXISTS:
    The dangerous bug in any multi-scale code is not a crash - it is a SHIFT. You average the series
    down, run the encoder, stretch the features back up, and everything is off by a few timesteps.
    Nothing errors. Accuracy barely moves. But NDCG quietly drops, because NDCG compares our per-timestep
    importance against WebTraffic's per-timestep ground truth, and a shifted map points at the wrong
    places. That is very hard to notice from a results table, so we test for it directly here.

    The test is simple: put a single spike (a "delta") into an otherwise flat series, push it through,
    and check the answer still peaks at the SAME index it went in at.

WHAT IT CHECKS:
    1. shapes   - every wrapper returns (B, d, T) with T unchanged (pooling heads depend on this).
    2. alignment- a spike at index i still comes out at index i.
    3. short    - very short UCR series (T = 8, 15) do not crash the pyramid.
    4. size     - how many parameters each multi-scale config adds over its plain baseline.

This script only BUILDS models and runs one forward pass. It does not train anything.

Run it:  python scripts/check_multiscale.py
"""
import os
import sys
from types import SimpleNamespace

import torch
from torch import nn

# make "import seanet" work when this file is run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seanet import features as FT
from seanet.config import load_config
from seanet.models.encoders import build_encoder
from seanet.models import build_model_from_config, num_params

PASS = "  [ok]  "
FAIL = "  [FAIL]"


class _Identity(nn.Module):
    """A fake encoder that returns its input unchanged. Used so we can test the WRAPPER on its own."""

    def __init__(self, n_in: int):
        super().__init__()
        self.d_out = n_in

    def forward(self, x):
        return x


@FT.register_encoder("test_identity")
def _build_identity(cfg, n_in: int) -> nn.Module:
    """Registered so MultiScalePyramid can build it by name, exactly like a real encoder."""
    return _Identity(n_in)


def _delta(length: int, index: int) -> torch.Tensor:
    """A flat series of zeros with a single 1.0 spike at `index`, shaped (1, 1, length)."""
    x = torch.zeros(1, 1, length)
    x[0, 0, index] = 1.0
    return x


def check_channel_shapes_and_alignment() -> bool:
    """The channel wrapper must keep T, and its rolling max must peak where the spike was."""
    print("\n1) multi-scale CHANNELS (Technique 3)")
    ok = True
    length, index = 200, 137
    x = _delta(length, index)

    chan = FT.MultiScaleChannels(windows=(3, 7, 15, 31), stats=("max", "min", "std"), add_diff_sign=True)
    out = chan(x)

    expected = 1 + 4 * 3 + 1                              # raw + (4 windows x 3 stats) + the sign channel
    if out.shape == (1, expected, length):
        print(f"{PASS} shape {tuple(out.shape)} - length kept, {expected} channels as expected")
    else:
        print(f"{FAIL} shape {tuple(out.shape)}, expected {(1, expected, length)}")
        ok = False

    # Alignment. Careful: a rolling MAX around a spike is FLAT, not pointy - every window that
    # contains the spike returns the same value, so with k=3 the output is equal at 136, 137 and 138.
    # argmax would just return the leftmost of that tie (136) and look like a shift when nothing moved.
    # The real invariant is the PLATEAU: it must be exactly k wide and centred on the spike.
    channels_per_window = len(chan.stats)                 # max, min, std -> the max is the 1st of each
    for w_i, k in enumerate(chan.windows):
        row = out[0, 1 + w_i * channels_per_window]       # the rolling-max channel for this window
        hit = (row >= row.max() - 1e-6).nonzero().flatten()
        first, last = int(hit[0]), int(hit[-1])
        centre, width = (first + last) / 2, len(hit)
        if centre == index and width == k:
            print(f"{PASS} rolling max k={k:2d}: plateau {first}..{last} "
                  f"(width {width}, centre {centre:.1f}) - centred on the spike")
        else:
            print(f"{FAIL} rolling max k={k:2d}: plateau {first}..{last} "
                  f"(width {width}, centre {centre:.1f}), expected width {k} centred on {index}")
            ok = False

    # a flat series must give std = 0 everywhere (the clamp inside must not invent a signal)
    flat_std = chan(torch.zeros(1, 1, 50))[0, 3]
    if float(flat_std.max()) < 1e-3:
        print(f"{PASS} std of a flat series is ~0 (max {float(flat_std.max()):.2e}) - no fake evidence")
    else:
        print(f"{FAIL} std of a flat series is {float(flat_std.max()):.4f}, should be ~0")
        ok = False
    return ok


def check_pyramid_alignment() -> bool:
    """The pyramid must keep T and must not shift the spike, at every fusion setting."""
    print("\n2) multi-scale PYRAMID (Technique 2a) - the important test")
    ok = True
    length, index = 256, 100
    x = _delta(length, index)
    base_cfg = SimpleNamespace(type="test_identity")      # a config block with just a type, like YAML gives

    for fusion in ("mean", "max", "attention", "concat"):
        pyr = FT.MultiScalePyramid(base_cfg, n_in=1, scales=(1, 2, 4, 8),
                                   fusion=fusion, d_attn=8, per_scale_bn=False)
        pyr.eval()                                        # eval mode: no dropout / BN updates
        with torch.no_grad():
            out = pyr(x)

        if out.shape[-1] != length:
            print(f"{FAIL} fusion={fusion:9s} length {out.shape[-1]}, expected {length}")
            ok = False
            continue

        peak = int(out[0].abs().mean(dim=0).argmax())     # where is the energy strongest?
        drift = abs(peak - index)
        # scale 8 averages 8 points into one, so a few timesteps of blur is expected and harmless.
        # Anything past half the coarsest scale means a real indexing bug, not blur.
        if drift <= 4:
            print(f"{PASS} fusion={fusion:9s} peak at {peak} (in at {index}, drift {drift}) - aligned")
        else:
            print(f"{FAIL} fusion={fusion:9s} peak at {peak} (in at {index}, drift {drift}) - SHIFTED")
            ok = False

    # the attention weights must be real weights: one per scale per timestep, adding up to 1
    pyr = FT.MultiScalePyramid(base_cfg, n_in=1, scales=(1, 2, 4, 8), fusion="attention", per_scale_bn=False)
    pyr.eval()
    with torch.no_grad():
        pyr(x)
    w = pyr.last_scale_weights                            # (B, S, 1, T)
    total = w.sum(dim=1)
    if w.shape == (1, 4, 1, length) and torch.allclose(total, torch.ones_like(total), atol=1e-5):
        print(f"{PASS} scale weights {tuple(w.shape)} and they sum to 1 at every timestep (plottable)")
    else:
        print(f"{FAIL} scale weights {tuple(w.shape)}, sums between {float(total.min()):.3f} "
              f"and {float(total.max()):.3f} (should all be 1)")
        ok = False
    return ok


def check_short_series() -> bool:
    """Very short UCR series must not crash (SmoothSubspace is only 15 long)."""
    print("\n3) short series (the shortest UCR sets)")
    ok = True
    base_cfg = SimpleNamespace(type="test_identity")      # a config block with just a type, like YAML gives
    for length in (8, 15, 24):
        try:
            pyr = FT.MultiScalePyramid(base_cfg, n_in=1, scales=(1, 2, 4, 8),
                                       fusion="attention", per_scale_bn=False)
            pyr.eval()
            with torch.no_grad():
                out = pyr(torch.randn(2, 1, length))
            assert out.shape[-1] == length, f"length changed to {out.shape[-1]}"
            print(f"{PASS} T={length:3d} -> {tuple(out.shape)}")
        except Exception as exc:                          # noqa: BLE001 - we want to report any failure
            print(f"{FAIL} T={length:3d} raised {type(exc).__name__}: {exc}")
            ok = False
    return ok


# How many classes to build the models with when counting parameters. WebTraffic has 10 classes, and
# the params column in results/SEA_NET/leaderboard.csv comes from real WebTraffic runs, so we use 10
# here too - otherwise the "baseline" column below would not match the leaderboard and comparing the
# two would mislead. (The class count only changes the tiny classifier head: 520 + 74*n_clz params.)
PARAM_COUNT_CLASSES = 10


def check_real_configs() -> bool:
    """Build every multi-scale config for real and show what it costs against its plain baseline."""
    print(f"\n4) the real multi-scale configs (built from YAML, {PARAM_COUNT_CLASSES} classes)")
    pairs = [
        ("seanet/seanet_bottleneck_mschan", "seanet/seanet_bottleneck_topk"),
        ("seanet/seanet_bottleneck_pyramid", "seanet/seanet_bottleneck_topk"),
        ("seanet/seanet_bottleneck_mschan_pyramid", "seanet/seanet_bottleneck_topk"),
        ("seanet/seanet_gated_mschan", "seanet/seanet_gated_mean_topk"),
        ("seanet/seanet_gated_pyramid", "seanet/seanet_gated_mean_topk"),
        ("seanet/seanet_inputgate_mschan", "seanet/seanet_inputgate_topk"),
    ]
    ok = True
    x = torch.randn(2, 1, 1008)                           # WebTraffic is 1008 long
    print(f"  {'config':40s} {'params':>9s} {'baseline':>9s} {'extra':>9s}")
    for new_name, base_name in pairs:
        try:
            new_cfg = load_config(overrides={"model": new_name})
            base_cfg = load_config(overrides={"model": base_name})
            n_clz = PARAM_COUNT_CLASSES
            new_net = build_model_from_config(new_cfg.model_config, n_clz=n_clz, n_in=1)
            base_net = build_model_from_config(base_cfg.model_config, n_clz=n_clz, n_in=1)

            new_net.eval()
            with torch.no_grad():
                out = new_net(x)
            # the interpretation MUST stay (batch, classes, T). This is the shape AOPCR and NDCG
            # read, so if a wrapper ever changed the length T, this is where we would catch it.
            if out["interpretation"].shape != (2, n_clz, 1008):
                print(f"{FAIL} {new_name}: interpretation {tuple(out['interpretation'].shape)}, "
                      f"expected {(2, n_clz, 1008)} - AOPCR/NDCG would break")
                ok = False
                continue

            n_new, n_base = num_params(new_net), num_params(base_net)
            print(f"  {new_name:40s} {n_new:9,d} {n_base:9,d} {n_new - n_base:+9,d}")
        except Exception as exc:                          # noqa: BLE001
            print(f"{FAIL} {new_name} raised {type(exc).__name__}: {exc}")
            ok = False
    return ok


def measure_receptive_field(encoder: nn.Module, length: int = 3001) -> int:
    """
    Measure how many input timesteps one output timestep can actually see.

    HOW: put a random series in, take the output at the MIDDLE position, and ask PyTorch which input
    positions that output depends on (autograd tells us - any input with a non-zero gradient was used).
    The distance from the first to the last such position IS the receptive field. Measuring it this way
    means we never have to work it out by hand from kernels and dilations, and it stays correct if the
    encoder changes.

    We use a random input, not zeros: ReLU has zero gradient at exactly 0, so an all-zero input would
    report a receptive field of nothing.
    """
    encoder = encoder.eval()                              # no dropout, no BatchNorm updates
    x = torch.randn(1, 1, length, requires_grad=True)
    out = encoder(x)
    out[0, :, out.shape[-1] // 2].sum().backward()
    used = (x.grad[0, 0].abs() > 0).nonzero().flatten()
    if len(used) == 0:
        return 0
    return int(used[-1] - used[0]) + 1


def check_receptive_fields() -> bool:
    """
    The most important check in this file: is each scale of the pyramid actually usable?

    A scale is only meaningful while the receptive field still FITS inside the shrunk series. Once the
    receptive field is bigger than the sequence, every position sees everything, the features stop
    varying along time, and that scale contributes smeared mush to the fusion. This is exactly what
    sank the first pyramid runs, so we now measure it instead of assuming.
    """
    print("\n5) receptive field vs scale length (WebTraffic is 1008 long)")
    series_length = 1008
    ok = True
    for config_name in ("seanet/seanet_bottleneck_pyramid", "seanet/seanet_gated_pyramid",
                        "seanet/seanet_bottleneck_mschan_pyramid"):
        try:
            cfg = load_config(overrides={"model": config_name})
            enc_cfg = cfg.model_config.encoder
            scales = tuple(getattr(enc_cfg, "scales", (1, 2, 4, 8)))
            # build ONLY the base encoder, on its own, to measure what one scale really sees
            base = build_encoder(enc_cfg.base, n_in=1)
            rf = measure_receptive_field(base)

            bad = [s for s in scales if rf > series_length / s]
            covers = rf * max(scales)                      # reach at the coarsest scale, in ORIGINAL steps
            mark = PASS if not bad else FAIL
            print(f"{mark} {config_name}")
            print(f"          base receptive field {rf}, scales {list(scales)}, "
                  f"coarsest reach {covers} original timesteps")
            for s in scales:
                ratio = rf / (series_length / s)
                note = "ok" if ratio <= 1 else "DEGENERATE - sees the whole shrunk series"
                print(f"          scale {s}: length {series_length // s:5d}  rf/length {ratio:5.2f}  {note}")
            if bad:
                print(f"          -> scales {bad} are degenerate. Use a SHALLOWER base "
                      f"(fewer n_blocks, smaller max_dilation).")
                ok = False
        except Exception as exc:                          # noqa: BLE001
            print(f"{FAIL} {config_name} raised {type(exc).__name__}: {exc}")
            ok = False
    return ok


def main() -> int:
    print("=" * 78)
    print("multi-scale encoder check - shapes, alignment, short series, size")
    print("=" * 78)
    results = [
        check_channel_shapes_and_alignment(),
        check_pyramid_alignment(),
        check_short_series(),
        check_real_configs(),
        check_receptive_fields(),
    ]
    print("\n" + "=" * 78)
    if all(results):
        print("ALL CHECKS PASSED - safe to run a smoke test.")
        return 0
    print("SOME CHECKS FAILED - fix these before training, the results would be misleading.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
