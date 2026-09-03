"""
scripts/check_new_features.py - one command that checks everything added on 2026-09-02.

Nothing here trains. It builds models, runs a forward pass, computes the loss, calls backward and
looks at the shapes and the gradients. On a laptop it takes about a minute.

    python scripts/check_new_features.py

What it checks, in order:
    1. MultiScaleChannels  - all five statistics, correct values, bad input refused
    2. MultiScalePyramid   - all four legal 1-D interpolation modes, image-only modes refused
    3a. TopKConjunctivePooling - still the plain top-k mean, untouched by the voting work
    3b. SimpleVotingPooling - every timestep votes once. Checks the forward pass really is a
        HARD vote and that gradients still flow (a plain argmax count cannot be trained at all)
    3c. TopKVotingPooling - only the top-k vote: shapes, finite logits, real gradients, the vote
        counts add up, and the threshold never produces a NaN even when it silences everything
    4. every model config in configs/models/ still builds, runs and trains a step
    5. the deployment bundle saves and reloads to bit-identical outputs

Why a script and not a unit-test file: this project has no test framework set up, and the point is
that YOU can run it and read the output, not that a CI robot can. It prints what it found and exits
non-zero if anything is wrong.
"""
import math
import os
import shutil
import sys
from types import SimpleNamespace

import torch
from torch import nn

# make "import seanet" work when this file is run directly from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seanet.config import available_models, load_config, model_folder_name   # noqa: E402
from seanet.deployment import load_bundle, save_bundle                        # noqa: E402
from seanet.models.build import build_model_from_config, num_params           # noqa: E402
from seanet.models.encoders import (ENCODER_REGISTRY, MultiScaleChannels,     # noqa: E402
                                    MultiScalePyramid, build_encoder)
from seanet.models.pooling import (SimpleVotingPooling, TopKConjunctivePooling,  # noqa: E402
                                   TopKVotingPooling, build_pooling)
from seanet.training import make_model                                        # noqa: E402

FAILURES = []


def check(label, condition, detail=""):
    """Print one PASS/FAIL line and remember the failures so the script can exit non-zero."""
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f"   {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ======================================================================================
section("1. MultiScaleChannels - the five rolling statistics")
# ======================================================================================
x = torch.randn(2, 1, 64)
for stats in [("mean",), ("max",), ("min",), ("std",), ("range",),
              ("mean", "max", "min", "std", "range"), ("max", "min", "std")]:
    m = MultiScaleChannels(windows=(3, 7, 15, 31), stats=stats, add_diff_sign=True)
    out = m(x)
    want = 1 + 4 * len(stats) + 1
    check(f"stats={str(stats):40s}", out.shape == (2, want, 64) and torch.isfinite(out).all(),
          f"-> {tuple(out.shape)}")

m = MultiScaleChannels(windows=(3,), stats=("max", "min"), add_diff_sign=False)
check("a repeated stat is dropped", MultiScaleChannels(windows=(3,), stats=("max", "max", "min")).stats
      == ("max", "min"))

sig = torch.tensor([[[1.0, 5.0, 2.0, 8.0, 3.0]]])
ch = MultiScaleChannels(windows=(3,), stats=("mean", "max", "min", "std", "range"),
                        add_diff_sign=False)(sig)[0]
check("range == max - min", torch.allclose(ch[5], ch[2] - ch[3]))
check("max is really the rolling max", torch.allclose(ch[2], torch.tensor([5.0, 5.0, 8.0, 8.0, 8.0])))

for bad, why in [(dict(windows=(4,)), "an even window"), (dict(stats=("median",)), "an unknown stat")]:
    try:
        MultiScaleChannels(**bad)
        check(f"{why} is refused", False)
    except ValueError:
        check(f"{why} is refused", True)


# ======================================================================================
section("2. MultiScalePyramid - configurable interpolation")
# ======================================================================================
base = SimpleNamespace(type="sea_mstcn_sep_bottleneck", d=16, n_blocks=2, dropout=0.1,
                       max_dilation=1, kernels=[5, 11], bottleneck_ratio=4)
for mode in MultiScalePyramid.KNOWN_INTERPOLATIONS:
    good = True
    for fusion in ("attention", "mean", "max", "concat"):
        p = MultiScalePyramid(base, n_in=1, scales=(1, 2, 4), fusion=fusion, interpolation=mode)
        o = p(x)
        good = good and o.shape == (2, p.d_out, 64) and torch.isfinite(o).all()
    check(f"interpolation={mode:14s} (all 4 fusions)", good)

for bad in ("bilinear", "bicubic", "trilinear"):
    try:
        MultiScalePyramid(base, interpolation=bad)
        check(f"the image-only mode {bad!r} is refused", False)
    except ValueError:
        check(f"the image-only mode {bad!r} is refused", True)


# ======================================================================================
section("3a. TopKConjunctivePooling - the plain top-k mean (must be untouched)")
# ======================================================================================
torch.manual_seed(0)
B, T, d, C = 4, 100, 16, 3
z = torch.randn(B, d, T)
y = torch.randint(0, C, (B,))
ce = nn.CrossEntropyLoss(label_smoothing=0.13)

# the top-k head must be the plain mean of the k largest values - nothing else
head = TopKConjunctivePooling(d, C, d_attn=8, dropout=0.0).eval()
with torch.no_grad():
    xp = head._prepare(z, None)
    g = head.attention_head(xp) * head.instance_classifier(xp)
    plain_formula = torch.topk(g, k=math.ceil(0.1 * T), dim=1).values.mean(dim=1)
    from_head = head(z)["bag_logits"]
check("bag_logits == mean of the k largest gated values",
      float((plain_formula - from_head).abs().max()) == 0.0)
check("it has NO voting settings on it",
      not any(hasattr(head, n) for n in
              ("pooling_method", "confidence_threshold", "vote_temperature", "vote_hard", "scale")),
      "(voting lives in TopKVotingPooling)")
check("it has exactly the parameters it always had",
      sum(pr.numel() for pr in head.parameters()) == 214)

h = TopKConjunctivePooling(d, C)
out = h(z); loss = ce(out["bag_logits"], y); loss.backward()
check("forward + backward still fine",
      out["bag_logits"].shape == (B, C) and out["interpretation"].shape == (B, C, T)
      and torch.isfinite(loss), f"loss={float(loss):.4f}")


section("3b. SimpleVotingPooling - every timestep votes once")
hsv = SimpleVotingPooling(d, C)
out = hsv(z)
loss = ce(out["bag_logits"], y)
loss.backward()
grads = [pr.grad for pr in hsv.parameters() if pr.grad is not None]
check("shapes are right",
      out["bag_logits"].shape == (B, C) and out["interpretation"].shape == (B, C, T))
check("loss is finite", bool(torch.isfinite(loss)), f"loss={float(loss):.4f}")
check("gradients actually flow (a plain argmax count would give NONE)",
      bool(grads) and any(float(gr.abs().sum()) > 0 for gr in grads),
      f"max|grad|={max(float(gr.abs().max()) for gr in grads):.3e}")
check("it adds NO parameters over the plain top-k head",
      sum(pr.numel() for pr in hsv.parameters()) == 214)

# the forward pass must be the REAL hard vote, not the smooth surrogate
hsv.eval()
with torch.no_grad():
    xp = hsv._prepare(z, None)
    gg = hsv.attention_head(xp) * hsv.instance_classifier(xp)
    plain = torch.nn.functional.one_hot(gg.argmax(dim=-1), num_classes=C).float().sum(dim=1)
    hsv(z)
check("forward output IS the plain argmax vote count",
      float((hsv.last_vote_counts - plain).abs().max()) == 0.0,
      f"first series: {[int(v) for v in hsv.last_vote_counts[0]]}")
check("every timestep voted exactly once (counts sum to T)",
      bool(torch.allclose(hsv.last_vote_counts.sum(dim=1), torch.full((B,), float(T)))))

# log(counts / T) and log(counts) must train identically - the difference is a per-row constant
la = ce(torch.log(plain / T + 1e-8), y)
lb = ce(torch.log(plain + 1e-8), y)
check("log(counts/T) == log(counts) as far as the loss is concerned",
      bool(torch.allclose(la, lb, atol=1e-6)), f"{float(la):.6f} vs {float(lb):.6f}")

cfg = SimpleNamespace(type="sea_simple_voting", d_attn=8, dropout=0.2,
                      positional_encoding=True, top_frac=0.1)   # top_frac is inherited + ignored
h = build_pooling(cfg, d_in=d, n_clz=C)
check("builds from a config, and an inherited top_frac is harmlessly ignored",
      isinstance(h, SimpleVotingPooling) and not hasattr(h, "top_frac"))


section("3c. TopKVotingPooling - only the top-k vote")
cases = {
    "voting (soft, default)": {},
    "voting, temperature=0.1": dict(temperature=0.1),
    "voting, hard (straight-through)": dict(hard=True),
    "voting + threshold 0.002": dict(confidence_threshold=0.002),
    "voting hard + threshold 0.002": dict(hard=True, confidence_threshold=0.002),
    "voting + threshold 0.999 (silences all)": dict(confidence_threshold=0.999),
    "voting, top_frac=1.0": dict(top_frac=1.0),
}
for label, kwargs in cases.items():
    h = TopKVotingPooling(d, C, **kwargs)
    out = h(z)
    loss = ce(out["bag_logits"], y)
    loss.backward()
    grads = [pr.grad for pr in h.parameters() if pr.grad is not None]
    ok = (out["bag_logits"].shape == (B, C)
          and out["interpretation"].shape == (B, C, T)
          and torch.isfinite(out["bag_logits"]).all()
          and torch.isfinite(loss)
          and any(float(gr.abs().sum()) > 0 for gr in grads))
    check(f"{label:40s}", ok, f"loss={float(loss):.4f}")

# the votes really are votes
hv = TopKVotingPooling(d, C, hard=True, top_frac=0.3).eval()
with torch.no_grad():
    hv(z)
counts = hv.last_vote_counts
check("hard votes are whole numbers summing to k",
      bool(torch.allclose(counts.sum(dim=1), torch.full((B,), 30.0)))
      and bool(torch.allclose(counts, counts.round())),
      f"first series: {[int(v) for v in counts[0]]}")
# a SOFT vote is still exactly one vote per timestep - it may just be split between classes,
# so the counts must still add up to the number of voters
hsoft = TopKVotingPooling(d, C, top_frac=0.3).eval()
with torch.no_grad():
    hsoft(z)
check("soft votes also sum to k (one vote each, possibly split)",
      bool(torch.allclose(hsoft.last_vote_counts.sum(dim=1), torch.full((B,), 30.0), atol=1e-4)),
      f"first series: {[round(float(v), 2) for v in hsoft.last_vote_counts[0]]}")

# log(share), not softmax(share): feeding the logits through softmax must give back the vote share
hs = TopKVotingPooling(d, C, top_frac=0.3).eval()
with torch.no_grad():
    logits = hs(z)["bag_logits"]
    share = hs.last_vote_counts / hs.last_vote_counts.sum(dim=1, keepdim=True)
    check("softmax(bag_logits) reproduces the vote share (scale starts at 1)",
          bool(torch.allclose(torch.softmax(logits, dim=1), share, atol=1e-5)))

# edge cases
o = TopKVotingPooling(d, 2, confidence_threshold=0.5)(torch.randn(2, d, 3))
check("C=2, T=3 still works", o["bag_logits"].shape == (2, 2) and torch.isfinite(o["bag_logits"]).all())
o = TopKVotingPooling(d, C, top_frac=5.0)(torch.randn(2, d, 4))
check("top_frac > 1 is clipped to T",
      o["bag_logits"].shape == (2, C) and torch.isfinite(o["bag_logits"]).all())
try:
    TopKVotingPooling(d, C, temperature=0.0)
    check("temperature <= 0 is refused", False)
except ValueError:
    check("temperature <= 0 is refused", True)

# built from a config
cfg = SimpleNamespace(type="sea_topk_voting", d_attn=8, dropout=0.2, positional_encoding=True,
                      top_frac=0.1, temperature=0.5, hard=True, confidence_threshold=0.002)
h = build_pooling(cfg, d_in=d, n_clz=C)
check("build_pooling wires the voting settings up",
      isinstance(h, TopKVotingPooling) and h.hard and h.confidence_threshold == 0.002
      and h.temperature == 0.5)
old_cfg = SimpleNamespace(type="sea_topk_conjunctive", d_attn=8, dropout=0.2,
                          positional_encoding=True, top_frac=0.1)
h = build_pooling(old_cfg, d_in=d, n_clz=C)
check("an existing topk config still builds the plain head",
      isinstance(h, TopKConjunctivePooling) and h.top_frac == 0.1)


# ======================================================================================
section("4. every model config still builds, runs and takes a training step")
# ======================================================================================
built, skipped, broken = 0, 0, []
for name in available_models():
    try:
        cfg = load_config(overrides={"model": name})
        net = build_model_from_config(cfg.model_config, 3, n_in=1)
    except NotImplementedError:
        skipped += 1
        continue
    except Exception as e:
        broken.append((name, f"build: {type(e).__name__}: {e}"))
        continue
    try:
        out = net(torch.randn(2, 1, 128))
        assert out["bag_logits"].shape == (2, 3)
        assert out["interpretation"].shape == (2, 3, 128)
        loss = ce(out["bag_logits"], torch.randint(0, 3, (2,)))
        loss.backward()
        assert torch.isfinite(loss)
        assert any(float(p.grad.abs().sum()) > 0 for p in net.parameters() if p.grad is not None)
        built += 1
    except Exception as e:
        broken.append((name, f"forward: {type(e).__name__}: {e}"))
check(f"{built} configs build + forward + backward ({skipped} placeholder skipped)", not broken)
for name, why in broken:
    print(f"        {name}: {why}")


# ======================================================================================
section("5. the deployment bundle saves and reloads exactly")
# ======================================================================================
OUT = os.path.join("results", "_selftest", "deploy")
shutil.rmtree(OUT, ignore_errors=True)
cfg = load_config(overrides={"model": "top/top_bottleneck_topk"})
model = make_model(3, torch.device("cpu"), model_cfg=cfg.model_config, n_in=1)
model.net.eval()
path = save_bundle(model, "WebTraffic", 0, OUT, model_cfg=cfg.model_config,
                   model_id=model_folder_name(cfg), metrics={"test_acc": 0.0},
                   series_length=1008, n_in=1)
check("bundle written", os.path.exists(path), f"{num_params(model.net)} params")
net2, payload = load_bundle(path)
xb = torch.randn(2, 1, 1008)
with torch.no_grad():
    a, b = model.net(xb), net2(xb)
for key in ("bag_logits", "interpretation"):
    check(f"reloaded model reproduces {key}", float((a[key] - b[key]).abs().max()) == 0.0)
for f in ("WebTraffic__seed0_config.yaml", "WebTraffic__seed0_meta.json", "README.md"):
    check(f"{f} written", os.path.exists(os.path.join(OUT, f)))
shutil.rmtree(os.path.join("results", "_selftest"), ignore_errors=True)


# ======================================================================================
print(f"\n{'=' * 78}")
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print("   -", f)
    sys.exit(1)
print("EVERYTHING PASSED")
