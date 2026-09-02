# 16. The deployment bundle (towards the ESP32)

`results.csv` only keeps numbers. You cannot put an accuracy on a microcontroller. This guide is
about the folder that keeps the **model itself**.

---

## 1. What gets saved, and when

After a real (non-smoke) run, `seanet/deployment.py` writes one bundle into

```text
results/top_results/SEA_NET/<model_id>/deploy/
```

One bundle per (dataset, seed). Every file shares the stem `<dataset>__seed<N>`:

| file | size | what it is |
|---|---|---|
| `<stem>.pt` | ~1.4 MB | **the important one.** Weights *and* the config that built them. |
| `<stem>_config.yaml` | ~1 KB | the same model config, readable. Drop it into `configs/models/` to retrain. |
| `<stem>_meta.json` | ~2 KB | input shape, preprocessing, classes, metrics, torch version, git commit, date. |
| `<stem>_traced.pt` | ~1.5 MB | TorchScript. `torch.jit.load` — runs with **no project code**. |
| `<stem>.onnx` | ~1.4 MB | ONNX opset 17. The starting point for every ESP32 toolchain. |
| `README.md` | — | written once per folder; the same instructions, next to the files. |

### When it is written

`configs/main.yaml`:

```yaml
output:
  save_deploy: webtraffic     # webtraffic (default) | always | never
```

* `webtraffic` — only for WebTraffic. That is the dataset we would actually deploy, and it stops a
  129-dataset UCR sweep from writing ~200 MB of weights nobody will use.
* `always` — every dataset.
* `never` — off.

Smoke runs never write one; a 3-epoch model is not something to deploy.

Saving is wrapped in `try/except`: if the disk is full or `onnx` is missing, it prints a note and
the training run finishes normally. Whatever failed is recorded inside `_meta.json`, so you can see
it later.

---

## 2. Getting the exact model back

```python
from seanet.deployment import load_bundle
import torch

net, meta = load_bundle("results/top_results/SEA_NET/<model_id>/deploy/WebTraffic__seed0.pt")

x = torch.zeros(1, 1, meta["series_length"])      # (batch, channels, time)
out = net(x)
out["bag_logits"]        # (1, C)  RAW logits - no softmax has been applied
out["interpretation"]    # (1, C, T)  importance of every timestep
```

`load_bundle` rebuilds the network **from the saved config** and then loads the weights with
`strict=True`. If anything at all does not line up it raises, instead of quietly handing you a
half-loaded model. That strictness is the point: it is what proves the bundle is complete.

Verified on `top/top_bottleneck_topk`: the rebuilt network reproduces both `bag_logits` and
`interpretation` with a maximum absolute difference of **0.0**.

### Without any project code at all

```python
import torch
m = torch.jit.load("WebTraffic__seed0_traced.pt")
bag_logits, interpretation = m(torch.zeros(1, 1, 1008))
```

Same file works from C++ (libtorch), which is useful for a laptop-side latency benchmark before the
board work starts.

---

## 3. Three things to remember on the device

**1. The input must be z-normalised per series.** Every series is turned into mean 0 / std 1 before
it ever reaches the model (`seanet/preprocessing.py`). If the device feeds raw counts, the model
sees a distribution it has never met and the prediction is meaningless. Do the normalisation in the
firmware, on the same window the model was trained on.

**2. `bag_logits` are raw logits.** Arg-max for the class; softmax only if you want probabilities.
The training loss did the softmax internally, so the model never learned to output one.

**3. The exported length is fixed.** The top-k head computes `k = ceil(top_frac * T)` in ordinary
Python, so the tracer freezes both `k` and `T` into the graph. Only the **batch** axis of the
`.onnx` and `_traced.pt` files is dynamic. Feeding a different length would silently use the wrong
`k`. Need another length? Re-export from the `.pt` bundle:

```python
from seanet.deployment import load_bundle, _ExportWrapper
import torch
net, meta = load_bundle("WebTraffic__seed0.pt")
torch.onnx.export(_ExportWrapper(net).eval(), torch.zeros(1, 1, 512), "model_T512.onnx",
                  input_names=["series"], output_names=["bag_logits", "interpretation"],
                  opset_version=17)
```

The `.pt` bundle itself has no length limit — it is a normal PyTorch module.

---

## 4. The road to the ESP32

Nothing here has been run on hardware yet; this is the plan, not a result.

```text
<stem>.onnx
   |
   |-- ESP-DL (Espressif's own converter)          <- try this first on an ESP32-S3
   |
   +-- onnx2tf / onnx-tf  ->  TFLite  ->  TFLite-Micro
```

Points to check when you get there, in this order:

1. **Ops.** Our encoder is depthwise-separable Conv1d + BatchNorm + ReLU + a linear head — all
   standard. The two things to watch are `topk` in the pooling head and, if you use it,
   `F.interpolate` in the pyramid encoder. If `topk` is not supported by the converter, run the
   encoder on-device and do the tiny pooling head in C by hand (it is one sort plus a mean).
2. **Size.** `top_bottleneck_topk` is 41 K parameters ≈ 160 KB as float32, ≈ 40 KB after int8
   quantisation. That fits comfortably; the activations are the real constraint — `(64, 1008)`
   float32 is ~258 KB per layer, so the length T is what will hurt, not the weights.
3. **Quantisation.** Do it post-training with a calibration set drawn from the WebTraffic training
   split, then re-measure accuracy AND AOPCR. Interpretability degrades faster than accuracy under
   quantisation, and AOPCR is the number this project is judged on.

---

## 5. Which models are worth deploying

The five in `configs/models/top/` — see `configs/models/top/README.md`. `top_bottleneck_topk` is the
main candidate: smallest model we have (41 K parameters, 11× fewer FLOPs than MILLET) and the only
one measured over 3 seeds.

```bash
python main.py webtraffic --model top/top_bottleneck_topk --smoke   # flow check
python main.py webtraffic --model top/top_bottleneck_topk           # the real run + the bundle
```
