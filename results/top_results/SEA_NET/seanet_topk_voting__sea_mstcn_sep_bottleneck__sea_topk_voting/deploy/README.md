# deploy/ - complete, reloadable copies of this model

One bundle per (dataset, seed). `WebTraffic__seed0.*` all belong together.

| file | what it is |
|---|---|
| `<stem>.pt` | weights **and** the config that built them. This one file is enough. |
| `<stem>_config.yaml` | the same model config, readable. Copy it into `configs/models/` to retrain. |
| `<stem>_meta.json` | input shape, preprocessing, classes, metrics, versions, git commit. |
| `<stem>_traced.pt` | TorchScript. `torch.jit.load(...)` - no project code needed. |
| `<stem>.onnx` | ONNX (opset 17, dynamic batch, **fixed length**). The starting point for ESP32. |

## Get the exact model back

```python
from seanet.deployment import load_bundle
net, meta = load_bundle("WebTraffic__seed0.pt")
import torch
out = net(torch.zeros(1, 1, meta["series_length"]))
out["bag_logits"]       # (1, C) raw logits - NO softmax has been applied
out["interpretation"]   # (1, C, T) importance of every timestep
```

## Without any project code

```python
import torch
m = torch.jit.load("WebTraffic__seed0_traced.pt")
bag_logits, interpretation = m(torch.zeros(1, 1, 1008))
```

## Towards the ESP32

Start from the `.onnx`. The usual road is ONNX -> TensorFlow -> TFLite -> TFLite-Micro, or your
vendor's own converter (ESP-DL for Espressif). Three things to remember when you get there:

1. **The input must be z-normalised per series** (mean 0, std 1) exactly as in training. If the
   device feeds raw counts, the model sees a distribution it has never met.
2. **`bag_logits` are raw logits.** Take the arg-max for the class, or a softmax for probabilities.
3. **The exported length is fixed** at whatever T the model was trained on. The top-k head turns
   `k = ceil(top_frac * T)` into a constant while tracing, so feeding a different length to the
   `.onnx` or `.pt` TorchScript file would use the wrong k. Need another length? Re-export from the
   bundle: `load_bundle(...)` then `torch.onnx.export(...)` with the new example input.
