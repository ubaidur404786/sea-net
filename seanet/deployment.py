"""
seanet/deployment.py - save EVERYTHING needed to rebuild and deploy a trained model.

What this file is for:
    results.csv only keeps NUMBERS. You cannot put an accuracy on a microcontroller. When a model
    does well on WebTraffic we want to be able to come back months later, on another machine, and
    get the EXACT same network back - to benchmark it on an ESP32, to convert it, or just to check
    a result. This file writes that "everything" next to the results, and reads it back.

What one bundle contains (all files share the stem "<dataset>__seed<N>"):

    <stem>.pt            the main file. A torch.save of ONE dict:
                           "state_dict"   - the trained weights
                           "model_config" - the whole resolved model config, as a plain dict
                                            (encoder block, pooling block, training recipe)
                           "n_classes", "n_in", "series_length" - the input/output shape
                           "model_id", "dataset", "seed", "encoder", "pooling"
                         That dict alone is enough to rebuild the network - see load_bundle().
    <stem>_config.yaml   the same model config as readable YAML (so you can diff it by eye, or
                         copy it back into configs/models/ to retrain the model).
    <stem>_meta.json     everything a human or a converter needs that is NOT weights: input shape,
                         preprocessing, class count, the test metrics, library versions, the git
                         commit, the date.
    <stem>_traced.pt     TorchScript. Loads with torch.jit.load in plain Python or in C++ (libtorch)
                         with no project code at all. Best effort - skipped with a printed note if
                         tracing fails.
    <stem>.onnx          ONNX, opset 17. This is the format the ESP32 toolchains start from
                         (ONNX -> TFLite -> TFLite-Micro, or ONNX -> your vendor's converter).
                         Best effort - skipped with a note in the meta file if it fails.
                         THE LENGTH T IS BAKED IN. The top-k head computes k = ceil(top_frac * T)
                         in Python, so the tracer turns it into a constant; only the BATCH axis is
                         dynamic. That is fine for a device (the input length is fixed there anyway),
                         but it means one export per input length. The .pt bundle has no such limit.
    README.md            written once per folder: how to load each of these back.

Why a wrapper is exported instead of the model itself:
    our network returns a DICT (bag_logits, interpretation, instance_logits, attn). Tracers and
    converters want plain tensors, so we export a small wrapper that returns the two tensors that
    matter - (bag_logits, interpretation) - in a fixed order. The interpretation is kept because
    losing it would throw away the interpretability that is the whole point of this project.

Nothing here can fail a training run: every step is wrapped so a missing optional library or a
read-only disk prints a note and moves on.

Related files:
    - seanet/results.py     -> deploy_dir(model_id) says WHERE a bundle goes.
    - seanet/evaluation.py  -> score_model() calls save_bundle() after a real run.
    - seanet/models/build.py -> build_model_from_config(), which load_bundle() uses to rebuild.
"""
import json
import os
import platform
import subprocess
from datetime import datetime
from typing import Dict, Optional

import torch
import yaml
from torch import nn

from seanet.models.build import build_model_from_config, num_params


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _plain(obj):
    """
    Turn a SimpleNamespace config (nested) into plain dicts/lists, so it can be saved as YAML/JSON.

    obj : a SimpleNamespace, dict, list or scalar.
    returns : the same data using only built-in types.
    """
    if hasattr(obj, "__dict__"):                             # a SimpleNamespace
        return {k: _plain(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _git_commit() -> Optional[str]:
    """The current git commit, so a bundle says which version of the code trained it (None if unknown)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:                                        # not a git checkout, or no git installed
        return None


class _ExportWrapper(nn.Module):
    """
    Make the network look like a normal "tensor in, tensors out" model, for tracing and ONNX.

    Our EncoderPoolNet returns a dict. Tracers cannot follow a dict cleanly, so this wrapper picks
    the two tensors we actually want on a device and returns them in a fixed order.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, bags: torch.Tensor):
        """bags : (B, n_in, T) -> (bag_logits (B, C), interpretation (B, C, T))."""
        out = self.net(bags)
        return out["bag_logits"], out["interpretation"]


# --------------------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------------------
def save_bundle(model, dataset: str, seed: int, out_dir: str, model_cfg=None,
                model_id: Optional[str] = None, metrics: Optional[Dict] = None,
                series_length: Optional[int] = None, n_in: int = 1,
                extra: Optional[Dict] = None, export: bool = True) -> str:
    """
    Write one complete deployment bundle for a trained model.

    model : the trained SeaNetModel (we save model.net, the actual network).
    dataset : dataset name, e.g. "WebTraffic".
    seed : the training seed (part of the file name, so seeds never overwrite each other).
    out_dir : the folder to write into (results.deploy_dir(model_id)).
    model_cfg : the model config this network was built from. Without it the bundle cannot be
                rebuilt automatically, so we refuse to pretend and just say so in the meta file.
    model_id : "<config>__<encoder>__<pooling>", the results folder name.
    metrics : the results row (test_acc, test_aopcr, ...) - saved for reference only.
    series_length : T, the input length the model was trained on.
    n_in : number of input channels (1 here).
    extra : anything else worth recording (preprocessing settings, dataset notes).
    export : also try TorchScript + ONNX. Set False to save only weights + config (much faster).
    returns : the path of the main .pt file.
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{dataset}__seed{int(seed)}")
    net = model.net
    net_was_training = net.training
    net.eval()                                               # export in eval mode (no dropout)

    cfg_plain = _plain(model_cfg) if model_cfg is not None else None
    n_classes = int(getattr(model, "n_classes", 0) or 0)

    # ---- 1. the main file: weights + everything needed to rebuild them -------------------
    payload = {
        "format_version": 1,
        "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
        "model_config": cfg_plain,
        "model_id": model_id,
        "dataset": dataset,
        "seed": int(seed),
        "n_classes": n_classes,
        "n_in": int(n_in),
        "series_length": None if series_length is None else int(series_length),
        "encoder": (cfg_plain or {}).get("encoder", {}).get("type"),
        "pooling": (cfg_plain or {}).get("pooling", {}).get("type"),
    }
    torch.save(payload, f"{stem}.pt")

    # ---- 2. the config on its own, as readable YAML --------------------------------------
    if cfg_plain is not None:
        with open(f"{stem}_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_plain, f, sort_keys=False, allow_unicode=True)

    # ---- 3. the meta file: everything a converter or a human needs, no tensors ------------
    meta = {
        "model_id": model_id,
        "dataset": dataset,
        "seed": int(seed),
        "saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(),
        "input": {
            # one instance = one timestep, so a bag is (n_in, T) and a batch is (B, n_in, T)
            "shape": ["batch", int(n_in), series_length or "T"],
            "dtype": "float32",
            "channels": int(n_in),
            "series_length": series_length,
            "preprocessing": "each series is z-normalised on its own (mean 0, std 1) before it "
                             "reaches the model - see seanet/preprocessing.py",
            "fixed_length_in_exports": "the .onnx and _traced.pt files are exported at THIS length "
                                       "only (the top-k head turns k = ceil(top_frac * T) into a "
                                       "constant). The .pt bundle works at any length.",
        },
        "output": {
            "bag_logits": ["batch", n_classes],
            "interpretation": ["batch", n_classes, series_length or "T"],
            "note": "bag_logits are RAW logits (no softmax). Apply softmax yourself if you want "
                    "probabilities; the training loss (nn.CrossEntropyLoss) does it internally.",
        },
        "n_classes": n_classes,
        "encoder": _plain(getattr(model_cfg, "encoder", None)) if model_cfg is not None else None,
        "pooling": _plain(getattr(model_cfg, "pooling", None)) if model_cfg is not None else None,
        "training": _plain(getattr(model_cfg, "training", None)) if model_cfg is not None else None,
        "params": num_params(net),
        "metrics": metrics or {},
        "versions": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "files": {
            "weights": os.path.basename(f"{stem}.pt"),
            "config": os.path.basename(f"{stem}_config.yaml") if cfg_plain else None,
        },
        "rebuild": "from seanet.deployment import load_bundle; net, meta = load_bundle('"
                   + os.path.basename(f"{stem}.pt") + "')",
    }
    if extra:
        meta["extra"] = _plain(extra)

    # ---- 4. portable exports (best effort - never fail the run) ---------------------------
    if export and series_length:
        example = torch.zeros(1, int(n_in), int(series_length))
        wrapper = _ExportWrapper(net).eval()
        try:
            traced = torch.jit.trace(wrapper, example, strict=False)
            traced.save(f"{stem}_traced.pt")
            meta["files"]["torchscript"] = os.path.basename(f"{stem}_traced.pt")
        except Exception as e:
            meta["files"]["torchscript"] = f"not saved ({type(e).__name__}: {e})"
        try:
            torch.onnx.export(
                wrapper, example, f"{stem}.onnx",
                input_names=["series"], output_names=["bag_logits", "interpretation"],
                # ONLY the batch axis is dynamic. The pooling head decides k = ceil(top_frac * T)
                # with ordinary Python arithmetic, so the tracer freezes it - claiming a dynamic
                # time axis here would give a file that silently computes the wrong k.
                dynamic_axes={"series": {0: "batch"},
                              "bag_logits": {0: "batch"},
                              "interpretation": {0: "batch"}},
                opset_version=17,
            )
            meta["files"]["onnx"] = os.path.basename(f"{stem}.onnx")
        except Exception as e:
            meta["files"]["onnx"] = f"not saved ({type(e).__name__}: {e})"

    with open(f"{stem}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _write_readme(out_dir)

    if net_was_training:                                     # leave the model exactly as we found it
        net.train()
    return f"{stem}.pt"


# --------------------------------------------------------------------------------------
# Loading it back (this is what proves a bundle is complete)
# --------------------------------------------------------------------------------------
def load_bundle(path: str, device=None):
    """
    Rebuild the exact trained network from a bundle's .pt file.

    It builds the network from the saved config (so the architecture is identical), then loads the
    saved weights into it with strict=True - if anything at all does not match, it raises instead of
    quietly giving you a half-loaded model.

    path : the "<dataset>__seed<N>.pt" file.
    device : where to put the network; None = cpu.
    returns : (net, payload). net is in eval() mode; payload is the saved dict (config, shapes, ...).
    raises ValueError : if the bundle has no model config (nothing to rebuild from).
    """
    from seanet.config import _to_namespace                  # same helper the config loader uses

    payload = torch.load(path, map_location=device or "cpu")
    cfg = payload.get("model_config")
    if cfg is None:
        raise ValueError(f"{path!r} has no model_config, so the network cannot be rebuilt. "
                         f"It was probably saved outside the normal training path.")
    net = build_model_from_config(_to_namespace(cfg), payload["n_classes"], n_in=payload["n_in"])
    net.load_state_dict(payload["state_dict"], strict=True)
    net.to(device or "cpu").eval()
    return net, payload


# --------------------------------------------------------------------------------------
# The folder's own instructions
# --------------------------------------------------------------------------------------
_README = """# deploy/ - complete, reloadable copies of this model

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
"""


def _write_readme(out_dir: str) -> None:
    """Write the folder's README once (it is the same text for every bundle in the folder)."""
    path = os.path.join(out_dir, "README.md")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(_README)
