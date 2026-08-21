# 13 - Debugging: walk the whole flow with breakpoints

Purpose: stop the code at chosen places and look at the real values (shapes, tensors, config) so
the pipeline can be understood end to end - data loading -> preprocessing -> model building ->
training -> scoring.

Always debug with `--smoke`: 3 epochs, nothing saved, nothing logged to MLflow. So you can stop,
step and restart as often as you like without polluting the results.

---

## 1. The one rule that breaks everything

**Press `F5` (Start Debugging). Not `Ctrl+F5`. Not the play button at the top right of the editor.**

`Ctrl+F5` and that button mean "Run **without** debugging" - the same command runs, but every
breakpoint is skipped, and it looks like your breakpoints "do not work".

## 2. Set the command

`.vscode/launch.json` holds one configuration. Its `"args"` line is just the command line split
into pieces:

```jsonc
"args": ["single", "Coffee", "--model", "seanet_bottleneck_topk", "--smoke"],
```

is exactly `python main.py single Coffee --model seanet_bottleneck_topk --smoke`.

Edit that line to debug something else:

```jsonc
"args": ["webtraffic", "--model", "seanet_bottleneck_topk", "--smoke"],   // the NDCG path
"args": ["run", "--model", "baselines/millet", "--smoke"],                // prints the full config
"args": ["analyse"],                                                       // the figure pipeline
```

Keep `--smoke` on anything that trains.

## 3. The keys

| key | what it does |
|---|---|
| `F5` | start / continue to the next breakpoint |
| `F10` | step **over** the current line (run it, stay here) |
| `F11` | step **into** the function on this line |
| `Shift+F11` | step out, back to the caller |
| `Shift+F5` | stop |

The **Debug Console** (bottom panel) is the useful part: while stopped, type any Python
expression - `train_full.n_clz`, `out["bag_logits"].shape`, `cfg.model_config.encoder.type` - and
it evaluates it in that frame.

---

## 4. Where to put the breakpoints

Line numbers move every time the code is edited, so these are given by **file and function**. Open
the file, find the function, click in the left gutter to place a red dot.

### The command arrives

| file | function | what you see |
|---|---|---|
| `main.py` | `_resolve_model` | the parsed flags becoming a config. Console: `cfg.model`, `cfg.seed`, `cfg.env` |
| `main.py` | `_run_context` | the device and smoke decision |
| `main.py` | `_train_and_save` | the single place a results row is produced and written |

### Configuration

| file | function | what you see |
|---|---|---|
| `seanet/config.py` | `load_config` | the four layers merging. Console: `main` after each `_merge` |
| `seanet/config.py` | `find_model_file` | which YAML a `--model` value resolved to |
| `seanet/config.py` | `model_folder_name` | how `<config>__<encoder>__<pooling>` is built |

### Data and preprocessing

| file | function | what you see |
|---|---|---|
| `seanet/data.py` | `load_dataset` | which dataset class was chosen. Console: `len(ds)`, `ds.n_clz`, `ds.get_bag(0).shape` |
| `seanet/preprocessing.py` | `prepare_splits` | whether a validation split was held out. Console: `len(train_full)`, `min_train_for_val` |
| `seanet/preprocessing.py` | `split_train_val` | the stratified split. Console: `torch.unique(targets, return_counts=True)` |

### Model building - **this is the shape line that matters**

| file | function | what you see |
|---|---|---|
| `seanet/training.py` | `fit_model`, at `n_in = int(train_full.get_bag(0).shape[1])` | the bag is `(T, n_in)`, so `n_in` is the channel count |
| `seanet/models/build.py` | `build_model_from_config` | YAML strings becoming real modules. `F10` twice, then look at `encoder` and `pool` |
| `seanet/models/encoders.py` | `build_encoder` | the registry lookup. Console: `sorted(ENCODER_REGISTRY)` |
| `seanet/models/pooling.py` | `build_pooling` | same for the head, plus which optional knobs it accepted |
| `seanet/models/build.py` | `EncoderPoolNet.forward` | the two halves. `F10` once: `timestep_embeddings.shape` is `(B, d, T)` |

### Training

| file | function | what you see |
|---|---|---|
| `seanet/training.py` | `SeaNetModel.fit`, inside the batch loop | Console: `batch["bags"].shape`, `batch["targets"]`, then after `out = self(...)`: `out.keys()`, `out["bag_logits"].shape` |
| `seanet/training.py` | `SeaNetModel.fit`, at `total = loss + ...` | the entropy focus penalty being added (only when the head returns `attn`) |
| `seanet/training.py` | `SeaNetModel.fit`, at `self.history.append(...)` | the per-epoch record: train loss/acc and val loss/acc |
| `seanet/training.py` | `SeaNetModel.fit`, at `if monitor_loss < best_loss` | early stopping deciding whether to keep these weights |

### Evaluation

| file | function | what you see |
|---|---|---|
| `seanet/evaluation.py` | `score_model` | `F10` past the two metric calls, then `cls`, `aopcr`, `ndcg` |
| `seanet/metrics.py` | `classification_metrics` | the `try` on MILLET's `evaluate`, and the AUROC fallback |
| `seanet/metrics.py` | `interpretability_metrics` | AOPCR and NDCG. This is the slow part - AOPCR re-runs the model many times |
| `seanet/evaluation.py` | `score_model`, at `row = {...}` | the finished results row, before it is saved |
| `seanet/training.py` | `save_history` | the history CSV and the two curve PNGs being written |
| `seanet/results.py` | `save_result_row` | the row going into `results.csv` (and the "keep the better result" rule) |

---

## 5. Reading a traceback

Because each component has one job, the last few frames tell you where to look:

| the traceback ends in | the problem is |
|---|---|
| `seanet/config.py` | a config typo, a missing key, an unknown `--model` |
| `seanet/data.py` | a missing or malformed data file |
| `seanet/preprocessing.py` | the split (usually a class with a single example) |
| `seanet/models/encoders.py` / `pooling.py` | an unknown `type:`, or a shape mismatch inside your new module |
| `seanet/training.py` | the training loop itself |
| `seanet/evaluation.py` / `metrics.py` | scoring - the model trained fine |
| `seanet/results.py` | writing the results file |
| `seanet/analysis/` | a figure; nothing to do with training |

Every run's full console output is also saved to
`results/SEA_NET/<model_id>/logs/<command>_<date-time>.log` (smoke runs go to `logs/smoke/`), and
`main.py` writes the traceback into that log too - so a crash on the cluster leaves a readable
record.

---

## 6. Quick checks without the debugger

Shapes, without training anything:

```python
import torch
from seanet.config import load_config
from seanet.models import build_model_from_config, num_params

cfg = load_config(overrides={"model": "seanet_bottleneck_topk"}).model_config
net = build_model_from_config(cfg, n_clz=3, n_in=1)
out = net(torch.randn(2, 1, 128))
print({k: tuple(v.shape) for k, v in out.items()}, num_params(net))
```

What a config actually resolved to:

```bash
python main.py run --model seanet_bottleneck_topk --smoke     # prints every resolved value first
```

Multi-scale alignment (a shift does not crash, it just destroys NDCG):

```bash
python scripts/check_multiscale.py
```

---

Next: [14 - Git](14_git.md)
