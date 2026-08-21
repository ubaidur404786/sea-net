# DEBUG_GUIDE.md — walk the whole flow with breakpoints

Purpose: stop the code at chosen lines and look at the real values (shapes,
tensors, config) so the pipeline can be explained end to end — data loading →
model building → training → scoring.

The debug run uses `--smoke` (3 epochs, result **not** saved), so it is safe to
stop, step and restart as often as you like.

---

## 1. The one rule that breaks everything

**Press `F5`. Not `Ctrl+F5`. Not the play button at the top-right of the editor.**

| you press | what VS Code does | breakpoints |
|---|---|---|
| `F5` — Start Debugging | runs **with** the debugger | stop |
| `Ctrl+F5` — Run Without Debugging | runs **without** the debugger | ignored |
| play button top-right of the editor | "Run Python File" = without debugger | ignored |

All three print the same `...debugpy\launcher...` line in the terminal, so a run
without debugging **looks** exactly like a real debug session. That is the trap.

**How to tell you are really debugging:** a floating toolbar appears at the top
with *pause, step over, step into, step out*. If you only see a stop button, the
debugger is off — you pressed the wrong key.

## 2. Setup (one time)

1. `Ctrl+Shift+P` → `Python: Select Interpreter` → choose **`millet`**
   (`C:\Users\HP\anaconda3\envs\millet\python.exe`).
2. `.vscode/launch.json` holds **one** config, `Debug main.py`. Nothing to do.

To debug a different command, edit its `args` line:

```json
"args": ["single", "Coffee", "--model", "sv2/seanet", "--smoke"],
```

That is the same as typing `python main.py single Coffee --model sv2/seanet --smoke`.

## 3. The keys

| what | how |
|---|---|
| breakpoint (red dot) | click the empty margin left of the line number |
| start | `F5` |
| continue to the next breakpoint | `F5` |
| `F10` | Step Over — run this line, stay in this file |
| `F11` | Step Into — go **inside** the function on this line |
| `Shift+F11` | Step Out — finish this function, return to the caller |
| `Shift+F5` | Stop |

While stopped, look at:
- **VARIABLES** panel (left) — every local variable and its value.
- **DEBUG CONSOLE** (bottom) — type any Python, e.g. `batch["bags"].shape`.
  Fastest way to show a shape to someone.
- **CALL STACK** panel — the chain of functions that led here. This *is* the flow.

## 4. The breakpoint list — the flow, in order

Put a red dot on **each** of these, press `F5`, then `F5` again at every stop.
Each stop is one stage of the pipeline.

**Line numbers move when you edit a file.** The code text in each row is the
part that matters — if a number is off, press `Ctrl+F` in that file and search
for the snippet shown instead.

### Stage 0 — the command is parsed

| file | line | what to show |
|---|---|---|
| `main.py` | **451** | `cfg, model_id, device, smoke = _run_context(args)` — `F10` once, then look at `cfg` (the YAML), `model_id`, `device`, `smoke=True` |

### Stage 1 — the data is loaded

| file | line | what to show |
|---|---|---|
| `seanet/data.py` | **299** | first line of `load_dataset()` — the single door every dataset goes through |
| `seanet/data.py` | **303** | `return UCRDataset(name, split)` — which of the 3 branches was taken |
| `seanet/train.py` | **389** | `train_full = D.load_dataset(name, "train")` — `F10`, then console: `len(train_full)` |
| `seanet/train.py` | **399** | `n_in = int(train_full.get_bag(0).shape[1])` — **the key shape line**. Console: `train_full.get_bag(0).shape` gives `(T, n_in)`, and `train_full.n_clz` |

**What to say here:** one *bag* = one time series. One *instance* = one timestep.
That is the MIL view the whole model is built on.

### Stage 2 — the model is built

| file | line | what to show |
|---|---|---|
| `seanet/train.py` | **407** | `batch_size = ...` — derived from dataset size, not fixed |
| `seanet/train.py` | **409** | `model = make_model(...)` — `F11` to go **into** the builder |
| `seanet/model.py` | **138** | inside `build_model_from_config()` — YAML strings become real modules. `F10` twice, then look at `encoder` and `pool` |
| `seanet/model.py` | **81** | `timestep_embeddings = self.feature_extractor(bags)` — encoder: `(B,C,T)` to `(B,d,T)` |
| `seanet/model.py` | **82** | `return self.pool(...)` — pooling head: `(B,d,T)` to the output dict |

At line 81/82, in the Debug Console:

```python
bags.shape                    # (B, C, T)  batch, channels, timesteps
timestep_embeddings.shape     # (B, d, T)  one embedding per timestep
```

That pair of shapes is the clearest way to explain the architecture.

### Stage 3 — training

| file | line | what to show |
|---|---|---|
| `seanet/train.py` | **417** | `model.fit(...)` — `F11` to enter the training loop |
| `seanet/train.py` | **254** | `for batch in loader:` — one pass over the data |
| `seanet/train.py` | **255** | console: `batch["bags"].shape`, `batch["targets"]` |
| `seanet/train.py` | **257** | `out = self(batch["bags"])` — `F10`, then `out.keys()`, `out["bag_logits"].shape` |
| `seanet/train.py` | **258** | `loss = criterion(...)` — `F10`, then `loss.item()` |

**Tip:** line 257 runs on every batch of every epoch. Right-click the red dot →
**Edit Breakpoint** → type `epoch == 2` to stop only once.

### Stage 4 — scoring

| file | line | what to show |
|---|---|---|
| `seanet/train.py` | **633** | `cls = safe_evaluate(model, test_ds)` — `F10` twice, then `cls`, `aopcr`, `ndcg` |
| `main.py` | **310** | `row["encoder"] = ...` — the finished results row. Console: `row` |

Because `--smoke` is on, the row is **not** written to `results/`. Say that out
loud during the demo — it is why the run is safe.

## 5. Shape cheat-sheet

For `Coffee`: 28 train series, 286 timesteps, 2 classes, 1 channel.

```
raw file            Coffee_TRAIN.tsv         28 rows x (1 label + 286 values)
one bag             (286, 1)                 (T, n_in)   T timesteps, 1 channel
one batch           bags  (B, 1, 286)        (B, C, T)   B = batch_size
after the encoder   (B, d, 286)              d = embedding size, T unchanged
after the pooling   bag_logits (B, 2)        one score per class
                    interpretation (B, 2, 286)  per-class, per-timestep
```

The important point: **T never shrinks inside the model.** That is what lets the
model say *which timestep* caused the prediction.

## 6. If it does not stop

Work down this list in order.

| symptom | cause | fix |
|---|---|---|
| ran straight through, no toolbar with step buttons | **you pressed `Ctrl+F5` or the play button** — the #1 cause | press `F5` from the Run and Debug panel |
| `breakpoint()` drops you to a `(Pdb)` prompt in the terminal | same thing — the debugger is off. With it on, `breakpoint()` stops in the VS Code UI, never in `(Pdb)` | press `F5` |
| BREAKPOINTS panel is empty | the margin clicks never registered | click the margin again, check the dot appears |
| breakpoints listed but all ignored | *Deactivate Breakpoints* toggle is on (a dot with a slash in the debug toolbar) | `Ctrl+Shift+P` then `Debug: Enable All Breakpoints` |
| dot is a **hollow** grey circle | not attached to a real line (blank line, comment, or a `def` line) | move it to the first real statement inside the function |
| it stops nowhere in one file only | that branch never runs (e.g. a WebTraffic line while running Coffee) | pick a line this command actually reaches |
| `ModuleNotFoundError: torch` | wrong interpreter | redo section 2 step 1 |

### Proving the debugger is on

`breakpoint()` is a plain Python built-in, and debugpy replaces it
(`sys.breakpointhook`) the moment it attaches. So it is a perfect test:

- stops in the **VS Code UI** (yellow line, VARIABLES filled) means the debugger is on
- drops to **`(Pdb)`** in the terminal means the debugger is off — Run, not Debug

If you add one temporarily, **delete it afterwards**. On Grid5000 a leftover
`breakpoint()` freezes a training job forever waiting for a debugger:

```powershell
git diff | Select-String "breakpoint\(\)"
```
