# 01 - Environment setup

There are two places this project runs, and they are **not** the same environment. The *code* is
identical; the python version, the conda env name, the CUDA build and the way a job starts are
not. Never copy an install command from one to the other without checking.

| | local | Grid5000 |
|---|---|---|
| what for | writing code, smoke tests | all real GPU training |
| how you start a run | `python main.py ...` in a terminal | `oarsub` -> `scripts/run_all.sh` |
| which env file the pipeline uses | `configs/environments/local.yaml` | `configs/environments/grid5000.yaml` |

---

## 1. Local (Windows or Linux, VS Code)

```bash
# a fresh environment (conda is what the cluster uses too, so it keeps the two similar)
conda create -n seanet python=3.10 -y
conda activate seanet

# PyTorch. Pick the line that matches YOUR machine from https://pytorch.org - the CPU build is
# fine for smoke tests, which is all the laptop is for.
pip install torch --index-url https://download.pytorch.org/whl/cpu

# everything else
pip install -r requirements.txt
```

`requirements_versioned.txt` holds the exact versions a working run was made with. Use it when
something behaves strangely and you want to rule the environment out:

```bash
pip install -r requirements_versioned.txt
```

### Check it works

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python main.py models                       # lists 72 model configs - no data needed
python main.py summary Coffee               # needs the data (see guide 02)
```

The real end-to-end check is a smoke run - 3 epochs, nothing saved:

```bash
python main.py single Coffee --model seanet_bottleneck_topk --smoke
```

If that ends with a table of numbers and no traceback, the whole pipeline works.

---

## 2. Grid5000

`scripts/env.sh` handles the per-site differences for you (Lille needs `module load conda`,
Sophia does not) - see [12_grid5000.md](12_grid5000.md) for the full walk-through. In short:

```bash
ssh <site>                       # e.g. ssh lille
cd SEA_NET
source scripts/env.sh            # activates the env and prints the torch / CUDA check
export SEANET_ENV=grid5000       # so every command uses configs/environments/grid5000.yaml
bash scripts/test_run.sh         # the smoke test, on the cluster
```

**Do not assume the server matches your laptop.** Before installing anything there, print what is
actually on it and read the output:

```bash
python -V
conda env list
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

---

## 3. Optional extras

| package | needed for | without it |
|---|---|---|
| `mlflow>=3.1` | run tracking | training still works; a one-line warning is printed |
| `optuna` | `python main.py optuna` | every other command works |
| `scipy` | the critical-difference bars in `analyse` | that one figure is skipped, not faked |
| `fvcore` or `thop` | FLOPs in `scripts/profile_models.py` | the FLOPs columns stay empty |

Nothing in the core pipeline hard-depends on these - each one degrades to a printed note.

---

Next: [02 - Datasets](02_datasets.md)
