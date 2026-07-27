"""
scripts/profile_models.py - measure the COST of every model: speed, FLOPs, memory, size.

Why this exists
---------------
results.csv records how ACCURATE each model is, but never how expensive it is to run. A paper that
claims "our model is smaller and just as good" has to show the cost side too, so this script
measures it. Nothing here trains anything - each model is built once, fed a fixed dummy batch, and
timed. It takes seconds per model.

What it measures, per model
---------------------------
    params        : number of weights (same number results.csv already stores - a cross-check)
    size_mb       : how big the saved weights are on disk
    flops_m       : millions of FLOPs for ONE series, counted analytically (see count_flops below)
    infer_ms      : milliseconds to classify ONE series (the average over many runs)
    throughput    : series per second
    peak_mem_mb   : peak GPU memory during a forward pass (NaN on CPU - see the note below)

The fair-comparison trick
-------------------------
Every model is measured on the SAME dummy input (same batch size, same series length, same number of
classes). That is deliberate: we are comparing ARCHITECTURES, so the input must be identical or the
numbers mean nothing. The shape is printed in the output and stored in the csv header comment, so
the paper can state it exactly.

How to run
----------
    python scripts/profile_models.py                     # every config, on the best device available
    python scripts/profile_models.py --device cpu        # force CPU (inference speed on CPU)
    python scripts/profile_models.py --length 500 --batch 32 --classes 5     # change the dummy input
    python scripts/profile_models.py --models sv2/seanet sv4/seanet_slim     # only these

Writes results/SEA_NET/profile.csv, which the paper figures read for the efficiency panels.
"""
import argparse
import glob
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seanet.config import load_config, model_folder_name            # noqa: E402
from seanet.data import chdir_to_repo_root                          # noqa: E402
from seanet.model import build_model_from_config, num_params, state_dict_size_mb   # noqa: E402
from seanet.results import RESULTS_ROOT                             # noqa: E402

PROFILE_CSV = os.path.join(RESULTS_ROOT, "profile.csv")

# the fixed dummy input every model is measured on (change with the command-line flags)
DEFAULT_LENGTH = 500          # series length T - around the middle of the UCR archive
DEFAULT_BATCH = 32            # batch size used for the timing
DEFAULT_CLASSES = 5           # number of classes (only changes the tiny classifier head)
WARMUP_RUNS = 5               # throw-away passes (the first ones are always slow - see below)
TIMED_RUNS = 30               # how many passes we actually average over


def count_flops(model, example: torch.Tensor) -> float:
    """
    Count the FLOPs of one forward pass, by hand, with no extra library.

    What a FLOP is here: one multiply-accumulate counted as 2 operations (one multiply + one add).
    This is the usual convention in papers.

    How we count without a library: PyTorch lets you attach a "forward hook" to a layer - a small
    function that runs every time that layer produces an output. We attach one to every Conv1d and
    Linear layer; the hook looks at the output shape and the layer's own settings and works out how
    much arithmetic that layer just did. Conv and Linear are where essentially all the arithmetic in
    these models happens (BatchNorm / ReLU / dropout are negligible), so this is an honest estimate
    and it needs no external package - which matters because the cluster env is fixed.

    Formula for a Conv1d: every output position needs (in_channels/groups * kernel_size) multiply-adds
    for each of the out_channels. So FLOPs = 2 * out_channels * out_length * (in_ch/groups * kernel).
    For a Linear: FLOPs = 2 * in_features * out_features per item.

    model : the network.
    example : one input batch, shape (B, C, T).
    returns : FLOPs for a SINGLE series (the batch total divided by the batch size), in millions.
    """
    total = [0.0]                                     # a list so the inner function can add to it

    def conv_hook(module, _inputs, output):
        out_len = output.shape[-1]
        kernel = module.kernel_size[0]
        per_position = (module.in_channels / module.groups) * kernel
        total[0] += 2.0 * module.out_channels * out_len * per_position * output.shape[0]

    def linear_hook(module, _inputs, output):
        n_items = output.numel() / module.out_features        # how many vectors went through
        total[0] += 2.0 * module.in_features * module.out_features * n_items

    handles = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv1d):
            handles.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, torch.nn.Linear):
            handles.append(layer.register_forward_hook(linear_hook))

    with torch.no_grad():
        model(example)
    for h in handles:                                  # always remove hooks, or they pile up
        h.remove()

    return total[0] / example.shape[0] / 1e6           # per single series, in millions


def time_inference(model, example: torch.Tensor, device: str) -> tuple:
    """
    Time one forward pass, carefully.

    Two details that make timing honest:

    1. WARM-UP. The first few passes are always slower than the rest - PyTorch is still allocating
       memory, picking convolution algorithms, and (on GPU) loading kernels. If you time those you
       measure the setup, not the model. So we run a few passes and throw them away first.

    2. torch.cuda.synchronize(). GPU work is ASYNCHRONOUS: the Python line returns immediately while
       the GPU is still computing. Without a synchronise you would time how fast Python queued the
       work, which is nonsense. synchronize() waits for the GPU to actually finish.

    model : the network (already on the device, in eval mode).
    example : the dummy batch.
    device : "cuda" or "cpu".
    returns : (milliseconds per single series, series per second).
    """
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):                   # 1. warm-up, results thrown away
            model(example)
        if device == "cuda":
            torch.cuda.synchronize()                   # 2. wait for the GPU before starting the clock

        start = time.perf_counter()
        for _ in range(TIMED_RUNS):
            model(example)
        if device == "cuda":
            torch.cuda.synchronize()                   # and wait again before stopping it
        elapsed = time.perf_counter() - start

    batch = example.shape[0]
    per_series_s = elapsed / (TIMED_RUNS * batch)
    return per_series_s * 1000.0, 1.0 / per_series_s


def peak_memory_mb(model, example: torch.Tensor, device: str) -> float:
    """
    Peak GPU memory used by one forward pass, in MB.

    Only meaningful on CUDA: PyTorch tracks its own GPU allocations exactly, so we can reset the
    counter, run one pass, and read the maximum. On CPU there is no equivalent honest number
    (Python's allocator does not tell us what the model alone used), so we return NaN rather than
    print a number we cannot stand behind.
    """
    if device != "cuda":
        return float("nan")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(example)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def list_config_names() -> list:
    """Every model config under configs/models/, as "sv4/seanet_slim"-style names."""
    names = []
    for path in sorted(glob.glob(os.path.join("configs", "models", "**", "*.yaml"), recursive=True)):
        rel = os.path.relpath(path, os.path.join("configs", "models"))
        names.append(rel.replace(os.sep, "/")[:-5])
    return names


def profile_one(config_name: str, device: str, length: int, batch: int, classes: int) -> dict:
    """
    Build ONE model from its config and measure everything about it.

    config_name : e.g. "sv4/seanet_slim".
    returns : a dict of measurements (or None if the config cannot be built, e.g. the placeholder).
    """
    cfg = load_config(overrides={"model": config_name})
    model_id = model_folder_name(cfg)
    model = build_model_from_config(cfg.model_config, n_clz=classes, n_in=1)
    model = model.to(device).eval()

    example = torch.randn(batch, 1, length, device=device)

    flops = count_flops(model, example)
    infer_ms, throughput = time_inference(model, example, device)
    mem = peak_memory_mb(model, example, device)

    return {
        "model": model_id,
        "config": config_name,
        "params": num_params(model),
        "size_mb": round(state_dict_size_mb(model), 4),
        "flops_m": round(flops, 3),
        "infer_ms": round(infer_ms, 4),
        "throughput": round(throughput, 1),
        "peak_mem_mb": round(mem, 3) if mem == mem else "",     # "" instead of nan for a clean csv
        "device": device,
        "batch": batch,
        "length": length,
        "n_classes": classes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH, help="dummy series length T")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="batch size for the timing")
    parser.add_argument("--classes", type=int, default=DEFAULT_CLASSES, help="number of classes")
    parser.add_argument("--models", nargs="*", help="only these configs (default: all of them)")
    parser.add_argument("--out", default=PROFILE_CSV, help="where to write the csv")
    args = parser.parse_args()
    chdir_to_repo_root()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    names = args.models or list_config_names()
    print(f"Profiling {len(names)} model(s) on {device}")
    print(f"  dummy input: batch={args.batch}, channels=1, length={args.length}, classes={args.classes}")
    print(f"  timing: {WARMUP_RUNS} warm-up passes (discarded) + {TIMED_RUNS} timed passes\n")
    if device == "cpu":
        print("  note: peak memory is only measurable on CUDA, so it will be left empty here.\n")

    rows, failed = [], []
    for name in names:
        try:
            row = profile_one(name, device, args.length, args.batch, args.classes)
        except NotImplementedError:
            failed.append((name, "placeholder config (not implemented)"))
            continue
        except Exception as exc:
            failed.append((name, repr(exc)))
            continue
        rows.append(row)
        print(f"  {row['model'][:62]:62s} {row['params']:>8,}p "
              f"{row['flops_m']:>9.2f}MF {row['infer_ms']:>8.3f}ms")

    if not rows:
        print("\nNothing profiled.")
        return

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("infer_ms").reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} model(s))")

    if failed:
        print(f"\n{len(failed)} config(s) skipped:")
        for name, why in failed:
            print(f"  {name}: {why}")

    print("\nNow draw the figures that use these numbers:")
    print("  python main.py paper")


if __name__ == "__main__":
    main()
