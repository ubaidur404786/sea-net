"""
seanet/config.py - reading the YAML config files.

What this file is for:
    The whole project is meant to be driven by configuration, not by numbers hardcoded in the
    Python code. This file is the one place that reads the config. You give it the path to
    configs/main.yaml; it also loads the matching model file (configs/models/<model>.yaml) and
    merges the two into a single config object you can read with dots, e.g.:

        cfg = load_config("configs/main.yaml")
        cfg.model                        -> "seanet"
        cfg.seed                         -> 0
        cfg.model_config.training.learning_rate   -> 0.00125

Input:
    The path to main.yaml (and, indirectly, the model file it points at).
Output:
    A config object (a nested types.SimpleNamespace) that the rest of the code reads.

Related files:
    - configs/main.yaml            -> the top-level settings (which model, which dataset, seed...).
    - configs/models/<model>.yaml  -> that model's encoder / pooling / training settings.
    - seanet/model.py              -> build_model_from_config() turns cfg.model_config into a network.
    - seanet/train.py              -> train_one_from_config() reads cfg.model_config.training.
    - main.py ("run" command)      -> calls load_config() and hands the config to the trainer.

Why a SimpleNamespace and not a plain dict:
    A dict makes you write cfg["model_config"]["training"]["learning_rate"], which is noisy.
    types.SimpleNamespace lets you write cfg.model_config.training.learning_rate, which reads like
    normal Python. It is part of the standard library, so there is nothing new to learn.
"""
import os
from types import SimpleNamespace
from typing import Dict, Optional

import yaml

# Where the config files live. Kept here so there is one place to change if the folder moves.
CONFIGS_DIR = "configs"
MODELS_DIR = os.path.join(CONFIGS_DIR, "models")


def _read_yaml(path: str) -> Dict:
    """
    Read one YAML file into a plain dict.

    path : the .yaml file to read.
    returns : a dict (an empty dict if the file is empty).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path!r}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _merge(base: Dict, extra: Dict) -> Dict:
    """
    Deep-merge two dicts. Values in `extra` win; nested dicts are merged key by key.

    This is used to lay command-line overrides on top of the file, and to attach the model file
    under the main config. It does not change the inputs; it returns a new dict.

    base : the starting dict.
    extra : the dict whose values take priority.
    returns : the merged dict.
    """
    out = dict(base)
    for key, value in extra.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _merge(out[key], value)   # both are dicts -> merge them
        else:
            out[key] = value                     # otherwise extra just replaces base
    return out


def _to_namespace(obj):
    """
    Recursively turn dicts into SimpleNamespaces, so config can be read with dots.

    Lists are walked too (in case a list holds dicts). Anything else (int, float, str, bool, a
    list of numbers like the kernel sizes) is returned unchanged.

    obj : a dict / list / scalar loaded from YAML.
    returns : the same data, but with every dict replaced by a SimpleNamespace.
    """
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(main_path: str = os.path.join(CONFIGS_DIR, "main.yaml"),
                overrides: Optional[Dict] = None,
                use_params: Optional[str] = None) -> SimpleNamespace:
    """
    Load main.yaml + the model file it points at, and return one merged config object.

    Steps:
      1. read main.yaml,
      2. work out the model name (an override wins over the file),
      3. read configs/models/<model>.yaml and attach it under the key "model_config",
      4. choose the DEFAULT recipe or Optuna's BEST recipe (both live in the SAME model file, under
         the "records" block - see record_metrics below), and merge the best params in if chosen,
      5. apply any leftover overrides,
      6. convert the whole thing to a SimpleNamespace so it can be read with dots.

    main_path : path to main.yaml.
    overrides : optional dict of values to override (e.g. {"model": "millet"} from the command
                line). Only "model" changes which model file is read; other keys are merged in
                after the model file is attached.
    use_params : force which recipe to use ("default" | "optuna_best" | "auto"); None reads the
                 model file's own `use_params` setting. The chosen recipe (and the default-vs-best
                 comparison) is stashed under the private key "_param_choice" for param_choice_message.
    returns : the merged config as a nested SimpleNamespace.
    """
    overrides = overrides or {}
    main = _read_yaml(main_path)

    # which model file to load (an override beats the value in main.yaml)
    model_name = overrides.get("model", main.get("model"))
    if not model_name:
        raise ValueError(f"No model specified in {main_path!r} and none given as an override.")
    model_path = os.path.join(MODELS_DIR, f"{model_name}.yaml")
    model_cfg = _read_yaml(model_path)

    # decide default vs Optuna-best recipe from the recorded results in the SAME file (no .best.yaml)
    model_cfg, choice = _select_params(model_cfg, use_params_override=use_params)

    # attach the model file under "model_config" and record the chosen model name at the top level
    merged = _merge(main, {"model": model_name, "model_config": model_cfg})
    # apply the remaining overrides (model was already handled, but merging it again is harmless)
    merged = _merge(merged, overrides)
    merged["_param_choice"] = choice                     # private: read by param_choice_message()

    return _to_namespace(merged)


# --------------------------------------------------------------------------------------
# Recorded results: default vs Optuna-best, stored in the SAME model yaml (no separate .best.yaml)
#
# We keep a small "records" block at the BOTTOM of each model file, below a marker line. Everything
# above the marker is ours to edit (encoder / pooling / training / optuna); everything below is
# managed by the code (rewritten in place - our comments above are never touched). It holds two
# entries, each with the hyperparameters used and the metrics they achieved:
#   records.default     -> metrics of ours last real `python main.py run` with the default recipe.
#   records.optuna_best -> the params Optuna found + their metrics (from `python main.py optuna`).
# The top-level `use_params` setting then picks which recipe a run trains with (default/optuna_best/
# auto). This lets you run normal first, then Optuna, then compare and switch - all in one file.
# --------------------------------------------------------------------------------------
RECORDS_MARKER = "# ===== AUTO-FILLED RESULTS (managed by main.py - do not edit below this line) ====="

_RECORDS_HELP = (
    "# These records let you compare the DEFAULT recipe against Optuna's BEST without opening MLflow,\n"
    "# and (with `use_params: auto`) train with whichever scored higher. They are filled in for you:\n"
    "#   records.default     : metrics from our last real `python main.py run` (default recipe).\n"
    "#   records.optuna_best : the params Optuna found + their metrics (`python main.py optuna`).\n"
)


def _select_params(model_cfg: Dict, use_params_override: Optional[str] = None):
    """
    Pick which recipe (default or Optuna-best) to use, and merge the best params in if chosen.

    model_cfg : the model config dict (may contain "records" and "use_params").
    use_params_override : force the mode; None reads model_cfg["use_params"] (default "default").
    returns : (possibly-updated model_cfg, choice-info dict).
    """
    records = model_cfg.get("records") or {}
    mode = use_params_override or model_cfg.get("use_params", "default")

    best_entry = records.get("optuna_best") or {}
    best_params = best_entry.get("params") or {}                 # nested: {encoder:{...}, training:{...}}
    default_acc = ((records.get("default") or {}).get("metrics") or {}).get("test_acc")
    best_acc = (best_entry.get("metrics") or {}).get("test_acc")
    has_best = bool(best_params)

    if mode == "optuna_best" and has_best:
        used = "optuna_best"
    elif (mode == "auto" and has_best and best_acc is not None
          and (default_acc is None or best_acc >= default_acc)):
        used = "optuna_best"                                     # auto: use best only if it scored higher
    else:
        used = "default"

    if used == "optuna_best":
        model_cfg = _merge(model_cfg, best_params)               # best params win over the base blocks

    choice = {"mode": mode, "used": used, "default_acc": default_acc,
              "optuna_acc": best_acc, "has_best": has_best}
    return model_cfg, choice


def param_choice_message(cfg) -> str:
    """
    A one-line, human-readable note about which recipe a run is using and how default vs best compare.

    cfg : a loaded config (must carry the private _param_choice stashed by load_config).
    returns : the message string (empty string if there is nothing useful to say).
    """
    pc = getattr(cfg, "_param_choice", None)
    if pc is None:
        return ""
    dacc, oacc, used = pc.default_acc, pc.optuna_acc, pc.used
    if not pc.has_best:                                          # no Optuna record yet
        return f"[params] using DEFAULT recipe (no Optuna record yet - run 'python main.py optuna --model {cfg.model}')."
    cmp = ""
    if dacc is not None and oacc is not None:
        winner = "optuna-best" if oacc >= dacc else "default"
        cmp = f" (test_acc: default {dacc:.4f} vs optuna-best {oacc:.4f}; better = {winner})"
    elif oacc is not None:
        cmp = f" (optuna-best test_acc {oacc:.4f}; default not recorded yet)"
    if used == "optuna_best":
        return f"[params] using OPTUNA-BEST recipe{cmp}."
    note = ""
    if dacc is not None and oacc is not None and oacc > dacc:
        note = (f"  NOTE: optuna-best scored higher - set `use_params: auto` (or optuna_best) in "
                f"configs/models/{cfg.model}.yaml to train with it.")
    return f"[params] using DEFAULT recipe{cmp}.{note}"


def read_records(model_name: str) -> Dict:
    """Read the "records" block from a model yaml (or an empty dict if the file/block is missing)."""
    path = os.path.join(MODELS_DIR, f"{model_name}.yaml")
    if not os.path.exists(path):
        return {}
    return _read_yaml(path).get("records") or {}


def write_records(model_name: str, records: Dict) -> str:
    """
    Rewrite the auto-managed "records" block at the bottom of a model yaml, keeping the rest verbatim.

    We split the file on RECORDS_MARKER: everything above it is preserved exactly (your hand-written
    config + comments), everything below is regenerated from `records`. PyYAML does not keep comments,
    so this marker approach is how we edit ONE part of the file without losing the documentation above.

    model_name : the model whose yaml to update.
    records : the full records dict to write (e.g. {"default": {...}, "optuna_best": {...}}).
    returns : the path written.
    """
    path = os.path.join(MODELS_DIR, f"{model_name}.yaml")
    with open(path, "r") as f:
        text = f.read()
    idx = text.find(RECORDS_MARKER)
    head = (text[:idx] if idx != -1 else text).rstrip()          # keep everything above the marker
    body = yaml.safe_dump({"records": records}, default_flow_style=False, sort_keys=False)
    with open(path, "w") as f:
        f.write(head + "\n\n" + RECORDS_MARKER + "\n" + _RECORDS_HELP + "\n" + body)
    return path


def record_metrics(model_name: str, which: str, metrics: Dict, params: Optional[Dict] = None) -> str:
    """
    Update one recorded entry (records.<which>) in a model yaml, keeping the other entry intact.

    model_name : the model whose yaml to update.
    which : "default" or "optuna_best".
    metrics : the metrics dict to store (e.g. {"test_acc": .., "test_loss": .., "test_auroc": ..}).
    params : optional nested params dict to store alongside (used for optuna_best; default has none).
    returns : the path written.
    """
    records = read_records(model_name)
    entry = dict(records.get(which) or {})
    if params is not None:
        entry["params"] = params
    entry["metrics"] = metrics
    records[which] = entry
    return write_records(model_name, records)


def to_flat_dict(cfg, prefix: str = "") -> Dict:
    """
    Flatten a config object into a flat dict with dotted keys, e.g.
    {"model": "seanet", "seed": 0, "model_config.training.learning_rate": 0.00125, ...}.

    This is handy for printing the whole resolved config in one place, and later for logging the
    config to MLflow (which wants flat key/value pairs).

    cfg : a config object (SimpleNamespace) or a piece of one.
    prefix : used by the recursion; leave it empty when you call this.
    returns : a flat dict of dotted-key -> value.
    """
    flat: Dict = {}
    items = vars(cfg).items() if isinstance(cfg, SimpleNamespace) else None
    if items is None:                                    # a scalar (or a list) -> store as-is
        flat[prefix.rstrip(".")] = cfg
        return flat
    for key, value in items:
        if key.startswith("_"):                          # skip private keys (e.g. _param_choice)
            continue
        full_key = f"{prefix}{key}"
        if isinstance(value, SimpleNamespace):           # nested config -> recurse
            flat.update(to_flat_dict(value, prefix=full_key + "."))
        else:                                            # a value (number / string / list)
            flat[full_key] = value
    return flat
