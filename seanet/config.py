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
                overrides: Optional[Dict] = None) -> SimpleNamespace:
    """
    Load main.yaml + the model file it points at, and return one merged config object.

    Steps:
      1. read main.yaml,
      2. work out the model name (an override wins over the file),
      3. read configs/models/<model>.yaml and attach it under the key "model_config",
      4. apply any leftover overrides,
      5. convert the whole thing to a SimpleNamespace so it can be read with dots.

    main_path : path to main.yaml.
    overrides : optional dict of values to override (e.g. {"model": "millet"} from the command
                line). Only "model" changes which model file is read; other keys are merged in
                after the model file is attached.
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

    # if Optuna has saved best hyperparameters for this model, merge them on top (they win over the
    # base file, so future runs automatically use the tuned values). Delete the .best.yaml to undo.
    best_path = os.path.join(MODELS_DIR, f"{model_name}.best.yaml")
    if os.path.exists(best_path):
        model_cfg = _merge(model_cfg, _read_yaml(best_path))

    # attach the model file under "model_config" and record the chosen model name at the top level
    merged = _merge(main, {"model": model_name, "model_config": model_cfg})
    # apply the remaining overrides (model was already handled, but merging it again is harmless)
    merged = _merge(merged, overrides)

    return _to_namespace(merged)


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
        full_key = f"{prefix}{key}"
        if isinstance(value, SimpleNamespace):           # nested config -> recurse
            flat.update(to_flat_dict(value, prefix=full_key + "."))
        else:                                            # a value (number / string / list)
            flat[full_key] = value
    return flat
