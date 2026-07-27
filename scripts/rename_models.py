"""
scripts/rename_models.py - move the results we ALREADY have onto the new naming convention.

Why this script exists
----------------------
In seanetv5 every encoder and pooling name gained an origin tag ("sea_" = ours, "mil_" = MILLET's,
reused unchanged), and a model id now joins its three parts with a double underscore:

    old:  seanet_slim__mstcn_sep_classwise_conjunctive
    new:  seanet_slim__sea_mstcn_sep__sea_classwise_conjunctive

The training results are exactly the same numbers - only the NAME changed. So there is no reason to
retrain anything. This script renames what is on disk instead:

    1. every model folder under results/SEA_NET/,
    2. the "model" / "encoder" / "pooling" columns INSIDE every csv (each model's results.csv and
       summary.csv, plus the shared model_comparison.csv, webtraffic_comparison.csv, leaderboard.csv).

Run it ONCE, on a clean git working tree, and check the output.

How to run
----------
    python scripts/rename_models.py            # DRY RUN: prints what it would do, changes nothing
    python scripts/rename_models.py --apply    # actually do it

After --apply, redraw the figures (this only re-plots, it never retrains):

    python main.py results
    python main.py web-compare
    python main.py leaderboard
    python main.py report

Safety
------
  - dry run is the DEFAULT, so you always look before you leap,
  - it refuses to overwrite an existing folder (that would mean two models colliding),
  - it only rewrites cells that actually match an old name, so running it twice is harmless.
"""
import argparse
import os
import re
import sys

# make "import seanet..." work when this file is run directly from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seanet.data import chdir_to_repo_root                     # noqa: E402
from seanet.results import RESULTS_ROOT                        # noqa: E402

# ---- the rename tables ----------------------------------------------------------------
# Longest name FIRST, so "mstcn_sep_gated" is matched before the shorter "mstcn_sep" can eat it.
ENCODER_RENAMES = [
    ("mstcn_sep_spiketrend", "sea_mstcn_sep_spiketrend"),
    ("mstcn_sep_bottleneck", "sea_mstcn_sep_bottleneck"),
    ("mstcn_sep_inputgate",  "sea_mstcn_sep_inputgate"),
    ("mstcn_sep_gated",      "sea_mstcn_sep_gated"),
    ("mstcn_sep_recon",      "sea_mstcn_sep_recon"),
    ("mstcn_sep",            "sea_mstcn_sep"),
    ("inceptiontime",        "mil_inceptiontime"),
    ("resnet",               "mil_resnet"),
    ("fcn",                  "mil_fcn"),
]

POOLING_RENAMES = [
    ("dualstream_conjunctive", "sea_dualstream_conjunctive"),
    ("classwise_conjunctive",  "sea_classwise_conjunctive"),
    ("softmax_conjunctive",    "sea_softmax_conjunctive"),
    ("adaptive_classwise",     "sea_adaptive_classwise"),
    ("topk_conjunctive",       "sea_topk_conjunctive"),
    ("gated_attention",        "sea_gated_attention"),
    ("attention_max",          "sea_attention_max"),
    ("conjunctive",            "mil_conjunctive"),
    ("attention",              "mil_attention"),
    ("instance",               "mil_instance"),
    ("additive",               "mil_additive"),
    ("gap",                    "mil_gap"),
]

# the shared tables that also hold model names
SHARED_CSVS = ["model_comparison.csv", "webtraffic_comparison.csv", "leaderboard.csv"]


def rename_one(name: str, table) -> str:
    """
    Apply a rename table to one name, and stop at the first match.

    We stop at the first hit on purpose: the tables are ordered longest-first, so the first match is
    always the most specific one. Without the stop, "mstcn_sep_gated" would match "mstcn_sep" too and
    we would end up with a mangled name.

    name : the old encoder / pooling name.
    table : ENCODER_RENAMES or POOLING_RENAMES.
    returns : the new name, or the name unchanged if nothing matched (already renamed, for example).
    """
    if name.startswith(("sea_", "mil_")):      # already done - running twice must be harmless
        return name
    for old, new in table:
        if name == old:
            return new
    return name


def new_model_id(old_id: str):
    """
    Turn one OLD model id into the new one.

        "seanet_slim__mstcn_sep_classwise_conjunctive"
        -> "seanet_slim__sea_mstcn_sep__sea_classwise_conjunctive"

    The hard part is that the old id glued the encoder and the pooling together with a single
    underscore, so "mstcn_sep_classwise_conjunctive" could be read several ways. We solve it by
    trying every known ENCODER as a prefix (longest first) and checking that what is left over is a
    known POOLING. Exactly one pair fits, and that is the split.

    old_id : the folder name as it is today.
    returns : (new id, encoder, pooling), or None if the id cannot be understood.
    """
    parts = old_id.split("__")
    if len(parts) == 3:                                       # already in the new 3-part shape
        return old_id, parts[1], parts[2]
    if len(parts) != 2:
        return None
    config, rest = parts

    for enc_old, enc_new in ENCODER_RENAMES:                  # longest encoder first
        if not rest.startswith(enc_old + "_"):
            continue
        pool_old = rest[len(enc_old) + 1:]                    # everything after "<encoder>_"
        for p_old, p_new in POOLING_RENAMES:
            if pool_old == p_old:
                return f"{config}__{enc_new}__{p_new}", enc_new, p_new
    return None


def rewrite_csv(path: str, id_map: dict, apply: bool) -> int:
    """
    Rewrite the model / encoder / pooling names inside one csv, as plain text.

    Why plain text and not pandas: these files are written by different tools (one of them pads the
    columns with spaces), and reading + rewriting with pandas would reformat every row. A text
    replace changes ONLY the names and leaves the rest of the file byte-for-byte identical.

    path : the csv to fix.
    id_map : {old model id -> new model id}.
    apply : False = only count what would change.
    returns : how many replacements were made.
    """
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    original = text

    # 1. full model ids (longest first, so a short id can never sit inside a longer one)
    for old in sorted(id_map, key=len, reverse=True):
        text = text.replace(old, id_map[old])
    # 2. the standalone "encoder" and "pooling" columns of results.csv. \b is a word boundary, so
    #    "fcn" matches the whole cell and never the "fcn" inside another word.
    for table in (ENCODER_RENAMES, POOLING_RENAMES):
        for old, new in table:
            text = re.sub(rf"(?<![\w-]){re.escape(old)}(?![\w-])", new, text)

    if text == original:
        return 0
    n = sum(1 for a, b in zip(original.split(","), text.split(",")) if a != b)
    if apply:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="actually rename (without this flag it is a dry run)")
    args = parser.parse_args()
    chdir_to_repo_root()

    if not os.path.isdir(RESULTS_ROOT):
        print(f"No results folder at {RESULTS_ROOT} - nothing to do.")
        return

    # ---- work out every rename first, so we can print the whole plan before touching anything ----
    id_map, skipped = {}, []
    for name in sorted(os.listdir(RESULTS_ROOT)):
        path = os.path.join(RESULTS_ROOT, name)
        if not os.path.isdir(path) or name in ("figures", "logs"):
            continue                                          # the shared folders are not models
        result = new_model_id(name)
        if result is None:
            skipped.append(name)
            continue
        new_id, _enc, _pool = result
        if new_id != name:
            id_map[name] = new_id

    mode = "APPLYING" if args.apply else "DRY RUN (nothing is changed; add --apply to do it)"
    print(f"=== rename_models.py - {mode} ===\n")

    if skipped:
        print(f"!! {len(skipped)} folder(s) could not be understood and are LEFT ALONE:")
        for name in skipped:
            print(f"     {name}")
        print()

    if not id_map:
        print("Every model folder is already on the new naming convention.")
    else:
        print(f"{len(id_map)} model folder(s) to rename:")
        for old, new in id_map.items():
            print(f"  {old}\n    -> {new}")
        print()

    # ---- 1. rename the folders ----
    clashes = [new for new in id_map.values() if os.path.exists(os.path.join(RESULTS_ROOT, new))]
    if clashes:
        print("STOPPING: these target folders already exist, renaming would overwrite them:")
        for name in clashes:
            print(f"     {name}")
        sys.exit(1)

    if args.apply:
        for old, new in id_map.items():
            os.rename(os.path.join(RESULTS_ROOT, old), os.path.join(RESULTS_ROOT, new))
        print(f"renamed {len(id_map)} folder(s).")

    # ---- 2. fix the names written INSIDE the csv files ----
    # after the folder rename the per-model csvs live under the NEW folder names
    targets = []
    for name in (id_map.values() if args.apply else id_map.keys()):
        for fname in ("results.csv", "summary.csv", "summary.md", "comparison_vs_millet.csv"):
            targets.append(os.path.join(RESULTS_ROOT, name, fname))
    # models that were already correctly named still have OLD encoder/pooling cells inside them
    for name in sorted(os.listdir(RESULTS_ROOT)):
        if os.path.isdir(os.path.join(RESULTS_ROOT, name)) and name not in ("figures", "logs"):
            for fname in ("results.csv", "summary.csv"):
                p = os.path.join(RESULTS_ROOT, name, fname)
                if p not in targets:
                    targets.append(p)
    targets += [os.path.join(RESULTS_ROOT, f) for f in SHARED_CSVS]

    total, touched = 0, 0
    for path in targets:
        n = rewrite_csv(path, id_map, args.apply)
        if n:
            touched += 1
            total += n
            print(f"  {'fixed' if args.apply else 'would fix'} {path}")
    print(f"\n{'fixed' if args.apply else 'would fix'} names inside {touched} file(s).")

    if args.apply:
        print("\nDone. Now redraw the tables and figures (no retraining happens):")
        print("  python main.py results")
        print("  python main.py web-compare")
        print("  python main.py leaderboard")
        print("  python main.py report")
    else:
        print("\nThis was a DRY RUN. Re-run with --apply to make the changes.")


if __name__ == "__main__":
    main()
