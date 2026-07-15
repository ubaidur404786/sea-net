# Session history — branch `seanetv1`

Handoff notes for whoever (human or Claude session) picks this branch up on another
Grid5000 site. Read this before touching `seanet/results.py`, `seanet/config.py`, or `main.py` —
it explains *why* they look the way they do, not just what they do.

## What changed on this branch, in order

1. **Fixed an `auto` recipe-selection deadlock in `seanet/config.py`.**
   `use_params: auto` used to be able to loop/deadlock when neither a `default` nor an
   `optuna_best` record existed yet. Fixed `_select_params` so `auto` falls back to `default`
   when there's nothing to compare against, and only switches to `optuna_best` once it's been
   recorded *and* it beat the default. See `param_choice_message` for the human-readable
   explanation printed at the top of every run.

2. **Added a settings fingerprint** (`seanet/config.py::settings_fingerprint`) — an 8-char
   SHA1 of the encoder + pooling + training config blocks, plus seed and preprocessing.
   Two runs only share a fingerprint if their actual hyperparameters match. Current
   fingerprints: `seanet` → `9eb2ac03`, `seanet_classwise` → `df4a9c11`,
   `seanet_conjunctive` → `69b09f02`. Changing a hyperparameter changes the fingerprint, which
   means a retrained recipe gets its own row/resume-key instead of silently overwriting the old
   one.

3. **Rewrote `seanet/results.py`** around a `model|settings|dataset` resume key
   (`done_key()` / `result_exists()`), so:
   - Each `(model, settings)` pair tracks its own progress across all 129 datasets
     independently. Re-running `python main.py train --model X` skips whatever's already done
     for that exact fingerprint and only trains what's left — safe to Ctrl+C and resume.
   - A new model, or the same model with changed hyperparameters (new fingerprint), retrains
     everything from scratch — nothing is silently reused across different settings.
   - Every result row is stamped `run_at`; results also get written per-invocation to
     `results/SEA_NET/runs/<datetime>_<model>_<command>/results.csv` in addition to the master
     `results/SEA_NET/results.csv`.
   - `build_best_results()` produces `results/SEA_NET/best_results.csv` — one row per dataset,
     the winning model + its settings fingerprint + when it ran, so you can see which head is
     actually winning where without re-deriving it from the raw csv.

4. **The old 129 pre-branch results were migrated, not discarded.** `ensure_migrated()` backs
   up the old `results.csv`/`done.txt` to `results/SEA_NET/archive/`, then tags every old row
   `settings=legacy` and clears `done.txt` of its old (unkeyed) entries. This was a deliberate
   choice: **old rows stay visible as history in the CSV**, but because they carry no real
   fingerprint, the new schema will retrain everything under `seanet`'s real settings hash from
   scratch. Verified lossless (comparison numbers identical pre/post migration: 26/14/45
   win/tie/loss vs MILLET, mean acc 0.8294 vs 0.8445).

5. **MLflow experiments are now named `<base>_<model>_<YYYY-MM-DD>`**
   (`seanet/tracking.py::experiment_name`), e.g. `SEA-Net_seanet_2026-07-15`. Wired through
   `optimize.py` and `benchmark.py` as well as the main train path, so browsing
   `mlflow ui --backend-store-uri sqlite:///mlflow.db` groups runs by model and day instead of
   dumping everything into one experiment.

6. **`main.py` is fully config-driven and takes `--model` on every training subcommand**
   (`train`, `webtraffic`, `single`, `results`), so any model in `configs/models/` can be swept
   or resumed without editing `main.py`. `cmd_train` prints a resume banner
   (`sweep_status()` → done vs. todo) before it starts and rebuilds `best_results.csv` +
   the model's own comparison-vs-MILLET table at the end.

7. **`python main.py report` now auto-updates `README.md`** between
   `<!-- BEGIN AUTO-RESULTS -->` / `<!-- END AUTO-RESULTS -->` markers — headline table (from
   `best_results.csv`, not a blended average across models), a per-model/per-settings summary
   table, and the figure embeds. The old hand-written "Results" section (which had drifted out
   of sync with the actual numbers) was replaced by this generated block. Re-running `report`
   is idempotent — it replaces the block in place.

8. **Explicit scope decision: results/tracking reorg only, no code-layout changes.** The user
   picked "results tree only" when asked about a broader folder reorg — `main.py`, `seanet/`,
   `configs/`, `millet/` stay exactly where they are. Don't restructure directories on a future
   pass unless asked again.

## Where things stand right now (2026-07-15)

- Both `seanet` and `seanet_classwise` currently resolve to the **default** recipe under
  `use_params: auto` (neither has a recorded `optuna_best` that beat its default yet).
- No full sweep has been run yet under the new schema — `results.csv` currently only has the
  129 `settings=legacy` rows plus a smoke-test entry from verifying the pipeline end-to-end.
- Motivation for the next sweep: `results/SEA_NET/pooling_benchmark.csv` showed
  `classwise_conjunctive` beating additive pooling on WebTraffic (acc 0.948 vs 0.938, NDCG
  0.7391) on a small 1-seed/5-dataset benchmark. Plain `conjunctive` actually topped that
  table at 0.960, so treat the classwise result as a hypothesis to test on the full 129
  datasets, not a settled win.
- Planned next step (not yet run — GPU was unavailable on Sophia, moving to Lille): full
  `seanet` and `seanet_classwise` sweeps over all 129 datasets, then compare both against
  MILLET via `python main.py results --model <name>` and refresh the README via
  `python main.py report`. See `GRID_CMD.md` for the exact commands.

## Non-obvious things to know before continuing

- The Bash sandbox on the dev machine silently fails (exit 1, no output) on any script that
  imports `torch` via `python -c`/heredoc. Not relevant on Grid5000 (no sandbox), but if you're
  debugging locally on a laptop dev environment, write the script to a file and run it directly
  instead of inlining it.
- `results.csv` is append-only by design — a csv-realigning tool on the build machine corrupts
  files that get rewritten in place, so `results.py` never rewrites the master file, only
  appends and derives (`best_results.csv`, `summary.csv`) from it.
- `data/`, `mlflow.db`, `mlruns/`, `/model/`, `*.pth` are all gitignored (see `.gitignore`) —
  none of that travels with `git push`/`git pull`. See `GRID_CMD.md` for how to move the
  dataset (`data/`, ~874 MB) between Grid5000 sites via `rsync`.
