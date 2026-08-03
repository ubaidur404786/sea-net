#!/bin/bash -l
# run_all.sh - the REAL training run. This is the part that runs ON THE COMPUTE NODE.
#
# What it does:
#   1) turns the environment on   (module load conda + activate, via env.sh)
#   2) trains the models you ask for, one after another
#   3) writes everything into ONE log file under logs/
#
# It does NOT send phone messages itself, on purpose: compute nodes have no internet. Instead it just
# SAVES the log, and scripts/notify.sh (on the frontend, which has internet) reads it and sends them.
#
# Resumable: if the job is killed, just start it again - finished datasets are skipped.
#
# --------------------------------------------------------------------------------------
# HOW TO CALL IT
#
#   bash scripts/run_all.sh                 # PHASE = web (default): every model on WebTraffic ONLY
#   bash scripts/run_all.sh web             # same as above, said explicitly
#   bash scripts/run_all.sh full            # PHASE = full: every model on ALL datasets (the real sweep)
#   bash scripts/run_all.sh web sv4/seanet_gated_last sv4/seanet_recon   # only these models (WebTraffic)
#   bash scripts/run_all.sh web sv6         # only the sv6 models (multi-scale), WebTraffic screen
#   bash scripts/run_all.sh full sv4        # only the sv4 models, full sweep
#
# ONE COMMAND PER LINE. If you paste two commands onto the same line they get joined into one, and every
# extra word is read as a model name - which is why a stray paste shows up as "config file not found".
#
# The plan: FIRST run the "web" phase to screen every model on WebTraffic fast (one dataset), read the
# accuracy + AOPCR, pick the winners, THEN run "full" on the winners for the whole benchmark.
#
# Passing model names as args is also how we SPLIT work over two nodes: give each node a different half.
# It is safe because every model writes to its own folder (results/SEA_NET/<model>/).
# --------------------------------------------------------------------------------------

cd "$(dirname "$0")/.."      # go to the project root
source scripts/env.sh        # turn the environment on (prints python + torch/cuda check)

# ---- the models, grouped by VERSION (the folder each config lives in) ----
# sv1 = paper baselines, sv2 = original SEA-Net, sv3 = pooling family, sv4 = new encoder+pooling work,
# sv5 = MIL pooling heads ported from the cancer/pathology papers (gated_attention, dualstream_conjunctive),
# sv6 = multi-scale encoder wrappers (channels + pyramid).
#
# BOTH the folders AND the files inside them are found automatically. The version names used to be
# hand-listed here, and that was a trap: adding configs/models/sv6/ was not enough, you also had to
# remember to edit this script, and if you forgot, "run_all.sh web sv6" fell through to "treat sv6 as a
# model name" and died with 'configs/models/sv6.yaml not found'. Now a new folder just works.
list_models() {
  local v="$1"
  for f in configs/models/"$v"/*.yaml; do
    [ -e "$f" ] || continue                        # folder empty -> skip
    echo "$v/$(basename "$f" .yaml)"
  done
}

list_versions() {                                  # every folder under configs/models/, e.g. sv1 .. sv6
  for d in configs/models/*/; do
    [ -d "$d" ] || continue
    basename "$d"
  done
}

ALL=""
for v in $(list_versions); do
  ALL="$ALL $(list_models "$v")"
done

# ---- phase: "web" (WebTraffic only, fast screen) or "full" (all datasets, real sweep) ----
PHASE="web"                                   # default
if [ "$1" = "web" ] || [ "$1" = "full" ]; then PHASE="$1"; shift; fi

# ---- which models to run ----
# You can pass:
#   nothing          -> ALL models (every version)
#   a version group  -> sv1 | sv2 | ... | sv6 | all   (any folder under configs/models/)
#   explicit names   -> sv6/seanet_bottleneck_mschan sv4/seanet_recon ...
# IMPORTANT: do NOT type "$SV4" from your shell - that variable only exists INSIDE this script, so from
# your terminal it is empty. Use the WORD  sv4  instead:   bash scripts/run_all.sh full sv4
MODELS=""
if [ "$#" -eq 0 ]; then
  MODELS="$ALL"
else
  for sel in "$@"; do
    [ -z "$sel" ] && continue                    # skip empty tokens (e.g. an unset "$SV4")
    if [ "$sel" = "all" ]; then
      MODELS="$MODELS $ALL"
    elif [ -d "configs/models/$sel" ]; then
      MODELS="$MODELS $(list_models "$sel")"     # a version folder like sv4 or sv6
    else
      MODELS="$MODELS $sel"                      # an explicit model name like sv4/seanet_recon
    fi
  done
fi
MODELS=$(echo $MODELS | xargs)                   # collapse whitespace / newlines, trim the ends

# guard: if nothing valid was selected, STOP LOUDLY instead of silently printing "ALL DONE".
if [ -z "$MODELS" ]; then
  echo "ERROR: no models selected (did you pass an empty \"\$SV4\"? use the word sv4 instead). Examples:"
  echo "  bash scripts/run_all.sh full sv4                 # all sv4 models, full sweep"
  echo "  bash scripts/run_all.sh web                      # all models, WebTraffic only"
  echo "  bash scripts/run_all.sh full sv4/seanet_recon sv4/seanet_slim_topk"
  exit 1
fi

# guard: check EVERY selected model has a config file BEFORE training anything. Without this a typo
# only shows up as a Python traceback once that model's turn comes - and with several typos you get
# several tracebacks scrolling past, which is what a joined copy-paste ("web" + the next command on the
# same line) looks like. Better to say exactly which names are wrong and stop.
MISSING=""
for m in $MODELS; do
  [ -f "configs/models/$m.yaml" ] || MISSING="$MISSING $m"
done
if [ -n "$MISSING" ]; then
  echo "ERROR: these are not model configs:$MISSING"
  echo "A model name looks like  <version folder>/<config file>, e.g. sv6/seanet_bottleneck_mschan."
  echo "Available version folders: $(list_versions | xargs)"
  echo "Tip: if you see the words of your NEXT command in that list, two commands were pasted onto one line."
  exit 1
fi

mkdir -p logs
# name the log after the phase + the first model, so two nodes / two phases never share a log file.
TAG=$(echo "$MODELS" | awk '{print $1}' | tr '/' '_')
LOG="logs/run_all_${PHASE}_${TAG}.log"
: > "$LOG"

# helper: print a line to the screen AND add it to the log file
say() { echo "$1" | tee -a "$LOG"; }

say "=== SEA-Net [$PHASE] run started on $(hostname) at $(date) ==="
say "models: $MODELS"

for m in $MODELS; do
  say ""
  say "############################################################"
  say "# MODEL: $m   (started $(date))"
  say "############################################################"
  if [ "$PHASE" = "web" ]; then
    # WebTraffic only: fast single-dataset run (dataset comes from configs/main.yaml = WebTraffic).
    python main.py run --model "$m" 2>&1 | tee -a "$LOG"
  else
    # full sweep: WebTraffic + all 128 UCR datasets (resumable).
    python main.py train --model "$m" 2>&1 | tee -a "$LOG"
  fi
done

say ""
say "=== ALL MODELS DONE ($(date)) ==="
if [ "$PHASE" = "web" ]; then
  say "This was the WebTraffic screen: read each model's test_acc / test_aopcr above, pick the winners,"
  say "then run the full sweep on them:   bash scripts/run_all.sh full <winner1> <winner2> ..."
else
  say "Next: python main.py results   (cross-model table)"
  say "      python main.py report    (all figures, comparison uses short m1/m2 labels + a legend)"
fi
