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
#   bash scripts/run_all.sh full "$SV4"     # only the sv4 models, full sweep
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
# sv1 = paper baselines, sv2 = original SEA-Net, sv3 = pooling family, sv4 = new encoder+pooling work.
SV1="sv1/millet sv1/fcn sv1/resnet"
SV2="sv2/seanet"
SV3="sv3/seanet_acp sv3/seanet_classwise sv3/seanet_conjunctive sv3/seanet_softmax"
SV4="sv4/seanet_slim sv4/seanet_spiketrend sv4/seanet_gated_max sv4/seanet_gated_mean sv4/seanet_gated_last sv4/seanet_bottleneck sv4/seanet_inputgate sv4/seanet_recon sv4/seanet_gated_last_topk sv4/seanet_gated_last_attnmax"
ALL="$SV1 $SV2 $SV3 $SV4"

# ---- phase: "web" (WebTraffic only, fast screen) or "full" (all datasets, real sweep) ----
PHASE="web"                                   # default
if [ "$1" = "web" ] || [ "$1" = "full" ]; then PHASE="$1"; shift; fi

# ---- which models: any names passed as args, else ALL of them ----
if [ "$#" -gt 0 ]; then MODELS="$*"; else MODELS="$ALL"; fi

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
