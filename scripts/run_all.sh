#!/bin/bash -l
# run_all.sh - the REAL run. It trains EVERY model, one after another, on the full
# sweep (WebTraffic + all UCR datasets).
#
# Good news: each model's sweep is RESUMABLE. If the job is killed (this happens with
# besteffort jobs), just start this script again and it continues from where it stopped
# (it skips datasets already finished). So it is safe to re-run.

cd "$(dirname "$0")/.."      # go to the project root
source scripts/env.sh        # turn the environment on

# the models to run, in order. These names match files in configs/models/<name>.yaml.
# (transformer is only a placeholder, so it is left out on purpose.)
MODELS="seanet seanet_acp seanet_classwise seanet_softmax seanet_conjunctive millet fcn resnet"

mkdir -p logs                # keep all log files in one folder

# ONE fixed log file for the whole run. We use a fixed name (not a time-stamp) so the phone
# tracker (scripts/notify.sh) always knows which file to watch - no guessing.
# ">" empties it at the start so each new run begins with a clean log.
# (The app also saves its own detailed per-model logs under results/SEA_NET/<model>/logs/.)
LOG="logs/run_all.log"
: > "$LOG"

# helper: print a line to the screen AND add it to the log file
say() { echo "$1" | tee -a "$LOG"; }

for m in $MODELS; do
  say ""
  say "############################################################"
  say "# MODEL: $m   (started $(date))"
  say "############################################################"
  # tee -a = show on screen AND append to the same log. "2>&1" also saves error messages.
  # We do NOT stop the loop if one model fails: we still try the next model.
  python main.py train --model "$m" 2>&1 | tee -a "$LOG"
done

say ""
say "=== ALL MODELS DONE ($(date)) ==="
say "Next: python main.py results   (comparison table)"
say "      python main.py report    (all figures)"
