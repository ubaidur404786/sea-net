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
STAMP=$(date +%Y%m%d_%H%M%S) # same time-stamp for every log file of this run

for m in $MODELS; do
  echo ""
  echo "############################################################"
  echo "# MODEL: $m   (started $(date))"
  echo "############################################################"
  # tee = show on screen AND save to a log file. "2>&1" also saves error messages.
  # We do NOT stop the loop if one model fails: we still try the next model.
  python main.py train --model "$m" 2>&1 | tee "logs/train_${m}_${STAMP}.log"
done

echo ""
echo "=== ALL MODELS DONE ($(date)) ==="
echo "Next: python main.py results   (comparison table)"
echo "      python main.py report    (all figures)"
