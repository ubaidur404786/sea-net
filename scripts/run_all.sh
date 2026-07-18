#!/bin/bash -l
# run_all.sh - the REAL training run. This is the part that runs ON THE COMPUTE NODE.
#
# What it does:
#   1) turns the environment on   (module load conda + activate, via env.sh)
#   2) trains EVERY model, one after another, on the full sweep (WebTraffic + all UCR)
#   3) writes everything into ONE log file: logs/run_all.log
#
# It does NOT send phone messages itself, on purpose: compute nodes have no internet.
# Instead it just SAVES the log, and scripts/notify.sh (running on the frontend, which
# does have internet) reads that same file and sends the messages. Our home folder is
# shared (NFS), so both machines see the same file.
#
# Resumable: if the job is killed (besteffort can be), just start it again and it
# continues from where it stopped (finished datasets are skipped).
#
# "#!/bin/bash -l" means "run me as a login shell", so the "module" command exists even
# when OAR runs this script on its own.

cd "$(dirname "$0")/.."      # go to the project root
source scripts/env.sh        # turn the environment on (prints python + torch/cuda check)

# the models to run, in order. These names match files in configs/models/<name>.yaml.
# (transformer is only a placeholder, so it is left out on purpose.)
MODELS="seanet seanet_acp seanet_classwise seanet_softmax seanet_conjunctive millet fcn resnet"

mkdir -p logs                # keep all log files in one folder

# ONE fixed log file for the whole run. A fixed name (not a time-stamp) means the phone
# watcher always knows which file to read - no guessing.
# ">" empties it at the start so each new run begins with a clean log.
# (The app also saves its own detailed per-model logs under results/SEA_NET/<model>/logs/.)
LOG="logs/run_all.log"
: > "$LOG"

# helper: print a line to the screen AND add it to the log file
say() { echo "$1" | tee -a "$LOG"; }

say "=== SEA-Net run started on $(hostname) at $(date) ==="

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
