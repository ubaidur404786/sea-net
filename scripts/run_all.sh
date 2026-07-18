#!/bin/bash -l
# run_all.sh - the REAL run, now FULLY self-contained. Started once (by hand or by OAR),
# it does EVERYTHING on its own:
#   1) turns the environment on            (module load conda + activate, via env.sh)
#   2) trains every model on the full sweep (WebTraffic + all UCR), resumable
#   3) saves one combined log              (logs/run_all.log)
#   4) messages your PHONE after each dataset, on failures, and at the end (Telegram)
# So you do NOT need a second terminal running notify.sh - this script does the notifying.
#
# Resumable: if the job is killed (besteffort can be), just start it again and it continues
# from where it stopped (finished datasets are skipped).
#
# INTERNET NOTE: compute nodes have no direct internet, so we send Telegram messages through
# Grid5000's HTTP proxy (PROXY below). Test the proxy once with:  scripts/test_run.sh has a
# note, or run the quick curl test from GRID5K_CMD_HELP.md. If your site's proxy name differs,
# change PROXY here.

cd "$(dirname "$0")/.."      # go to the project root
source scripts/env.sh        # turn the environment on (prints python + torch/cuda check)

# ---------------------------------------------------------------------------------------
# Phone notifications. These turn ON only if scripts/telegram_secrets.sh exists. If sending
# ever fails (e.g. proxy blocked), we ignore the error so training is NEVER interrupted.
# ---------------------------------------------------------------------------------------
PROXY="http://proxy:3128"                    # Grid5000 proxy that gives compute nodes internet
SECRETS="scripts/telegram_secrets.sh"
NOTIFY=0
if [ -f "$SECRETS" ]; then
  source "$SECRETS"                          # loads TOKEN and CHAT_ID (this file is git-ignored)
  export http_proxy="$PROXY" https_proxy="$PROXY"   # so curl on the node can reach the internet
  NOTIFY=1
fi

# send one Telegram message. Does nothing if notifications are off, and never breaks the run
# (--max-time stops it hanging, "|| true" swallows any error).
send() {
  [ "$NOTIFY" -eq 1 ] || return 0
  curl -s --max-time 15 -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
       -d chat_id="${CHAT_ID}" --data-urlencode text="$1" >/dev/null 2>&1 || true
}

# the models to run, in order. These names match files in configs/models/<name>.yaml.
# (transformer is only a placeholder, so it is left out on purpose.)
MODELS="seanet seanet_acp seanet_classwise seanet_softmax seanet_conjunctive millet fcn resnet"

mkdir -p logs                # keep all log files in one folder
LOG="logs/run_all.log"       # one fixed combined log for the whole run
: > "$LOG"                   # start clean each run

# helper: print a line to the screen AND add it to the log file
say() { echo "$1" | tee -a "$LOG"; }

# a first message so you know it really started (includes which node + the cuda check)
CUDA=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null)
send "SEA-Net run started on $(hostname) [$CUDA] - $(date)"

for m in $MODELS; do
  say ""
  say "############################################################"
  say "# MODEL: $m   (started $(date))"
  say "############################################################"
  send "MODEL $m started"

  # Run training and watch its output line by line.
  #   tee -a "$LOG"  -> writes every line (incl. live progress) to the combined log
  #   the while loop -> reacts to the important lines and pings your phone
  # We do NOT stop the loop if one model fails: we still try the next model.
  python main.py train --model "$m" 2>&1 | tee -a "$LOG" | while read -r line; do
    case "$line" in
      *DONE*)   send "$line" ;;             # a dataset finished -> send its full result line
      *FAILED*) send "FAILED: $line" ;;     # a dataset failed  -> tell me right away
    esac
  done
done

say ""
say "=== ALL MODELS DONE ($(date)) ==="
send "ALL MODELS DONE - $(date)"
say "Next: python main.py results   (comparison table)"
say "      python main.py report    (all figures)"
