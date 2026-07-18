#!/bin/bash
# launch.sh - ONE command that starts everything. RUN THIS ON THE FRONTEND (flille).
#
#   bash scripts/launch.sh
#
# What it does, in order:
#   1) checks your Telegram setup and SENDS A TEST MESSAGE (stops early if it fails,
#      so you never start a long run with broken notifications)
#   2) asks OAR for a compute node and submits the training there (scripts/run_all.sh)
#   3) starts the phone watcher in the BACKGROUND with nohup, so it keeps running after
#      you close the laptop - no tmux needed
#   4) prints how to check on everything
#
# Why it is split this way (important):
#   The compute NODE has no internet, so it only WRITES the log (logs/run_all.log).
#   The FRONTEND has internet, so it READS that log and sends the Telegram messages.
#   Home is shared (NFS), so both machines see the same file.

cd "$(dirname "$0")/.."      # go to the project root

# ---------------------------------------------------------------------------------------
# Settings you may want to change
# ---------------------------------------------------------------------------------------
CLUSTER="chuc"                    # which cluster to ask for
WALLTIME="12:00:00"               # how long to ask for (hours:minutes:seconds)
LOG="logs/run_all.log"            # the log the node writes and the watcher reads
EVERY="${NOTIFY_EVERY:-1}"        # phone ping every N finished datasets (1 = every dataset)
SECRETS="scripts/telegram_secrets.sh"

mkdir -p logs

# ---------------------------------------------------------------------------------------
# 0) safety checks - make sure we are in the right place with the right tools
# ---------------------------------------------------------------------------------------
if ! command -v oarsub >/dev/null; then
  echo "ERROR: 'oarsub' not found. Run this on the Grid5000 FRONTEND (flille), not on a node."
  exit 1
fi

if [ ! -f "$SECRETS" ]; then
  echo "ERROR: missing $SECRETS"
  echo "Create it (it is git-ignored) with your bot TOKEN and CHAT_ID:"
  echo "  cp scripts/telegram_secrets.example.sh $SECRETS   # then edit it"
  exit 1
fi
source "$SECRETS"                 # loads TOKEN and CHAT_ID

# ---------------------------------------------------------------------------------------
# 1) TEST the phone connection before doing anything heavy.
#    Telegram answers with '"ok":true' when the message really went through, so we check
#    for that instead of just hoping curl worked.
# ---------------------------------------------------------------------------------------
echo "==> testing Telegram connection ..."
RESP=$(curl -s --max-time 15 -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
        -d chat_id="${CHAT_ID}" \
        --data-urlencode text="SEA-Net launcher: connection test OK ($(date))")
case "$RESP" in
  *'"ok":true'*)
    echo "    phone connection OK - check your Telegram, you should see a test message." ;;
  *)
    echo "ERROR: could not send a Telegram message. Telegram replied:"
    echo "    $RESP"
    echo "Fix TOKEN / CHAT_ID in $SECRETS and try again."
    exit 1 ;;
esac

# ---------------------------------------------------------------------------------------
# 2) ask OAR for a node and submit the training on it
#    -t besteffort = starts fast and can be killed; our run is resumable so that is fine.
#    The job runs by itself, so your laptop can be closed.
# ---------------------------------------------------------------------------------------
echo "==> asking OAR for a node on '$CLUSTER' (walltime $WALLTIME, besteffort) ..."
OUT=$(oarsub -t besteffort -q besteffort -p "$CLUSTER" -l walltime="$WALLTIME" \
             -E logs/run_all.err \
             "$PWD/scripts/run_all.sh" 2>&1)
echo "$OUT"

# pull the job id out of the oarsub output (the line looks like: OAR_JOB_ID=2170171)
JOB=$(echo "$OUT" | grep -o 'OAR_JOB_ID=[0-9]*' | cut -d= -f2)
if [ -z "$JOB" ]; then
  echo "ERROR: could not read a job id from oarsub - the job was probably not submitted."
  exit 1
fi
echo "    submitted job $JOB"

# ---------------------------------------------------------------------------------------
# 3) start the phone watcher in the background.
#    nohup + & = it keeps running after you log out / close the laptop (no tmux needed).
#    It reads $LOG (written by the node) and sends the important lines to your phone.
# ---------------------------------------------------------------------------------------
echo "==> starting the phone watcher in the background (every $EVERY dataset) ..."
nohup env NOTIFY_EVERY="$EVERY" bash scripts/notify.sh "$LOG" > logs/notify.out 2>&1 &
WATCHER=$!
echo "    watcher running as process $WATCHER (output: logs/notify.out)"

# ---------------------------------------------------------------------------------------
# 4) tell the user how to check on it
# ---------------------------------------------------------------------------------------
cat <<EOF

============================================================
Everything is started. You can close your laptop now.
============================================================
  training job : $JOB   (on cluster $CLUSTER)
  live log     : $LOG
  phone pings  : every $EVERY finished dataset(s)

Check on it later:
  oarstat -u                 # is the job still Running?
  tail -f $LOG               # live output (Ctrl+C only stops watching)
  cat logs/run_all.err       # errors from the job, if any
  cat logs/notify.out        # output of the phone watcher

Stop things:
  oardel $JOB                # stop the training job
  kill $WATCHER              # stop the phone watcher

If besteffort gets killed, just run this script again - the training resumes
from where it stopped (finished datasets are skipped).
EOF
