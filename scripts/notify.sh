#!/bin/bash
# notify.sh - send live training updates to your PHONE via a free Telegram bot.
#
# IMPORTANT: run this ON THE FRONTEND (flille), NOT on the compute node.
# Why: the frontend has internet, the compute nodes usually do NOT. The training job
# writes its log to your shared home folder (NFS), so the frontend can read that same
# file live and forward the important lines to Telegram. Your phone then gets a push
# notification - even if your laptop is closed.
#
# One-time setup (see GRID5K_CMD_HELP.md, section 8):
#   1) In Telegram, talk to @BotFather -> /newbot -> copy the TOKEN it gives you.
#   2) Send any message to your new bot once.
#   3) Get your CHAT_ID (the guide shows the exact curl command).
#   4) Put TOKEN and CHAT_ID in scripts/telegram_secrets.sh (that file is git-ignored,
#      so your secret token never gets committed).

# load our private token + chat id from a file that git does NOT track.
# keeping secrets out of the code is a good habit - the token controls the whole bot.
SECRETS="$(dirname "$0")/telegram_secrets.sh"
if [ -f "$SECRETS" ]; then
  source "$SECRETS"
else
  echo "missing $SECRETS - create it with your TOKEN and CHAT_ID (see GRID5K_CMD_HELP.md section 8)"
  exit 1
fi

# which log to watch. Default is the fixed file run_all.sh writes to, so you don't have
# to guess a name. You can still pass a different file: ./scripts/notify.sh some.log
LOGFILE="${1:-logs/run_all.log}"

# how often to send a "progress" ping. The full run finishes ~1000 datasets, so sending
# every single one would flood your phone. We send one progress line every EVERY datasets.
# Default is 25. For a quick TEST you can make it ping every dataset like this:
#   NOTIFY_EVERY=1 bash scripts/notify.sh
EVERY="${NOTIFY_EVERY:1}"

# small helper that posts one text message to your bot
send() {
  curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
       -d chat_id="${CHAT_ID}" --data-urlencode text="$1" >/dev/null
}

# make sure the log file exists before we watch it, so tail never fails if the run
# has not started yet (the folder may not exist on a fresh checkout).
mkdir -p "$(dirname "$LOGFILE")"
touch "$LOGFILE"

send "SEA-Net tracker started, watching $(basename "$LOGFILE")"

# tail -F follows the file even if it does not exist yet (it waits for it) or is recreated.
# -n0 means "start from the end", so we only get NEW lines from now on.
#
# What we forward to your phone (and what we skip):
#   - a model header  (MODEL:)            -> always send  (tells you model X of 8 started)
#   - a failure       (FAILED)            -> always send  (you want to know immediately)
#   - the final line  (ALL MODELS DONE)   -> always send
#   - a finished dataset (DONE)           -> send only every EVERY-th one, as a progress ping
#     (we count them so your phone gets ~40 progress pings, not ~1000)
count=0
tail -n0 -F "$LOGFILE" | while read -r line; do
  case "$line" in
    *"MODEL:"*|*"ALL MODELS DONE"*|*FAILED*)
      send "$line"
      ;;
    *DONE*)
      count=$((count + 1))
      # ping on the VERY FIRST finished dataset (so you quickly see it is really tracking),
      # then only every EVERY-th one after that (so your phone is not flooded).
      if [ "$count" -eq 1 ] || [ $((count % EVERY)) -eq 0 ]; then
        send "progress [$count done]: $line"
      fi
      ;;
  esac
done
