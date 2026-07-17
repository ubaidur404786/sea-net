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

LOGFILE="$1"     # the log file to watch, e.g. logs/train_seanet_20260717_120000.log

if [ -z "$LOGFILE" ]; then
  echo "usage: ./scripts/notify.sh <logfile>"
  echo "example: ./scripts/notify.sh logs/train_seanet_20260717_120000.log"
  exit 1
fi

# small helper that posts one text message to your bot
send() {
  curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
       -d chat_id="${CHAT_ID}" --data-urlencode text="$1" >/dev/null
}

send "SEA-Net tracker started, watching $(basename "$LOGFILE")"

# tail -F follows the file even if it is created a bit later or rotated.
# We only forward the IMPORTANT lines (DONE / FAILED / model headers) so your phone
# does not get spammed with every single line.
tail -n0 -F "$LOGFILE" | while read -r line; do
  case "$line" in
    *DONE*|*FAILED*|*"MODEL:"*|*"ALL MODELS DONE"*)
      send "$line"
      ;;
  esac
done
