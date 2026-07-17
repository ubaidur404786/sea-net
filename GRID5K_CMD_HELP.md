# Grid5000 Command Help — SEA-Net (Lille)

This is a start-to-end guide for running SEA-Net on the Grid5000 **Lille** server.
It is written so that next time you can just follow the steps without thinking hard.

Everything here is tested against **your real server environment**:

| Thing | Value on Lille (`flille`) |
|---|---|
| Load conda | `module load conda` (Grid5000 conda 23.5.0) |
| Activate env | `conda activate seanet` |
| Env path | `~/miniforge3/envs/seanet` |
| Python | 3.10.20 |
| torch | 2.0.1+cu118 (built for CUDA 11.8) |
| Project folder | `~/projects/sea-net` |

> **Golden rule:** the *code* is the same on the laptop and the server, but the
> *environment* is different. On the server there is **no `~/.bashrc`** and **no conda on
> the PATH**, so you must load conda every time with `module load conda`.

---

## The big picture (read this once)

There are 3 machines in play:

1. **Your laptop** — you SSH from here. If you close it, the SSH connection dies.
2. **The frontend** (`flille`) — where you land after SSH. It **has internet**. It is only
   for light work: editing, submitting jobs, watching logs. **Do NOT train here.**
3. **A compute node** — a real GPU machine you *reserve*. Training happens here. Compute
   nodes usually have **no internet**.

So the plan is:
- reserve a GPU node,
- run the training there,
- keep it alive even if the laptop closes (using `tmux`),
- watch progress on your phone (using a free Telegram bot that runs on the frontend).

---

## Step 0 — Connect and where you land

After you SSH in, you land on the frontend and see something like:

```
urehman@flille:~/projects/sea-net$
```

`flille` = the Lille frontend. Good. Now go to the project (if not already there):

```bash
cd ~/projects/sea-net
```

---

## Step 1 — Turn ON the environment

Do this **every time** you open a new shell on the server:

```bash
module load conda
conda activate seanet
```

Check it worked:

```bash
which python
# should print: /home/urehman/miniforge3/envs/seanet/bin/python
python -c "import torch; print(torch.__version__)"
# should print: 2.0.1+cu118
```

**What each line does (teacher note):**
- `module load conda` — Grid5000 keeps software in "modules". This turns conda on for
  this shell. We need it because there is no `~/.bashrc` to do it automatically.
- `conda activate seanet` — switches on our project env (the one with torch, mlflow, etc.).

We already put all of this inside **`scripts/env.sh`**, so later you can simply do
`source scripts/env.sh` instead of typing the two lines.

---

## Step 2 — Reserve a compute node (the important part)

Grid5000 uses a scheduler called **OAR**. You ask it for a machine with `oarsub`.
There are two modes. Here is the clear difference:

### 2a. Interactive mode (`-I`) — you get a live shell

```bash
oarsub -I -l gpu=1,walltime=4:00:00
```

- `-I` = interactive. When the node is free, OAR **drops you into a shell on the node**.
- `-l gpu=1` = give me 1 GPU.
- `walltime=4:00:00` = for 4 hours (format is `hours:minutes:seconds`).

You then type commands by hand on the node. Best for **testing** and for **watching a run
live**. The reservation is **guaranteed** for the whole 4 hours once granted.

> Tip: to pick a specific GPU cluster on Lille, add `-p`, e.g.
> `oarsub -I -p "cluster='chifflot'" -l gpu=1,walltime=4:00:00`.
> Check the Lille hardware page for cluster names if `gpu=1` alone can't find a node.

### 2b. Besteffort mode (`-t besteffort`) — runs by itself, no babysitting

```bash
oarsub -t besteffort -l gpu=1,walltime=24:00:00 "$HOME/projects/sea-net/scripts/run_all.sh"
```

- `-t besteffort` = low-priority job. It runs a **script** for you on the node, with **no
  live shell**. You don't need to stay connected at all — your laptop can be fully off.
- The catch: a besteffort job **can be killed at any moment** if someone with a normal job
  needs that machine. That's fine for us, because `run_all.sh` is **resumable** — just
  submit it again and it continues where it stopped.

### Interactive vs Besteffort — quick table

| | Interactive (`-I`) | Besteffort (`-t besteffort`) |
|---|---|---|
| You get | a live shell on the node | a job that runs a script by itself |
| You must stay connected? | yes (unless you use `tmux`, see Step 3) | no — laptop can be off |
| Can it be killed early? | no, guaranteed for the walltime | yes, anytime someone needs the node |
| Wait time to start | can wait if busy | usually starts fast (fills gaps) |
| Best for | testing, watching live | long full runs, hands-off |
| Time limit | short (hours) | can be very long (our run is resumable) |

**Simple advice:**
- First time / testing → **interactive** (Step 2a), so you can watch and learn.
- The real full run of all models → **besteffort** (Step 2b), so it survives and needs no
  babysitting.

### Handy OAR commands

```bash
oarstat -u            # list my jobs and their state (Waiting / Running)
oarsub -C <job_id>    # re-connect to my running interactive job (get the shell back)
oardel <job_id>       # cancel a job
```

---

## Step 3 — Keep it alive when the laptop closes (`tmux`)

If you use **interactive** mode and close your laptop, the SSH dies and the job dies with
it. The fix is `tmux`: it keeps a shell running on the **frontend** even after you
disconnect.

```bash
# on the frontend, BEFORE you run oarsub:
tmux new -s train        # start a named session "train"
# ... now do Step 2a (oarsub -I) and run your scripts inside here ...
```

To leave it running and close your laptop: press **Ctrl+b** then **d** (this "detaches").
Later, reconnect and get it back:

```bash
tmux attach -t train     # come back to the same session
```

> With **besteffort** (Step 2b) you don't even need `tmux` — the job already runs without
> you. `tmux` is mainly for the interactive case.

---

## Step 4 — Run the TEST script FIRST (always)

Before the big run, prove the pipeline works with a tiny smoke test. This trains for only
3 epochs on one small dataset and saves nothing — it just checks for errors.

Get an interactive node (Step 2a), then:

```bash
cd ~/projects/sea-net
bash scripts/test_run.sh
```

If it finishes with accuracy/loss numbers and **no red errors**, your setup is good and you
can move on. If it errors, fix that first (usually a missing package or wrong env).

---

## Step 5 — Run the REAL script (all models, step by step)

This trains **every** model, one after another, on the full sweep (WebTraffic + all UCR).
The models it runs (from `configs/models/`):

```
seanet  seanet_acp  seanet_classwise  seanet_softmax  seanet_conjunctive  millet  fcn  resnet
```

### Option A — inside an interactive node (watch it live)

```bash
cd ~/projects/sea-net
bash scripts/run_all.sh
```

Each model's output is shown on screen **and** saved to `logs/train_<model>_<time>.log`.

### Option B — besteffort (hands-off, recommended for the full run)

From the frontend (no node needed first — OAR gets one for you):

```bash
cd ~/projects/sea-net
oarsub -t besteffort -l gpu=1,walltime=24:00:00 "$HOME/projects/sea-net/scripts/run_all.sh"
```

Now close your laptop if you want. If the job gets killed, just run the same `oarsub`
line again — it resumes automatically.

---

## Step 6 — See the results

**While it runs — watch the log live:**

```bash
tail -f logs/train_seanet_*.log        # follow the newest seanet log (Ctrl+C to stop watching)
```

**Numbers per model** (saved automatically):

```bash
cat results/SEA_NET/*/results.csv      # each model has its own folder + results.csv
```

**Build the comparison table and figures** (after the run finishes):

```bash
python main.py results     # win/tie/loss comparison vs MILLET
python main.py report      # draws all the figures under results/SEA_NET/
```

**Browse everything in MLflow on your laptop later:** copy the `mlflow.db` file back and
open the web page. Full steps are in `seanet/MLFLOW_GUIDE.md` (section 4). Short version,
run on your **laptop**:

```bash
scp urehman@access.grid5000.fr:~/projects/sea-net/mlflow.db ./
mlflow ui --backend-store-uri sqlite:///mlflow.db      # then open http://127.0.0.1:5000
```

---

## Step 7 — Free live tracking on your PHONE (Telegram bot)

Goal: watch progress from your phone, for free, even with the laptop closed.

**Why Telegram:** it is completely free, has phone push notifications, and sending a
message is just one simple web request. Our watcher runs on the **frontend** (which has
internet) and reads the log file the node writes to shared home — so nothing extra is
needed on the compute node.

### One-time setup (5 minutes)

You do NOT create a bot from a menu. In Telegram you make a bot by chatting with a special
bot called **@BotFather** (BotFather is a robot that builds other robots).

1. On your phone, open Telegram, tap search (🔍), type **BotFather**, open the one with the
   blue checkmark ✔️, tap **START**.
2. Send `/newbot`. Give it a name, then a username that ends in `bot` (e.g. `seanet_xxx_bot`).
   BotFather replies with a **token** (looks like `123456789:AAE...`). That token is your
   `TOKEN`.
3. Open your new bot (search its username), tap **START**, and send it any message like `hi`.
   This matters: a bot can't message you until you message it first.
4. Get your **chat id**. On the frontend, run (paste your real token in place of `<TOKEN>`):

   ```bash
   curl -s "https://api.telegram.org/bot<TOKEN>/getUpdates"
   ```

   In the output find `"chat":{"id":<NUMBER>,...`. That `<NUMBER>` is your `CHAT_ID`.
   (If the output is `{"ok":true,"result":[]}`, you skipped step 3 — send the bot a message,
   then run this again.)
5. Put your `TOKEN` and `CHAT_ID` into a **private, git-ignored** file. On the frontend:

   ```bash
   cd ~/projects/sea-net
   cp scripts/telegram_secrets.example.sh scripts/telegram_secrets.sh
   nano scripts/telegram_secrets.sh      # paste your real TOKEN and CHAT_ID, then save
   ```

   We keep the token in `scripts/telegram_secrets.sh` (git ignores it) instead of inside
   `notify.sh`, so the secret never gets committed to git. `notify.sh` loads it automatically.

6. Test it (frontend) — this should ping your phone:

   ```bash
   source scripts/telegram_secrets.sh
   curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
        -d chat_id="${CHAT_ID}" --data-urlencode text="SEA-Net test - it works!"
   ```

### Use it

Start your training first (Step 5), note the log file name it created, then on the
**frontend** (inside a `tmux` session so it keeps running):

```bash
tmux new -s notify
cd ~/projects/sea-net
bash scripts/notify.sh logs/train_seanet_20260717_120000.log   # use your real log name
# Ctrl+b then d to detach; your phone keeps getting updates
```

You'll get a phone message for each model that starts, each dataset that finishes (DONE),
each failure (FAILED), and when everything is done. Laptop can be closed the whole time.

> Want a full live *dashboard* (charts) on your phone instead of text messages? That is
> possible with Weights & Biases (free tier), but it needs extra proxy setup because the
> compute nodes have no internet, and a small code change to log to it. Telegram is the
> simplest free option, so we start there.

---

## Step 8 — Launch AUTOMATICALLY (batch job) + the git update/pull workflow

This is the hands-off way: edit code on the laptop, push it, pull it on the server, and let
OAR start the run **by itself** when a node is free — even with your laptop closed.

### Interactive vs batch (why we switch)

`oarsub -I` (interactive) is NOT automatic: it freezes your terminal until a node is free,
you must stay connected, and closing the laptop loses it. If the cluster is busy you might
wait many hours staring at:

```
# Interactive mode: waiting...
# [..] Start prediction: <tomorrow>
```

Instead, submit a **batch** job (drop `-I`, pass a script). OAR runs the script for you the
moment a node frees up. You get your prompt back right away and can log off.

### Will my run survive if the internet / laptop drops? (IMPORTANT)

Your connection is: **laptop → (Wi-Fi / mobile hotspot) → frontend → node.** If the hotspot
drops or the laptop sleeps, your SSH dies. What happens next depends on HOW you launched:

| How you launched | Hotspot drops / laptop closes → |
|---|---|
| `oarsub -I` then ran the script by hand (no tmux) | ❌ SSH dies → job dies → training **stops** |
| `oarsub -I` **inside `tmux` on the frontend** | ✅ tmux keeps it alive → training **continues** |
| **besteffort / batch** job (`oarsub ... script.sh`) | ✅ runs on the node by itself → training **continues** |

So a plain interactive run is **not** safe against a disconnect. For an unattended run
(hotspot may drop, laptop may sleep), use **besteffort** — that is exactly what it is for.
Either way our run is **resumable**: after any stop, run `run_all.sh` again and it continues
from where it left off (only the dataset it was mid-way through is repeated).

The phone tracker (`notify.sh`) runs on the frontend, so put it inside `tmux` too — then it
keeps pinging your phone even after your laptop disconnects.

### The full automatic flow

**1) On the laptop — save and push your code:**
```bash
git add -A
git commit -m "what I changed"
git push
```

**2) On the frontend — get that code and turn the env on:**
```bash
cd ~/projects/sea-net
git pull
module load conda
conda activate seanet
```

**3) Submit the besteffort job (survives a disconnect, laptop can be closed):**
```bash
mkdir -p logs
oarsub -t besteffort -q besteffort -p chuc -l walltime=12:00:00 \
       -E logs/run_all.err \
       ~/projects/sea-net/scripts/run_all.sh
```
- no `-I` = OAR runs the script for you, independent of your SSH.
- `run_all.sh` writes its own combined log to `logs/run_all.log` (that is the file the phone
  tracker watches), so we only need `-E` here to also capture any OAR-level errors.
- prints an `OAR_JOB_ID` and returns your prompt immediately. `run_all.sh` is resumable, so
  if besteffort gets killed just submit the same line again.

**4) Watch it / track on phone:**
```bash
oarstat -u                    # Waiting or Running?
tail -f logs/run_all.log      # live output once it starts

# phone tracking (own tmux; tail -F waits for the file, so start it any time):
tmux new -s notify
bash scripts/notify.sh logs/run_all.log
# Ctrl+b then d to detach
```

> **Prefer a guaranteed (non-killable) slot instead of besteffort?** Drop `-t besteffort`
> and `-q besteffort` and it becomes a normal batch job (waits its turn, then runs to the
> end): `oarsub -q default -p chuc -l walltime=4:00:00 -E logs/run_all.err ~/projects/sea-net/scripts/run_all.sh`

> **Do a smoke test first the same way** (replace the script):
> ```bash
> oarsub -t besteffort -q besteffort -p chuc -l walltime=0:30:00 \
>        -E logs/test.err \
>        ~/projects/sea-net/scripts/test_run.sh
> ```

### Useful job commands
```bash
oarstat -u            # my jobs and their state
oardel <job_id>       # cancel a job (e.g. an interactive one stuck waiting)
cat logs/run_all.err  # read errors if something failed
```

---

## Quick cheat-sheet (copy-paste order)

```bash
# 1) connect, then on the frontend:
cd ~/projects/sea-net
tmux new -s train

# 2) turn env on
module load conda
conda activate seanet

# 3) get a GPU node (interactive, 4h)
oarsub -I -l gpu=1,walltime=4:00:00

# 4) on the node: test first, then the real run
cd ~/projects/sea-net
bash scripts/test_run.sh
bash scripts/run_all.sh

# --- OR, hands-off full run (from the frontend, no node needed first) ---
oarsub -t besteffort -l gpu=1,walltime=24:00:00 "$HOME/projects/sea-net/scripts/run_all.sh"

# 5) phone tracking (frontend, in its own tmux)
bash scripts/notify.sh logs/train_seanet_<stamp>.log
```

---

## Troubleshooting

- **`conda: command not found`** → you forgot `module load conda`.
- **`Could not find conda environment: seanet`** → run once:
  `conda config --append envs_dirs ~/miniforge3/envs`, or activate by full path:
  `conda activate ~/miniforge3/envs/seanet`.
- **`cuda available: False` on the frontend** → normal, the frontend has no GPU. It should
  be `True` on a reserved GPU node.
- **Besteffort job disappeared** → it was killed (that's expected). Re-submit the same
  `oarsub` line; `run_all.sh` resumes automatically.
- **No Telegram messages** → check `TOKEN`/`CHAT_ID` in `scripts/notify.sh`, and make sure
  you ran `notify.sh` on the **frontend** (nodes have no internet).
- **`module: command not found` inside a besteffort script** → the scripts start with
  `#!/bin/bash -l` (login shell) exactly to avoid this; make sure you didn't change that
  first line.
