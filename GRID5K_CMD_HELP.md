# Grid5000 Command Help — SEA-Net (Lille + Sophia)

This is a start-to-end guide for running SEA-Net on Grid5000.
It is written so that next time you can just follow the steps without thinking hard.

Steps 0–8 are written for **Lille**. Step 9 shows what is different on **Sophia**
(that is where the full run actually finished). Step 10 is figures + reports.

Everything here is tested against **your real server environment**:

| Thing | Lille (`flille`) | Sophia (`fsophia`) |
|---|---|---|
| Load conda | `module load conda` (needed) | already on, `(base)` is active |
| Activate env | `conda activate seanet` | `conda activate seanet` |
| Env path | `~/miniforge3/envs/seanet` | `~/miniforge3/envs/seanet` |
| Python | 3.10.20 | 3.10.20 |
| torch | 2.0.1+cu118 | 2.0.1+cu118 |
| Cluster to ask for | `-p chuc` | `-p "cluster='esterel40'"` |
| Project folder | `~/projects/sea-net` | `~/projects/sea-net` |

> **Golden rule 1:** the *code* is the same on the laptop and the server, but the
> *environment* is different. On Lille there is **no `~/.bashrc`** and **no conda on
> the PATH**, so you must load conda with `module load conda`. `scripts/env.sh` handles
> both sites for you, so prefer `source scripts/env.sh` over typing it by hand.

> **Golden rule 2:** each site has its **own home folder**. Lille and Sophia do NOT share
> files. So on a new site you must `git clone`/`git pull` again, `mkdir -p logs` again, and
> recreate `scripts/telegram_secrets.sh` again (it is git-ignored, so git never copies it).

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
conda activate /home/urehman/miniforge3/envs/seanet
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

## Step 9 — Running on the SOPHIA site (2 nodes at once) ✅ this is what worked

When Lille had no free nodes we moved to Sophia. Three things are different there.

### 9a. What is different on Sophia

| | Lille | Sophia |
|---|---|---|
| conda | `module load conda` first | already active as `(base)` |
| Ask for a cluster | `-p chuc` (short name works) | `-p "cluster='esterel40'"` (SQL form **required**) |
| Home folder | Lille's own home | a **different** home — nothing carries over |

If you type `-p esterel` on Sophia you get `Bad resource request (column esterel does not
exist)`. Sophia's OAR wants the full SQL form with quotes.

### 9b. First time on a new site (do this once)

```bash
cd ~/projects/sea-net
git checkout -b seanetv2 origin/seanetv2   # only if you are on the wrong branch
git config core.fileMode false             # stop chmod +x from looking like a code change
git pull
chmod +x scripts/*.sh                      # or the job dies with "Permission denied"
mkdir -p logs                              # or OAR cannot create the -E error file -> job state F
```

Recreate the phone secrets (git-ignored, so it is NOT in the repo):

```bash
cat > scripts/telegram_secrets.sh <<'EOF'
TOKEN="your-bot-token-here"
CHAT_ID="your-chat-id-here"
EOF
chmod 600 scripts/telegram_secrets.sh
```

### 9c. Check a node really has a GPU

```bash
oarnodes --sql "gpu_count > 0" | grep network_address | sort -u
```

Whatever node names appear in that list have GPUs. `esterel40` and `esterel43` were in it.

### 9d. Split the 8 models over 2 nodes (the actual working commands)

`run_all.sh` accepts a list of models. Give each node a **different half** — that is safe
because every model writes into its own folder `results/SEA_NET/<model>/`, so two nodes
never touch the same file.

```bash
cd ~/projects/sea-net
mkdir -p logs

# Job A - first half, on esterel40
oarsub -t besteffort -q besteffort -p "cluster='esterel40'" -l host=1,walltime=12:00:00 \
  -E logs/jobA.err \
  "$HOME/projects/sea-net/scripts/run_all.sh seanet seanet_acp seanet_classwise seanet_softmax"

# Job B - second half, on esterel43
oarsub -t besteffort -q besteffort -p "cluster='esterel43'" -l host=1,walltime=12:00:00 \
  -E logs/jobB.err \
  "$HOME/projects/sea-net/scripts/run_all.sh seanet_conjunctive millet fcn resnet"
```

Each job writes its own log, named after its **first** model:
`logs/run_all_seanet.log` and `logs/run_all_seanet_conjunctive.log`.

### 9e. Verify (wait ~30 seconds after submitting)

```bash
oarstat -u                                     # both jobs should be R (running), not F
head -5 logs/run_all_seanet.log                # look for: cuda available: True
head -5 logs/run_all_seanet_conjunctive.log
cat logs/jobA.err logs/jobB.err                # should exist and be empty
```

### 9f. Phone tracking for both jobs

```bash
nohup env NOTIFY_EVERY=1 bash scripts/notify.sh logs/run_all_seanet.log > logs/notifyA.out 2>&1 &
nohup env NOTIFY_EVERY=1 bash scripts/notify.sh logs/run_all_seanet_conjunctive.log > logs/notifyB.out 2>&1 &
```

`nohup ... &` = keep running after you log out, so you can close the laptop.

> ⚠️ **One warning about running 2 jobs at once:** both write to the same `mlflow.db`
> (SQLite). SQLite allows only one writer at a time, so you may see a "database is locked"
> message. It does not lose your results (the CSVs are the real source of truth) but if it
> gets noisy, run the two jobs one after the other instead.

---

## Step 10 — Figures, tables and reports (after training finishes)

Training only saves **numbers** (`results.csv` per model). This step turns the numbers into
**tables and PNG figures**. It is fast and needs **no GPU** — run it on the **frontend**.

### 10a. Turn the environment on (frontend)

```bash
cd ~/projects/sea-net
source scripts/env.sh
```

`cuda available: False` here is normal — the frontend has no GPU, and drawing figures
does not need one.

### 10b. Check everything really finished

```bash
tail -3 logs/run_all_seanet.log                # should end with "=== ALL MODELS DONE ... ==="
tail -3 logs/run_all_seanet_conjunctive.log
ls results/SEA_NET/                            # one folder per model
wc -l results/SEA_NET/*/results.csv            # rows = datasets finished (+1 header line)
```

Each model should have roughly the same number of rows. If one is much smaller, that model
did not finish — resubmit its job before making figures.

### 10c. Build the dataset overview table (needed for one figure)

```bash
python main.py summary --all
```

This writes `results/SEA_NET/data_summary.csv` (size, length and number of classes of every
dataset). `report` uses it to draw `data_summary.png`. If you skip this, everything else
still works — you just don't get that one figure.

### 10d. Build the comparison tables

```bash
python main.py results
```

What it does, per model: reads that model's `results.csv`, lines it up against MILLET's
published numbers on the **85 datasets MILLET published** (the only fair one-to-one
comparison), and writes:

```
results/SEA_NET/<model>/comparison_vs_millet.csv   # one row per dataset: us vs MILLET
results/SEA_NET/<model>/summary.csv + summary.md   # that model's means + win/tie/loss
results/SEA_NET/model_comparison.csv               # THE ranking table: which model wins
```

`model_comparison.csv` is the one to look at first — it answers "did our new pooling heads
beat MILLET Conjunctive?".

For just one model:

```bash
python main.py results --model seanet_acp
```

### 10e. Draw every figure

```bash
python main.py report
```

This runs `results` again internally (so the tables and figures can never disagree) and then
draws the PNGs. It uses matplotlib's `Agg` backend, which draws straight to a file instead of
opening a window — that is why it works fine over SSH with no screen.

Per model, into `results/SEA_NET/<model>/figures/`:

| File | What it shows |
|---|---|
| `results.png` | the model alone: accuracy / loss / AOPCR spread, accuracy vs series length |
| `acc_scatter.png` | our accuracy vs MILLET's, one dot per dataset — **above** the line = we win |
| `loss_scatter.png` | our loss vs MILLET's — **below** the line = we win (lower loss is better) |
| `aopcr_scatter.png` | our AOPCR vs MILLET's — **above** the line = we win |
| `win_tie_loss.png` | win / tie / loss bars for accuracy, loss and AOPCR |
| `means.png` | our mean vs MILLET's mean, all three metrics side by side |
| `acc_diff.png` | per-dataset accuracy gap, sorted (green = we win) |

Once, into `results/SEA_NET/figures/`:

| File | What it shows |
|---|---|
| `model_comparison.png` | **every** model's mean accuracy / loss / AOPCR next to MILLET's |
| `data_summary.png` | overview of the datasets (from step 10c) |

At the end it prints the ranking and the full list of files it wrote.

### 10f. Per-sample explanation figures (optional)

The figures above compare *scores*. This one shows *why* the model made one prediction —
which time steps it looked at:

```bash
python main.py interpret --model seanet
```

### 10g. Look at everything

```bash
cat results/SEA_NET/model_comparison.csv        # the ranking
cat results/SEA_NET/seanet_acp/summary.md       # one model's summary, nicely formatted
find results/SEA_NET -name "*.png" | sort       # every figure that was drawn
```

### 10h. Copy the figures to your laptop

PNGs cannot be viewed over plain SSH, so pull them down. Run this **on your laptop**, not on
the server (`fsophia` = Sophia; use `flille` for Lille):

```bash
# everything: tables + figures
scp -r urehman@access.grid5000.fr:fsophia/projects/sea-net/results/SEA_NET ./results_from_grid5000

# or just the one summary figure
scp urehman@access.grid5000.fr:fsophia/projects/sea-net/results/SEA_NET/figures/model_comparison.png ./
```

And the MLflow database, to browse every run in a web page:

```bash
scp urehman@access.grid5000.fr:fsophia/projects/sea-net/mlflow.db ./
mlflow ui --backend-store-uri sqlite:///mlflow.db     # open http://127.0.0.1:5000
```

### 10i. Save the results into git

Run on the **server**, so the numbers are backed up and your laptop can pull them:

```bash
cd ~/projects/sea-net
git add results/SEA_NET
git commit -m "results: full sweep for all 8 models (Sophia, 2 nodes)"
git push
```

---

## Quick cheat-sheet (copy-paste order)

```bash
# 1) connect, then on the frontend:
cd ~/projects/sea-net
tmux new -s train

# tmux attach -t train (Attach to a session in same previous one)
#tmux kill-session -t train




# 2) get a GPU node (interactive, 4h)
oarsub -I -l gpu=1,walltime=6:00:00

# 3) turn env on
module load conda
conda activate /home/urehman/miniforge3/envs/seanet  



# 4) on the node: test first, then the real run
cd ~/projects/sea-net
bash scripts/test_run.sh
bash scripts/run_all.sh


# --- OR, hands-off full run (from the frontend, no node needed first) ---
oarsub -t besteffort -l gpu=1,walltime=24:00:00 "$HOME/projects/sea-net/scripts/run_all.sh"

# 5) phone tracking (frontend, in its own tmux)
bash scripts/notify.sh logs/train_seanet_<stamp>.log

# --- SOPHIA, 2 nodes, hands-off (what actually worked) ---
cd ~/projects/sea-net && mkdir -p logs && chmod +x scripts/*.sh
oarsub -t besteffort -q besteffort -p "cluster='esterel40'" -l host=1,walltime=12:00:00 \
  -E logs/jobA.err "$HOME/projects/sea-net/scripts/run_all.sh seanet seanet_acp seanet_classwise seanet_softmax"
oarsub -t besteffort -q besteffort -p "cluster='esterel43'" -l host=1,walltime=12:00:00 \
  -E logs/jobB.err "$HOME/projects/sea-net/scripts/run_all.sh seanet_conjunctive millet fcn resnet"
oarstat -u

# 6) AFTER training: tables + figures (frontend, no GPU needed)
source scripts/env.sh
python main.py summary --all      # dataset overview table
python main.py results            # comparison tables + model_comparison.csv
python main.py report             # every figure (per-model + cross-model + tiers + winner)
python main.py web-compare        # WebTraffic table + accuracy-TIER figures (>=95%..>=90%) + winner
```

NOTE on parallel nodes: every model writes to its OWN folder (results/SEA_NET/<model>/), so two
nodes never clash. The comparison tables/figures are REBUILT (overwritten) from whatever model
folders exist on the machine you run them on - so first gather all results/SEA_NET/<model>/ folders
onto one place, then run results/report/web-compare there.

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
- **`Permission denied` in `logs/jobA.err`, job goes to state F instantly** → the scripts
  lost their execute bit. Fix: `chmod +x scripts/*.sh` (and `git config core.fileMode false`
  so git stops treating that as a change).
- **Job goes to state F instantly and `logs/jobA.err` does not even exist** → the `logs/`
  folder is missing, so OAR could not create the `-E` file. Fix: `mkdir -p logs`.
- **`Bad resource request (column esterel does not exist)`** → you are on Sophia. Use the
  SQL form: `-p "cluster='esterel40'"`, not `-p esterel`.
- **`fatal: Not possible to fast-forward`** → you are on the wrong branch on that site.
  Fix: `git checkout -b seanetv2 origin/seanetv2`.
- **`Your local changes to scripts/env.sh would be overwritten`** → `chmod +x` changed the
  file-mode bits. Fix: `git config core.fileMode false` then `git checkout -- scripts/env.sh`.
- **git asks for the token on every pull** → run once:
  `git config --global credential.helper store` and `git config --global pull.rebase false`,
  then pull once and type the token; after that it is remembered.
- **`python main.py report` says "No model has any results yet"** → you are in the wrong
  folder, or training wrote to a different site's home. Check `ls results/SEA_NET/`.
- **`database is locked`** → two jobs writing `mlflow.db` at the same time. Harmless for the
  CSV results; run the jobs one after the other if you want it clean.
- **`Your local changes ... would be overwritten by merge`** → you edited a file on the laptop
  AND the server changed the same file. Commit yours first, then pull:
  `git add <file>` + `git commit -m "..."`, then `git pull origin seanetv2`.
- **`CONFLICT (content): Merge conflict in <file>`** → both sides changed the same lines.
  Open the file and look for the three conflict marker lines git inserted (a line of `<`, a
  line of `=`, and a line of `>`). Keep the text you want, delete all three marker lines,
  then `git add <file>` and `git commit`.
