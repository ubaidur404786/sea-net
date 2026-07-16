# GRID_CMD — Grid5000 helper

Everything needed to set this project up on a Grid5000 site from scratch, submit a job, run the
sweeps, and move code + data to another site (e.g. Sophia had no free GPU, so we're moving to
Lille). See `SESSION_HISTORY.md` for *why* the code looks the way it does right now, and
`MLFLOW_GUIDE.md` for pulling `mlflow.db`/`mlartifacts` back to your laptop.

---

## 1. First-time setup on a new site (e.g. Lille)

```bash
ssh <user>@lille.grid5000.fr        # or from inside another site: ssh lille.g5k

git clone https://github.com/ubaidur404786/sea-net.git
cd sea-net
git checkout seanetv1
```

Create the conda env (miniforge already assumed installed under `~/miniforge3`; if not, install
it first the same way it was set up on Sophia):

```bash
module load conda
conda create -y -p ~/miniforge3/envs/seanet python=3.10
conda activate /home/$USER/miniforge3/envs/seanet

# torch is a CUDA build - install it explicitly BEFORE the rest, or pip will pull a CPU wheel
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_versioned.txt   # exact versions verified working on Sophia
```

Then get the data (see §4) before running anything real — `data/` is gitignored and does not
come with `git clone`.

## 2. Activate env (every new shell)

Option 1:
```bash
module load conda
conda activate /home/urehman/miniforge3/envs/seanet
```

Option 2:
```bash
source /home/urehman/miniforge3/etc/profile.d/conda.sh
conda activate seanet
```

## 3. Submit a job (`oarsub`)

```bash
oarsub -I -t besteffort -p "host='chifflot-1'" -l host=1,walltime=12:00:00
# or a different node:
oarsub -I -t besteffort -p "host='esterel32-1'" -l host=1,walltime=12:00:00
```

See jobs:
```bash
oarstat -u
```

Delete a job:
```bash
oardel <job_id>
```

`besteffort` jobs can be killed at any time by higher-priority jobs — that's fine here, every
training command is resumable (see `SESSION_HISTORY.md` §3): re-running the same
`python main.py train --model X` command skips whatever already finished for that exact
model+settings fingerprint.

## 4. Moving the dataset between sites (Sophia → Lille)

`data/` (~874 MB) is gitignored on purpose (license terms, too large for git) — it has to be
copied by hand, once, per site. From a shell **on Sophia** (site frontends can reach each other
directly over the Grid5000 internal network, no need to go through `access.grid5000.fr`):

```bash
rsync -avzP ~/projects/sea-net/data/ <user>@lille.grid5000.fr:~/projects/sea-net/data/
```

(`-P` shows progress and lets you resume if it's interrupted — useful for an 874 MB transfer.)
If `projects/sea-net` doesn't exist yet on Lille, do §1 there first so the destination directory
exists, then run the `rsync`.

Sanity-check after transfer:
```bash
ssh <user>@lille.grid5000.fr 'du -sh ~/projects/sea-net/data && find ~/projects/sea-net/data/UCR -maxdepth 1 -type d | wc -l'
# expect: ~874M total, 129 (128 UCR dirs + the adjusted/ folder)
```

## 5. Push code from Sophia → pull on Lille

On Sophia, once your changes are committed:
```bash
git push origin seanetv1
```

On Lille:
```bash
cd ~/projects/sea-net
git pull origin seanetv1
```

Do this *before* the data rsync in §4 if the destination repo doesn't exist yet — you need the
`sea-net/` directory (and its `.gitignore`) in place first.

## 6. Run the sweeps (on whichever site has the free GPU)

```bash
python main.py train --model seanet             # 129 datasets, ~11.6 h, resumable
python main.py train --model seanet_classwise   # 129 datasets, ~11.6 h, resumable
```

Compare against MILLET afterwards:
```bash
python main.py results --model seanet
python main.py results --model seanet_classwise
python main.py results             # best-per-dataset -> results/SEA_NET/best_results.csv
python main.py report              # figures + README auto-update
```

If a `besteffort` job gets killed partway through, just re-submit (§3) and re-run the **same**
`train` command — it prints a resume banner (`resuming: N done, M to go`) and continues instead
of restarting.

## 7. Optuna

```bash
sed -i 's/enabled: false/enabled: true/' configs/models/seanet.yaml
python main.py optuna --model seanet --dataset WebTraffic
```

## 8. Getting results back off Grid5000

See `MLFLOW_GUIDE.md` for pulling `mlflow.db` / `mlartifacts` / `results/` back to your laptop
with `scp`/`rsync` — same pattern as §4, just aimed at `access.grid5000.fr` instead of a
site-to-site hop.
