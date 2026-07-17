#!/bin/bash
# env.sh - turn ON our project environment on Grid5000 (Lille).
#
# We keep these lines in ONE place and the other scripts do:  source scripts/env.sh
# You can also run it by hand:  source scripts/env.sh
#
# Why we need this on the server (and not on the laptop):
# Grid5000 has NO ~/.bashrc and conda is NOT on the PATH by default. So we must load
# conda ourselves every time. "module load" is Grid5000's way of switching software on.

# 1) load Grid5000's conda (version 23.5.0)
module load conda

# 2) turn on our project env. The short name "seanet" works because we ran once:
#      conda config --append envs_dirs ~/miniforge3/envs
#    (if the name ever fails, use the full path: conda activate ~/miniforge3/envs/seanet)
conda activate seanet

# 3) quick proof the env is really on (so we SEE it before training)
echo "[env] python: $(which python)"
python -c "import torch; print('[env] torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
