#!/bin/bash
# env.sh - turn ON our project environment on Grid5000.
#
# We keep these lines in ONE place and the other scripts do:  source scripts/env.sh
# You can also run it by hand:  source scripts/env.sh
#
# IMPORTANT - the Grid5000 sites are set up differently, so we detect and handle both:
#   Lille  (flille) : no ~/.bashrc, conda NOT on the PATH, and ~/miniforge3 is incomplete
#                     -> we must run "module load conda" first
#   Sophia (fsophia): a full ~/miniforge3 install, conda already active as (base)
#                     -> nothing to load, just activate
# This script figures out which case it is, so the SAME script works everywhere.

# 1) make sure the "conda" command exists in this shell
# if ! command -v conda >/dev/null 2>&1; then
#   if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
#     # a full personal miniforge install (Sophia) - switch it on directly
#     source "$HOME/miniforge3/etc/profile.d/conda.sh"
#   else
#     # no personal conda to source (Lille) - use Grid5000's module system instead
#     module load conda
#   fi
# fi

# 2) turn on our project env.
# Try the short name first. Some sites' conda does not know the name (because the env was
# made by a different conda), so we fall back to the full folder path.
# ENV_NAME="seanet"
# ENV_PATH="$HOME/miniforge3/envs/seanet"
# conda activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_PATH"
module load conda
conda activate /home/urehman/miniforge3/envs/seanet
# # 3) quick proof the env is really on (so we SEE it before training starts)
# echo "[env] host  : $(hostname)"
# echo "[env] python: $(which python)"
python -c "import torch; print('[env] torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
