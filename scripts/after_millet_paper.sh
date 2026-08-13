#!/usr/bin/env bash
# scripts/after_millet_paper.sh
#
# Everything to run ONCE, on the server, after the sv1/millet_paper sweep has finished.
#
# Why this file exists: the order matters. The four crashed datasets must be re-trained
# BEFORE the leaderboard is rebuilt, otherwise the leaderboard just re-reads the old
# results.csv and nothing changes. And `paper` reads the leaderboard, so it goes last.
#
# Run it from the project root, inside the seanet conda env:
#     bash scripts/after_millet_paper.sh
#
# It is safe to run twice: training skips datasets already listed in done_train_dataset.txt,
# and the rebuild steps just overwrite their own output files.

set -e   # stop at the first error instead of carrying on with half-built results

# The four datasets that crashed on the manual_pad bug (see workflow.md, "Adjustments to
# millet" #3). They are the only ones in the archive short enough to trigger it.
SHORT_DATASETS="SmoothSubspace ItalyPowerDemand Chinatown MelbournePedestrian"

echo "=============================================================="
echo "STEP 1  the four short datasets, full-fidelity MILLET recipe"
echo "        (1500 epochs, no early stopping - this is the slow one)"
echo "=============================================================="
python main.py train --model sv1/millet_paper --only $SHORT_DATASETS

echo
echo "=============================================================="
echo "STEP 2  the same four for the controlled MILLET baseline"
echo "=============================================================="
python main.py train --model sv1/millet --only $SHORT_DATASETS

echo
echo "=============================================================="
echo "STEP 3  rebuild every derived file, in this order"
echo "=============================================================="
python main.py leaderboard    # one row per model, reads every results.csv
python main.py results        # per-model result tables
python main.py report         # per-model figures
python main.py web-compare    # the WebTraffic side-by-side
python main.py paper          # every figure and LaTeX table used in the report

echo
echo "=============================================================="
echo "STEP 4  seed spread, paired test and the ensemble (Table 16)"
echo "=============================================================="
python scripts/ensemble_vote.py \
    --models sv4/seanet_bottleneck_topk sv4/seanet_inputgate_adaptive \
    --baseline sv1/millet

echo
echo "=============================================================="
echo "DONE. Check these before pulling to the laptop:"
echo "  - ucr85_n is 85 for millet_paper in results/SEA_NET/leaderboard.csv"
echo "  - the log says '0 failed'"
echo "Then:  git add results/ && git commit -m 'full millet_paper sweep' && git push"
echo "=============================================================="
