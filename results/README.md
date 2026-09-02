# `results/` — what is where

Four folders, and only two of them are ours.

```text
results/
├── old_results/      ARCHIVE. Everything from the finished 72-model sweep. Never deleted.
│   ├── SEA_NET/        one folder per model + leaderboard.csv, model_comparison.csv, ...
│   └── analysis/       the cross-model figures and tables that went with them
├── top_results/      LIVE. Where every new run writes. Starts empty.
│   ├── SEA_NET/
│   └── analysis/
├── UCR/              NOT ours - MILLET's PUBLISHED numbers for the 85 UCR datasets
└── WebTraffic/       NOT ours - MILLET's PUBLISHED numbers for WebTraffic
```

## Why `UCR/` and `WebTraffic/` did not move

They are **inputs, not outputs**. They hold the numbers from the MILLET paper that every one of our
comparisons is measured against (`results/UCR/InceptionTime/test_acc.csv` and friends), and
`seanet/results.py` reads them from those exact paths. Moving them into `old_results/` would say
they are old results of ours, which they are not.

## Which folder does a command use?

`configs/main.yaml` decides:

```yaml
output:
  results_dir: results/top_results/SEA_NET
  analysis_dir: results/top_results/analysis
```

So by default everything — training, `leaderboard`, `analyse`, `report` — works on **top_results**.
To read the archive instead, pass the flag; nothing needs editing:

```bash
python main.py leaderboard --results-dir results/old_results/SEA_NET
python main.py analyse     --results-dir results/old_results/SEA_NET \
                           --analysis-dir results/old_results/analysis
```

## Inside one model folder

```text
top_results/SEA_NET/<config>__<encoder>__<pooling>/
  results.csv              one row per (dataset, seed)
  done_train_dataset.txt   the resume list
  history/                 per-epoch loss + accuracy, and their two curves
  predictions/             per-series test probabilities (for ensembling)
  figures/                 this model's comparison figures
  interpretation/          per-sample explanation figures
  deploy/                  NEW: the complete model - weights + config + ONNX (see below)
  logs/
```

`deploy/` is the one that matters for the ESP32 work: it holds the weights **and** the config that
built them, so the exact trained network can be rebuilt on any machine. Details in
`guide/16_deployment_bundle.md` and in the README inside each `deploy/` folder.

## Nothing was deleted

`old_results/` is a plain move. Every number quoted in `PROJECT_STATE.md`, the README and the paper
still exists, at the same file names, one folder deeper.
