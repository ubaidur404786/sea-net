"""
seanet - all of OUR code for the SEA-Net project.

The repository has three top-level pieces, and the split is the point:

    seanet/     our code               <- you are here
    millet/     the MILLET baseline    <- upstream code, kept unchanged so it stays diffable
    scripts/    stand-alone utilities  <- profiling, ensembling, config generation, Grid5000

The pipeline, in the order it actually runs
-------------------------------------------
    config.py         read configs/ (main + environment + model + command-line flags)
    data.py           load a dataset by name (WebTraffic or any of the 128 UCR sets)
    preprocessing.py  normalise, and split train / validation
    models/           ENCODER  ->  MIL POOLING HEAD  (each switchable on its own)
    training.py       the one training loop, with a per-epoch train/val history
    evaluation.py     score the trained model -> one results row
    metrics.py        the numbers themselves: accuracy, AOPCR, NDCG
    results.py        save the row, remember what is done, build the leaderboard
    analysis/         turn all the rows into comparison figures and tables
    tracking.py       MLflow, the single interface used by everything
    optuna_search.py  OPTIONAL hyperparameter search - it calls the same training pipeline
    interpretability.py  per-sample explanation figures
    utils.py          device, seeds, and the run log

Each module does one job, so when something breaks you know which file to open.

You do not import these to run things. Everything is run from main.py in the repo root:

    python main.py -h                                   list every command
    python main.py single Coffee --model seanet_bottleneck_topk --smoke

Documentation lives in guide/.
"""
