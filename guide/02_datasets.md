# 02 - Datasets

The data is **not** in git (about 870 MB, and the UCR archive has its own licence). This guide is
how to put it on a new machine.

## What the project uses

| dataset | what it is | why we use it |
|---|---|---|
| **WebTraffic** | synthetic web-traffic series with injected anomalies, from the MILLET paper | the **only** dataset with per-timestep ground truth, so it is the only one where NDCG can be measured and where an explanation figure can be *checked* |
| **UCR 2018** | 128 real time-series classification datasets | the standard benchmark; 85 of them have published MILLET numbers we compare against |

## The folder layout the code expects

```text
data/
├── UCR/
│   ├── Coffee/
│   │   ├── Coffee_TRAIN.tsv
│   │   └── Coffee_TEST.tsv
│   ├── ...                                      (128 folders)
│   └── Missing_value_and_variable_length_datasets_adjusted/
│       ├── DodgerLoopDay/                       (the 15 fixed copies - see below)
│       └── ...
└── WebTraffic/
    ├── WebTraffic_TRAIN.csv
    ├── WebTraffic_TRAIN_metadata.json
    ├── WebTraffic_TEST.csv
    └── WebTraffic_TEST_metadata.json
```

`seanet/data.py` is the **only** file that reads these paths. Nothing else in the project touches
a data file directly.

## 1. UCR 2018

Download the archive from <https://www.cs.ucr.edu/~eamonn/time_series_data_2018/> (it is password
protected; the password is on that page), unzip it, and put the per-dataset folders under
`data/UCR/`. Keep the `Missing_value_and_variable_length_datasets_adjusted/` folder that ships with
it - we need it.

**Why that adjusted folder matters.** 15 UCR datasets have missing values or series of different
lengths, so their normal `.tsv` files contain `NaN`. Normalising a series with a NaN in it makes
the whole series NaN, the loss becomes NaN, and training silently produces garbage. The archive
ships fixed copies of exactly those 15, and `seanet/data.py` reads the fixed copy for those names
(`AdjustedUCRDataset`). We do this by subclassing in our own code - the `millet/` folder is not
edited.

Check the archive is complete:

```bash
python -c "import sys; sys.path.insert(0,'.'); from seanet.data import discover_ucr_datasets; discover_ucr_datasets()"
```

It compares what is on disk against the 128-name list and complains about anything missing.

## 2. WebTraffic

It is **generated**, not downloaded. The generator is MILLET's, in
`millet/data/web_traffic_generation.py`, and `millet/notebooks/web_traffic_generation_example.ipynb`
shows how it is called. If you already have the four files, just copy them into
`data/WebTraffic/` - they are only ~19 MB and regenerating with a different seed would give you a
different dataset, which would make your numbers incomparable with the ones in `results/`.

## 3. Check everything

```bash
python main.py summary Coffee        # one dataset
python main.py summary --all         # every dataset, writes results/SEA_NET/data_summary.csv
```

`summary` also runs a sanity check per dataset (label column convention, class count, no NaN) and
writes one row per dataset to `results/SEA_NET/data_summary.csv`:

```text
dataset, split sizes, series_length, n_classes, class balance, used_adjusted_folder
```

## Copying the data to Grid5000

```bash
rsync -av --progress data/ <site>:~/SEA_NET/data/
```

Do it once. It is the slowest part of setting the cluster up, and the data never changes.

---

To add a dataset of your own, see [08 - Adding a dataset](08_adding_a_dataset.md).

Next: [03 - Running experiments](03_running_experiments.md)
