# 08 - Adding a new dataset

Everything in the project asks `seanet/data.py: load_dataset(name, split)` for its data. Nothing
else reads a data file. So adding a dataset is: make `load_dataset` able to return it.

## The easy case: it is already in UCR format

If your file is a tab-separated table with **no header**, where **column 0 is the class label** and
columns 1.. are the series values, you do not have to write any code at all:

```text
data/UCR/MyDataset/
├── MyDataset_TRAIN.tsv
└── MyDataset_TEST.tsv
```

Then:

```bash
python main.py summary MyDataset                                  # check it parses
python main.py single MyDataset --model seanet_bottleneck_topk --smoke
python main.py single MyDataset --model seanet_bottleneck_topk
```

`load_dataset` falls through to MILLET's `UCRDataset`, which handles the reading and the
z-normalisation. `summary` runs `assert_label_column_convention`, which catches a transposed or
corrupted file **before** you train on it - read what it prints.

### If you also want it in the 128-dataset sweep

Add its name to `UCR_128_DATASETS` in `seanet/data.py`. `sweep_order()` builds the sweep list from
that, so `python main.py train` will then include it. Leave it out and it is still trainable with
`single` or `train --only MyDataset` - which is usually what you want for a one-off dataset,
because adding it to the list changes the sweep for every model and makes old and new results
cover different sets.

---

## The harder case: a different file format

Write a small dataset class. It must subclass MILLET's `MILTSCDataset`, which is what defines a
"bag" for the whole project and does the z-normalisation.

In `seanet/data.py`:

```python
class MyDataset(MILTSCDataset):
    """
    One-line description: where the data comes from and what one series is.

    split : "train" or "test".
    """

    def __init__(self, split: str):
        super().__init__("MyDataset", split)

    def get_time_series_collection_and_targets(self, split: str):
        """
        Read the files and hand back the data. This is the ONE method a dataset must provide.

        returns : (ts_collection, targets)
                  ts_collection : a list of tensors, one per series, each shaped (T, 1)
                  targets       : a LongTensor of class indices, 0..n_clz-1, one per series
        """
        path = os.path.join("data", "MyDataset", f"MyDataset_{split.upper()}.csv")
        frame = pd.read_csv(path)
        targets = torch.as_tensor(frame.iloc[:, 0].to_numpy()).long()
        series = [torch.as_tensor(row).float().reshape(-1, 1)
                  for row in frame.iloc[:, 1:].to_numpy()]
        return series, targets
```

Then teach `load_dataset` about it:

```python
def load_dataset(name: str, split: str) -> MILTSCDataset:
    if name == WEB_TRAFFIC:
        return WebTrafficDataset(split)
    if name == "MyDataset":                  # <- the new line
        return MyDataset(split)
    if is_adjusted(name):
        return AdjustedUCRDataset(name, split)
    return UCRDataset(name, split)
```

And, if `python main.py summary MyDataset` should work, teach `read_raw_frame` how to read the raw
file too (it is used by the sanity check and the summary; the pattern for WebTraffic is right
there).

**Class labels must be 0-based consecutive integers** (0, 1, 2, ...). Labels like `-1 / 1` or
`1 / 2` break the cross-entropy loss. Remap them in
`get_time_series_collection_and_targets`, not later.

---

## Per-timestep ground truth, and why WebTraffic is special

NDCG@n asks "of the timesteps that really are important, how many did the model rank at the top?".
That needs a per-timestep label, and WebTraffic is the only dataset we have that ships one. So:

* on any other dataset, `test_ndcg` is empty in `results.csv`. That is correct, not a bug.
* AOPCR still works everywhere - it needs no labels, only the model's own predictions.
* the explanation figures (`python main.py interpret`) still draw everywhere, but they cannot
  shade the *true* important region, so you cannot **check** the highlight. The command prints a
  warning saying exactly that when you point it at a UCR dataset.

If your dataset does have per-timestep labels, expose them the way `WebTrafficDataset` does
(`get_instance_targets`) and NDCG will start working for it automatically - `metrics.py` calls
MILLET's `evaluate_interpretability`, which looks for them.

---

## Check the whole thing

```bash
python main.py summary MyDataset                                    # parses, sane shapes
python main.py single MyDataset --model seanet_bottleneck_topk --smoke   # trains end to end
python main.py single MyDataset --model baselines/millet --smoke         # and with the baseline
```

Same pipeline, same commands, no training code duplicated - which is the point.

---

Next: [09 - MLflow](09_mlflow.md)
