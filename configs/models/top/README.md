# `configs/models/top/` — the short list

This folder holds only the models we decided to **keep and keep using**. Everything else stays
where it is: `configs/models/seanet/` is still the full 61-combination sweep, `baselines/` is still
MILLET and friends, `ablations/` is still the one-knob-at-a-time studies. Nothing was moved or
deleted — this folder is a *short list*, not a new home.

## Nothing is copied

Each file here is 5–10 lines. It says which config it starts from and then only what differs:

```yaml
extends: seanet/seanet_bottleneck_topk   # the parent, spelled as you would pass it to --model
name: top_bottleneck_topk                # everything below this line overrides the parent
```

The loader (`seanet/config.read_model_file`) deep-merges the parent and the child, so a top config
can change one nested value — say only `pooling.pooling_method` — and inherit all the rest. If the
parent is ever re-tuned, every top model that extends it follows automatically. A chain
(A extends B extends C) works; a loop is refused with a clear message.

The parent's `records:` block is **not** inherited: those numbers came from the parent's own runs.
A top model records its own results the first time you train it.

## Why these five

Chosen from `results/old_results/SEA_NET/leaderboard.csv` (72 trained models) so that the short
list covers all three things we care about, not just accuracy:

| file | parent | why it is here | WebTraffic acc | AOPCR | NDCG@n | params |
|---|---|---|---|---|---|---|
| `top_gated_mean_topk.yaml` | `seanet/seanet_gated_mean_topk` | best accuracy of all 72 | 0.9547 | 2.225 | 0.750 | 61,740 |
| `top_bottleneck_topk.yaml` | `seanet/seanet_bottleneck_topk` | **the ESP32 candidate** — smallest model, and the one we ran over 3 seeds | 0.938 | 2.778 | 0.777 | 41,324 |
| `top_topk_nofocus.yaml` | `ablations/seanet_topk_nofocus` | same tiny encoder, best NDCG@n we have | 0.950 | 2.303 | 0.765 | 41,324 |
| `top_inputgate_topk.yaml` | `seanet/seanet_inputgate_topk` | best AOPCR in the top ten | 0.946 | 2.803 | 0.748 | 58,092 |
| `top_spiketrend_topk.yaml` | `seanet/seanet_spiketrend_topk` | best all-rounder of the dual-branch encoders | 0.950 | 2.316 | 0.756 | 67,020 |

All five use the same MIL head, `sea_topk_conjunctive` — that is the head this project's results
kept picking, and it is also the head the new `voting` experiments plug into.

## Adding another one later

Three lines and it is in:

```yaml
# configs/models/top/top_<short name>.yaml
extends: seanet/<the config that did well>
name: top_<short name>
```

Then run it. Two rules keep the folder tidy:

1. **File names start with `top_`.** The results folder is named after the *file*, so a `top_` name
   means its results never collide with the parent's older results.
2. **One line of "why" at the top of the file**, and a row in the table above. A short list is only
   useful while you can still see why each entry earned its place.

## Running them

```bash
python main.py models                                  # lists these next to everything else
python main.py webtraffic --model top/top_bottleneck_topk --smoke   # 3-epoch flow check
python main.py webtraffic --model top/top_bottleneck_topk           # the real run
```

Results land in `results/top_results/SEA_NET/<name>__<encoder>__<pooling>/`, and — because these
are the models we want to deploy — each real run also writes a `deploy/` bundle next to them
(weights + config + input shape + ONNX/TorchScript). See `guide/16_deployment_bundle.md`.
