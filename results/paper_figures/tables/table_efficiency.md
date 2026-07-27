### Computational cost of the 15 most accurate models. FLOPs, latency and peak memory are measured on a fixed dummy input so that architectures are compared on equal footing; see the profiling script for the exact shape. Best value per column in bold.

| Model | Acc. | Params | Size (MB) |
|---|---|---|---|
| gated_mean_topk | **0.954** | 61,740 | 1.50 |
| conjunctive | **0.954** | 269,083 | 3.54 |
| gated_max_topk | 0.952 | 61,740 | 1.50 |
| spiketrend_topk | 0.950 | 67,020 | 1.52 |
| classwise | 0.946 | 269,164 | 3.54 |
| recon_adaptive | 0.946 | 62,935 | 1.51 |
| inputgate_topk | 0.946 | 58,092 | 1.49 |
| gated_max | 0.944 | 61,740 | 1.50 |
| gated_last | 0.942 | 61,740 | 1.50 |
| SEA-Net | 0.942 | 269,083 | 3.54 |
| gated_last_topk | 0.942 | 61,740 | 1.50 |
| bottleneck_softmax | 0.940 | 41,325 | 1.43 |
| recon_topk | 0.940 | 62,925 | 1.51 |
| gated_last_adaptive | 0.938 | 61,750 | 1.50 |
| inputgate_adaptive | 0.938 | 58,102 | 1.49 |
