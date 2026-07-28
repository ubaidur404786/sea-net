### Computational cost of the 15 most accurate models. FLOPs, latency and peak memory are measured on a fixed dummy input so that architectures are compared on equal footing; see the profiling script for the exact shape. Best value per column in bold.

| Model | Acc. | Params | Size (MB) | FLOPs (M) | Latency (ms) | Series/s | Mem (MB) |
|---|---|---|---|---|---|---|---|
| sea_gated-me…ea_topk_conj | **0.954** | 61,740 | 1.50 | 54.1 | 2.109 | 474.2 | 37.5 |
| sea_mstcn__mil_conj | **0.954** | 269,083 | 3.54 | 259.1 | 1.139 | 877.9 | 66.4 |
| sea_gated-ma…ea_topk_conj | 0.952 | 61,740 | 1.50 | 54.1 | 2.572 | 388.8 | 37.5 |
| sea_spiketre…ea_topk_conj | 0.950 | 67,020 | 1.52 | 59.2 | 1.087 | 919.5 | 37.6 |
| sea_mstcn-cl…sea_cls_conj | 0.946 | 269,164 | 3.54 | 259.1 | 0.759 | 1317.1 | 66.4 |
| sea_recon__sea_adapt_cls | 0.946 | 62,935 | 1.51 | 59.3 | **0.383** | 2610.1 | 37.6 |
| sea_inputgat…ea_topk_conj | 0.946 | 58,092 | 1.49 | 54.5 | 2.042 | 489.8 | 37.5 |
| sea_gated-ma…sea_cls_conj | 0.944 | 61,740 | 1.50 | 54.1 | 1.516 | 659.6 | 37.5 |
| sea_gated-la…sea_cls_conj | 0.942 | 61,740 | 1.50 | 54.1 | 1.002 | 998.4 | 37.5 |
| sea_mstcn__mil_add | 0.942 | 269,083 | 3.54 | 259.1 | 0.712 | 1403.8 | 66.4 |
| sea_gated-la…ea_topk_conj | 0.942 | 61,740 | 1.50 | 54.1 | 1.167 | 857.0 | 37.5 |
| sea_bottlene…_sea_sm_conj | 0.940 | 41,325 | 1.43 | **37.7** | 0.906 | 1103.1 | 41.4 |
| sea_recon__sea_topk_conj | 0.940 | 62,925 | 1.51 | 59.3 | 0.737 | 1357.2 | 37.6 |
| sea_inputgat…ea_adapt_cls | 0.938 | 58,102 | 1.49 | 54.5 | 2.594 | 385.5 | 37.5 |
| sea_bottlene…ea_topk_conj | 0.938 | 41,324 | 1.43 | **37.7** | 0.631 | 1585.9 | 41.4 |
