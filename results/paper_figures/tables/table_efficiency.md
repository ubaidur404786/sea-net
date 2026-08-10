### Computational cost of the 15 most accurate models. FLOPs, latency and peak memory are measured on a fixed dummy input so that architectures are compared on equal footing; see the profiling script for the exact shape. Best value per column in bold.

| Model | Acc. | Params | Size (MB) | FLOPs (M) | Latency (ms) | Series/s | Mem (MB) |
|---|---|---|---|---|---|---|---|
| sea_gated-me…ea_topk_conj | **0.955** | 61,740 | 1.50 | 109.7 | 0.128 | 7821.2 | 64.9 |
| sea_mstcn__mil_conj | 0.954 | 269,083 | 3.54 | 523.6 | 0.358 | 2790.0 | 123.7 |
| sea_gated-ma…ea_topk_conj | 0.952 | 61,740 | 1.50 | 109.7 | 0.128 | 7820.5 | 64.9 |
| sea_bottlene…ea_topk_conj | 0.950 | 41,324 | 1.43 | **76.7** | 0.130 | 7683.7 | 179.0 |
| sea_spiketre…ea_topk_conj | 0.950 | 67,020 | 1.52 | 120.1 | 0.132 | 7564.1 | 64.9 |
| sea_channels…ea_topk_conj | 0.948 | 69,768 | 1.54 | 134.1 | 0.138 | 7244.5 | 66.6 |
| sea_channels…ea_topk_conj | 0.948 | 67,592 | 1.53 | 121.5 | 0.135 | 7408.4 | 66.6 |
| sea_bottlene…ea_topk_conj | 0.947 | 41,324 | 1.43 | **76.7** | 0.131 | 7664.5 | 179.0 |
| sea_mstcn-cl…sea_cls_conj | 0.946 | 269,164 | 3.54 | 523.7 | 0.359 | 2784.6 | 123.7 |
| sea_inputgat…ea_topk_conj | 0.946 | 58,092 | 1.49 | 110.6 | 0.130 | 7702.5 | 64.9 |
| sea_recon__sea_adapt_cls | 0.946 | 62,935 | 1.51 | 120.2 | 0.129 | 7779.4 | 145.8 |
| sea_gated-ma…sea_cls_conj | 0.944 | 61,740 | 1.50 | 109.7 | 0.125 | 8011.0 | 64.9 |
| sea_gated-la…ea_topk_conj | 0.942 | 61,740 | 1.50 | 109.7 | 0.128 | 7846.2 | 64.9 |
| sea_mstcn__mil_add | 0.942 | 269,083 | 3.54 | 523.6 | 0.360 | 2776.9 | 123.7 |
| sea_gated-la…sea_cls_conj | 0.942 | 61,740 | 1.50 | 109.7 | **0.124** | 8047.0 | 64.9 |
