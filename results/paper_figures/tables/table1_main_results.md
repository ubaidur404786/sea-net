### Main results: the 10 best models by WebTraffic accuracy. UCR-85 columns are the mean over the 85 datasets the MILLET paper reports, and W/T/L is the per-dataset win/tie/loss record against it; a dash means the model was screened on WebTraffic only. Best value per column in bold.

| Model | Origin | Acc. | AOPCR | NDCG | Loss | UCR-85 Acc. | W/T/L | Params | Size (MB) | FLOPs (M) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sea_gated-me…ea_topk_conj | ours | **0.955** | 2.23 | 0.750 | **0.254** | 0.8083 | 19/13/53 | 61,740 | 1.50 | 109.7 | 0.128 |
| sea_mstcn__mil_conj | half-ours | 0.954 | 1.50 | 0.698 | 0.262 | 0.8238 | 23/20/42 | 269,083 | 3.54 | 523.6 | 0.358 |
| sea_gated-ma…ea_topk_conj | ours | 0.952 | 2.27 | 0.719 | 0.308 | 0.8108 | 22/16/47 | 61,740 | 1.50 | 109.7 | 0.128 |
| sea_bottlene…ea_topk_conj | ours | 0.950 | 2.30 | 0.765 | 0.263 | -- | -- | 41,324 | 1.43 | 76.7 | 0.130 |
| sea_spiketre…ea_topk_conj | ours | 0.950 | 2.32 | 0.756 | 0.288 | 0.8089 | 19/23/43 | 67,020 | 1.52 | 120.1 | 0.132 |
| sea_channels…ea_topk_conj | ours | 0.948 | 2.32 | 0.757 | 0.318 | -- | -- | 69,768 | 1.54 | 134.1 | 0.138 |
| sea_channels…ea_topk_conj | ours | 0.948 | 2.22 | 0.717 | 0.289 | -- | -- | 67,592 | 1.53 | 121.5 | 0.135 |
| sea_bottlene…ea_topk_conj | ours | 0.947 | 2.62 | **0.772** | 0.282 | 0.8097 | 20/15/50 | 41,324 | 1.43 | 76.7 | 0.131 |
| sea_mstcn-cl…sea_cls_conj | ours | 0.946 | 1.72 | 0.686 | 0.292 | **0.8292** | 27/19/39 | 269,164 | 3.54 | 523.7 | 0.359 |
| sea_inputgat…ea_topk_conj | ours | 0.946 | **2.80** | 0.748 | 0.261 | 0.8131 | 23/16/46 | 58,092 | 1.49 | 110.6 | 0.130 |
