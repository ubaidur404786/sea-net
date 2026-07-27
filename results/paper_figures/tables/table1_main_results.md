### Main results: the 10 best models by WebTraffic accuracy. UCR-85 columns are the mean over the 85 datasets the MILLET paper reports, and W/T/L is the per-dataset win/tie/loss record against it; a dash means the model was screened on WebTraffic only. Best value per column in bold.

| Model | Origin | Acc. | AOPCR | NDCG | Loss | UCR-85 Acc. | W/T/L | Params | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|
| gated_mean_topk | ours | **0.954** | 2.29 | **0.772** | **0.255** | 0.8083 | 19/13/53 | 61,740 | 1.50 |
| conjunctive | half-ours | **0.954** | 1.50 | 0.698 | 0.262 | 0.8238 | 23/20/42 | 269,083 | 3.54 |
| gated_max_topk | ours | 0.952 | 2.27 | 0.719 | 0.308 | 0.8108 | 22/16/47 | 61,740 | 1.50 |
| spiketrend_topk | ours | 0.950 | 2.32 | 0.756 | 0.288 | -- | -- | 67,020 | 1.52 |
| classwise | ours | 0.946 | 1.72 | 0.686 | 0.292 | 0.8292 | 27/19/39 | 269,164 | 3.54 |
| recon_adaptive | ours | 0.946 | 2.60 | 0.768 | 0.296 | 0.8140 | 19/18/48 | 62,935 | 1.51 |
| inputgate_topk | ours | 0.946 | **2.80** | 0.748 | 0.261 | 0.8131 | 23/16/46 | 58,092 | 1.49 |
| gated_max | ours | 0.944 | 1.89 | 0.592 | 0.306 | 0.8237 | 20/17/48 | 61,740 | 1.50 |
| gated_last | ours | 0.942 | 2.52 | 0.742 | 0.327 | 0.7888 | 17/12/56 | 61,740 | 1.50 |
| SEA-Net | half-ours | 0.942 | 1.58 | 0.729 | 0.260 | **0.8298** | 24/19/42 | 269,083 | 3.54 |
