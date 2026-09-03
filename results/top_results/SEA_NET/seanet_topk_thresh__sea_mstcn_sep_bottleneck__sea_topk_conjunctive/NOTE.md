# Legacy run - not reproducible from the current code

These numbers (WebTraffic seed 0: 0.930 acc / 2.604 AOPCR / 0.744 NDCG) came from a short-
lived version of `sea_topk_conjunctive` that had a `confidence_threshold` built into it: the
top-k timesteps whose best two classes were nearly tied were dropped from the MEAN.

On 2026-09-03 that head was put back to its original form - plain top-k mean, no threshold -
and the threshold idea moved to the separate voting head (`sea_topk_voting`), where an
undecided timestep casting a full vote is the thing that actually causes harm.

So:

- The `pooling` column here says `sea_topk_conjunctive`, but TODAY'S `sea_topk_conjunctive`
  does NOT do this. Do not read this row as a score for the plain top-k head.
- `configs/models/ablations/seanet_topk_thresh.yaml` was removed; re-running is not possible
  without re-adding the threshold to the head.
- The plain top-k head's real score on the same encoder and seed is 0.942 / 2.951 / 0.786
  (`top_bottleneck_topk`).

Kept because we do not delete results.
