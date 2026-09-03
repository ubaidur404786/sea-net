# 15. Voting pooling, and the ambiguity threshold

This guide explains **`sea_topk_voting`** — a pooling head where the top-k timesteps **vote** for a
class instead of having their evidence averaged — and its `confidence_threshold` setting, which
stops a timestep that cannot decide between its two best classes from voting at all.

> **It is its own head.** Voting started life as an option *inside* `sea_topk_conjunctive`. That was
> a mistake: it put an experiment inside our best model, so a bug in the experiment could change the
> model we actually ship. On 2026-09-03 the two were split. `sea_topk_conjunctive` is now back to
> exactly what it always was — the plain mean of the top-k — and every voting setting lives in
> `sea_topk_voting`. The split was checked to be numerically exact: the new head reproduces the old
> voting code to a maximum absolute difference of **0.0**, so the results in §6 still stand.

| head | what it does with the top-k timesteps |
|---|---|
| `sea_topk_conjunctive` | averages their gated evidence (per class). **Our best model. Untouched.** |
| `sea_topk_voting` | lets them vote, one vote each. The experiment. |

It also answers, honestly, the question that matters most: **can this thing even be trained?**

---

## 1. What the head does today (`topk_mean`)

For every timestep `j` and every class `k`:

```text
a_j^k = sigmoid(attention(z_j))_k      per-class gate,      in (0, 1)
p_j^k = classifier(z_j)_k              per-timestep logit,  unbounded
g_j^k = a_j^k * p_j^k                  "gated evidence"
```

Then keep the `k = ceil(top_frac * T)` largest `g` values **per class** and average them:

```text
Y^k = mean of the k largest g_j^k
```

On WebTraffic, `T = 1008` and `top_frac = 0.1`, so `k = 101` timesteps per class.

---

## 2. What voting does instead

The supervisor's picture: take the 30 selected time points, look at the 3 class scores at each one,
give one vote to the winner, and count.

```text
class 1 wins at 15 points
class 2 wins at 10 points     ->   [15, 10, 5]
class 3 wins at  5 points
```

Two things had to change to make that a real algorithm.

### 2.1 The k timesteps must be shared between the classes

`topk_mean` picks a **different** top-k for every class. A vote between classes only makes sense
if all classes are judged at the **same** time points — otherwise class 1's votes come from one set
of points and class 2's from another, and the counts are not comparable.

So voting scores every timestep by its own strongest class,

```text
s_j = max_k g_j^k
```

and takes the `k` best `s_j`. One shared set of timesteps, then everyone votes.

### 2.2 The counts must be turned into logits, not probabilities

This is the part that is easy to get wrong.

The training loss is `nn.CrossEntropyLoss` (`seanet/training.py`,
`make_label_smoothing_criterion`). It applies its **own** `log_softmax` internally. So whatever we
put in `bag_logits` must be **raw logits**. Adding a softmax before it would apply softmax twice:
the gradients get flattened and the model trains badly.

Now, the vote share

```text
share^k = n^k / n_kept          (n = vote counts, n_kept = how many points voted)
```

is already a probability vector — it sums to 1. So what are the logits that produce exactly this
distribution? The logarithm, because

```text
softmax(log share) == share
```

That is what the code returns:

```python
bag_logits = softplus(self.scale) * torch.log(share)
```

| what you might write | what CrossEntropyLoss then computes | verdict |
|---|---|---|
| `softmax(counts)` | `softmax(softmax(counts))` | wrong — double softmax |
| `counts` | `softmax(counts)` | wrong scale — counts go up to k, and k depends on T, so a long series behaves differently from a short one |
| `counts / k` | `softmax(counts / k)` | better, but still an exponential distortion of the vote |
| **`log(counts / n_kept)`** | **`counts / n_kept`** | **the loss sees exactly the vote share** |

`scale` is one learned number (starts at exactly 1.0). It is there because a pure vote share can
never be more confident than "all k points agree", which would cap how low the loss can go; `scale`
lets the model sharpen or flatten that distribution as training needs. (§6 shows it did its job.)

Dividing by `n_kept` and not by a fixed `k` matters once the threshold is on — see §4.

---

## 3. Is hard voting differentiable? No. Here is the honest answer.

The seven questions, one by one.

**1. Why voting could make sense.**
`topk_mean` adds up *magnitudes*. One timestep with an enormous `g` can outvote 50 timesteps that
mildly disagree — the decision is then really made by one point. Voting makes every selected
timestep count the same, so the prediction becomes "most of the evidence points at class 1", which
is closer to what MIL claims to be doing, and much harder for one outlier to hijack. In MIL terms
it moves the head from a *max-like* aggregator towards a *majority* aggregator.

**2. What information is lost.** The magnitude. `[15, 10, 5]` cannot tell the difference between
"class 1 barely won 15 points" and "class 1 crushed 15 points". Ranking, distance from the decision
boundary and confidence per point are all discarded. That is a real loss — on a dataset where the
evidence is one enormous spike, voting should do *worse* than `topk_mean`, and that is exactly the
kind of result worth reporting.

**3. Is hard arg-max voting differentiable?** **No.** `argmax` is piecewise constant: nudge a weight
and the winner does not change, so the derivative is 0 almost everywhere, and where it does change
it jumps, so the derivative is undefined. Gradient descent gets nothing.

**4. Would it break gradient flow?** Completely. Every gradient reaching the encoder passes through
the pooling head. With a hard vote, `d loss / d g = 0`, so the encoder, the attention head and the
instance classifier would **all** get zero gradient. The model would never learn anything —
training would run, the loss would sit still, and the result would look like a bug rather than a
finding.

So the implementation uses **soft votes** by default:

```text
v_j = softmax(g_j / vote_temperature)     each point still casts exactly ONE vote,
                                          but it may split it between classes
```

The counts still sum to `n_kept`, so it is still a vote; it is just a vote that can be
differentiated. A small `vote_temperature` (0.5, 0.2, 0.1) makes it sharper — closer and closer to
"all of it to the winner" — at the cost of smaller gradients.

`hard: true` gives the strict version with a **straight-through estimator**: forward pass uses the
real 1/0 vote, backward pass borrows the soft vote's gradient. It trains, but the gradient no longer
matches the function that was computed, so it is a *biased* gradient. Use it as an experiment
(`ablations/seanet_topk_voting_hard.yaml`) and compare, do not assume it is better.

**5. Could this improve accuracy?** Possibly, on data where the class evidence is spread over many
timesteps and a few points are extreme. On WebTraffic our best models already lean on a small
number of sharp points, so this is genuinely an open question — which is why it is an ablation
config and not a change to the default.

**6. Failure cases to expect.**

* Early training: `g` is near zero everywhere, so `softmax(g)` is nearly uniform, every point votes
  ~1/C for every class, the logits are ~`log(1/C)` for all classes, and gradients are small. Warm-up
  may be slow. If it stalls, lower `temperature`.
* Ties: with `C = 3` and `k = 101`, a 34/34/33 split is a near-coin-flip prediction. Voting has a
  coarser resolution than a mean.
* Confidence ceiling: without `scale` the model could never be more sure than 100 % of votes.
  (Measured in §6: this did NOT bite - the head reached 100 % training accuracy.)
* Calibration: vote shares are not well-calibrated probabilities. AUROC should still be fine
  (it only needs the ranking), but the test *loss* may look worse than the accuracy suggests.
* AOPCR / NDCG are computed from `interpretation`, which is still `g` — unchanged — so the
  interpretability numbers stay comparable to every model already in the leaderboard.

**7. Could the threshold help or hurt?** Both are plausible. Removing points that are genuinely
undecided should sharpen the vote. But the same points may be *weak evidence rather than no
evidence*, and removing them throws that away — and it removes them from the loss, so the encoder
stops being taught anything about those regions. Sweep it and look at accuracy **and** AOPCR
together.

---

## 4. The ambiguity threshold

### What it is measured on, and why

The natural thing to write is "if `g_top1 - g_top2 < 0.002`, drop the point". The trouble is that
`g` is an **unnormalised logit**: it has no fixed scale. Early in training all the `g` values are
tiny and almost every point looks ambiguous; later they grow and almost none does. The same 0.002
would mean two completely different things at two moments of the same run, and different things
again on another dataset.

So the margin is measured on the per-timestep class **probabilities**:

```text
q_j    = softmax over classes of g_j        # sums to 1 at every timestep
margin = q_top1 - q_top2                    # always in [0, 1]
```

Now `0.002` means "the best two classes are within 0.2 percentage points of each other" — the same
statement everywhere. **Your 0.001 / 0.002 values are usable on this scale**; they were not on the
raw `g` scale. Expect them to be *small*, though: sweep upwards.

```text
0.0    off (the default - the head behaves exactly as before)
0.002  drops only near-perfect ties
0.01   noticeable
0.05   aggressive
0.10+  expect accuracy to move; check whether it moved the right way
```

### Where it is applied

It applies **before** selection - ambiguous points are pushed to the back of the queue for the
top-k - **and** again after, so one that still gets picked is silenced. A vote from an undecided
point is exactly what we want gone, and the slot is better spent on a confident point.

The threshold exists only on the voting head. It used to also apply to the top-k *mean*, but that
put an experimental knob inside our best model for no strong reason: an averaged value is already
weighted by how strong the evidence is, so a near-tied point contributes little anyway. In a vote it
contributes a **whole vote**, which is where the idea actually earns its place.

### The empty-selection problem

If the threshold removes every selected timestep of a series, a plain weighted mean would be
`0 / 0 = NaN`, the loss would become NaN, and the whole run would die. Two guards stop that:

1. `_never_empty()` — any (series, class) column whose weights all became 0 is put back to all
   ones, i.e. that series simply ignores the threshold.
2. the denominator is `w.sum().clamp_min(1.0)`, never a bare `k`.

Verified with `confidence_threshold: 0.999` (which kills essentially everything): the logits stay
finite and the loss stays finite.

---

## 5. The configs to run

All four extend `seanet/seanet_bottleneck_topk`, so the *only* difference is the pooling knob.

```bash
# the baseline these are measured against (0.942 acc / 2.951 AOPCR / 0.786 NDCG, same encoder)
python main.py webtraffic --model top/top_bottleneck_topk

# 1. soft voting
python main.py webtraffic --model ablations/seanet_topk_voting --smoke   # flow check first
python main.py webtraffic --model ablations/seanet_topk_voting

# 2. strict voting (straight-through)
python main.py webtraffic --model ablations/seanet_topk_voting_hard

# 3. voting with the ambiguity threshold
python main.py webtraffic --model ablations/seanet_topk_voting_thresh
```

Then compare:

```bash
python main.py leaderboard
```

Watch **three** numbers, not one: `web_acc`, `web_aopcr` and `web_ndcg`. A voting head that gains
0.3 % accuracy and loses 0.5 AOPCR is not an improvement for this project.

---

## 6. THE RESULTS (first run, 2026-09-03, WebTraffic, seed 0)

All five trained on Grid5000. Same encoder (`sea_mstcn_sep_bottleneck`, 41 K params), same recipe -
only the pooling knob differs, so the comparison is clean.

| model | pooling | acc | AOPCR | NDCG@n |
|---|---|---|---|---|
| `top_bottleneck_topk` | topk_mean (baseline) | **0.942** | **2.951** | **0.786** |
| `seanet_topk_thresh` | topk_mean + threshold (head since removed, see note) | 0.930 | 2.604 | 0.744 |
| `seanet_topk_voting_thresh` | voting + threshold | 0.886 | 2.195 | 0.604 |
| `seanet_topk_voting` | voting (soft) | 0.870 | 2.212 | 0.601 |
| `seanet_topk_voting_hard` | voting (strict, straight-through) | 0.672 | 1.770 | 0.697 |
| MILLET (paper, published) | — | 0.924 | — | 0.674 |

**Voting lost, on every metric.** That is a real result, not a bug - see the diagnosis below.

### Why - read the training curves, not just the scores

WebTraffic has **10 classes** and we train with `label_smoothing: 0.13`. That puts a floor under the
training loss: the lowest value `nn.CrossEntropyLoss` can ever return is the entropy of the smoothed
target, **0.618**. Measuring each run against that floor separates "could not learn" from "learned
fine but does not generalise":

| model | best train_acc | min train_loss | above the 0.618 floor | val_acc |
|---|---|---|---|---|
| baseline | 1.000 | 0.638 | +0.020 | 0.910 |
| threshold | 1.000 | 0.639 | +0.021 | 0.880 |
| voting (soft) | **1.000** | 0.656 | **+0.038** | 0.840 |
| voting + threshold | 1.000 | 0.643 | +0.025 | 0.870 |
| voting (hard) | **0.757** | 1.128 | **+0.510** | 0.640 |

Two different failures, and it matters which is which:

**Soft voting fits the training set perfectly** (train_acc 1.000, loss within 0.04 of the theoretical
floor). So the "confidence ceiling" this guide warned about did NOT bite - the learned `vote_scale`
did its job. The head is trainable; it simply **generalises worse** (val_acc 0.84 vs 0.91). Discarding
the magnitude costs real information, exactly as §3.2 predicted, and on WebTraffic that information
was worth about 7 accuracy points.

**Hard voting could not even fit the training data** (train_acc 0.757, loss 0.51 above the floor).
This is the straight-through estimator's bias showing up as predicted in §3.4: the gradient the
encoder receives does not match the function the forward pass computed, so optimisation stalls well
short of a fit. Its higher NDCG (0.697) next to its terrible accuracy is not a redeeming feature -
an under-fitted model can still rank timesteps sensibly while getting the class wrong.

### The threshold

On its own it cost about 1 point of accuracy (0.942 -> 0.930) and 0.35 AOPCR. Combined with voting it
*helped* (0.870 -> 0.886), which fits the theory: voting is the method that actually suffers from
undecided points, because each one still casts a full vote. But both gaps are inside single-seed
noise (this project has seen the same config move 0.938 -> 0.888 on a re-run), so neither is
established yet.

### What this does and does not settle

- **Settled:** hard/strict voting is not viable here. Do not spend more GPU time on it.
- **Settled enough:** soft voting is trainable but worse than the top-k mean on WebTraffic.
- **NOT settled:** everything above is **one seed**. Before this goes in the paper, run seeds 1 and 2
  for the baseline and soft voting at least:
  ```bash
  for s in 1 2; do
    python main.py webtraffic --model top/top_bottleneck_topk      --seed $s
    python main.py webtraffic --model ablations/seanet_topk_voting --seed $s
  done
  ```
- **Untested:** whether voting helps on data where evidence is spread out rather than spiky. Our
  best WebTraffic models lean on a few sharp points - the regime where §3.5 predicted voting would
  lose. A UCR dataset with diffuse evidence is the fair test, if it is worth the time.

The honest one-line summary for the paper: *we tested majority voting over the top-k instances as an
alternative to averaging their gated evidence; it trains but generalises worse, and the strict
(non-differentiable) form fails to fit at all.* That is a legitimate negative result about a
reasonable idea, and it is worth one paragraph.

---

## 7. Reading the votes

The head keeps the last batch's counts, so you can look at them:

```python
net.pool.last_vote_counts        # (B, C), e.g. tensor([[19., 8., 3.], ...])
```

Each row sums to the number of timesteps that actually voted - with `hard: true` those are whole
numbers, with soft votes they are fractions that still add up to the same total.

---

## 8. The settings, in one place

```yaml
pooling:
  type: sea_topk_voting
  d_attn: 8
  dropout: 0.2
  positional_encoding: true
  top_frac: 0.1             # fraction of timesteps allowed to vote (0.1 = the best 10 %)
  temperature: 0.5          # < 1 sharpens each vote toward its winner
  hard: false               # true = a real 1/0 vote (straight-through backward). See §3 and §6.
  confidence_threshold: 0.0 # > 0 stops near-tied timesteps from voting. Measured on probabilities.
```
