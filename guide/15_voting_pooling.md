# 15. Voting pooling, and the ambiguity threshold

This guide explains the two new options on `sea_topk_conjunctive`:

* `pooling_method: voting` — the top-k timesteps **vote** instead of being averaged,
* `confidence_threshold` — timesteps that cannot decide between their two best classes are thrown
  out before the decision is made.

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
bag_logits = softplus(vote_scale) * log(share)
```

| what you might write | what CrossEntropyLoss then computes | verdict |
|---|---|---|
| `softmax(counts)` | `softmax(softmax(counts))` | wrong — double softmax |
| `counts` | `softmax(counts)` | wrong scale — counts go up to k, and k depends on T, so a long series behaves differently from a short one |
| `counts / k` | `softmax(counts / k)` | better, but still an exponential distortion of the vote |
| **`log(counts / n_kept)`** | **`counts / n_kept`** | **the loss sees exactly the vote share** |

`vote_scale` is one learned number (starts at exactly 1.0). It is there because a pure vote share
can never be more confident than "all k points agree", which caps how low the loss can go;
`vote_scale` lets the model sharpen or flatten that distribution as training needs.

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

`vote_hard: true` gives the strict version with a **straight-through estimator**: forward pass uses
the real 1/0 vote, backward pass borrows the soft vote's gradient. It trains, but the gradient no
longer matches the function that was computed, so it is a *biased* gradient. Use it as an
experiment (`ablations/seanet_topk_voting_hard.yaml`) and compare, do not assume it is better.

**5. Could this improve accuracy?** Possibly, on data where the class evidence is spread over many
timesteps and a few points are extreme. On WebTraffic our best models already lean on a small
number of sharp points, so this is genuinely an open question — which is why it is an ablation
config and not a change to the default.

**6. Failure cases to expect.**

* Early training: `g` is near zero everywhere, so `softmax(g)` is nearly uniform, every point votes
  ~1/C for every class, the logits are ~`log(1/C)` for all classes, and gradients are small. Warm-up
  may be slow. If it stalls, lower `vote_temperature`.
* Ties: with `C = 3` and `k = 101`, a 34/34/33 split is a near-coin-flip prediction. Voting has a
  coarser resolution than a mean.
* Confidence ceiling: without `vote_scale` the model could never be more sure than 100 % of votes.
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

| method | where | why there |
|---|---|---|
| `voting` | **before** selection (ambiguous points are pushed out of the top-k) **and** silenced if they still get picked | a vote from an undecided point is exactly what we want gone, and the slot is better used by a confident point |
| `topk_mean` | **after** selection (removed from the average) | the per-class top-k is what defines this head; changing the selection would change the model, not just filter it |

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
# the baseline these are all measured against (already trained: 0.938 acc, 2.778 AOPCR)
python main.py webtraffic --model seanet/seanet_bottleneck_topk

# 1. soft voting
python main.py webtraffic --model ablations/seanet_topk_voting --smoke   # flow check first
python main.py webtraffic --model ablations/seanet_topk_voting

# 2. strict voting (straight-through)
python main.py webtraffic --model ablations/seanet_topk_voting_hard

# 3. the threshold on its own, no voting
python main.py webtraffic --model ablations/seanet_topk_thresh

# 4. both together - only after 1 and 3, so you know which one did what
python main.py webtraffic --model ablations/seanet_topk_voting_thresh
```

Then compare:

```bash
python main.py leaderboard
```

Watch **three** numbers, not one: `web_acc`, `web_aopcr` and `web_ndcg`. A voting head that gains
0.3 % accuracy and loses 0.5 AOPCR is not an improvement for this project.

---

## 6. Reading the votes

The head keeps the last batch's counts, so you can look at them:

```python
net.pool.last_vote_counts        # (B, C), e.g. tensor([[19., 8., 3.], ...])
```

Each row sums to the number of timesteps that actually voted.
