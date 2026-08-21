# 07 - Adding a new MIL pooling method

A MIL pooling head takes the encoder's per-timestep representation and produces both the
prediction **and** the explanation:

```text
(B, d_in, T)   ->   MIL POOLING HEAD   ->   {"bag_logits":     (B, n_clz),
                                             "interpretation": (B, n_clz, T),
                                             "attn":           (B, T, 1)}
```

| key | what it is | who reads it |
|---|---|---|
| `bag_logits` | one score per class for the whole series | the loss, accuracy, AUROC |
| `interpretation` | how much each timestep pushed each class | AOPCR, NDCG, the explanation figures |
| `attn` | the attention gate over time (attention-based heads only) | the entropy "focus" penalty in the loss |

`attn` is **optional**: a head without attention (global average pooling, instance pooling) simply
does not return it, and `training.py` skips the focus penalty for that head. Everything still works.

**The classifier lives inside the head, and it has to.** A MIL head must score every timestep
before it aggregates, because that per-timestep class score *is* `interpretation`. So a head is
"aggregation + classification" together, not aggregation alone.

---

## Three steps

### 1. Write the class

In `seanet/models/pooling.py`. Subclass `_AttnPoolingBase` if you want attention (it gives you the
positional encoding and the `(B, d, T) -> (B, T, d)` transpose for free):

```python
class MyPooling(_AttnPoolingBase):
    """
    One-line description of what this head changes about Conjunctive pooling.

    d_in : the encoder's output width.
    n_clz : number of classes.
    d_attn : width of the attention MLP.
    """

    def __init__(self, d_in: int, n_clz: int, d_attn: int = 8, dropout: float = 0.1,
                 apply_positional_encoding: bool = True, my_knob: float = 1.0):
        super().__init__(d_in, n_clz, dropout, apply_positional_encoding)
        self.attention = nn.Sequential(          # the gate: one weight per timestep
            nn.Linear(d_in, d_attn), nn.Tanh(), nn.Linear(d_attn, 1),
        )
        self.instance_classifier = nn.Linear(d_in, n_clz)   # THE CLASSIFIER (see above)
        self.my_knob = my_knob

    def forward(self, instance_embeddings, pos=None):
        """instance_embeddings : (B, d_in, T) -> the output dict."""
        z = self._prepare(instance_embeddings, pos)          # (B, T, d_in)
        a = torch.sigmoid(self.attention(z))                 # (B, T, 1)  the gate
        logits = self.instance_classifier(z)                 # (B, T, n_clz) per-timestep class score
        weighted = a * logits                                # (B, T, n_clz)
        bag_logits = weighted.mean(dim=1) * self.my_knob     # (B, n_clz)  aggregate over time
        return {
            "bag_logits": bag_logits,
            "interpretation": weighted.transpose(1, 2),      # (B, n_clz, T)  <- note the transpose
            "attn": a,
        }
```

Two things that are easy to get wrong:

* `interpretation` is `(B, n_clz, T)`, while the working tensor is `(B, T, n_clz)`. Transpose it.
* the aggregation over time must be **length-invariant** (mean, softmax-weighted sum,
  soft-argmax). Every dataset has a different `T`; anything that assumes a fixed length breaks on
  the next dataset.

### 2. Register it

At the bottom of the same file, next to the other `register_pooling` calls:

```python
register_pooling("sea_my_pooling", MyPooling, has_attn=True)
```

`sea_` = ours, `mil_` = MILLET's reused unchanged.

`build_pooling` passes head-specific settings only when your constructor actually accepts them -
it inspects the signature. `d_attn`, `temperature`, `init_beta`, `top_frac` and `init_lam` are
already recognised; to add `my_knob`, put it in that tuple in `build_pooling`, or name your knob
one of the existing ones.

### 3. Write a config that uses it

`configs/models/seanet/seanet_bottleneck_mypool.yaml`:

```yaml
name: seanet_bottleneck_mypool
use_params: default

encoder:                              # unchanged - that is the point
  type: sea_mstcn_sep_bottleneck
  d: 64
  n_blocks: 4
  dropout: 0.2
  max_dilation: 16
  kernels: [5, 11, 23]
  bottleneck_ratio: 4

pooling:
  type: sea_my_pooling                # <- the name you registered
  d_attn: 8
  dropout: 0.2
  positional_encoding: true

training:
  n_epochs: 400
  patience: 60
  max_batch: 16
  learning_rate: 0.00125
  weight_decay: 0.00012
  label_smoothing: 0.13
  lambda_entropy: 0.01
  min_train_for_val: 100
  val_frac: 0.2
  optimizer: adam
```

---

## Check it

```python
import torch
from seanet.config import load_config
from seanet.models import build_model_from_config

cfg = load_config(overrides={"model": "seanet_bottleneck_mypool"}).model_config
net = build_model_from_config(cfg, n_clz=3, n_in=1)
out = net(torch.randn(2, 1, 128))
print({k: tuple(v.shape) for k, v in out.items()})
# bag_logits (2, 3) | interpretation (2, 3, 128) | attn (2, 128, 1)
```

Then:

```bash
python main.py single Coffee --model seanet_bottleneck_mypool --smoke
python main.py webtraffic --model seanet_bottleneck_mypool --smoke   # the NDCG path
```

WebTraffic is the one that exercises NDCG, so run that smoke too - it is where a wrong
`interpretation` shape or a misaligned map shows up.

---

## The heads that already exist

```bash
python -c "import sys; sys.path.insert(0,'.'); from seanet.models import POOLING_REGISTRY; print(sorted(POOLING_REGISTRY))"
```

From MILLET (`mil_`): `additive`, `conjunctive`, `attention`, `instance`, `gap`.

Ours (`sea_`), each fixing one weak point of Conjunctive pooling:

| head | the weak point it fixes |
|---|---|
| `sea_classwise_conjunctive` | the gate is shared by all classes -> one gate per class |
| `sea_softmax_conjunctive` | the gate is unnormalised and divided by T -> softmax over time, with a temperature |
| `sea_adaptive_classwise` | the aggregator is a fixed mean -> a learnable blend between mean and max |
| `sea_topk_conjunctive` | the mean dilutes a short anomaly over a long series -> average only the top fraction |
| `sea_attention_max` | — the max-pooled variant |
| `sea_gated_attention` | gated attention, ported from the pathology MIL literature |
| `sea_dualstream_conjunctive` | two attention streams |

Each is a strict generalisation of Conjunctive, so a well-trained model can only match or beat it.

---

Next: [08 - Adding a dataset](08_adding_a_dataset.md)
