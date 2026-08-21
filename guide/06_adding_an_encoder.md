# 06 - Adding a new encoder

An encoder turns a batch of series into a per-timestep representation:

```text
(B, n_in, T)   ->   ENCODER   ->   (B, d_out, T)
```

`B` = batch, `n_in` = input channels (always 1 here: one instance is one timestep), `T` = length.

**The contract, in full:**

1. it is an `nn.Module` whose `forward(x)` takes `(B, n_in, T)` and returns `(B, d_out, T)`;
2. **the length `T` must come out unchanged** - the pooling head produces one importance value per
   timestep, and AOPCR/NDCG line those up against the input. An encoder that shortens the series
   silently misaligns every explanation;
3. it has an attribute `.d_out` saying how wide its output is. The pooling head reads it, so it
   works with any pooling head without either side knowing about the other.

That is the whole interface. Nothing else in the project needs to change.

---

## Three steps

### 1. Write the module

In `seanet/models/encoders.py`:

```python
class MyEncoder(nn.Module):
    """
    One-line description of the idea.

    n_in : input channels (1).
    d : how wide the representation is.
    """

    def __init__(self, n_in: int = 1, d: int = 64, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Conv1d(n_in, d, kernel_size=1)
        # padding = (kernel_size - 1) // 2 with stride 1 keeps T the same - this is the bit
        # that matters, see the contract above
        self.body = nn.Sequential(
            nn.Conv1d(d, d, kernel_size=5, padding=2),
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.d_out = d                       # REQUIRED - the pooling head reads this

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, n_in, T) -> (B, d, T)."""
        return self.body(self.stem(x))
```

### 2. Register it

Right below the class, with the `sea_` prefix (ours) or `mil_` (reused from MILLET):

```python
@register_encoder("sea_my_encoder")
def _build_my_encoder(cfg, n_in: int) -> nn.Module:
    """Build MyEncoder from its config block."""
    return MyEncoder(n_in=n_in, d=cfg.d, dropout=cfg.dropout)
```

The builder reads its numbers from `cfg` - the `encoder:` block of the model YAML - so they are
config, not code. Use `getattr(cfg, "name", default)` for anything optional, so old configs that
do not mention it keep working.

### 3. Write a config that uses it

`configs/models/seanet/seanet_my_encoder_topk.yaml`:

```yaml
name: seanet_my_encoder_topk
use_params: default

encoder:
  type: sea_my_encoder        # <- the name you registered
  d: 64
  dropout: 0.2

pooling:                      # unchanged - that is the point
  type: sea_topk_conjunctive
  d_attn: 8
  dropout: 0.2
  positional_encoding: true
  top_frac: 0.1

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

Keep the `training:` block identical to the other configs, otherwise a difference in the results
could come from the recipe rather than from your encoder.

---

## Check it before training anything

```python
import torch
from seanet.config import load_config
from seanet.models import build_model_from_config, num_params

cfg = load_config(overrides={"model": "seanet_my_encoder_topk"}).model_config
net = build_model_from_config(cfg, n_clz=3, n_in=1)
out = net(torch.randn(2, 1, 128))
print(out["bag_logits"].shape)        # (2, 3)
print(out["interpretation"].shape)    # (2, 3, 128)  <- the 128 MUST match the input length
print(num_params(net))
```

If `interpretation` is not `(B, n_clz, T)` with the original `T`, your encoder changed the length.
Fix that before going further.

Then the real check:

```bash
python main.py single Coffee --model seanet_my_encoder_topk --smoke
```

If your encoder does anything multi-scale (downsampling and upsampling again), also run the
alignment test - a shift of a few timesteps does not crash and barely moves accuracy, but it
quietly destroys NDCG:

```bash
python scripts/check_multiscale.py
```

---

## Pair it with every pooling head

The point of the split is that you get the whole row for free:

```bash
python scripts/make_cross_configs.py     # writes one config per encoder x pooling pair
python main.py models                    # check yours appeared
```

`make_cross_configs.py` never overwrites an existing file, so it is safe to re-run.

---

## The encoders that already exist

```bash
python -c "import sys; sys.path.insert(0,'.'); from seanet.models import ENCODER_REGISTRY; print(sorted(ENCODER_REGISTRY))"
```

Ours (`sea_`): plain multi-scale separable TCN, plus variants that add self-gating, an input gate,
a spike/trend branch, a bottleneck, a reconstruction branch, and two multi-scale *wrappers*
(rolling-statistic channels, and a downsampling pyramid) that wrap another encoder instead of
replacing it. Reused from MILLET (`mil_`): InceptionTime, FCN, ResNet.

---

Next: [07 - Adding a MIL pooling method](07_adding_a_pooling_method.md)
