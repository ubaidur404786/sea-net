"""
seanet.models - the model half of the project: encoder + MIL pooling head.

The one shape every model in this repo has:

    series (B, 1, T)
        -> ENCODER          -> timestep embeddings (B, d, T)      [models/encoders.py]
        -> MIL POOLING HEAD -> bag_logits (B, n_clz)              [models/pooling.py]
                               interpretation (B, n_clz, T)
                               attn (B, T, 1)

The two parts are INDEPENDENTLY SWITCHABLE. Each one is looked up by a string name in its own
registry, so a config can pair any encoder with any pooling head:

    encoder: {type: sea_mstcn_sep_bottleneck}   +   pooling: {type: sea_topk_conjunctive}
    encoder: {type: mil_inceptiontime}          +   pooling: {type: sea_topk_conjunctive}
    encoder: {type: sea_mstcn_sep_bottleneck}   +   pooling: {type: mil_conjunctive}

Nothing outside this package needs to change when you add a new encoder or a new pooling head -
see guide/06_adding_an_encoder.md and guide/07_adding_a_pooling_method.md.

Where is the "classification head"?
    It is INSIDE the pooling head, and that is on purpose, not an oversight. A MIL pooling head
    has to classify every timestep BEFORE it aggregates them - that per-timestep class score IS
    the interpretation map that AOPCR and NDCG measure. Pulling the classifier out into its own
    module would break the interpretability, so the honest picture is:

        encoder -> [ per-instance classifier + aggregation ] -> bag logits + interpretation
                   \_______________ the pooling head ______/

    models/pooling.py lists which linear layer plays the classifier role in each head.

The modules:
    encoders.py : every encoder + build_encoder() + the @register_encoder registry
    pooling.py  : every MIL pooling head + build_pooling() + the register_pooling registry
    build.py    : the glue (EncoderPoolNet) and build_model_from_config()

Naming rule (kept from seanetv5): "sea_" = written by us, "mil_" = MILLET's own code reused
unchanged from millet/. So an encoder called mil_inceptiontime is the baseline's, and one called
sea_mstcn_sep_bottleneck is ours.
"""
from seanet.models.build import (  # noqa: F401  (re-exported: this is the package's public API)
    EncoderPoolNet,
    build_model_from_config,
    make_sea_net,
    make_baseline,
    num_params,
    state_dict_size_mb,
)
from seanet.models.encoders import build_encoder, register_encoder, ENCODER_REGISTRY  # noqa: F401
from seanet.models.pooling import build_pooling, register_pooling, POOLING_REGISTRY  # noqa: F401
