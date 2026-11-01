"""TE delayed weight-gradient (``--delay-wgrad-compute``) helpers for Qwen3.8-Next modules."""

import copy


def without_delayed_wgrad(config):
    """``config`` with ``delay_wgrad_compute`` off, for TE linears that no ``backward_dw`` reaches.

    With ``--delay-wgrad-compute`` TE defers a linear's weight gradient until its ``backward_dw()`` is called. The
    EP-overlap schedule calls it only for the layer's ``self_attention``, shared experts and routed experts; the
    PLE projections live in the attention-site hyper-connection, so their deferred gradient would never be
    computed and their weights would silently stop learning. Computing it in the backward (TE's default) costs
    only the overlap of two small GEMMs. A shallow copy: every other field, and the object every other module
    holds, stay the same.
    """
    if not getattr(config, "delay_wgrad_compute", False):  # config-access-exempt: absent from test configs
        return config
    config = copy.copy(config)
    config.delay_wgrad_compute = False
    return config
