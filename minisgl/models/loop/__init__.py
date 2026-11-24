"""`loopres.model.LoopResModel` served from its checkpoint, whose config.json carries `LoopResConfig`'s loop keys."""

from __future__ import annotations

from .common import cache_layers  # noqa: F401  (engine sizes the KV cache through this)


def LoopResForCausalLM(config):
    from .loopres import Model

    return Model(config)
