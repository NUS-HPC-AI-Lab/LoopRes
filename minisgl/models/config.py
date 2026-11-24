from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    architectures: list[str]
    # A loopres checkpoint's `LoopResConfig` keys, `base_layers` added; `num_layers` then counts its KV-cache
    # layers.
    loop: Dict[str, Any] | None = None

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        architectures = getattr(config, "architectures", ["Qwen3ForCausalLM"])

        num_layers, loop = config.num_hidden_layers, None
        if getattr(config, "loop_layers", None):
            from .loop import cache_layers

            loop = {
                "steps": config.steps,
                "loop_layers": tuple(config.loop_layers),
                "attn_res_blocks": config.attn_res_blocks,
                "previous_step": config.previous_step,
                "base_layers": num_layers,
            }
            num_layers = cache_layers(loop)
        return cls(
            num_layers=num_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                scaling=getattr(config, "rope_scaling", None),
            ),
            architectures=architectures,
            loop=loop,
        )
