import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from loopres.model import LoopResConfig, LoopResForCausalLM


def tiny_config(attn_implementation: str = "eager", layers: int = 2, **loop) -> Qwen3Config:
    """A small Qwen3 configuration with tied embeddings; `loop` keys make it a `LoopResConfig`."""
    cls = LoopResConfig if loop else Qwen3Config
    return cls(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=True,
        sliding_window=None,
        attn_implementation=attn_implementation,
        **loop,
    )


def tiny_lm(attn_implementation: str = "eager", seed: int = 0, layers: int = 2) -> Qwen3ForCausalLM:
    """A small random Qwen3, in float32."""
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(tiny_config(attn_implementation, layers)).eval()


def tiny_loopres(attn_implementation: str = "eager", seed: int = 0, layers: int = 6, **loop) -> LoopResForCausalLM:
    """A small random `LoopResForCausalLM`, in float32: three steps over core layers 1-5 in two blocks unless `loop`
    says otherwise."""
    loop = {"steps": 3, "loop_layers": [1, 5], "attn_res_blocks": 2, "previous_step": 1} | loop
    torch.manual_seed(seed)
    return LoopResForCausalLM(tiny_config(attn_implementation, layers, **loop)).eval()
