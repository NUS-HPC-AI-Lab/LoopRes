"""What the looped model is built from: the decoder layer that writes one KV-cache layer per step it runs, the model
shell around its `hidden`, and the AttnRes mixture as one Triton kernel."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

import torch
import triton
import triton.language as tl
from minisgl.core import get_global_ctx
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
)

from ..base import BaseLLMModel
from ..utils import GatedMLP, RopeAttn

if TYPE_CHECKING:
    from ..config import ModelConfig


class Vectors(BaseOP):
    """A loopres `ParameterList` of vectors, under the same names."""

    def __init__(self, count: int, size: int):
        for index in range(count):
            setattr(self, str(index), torch.empty(size))

    def __getitem__(self, index: int) -> torch.Tensor:
        return getattr(self, str(index))


class LoopLayer(BaseOP):
    """A decoder layer whose attention, at the `slot`-th step it runs, writes KV-cache layer `cache_ids[slot]`."""

    def __init__(self, config: ModelConfig, cache_ids: Sequence[int]):
        self.self_attn = RopeAttn(config, cache_ids[0], has_qk_norm=True)
        self.mlp = GatedMLP(config)
        self.input_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        first = self.self_attn.attn
        self._attn = [first] + [
            AttentionLayer(
                layer_id=cache_id, num_qo_heads=config.num_qo_heads, num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim, rotary_config=config.rotary_config, q_norm=first.q_norm, k_norm=first.k_norm,
            )
            for cache_id in cache_ids[1:]
        ]

    def attention(self, x: torch.Tensor, slot: int = 0) -> torch.Tensor:
        """The attention update of `x`, the normed input."""
        return self.self_attn.o_proj.forward(self._attn[slot].forward(self.self_attn.qkv_proj.forward(x)))

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None, slot: int = 0):
        """The stock layer: `x` and `residual` as `Qwen3DecoderLayer` takes and returns them."""
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.attention(x, slot)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        return self.mlp.forward(x), residual


class LoopModel(BaseLLMModel):
    """The served model: `hidden` maps the batch's token ids to the final normed hidden state the LM head reads,
    under the training-side checkpoint names."""

    def __init__(self, config: ModelConfig, layers: List[BaseOP]):
        self.model = _Stack(config, layers)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    @abstractmethod
    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def forward(self) -> torch.Tensor:
        return self.lm_head.forward(self.hidden(get_global_ctx().batch.input_ids))

    def normed(self, x: torch.Tensor) -> torch.Tensor:
        """`x` under the final norm."""
        return self.model.norm.forward(x)[0]


class _Stack(BaseOP):
    def __init__(self, config: ModelConfig, layers: List[BaseOP]):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(layers)
        self.norm = RMSNormFused(config.hidden_size, config.rms_norm_eps)


def layers(config: ModelConfig, first: int, last: int, steps: int) -> List[BaseOP]:
    """The base stack with layers `[first, last)` run `steps` times and the others once, in `stack_ids`'s layout."""
    return [LoopLayer(config, ids) for ids in stack_ids(config.loop["base_layers"], first, last, steps)]


def run(layers: Sequence[LoopLayer], x: torch.Tensor) -> torch.Tensor:
    """The stock residual stream through `layers` from `x`. `x` is left as it is, which the fused norms, adding
    into the stream in place, would not."""
    x, residual = x.clone(), None
    for layer in layers:
        x, residual = layer.forward(x, residual)
    return x if residual is None else x + residual


def cache_layers(loop: Dict[str, Any]) -> int:
    """The KV-cache layers of `stack_ids`'s layout: the base layers plus the extra runs of the looped ones."""
    first, last = loop["loop_layers"]
    return loop["base_layers"] + (loop["steps"] - 1) * (last - first)


def stack_ids(layers: int, first: int, last: int, steps: int) -> List[List[int]]:
    """Each layer's KV-cache layers when layers `[first, last)` run `steps` times and the others once, in order of
    first run: the layers before, the looped ones step by step, the layers after."""
    ids: List[List[int]] = [[] for _ in range(layers)]
    count = 0
    for layer in [*range(first), *(index for _ in range(steps) for index in range(first, last)), *range(last, layers)]:
        ids[layer].append(count)
        count += 1
    return ids


@triton.jit
def _mix_kernel(
    out_ptr, sources, biases, key_ptr, norm_ptr, key_eps, norm_eps,
    HIDDEN: tl.constexpr, BLOCK: tl.constexpr, SOURCES: tl.constexpr, NORM: tl.constexpr,
):
    col = tl.arange(0, BLOCK)
    mask = col < HIDDEN
    row = tl.program_id(0).to(tl.int64)
    offsets = row * HIDDEN + col
    dtype = out_ptr.dtype.element_ty
    key = tl.load(key_ptr + col, mask=mask, other=0.0)
    peak = float("-inf")
    total = 0.0
    acc = tl.zeros([BLOCK], tl.float32)
    # Online softmax over the sources, each read once.
    for index in tl.static_range(SOURCES):
        source = tl.load(sources[index] + offsets, mask=mask, other=0.0).to(tl.float32)
        logit = biases[index]
        logit += tl.sum(source * key, 0) * tl.rsqrt(tl.sum(source * source, 0) / HIDDEN + key_eps)
        new_peak = tl.maximum(peak, logit)
        scale = tl.exp(peak - new_peak)
        weight = tl.exp(logit - new_peak)
        total = total * scale + weight
        acc = acc * scale + weight * source
        peak = new_peak
    out = (acc / total).to(dtype)
    if NORM:
        # As HF's RMSNorm: normalised in float32, cast, then scaled.
        x = out.to(tl.float32)
        x = (x * tl.rsqrt(tl.sum(x * x, 0) / HIDDEN + norm_eps)).to(dtype).to(tl.float32)
        out = (x * tl.load(norm_ptr + col, mask=mask, other=0.0).to(tl.float32)).to(dtype)
    tl.store(out_ptr + offsets, out, mask=mask)


def mix(
    sources: List[torch.Tensor], bias: List[float], key: torch.Tensor, key_eps: float, norm: RMSNormFused | None = None
) -> torch.Tensor:
    """loopres's AttnRes mixture of `sources`: their average under the softmax of `bias` plus each source's dot
    product with `key` over its root mean square; then under `norm` that RMS norm. One kernel, one program per
    row."""
    rows, hidden = sources[0].shape
    out = torch.empty_like(sources[0])
    _mix_kernel[(rows,)](
        out, tuple(sources), tuple(bias), key, sources[0] if norm is None else norm.weight, key_eps,
        0.0 if norm is None else norm.eps,
        HIDDEN=hidden, BLOCK=triton.next_power_of_2(hidden), SOURCES=len(sources), NORM=norm is not None,
    )
    return out
