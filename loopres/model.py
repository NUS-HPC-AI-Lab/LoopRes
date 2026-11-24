"""A Qwen3 causal LM under the LoopRes loop.

LoopRes splits the decoder stack into three parts: a prelude (the layers before `loop_layers`), a core (the layers
inside `loop_layers`) and a coda (the layers after). The prelude and the coda run once per token; the core runs
`steps` times, spending more depth per token without more weights.

The core is cut into `attn_res_blocks` blocks of consecutive layers. Inside the core, a sub-layer does not read the
plain residual stream. It reads a learned softmax average over a set of sources: the prelude output, the block
outputs of the last `previous_step` loop steps, the block outputs finished so far in the current step, and the
update sum -- the attention and MLP updates accumulated inside the current block. Every source carries a trained
static key vector; the key's dot with the sub-layer's query, plus a second query's dot with the source's RMS norm,
is the source's softmax logit (`loopres.kernels.static_read`). Queries start at zero, so an untrained read weighs
every source alike. After the last step's last block, a readout read of the same sources gives the loop's output,
which the coda takes.

`LoopResForCausalLM` is `Qwen3ForCausalLM` over this stack, registered with `AutoModelForCausalLM`, so a checkpoint
loads, generates and saves as any transformers model; decoding caches each (layer, loop step) pass's keys and
values as its own KV-cache layer. `build` constructs the model from a run config's `model` section: `name` and
`attn_implementation`, plus for LoopRes `LoopResConfig`'s keys; without them it returns the stock pretrained model,
the SFT baseline.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
)

from loopres.kernels import static_read


class LoopResConfig(Qwen3Config):
    """`Qwen3Config` plus the loop: the core layers `[first, last)` of `loop_layers` run `steps` times in
    `attn_res_blocks` blocks of consecutive layers, every read reaching the blocks of the last `previous_step`
    steps."""

    model_type = "loopres"

    def __init__(
        self,
        steps: int = 3,
        loop_layers: tuple[int, int] = (7, 21),
        attn_res_blocks: int = 2,
        previous_step: int = 1,
        **kwargs: Any,
    ) -> None:
        first, last = loop_layers
        if attn_res_blocks < 1 or (last - first) % attn_res_blocks:
            raise ValueError("attn_res_blocks must be positive and divide the core layers")
        self.steps = steps
        self.loop_layers = list(loop_layers)
        self.attn_res_blocks = attn_res_blocks
        self.previous_step = int(previous_step)
        super().__init__(**kwargs)


class LoopLayer(Qwen3DecoderLayer):
    """`Qwen3DecoderLayer` extended with the LoopRes read; called without `sources` it is the stock layer, so the
    prelude and the coda run unchanged.

    Under LoopRes, `sources` is the list of (tensor, static key) pairs this layer may read, `hidden_states` is the
    current block's update sum (`None` when the block has not produced an update yet), and `update_key` is the sum's
    own static key. Each sub-layer replaces its input with `static_read` over the sources and the sum, and adds its
    output to the sum."""

    def forward(self, hidden_states, sources=None, step: int = 0, update_key=None, **attention):

        # -----------------------------------------------------------------------
        # 1. Attention: read the sources, add the attention output to the update sum.
        # -----------------------------------------------------------------------
        read = hidden_states
        if sources is not None:
            # At the block's start there is no update sum to read yet.
            pairs = [*sources] if hidden_states is None else [*sources, (hidden_states, update_key)]
            read = static_read(pairs, self.attn_res_query[step], self.attn_res_value_query[step], self.attn_res_norm)
        update = self.self_attn(hidden_states=self.input_layernorm(read), **attention)[0]
        hidden_states = update if hidden_states is None else hidden_states + update

        # -----------------------------------------------------------------------
        # 2. MLP: read again, the grown sum included, and add the MLP output to it.
        # -----------------------------------------------------------------------
        read = hidden_states
        if sources is not None:
            pairs = [*sources, (hidden_states, update_key)]
            read = static_read(pairs, self.mlp_res_query[step], self.mlp_res_value_query[step], self.mlp_res_norm)
        return (hidden_states + self.mlp(self.post_attention_layernorm(read)),)


class LoopResModel(Qwen3Model):
    """The LoopRes decoder stack; the module docstring explains the method. Each static key is stored next to what
    it keys: `prelude_key` for the prelude output, `block_keys[t][b]` for block `b` of loop step `t`, and
    `update_keys[t][b]` for the update sum inside that block."""

    config_class = LoopResConfig

    def __init__(self, config: LoopResConfig) -> None:
        super().__init__(config)
        first, last = config.loop_layers
        hidden, eps = config.hidden_size, config.rms_norm_eps

        # -----------------------------------------------------------------------
        # 1. Core layers: per loop step, each sub-layer's query, value query and read norm. A coda layer's KV-cache
        #    layer comes after every (core layer, step) pass's.
        # -----------------------------------------------------------------------
        for index, layer in enumerate(self.layers):
            layer.__class__ = LoopLayer
            if first <= index < last:
                for name in ("attn_res_query", "mlp_res_query", "attn_res_value_query", "mlp_res_value_query"):
                    # Zero queries weigh every source alike at the start; kept as 1-D vectors, which the optimizer
                    # exempts from weight decay.
                    setattr(
                        layer, name, nn.ParameterList(nn.Parameter(torch.zeros(hidden)) for _ in range(config.steps))
                    )
                layer.attn_res_norm = Qwen3RMSNorm(hidden, eps=eps)
                layer.mlp_res_norm = Qwen3RMSNorm(hidden, eps=eps)
            elif index >= last:
                layer.self_attn.layer_idx = index + (config.steps - 1) * (last - first)

        # -----------------------------------------------------------------------
        # 2. The readout: its query, value query and norm.
        # -----------------------------------------------------------------------
        self.attn_res_output_query = nn.Parameter(torch.zeros(hidden))
        self.attn_res_output_value_query = nn.Parameter(torch.zeros(hidden))
        self.attn_res_output_norm = Qwen3RMSNorm(hidden, eps=eps)

        # -----------------------------------------------------------------------
        # 3. The static keys, one per source, drawn from N(0, 1).
        # -----------------------------------------------------------------------
        self.prelude_key = nn.Parameter(torch.randn(hidden))
        self.block_keys = nn.ModuleList(
            nn.ParameterList(nn.Parameter(torch.randn(hidden)) for _ in range(config.attn_res_blocks))
            for _ in range(config.steps)
        )
        self.update_keys = nn.ModuleList(
            nn.ParameterList(nn.Parameter(torch.randn(hidden)) for _ in range(config.attn_res_blocks))
            for _ in range(config.steps)
        )

    def _init_weights(self, module: nn.Module) -> None:
        """`from_pretrained` materialises the parameters the base checkpoint lacks through this, not the
        constructor: queries zero, static keys N(0, 1)."""
        super()._init_weights(module)
        # `Qwen3Model.__init__`'s own `post_init` lands here before the loop parameters exist.
        if module is self and hasattr(module, "attn_res_output_query"):
            module.attn_res_output_query.data.zero_()
            module.attn_res_output_value_query.data.zero_()
            module.prelude_key.data.normal_()
            for keys in (*module.block_keys, *module.update_keys):
                for key in keys:
                    key.data.normal_()
        elif isinstance(module, LoopLayer) and hasattr(module, "attn_res_query"):
            for name in ("attn_res_query", "mlp_res_query", "attn_res_value_query", "mlp_res_value_query"):
                for query in getattr(module, name):
                    query.data.zero_()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **_: Any,
    ) -> BaseModelOutputWithPast:
        """The last loop step's readout, through the coda and the final norm."""

        # -----------------------------------------------------------------------
        # 1. What every attention takes: the KV cache, the positions, their rotary embeddings and the causal mask.
        # -----------------------------------------------------------------------
        prelude = self.embed_tokens(input_ids)
        if use_cache is None:
            use_cache = self.config.use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if cache_position is None:
            past = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past, past + prelude.shape[1], device=prelude.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        attention = {
            "position_ids": position_ids,
            "position_embeddings": self.rotary_emb(prelude, position_ids),
            "attention_mask": self._update_causal_mask(attention_mask, prelude, cache_position, past_key_values, False),
            "past_key_value": past_key_values,
            "cache_position": cache_position,
        }

        # -----------------------------------------------------------------------
        # 2. Run the prelude once: the embeddings through the layers before the core.
        # -----------------------------------------------------------------------
        first, last = self.config.loop_layers
        for layer in self.layers[:first]:
            prelude = layer(prelude, **attention)[0]

        # -----------------------------------------------------------------------
        # 3. The window: the (tensor, static key) sources every read sees, starting from the prelude output.
        # -----------------------------------------------------------------------
        window = [(prelude, self.prelude_key)]
        size = (last - first) // self.config.attn_res_blocks
        for step in range(self.config.steps):

            # -----------------------------------------------------------------------
            # 4. Run the core blocks: each sub-layer reads the window, the step's finished blocks and its update
            #    sum; its attention writes the (layer, step) pass's own KV-cache layer.
            # -----------------------------------------------------------------------
            blocks, hidden = [], None
            for index in range(first, last):
                self.layers[index].self_attn.layer_idx = index + step * (last - first)
                if not (index - first) % size:
                    # A new block starts its update sum from nothing.
                    hidden = None
                # The sources go in as a tuple: gradient checkpointing re-runs the layer on the ones it first saw.
                hidden = self.layers[index](
                    hidden, step=step, sources=(*window, *blocks),
                    update_key=self.update_keys[step][len(blocks)], **attention,
                )[0]
                if not (index + 1 - first) % size:
                    # A finished block becomes a source, paired with its key.
                    blocks.append((hidden, self.block_keys[step][len(blocks)]))

            # -----------------------------------------------------------------------
            # 5. Slide the window between steps: this step's blocks enter, the oldest step's leave.
            # -----------------------------------------------------------------------
            if step + 1 < self.config.steps:
                window += blocks
                if step >= self.config.previous_step:
                    # Only the last `previous_step` steps are ever read again; older blocks are freed.
                    del window[1 : 1 + self.config.attn_res_blocks]

        # -----------------------------------------------------------------------
        # 6. The readout, once, over the last step's sources; the coda and the final norm follow.
        # -----------------------------------------------------------------------
        hidden = static_read(
            [*window, *blocks], self.attn_res_output_query,
            self.attn_res_output_value_query, self.attn_res_output_norm,
        )
        for layer in self.layers[last:]:
            hidden = layer(hidden, **attention)[0]
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden), past_key_values=past_key_values)


class LoopResForCausalLM(Qwen3ForCausalLM):
    """`Qwen3ForCausalLM` with `LoopResModel` as its decoder stack: `from_pretrained`, `generate` and
    `save_pretrained` are transformers' own."""

    config_class = LoopResConfig

    def __init__(self, config: LoopResConfig) -> None:
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = LoopResModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


AutoConfig.register("loopres", LoopResConfig)
AutoModelForCausalLM.register(LoopResConfig, LoopResForCausalLM)


def build(config: Mapping[str, Any], dtype: torch.dtype = torch.bfloat16) -> Qwen3ForCausalLM:
    """The model around the pretrained `config['name']` under `config['attn_implementation']`: a
    `LoopResForCausalLM` of the remaining keys, or with none the stock pretrained model."""
    options = dict(config)
    name, attention = options.pop("name"), options.pop("attn_implementation")
    if not options:
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation=attention)
    loop = LoopResConfig.from_pretrained(name, **options)
    return LoopResForCausalLM.from_pretrained(name, config=loop, torch_dtype=dtype, attn_implementation=attention)
