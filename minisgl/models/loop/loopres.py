"""`loopres.model.LoopResModel`: the prelude once, the core `steps` times under block AttnRes, the coda once on the
last step's readout.

Each core sub-layer reads the mixture of the prelude output, the core blocks of `previous_step` earlier steps, the
step's completed blocks and the current block's update sum; the last step's readout reads the same sources and
feeds the coda. A source's logit is its trained key against the read's query — a constant computed once loaded — plus its
RMS-normed value against the read's value query.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from minisgl.layers import BaseOP, RMSNormFused

from .common import LoopModel, Vectors, layers, mix, run

if TYPE_CHECKING:
    from ..config import ModelConfig


class StepVectors(BaseOP):
    """Per step, a loopres `ParameterList` of vectors, under the same names."""

    def __init__(self, steps: int, count: int, size: int):
        for step in range(steps):
            setattr(self, str(step), Vectors(count, size))

    def __getitem__(self, step: int) -> Vectors:
        return getattr(self, str(step))


class Model(LoopModel):
    def __init__(self, config: ModelConfig):
        loop = config.loop
        self._steps = loop["steps"]
        self._first, self._last = loop["loop_layers"]
        self._blocks = loop["attn_res_blocks"]
        self._size = (self._last - self._first) // self._blocks
        self._previous = int(loop["previous_step"])
        stack = layers(config, self._first, self._last, self._steps)
        hidden = config.hidden_size
        for layer in stack[self._first : self._last]:
            # Each core sub-layer's per-step AttnRes queries and value queries, and its read norms.
            for sub in ("attn", "mlp"):
                setattr(layer, f"{sub}_res_query", Vectors(self._steps, hidden))
                setattr(layer, f"{sub}_res_value_query", Vectors(self._steps, hidden))
                setattr(layer, f"{sub}_res_norm", RMSNormFused(hidden, config.rms_norm_eps))
        super().__init__(config, stack)
        self.model.attn_res_output_query = torch.empty(hidden)
        self.model.attn_res_output_norm = RMSNormFused(hidden, config.rms_norm_eps)
        self.model.attn_res_output_value_query = torch.empty(hidden)
        # One static key per source, under the training-side names.
        self.model.prelude_key = torch.empty(hidden)
        self.model.block_keys = StepVectors(self._steps, self._blocks, hidden)
        self.model.update_keys = StepVectors(self._steps, self._blocks, hidden)

    def _read(self, query: torch.Tensor, value_query: torch.Tensor, norm: RMSNormFused, keys: List[torch.Tensor]) -> tuple:
        """`mix`'s bias, key and key epsilon for a read of the sources with static keys `keys` under `query`: the
        bias from the static keys' logits, the key from the value query under the read's norm."""
        bias = (torch.stack(keys).float() @ query.float()).tolist()
        return bias, norm.weight.float() * value_query.float(), norm.eps

    def _layer_read(self, layer: BaseOP, sub: str, step: int, keys: List[torch.Tensor]) -> tuple:
        return self._read(
            getattr(layer, f"{sub}_res_query")[step],
            getattr(layer, f"{sub}_res_value_query")[step],
            getattr(layer, f"{sub}_res_norm"),
            keys,
        )

    def after_load(self) -> None:
        # The static keys of what each read sees, as the training-side `LoopResModel.forward` pairs them: the
        # prelude's, the window steps' block keys, the step's completed block keys and the update sum's.
        core_layers = self.model.layers.op_list[self._first : self._last]
        self._reads = []
        for step in range(self._steps):
            window = [self.model.prelude_key]
            for earlier in range(max(step - self._previous, 0), step):
                window += [self.model.block_keys[earlier][block] for block in range(self._blocks)]
            reads = []
            for offset, layer in enumerate(core_layers):
                done = offset // self._size
                sourced = [*window, *(self.model.block_keys[step][block] for block in range(done))]
                update = self.model.update_keys[step][done]
                # A block's first attention read has no update sum yet.
                attn = [*sourced] + [update] * bool(offset % self._size)
                reads.append((
                    self._layer_read(layer, "attn", step, attn),
                    self._layer_read(layer, "mlp", step, [*sourced, update]),
                ))
            self._reads.append(reads)
        # The readout, once, on the last step's window and blocks.
        self._readout = self._read(
            self.model.attn_res_output_query, self.model.attn_res_output_value_query, self.model.attn_res_output_norm,
            [*window, *(self.model.block_keys[self._steps - 1][block] for block in range(self._blocks))],
        )

    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        layers = self.model.layers.op_list
        prelude = run(layers[: self._first], self.model.embed_tokens.forward(input_ids))
        history = []
        for step in range(self._steps):
            inputs = [prelude]
            for kept in history:
                inputs += kept
            blocks, update = [], None
            for offset, layer in enumerate(layers[self._first : self._last]):
                if not offset % self._size:
                    update = None
                update = _sub_layers(layer, [*inputs, *blocks], update, self._reads[step][offset], step)
                if not (offset + 1) % self._size:
                    blocks.append(update)
            if step + 1 < self._steps:
                history.append(blocks)
                if len(history) > self._previous:
                    # A step out of the window is never read again: its blocks are freed, not indexed past.
                    del history[0]
        # The readout, once, on the last step's sources; the coda and the final norm follow.
        state = mix([*inputs, *blocks], *self._readout)
        return self.normed(run(layers[self._last :], state))


def _sub_layers(layer: BaseOP, sources: List[torch.Tensor], update, reads: tuple, slot: int) -> torch.Tensor:
    """The block's update sum after `layer`'s two sub-layers, each reading `sources` and the update sum, `None` at the
    block's start, by its read in `reads`; each mix runs under the sub-layer's pre-norm, fused into the kernel."""
    attn, mlp = reads
    x = mix([*sources, *([] if update is None else [update])], *attn, norm=layer.input_layernorm)
    out = layer.attention(x, slot)
    update = out if update is None else update + out
    x = mix([*sources, update], *mlp, norm=layer.post_attention_layernorm)
    return update + layer.mlp.forward(x)
