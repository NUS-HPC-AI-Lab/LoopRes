from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch, SamplingParams

# Random values a sampling kernel draws per row at most, flashinfer's own offset step.
RANDOMS_PER_TOKEN = 32


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    # Under a presence penalty: each row's table index and penalty.
    rows: torch.Tensor | None = None
    penalties: torch.Tensor | None = None
    # Under a request's seed: each row's Philox seed and offset.
    rng: dict = field(default_factory=dict)


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
    rng: dict,
) -> torch.Tensor:
    import flashinfer.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs, **rng)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p, **rng)

    # flashinfer's top-k sampling kernels can spin forever on some distributions: each row keeps its top k here,
    # then, renormalized, the head within top_p, and draws among them.
    kept, order = probs.topk(int(top_k.max()), dim=-1)
    ranks = torch.arange(kept.shape[-1], device=kept.device)
    kept = kept.masked_fill(ranks >= top_k[:, None], 0)
    kept = kept / kept.sum(-1, keepdim=True)
    if top_p is not None:
        kept = kept.masked_fill(kept.cumsum(-1) - kept > top_p[:, None], 0)
        kept = kept / kept.sum(-1, keepdim=True)
    drawn = sampling.sampling_from_probs(kept, **rng)
    return order.gather(-1, drawn[:, None].long()).squeeze(-1)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int
    # One row per table index: the tokens its request has generated.
    rows: int = 1

    def __post_init__(self) -> None:
        self.generated = torch.zeros(self.rows, self.vocab_size, dtype=torch.bool, device=self.device)

    def reset(self, table_idx: int, tokens: torch.Tensor) -> None:
        """Start `table_idx`'s row over with `tokens`, the tokens its request has generated so far."""
        self.generated[table_idx].zero_()
        if len(tokens):
            self.generated[table_idx, tokens.to(self.device, torch.int64)] = True

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        args = self._prepare(params)
        if any(p.presence_penalty for p in params):
            args.rows = make_device_tensor([r.table_idx for r in batch.reqs], torch.int64, self.device)
            args.penalties = make_device_tensor([p.presence_penalty for p in params], torch.float32, self.device)
        if any(p.seed is not None for p in params):
            # A request's own Philox stream: its seed, a request without one its uid, at a fresh offset per token.
            seeds = [r.uid if p.seed is None else p.seed for r, p in zip(batch.reqs, params)]
            offsets = [RANDOMS_PER_TOKEN * (r.device_len - r.prompt_len) for r in batch.reqs]
            args.rng = {"seed": make_device_tensor(seeds, torch.int64, self.device)}
            args.rng["offset"] = make_device_tensor(offsets, torch.int64, self.device)
        return args

    def _prepare(self, params: List[SamplingParams]) -> BatchSamplingArgs:
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.penalties is not None:
                logits = logits.float() - args.penalties[:, None] * self.generated[args.rows]
            if args.temperatures is None:  # greedy sampling
                tokens = torch.argmax(logits, dim=-1)
            else:
                tokens = sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p, args.rng)
            if args.rows is not None:
                self.generated[args.rows, tokens.to(torch.int64)] = True
            return tokens
