"""The LoopRes read: a learned softmax average over a list of sources.

A source is a hidden-state tensor paired with its trained static key. Its weight comes from one logit: the key's dot
with the read's query, plus a second (value) query's dot with the source's RMS norm. `static_read` computes the
average in float32; on CUDA a Triton kernel reads each source once forward and twice backward and never materialises
the sources stacked, on CPU plain torch does the same math.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _mix_forward(
    source_ptrs, key_ptr, bias_ptr, out_ptr, rows, eps,
    HIDDEN: tl.constexpr, SOURCES: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_HIDDEN: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, BLOCK_HIDDEN)
    mask = (row[:, None] < rows) & (col[None, :] < HIDDEN)
    offsets = row[:, None].to(tl.int64) * HIDDEN + col[None, :]
    key = tl.load(key_ptr + col, mask=col < HIDDEN, other=0.0)
    dtype = out_ptr.dtype.element_ty
    # Online softmax: each source row is read once.
    peak = tl.full([BLOCK_ROWS], float("-inf"), tl.float32)
    total = tl.zeros([BLOCK_ROWS], tl.float32)
    acc = tl.zeros([BLOCK_ROWS, BLOCK_HIDDEN], tl.float32)
    for index in tl.static_range(SOURCES):
        source_ptr = tl.load(source_ptrs + index).to(tl.pointer_type(dtype))
        source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        logit = tl.sum(source * key[None, :], 1) * tl.rsqrt(tl.sum(source * source, 1) / HIDDEN + eps)
        logit += tl.load(bias_ptr + index)
        new_peak = tl.maximum(peak, logit)
        scale = tl.exp(peak - new_peak)
        weight = tl.exp(logit - new_peak)
        total = total * scale + weight
        acc = acc * scale[:, None] + weight[:, None] * source
        peak = new_peak
    tl.store(out_ptr + offsets, (acc / total[:, None]).to(dtype), mask=mask)


@triton.jit
def _mix_backward(
    source_ptrs, source_grad_ptrs, key_ptr, bias_ptr, grad_ptr, key_grad_ptr, bias_grad_ptr, rows, eps,
    HIDDEN: tl.constexpr, SOURCES: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_HIDDEN: tl.constexpr,
    BLOCK_SOURCES: tl.constexpr,
):
    program = tl.program_id(0)
    col = tl.arange(0, BLOCK_HIDDEN)
    key = tl.load(key_ptr + col, mask=col < HIDDEN, other=0.0)
    key_grad = tl.zeros([BLOCK_HIDDEN], tl.float32)
    slot = tl.arange(0, BLOCK_SOURCES)
    bias_grad = tl.zeros([BLOCK_SOURCES], tl.float32)
    dtype = grad_ptr.dtype.element_ty
    # Each program strides over row blocks and writes one partial key and bias gradient.
    for start in range(program * BLOCK_ROWS, rows, tl.num_programs(0) * BLOCK_ROWS):
        row = start + tl.arange(0, BLOCK_ROWS)
        mask = (row[:, None] < rows) & (col[None, :] < HIDDEN)
        offsets = row[:, None].to(tl.int64) * HIDDEN + col[None, :]
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        # First pass: the softmax's peak and normalizer, and grad . out as the softmax average of grad . source.
        peak = tl.full([BLOCK_ROWS], float("-inf"), tl.float32)
        total = tl.zeros([BLOCK_ROWS], tl.float32)
        grad_out = tl.zeros([BLOCK_ROWS], tl.float32)
        for index in tl.static_range(SOURCES):
            source_ptr = tl.load(source_ptrs + index).to(tl.pointer_type(dtype))
            source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            logit = tl.sum(source * key[None, :], 1) * tl.rsqrt(tl.sum(source * source, 1) / HIDDEN + eps)
            logit += tl.load(bias_ptr + index)
            new_peak = tl.maximum(peak, logit)
            scale = tl.exp(peak - new_peak)
            weight = tl.exp(logit - new_peak)
            total = total * scale + weight
            grad_out = grad_out * scale + weight * tl.sum(grad * source, 1)
            peak = new_peak
        grad_out = grad_out / total
        # Second pass: each source's weight and logit gradient, then its own, the key's and the bias' gradients.
        for index in tl.static_range(SOURCES):
            source_ptr = tl.load(source_ptrs + index).to(tl.pointer_type(dtype))
            source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            dot = tl.sum(source * key[None, :], 1)
            inverse = tl.rsqrt(tl.sum(source * source, 1) / HIDDEN + eps)
            logit = dot * inverse + tl.load(bias_ptr + index)
            weight = tl.exp(logit - peak) / total
            logit_grad = weight * (tl.sum(grad * source, 1) - grad_out)
            bias_grad += tl.where(slot == index, tl.sum(logit_grad, 0), 0.0)
            source_grad = (
                weight[:, None] * grad
                + (logit_grad * inverse)[:, None] * key[None, :]
                - (logit_grad * dot * inverse * inverse * inverse / HIDDEN)[:, None] * source
            )
            source_grad_ptr = tl.load(source_grad_ptrs + index).to(tl.pointer_type(dtype))
            tl.store(source_grad_ptr + offsets, source_grad.to(dtype), mask=mask)
            key_grad += tl.sum((logit_grad * inverse)[:, None] * source, 0)
    tl.store(key_grad_ptr + program * HIDDEN + col, key_grad, mask=col < HIDDEN)
    tl.store(bias_grad_ptr + program * BLOCK_SOURCES + slot, bias_grad, mask=slot < SOURCES)


def _pointers(tensors: list[torch.Tensor]) -> torch.Tensor:
    addresses = torch.tensor([tensor.data_ptr() for tensor in tensors], dtype=torch.int64)
    return addresses.pin_memory().to(tensors[0].device, non_blocking=True)


class _Mix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, key, bias, eps, *sources):
        sources = [source.contiguous() for source in sources]
        bias = bias.float().contiguous()
        hidden = key.numel()
        rows = sources[0].numel() // hidden
        block_hidden = triton.next_power_of_2(hidden)
        block_rows = max(1, 2048 // block_hidden)
        out = torch.empty_like(sources[0])
        _mix_forward[(triton.cdiv(rows, block_rows),)](
            _pointers(sources), key, bias, out, rows, eps, hidden, len(sources),
            # Few warps keep the row reductions near one warp; wider rows need more.
            BLOCK_ROWS=block_rows, BLOCK_HIDDEN=block_hidden, num_warps=max(2, block_hidden // 1024),
        )
        ctx.save_for_backward(key, bias, *sources)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad):
        key, bias, *sources = ctx.saved_tensors
        hidden = key.numel()
        rows = sources[0].numel() // hidden
        block_hidden = triton.next_power_of_2(hidden)
        block_rows = max(1, 2048 // block_hidden)
        source_grads = [torch.empty_like(source) for source in sources]
        processors = torch.cuda.get_device_properties(key.device).multi_processor_count
        programs = min(triton.cdiv(rows, block_rows), 4 * processors)
        key_grads = torch.empty(programs, hidden, dtype=torch.float32, device=key.device)
        slots = triton.next_power_of_2(len(sources))
        bias_grads = torch.empty(programs, slots, dtype=torch.float32, device=key.device)
        _mix_backward[(programs,)](
            _pointers(sources), _pointers(source_grads), key, bias, grad.contiguous(), key_grads, bias_grads,
            rows, ctx.eps, hidden, len(sources),
            BLOCK_ROWS=block_rows, BLOCK_HIDDEN=block_hidden, num_warps=max(2, block_hidden // 1024),
            BLOCK_SOURCES=slots,
        )
        return key_grads.sum(0), bias_grads.sum(0)[: len(sources)], None, *source_grads


def static_read(
    sources: list[tuple[torch.Tensor, torch.Tensor]], query: torch.Tensor, value_query: torch.Tensor, norm: nn.Module
) -> torch.Tensor:
    """The weighted average of `sources`, each a `[..., hidden]` tensor paired with its `[hidden]` static key; a
    source's softmax logit is `key . query + value_query . norm(tensor)`, in float32."""
    tensors = [tensor for tensor, _ in sources]
    bias = torch.stack([static_key for _, static_key in sources]).float() @ query.float()
    # The norm's weight folds into the query: norm(s) @ q = (s @ (weight * q)) * rsqrt(mean(s^2) + eps).
    key = norm.weight.float() * value_query.float()
    eps = norm.variance_epsilon
    if tensors[0].is_cuda:
        # The kernel reads every source as the first's dtype; without FSDP's uniform bfloat16 the prelude arrives
        # in float32 beside bfloat16 update sums.
        tensors = [tensor.to(tensors[0].dtype) for tensor in tensors]
        return _Mix.apply(key, bias, eps, *tensors)
    logits = torch.stack(
        [(tensor.float() @ key) * torch.rsqrt(tensor.float().pow(2).mean(-1) + eps) for tensor in tensors]
    )
    weights = (logits + bias.view(-1, *[1] * (logits.dim() - 1))).softmax(0)
    return sum(weight.unsqueeze(-1) * tensor.float() for weight, tensor in zip(weights, tensors)).to(tensors[0].dtype)
