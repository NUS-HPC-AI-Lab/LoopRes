"""Run by path on a free GPU: `CUDA_VISIBLE_DEVICES=<gpu> python -m unittest tests/gpu/test_loopres.py`."""

import unittest

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from loopres.kernels import static_read
from loopres.model import LoopResForCausalLM
from tests.fakes import tiny_loopres
from tests.test_kernels import reference

# The window of previous steps whose core blocks each core sub-layer and the readout read.
VARIANTS = [{"previous_step": 1}, {"previous_step": 2}]


def loopres_model(**variant) -> LoopResForCausalLM:
    model = tiny_loopres("flash_attention_2", **variant)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "res_" in name:
                parameter.add_(torch.randn_like(parameter) * 0.5)
    return model.to("cuda", torch.bfloat16)


class KernelTest(unittest.TestCase):
    def test_the_kernel_matches_the_float64_mixture_in_values_and_gradients(self) -> None:
        torch.manual_seed(0)
        # Hidden 48 exercises the kernel's masking past a non-power-of-two row.
        for dtype, close in ((torch.float32, 1.0e-3), (torch.bfloat16, 3.0e-2)):
            for count, hidden in ((2, 32), (5, 48)):
                with self.subTest(dtype=dtype, sources=count, hidden=hidden):
                    norm = Qwen3RMSNorm(hidden, eps=1.0e-6).to("cuda", dtype)
                    with torch.no_grad():
                        norm.weight.add_(torch.randn_like(norm.weight) * 0.1)
                    sources = [torch.randn(3, 67, hidden, device="cuda", dtype=dtype, requires_grad=True) for _ in range(count)]
                    query = torch.randn(hidden, device="cuda", dtype=dtype, requires_grad=True)
                    keys = torch.randn(count, hidden, device="cuda", dtype=dtype, requires_grad=True)
                    value_query = torch.randn(hidden, device="cuda", dtype=dtype, requires_grad=True)
                    leaves = [query, keys, value_query, norm.weight, *sources]

                    out = static_read(list(zip(sources, keys)), query, value_query, norm)
                    grad = torch.randn_like(out)
                    out.backward(grad)

                    doubles = [leaf.detach().double().requires_grad_() for leaf in leaves]
                    d_query, d_keys, d_value_query, d_weight, *d_sources = doubles
                    expected = reference(d_sources, d_query, d_keys, d_value_query, d_weight, 1.0e-6)
                    expected.backward(grad.double())

                    torch.testing.assert_close(out.double(), expected, rtol=close, atol=close)
                    for leaf, double in zip(leaves, doubles):
                        torch.testing.assert_close(leaf.grad.double(), double.grad, rtol=close, atol=close)


class LoopResTest(unittest.TestCase):
    @torch.no_grad()
    def test_a_packed_row_is_its_documents_run_apart(self) -> None:
        for variant in VARIANTS:
            with self.subTest(**variant):
                model = loopres_model(**variant)
                documents = [torch.randint(0, 97, (1, n), device="cuda") for n in (7, 12, 5)]
                positions = torch.cat([torch.arange(d.shape[1], device="cuda") for d in documents])[None]
                packed = model.model(torch.cat(documents, dim=1), position_ids=positions, use_cache=False)
                expected = [model.model(document, use_cache=False).last_hidden_state for document in documents]
                torch.testing.assert_close(packed.last_hidden_state, torch.cat(expected, dim=1), atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
