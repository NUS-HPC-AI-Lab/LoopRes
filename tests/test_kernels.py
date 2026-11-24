import unittest

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from loopres.kernels import static_read


def reference(sources, query, keys, value_query, weight, eps) -> torch.Tensor:
    """The documented mixture in float64: softmax over static-key dot plus RMS-normed value dot, sources averaged."""
    key = weight * value_query
    logits = []
    for index, source in enumerate(sources):
        value = (source @ key) * torch.rsqrt(source.pow(2).mean(-1) + eps)
        logits.append(keys[index] @ query + value)
    weights = torch.stack(logits).softmax(0)
    return sum(w.unsqueeze(-1) * source for w, source in zip(weights, sources))


class StaticReadTest(unittest.TestCase):
    def test_the_cpu_read_matches_the_float64_mixture_in_values_and_gradients(self) -> None:
        torch.manual_seed(0)
        hidden = 16
        eps = 1.0e-6
        norm = Qwen3RMSNorm(hidden, eps=eps)
        with torch.no_grad():
            norm.weight.add_(torch.randn(hidden) * 0.1)
        sources = [torch.randn(2, 5, hidden, requires_grad=True) for _ in range(3)]
        query = torch.randn(hidden, requires_grad=True)
        keys = torch.randn(3, hidden, requires_grad=True)
        value_query = torch.randn(hidden, requires_grad=True)
        leaves = [query, keys, value_query, norm.weight, *sources]

        out = static_read(list(zip(sources, keys)), query, value_query, norm)
        grad = torch.randn_like(out)
        out.backward(grad)

        doubles = [leaf.detach().double().requires_grad_() for leaf in leaves]
        d_query, d_keys, d_value_query, d_weight, *d_sources = doubles
        expected = reference(d_sources, d_query, d_keys, d_value_query, d_weight, eps)
        expected.backward(grad.double())

        torch.testing.assert_close(out.double(), expected, rtol=1.0e-4, atol=1.0e-4)
        for leaf, double in zip(leaves, doubles):
            torch.testing.assert_close(leaf.grad.double(), double.grad, rtol=1.0e-4, atol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
