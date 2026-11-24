# minisgl

[mini-sglang](https://github.com/sgl-project/mini-sglang) (commit `9a91cfa`)
serving loopres checkpoints: paged KV cache, continuous batching, overlap
scheduling and CUDA graphs, on one GPU offline or behind an OpenAI-compatible
server. MIT licensed ([LICENSE](LICENSE)).

Against mini-sglang:

- `models/loop/` serves LoopRes checkpoints, whose config.json carries
  `LoopResConfig`'s loop keys: `loopres.py` runs
  `loopres.model.LoopResForCausalLM`'s forward, `common.py` holds what it
  builds on, the AttnRes mixture's Triton kernel included.
  Each (layer, step) pass writes its own KV-cache layer; a read's static-key
  logits fold into the mixture's biases once the weights load. SFT baseline
  checkpoints are stock Qwen3 and take the stock path.
- `SamplingParams.presence_penalty`, taken off the logit of every token a
  request has generated (`engine/sample.py`).
- Top-k sampling keeps each row's top k, then its head within top-p, in torch
  and draws among them with flashinfer's `sampling_from_probs`: flashinfer's
  top-k sampling kernels can spin forever on some distributions.
- A request stops at any of Qwen3's stop tokens, and, when the server runs
  with `--repeat-check`, once `scheduler/repeat.py` finds its reply ends in a
  repeat.
- Admission reserves a decaying share of each request's remaining output, as
  SGLang's `new_token_ratio` does; when the next decode steps would not fit,
  the requests with the most output left go back to the queue.
- `LLM.score` returns every prompt position's logits.
- No prefix cache unless `--cache-type radix`: upstream's radix cache frees a
  request's prompt pages that an identical prompt prefilled alongside it
  cached first, while the request still reads them.
- Serving is one GPU per server; upstream's tensor parallelism is removed.

## Environment

A venv apart from the training environment, from the repository's
`requirements_minisgl.txt`. Native kernels compile on first use with `nvcc`.

```bash
python -m venv <venv>
<venv>/bin/pip install -r requirements_minisgl.txt
```

## Use

`scripts.serve` serves a checkpoint on a set of GPUs for `scripts.eval
--servers` (the repository's README gives both). The parity check, one GPU
per step:

```bash
# the training environment, from the repository root
PYTHONPATH=. python tests/gpu/test_minisgl_parity.py reference --checkpoint outputs/<run_name>/checkpoint-<step> --output <reference.pt>
# the venv, from the repository root; PYTHONPATH reaches minisgl/, which the venv does not install
PYTHONPATH=. python tests/gpu/test_minisgl_parity.py compare --model outputs/<run_name>/checkpoint-<step> --reference <reference.pt>
```

```python
from minisgl.core import SamplingParams
from minisgl.llm import LLM

llm = LLM("outputs/<run_name>/checkpoint-<step>")
llm.generate(prompt_ids, SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384))
```

`python -m minisgl --model-path <checkpoint> --port 30080 --sampling '{"temperature": 0.6, "max_tokens": 32768}'`
serves `/generate_ids`, token ids in and the generated token ids out, sampling
as `--sampling` says where a request leaves a field out, beside `/v1/models`,
`/v1/chat/completions` and `/v1/completions`. A server also takes the next
port, for its engine's rendezvous.
