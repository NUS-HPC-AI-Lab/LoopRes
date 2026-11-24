"""Check a `minisgl` model against its loopres checkpoint, in two steps on one GPU each.

    # the training environment: the first prompts' greedy replies, and every reply position's logits
    PYTHONPATH=. python tests/gpu/test_minisgl_parity.py reference --checkpoint <checkpoint> --output <reference.pt>
    # minisgl's environment, on the same checkpoint directory; PYTHONPATH reaches `minisgl/`, which the venv does not install
    PYTHONPATH=. python tests/gpu/test_minisgl_parity.py compare --model <checkpoint> --reference <reference.pt>

`compare` prints, over the reply positions, how often minisgl's prefill logits share the reference's top token, the
reference's largest margin between its top two where they do not, and the mean and largest difference over the
reference's top 20 logits; then, for minisgl's own greedy replies, where each first differs from the reference's and the
reference's margin between the two tokens there. Matching implementations disagree on the top token only at near ties,
margins within bfloat16 noise (an eighth at logits around 20), with a mean difference of a few hundredths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

PROMPTS, REPLY, TOP = 8, 256, 20


@torch.no_grad()
def reference(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM

    from loopres.data import chat_prompt, load_tokenizer

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype="auto").cuda().eval()
    tokenizer = load_tokenizer(str(args.checkpoint))
    rows = [json.loads(line) for line in open(args.prompts)][:PROMPTS]
    texts = [chat_prompt(tokenizer, [{"role": "user", "content": row["problem"]}]) for row in rows]
    prompts = [tokenizer(text, add_special_tokens=False).input_ids for text in texts]
    vocabulary = tokenizer.get_vocab()
    # Qwen3's stop texts; the tokenizer stops at those its vocabulary holds.
    stops = {tokenizer.eos_token_id, *(vocabulary[t] for t in ["<|im_end|>", "<|endoftext|>"] if t in vocabulary)}
    replies, tops = [], []
    for prompt in prompts:
        # Greedy, each token from a full forward: no cache on the training side.
        ids = list(prompt)
        for _ in range(REPLY):
            token = int(model(torch.tensor([ids], device="cuda"), use_cache=False).logits[0, -1].argmax())
            if token in stops:
                break
            ids.append(token)
        replies.append(ids[len(prompt) :])
        logits = model(torch.tensor([ids], device="cuda"), use_cache=False).logits[0, len(prompt) - 1 : -1].float()
        tops.append(logits.topk(TOP))
    torch.save({
        "prompts": prompts, "replies": replies, "ids": [t.indices.cpu() for t in tops],
        "values": [t.values.cpu() for t in tops],
    }, args.output)


def compare(args: argparse.Namespace) -> None:
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    ref = torch.load(args.reference)
    llm = LLM(str(args.model), cache_type="naive", memory_ratio=args.memory_ratio, cuda_graph_max_bs=PROMPTS)
    sequences = [prompt + reply for prompt, reply in zip(ref["prompts"], ref["replies"])]
    agree = positions = 0
    tie, differences = 0.0, []
    for prompt, reply, logits, ids, values in zip(ref["prompts"], ref["replies"], llm.score(sequences), ref["ids"],
                                                  ref["values"]):
        logits = logits[len(prompt) - 1 : len(prompt) - 1 + len(reply)]
        same = logits.argmax(-1) == ids[:, 0]
        agree += same.sum().item()
        positions += len(reply)
        tie = max([tie, *(values[~same, 0] - values[~same, 1]).tolist()])
        differences.append((logits.gather(-1, ids) - values).abs())
    differences = torch.cat(differences)
    print(f"prefill: top token agrees at {agree}/{positions} positions, the largest reference margin where not "
          f"{tie:.3f}; top-{TOP} logit difference mean {differences.mean():.4f}, largest {differences.max():.3f}")
    replies = [r["token_ids"] for r in llm.generate(ref["prompts"], SamplingParams(max_tokens=REPLY))]
    for index, (ours, theirs) in enumerate(zip(ref["replies"], replies)):
        same = next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b), min(len(ours), len(theirs)))
        line = f"greedy {index}: agrees for {same} of {len(ours)} tokens"
        if same < min(len(ours), len(theirs)):
            ids, values = ref["ids"][index][same].tolist(), ref["values"][index][same]
            theirs_value = values[ids.index(theirs[same])].item() if theirs[same] in ids else float("-inf")
            line += f"; reference margin there {values[0].item() - theirs_value:.3f}"
        print(line)
    llm.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser("reference")
    step.add_argument("--checkpoint", type=Path, required=True)
    step.add_argument("--output", type=Path, required=True)
    step.add_argument("--prompts", type=Path, default=Path("data/benchmarks/MATH-500.jsonl"),
                      help="a jsonl of rows with a `problem` field")
    step = commands.add_parser("compare")
    step.add_argument("--model", type=Path, required=True)
    step.add_argument("--reference", type=Path, required=True)
    step.add_argument("--memory-ratio", type=float, default=0.3, help="the share of free GPU memory minisgl takes")
    args = parser.parse_args()
    {"reference": reference, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
