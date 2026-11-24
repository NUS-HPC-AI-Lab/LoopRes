"""Generate one reply from a checkpoint and print its thinking and its content.

    CUDA_VISIBLE_DEVICES=0 python -m scripts.infer --checkpoint outputs/<run_name>/checkpoint-<step> \\
        --prompt "<prompt>" --max-new-tokens 32768 --device cuda

A reply samples as `configs/eval.yaml`'s `sampling` and stops at Qwen3's stop tokens.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from transformers import AutoModelForCausalLM

from loopres.data import chat_prompt, load_tokenizer  # importing loopres registers LoopResForCausalLM

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="Give me a short introduction to large language model.")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # load the tokenizer and the model
    tokenizer = load_tokenizer(str(args.checkpoint))
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype="auto").to(args.device).eval()

    # prepare the model input
    text = chat_prompt(tokenizer, [{"role": "user", "content": args.prompt}])
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(args.device)

    # conduct text completion; the checkpoint's generation config stops the reply at its turn end
    sampling = yaml.safe_load((ROOT / "configs/eval.yaml").read_text())["sampling"]
    generated = model.generate(**inputs, do_sample=True, max_new_tokens=args.max_new_tokens, **sampling)
    output_ids = generated[0, inputs.input_ids.shape[1] :].tolist()

    # parse the thinking content: through the last `</think>`, none without one
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    index = len(output_ids) - output_ids[::-1].index(think_end) if think_end in output_ids else 0
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)


if __name__ == "__main__":
    main()
