"""Download the `{question, generation}` jsonl splits from Hugging Face.

    python -m scripts.download_data [--subsets Qwen3-235B-A22B Qwen3-8B]

fetches each named subset of the dataset repository into `--output`, both by default: a train split of
the same questions, answered by Qwen3-235B-A22B (`Qwen3-235B-A22B`) or by Qwen3-8B (`Qwen3-8B`).
`scripts.prepare_data` renders a split directory into the tokenised dataset the run configs read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="oahzxl/LoopRes-Qwen3-SFT")
    parser.add_argument(
        "--subsets", nargs="+", default=["Qwen3-235B-A22B", "Qwen3-8B"], help="Qwen3-235B-A22B and/or Qwen3-8B"
    )
    parser.add_argument("--output", type=Path, default=Path("data/LoopRes-Qwen3-SFT"))
    args = parser.parse_args()

    patterns = [f"{subset}/*.jsonl" for subset in args.subsets]
    snapshot_download(args.dataset, repo_type="dataset", allow_patterns=patterns, local_dir=args.output)
    for split in sorted(args.output.glob("*/*.jsonl")):
        print(split)


if __name__ == "__main__":
    main()
