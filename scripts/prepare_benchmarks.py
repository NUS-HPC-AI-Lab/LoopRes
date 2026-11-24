"""Set up `data/benchmarks`: the benchmark files the generated math tasks read, and EvalPlus's problems and image.

    python -m scripts.prepare_benchmarks --output data/benchmarks

writes each dataset below, at the revision named, as its test split one row per line, the columns unchanged: the
three `configs/eval.yaml` tasks whose `dataset_path` is `json` -- each task YAML reads its own problem column --
and MATH-500, which `tests/gpu/test_minisgl_parity.py` prompts from. The `math500` and `olympiadbench` tasks read
their datasets from the hub themselves. For HumanEval+ / MBPP+ it downloads the official EvalPlus problem files
and builds the EvalPlus Docker image as the Apptainer sandbox `scripts.eval` runs, each kept once set up.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import urllib.request
from pathlib import Path

from datasets import load_dataset

# Each file's dataset and revision.
BENCHMARKS = {
    "AIME25": ("math-ai/aime25", "563bb8404243c5f09de6ec262f2db674fe5bce9b"),
    "AIME26": ("math-ai/aime26", "79037aebdb6580008fb960d17cb21fd3099083e3"),
    "AMC23": ("math-ai/amc23", "80815d37005feb82cd7f8fbc6901d5d3eff43057"),
    "MATH-500": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"),
}
# The official EvalPlus problem file releases.
EVALPLUS = {
    "HumanEvalPlus-v0.1.10.jsonl":
        "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz",
    "MbppPlus-v0.2.0.jsonl":
        "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz",
}
IMAGE = "docker://ganler/evalplus:v0.3.1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for name, (dataset, revision) in BENCHMARKS.items():
        rows = load_dataset(dataset, split="test", revision=revision)
        path = args.output / f"{name}.jsonl"
        with path.open("w") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")
        print(f"{path}: {len(rows)} problems", flush=True)
    for name, url in EVALPLUS.items():
        path = args.output / name
        if not path.exists():
            with urllib.request.urlopen(url) as response:
                path.write_bytes(gzip.decompress(response.read()))
        print(f"{path}: {sum(1 for _ in path.open())} problems", flush=True)
    sandbox = args.output / "evalplus-v0.3.1"
    if not sandbox.exists():
        subprocess.run(["apptainer", "build", "--sandbox", str(sandbox), IMAGE], check=True)
    print(f"{sandbox}: the EvalPlus image", flush=True)


if __name__ == "__main__":
    main()
