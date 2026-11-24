"""Render `{question, generation}` jsonl splits through the chat template into tokenised datasets.

Each split `<source>/<split>.jsonl` becomes the dataset `<output>/<split>`; a
conversation longer than `--max-length` tokens, or whose text holds a special token's, is dropped. `--procs` encodes in
that many processes, in the rows' order.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm
from loopres.data import encode, load_tokenizer, special_pattern, write_dataset

tokenizer = None
specials = None


def init(name: str) -> None:
    global tokenizer, specials
    tokenizer = load_tokenizer(name)
    specials = special_pattern(tokenizer)


def encode_row(line: str) -> tuple[np.ndarray, int] | None:
    row = json.loads(line)
    if specials.search(row["question"]) or specials.search(row["generation"]):
        return None
    tokens, prompt_length = encode(tokenizer, row["question"], row["generation"])
    return np.array(tokens, dtype=np.int32), prompt_length


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--procs", type=int, default=1)
    args = parser.parse_args()

    for split in args.splits:
        rows = (args.source / f"{split}.jsonl").open().readlines()
        with Pool(args.procs, initializer=init, initargs=(args.tokenizer,)) as pool:
            documents = list(tqdm(pool.imap(encode_row, rows, chunksize=64), total=len(rows), desc=split))
        kept = [document for document in documents if document is not None and len(document[0]) <= args.max_length]
        meta = {"source": str(args.source / f"{split}.jsonl"), "tokenizer": args.tokenizer}
        meta |= {"max_length": args.max_length, "dropped": len(rows) - len(kept)}
        write_dataset(args.output / split, kept, meta)
        print(f"{split}: {len(kept)} documents kept, {len(rows) - len(kept)} dropped")


if __name__ == "__main__":
    main()
