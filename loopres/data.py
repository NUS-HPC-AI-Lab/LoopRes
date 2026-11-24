"""Chat-format SFT documents and the packed samples training reads.

A dataset is a directory of `tokens.npy` (every document's tokens, concatenated),
`offsets.npy` (document boundaries) and `prompt_lengths.npy`; only the tokens
after a document's prompt carry a label. A packed sample concatenates whole
documents, with positions restarting at each one, which the flash-attention
path reads as document boundaries.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


def load_tokenizer(name: str):
    """`name`'s tokenizer, decoding tokens to their text as is."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    # Cleanup would drop the space before punctuation and contractions, string literals' included.
    tokenizer.clean_up_tokenization_spaces = False
    return tokenizer


def special_pattern(tokenizer) -> re.Pattern:
    """Matches the text of any of `tokenizer`'s special added tokens, which it would read as that token."""
    specials = [token.content for token in tokenizer.added_tokens_decoder.values() if token.special]
    return re.compile("|".join(map(re.escape, specials)))


def chat_prompt(tokenizer, messages: list[dict], add_generation_prompt: bool = True) -> str:
    """`messages` under the chat template, under `add_generation_prompt` the assistant turn left to the model."""
    # The reply's think block is left open for the model to close.
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=True
    )


def encode(tokenizer, question: str, answer: str) -> tuple[list[int], int]:
    """A conversation's tokens and its prompt's length."""
    prompt = chat_prompt(tokenizer, [{"role": "user", "content": question}])
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("the chat template does not extend the generation prompt")
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(full[len(prompt) :], add_special_tokens=False).input_ids
    return prompt_ids + answer_ids, len(prompt_ids)


def write_dataset(directory: Path, documents: list[tuple[list[int], int]], meta: dict) -> None:
    directory.mkdir(parents=True)
    lengths = [len(tokens) for tokens, _ in documents]
    np.save(directory / "tokens.npy", np.concatenate([tokens for tokens, _ in documents]).astype(np.int32))
    np.save(directory / "offsets.npy", np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))
    np.save(directory / "prompt_lengths.npy", np.array([length for _, length in documents], dtype=np.int32))
    (directory / "meta.json").write_text(json.dumps({**meta, "documents": len(documents), "tokens": sum(lengths)}))


class Dataset:
    def __init__(self, directory: Path) -> None:
        self.tokens = np.load(directory / "tokens.npy", mmap_mode="r")
        self.offsets = np.load(directory / "offsets.npy")
        self.prompt_lengths = np.load(directory / "prompt_lengths.npy")
        self.lengths = np.diff(self.offsets)

    def __len__(self) -> int:
        return len(self.lengths)

    def document(self, index: int) -> tuple[np.ndarray, int]:
        return self.tokens[self.offsets[index] : self.offsets[index + 1]], int(self.prompt_lengths[index])


def pack(lengths: np.ndarray, sample_length: int, seed: int) -> list[list[int]]:
    """Documents in a seeded order, filled greedily into samples of at most `sample_length` tokens."""
    samples, sample, used = [], [], 0
    for index in np.random.default_rng(seed).permutation(len(lengths)).tolist():
        if sample and used + lengths[index] > sample_length:
            samples.append(sample)
            sample, used = [], 0
        sample.append(index)
        used += int(lengths[index])
    return samples + [sample] if sample else samples


def collate(dataset: Dataset, samples: list[list[int]]) -> dict[str, torch.Tensor]:
    """The samples' documents as one tensor of batch dimension 1: each token's label is its document's next
    answer token."""
    ids, positions, labels = [], [], []
    for index in [index for sample in samples for index in sample]:
        tokens, prompt_length = dataset.document(index)
        target = np.full(len(tokens), -100, dtype=np.int64)
        target[prompt_length - 1 : -1] = tokens[prompt_length:]
        ids.append(tokens.astype(np.int64))
        positions.append(np.arange(len(tokens)))
        labels.append(target)
    batch = {"input_ids": ids, "position_ids": positions, "labels": labels}
    return {name: torch.from_numpy(np.concatenate(parts))[None] for name, parts in batch.items()}
