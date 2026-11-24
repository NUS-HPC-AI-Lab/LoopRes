"""Whether a request's reply ends in a repeat, checked every INTERVAL generated tokens: a span repeated to its end at
least REPEAT_PASSES times with every digit taken as the same character, the repeats past the first spanning
REPEAT_CHARACTERS or more.

Each request keeps its reply's text, extended at each check by decoding only the tokens since the last one, as
SGLang's incremental detokenizer does. A repeat spans REPEAT_CHARACTERS or more, so the text's last GATE digit-masked
characters then also occur earlier in it; when they do not, the reply ends in no repeat and the search is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict

import torch

INTERVAL = 1024
REPEAT_PASSES = 4
REPEAT_CHARACTERS = 3000
GATE = 200
DIGITS = str.maketrans("0123456789", "0000000000")


def ends_in_repeat(text: str) -> bool:
    """Whether `text` ends in a span repeated at least REPEAT_PASSES times, digits masked, the repeats past the first
    spanning REPEAT_CHARACTERS or more."""
    masked = text.translate(DIGITS)
    for period in range(1, len(text) // REPEAT_PASSES + 1):
        if masked[-period:] * REPEAT_PASSES != masked[-REPEAT_PASSES * period :]:
            continue
        start = len(text) - REPEAT_PASSES * period
        while start > 0 and masked[start - 1] == masked[start - 1 + period]:
            start -= 1
        if len(text) - start - period >= REPEAT_CHARACTERS:
            return True
    return False


@dataclass
class _Reply:
    text: str = ""
    # Tokens decoded into `text`, and the start of the last decoded span, which the next decode repeats for context.
    read: int = 0
    surround: int = 0


class RepeatCheck:
    def __init__(self, tokenizer) -> None:
        assert GATE <= REPEAT_CHARACTERS
        self.tokenizer = tokenizer
        self.replies: Dict[int, _Reply] = {}

    def __call__(self, uid: int, generated: torch.Tensor) -> bool:
        """At every INTERVAL-th token of `generated`, the request's tokens so far, whether its reply ends in a repeat."""
        if not len(generated) or len(generated) % INTERVAL:
            return False
        generated = generated.tolist()
        reply = self.replies.setdefault(uid, _Reply())
        decode = partial(self.tokenizer.decode, skip_special_tokens=True)
        context = decode(generated[reply.surround : reply.read])
        new = decode(generated[reply.surround :])[len(context) :]
        text = reply.text + new
        # A span ending inside a character decodes to a replacement character; it is decoded again next time.
        if not new.endswith("�"):
            reply.text, reply.surround, reply.read = text, reply.read, len(generated)
        masked = text.translate(DIGITS)
        if masked.find(masked[-GATE:], 0, len(masked) - 1) < 0:
            return False
        return ends_in_repeat(text)

    def forget(self, uid: int) -> None:
        self.replies.pop(uid, None)
