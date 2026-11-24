import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from scripts import eval as evaluation
from scripts.tasks.utils import process_results


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {f"t{i}": i for i in range(97)}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="t0"))
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="t0", eos_token="t1")


class ServedTest(unittest.TestCase):
    def test_a_limit_keeps_evenly_spaced_documents(self) -> None:
        self.assertEqual(evaluation.spaced(10, 4), [0, 2, 5, 7])
        self.assertEqual(evaluation.spaced(3, None), [0, 1, 2])
        self.assertEqual(evaluation.spaced(3, 8), [0, 1, 2])
        kept = evaluation.spaced(675, 300)
        self.assertEqual(len(set(kept)), 300)
        self.assertEqual(kept, sorted(kept))
        self.assertLess(kept[-1], 675)

    def test_served_requests_carry_their_seeds_end_before_the_stop_token_and_are_reused_on_a_rerun(self) -> None:
        calls = []

        def urlopen(request):
            body = json.loads(request.data)
            calls.append((request.full_url, body["input_ids"], body["seed"]))
            return io.BytesIO(json.dumps({"token_ids": [9, body["input_ids"][0], 1, 4]}).encode())

        tokenizer = tiny_tokenizer()
        with tempfile.TemporaryDirectory() as directory, mock.patch("urllib.request.urlopen", urlopen):
            replies = Path(directory) / "replies.jsonl"
            servers = evaluation.Servers(["http://a", "http://b"], tokenizer, replies, seed=0)
            self.assertEqual(servers.generate([[5], [6], [7], [5]]), [[9, 5], [9, 6], [9, 7], [9, 5]])
            urls = sorted(url for url, _, _ in calls)
            self.assertEqual(urls, ["http://a/generate_ids"] * 2 + ["http://b/generate_ids"] * 2)
            seeds = {(tuple(prompt), seed) for _, prompt, seed in calls}
            # The two asks of one prompt sample under different seeds.
            self.assertEqual(len(seeds), 4)
            # A rerun under the seed takes the four saved replies and requests only the third ask of [5].
            again = evaluation.Servers(["http://a"], tokenizer, replies, seed=0)
            self.assertEqual(again.generate([[6], [5], [7], [5], [5]]), [[9, 6], [9, 5], [9, 7], [9, 5], [9, 5]])
            self.assertEqual(len(calls), 5)
            # Another seed samples every request afresh.
            evaluation.Servers(["http://a"], tokenizer, replies, seed=1).generate([[5]])
            self.assertEqual(len(calls), 6)
            self.assertNotIn(calls[-1][2], {seed for _, _, seed in calls[:-1]})


class ReplyTest(unittest.TestCase):
    def test_math_grading_reads_the_whole_reply_and_averages_the_samples(self) -> None:
        doc = {"answer": "\\frac12"}
        right = "<think>so x = 1/2</think>\nThus \\boxed{0.5}."
        wrong = "<think>so x = 1/2</think>\nThus \\boxed{3}."
        self.assertEqual(process_results(doc, [[right]]), {"math_verify": 1})
        self.assertEqual(process_results(doc, [[wrong]]), {"math_verify": 0})
        # Unclosed thinking is graded too.
        self.assertEqual(process_results(doc, [["<think>\\boxed{1/2} and then"]]), {"math_verify": 1})
        self.assertEqual(process_results(doc, [[right, wrong, wrong, right]]), {"math_verify": 0.5})
        tuple_doc = {"answer": "$(1,8,19), (2,7,13)$"}
        self.assertEqual(process_results(tuple_doc, [["</think> \\boxed{(1,8,19), (2,7,13)}"]]), {"math_verify": 1})

    def test_the_solution_is_the_block_defining_the_entry_point(self) -> None:
        demo = "```python\nprint(f(2))\n```"
        helper = "```python\ndef g(x):\n    return x\n```"
        solution = "```python\ndef f(x):\n    return 2 * x\n```"
        self.assertEqual(
            evaluation.extract_code(f"<think>{helper}</think>{solution}\n{demo}", "f"), "def f(x):\n    return 2 * x\n"
        )
        self.assertEqual(evaluation.extract_code(f"{demo}\n{helper}", "f"), "def g(x):\n    return x\n")
        self.assertEqual(evaluation.extract_code(f"{demo}\n```python\nx = 1\n```", "f"), "x = 1\n")
        self.assertEqual(evaluation.extract_code("no code", "f"), "")


if __name__ == "__main__":
    unittest.main()
