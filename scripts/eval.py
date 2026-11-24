"""Score the model behind `scripts.serve`'s servers on the benchmarks of the evaluation config.

    python -m scripts.eval --servers http://127.0.0.1:<port> ... --output <dir> \
        [--benchmarks <name> ...] [--samples <count>]

The benchmarks are lm-evaluation-harness tasks, under the chat template unless the evaluation config's entry sets
`chat_template: false`, and HumanEval+ / MBPP+ through EvalPlus; the tasks the harness lacks are YAML under
`scripts/tasks`. A benchmark's score averages its `samples` replies of every document or problem. Every benchmark's
replies are requested at once, sampling as the servers were started, each request under a seed of its own that
`--seed` decides; the n-th repeat of a run takes seed n. Each reply is appended to `<output>/replies.jsonl` as it
finishes; a rerun on the same `--output` and `--seed` takes the replies saved there. `--samples` replaces every
benchmark's `samples`. The scores land in `<output>/summary.json`, every record in `<output>/<benchmark>.jsonl`.

EvalPlus problems take an instruction as the user turn, MBPP+'s with the problem's own test cases, and the code
block `extract_code` finds in the whole reply, thinking included, is scored inside the EvalPlus image under
Apptainer. An entry's `image` is the Apptainer sandbox `scripts.prepare_benchmarks` builds from the official
EvalPlus Docker image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import re
import resource
import shutil
import subprocess
import threading
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from loopres.data import chat_prompt, load_tokenizer

# Requests in flight per server, minisgl's running-request limit.
SERVED_SLOTS = 256


class Servers:
    """The minisgl servers of one model, which sample as `scripts.serve` set them, each request under a seed of its own:
    of `seed`, its prompt and how many times the call has asked that prompt already. Every call's requests share
    `SERVED_SLOTS` slots per server, a request going to whichever server frees a slot first. Every finished reply is
    appended to `replies` at once, and a later run on the same file and `seed` takes its saved replies in place of
    requesting them again."""

    def __init__(self, urls: list[str], tokenizer, replies: Path, seed: int) -> None:
        self.urls, self.tokenizer, self.replies, self.seed = urls, tokenizer, replies, seed
        vocabulary = tokenizer.get_vocab()
        # Qwen3's stop texts; the tokenizer stops at those its vocabulary holds.
        stops = ["<|im_end|>", "<|endoftext|>"]
        self.stop_ids = {tokenizer.eos_token_id, *(vocabulary[token] for token in stops if token in vocabulary)}
        self.saved = {}
        if replies.exists():
            with replies.open() as file:
                self.saved = {(r["prompt"], r["seed"]): r["tokens"] for r in map(json.loads, file)}
        self.lock = threading.Lock()
        self.free = queue.Queue()
        for _ in range(SERVED_SLOTS):
            for url in urls:
                self.free.put(url)
        self.pool = ThreadPoolExecutor(SERVED_SLOTS * len(urls))

    @classmethod
    def connect(cls, urls: list[str], replies: Path, seed: int) -> Servers:
        """`urls`' servers, with the tokenizer of the checkpoint the first one serves."""
        with urllib.request.urlopen(f"{urls[0]}/v1/models") as response:
            model = json.load(response)["data"][0]["root"]
        return cls(urls, load_tokenizer(model), replies, seed)

    def generate(self, prompts: list[list[int]]) -> list[list[int]]:
        """Each prompt's continuation up to and excluding its stop token, the prompts spread over the servers."""
        # Each prompt's reply key, its hash and a seed of its own: the n-th ask of one prompt takes its n-th seed.
        asked = defaultdict(int)
        keyed = []
        for prompt in prompts:
            key = hashlib.sha1(json.dumps(prompt).encode()).hexdigest()
            digest = hashlib.sha1(f"{self.seed}:{key}:{asked[key]}".encode()).digest()
            keyed.append((key, int.from_bytes(digest[:8], "big") >> 1))
            asked[key] += 1
        outputs = [self.saved.get(reply_key) for reply_key in keyed]

        def one(index: int) -> list[int]:
            key, seed = keyed[index]
            server = self.free.get()
            body = json.dumps({"input_ids": prompts[index], "seed": seed}).encode()
            request = urllib.request.Request(f"{server}/generate_ids", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(request) as response:
                tokens = json.load(response)["token_ids"]
            self.free.put(server)
            tokens = tokens[: next((i for i, token in enumerate(tokens) if token in self.stop_ids), len(tokens))]
            record = {"prompt": key, "seed": seed, "tokens": tokens}
            with self.lock, self.replies.open("a") as file:
                file.write(json.dumps(record | {"text": self.tokenizer.decode(tokens)}) + "\n")
                self.saved[key, seed] = tokens
            return tokens

        missing = [index for index, output in enumerate(outputs) if output is None]
        for index, tokens in zip(missing, self.pool.map(one, missing)):
            outputs[index] = tokens
        return outputs


def run(
    servers: Servers, evaluation: Mapping[str, Any], names: list[str], output: Path, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Each benchmark's scores; a benchmark whose entry holds a `limit` keeps that many documents of each leaf task,
    or of its EvalPlus problems, evenly spaced.

    Every benchmark's replies are requested at once, so no benchmark's last replies leave the servers idle; the
    benchmarks then score in turn from the saved replies on the main thread, which `math_verify`'s timeout needs.
    `output` takes `<benchmark>.jsonl`, every record, and `summary.json`.
    """
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = servers.tokenizer
    sampling, max_new_tokens = evaluation["sampling"], evaluation["max_new_tokens"]
    gen_kwargs = {"do_sample": sampling["temperature"] > 0, **sampling, "max_gen_toks": max_new_tokens}

    def task_documents(entry: Mapping[str, Any]) -> dict:
        """Each leaf task's kept document indices, every document repeated `samples` times."""
        from lm_eval.tasks import TaskManager, get_task_dict

        def leaves(tree):
            for value in tree.values():
                yield from leaves(value) if isinstance(value, dict) else [value]

        documents = {}
        manager = TaskManager(include_path=str(Path(__file__).parent / "tasks"))
        for task in leaves(get_task_dict([entry["task"]], manager)):
            task.config.repeats = entry.get("samples", 1)
            documents[task] = spaced(len(task.eval_docs), entry.get("limit"))
        return documents

    def code_problems(entry: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
        """Every EvalPlus problem, and the kept ones, each repeated `samples` times."""
        problems = [json.loads(line) for line in Path(entry["problems"]).open()]
        kept = [problems[index] for index in spaced(len(problems), entry.get("limit"))]
        return problems, [problem for problem in kept for _ in range(entry.get("samples", 1))]

    def request(name: str) -> None:
        """`name`'s replies requested from `Servers`, which saves them, without scoring them."""
        entry = evaluation["benchmarks"][name]
        if "task" in entry:
            try:
                harness(harness_lm(servers, requesting=True), task_documents(entry), entry, gen_kwargs, seed)
            except Requested:
                pass
        else:
            evalplus_replies(servers.generate, tokenizer, entry["evalplus"], code_problems(entry)[1])

    with ThreadPoolExecutor(len(names)) as pool:
        list(pool.map(request, names))

    lm = harness_lm(servers)
    summary = {}
    for name in names:
        entry = evaluation["benchmarks"][name]
        if "task" in entry:
            records = harness(lm, task_documents(entry), entry, gen_kwargs, seed)
        else:
            problems, kept = code_problems(entry)
            records = evalplus_replies(servers.generate, tokenizer, entry["evalplus"], kept)
        with (output / f"{name}.jsonl").open("w") as file:
            file.writelines(json.dumps(record, default=str) + "\n" for record in records)
        if "task" in entry:
            # Each `<metric>,<filter>` averaged over every document of every leaf task.
            totals, counts = defaultdict(float), defaultdict(int)
            for record in records:
                for metric in record["metrics"]:
                    key = f"{metric},{record['filter']}"
                    totals[key] += float(record[metric])
                    counts[key] += 1
            scores = {key: totals[key] / counts[key] for key in totals}
            scores["documents"] = len(records) // len({record["filter"] for record in records})
        else:
            replies = defaultdict(list)
            for record in records:
                replies[record["task_id"]].append(record["solution"])
            data = Path(entry["problems"]).parent
            scores = score_evalplus(entry["evalplus"], entry["image"], problems, replies, data, output / name)
            scores["documents"] = len(replies)
        summary[name] = scores
        print(name, scores, flush=True)
    (output / "summary.json").write_text(json.dumps({"servers": servers.urls} | summary, indent=2))
    return summary


def spaced(count: int, limit: int | None) -> list[int]:
    """`limit` of `count` documents' indices, evenly spaced from the first; every index without a `limit`."""
    limit = count if limit is None else min(limit, count)
    return [index * count // limit for index in range(limit)]


class Requested(Exception):
    """Raised by `harness_lm`'s requesting model once its replies are saved, which stops the harness before it
    scores them."""


def harness_lm(servers: Servers, requesting: bool = False):
    """The harness's model generating through `servers`; it scores no likelihoods. A `requesting` model raises
    `Requested` in place of returning its replies."""
    from lm_eval.api.model import LM

    tokenizer = servers.tokenizer

    class HarnessLM(LM):
        tokenizer_name = tokenizer.name_or_path

        def apply_chat_template(self, chat_history, add_generation_prompt=True):
            return chat_prompt(tokenizer, chat_history, add_generation_prompt)

        def generate_until(self, requests, disable_tqdm=False):
            # The chat template writes any BOS token itself.
            prompts = tokenizer([request.args[0] for request in requests], add_special_tokens=False).input_ids
            outputs = servers.generate(prompts)
            if requesting:
                raise Requested
            return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        def loglikelihood(self, requests):
            raise NotImplementedError

        loglikelihood_rolling = loglikelihood

    return HarnessLM()


def harness(lm, documents: dict, entry: Mapping[str, Any], gen_kwargs: dict, seed: int) -> list[dict]:
    """The harness's records for `documents`, each leaf task's document indices, one per document and filter;
    `seed` seeds Python, NumPy and torch, and leaves the harness's few-shot seed as it is."""
    from lm_eval import simple_evaluate

    results = simple_evaluate(
        model=lm,
        tasks=list(documents),
        num_fewshot=entry.get("num_fewshot"),
        samples={task.config.task: indices for task, indices in documents.items()},
        apply_chat_template=entry.get("chat_template", True),
        gen_kwargs=gen_kwargs,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        log_samples=True,
    )
    return [record for records in results["samples"].values() for record in records]


def evalplus_replies(decode: Callable, tokenizer, dataset: str, problems) -> list[dict]:
    """Each problem's solution, `extract_code`'s of its reply to the `dataset` instruction, generated by `decode`.
    The instruction, for `humaneval` or `mbpp`, takes the problem as the user turn, the assistant turn left to the
    model."""
    if dataset == "mbpp":
        instruction = """
Please solve the programming task with test cases below in Python. Make sure your code satisfies the following requirements:
1. The function name and signature must match exactly as specified in the test cases.
2. Your code should be wrapped in a markdown code block without including any test cases.

Task:
{problem}

Test Cases:
```python
{assertion}
```
""".strip()
        contents = [instruction.format(problem=p["prompt"].strip(), assertion=p["assertion"]) for p in problems]
    else:
        instruction = """
Please solve the programming task below in Python. Code should be wrapped in a markdown code block.

```python
{problem}
```
""".strip()
        contents = [instruction.format(problem=p["prompt"].strip()) for p in problems]
    # The chat template writes any BOS token itself.
    texts = [chat_prompt(tokenizer, [{"role": "user", "content": content}]) for content in contents]
    prompts = tokenizer(texts, add_special_tokens=False).input_ids
    outputs = decode(prompts)
    return [
        {"task_id": problem["task_id"],
         "solution": extract_code(tokenizer.decode(tokens, skip_special_tokens=True), problem["entry_point"]),
         "output_tokens": len(tokens)}
        for problem, tokens in zip(problems, outputs)
    ]


def extract_code(reply: str, entry_point: str) -> str:
    """The reply's first ```python block defining `entry_point`, else its first defining any function, else its last
    block; empty without one."""
    blocks = re.findall(r"```python\n(.*?)```", reply, re.DOTALL)
    defining = [block for block in blocks if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", block)]
    defining = defining or [block for block in blocks if re.search(r"\bdef\s+\w+\s*\(", block)]
    return (defining or blocks[-1:] or [""])[0]


def score_evalplus(
    dataset: str, image: str, problems: list[dict], replies: Mapping[str, list[str]], data: Path, output: Path
) -> dict[str, float]:
    """pass@1 of `replies`, each problem's sampled solutions, on the base and the plus tests, averaged over each
    problem's samples and then the problems; run inside the Apptainer sandbox `image` without a network."""
    cache = output / ".cache" / "evalplus"
    # A scoring cut short leaves its cache and its results, which EvalPlus would reuse.
    cache.mkdir(parents=True, exist_ok=True)
    (output / "samples_eval_results.json").unlink(missing_ok=True)
    # The problem files the EvalPlus image reads from its cache.
    for name in ("HumanEvalPlus-v0.1.10.jsonl", "MbppPlus-v0.2.0.jsonl"):
        (cache / name).write_bytes((data / name).read_bytes())
    samples = [
        {"task_id": p["task_id"], "solution": solution}
        for p in problems
        # A problem not evaluated submits a pass stub: the scorer refuses a file missing any problem.
        for solution in replies.get(p["task_id"], [p["prompt"] + "    pass\n"])
    ]
    with (output / "samples.jsonl").open("w") as file:
        file.writelines(json.dumps(sample) + "\n" for sample in samples)
    container = ["apptainer", "exec", "--contain", "--cleanenv", "--net", "--network", "none"]
    container += ["--env", "XDG_CACHE_HOME=/app/.cache", "--bind", f"{output.resolve()}:/app", image]
    subprocess.run(container + ["evalplus.evaluate", dataset, "--samples", "/app/samples.jsonl"], check=True)
    # The problem copies and the expected outputs EvalPlus computes are rebuilt on every scoring.
    shutil.rmtree(cache.parent)
    results: dict[str, Any] = json.loads((output / "samples_eval_results.json").read_text())

    def passed(kind: str) -> float:
        return sum(
            sum(status[f"{kind}_status"] == "pass" for status in results["eval"][task]) / len(results["eval"][task])
            for task in replies
        ) / len(replies)

    return {"pass_at_1,base": passed("base"), "pass_at_1,plus": passed("plus")}


def main() -> None:
    # The requests in flight hold `SERVED_SLOTS` sockets per server, over the usual 1024-descriptor soft limit.
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--servers", nargs="+", required=True, help="scripts.serve's URLs to generate through")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--benchmarks", nargs="+")
    parser.add_argument("--seed", type=int, default=0, help="each benchmark's sampling seed; a repeat takes the next")
    parser.add_argument("--samples", type=int, help="replies per document or problem, for every benchmark")
    args = parser.parse_args()
    evaluation = yaml.safe_load(args.evaluation_config.read_text())
    names = args.benchmarks or list(evaluation["benchmarks"])
    if args.samples:
        for name in names:
            evaluation["benchmarks"][name]["samples"] = args.samples

    args.output.mkdir(parents=True, exist_ok=True)  # the servers append replies there before run() begins
    servers = Servers.connect(args.servers, args.output / "replies.jsonl", args.seed)
    run(servers, evaluation, names, args.output, args.seed)


if __name__ == "__main__":
    main()
