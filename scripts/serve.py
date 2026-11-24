"""Serve a checkpoint through minisgl, one server per GPU, for `scripts.eval --servers`.

    python -m scripts.serve --checkpoint outputs/<run_name>/checkpoint-<step> --gpus 0 1 2 3 [--port 30080]

starts one minisgl server per GPU, on every other port from `--port` (each also takes the next), and prints their
URLs once all answer; they stop with this process. When minisgl runs in its own venv, `--python` names that venv's
Python. A request samples as the evaluation config's `sampling` and `max_new_tokens`, each flag given replacing one
of them; `scripts.eval` seeds each request. `--repeat-check` has the servers stop a reply that ends in a repeat.
Each server logs to `<checkpoint>/server<gpu>.log`.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--port", type=int, default=30080)
    parser.add_argument("--python", default=sys.executable, help="the Python of minisgl's environment")
    parser.add_argument("--evaluation-config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--repeat-check", action="store_true", help="servers stop a reply ending in a repeat")
    args = parser.parse_args()
    # A termination runs the clean-up below, so the servers stop with this process.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    evaluation = yaml.safe_load(args.evaluation_config.read_text())
    sampling = evaluation["sampling"] | {"max_tokens": evaluation["max_new_tokens"]}
    flags = {"max_tokens": args.max_new_tokens, "temperature": args.temperature, "top_p": args.top_p}
    flags |= {"top_k": args.top_k, "presence_penalty": args.presence_penalty}
    sampling |= {key: value for key, value in flags.items() if value is not None}

    urls, servers = [], []
    for index, gpu in enumerate(args.gpus):
        port = args.port + 2 * index
        command = [args.python, "-m", "minisgl", "--model-path", str(args.checkpoint), "--port", str(port)]
        command += ["--sampling", json.dumps(sampling)] + ["--repeat-check"] * args.repeat_check
        env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(ROOT)}
        log = (args.checkpoint / f"server{gpu}.log").open("w")
        # Each server leads a process group of its own: the clean-up below reaps minisgl's workers with it.
        servers.append(
            subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        )
        urls.append(f"http://127.0.0.1:{port}")
    try:
        waiting = set(urls)
        while waiting:
            for server, gpu in zip(servers, args.gpus):
                if server.poll() is not None:
                    raise SystemExit(f"the server on GPU {gpu} exited: {args.checkpoint / f'server{gpu}.log'}")
            for url in list(waiting):
                try:
                    urllib.request.urlopen(f"{url}/v1/models", timeout=2)
                    waiting.discard(url)
                except OSError:
                    pass
            time.sleep(5)
        print(f"serving {args.checkpoint} with {json.dumps(sampling)} at", *urls, flush=True)
        for server in servers:
            server.wait()
    finally:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for server in servers:
                try:
                    os.killpg(server.pid, sig)
                except ProcessLookupError:
                    pass
            time.sleep(5)


if __name__ == "__main__":
    main()
