<div align="center">

# LoopRes

### Unlocking the Power of Loops in Pretrained LLMs with Loop Residuals

[![arXiv](https://img.shields.io/badge/arXiv-2611.31094-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2611.31094)
[![Dataset](https://img.shields.io/badge/Dataset-LoopRes--Qwen3--SFT-ffd21e?logo=huggingface)](https://huggingface.co/datasets/oahzxl/LoopRes-Qwen3-SFT)
[![Model](https://img.shields.io/badge/Model-LoopRes--Qwen3--1.7B-ffd21e?logo=huggingface)](https://huggingface.co/oahzxl/LoopRes-Qwen3-1.7B)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Method](#method) · [Installation](#quick-start) · [Training](#training) · [Inference](#inference) · [Evaluation](#evaluation) · [Citation](#citation) · [Acknowledgement](#acknowledgement)

</div>

This is the official implementation of
**[LoopRes](https://arxiv.org/abs/2611.31094)**. LoopRes turns pretrained LLMs
into looped transformers through finetuning. Each looped layer attends directly
to the outputs of earlier loops through loop residuals. Each of those outputs
carries a trainable loop-aware key, and each layer only sees a window of the
latest W loops. At 2× compute, LoopRes improves Qwen3-1.7B-Base by **29.1%**
over SFT, more than twice the gain of the best previous method. The gains hold
across 6 pretrained LLMs from 3 model families, up to 30B, on 9 math and code
benchmarks. From 1.7B to 8B, LoopRes also scales more steeply with test-time
compute than SFT.

<a id="method"></a>

## 🎉 Method

LoopRes applies a block of decoder layers several loop steps per token: a
prelude runs once, the core layers loop under block attention residuals whose
sources carry trained static keys and value queries, and a coda reads the last
step out. Models train on SFT data and score on math and code benchmarks;
`loopres/model.py`'s docstring gives the method and every config key.

## 🗂 Layout

```text
configs/    example run configs; eval.yaml, the benchmarks protocol
loopres/    the model, its Triton read kernel and the data pipeline
minisgl/    mini-sglang serving checkpoints, in its own venv
scripts/    data download and preparation, training, inference, serving and evaluation
tests/      CPU suite; tests/gpu/ runs by path on a GPU
data/       datasets; outputs/ run outputs (both gitignored)
```

<a id="quick-start"></a>

## 🛠 Dependencies and Installation

Create the LoopRes environment and install its dependencies:

```bash
conda create -n loopres python=3.11 -y
conda activate loopres
pip install -e .
pip install flash-attn  # add --no-build-isolation if the build cannot import torch
```

Create the `minisgl` environment for serving and install its dependencies:

```bash
conda create -n minisgl python=3.11 -y
conda activate minisgl
pip install -r requirements_minisgl.txt
```

## 📦 Data

The training data is
[oahzxl/LoopRes-Qwen3-SFT](https://huggingface.co/datasets/oahzxl/LoopRes-Qwen3-SFT):
`{question, generation}` jsonl, the same questions answered by Qwen3-8B
(`Qwen3-8B/`) and by Qwen3-235B-A22B (`Qwen3-235B-A22B/`), one train split
each.

Download the jsonl splits; Qwen3-8B is the other subset:

```bash
conda activate loopres
python -m scripts.download_data --subsets Qwen3-235B-A22B --output data/LoopRes-Qwen3-SFT
```

Render the subset through the chat template into the tokenised dataset the
configs read:

```bash
conda activate loopres
python -m scripts.prepare_data --source data/LoopRes-Qwen3-SFT/Qwen3-235B-A22B \
  --output data/Qwen3-235B-A22B-16k --max-length 16384
```

<a id="training"></a>

## 🏋️ Training

Train LoopRes in the `loopres` environment:

```bash
conda activate loopres
torchrun --nproc_per_node <gpus> -m scripts.train --config configs/loopres.yaml
```

`configs/loopres.yaml` trains LoopRes on the `data/Qwen3-235B-A22B-16k` dataset
built above; `configs/sft.yaml` is the baseline, the pretrained model at one
step. To train on the Qwen3-8B replies, download and prepare that subset the
same way and point the config's `data:` paths at its dataset. `--resume`
continues from the latest complete checkpoint. A checkpoint directory is a
transformers checkpoint: config.json, bfloat16 weights and the tokenizer; the
run config is copied to the output directory as `config.yaml`.

<a id="inference"></a>

## 🧱 Inference

Download the model:

```bash
conda activate loopres
huggingface-cli download oahzxl/LoopRes-Qwen3-1.7B --local-dir checkpoints/LoopRes-Qwen3-1.7B
```

Load a checkpoint:

```python
import loopres  # required to load a LoopRes checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "checkpoints/LoopRes-Qwen3-1.7B"
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto").cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

`scripts.infer` does that and prints one reply's thinking and its content,
sampling as `configs/eval.yaml`'s `sampling`:

```bash
conda activate loopres
python -m scripts.infer --checkpoint checkpoints/LoopRes-Qwen3-1.7B \
  --prompt "Give me a short introduction to large language model." \
  --max-new-tokens 32768 --device cuda
```

<a id="evaluation"></a>

## 📊 Evaluation

EvalPlus runs in an
[Apptainer](https://apptainer.org/docs/admin/main/installation.html) sandbox.
Apptainer is not on PyPI; install it with Conda:

```bash
conda activate loopres
conda install -c conda-forge apptainer=1.5.4 -y
```

Prepare the benchmark files, EvalPlus's problems and its image (an Apptainer
sandbox), kept once set up; math500 and olympiadbench come from the hub. This
runs in the `loopres` environment:

```bash
conda activate loopres
python -m scripts.prepare_benchmarks --output data/benchmarks
```

Serve a checkpoint in the `minisgl` environment, one server per GPU:

```bash
conda activate minisgl
python -m scripts.serve --checkpoint checkpoints/LoopRes-Qwen3-1.7B --gpus 0 1 2 3
```

Request and score in the `loopres` environment. Replies are saved as they finish
and a rerun on the same `--output` and `--seed` reuses them:

```bash
conda activate loopres
python -m scripts.eval --servers http://127.0.0.1:30080 http://127.0.0.1:30082 \
  http://127.0.0.1:30084 http://127.0.0.1:30086 \
  --output outputs/loopres/eval32k-12000 [--benchmarks aime25 amc23 ...] [--samples <count>]
```

<a id="citation"></a>

## 📝 Citation

If you find LoopRes useful, please cite:

```bibtex
@misc{zhao2026unlockingpowerloopspretrained,
      title={Unlocking the Power of Loops in Pretrained LLMs with Loop Residuals},
      author={Xuanlei Zhao and Hailun Xu and Kai Wang and Yang Luo and Yingxuan Ren and Zirui Zhu and Yuxuan Lou and Kun Xu and Yan Wang and Kanchan Sarkar and Yang You},
      year={2026},
      eprint={2611.31094},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2611.31094},
}
```

<a id="acknowledgement"></a>

## 🙏 Acknowledgement

We thank the following projects for their open-source code:

- [TaH](https://github.com/thu-nics/TaH)
- [mini-sglang](https://github.com/sgl-project/mini-sglang)

## 📄 License

MIT (`LICENSE`). `minisgl/` is vendored from
[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang) at
`9a91cfa`, MIT, with the modifications `minisgl/README.md` lists.
