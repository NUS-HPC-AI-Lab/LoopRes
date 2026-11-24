"""Train a loop model on packed SFT samples.

    torchrun --nproc_per_node <gpus> -m scripts.train --config configs/<run_name>.yaml

Every epoch repacks the documents in a new order; each rank forwards `per_device_train_batch_size` packed
samples per micro-batch, their documents as one varlen sequence. `training.max_steps`, when set, ends the run after that many
training steps of the `epochs` order. The loss is the cross-entropy of the model's output, summed over tokens and divided
by the labelled tokens of the whole step's batch, across every rank. A checkpoint is written every `save_steps` and at
the end, as a transformers checkpoint in bfloat16 with the base model's tokenizer; the run config is copied to the
output directory as `config.yaml`.
A checkpoint also holds the sharded model and optimizer state, each rank's RNG state and `trainer.json`, written
last; `--resume` continues from the output directory's latest complete checkpoint at the next training step of
the same precomputed data order, on the same ranks and batch.
Under `torchrun` the model trains under FSDP2 in ZeRO-2 mode: parameters, gradients and optimizer state sharded
across the ranks; a decoder layer is gathered from its first forward to its backward, and again in backward
before each loop step's checkpoint recompute.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
import yaml
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint as recompute
from transformers import AutoConfig, AutoModelForCausalLM, get_scheduler

from loopres.data import Dataset, collate, load_tokenizer, pack
from loopres.model import LoopResConfig, LoopResForCausalLM, build


class TrainModel(nn.Module):
    """The unit training shards and calls: the model and its loss in one forward, since FSDP gathers the sharded
    LM head only inside the wrapped module's own forward."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """The cross-entropy at every labelled token, `[tokens]`, the logits chunked and recomputed in
        backward."""
        hidden = self.model.model(
            batch["input_ids"], position_ids=batch["position_ids"], use_cache=False
        ).last_hidden_state
        labelled = batch["labels"] != -100
        targets = batch["labels"][labelled]

        def token_ce(hidden: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(self.model.lm_head(hidden).float(), target, reduction="none")

        # 2048 tokens' logits at a time, recomputed in backward instead of stored.
        parts = zip(hidden[labelled].split(2048), targets.split(2048))
        if not torch.is_grad_enabled():
            return torch.cat([token_ce(*part) for part in parts])
        return torch.cat([recompute(token_ce, *part, use_reentrant=False) for part in parts])


def checkpoint_step(directory: Path) -> int:
    return int(directory.name.removeprefix("checkpoint-"))


def save_state(directory: Path, train_model: TrainModel, optimizer, trainer: dict, rank: int, distributed: bool) -> None:
    """Sharded model and optimizer state and this rank's RNG state into `directory`, which rank 0 has made; rank 0
    writes `trainer.json` once every rank's state is in."""
    model_state, optimizer_state = get_state_dict(train_model, optimizer)
    dcp.save({"model": model_state, "optimizer": optimizer_state}, checkpoint_id=directory / "state")
    rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    torch.save(rng | {"cuda": torch.cuda.get_rng_state()}, directory / "state" / f"rng-{rank}.pt")
    if distributed:
        dist.barrier()
    if rank == 0:
        (directory / "trainer.json").write_text(json.dumps(trainer, indent=2))


def load_state(directory: Path, train_model: TrainModel, optimizer, rank: int) -> dict:
    """The model, optimizer and RNG state `save_state` wrote, and its `trainer.json`."""
    model_state, optimizer_state = get_state_dict(train_model, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=directory / "state")
    set_state_dict(train_model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"])
    rng = torch.load(directory / "state" / f"rng-{rank}.pt", weights_only=False)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state(rng["cuda"])
    return json.loads((directory / "trainer.json").read_text())


def parameter_groups(train_model: TrainModel, weight_decay: float) -> list[dict]:
    """Matrices carry the run's weight decay, vectors and norms none."""
    parameters = [parameter for parameter in train_model.parameters() if parameter.requires_grad]
    return [
        {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": weight_decay},
        {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="continue from the latest complete checkpoint")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    training = config["training"]

    # -----------------------------------------------------------------------
    # 1. Set up: the process group under torchrun, the seed and the output directory.
    # -----------------------------------------------------------------------
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.manual_seed(training["seed"])
    output = Path(training["output_dir"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=args.resume)
        (output / "config.yaml").write_text(args.config.read_text())

    # -----------------------------------------------------------------------
    # 2. Build the model, wrap it with its loss, and shard it across the ranks when distributed.
    # -----------------------------------------------------------------------
    model = build(config["model"], torch.float32).cuda()
    if training["gradient_checkpointing"]:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    train_model = TrainModel(model).train()
    if distributed:
        # FSDP2 in ZeRO-2 mode: every decoder layer a unit, the rest the root, each kept gathered from forward to
        # backward; a layer's checkpoint recompute runs inside the unit's own call, which FSDP gathers for again.
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.model.layers:
            fully_shard(layer, mp_policy=policy, reshard_after_forward=False)
        fully_shard(train_model, mp_policy=policy, reshard_after_forward=False)

    # -----------------------------------------------------------------------
    # 3. Lay out the data: pack each epoch's documents into samples and fix every step's micro-batches up front.
    # -----------------------------------------------------------------------
    train_dataset = Dataset(Path(config["data"]["train"]))
    sample_length, accumulation = config["data"]["sample_length"], training["gradient_accumulation_steps"]
    batch_size = training["per_device_train_batch_size"]
    step_samples = accumulation * batch_size
    # Each epoch's samples for this rank, every rank holding the same count; then every training step's
    # (epoch, first sample of its micro-batches), fixed by the seed, so a step count is a data position.
    epochs = []
    for epoch in range(training["epochs"]):
        samples = pack(train_dataset.lengths, sample_length, training["seed"] + epoch)
        epochs.append(samples[: len(samples) // world * world][rank::world])
    schedule = [
        (epoch, start)
        for epoch, samples in enumerate(epochs)
        for start in range(0, len(samples) // step_samples * step_samples, step_samples)
    ][: training.get("max_steps")]
    total = len(schedule)
    # What must match for a resumed run to read the same micro-batches in the same order.
    alignment = {
        "world": world,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "sample_length": sample_length,
        "seed": training["seed"],
        "epochs": training["epochs"],
        "train": config["data"]["train"],
        "documents": len(train_dataset),
    }
    if "max_steps" in training:
        alignment["max_steps"] = training["max_steps"]

    # -----------------------------------------------------------------------
    # 4. The optimizer and the learning-rate schedule.
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        parameter_groups(train_model, training["weight_decay"]),
        lr=training["learning_rate"],
        betas=(training["adam_beta1"], training["adam_beta2"]),
    )
    scheduler = get_scheduler(
        training["lr_scheduler_type"],
        optimizer,
        num_warmup_steps=math.ceil(training["warmup_ratio"] * total),
        num_training_steps=total,
        scheduler_specific_kwargs=training.get("lr_scheduler_kwargs", {}),
    )

    # -----------------------------------------------------------------------
    # 5. --resume: reload the newest complete checkpoint and check it matches this run's data layout.
    # -----------------------------------------------------------------------
    step, run_id = 0, None
    if args.resume:
        latest = max((path.parent for path in output.glob("checkpoint-*/trainer.json")), key=checkpoint_step)
        latest_step = checkpoint_step(latest)
        # A checkpoint past the latest complete one was cut short; training would reach its directory again.
        later = sorted(str(path) for path in output.glob("checkpoint-*") if checkpoint_step(path) > latest_step)
        if later:
            raise FileExistsError(f"incomplete checkpoints after {latest}: {later}")
        trainer = load_state(latest, train_model, optimizer, rank)
        if trainer["alignment"] != alignment:
            raise ValueError(f"{latest} was trained as {trainer['alignment']}, not {alignment}")
        scheduler.load_state_dict(trainer["scheduler"])
        step, run_id = trainer["step"], trainer["wandb"]

    # -----------------------------------------------------------------------
    # 6. Weights & Biases logging, resuming the checkpoint's run when continuing.
    # -----------------------------------------------------------------------
    run = None
    if rank == 0 and training["report_to"] == "wandb":
        import wandb

        project, name = os.environ.get("WANDB_PROJECT", "LoopRes"), training["run_name"]
        run = wandb.init(project=project, name=name, config=config, id=run_id, resume="must" if run_id else None)

    for epoch, start in schedule[step:]:

        # -----------------------------------------------------------------------
        # 7. One training step: forward and backward over its micro-batches.
        # -----------------------------------------------------------------------
        started = time.perf_counter()
        # Summed cross-entropy and labelled tokens.
        sums = torch.zeros(2, device="cuda")
        samples = epochs[epoch]
        micro_batches = [
            collate(train_dataset, samples[micro : micro + batch_size])
            for micro in range(start, start + step_samples, batch_size)
        ]
        tokens = torch.tensor(float(sum((batch["labels"] != -100).sum().item() for batch in micro_batches)), device="cuda")
        if distributed:
            # The global labelled-token count; FSDP averages gradients over the ranks, so each rank divides by its
            # equal share of it and the summed update comes out as sum(ce) / global tokens.
            dist.all_reduce(tokens)
        normalizer = tokens.item() / world
        for micro, batch in enumerate(micro_batches):
            batch = {name: value.cuda() for name, value in batch.items()}
            if distributed:
                train_model.set_requires_gradient_sync(micro == accumulation - 1)
            with torch.autocast("cuda", torch.bfloat16):
                ce = train_model(batch)
            # The loss: the cross-entropy over the step's labelled tokens, every rank's.
            (ce.sum() / normalizer).backward()
            sums += torch.stack([ce.detach().sum(), ce.new_tensor(len(ce))])

        # -----------------------------------------------------------------------
        # 8. Apply the update: clip the gradient, step the optimizer and the schedule.
        # -----------------------------------------------------------------------
        grad_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), training["max_grad_norm"])
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        # -----------------------------------------------------------------------
        # 9. Log metrics.
        # -----------------------------------------------------------------------
        if distributed:
            dist.all_reduce(sums)
        metrics = {"train/ce": (sums[0] / sums[1]).item()}
        metrics["train/lr"] = scheduler.get_last_lr()[0]
        metrics["train/grad_norm"] = grad_norm.item()
        metrics["train/epoch"] = epoch
        metrics["train/step_seconds"] = time.perf_counter() - started

        # -----------------------------------------------------------------------
        # 10. Every save_steps: write the full model, then the sharded optimizer and trainer state.
        # -----------------------------------------------------------------------
        if step % training["save_steps"] == 0 or step == total:
            # A float32 copy on every rank's GPU; rank 0 writes it in bfloat16 with the tokenizer.
            state = train_model.state_dict()
            if distributed:
                state = get_model_state_dict(train_model, options=StateDictOptions(full_state_dict=True))
            options = dict(config["model"])
            base = options.pop("name")
            options.pop("attn_implementation")
            copy = (
                LoopResForCausalLM(LoopResConfig.from_pretrained(base, **options))
                if options
                else AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(base))
            ).cuda()
            copy.load_state_dict({name.removeprefix("model."): value for name, value in state.items()})
            checkpoint = output / f"checkpoint-{step}"
            if rank == 0:
                tokenizer = load_tokenizer(base)
                # A reply ends at the chat template's turn end, which the base model's generation config, trained
                # without it, does not stop on.
                end = tokenizer.convert_tokens_to_ids("<|im_end|>")
                copy.generation_config.eos_token_id = [end, tokenizer.eos_token_id]
                copy.generation_config.pad_token_id = tokenizer.eos_token_id
                copy.to(torch.bfloat16).save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
            if distributed:
                dist.barrier()
            trainer = {"step": step, "scheduler": scheduler.state_dict(), "alignment": alignment}
            trainer["wandb"] = run.id if run else None
            save_state(checkpoint, train_model, optimizer, trainer, rank, distributed)
            del copy, state
            torch.cuda.empty_cache()
        if rank == 0:
            printed = " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
            print(f"step {step}/{total} {printed}", flush=True)
            if run:
                run.log(metrics, step=step)
    if run:
        run.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
