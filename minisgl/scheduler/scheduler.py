from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .repeat import RepeatCheck
from .table import TableManager
from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]

# Qwen3's reply end and document end; a request stops at either, beside the tokenizer's own end of text.
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]
# SGLang's: admission reserves this share of a request's remaining output, decaying from INIT to MIN over DECAY_STEPS
# decode steps and back to INIT after a preemption.
INIT_NEW_TOKEN_RATIO, MIN_NEW_TOKEN_RATIO, DECAY_STEPS = 0.7, 0.098, 600
STATS_STEPS = 500


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        vocab = self.tokenizer.get_vocab()
        self.stop_ids = {self.eos_token_id} | {vocab[token] for token in STOP_TOKENS if token in vocab}
        self.repeat_check = RepeatCheck(self.tokenizer) if config.repeat_check else None
        self.decode_manager.new_token_ratio = INIT_NEW_TOKEN_RATIO
        # Running requests taken out of the batch, freed and queued again once their in-flight tokens are processed.
        self.preempting: List[Req] = []
        self.preemptions = self.decode_steps = 0
        self.start_time = time.perf_counter()
        self.token_pool = self.table_manager.token_pool
        # Recorded after the last launched batch writes its next tokens into `token_pool`: that batch can still carry a
        # request it finished, whose freed row a new request may take, so new rows are written only after it.
        self.token_pool_written: torch.cuda.Event | None = None
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        self._requeue_preempted()
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)
        self._requeue_preempted()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                # A request the last batch finished rode this one too, under overlap scheduling: a further reply
                # could reach the frontend beside its finished one and hide it.
                if isinstance(req, ChunkedReq) or req in self.finished_reqs:
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token in self.stop_ids
                if self.repeat_check:
                    finished = finished or self.repeat_check(req.uid, req.generated)
                    if finished:
                        self.repeat_check.forget(req.uid)
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                if finished:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        self._preempt()
        if self.token_pool_written is not None:
            self.stream.wait_event(self.token_pool_written)
        # TODO: support other policies: e.g. DECODE first
        batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
        if batch is None and (batch := self.decode_manager.schedule_next_batch()):
            self._decode_step()
        return self._prepare_batch(batch) if batch else None

    def _preempt(self) -> None:
        """When the KV cache could not take two more decode steps, take the running requests with the most output
        left out of the batch until it could."""
        running = self.decode_manager.running_reqs
        headroom = 2 * len(running) * self.cache_manager.page_size
        available = self.cache_manager.available_size
        for req in sorted(running, key=lambda req: req.remain_len, reverse=True):
            if available >= headroom:
                break
            self.decode_manager.remove_req(req)
            self.preempting.append(req)
            available += req.device_len
            headroom -= 2 * self.cache_manager.page_size
        if self.preempting:
            self.decode_manager.new_token_ratio = INIT_NEW_TOKEN_RATIO

    def _requeue_preempted(self) -> None:
        """Free the preempted requests that did not finish and queue them again, behind any chunked prefill."""
        resumed = []
        for req in self.preempting:
            if not req.can_decode or req in self.finished_reqs:
                continue
            self._free_req_resources(req)
            params = replace(req.sampling_params, max_tokens=req.remain_len)
            resumed.append(PendingReq(req.uid, req.input_ids[: req.host_len].clone(), params, req.prompt_len))
        self.preemptions += len(resumed)
        self.preempting = []
        pending = self.prefill_manager.pending_list
        chunked = sum(p.chunked_req is not None for p in pending)
        pending[chunked:chunked] = resumed

    def _decode_step(self) -> None:
        manager = self.decode_manager
        decay = (INIT_NEW_TOKEN_RATIO - MIN_NEW_TOKEN_RATIO) / DECAY_STEPS
        manager.new_token_ratio = max(MIN_NEW_TOKEN_RATIO, manager.new_token_ratio - decay)
        self.decode_steps += 1
        if self.decode_steps % STATS_STEPS:
            return
        used = self.cache_manager.num_pages * self.cache_manager.page_size - self.cache_manager.available_size
        logger.info_rank0(
            f"stats t={time.perf_counter() - self.start_time:.0f}s decode_steps={self.decode_steps} "
            f"running={len(manager.running_reqs)} queued={len(self.prefill_manager.pending_list)} "
            f"used_tokens={used} available_tokens={self.cache_manager.available_size} "
            f"ratio={manager.new_token_ratio:.3f} preemptions={self.preemptions}"
        )

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.token_pool_written = torch.cuda.Event()
        self.token_pool_written.record()
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
