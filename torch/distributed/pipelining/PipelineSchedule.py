# Copyright (c) Meta Platforms, Inc. and affiliates

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.profiler import record_function

from ._IR import Pipe
from .microbatch import merge_chunks, split_args_kwargs_into_chunks
from .PipelineStage import _PipelineStageBase


__all__ = [
    "PipelineScheduleSingle",
    "PipelineScheduleMulti",
    "Schedule1F1B",
    "ScheduleGPipe",
    "ScheduleInterleaved1F1B",
    "ScheduleLoopedBFS",
]

logger = logging.getLogger(__name__)


class _ComputationType(Enum):
    FORWARD = 1
    BACKWARD = 2


class _PipelineSchedule(ABC):
    def __init__(
        self,
        n_microbatches: int,
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        # From arguments
        self._n_microbatches = n_microbatches
        self._loss_fn = loss_fn
        self._output_merge_spec = output_merge_spec
        # Derived
        self._has_backward = self._loss_fn is not None
        # To be filled by subclasses
        self._pipe_info: Optional[Pipe.PipeInfo] = None

        # Holds the losses for each microbatch.
        self._internal_losses: List[torch.Tensor] = []
        logger.info(f"Using {self.__class__.__name__}")  # noqa: G004

    def _maybe_compute_loss(self, stage, output, target_mbs, mb_index):
        if stage.is_last and self._has_backward:
            loss = self._compute_loss(output, target_mbs[mb_index])  # type: ignore[index]
            self._internal_losses.append(loss)

    def _maybe_get_loss(self, stage, mb_index):
        valid_index = 0 <= mb_index < len(self._internal_losses)
        if stage.is_last and self._has_backward and valid_index:
            return self._internal_losses[mb_index]
        elif len(self._internal_losses) != 0 and not valid_index:
            raise RuntimeError(
                f"Loss for microbatch {mb_index} is not available. "
                f"Available losses for microbatches: {self._internal_losses}"
            )
        else:
            return None

    def _update_losses(self, stages, losses):
        """
        Update the losses to those in the internal state
        """
        # if stages not a list turn into a list
        if not isinstance(stages, list):
            stages = [stages]
        contains_last_stage = any(stage.is_last for stage in stages)

        # Return losses if there is a container passed in
        if contains_last_stage and losses is not None:
            if len(self._internal_losses) != self._n_microbatches:
                raise RuntimeError(
                    f"Expecting {self._n_microbatches} losses but got {len(self._internal_losses)}"
                )

            # Clean external container first
            losses.clear()
            # Copy internal losses to external container
            losses.extend(self._internal_losses)

        self._internal_losses.clear()

    @abstractmethod
    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the schedule
        implementation.

        Args:
            microbatches: list of microbatch args.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """
        raise NotImplementedError

    def _check_inputs(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Pre-process/check inputs
        """

        def check_type_and_len(mbs, name: str):
            if not isinstance(mbs, list):
                raise TypeError(f"{name} must be a list but got a {type(mbs)}")
            if len(mbs) != self._n_microbatches:
                raise ValueError(
                    f"Expecting {self._n_microbatches} {name} but got {len(mbs)}"
                )

        if arg_mbs is not None:
            check_type_and_len(arg_mbs, "arg_mbs")
        else:
            arg_mbs = [()] * self._n_microbatches

        if kwarg_mbs is not None:
            check_type_and_len(kwarg_mbs, "kwarg_mbs")
        else:
            kwarg_mbs = [{}] * self._n_microbatches

        if target_mbs is not None:
            check_type_and_len(target_mbs, "target_mbs")

        if losses is not None:
            if not isinstance(losses, list):
                raise TypeError(f"losses must be a list but got a {type(losses)}")

        return arg_mbs, kwarg_mbs

    def _compute_loss(self, output, target):
        return self._loss_fn(output, target)  # type: ignore[misc]

    def _split_inputs(
        self,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Splits a full-batch input into chunks (i.e. microbatches) and returns
        the chunks
        """
        if self._pipe_info is not None:
            # Use spec from `pipe_info`
            args_chunk_spec = self._pipe_info.args_chunk_spec
            kwargs_chunk_spec = self._pipe_info.kwargs_chunk_spec
        else:
            # Use default spec from `microbatch.py` (i.e. chunk dim 0 for each arg/kwarg)
            args_chunk_spec = None
            kwargs_chunk_spec = None

        if args or kwargs:
            args_split, kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self._n_microbatches,
                args_chunk_spec,
                kwargs_chunk_spec,
            )
            return args_split, kwargs_split
        else:
            # Empty inputs (e.g. when called on middle stages)
            # Return a list of empty tuples/dicts with matching length as chunks
            return [()] * self._n_microbatches, [{}] * self._n_microbatches

    def _merge_outputs(self, output_chunks: List[Any]) -> Any:
        """
        Merge output chunks back to a batch state.
        If output_merge_spec is None, the utility will merge output chunks by dimension 0 (batch dim).
        """
        return merge_chunks(
            output_chunks,
            self._output_merge_spec,
        )


def _batch_p2p(p2p_ops: List[dist.P2POp], desc: Optional[str] = None):
    """
    Simple wrapper over batch_isend_irecv from torch.distributed, which just adds a descriptive logger on top.
    """
    desc_str = f"{desc}, " if desc else ""
    logger.debug(f"batch_p2p {desc_str}{p2p_ops}")  # noqa: G004
    return dist.batch_isend_irecv(p2p_ops).pop()


def _sorted_batch_p2p(
    p2p_ops: List[dist.P2POp], desc: Optional[str] = None
) -> Dict[int, dist.Work]:
    """
    Sorts the list of P2P ops by the peer rank, and then calls
    batch_isend_irecv. Return a dictionary of works by peer rank. This function
    helps us avoid hangs in case of skip connections.
    """
    # Arrange p2p_ops by peer rank:
    #   int is the peer rank;
    #   List is the list of ops towards the peer
    ops_by_peer: Dict[int, List[dist.P2POp]] = defaultdict(list)
    work_by_peer: Dict[int, dist.Work] = {}
    if len(p2p_ops) == 0:
        return work_by_peer

    # Classify the ops by peer rank
    for op in p2p_ops:
        ops_by_peer[op.peer].append(op)

    # Call batch_isend_irecv per peer, in sorted order of the peers (to avoid hangs)
    for peer, ops in sorted(ops_by_peer.items()):
        work_by_peer[peer] = _batch_p2p(ops, desc=desc)

    return work_by_peer


class PipelineScheduleSingle(_PipelineSchedule):
    """
    Base class for single-stage schedules.
    Implements the `step` method.
    Derived classes should implement `_step_microbatches`.
    """

    def __init__(
        self,
        stage: _PipelineStageBase,
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        # Init parent
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            output_merge_spec=output_merge_spec,
        )
        self._pipe_info = (
            stage.pipe_info if hasattr(stage, "pipe_info") else None  # type: ignore[attr-defined]
        )
        # Self attributes
        self._stage = stage
        self._num_stages = stage.num_stages
        # Set the same has_backward flag for stage object
        self._stage.has_backward = self._has_backward

    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        # Clean per iteration
        self._stage.clear_runtime_states()

        # Split inputs into microbatches
        args_split, kwargs_split = self._split_inputs(args, kwargs)

        # Split target into microbatches
        if target is not None:
            targets_split = list(torch.tensor_split(target, self._n_microbatches))
        else:
            targets_split = None

        # Run microbatches
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)

        # Return merged results per original format
        if self._stage.is_last:
            return self._merge_outputs(self._stage.output_chunks)
        else:
            return None


class ScheduleGPipe(PipelineScheduleSingle):
    """
    The GPipe schedule.
    Will go through all the microbatches in a fill-drain manner.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the GPipe schedule.

        Args:
            microbatches: list of microbatch args.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

        # Delay send waits
        fwd_sends_to_wait: List[dist.Work] = []

        # Run microbatches
        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops()
                works = _sorted_batch_p2p(ops, desc="fwd_recv")
                for work in works.values():
                    work.wait()

                output = self._stage.forward_one_chunk(arg_mbs[i], kwarg_mbs[i])  # type: ignore[index]

                ops = self._stage.get_fwd_send_ops()
                works = _sorted_batch_p2p(ops, desc="fwd_send")
                fwd_sends_to_wait.extend(works.values())

            logger.debug(
                f"[{self._stage.stage_index}] Forwarded microbatch {i}"  # noqa: G004
            )

            self._maybe_compute_loss(self._stage, output, target_mbs, i)

        # Wait for all forward sends to finish
        # This should not have performance impact because by the time the first
        # backward arrives all the forward sends should have been finished.
        for work in fwd_sends_to_wait:
            work.wait()

        # No loss function, no need to run backward
        if not self._has_backward:
            return

        # Run backward
        # Delay send waits
        bwd_sends_to_wait: List[dist.Work] = []
        for i in range(self._n_microbatches):
            # set library-specific data-parallel config flags to ensure gradient accumulation across microbatches
            self._stage._configure_data_parallel_mode(i == self._n_microbatches - 1)

            with record_function(f"Backward {i}"):
                ops = self._stage.get_bwd_recv_ops()
                works = _sorted_batch_p2p(ops, desc="bwd_recv")
                for work in works.values():
                    work.wait()

                loss = self._maybe_get_loss(self._stage, i)
                self._stage.backward_one_chunk(loss=loss)

                ops = self._stage.get_bwd_send_ops()
                works = _sorted_batch_p2p(ops, desc="bwd_send")
                bwd_sends_to_wait.extend(works.values())

            logger.debug(
                f"[{self._stage.stage_index}] Backwarded microbatch {i}"  # noqa: G004
            )

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)

        # Wait for all backward sends to finish
        for work in bwd_sends_to_wait:
            work.wait()


class Schedule1F1B(PipelineScheduleSingle):
    """
    The 1F1B schedule.
    Will perform one forward and one backward on the microbatches in steady state.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the 1F1B schedule.

        Args:
            microbatches: list of microbatch args.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

        # Example, 4 GPUs, 8 microbatches
        # Stage 0: 6 warmup, 2 1f1b, 6 cooldown
        # Stage 1: 4 warmup, 4 1f1b, 4 cooldown
        # Stage 2: 2 warmup, 6 1f1b, 2 cooldown
        # Stage 3: 0 warmup, 8 1f1b, 0 cooldown
        # fwd only
        warmup_steps = min(
            self._n_microbatches,
            2 * (self._num_stages - self._stage.stage_index - 1),
        )
        # fwd + bwd
        main_1f1b_steps = self._n_microbatches - warmup_steps
        # bwd only
        cooldown_steps = (2 * self._n_microbatches) - (
            warmup_steps + (2 * main_1f1b_steps)
        )
        total_steps = warmup_steps + main_1f1b_steps + cooldown_steps
        logger.debug(
            f"Stage {self._stage.stage_index}: "  # noqa: G004
            f"Warmup steps: {warmup_steps}, "
            f"Main 1F1B steps: {main_1f1b_steps}, "
            f"Cooldown steps: {cooldown_steps}, "
            f"Total steps: {total_steps}"
        )

        # Delay send waits
        fwd_sends_to_wait: List[dist.Work] = []
        bwd_sends_to_wait: List[dist.Work] = []

        def step_has_forward(i):
            assert i >= 0, i
            return i < self._n_microbatches

        def step_has_backward(i):
            assert i < total_steps, i
            return i >= warmup_steps and self._has_backward

        def is_1f1b_step(i):
            return step_has_forward(i) and step_has_backward(i)

        def is_warmup_step(i):
            return step_has_forward(i) and not step_has_backward(i)

        def is_cooldown_step(i):
            return not step_has_forward(i) and step_has_backward(i)

        def should_coalesce_fwd_send_bwd_recv(step):
            return (
                is_1f1b_step(step)
                or (is_warmup_step(step) and is_cooldown_step(step + 1))
                or (step >= 1 and is_warmup_step(step - 1) and is_cooldown_step(step))
            )

        def should_coalesce_bwd_send_fwd_recv(bwd_send_step):
            # The backward send to prev stage should be coalesced with the fwd recv from the previous stage
            return bwd_send_step >= warmup_steps and is_1f1b_step(bwd_send_step + 1)

        # bwd chunk counter
        bwd_mb_index = 0
        self._stage._configure_data_parallel_mode(last_backward=False)
        for i in range(total_steps):
            if step_has_forward(i):
                with record_function(f"Forward {i}"):
                    ops = self._stage.get_fwd_recv_ops()
                    desc = "fwd_recv"
                    if should_coalesce_bwd_send_fwd_recv(i - 1):
                        desc += "_bwd_send"
                        ops.extend(self._stage.get_bwd_send_ops())

                    works = _sorted_batch_p2p(ops, desc=desc)
                    for work in works.values():
                        work.wait()

                    output = self._stage.forward_one_chunk(arg_mbs[i], kwarg_mbs[i])  # type: ignore[index]

                    if not should_coalesce_fwd_send_bwd_recv(i):
                        ops = self._stage.get_fwd_send_ops()
                        works = _sorted_batch_p2p(ops, desc="fwd_send")
                        fwd_sends_to_wait.extend(works.values())

                self._maybe_compute_loss(self._stage, output, target_mbs, i)

            if step_has_backward(i):
                self._stage._configure_data_parallel_mode(
                    last_backward=(i == total_steps - 1)
                )
                with record_function(f"Backward {bwd_mb_index}"):
                    ops = self._stage.get_bwd_recv_ops()
                    desc = "bwd_recv"
                    if should_coalesce_fwd_send_bwd_recv(i):
                        ops.extend(self._stage.get_fwd_send_ops())
                        desc += "_fwd_send"

                    works = _sorted_batch_p2p(ops, desc=desc)
                    for work in works.values():
                        work.wait()

                    loss = self._maybe_get_loss(self._stage, bwd_mb_index)
                    self._stage.backward_one_chunk(loss=loss)

                    if not should_coalesce_bwd_send_fwd_recv(i):
                        # see Note: coalesced bwd-send/fwd-recv
                        ops = self._stage.get_bwd_send_ops()
                        works = _sorted_batch_p2p(ops, desc="bwd_send")
                        bwd_sends_to_wait.extend(works.values())

                    bwd_mb_index += 1

        # Wait for all forward sends to finish
        for work in fwd_sends_to_wait:
            work.wait()

        # Wait for all backward sends to finish
        for work in bwd_sends_to_wait:
            work.wait()

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)


class PipelineScheduleMulti(_PipelineSchedule):
    """
    Base class for multi-stage schedules.
    Implements the `step` method.
    Derived classes should implement `_step_microbatches`.
    """

    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        if len(stages) <= 1:
            raise ValueError(
                f"Multi-stage schedule expects at least two stages but got {len(stages)}"
            )
        # Init parent
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            output_merge_spec=output_merge_spec,
        )
        self._pipe_info = (
            stages[0].pipe_info if hasattr(stages[0], "pipe_info") else None  # type: ignore[attr-defined]
        )
        # Self attributes
        self._stages = stages
        self._num_stages = stages[0].num_stages
        # Set the same has_backward flag for stage object
        for stage in self._stages:
            stage.has_backward = self._has_backward

        self._should_compute_loss = (
            lambda stage: stage.is_last and self._loss_fn is not None
        )

    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        # Clean per iteration
        for stage in self._stages:
            stage.clear_runtime_states()

        # Split inputs into microbatches
        args_split, kwargs_split = self._split_inputs(args, kwargs)

        # Split target into microbatches
        if target is not None:
            targets_split = list(torch.tensor_split(target, self._n_microbatches))
        else:
            targets_split = None

        # Run microbatches
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)

        # Return merged results per original format
        for stage in self._stages:
            if stage.is_last:
                return self._merge_outputs(stage.output_chunks)
        # Does not contain the last stage
        return None


class ScheduleLoopedBFS(PipelineScheduleMulti):
    """
    Breadth-First Pipeline Parallelism.
    See https://arxiv.org/abs/2211.05953 for details.
    Simliar to Interleaved 1F1B, Looped BFS supports multiple stages per rank.
    What is different is that when microbatches are ready for multiple local
    stages, Loops BFS will prioritizes the earlier stage, running all available
    microbatches at once.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,  # TODO
        losses: Optional[List] = None,  # TODO
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the Looped BFS schedule.

        Args:
            microbatches: list of microbatch args.
        """
        # Pre-process inputs
        if arg_mbs is not None:
            # TODO: fix this so it is preset
            self._n_microbatches = len(arg_mbs)
            assert len(arg_mbs) == self._n_microbatches
        else:
            arg_mbs = [()] * self._n_microbatches

        if kwarg_mbs is not None:
            assert len(kwarg_mbs) == self._n_microbatches
        else:
            kwarg_mbs = [{}] * self._n_microbatches

        for stage in self._stages:
            for i in range(self._n_microbatches):
                with record_function(f"Stage {stage.stage_index} Forward"):
                    ops = stage.get_fwd_recv_ops()
                    if ops:
                        _batch_p2p(ops, desc="fwd_recv").wait()

                    output = stage.forward_one_chunk(arg_mbs[i], kwarg_mbs[i])
                    self._maybe_compute_loss(stage, output, target_mbs, i)

                    ops = stage.get_fwd_send_ops()
                    if ops:
                        _batch_p2p(ops, desc="fwd_send")

        for stage in reversed(self._stages):
            for i in range(self._n_microbatches):
                stage._configure_data_parallel_mode(i == self._n_microbatches - 1)
                with record_function(f"Stage {stage.stage_index} Backward"):
                    ops = stage.get_bwd_recv_ops()
                    if ops:
                        _batch_p2p(ops, desc="bwd_recv").wait()

                    loss = self._maybe_get_loss(stage, i)
                    stage.backward_one_chunk(loss=loss)

                    ops = stage.get_bwd_send_ops()
                    if ops:
                        _batch_p2p(ops, desc="bwd_send")

        self._update_losses(self._stages, losses)


class ScheduleInterleaved1F1B(PipelineScheduleMulti):
    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        self.pp_group_size = stages[0].group_size
        # TODO: is this limitation a must?
        if n_microbatches % self.pp_group_size != 0:
            raise ValueError(
                f"Interleaved 1F1B schedule requires the number of microbatches ({n_microbatches}) \
                to be a multiple of the number of pipeline ranks ({self.pp_group_size})."
            )

        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            output_merge_spec=output_merge_spec,
        )

        self.n_local_stages = len(stages)
        self.rank = stages[0].group_rank
        self.group = stages[0].group

        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: List[
            List[Optional[Tuple[_ComputationType, int, int]]]
        ] = []
        # ========================================================================
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order.append(rank_ops)

    def _calculate_single_rank_operations(
        self, rank
    ) -> List[Optional[Tuple[_ComputationType, int, int]]]:
        def get_rank_warmup_steps(rank):
            # Increment warmup_steps by 2 for each hop away
            warmup_steps = (self.n_local_stages - 1) * self.pp_group_size
            warmup_steps += 2 * ((self.pp_group_size - 1) - rank)
            return min(warmup_steps, self._n_microbatches * self.n_local_stages)

        warmup_steps = get_rank_warmup_steps(rank)
        fwd_bwd_steps = (self.n_local_stages * self._n_microbatches) - warmup_steps
        cooldown_steps = (self.n_local_stages * self._n_microbatches) - fwd_bwd_steps

        assert (
            warmup_steps + fwd_bwd_steps * 2 + cooldown_steps
            == self.n_local_stages * self._n_microbatches * 2
        )
        total_steps = warmup_steps + fwd_bwd_steps + cooldown_steps

        logger.debug(
            "rank %s, warmup_steps %s, 1f1b %s, cooldown_steps %s",
            rank,
            warmup_steps,
            fwd_bwd_steps,
            cooldown_steps,
        )

        # Calculates the stage index based on step and pp_group_size
        def forward_stage_index(step):
            # Get the local index from 0 to n_local_stages-1
            local_index = (step // self.pp_group_size) % self.n_local_stages
            return (local_index * self.pp_group_size) + rank

        def backward_stage_index(step):
            local_index = (
                self.n_local_stages
                - 1
                - ((step - warmup_steps) // self.pp_group_size) % self.n_local_stages
            )
            return (local_index * self.pp_group_size) + rank

        fwd_stage_mb_index: Dict[int, int] = defaultdict(int)
        bwd_stage_mb_index: Dict[int, int] = defaultdict(int)

        # Store the list of operations used for that rank
        rank_ops: List[Optional[Tuple[_ComputationType, int, int]]] = []
        # Pre-padding, rank starts with no-ops based on the warmup.
        for _ in range(rank):
            rank_ops.append(None)

        # These is used to calculate the number of slots to fill with no-ops, to account for the delay in warmup
        # when we want to wait for the backward to trickle back up
        # first backward is going to be pp_group_size - 1 hops away,
        post_warmup_steps = 2 if rank == 0 else 0

        for step in range(total_steps):
            # Warmup phase
            if step < warmup_steps:
                fwd_stage_index = forward_stage_index(step)
                # This will assign the current microbatch index and update it for future steps
                fwd_stage_mb_index[fwd_stage_index] = (
                    mb_index := fwd_stage_mb_index[fwd_stage_index]
                ) + 1
                rank_ops.append((_ComputationType.FORWARD, mb_index, fwd_stage_index))
                if step == warmup_steps - 1:
                    # This is the last step in the warmup phase, so we need to wait for the backward to trickle back up
                    while post_warmup_steps > 0:
                        rank_ops.append(None)
                        post_warmup_steps -= 1
            # 1F1B Phase (forward and backward)
            elif warmup_steps <= step < warmup_steps + fwd_bwd_steps:
                fwd_stage_index = forward_stage_index(step)
                fwd_stage_mb_index[fwd_stage_index] = (
                    fwd_mb_index := fwd_stage_mb_index[fwd_stage_index]
                ) + 1
                rank_ops.append(
                    (_ComputationType.FORWARD, fwd_mb_index, fwd_stage_index)
                )

                bwd_stage_index = backward_stage_index(step)
                bwd_stage_mb_index[bwd_stage_index] = (
                    bwd_mb_index := bwd_stage_mb_index[bwd_stage_index]
                ) + 1
                rank_ops.append(
                    (_ComputationType.BACKWARD, bwd_mb_index, bwd_stage_index)
                )
            # Cooldown phase
            else:
                # During cooldown phase, we need steps to align with 1f1b happening in other ranks
                rank_ops.append(None)
                bwd_stage_index = backward_stage_index(step)
                bwd_stage_mb_index[bwd_stage_index] = (
                    bwd_mb_index := bwd_stage_mb_index[bwd_stage_index]
                ) + 1
                rank_ops.append(
                    (_ComputationType.BACKWARD, bwd_mb_index, bwd_stage_index)
                )

        # Post padding
        for _ in range(self.pp_group_size - rank - 1):
            rank_ops.append(None)
        return rank_ops

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Operate on the microbatches using the interleaved 1f1b schedule (https://arxiv.org/pdf/2104.04473.pdf).

        TODO: Interleaved 1F1B does not use sorted_batch_isend_irecv(). As a result, this schedule does
        not support models with skip connections.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

        # Based on the plan in Step 1 created in __init__:
        # 2. Perform communication based on the pipeline_order
        stage_index_to_stage: Dict[int, _PipelineStageBase] = {
            stage.stage_index: stage for stage in self._stages
        }
        prev_rank: int = (self.rank - 1) % self.pp_group_size
        next_rank: int = (self.rank + 1) % self.pp_group_size

        for time_step, action in enumerate(self.pipeline_order[self.rank]):
            prev_rank_ops = self.pipeline_order[prev_rank]
            next_rank_ops = self.pipeline_order[next_rank]
            ops: List[dist.P2POp] = []
            if action is not None:
                computation_type, mb_index, stage_index = action
                if computation_type == _ComputationType.FORWARD:
                    # perform forward computation
                    stage = stage_index_to_stage[stage_index]
                    output = stage.forward_one_chunk(
                        arg_mbs[mb_index], kwarg_mbs[mb_index]
                    )
                    self._maybe_compute_loss(stage, output, target_mbs, mb_index)
                    ops.extend(stage.get_fwd_send_ops())
                elif computation_type == _ComputationType.BACKWARD:
                    # perform backward computation
                    stage = stage_index_to_stage[stage_index]
                    loss = self._maybe_get_loss(stage, mb_index)
                    stage.backward_one_chunk(loss=loss)
                    if stage_index == 4 and mb_index == 4:
                        pass
                    ops.extend(stage.get_bwd_send_ops())
                else:
                    raise ValueError(f"Unknown computation type {computation_type}")

            # look at neighboring ranks to see if I need to include any recv communication
            if time_step < len(prev_rank_ops):
                prev_rank_action = prev_rank_ops[time_step]
            else:
                prev_rank_action = None
            if prev_rank_action is not None:
                computation_type, mb_index, stage_index = prev_rank_action
                # only handle sends for the forward from a previous rank
                if computation_type == _ComputationType.FORWARD:
                    if stage_index != self._num_stages - 1:
                        stage = stage_index_to_stage[stage_index + 1]
                        ops.extend(stage.get_fwd_recv_ops())
                elif computation_type == _ComputationType.BACKWARD:
                    pass
                else:
                    raise ValueError(f"Unknown computation type {computation_type}")

            if time_step < len(next_rank_ops):
                next_rank_action = next_rank_ops[time_step]
            else:
                next_rank_action = None
            if next_rank_action is not None:
                computation_type, mb_index, stage_index = next_rank_action
                # only handle receives for the backwards from a next rank
                if computation_type == _ComputationType.FORWARD:
                    pass
                elif computation_type == _ComputationType.BACKWARD:
                    if stage_index != 0:
                        stage = stage_index_to_stage[stage_index - 1]
                        ops.extend(stage.get_bwd_recv_ops())
                else:
                    raise ValueError(f"Unknown computation type {computation_type}")

            # do the communication
            if ops:
                dist.batch_isend_irecv(ops).pop().wait()
        # Return losses if there is a container passed in
        self._update_losses(self._stages, losses)
