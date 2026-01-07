# Copyright (c) OpenMMLab. All rights reserved.
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch
from torch.profiler import record_function

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.model_inputs import ModelInputs, VisionModelInputs
from lmdeploy.utils import get_logger

if TYPE_CHECKING:
    from lmdeploy.pytorch.adapter.adapter import AdapterManager
    from lmdeploy.pytorch.paging import Scheduler
    from lmdeploy.pytorch.strategies.base.engine import EngineStrategy
    from lmdeploy.pytorch.strategies.base.model_agent import ModelAgentStrategy
    from lmdeploy.pytorch.strategies.base.sampling import SamplingStrategy

    from .engine import Engine, SeqList
    from .executor import ExecutorBase

logger = get_logger('lmdeploy')


def _tensorlize_block_offsets(block_offsets, dtype=torch.int32):
    """Tensorlize block_offsets."""
    # copy on numpy is faster than torch.nn.utils.rnn.pad_sequence
    batch_size = len(block_offsets)
    max_len = max([len(off) for off in block_offsets])
    out = np.zeros((batch_size, max_len), dtype=block_offsets[0].dtype)

    for idx, off in enumerate(block_offsets):
        off_len = len(off)
        out[idx, :off_len] = off
    return torch.as_tensor(out, dtype=dtype)


@dataclass
class _DelayedCachePinnedBuffers:
    """Pinned host buffers for delayed-cache/FOCUS metadata.

    NOTE: InputsMakerAsync may prefetch the next forward while the current
    forward is still consuming host tensors, so keep a small pool of these
    buffers.
    """

    max_batches: int
    max_tokens: int
    processing_q_lens: torch.Tensor
    processing_indices: torch.Tensor
    focus_block_progress: Optional[torch.Tensor] = None
    focus_avg_tokens: Optional[torch.Tensor] = None
    focus_mask_seq_offsets: Optional[torch.Tensor] = None
    focus_mask_indices: Optional[torch.Tensor] = None


@dataclass
class InputsMakerConfig:
    """Input maker config.

    This config is added for Dependency Injection
    """
    max_batches: int
    max_prefill_token_num: int
    role: EngineRole
    is_ssm: bool = False
    dp: int = 1
    spec_decoding: bool = False

    @staticmethod
    def from_engine(engine: 'Engine'):
        cache_config = engine.cache_config
        return InputsMakerConfig(
            spec_decoding=engine.specdecode_config is not None,
            max_batches=cache_config.max_batches,
            max_prefill_token_num=cache_config.max_prefill_token_num,
            role=cache_config.role,
            is_ssm=len(cache_config.states_shapes) > 0,
            dp=engine.dist_config.dp,
        )


class InputsMakerAsync:

    def __init__(
        self,
        executor: 'ExecutorBase',
        scheduler: 'Scheduler',
        adapter_manager: 'AdapterManager',
        engine_strategy: 'EngineStrategy',
        sampling_strategy: 'SamplingStrategy',
        model_agent_strategy: 'ModelAgentStrategy',
        config: InputsMakerConfig,
    ):
        self.executor = executor
        self.scheduler = scheduler
        self.adapter_manager = adapter_manager
        self.config = config
        self.spec_decoding = config.spec_decoding

        # strategies
        self.engine_strategy = engine_strategy
        self.sampling_strategy = sampling_strategy
        self.model_agent_strategy = model_agent_strategy

        self._init_do_prefill(config)

        # record for next forward.
        self.next_is_prefill = True
        self.forward_inputs = None

        # Reusable pinned buffers for delayed-cache / FOCUS metadata.
        # Double-buffering is required because InputsMakerAsync may prefetch the
        # next inputs while the current forward is still consuming host tensors.
        self._delayed_cache_pinned_pool_size = 2
        self._delayed_cache_pinned_buffers: List[Optional[_DelayedCachePinnedBuffers]] = [
            None
        ] * self._delayed_cache_pinned_pool_size
        self._delayed_cache_pinned_buffer_idx = 0

    def _init_do_prefill(self, config: InputsMakerConfig):
        if config.role == EngineRole.Prefill:
            self.do_prefill = self.do_prefill_pnode
        elif config.dp == 1:
            self.do_prefill = self.do_prefill_default
        else:
            self.do_prefill = self.do_prefill_dp

    def _create_vision_model_inputs(self, messages: 'SeqList', model_inputs: ModelInputs):
        """Create vision model inputs."""
        batch_size = len(messages)

        def __get_vlm_embeddings():
            """Get vlm input embeddings and indexings."""
            max_q_seq_length = model_inputs.seq_length.max().item()
            input_embeddings = [[
                emb.embeddings if isinstance(emb.embeddings, torch.Tensor) else torch.as_tensor(emb.embeddings)
                for emb in msg.input_embeddings
            ] for msg in messages]
            input_embedding_ranges = [
                torch.tensor([[emb.start, emb.end] for emb in msg.input_embeddings]) for msg in messages
            ]
            input_embedding_indexing = torch.zeros((batch_size, max_q_seq_length), dtype=torch.bool)
            for msg_id, msg in enumerate(messages):
                num_history_ids = msg.num_history_ids
                for emb in msg.input_embeddings:
                    # make slice index relative to embeddings
                    emb_start = emb.start - num_history_ids
                    emb_end = emb.end - num_history_ids
                    input_embedding_indexing[msg_id][emb_start:emb_end] = True
            return (input_embeddings, input_embedding_indexing, input_embedding_ranges)

        def __has_values(input_multimodals):
            for input_mm in input_multimodals:
                for val in input_mm.values():
                    if len(val) > 0:
                        return True
            return False

        has_embedding = any([len(msg.history_embeddings) > 0 for msg in messages])
        if has_embedding:
            has_embedding = any([len(msg.input_embeddings) > 0 for msg in messages])

        has_multimodal = any([not msg.history_multimodals.empty() for msg in messages])
        input_multimodals = None
        if has_multimodal:
            input_multimodals = [msg.get_input_multimodals() for msg in messages]
            has_multimodal = __has_values(input_multimodals)
            if not has_multimodal:
                # no multimodal inputs
                input_multimodals = None

        if not has_embedding and not has_multimodal:
            # no vision inputs
            return None

        if has_embedding:
            # for inputs with embeddings
            (input_embeddings, input_embedding_indexing, input_embedding_ranges) = __get_vlm_embeddings()
        else:
            input_embeddings = None
            input_embedding_indexing = None
            input_embedding_ranges = None

        history_lengths = model_inputs.history_lengths
        vision_embedding_inputs = VisionModelInputs(history_lengths=history_lengths,
                                                    input_embeddings=input_embeddings,
                                                    input_embedding_indexing=input_embedding_indexing,
                                                    input_embedding_ranges=input_embedding_ranges,
                                                    input_multimodals=input_multimodals)
        return vision_embedding_inputs

    @property
    def torch_int_dtype(self):
        """Return int32 for cuda, int64 for others."""
        if self.executor.device_type == 'cuda':
            return torch.int32
        return torch.int64

    def _get_delayed_cache_pinned_buffers(self, max_tokens: int, focus_enabled: bool) -> _DelayedCachePinnedBuffers:
        """Get a pooled pinned-buffer set for delayed-cache/FOCUS metadata."""
        max_batches = self.config.max_batches

        idx = self._delayed_cache_pinned_buffer_idx
        self._delayed_cache_pinned_buffer_idx = (idx + 1) % self._delayed_cache_pinned_pool_size

        buf = self._delayed_cache_pinned_buffers[idx]
        need_capacity = (buf is None or buf.max_batches < max_batches or buf.max_tokens < max_tokens)
        need_focus = focus_enabled and (buf is None or buf.focus_mask_indices is None or buf.focus_block_progress is None
                                        or buf.focus_avg_tokens is None)
        if need_capacity or need_focus:
            processing_q_lens = torch.empty((max_batches, ), dtype=torch.long, device='cpu', pin_memory=True)
            processing_indices = torch.empty((max_tokens, ), dtype=torch.long, device='cpu', pin_memory=True)

            focus_block_progress = None
            focus_avg_tokens = None
            focus_mask_seq_offsets = None
            focus_mask_indices = None
            if focus_enabled:
                focus_block_progress = torch.empty((max_batches, ), dtype=torch.int32, device='cpu', pin_memory=True)
                focus_avg_tokens = torch.empty((max_batches, ), dtype=torch.float32, device='cpu', pin_memory=True)
                focus_mask_seq_offsets = torch.empty((max_batches + 1, ), dtype=torch.int32, device='cpu', pin_memory=True)
                focus_mask_indices = torch.empty((max_tokens, ), dtype=torch.long, device='cpu', pin_memory=True)

            buf = _DelayedCachePinnedBuffers(
                max_batches=max_batches,
                max_tokens=max_tokens,
                processing_q_lens=processing_q_lens,
                processing_indices=processing_indices,
                focus_block_progress=focus_block_progress,
                focus_avg_tokens=focus_avg_tokens,
                focus_mask_seq_offsets=focus_mask_seq_offsets,
                focus_mask_indices=focus_mask_indices,
            )
            self._delayed_cache_pinned_buffers[idx] = buf

        return buf

    @torch.inference_mode()
    @record_function('create_model_inputs')
    def create_model_inputs(self, messages: 'SeqList', is_prefill: bool):
        """Create model inputs from messages.

        Args:
            messages (SeqList): The input messages.
        """
        batch_size = len(messages)
        # history lengths
        history_lengths = torch.tensor([msg.num_history_ids for msg in messages])

        # input ids
        token_ids = [msg.token_ids for msg in messages]

        input_ids = torch.as_tensor(np.concatenate(token_ids))[None]

        # seqlens
        is_decoding = not is_prefill
        if not is_decoding:
            seq_length = [len(tokens) for tokens in token_ids]
            seq_length = torch.tensor(seq_length, dtype=torch.long)
            max_q_seqlen = seq_length.max().item()
        else:
            max_q_seqlen = len(token_ids[0]) if token_ids else 0
            seq_length = torch.full((batch_size, ), max_q_seqlen, dtype=torch.long)

        processing_indices = None
        processing_q_lens = None
        focus_block_progress = None
        focus_mask_global_indices = None
        focus_mask_seq_offsets = None
        focus_avg_tensor = None
        dllm_cfg = getattr(self.executor.misc_config, 'dllm_config', None)
        enable_delayed = bool(dllm_cfg and getattr(dllm_cfg, 'enable_delayed_cache', False) and is_decoding)
        focus_enabled = bool(enable_delayed and getattr(dllm_cfg, 'enable_focus', False))
        if enable_delayed:
            dllm_block_len = dllm_cfg.block_length
            max_total_proc = self.config.max_batches * dllm_block_len
            pinned = self._get_delayed_cache_pinned_buffers(max_total_proc, focus_enabled=focus_enabled)
            processing_q_lens = pinned.processing_q_lens[:batch_size]
            processing_indices_buffer = pinned.processing_indices
            proc_write = 0

            focus_mask_buffer = None
            mask_write = 0
            if focus_enabled:
                focus_block_progress = pinned.focus_block_progress[:batch_size]
                focus_avg_tensor = pinned.focus_avg_tokens[:batch_size]
                focus_mask_seq_offsets = pinned.focus_mask_seq_offsets[:batch_size + 1]
                focus_mask_seq_offsets[0] = 0
                focus_mask_buffer = pinned.focus_mask_indices

            for msg_idx, msg in enumerate(messages):
                indices = msg.get_processing_indices()
                proc_len = indices.numel()
                processing_q_lens[msg_idx] = proc_len
                processing_indices_buffer[proc_write:proc_write + proc_len].copy_(indices, non_blocking=False)

                if focus_enabled:
                    focus_info = msg.get_focus_info()
                    focus_block_progress[msg_idx] = focus_info.rightmost_processed
                    focus_avg_tensor[msg_idx] = float(focus_info.avg_decoded_tokens)
                    local_indices = focus_info.mask_local_indices
                    local_count = local_indices.numel()
                    dst = focus_mask_buffer[mask_write:mask_write + local_count]
                    torch.add(local_indices, proc_write, out=dst)
                    mask_write += local_count
                    focus_mask_seq_offsets[msg_idx + 1] = mask_write
                proc_write += proc_len

            processing_indices = processing_indices_buffer[:proc_write]
            if focus_enabled:
                focus_mask_global_indices = focus_mask_buffer[:mask_write]
        kv_seqlens = seq_length + history_lengths
        max_kv_seqlen = kv_seqlens.max().item()
        sum_kv_seqlen = kv_seqlens.sum().item()

        # block offsets
        block_offsets = self.scheduler.get_block_tables(messages)
        block_offsets = _tensorlize_block_offsets(block_offsets, dtype=self.torch_int_dtype)

        # num_ignored_history
        num_ignored_history = torch.tensor([msg.num_ignored_history for msg in messages])

        # model_metas
        model_metas = [msg.model_meta for msg in messages]

        # create model inputs for all required fields
        model_inputs = ModelInputs(
            input_ids=input_ids,
            seq_length=seq_length,
            history_lengths=history_lengths,
            block_offsets=block_offsets,
            is_decoding=is_decoding,
            num_ignored_history=num_ignored_history,
            max_q_seqlen=max_q_seqlen,
            max_kv_seqlen=max_kv_seqlen,
            sum_kv_seqlen=sum_kv_seqlen,
            model_metas=model_metas,
        )

        if processing_indices is not None:
            model_inputs.processing_indices = processing_indices
            model_inputs.processing_q_lens = processing_q_lens
            from .engine import build_delayed_cache_ragged_metadata
            tile_to_seq, seq_tile_offsets, max_proc_q_len = build_delayed_cache_ragged_metadata(
                processing_q_lens,
                self.executor.model_config,
                self.executor.cache_config.block_size,
            )
            model_inputs.processing_max_q_len = max_proc_q_len
            model_inputs.ragged_tile_to_seq = tile_to_seq
            model_inputs.ragged_seq_tile_offsets = seq_tile_offsets
            if focus_enabled:
                model_inputs.focus_block_progress = focus_block_progress
                model_inputs.focus_avg_tokens = focus_avg_tensor
                model_inputs.focus_mask_global_indices = focus_mask_global_indices
                model_inputs.focus_mask_seq_offsets = focus_mask_seq_offsets

        # adapters
        local_adapter_ids = None
        if self.adapter_manager.num_adapters() > 1:
            adapter_names = [msg.adapter_name for msg in messages]
            local_adapter_ids = self.adapter_manager.get_adapter_ids(adapter_names)
            local_adapter_ids = seq_length.new_tensor(local_adapter_ids)
            model_inputs.local_adapter_ids = local_adapter_ids

        # cross for mllama
        cross_length = torch.tensor([msg.num_cross for msg in messages])
        history_cross_length = torch.tensor([msg.num_history_cross for msg in messages])
        if (cross_length + history_cross_length).max().item() > 0:
            model_inputs.cross_length = cross_length
            model_inputs.history_cross_length = history_cross_length

        # vision inputs
        vision_model_inputs = self._create_vision_model_inputs(messages, model_inputs)
        model_inputs.vision_inputs = vision_model_inputs

        # ssm
        if self.config.is_ssm:
            state_offsets = torch.tensor([msg.logical_state for msg in messages])
            model_inputs.state_offsets = state_offsets

        return model_inputs

    @torch.inference_mode()
    @record_function('make_forward_inputs')
    def _make_forward_inputs(self, prefill: bool, enable_empty: bool = False):
        """Make forward inputs for ModelAgent._async_step_background()"""

        def __need_logits(seqs: 'SeqList'):
            """Need logits."""
            if self.spec_decoding:
                return True
            return any(seq.return_logits for seq in seqs)

        def __need_routed_experts(seqs: 'SeqList'):
            """Need routed experts."""
            return any(seq.return_routed_experts for seq in seqs)

        def __need_schedule_again(prefill: bool, scheduler_output):
            """Need schedule again."""
            # only reschedule when prefill
            if not prefill:
                return False
            # schedule decoding if no valid prefill reqs.
            if len(scheduler_output.running) > 0:
                return False
            # disable decoding for prefill role
            if (self.config.role == EngineRole.Prefill):
                return False
            # disable decoding if no running reqs.
            if not self.scheduler.has_ready():
                logger.warning('No running sequences for decoding scheduling after prefill scheduling.')
                return False
            return True

        scheduler = self.scheduler
        logger.debug(f'Make forward inputs with prefill={prefill}, enable_empty={enable_empty}')

        prealloc_size = self.engine_strategy.get_prealloc_size(not prefill)
        scheduler_output = scheduler.schedule(is_prefill=prefill, prealloc_size=prealloc_size)

        if enable_empty and len(scheduler_output.running) == 0:
            return None

        if __need_schedule_again(prefill, scheduler_output):
            prefill = False
            prealloc_size = self.engine_strategy.get_prealloc_size(not prefill)
            scheduler_output = scheduler.schedule(is_prefill=prefill, prealloc_size=prealloc_size)

        num_loops = self.engine_strategy.get_num_loops(not prefill)
        running = scheduler_output.running
        swap_in_map = scheduler_output.swap_in_map
        swap_out_map = scheduler_output.swap_out_map

        if len(running) == 0:
            return None

        # create inputs
        inputs = self.create_model_inputs(running, prefill)
        sampling_inputs = self.sampling_strategy.make_sampling_inputs(running)
        return_logits = __need_logits(running)
        return_routed_experts = __need_routed_experts(running)
        extra_inputs = self.model_agent_strategy.make_extra_inputs(running)
        stopping_criteria = self.model_agent_strategy.make_stopping_criteria(running)

        sync_long_context = inputs.input_ids.numel() > self.config.max_prefill_token_num

        return dict(
            running=running,
            inputs=inputs,
            swap_in_map=swap_in_map,
            swap_out_map=swap_out_map,
            loop_count=num_loops,
            sampling_inputs=sampling_inputs,
            stopping_criteria=stopping_criteria,
            return_logits=return_logits,
            is_dummy=False,
            sync_long_context=sync_long_context,
            extra_inputs=extra_inputs,
            return_routed_experts=return_routed_experts,
        )

    def do_prefill_pnode(self):
        return True

    def do_prefill_dp(self):
        scheduler = self.scheduler

        if self.next_is_prefill:
            ret = scheduler.has_waiting()
        else:
            ret = not scheduler.has_ready()
        return ret

    def do_prefill_default(self):
        # decoding if no waiting
        scheduler = self.scheduler
        if not scheduler.has_waiting():
            return False
        num_ready = scheduler.num_ready()
        num_waiting = scheduler.num_waiting()
        max_batches = self.config.max_batches
        # prefill if too much waiting
        permitted_waiting = 4 if (self.config.role != EngineRole.Prefill) else 1
        if num_waiting >= permitted_waiting:
            return True
        # prefill if no enough running
        if num_ready < max_batches * 0.5:
            return True
        # decoding
        return False

    async def _send_next_inputs_impl(self, prefill: bool = None, enable_empty: bool = False):
        forward_inputs = self._make_forward_inputs(prefill, enable_empty)
        if forward_inputs is None:
            return None, None
        next_running = forward_inputs.pop('running')
        inputs = forward_inputs['inputs']
        logger.debug(f'Sending forward inputs: {inputs.log_info()}')
        if logger.level <= logging.DEBUG:
            session_ids = [seq.session_id for seq in next_running]
            logger.debug(f'Forward session_ids: {session_ids}')
        self.next_is_prefill = inputs.is_decoding
        await self.executor.forward_async(forward_inputs)
        self.forward_inputs = forward_inputs
        return forward_inputs, next_running

    async def send_next_inputs(self):
        prefill = self.do_prefill()
        return await self._send_next_inputs_impl(prefill)

    async def prefetch_next_inputs(self):
        enable = False
        scheduler = self.scheduler
        prefill = self.do_prefill()
        if prefill:
            enable = True
        else:
            num_ready = scheduler.num_ready()
            is_decoding = self.forward_inputs['inputs'].is_decoding
            running_threshold = (self.config.max_batches // 4) if is_decoding or self.spec_decoding else 0

            if num_ready > running_threshold:
                enable = True

        if enable:
            # send next forward
            logger.debug('Prefetching next forward inputs.')
            return await self._send_next_inputs_impl(prefill, True)
        else:
            return None, None


def build_inputs_maker(engine: 'Engine'):
    """Build inputs makers."""
    config = InputsMakerConfig.from_engine(engine)
    return InputsMakerAsync(
        executor=engine.executor,
        scheduler=engine.scheduler,
        adapter_manager=engine.adapter_manager,
        engine_strategy=engine.engine_strategy,
        sampling_strategy=engine.sampling_strategy,
        model_agent_strategy=engine.model_agent_strategy,
        config=config,
    )
