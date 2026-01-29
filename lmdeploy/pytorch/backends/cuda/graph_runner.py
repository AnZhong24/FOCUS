# Copyright (c) OpenMMLab. All rights reserved.
import functools
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.profiler import record_function

from lmdeploy.pytorch.backends.deepep_moe_checker import get_moe_backend
from lmdeploy.pytorch.backends.selector import get_backend
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.pytorch.strategies.base import StrategyFactoryBase
from lmdeploy.utils import get_logger

from ..graph_runner import GraphRunner
from .attention import TritonAttentionMetadata

logger = get_logger('lmdeploy')


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


@functools.lru_cache
def _get_capture_batch_size_impl(max_batches: int):
    """Capture batch size."""
    ret = []
    batch_size = 1
    batch_step = 256
    # power of 2
    while batch_size <= min(batch_step, max_batches):
        ret.append(batch_size)
        batch_size *= 2

    # step
    ret += list(range(batch_size, max_batches + 1, batch_step))

    if max_batches != ret[-1]:
        ret.append(max_batches)
    return ret


def _false(*args, **kwargs):
    """Default value of not support cuda graph."""
    return False


class CUDASingleGraphRunner:
    """Cuda single graph runner."""

    def __init__(
        self,
        model: torch.nn.Module,
        max_batches: int,
        max_tokens: int,
        num_blocks: int,
        is_decoding: bool,
        pool: Tuple[int, int],
        model_config: ModelConfig,
        device: torch.device,
        decode_query_len: int = 1,
        block_size: int = 1,
        use_delayed_cache: bool = False,
        max_buffer_batch_size: int = None,
    ):
        self.model = model
        self.ctx_mgr = model.ctx_mgr
        self.model_config = model_config

        self.meta = CudaGraphMeta(
            max_batchs=max_batches,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            is_decoding=is_decoding,
            device=device,
            input_buffers=dict(),
            output_buffers=dict(),
            vocab_size=self.model_config.vocab_size,
            use_mla_fp8_cache=getattr(self.model_config, 'use_mla_fp8_cache', False),
            use_flash_mla=getattr(self.model_config, 'use_flash_mla', False),
            mla_index_topk=getattr(self.model_config, 'mla_index_topk', None),
            decode_query_len=decode_query_len,
            use_fa3_decoding=model_config.model_paradigm == 'ar_spec',
            block_size=block_size,
            use_delayed_cache=use_delayed_cache,
            max_buffer_batch_size=max_buffer_batch_size,
        )
        self.device = device
        self.max_batches = max_batches
        self.max_tokens = max_tokens
        self.num_blocks = num_blocks
        self.is_decoding = is_decoding
        self.pool = pool
        self._graph: torch.cuda.CUDAGraph = None

    @record_function('capture_cudagraph')
    def capture(self, **kwargs):
        """Capture graph."""
        logger.debug(f'Capturing graph with meta: {self.meta}')
        self.meta.input_buffers = self.model.make_buffers_cudagraph(self.meta, **kwargs)
        padded_kwargs = self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        current_stream = torch.cuda.current_stream()

        # warmup
        warmup_output = self.model(**padded_kwargs)
        warmup_buffers = self.model.make_output_buffers(warmup_output)

        self._graph = torch.cuda.CUDAGraph()
        # unsafe kernel call in other thread might invalid the capture
        # so we set thread_safe capture mode here.
        with torch.cuda.graph(self._graph, pool=self.pool, stream=current_stream, capture_error_mode='thread_local'):
            output = self.model(**padded_kwargs)

        output_buffers = self.model.make_output_buffers(output)
        self.meta.output_buffers = output_buffers
        output = self.model.get_outputs_cudagraph(warmup_buffers, **kwargs)
        return output

    @record_function('forward_cudagraph')
    def forward(self, **kwargs):
        """forward."""
        assert self._graph is not None
        self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        self._graph.replay()
        output_buffers = self.meta.output_buffers
        output = self.model.get_outputs_cudagraph(output_buffers, **kwargs)
        return output

    def __del__(self):
        """del."""
        del self._graph


class CUDAGraphRunner(GraphRunner):
    """Cuda graph runner."""

    def __init__(self, model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                 backend_config: BackendConfig, device: torch.device):
        super().__init__(model, model_config, cache_config, backend_config, device)
        self.max_batches = cache_config.max_batches
        self.max_tokens = cache_config.max_prefill_token_num
        self.num_blocks = cache_config.num_gpu_blocks

        self.enable_graph = self.check_enable_graph()

        self.graph_pool_handle = torch.cuda.graph_pool_handle()
        self._runner_map: Dict[Any, CUDASingleGraphRunner] = dict()
        self.has_try_compile_model: bool = False

        # strategy factory
        build_ctx = model.ctx_mgr.build_ctx
        strategy_factory: StrategyFactoryBase = build_ctx.strategy_factory
        self.cudagraph_strategy = strategy_factory.build_cudagraph_strategy()
        self._cached_capture_batch_sizes: Optional[List[int]] = None

    def check_enable_graph(self):
        """Check enable graph."""
        if self.backend_config.eager_mode:
            return _false

        return getattr(self.model, 'support_cuda_graph', _false)

    def _try_compile_model_once(self):
        if self.has_try_compile_model:
            return

        # TODO: recovery it when torch.compile is stable (should be add a flag to enable it?)
        # if hasattr(self.model, 'compile_model'):
        #     method = getattr(self.model, 'compile_model')
        #     method()

        self.has_try_compile_model = True

    def _round_capture_batch_size(self, batch_size: int):
        cap_sizes = self.get_capture_batch_sizes()
        for size in cap_sizes:
            if size >= batch_size:
                return size
        assert False, f'Unsupported batch_size={batch_size}'

    def _get_capture_tokens(self, batch_size: int, origin_batch_size: int = None):
        """Get capture tokens."""
        capture_func = self._round_capture_batch_size
        strategy = getattr(self, 'cudagraph_strategy', None)
        if strategy is None:
            return capture_func(batch_size)
        if origin_batch_size is None:
            origin_batch_size = batch_size
        return strategy.get_capture_batch_size(target_batch_size=batch_size,
                                               origin_batch_size=origin_batch_size,
                                               capture_func=capture_func)

    def get_graph_key(self, input_ids: torch.Tensor, position_ids: torch.Tensor, past_key_values: List,
                      attn_metadata: TritonAttentionMetadata, inputs_embeds: torch.Tensor, **kwargs):
        """Get graph key."""
        context = self.ctx_mgr.current_context()
        is_decoding = context.is_decoding
        origin_batch_size = attn_metadata.q_seqlens.size(0)
        meta = self.get_meta()
        enable_microbatch = get_step_ctx_manager().current_context().enable_microbatch
        # for draft model to distinguish inputs from target model and itself
        num_tokens = input_ids.size(1)
        use_delayed_cache = context.use_delayed_cache
        is_dllm_paradigm = self.model_config.model_paradigm == 'dllm'

        # DLLM + delayed-cache: key only by rounded total tokens (bucket).
        # (Batch-size rounding via batch*block is for vanilla DLLM only.)
        if is_dllm_paradigm and use_delayed_cache:
            # Use block_size as query_key (same as vanilla DLLM) to enable
            # reusing warmup graphs.
            block_size = self.cudagraph_strategy.block_size
            query_key = block_size

            # Use ceil(num_tokens / block_size) as the batch_size key to allow
            # warmup graphs to be reused across different actual batch sizes.
            # Round to the nearest captured graph size.
            raw_batch_size = math.ceil(num_tokens / block_size)
            batch_size = self._round_capture_batch_size(raw_batch_size)
            enable_microbatch = False
            return (batch_size, is_decoding, enable_microbatch, query_key)

        query_key = num_tokens // origin_batch_size
        if meta.padding_batch_size is None:
            batch_size = self._get_capture_tokens(origin_batch_size, origin_batch_size=origin_batch_size)
        else:
            batch_size = self._get_capture_tokens(meta.padding_batch_size, origin_batch_size=origin_batch_size)
        return (batch_size, is_decoding, enable_microbatch, query_key)

    def _prepare_inputs(self, **kwargs):
        """Prepare inputs."""
        assert 'attn_metadata' in kwargs, 'attn_metadata is required for cudagraph.'
        attn_metadata: TritonAttentionMetadata = kwargs['attn_metadata']
        if not attn_metadata.block_offsets.dtype == torch.int32:
            attn_metadata.block_offsets = attn_metadata.block_offsets.to(torch.int32)
        return kwargs

    def _get_max_tokens(self, graph_key: tuple, input_ids: torch.Tensor, q_seqlens: torch.Tensor, use_delayed_cache: bool):
        max_batches = graph_key[0]
        is_decoding = graph_key[1]
        assert is_decoding
        if use_delayed_cache:
            num_tokens = input_ids.size(1)
            max_tokens = self.cudagraph_strategy.get_capture_token_bucket(num_tokens)
            return max_tokens
        origin_batch_size = q_seqlens.size(0)
        num_tokens = input_ids.size(1)
        return self.cudagraph_strategy.get_max_tokens(max_batches, origin_batch_size, num_tokens)

    def __call__(self, **kwargs):
        """call."""
        if not self.backend_config.eager_mode and get_backend().get_name() == 'cuda':
            self._try_compile_model_once()

        kwargs = self._prepare_inputs(**kwargs)
        enable_graph = self.enable_graph(**kwargs)

        if not enable_graph:
            with record_function('forward_eager'):
                output = self.model(**kwargs)
                return self.model.make_output_buffers(output)

        # FOCUS changes tensor shapes inside layer 1. When enabled, run the
        # prefix (layers 0-1) eagerly to perform token eviction, then capture
        # and replay a CUDA graph for the remaining suffix.
        context = self.ctx_mgr.current_context()
        if context.is_decoding and context.focus_enabled():
            prefix_fn = self.model.forward_focus_prefix
            suffix_fn = self.model.get_focus_suffix_model

            # stage 1: eager prefix (includes FOCUS eviction)
            hidden_states, residual, query_states = prefix_fn(**kwargs)

            # post-eviction inputs live on the context
            post_eviction_input_ids = context.input_ids
            post_eviction_position_ids = context.position_ids
            post_eviction_attn_metadata = context.attn_metadata

            # graph key based on post-eviction (stable) shapes
            suffix_graph_key = self.get_graph_key(
                input_ids=post_eviction_input_ids,
                position_ids=post_eviction_position_ids,
                past_key_values=kwargs['past_key_values'],
                attn_metadata=post_eviction_attn_metadata,
                inputs_embeds=None,
            )
            max_batches = suffix_graph_key[0]
            is_decoding = suffix_graph_key[1]
            use_delayed_cache = getattr(post_eviction_attn_metadata, 'use_delayed_cache', False)
            decode_query_len = suffix_graph_key[3]

            runner = self._runner_map.get(suffix_graph_key, None)
            if runner is None:
                max_tokens = self._get_max_tokens(suffix_graph_key,
                                                  post_eviction_input_ids,
                                                  post_eviction_attn_metadata.q_seqlens,
                                                  use_delayed_cache=use_delayed_cache)
                suffix_model = suffix_fn()
                runner = CUDASingleGraphRunner(
                    suffix_model,
                    max_batches=max_batches,
                    max_tokens=max_tokens,
                    num_blocks=self.num_blocks,
                    is_decoding=is_decoding,
                    pool=self.graph_pool_handle,
                    model_config=self.model_config,
                    device=self.device,
                    decode_query_len=decode_query_len,
                    block_size=self.cache_config.block_size,
                    use_delayed_cache=use_delayed_cache,
                    max_buffer_batch_size=self.max_batches if use_delayed_cache else None,
                )
                output = runner.capture(
                    input_ids=post_eviction_input_ids,
                    position_ids=post_eviction_position_ids,
                    past_key_values=kwargs['past_key_values'],
                    attn_metadata=post_eviction_attn_metadata,
                    inputs_embeds=None,
                    hidden_states=hidden_states,
                    residual=residual,
                    query_states=query_states,
                )
                self._runner_map[suffix_graph_key] = runner
                return output
            else:
                output = runner.forward(
                    input_ids=post_eviction_input_ids,
                    position_ids=post_eviction_position_ids,
                    past_key_values=kwargs['past_key_values'],
                    attn_metadata=post_eviction_attn_metadata,
                    inputs_embeds=None,
                    hidden_states=hidden_states,
                    residual=residual,
                    query_states=query_states,
                )
                return output

        graph_key = self.get_graph_key(**kwargs)
        max_batches = graph_key[0]
        is_decoding = graph_key[1]
        use_delayed_cache = kwargs['attn_metadata'].use_delayed_cache
        decode_query_len = graph_key[3]
        runner = self._runner_map.get(graph_key, None)
        if runner is None:
            max_tokens = self._get_max_tokens(graph_key,
                                              kwargs['input_ids'],
                                              kwargs['attn_metadata'].q_seqlens,
                                              use_delayed_cache=use_delayed_cache)
            runner = CUDASingleGraphRunner(
                self.model,
                max_batches=max_batches,
                max_tokens=max_tokens,
                num_blocks=self.num_blocks,
                is_decoding=is_decoding,
                pool=self.graph_pool_handle,
                model_config=self.model_config,
                device=self.device,
                decode_query_len=decode_query_len,
                block_size=self.cache_config.block_size,
                use_delayed_cache=use_delayed_cache,
                max_buffer_batch_size=self.max_batches if use_delayed_cache else None,
            )
            output = runner.capture(**kwargs)
            self._runner_map[graph_key] = runner
            # SSM would update the state in capture(warmup), replay the graph will leads unexpected state update.
            return output
        else:
            runner = self._runner_map[graph_key]
            output = runner.forward(**kwargs)
            return output

    @record_function('prepare_inputs_for_generation')
    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """Prepare inputs."""

        if get_moe_backend().use_deepep_moe_backend():
            from dlblas.layers.moe.token_dispatcher import DeepEPBuffer, DeepEPMode
            deepep_mode = DeepEPMode.LOW_LATENCY if context.is_decoding else DeepEPMode.NORMAL
            DeepEPBuffer.set_deepep_mode(deepep_mode)

        return self.model.prepare_inputs_for_generation(
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            context=context,
        )

    def reset(self):
        """Remove all graphs to prevent hanging on exit."""
        self._runner_map.clear()

    def update_inputs(self, inputs):
        """Update inputs."""
        if self.backend_config.eager_mode:
            return inputs
        is_decoding = inputs.is_decoding
        dp_meta = inputs.dp_meta
        if is_decoding and dp_meta is not None:
            meta = self.get_meta()
            padding_batch_size = meta.padding_batch_size
            if padding_batch_size is not None:
                tp_size = self.cudagraph_strategy.get_capture_batch_size(
                    target_batch_size=padding_batch_size,
                    origin_batch_size=padding_batch_size,
                    capture_func=self._get_capture_tokens,
                )
                dp_meta.sync_tp_size(tp_size)
        return inputs

    def get_capture_batch_sizes(self) -> List[int]:
        """Capture batch sizes."""
        if self._cached_capture_batch_sizes is not None:
            return self._cached_capture_batch_sizes

        max_batches = self.cache_config.max_batches
        strategy = self.cudagraph_strategy

        # DLLM uses a total-token bucketing rule (see DLLMCudagraphStrategy) that
        # can map many origin batch sizes to intermediate capture batch sizes
        # (e.g. multiples of 8 when block_size=32). Warmup should pre-capture
        # those buckets to avoid late graph captures during benchmarking.
        if self.model_config.model_paradigm == 'dllm':
            capture_func = lambda x: x
            sizes = set()
            for origin_batch_size in range(1, max_batches + 1):
                cap = int(
                    strategy.get_capture_batch_size(
                        target_batch_size=origin_batch_size,
                        origin_batch_size=origin_batch_size,
                        capture_func=capture_func,
                    ))
                if 1 <= cap <= max_batches:
                    sizes.add(cap)
            sizes.add(1)
            sizes.add(max_batches)
            self._cached_capture_batch_sizes = sorted(sizes)
            return self._cached_capture_batch_sizes

        self._cached_capture_batch_sizes = _get_capture_batch_size_impl(max_batches)
        return self._cached_capture_batch_sizes
