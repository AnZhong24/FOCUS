import numpy as np
import pytest
import torch

from lmdeploy.pytorch import consts
from lmdeploy.pytorch.config import CacheConfig, DLLMConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import BuildModelContext, ModelInputs, StepContext
from lmdeploy.pytorch.strategies.dllm.model_agent import DLLMModelAgentStrategy
from lmdeploy.pytorch.strategies.dllm.sequence import DelayedCacheState, SubBlockDecodeState


def _make_model_inputs(**kwargs) -> ModelInputs:
    base = dict(
        input_ids=torch.arange(32, dtype=torch.long).reshape(1, 32),
        seq_length=torch.tensor([32], dtype=torch.long),
        history_lengths=torch.tensor([64], dtype=torch.long),
        block_offsets=torch.tensor([[0, 1]], dtype=torch.int32),
        is_decoding=True,
        num_ignored_history=torch.tensor([0], dtype=torch.long),
        max_q_seqlen=32,
        max_kv_seqlen=96,
        sum_kv_seqlen=96,
        processing_indices=torch.arange(8, 16, dtype=torch.long),
        processing_q_lens=torch.tensor([8], dtype=torch.long),
        processing_fill_kv_lens=torch.tensor([80], dtype=torch.long),
        processing_attn_kv_lens=torch.tensor([96], dtype=torch.long),
        processing_target_starts=torch.tensor([8], dtype=torch.long),
        processing_target_ends=torch.tensor([16], dtype=torch.long),
        processing_max_q_len=8,
        ragged_tile_to_seq=torch.tensor([0], dtype=torch.int32),
        ragged_seq_tile_offsets=torch.tensor([0], dtype=torch.int32),
    )
    base.update(kwargs)
    return ModelInputs(**base)


def _make_model_config() -> ModelConfig:
    return ModelConfig(hidden_size=16,
                       num_layers=1,
                       num_attention_heads=4,
                       num_key_value_heads=4,
                       bos_token_id=0,
                       eos_token_id=[1],
                       head_dim=4,
                       model_paradigm='dllm',
                       dllm_block_length=32)


def test_sub_block_decode_state_switches_from_full_block_to_sub_block():
    state = SubBlockDecodeState.new(block_length=32, sub_block_size=8)
    dllm_mask = np.full((32,), consts.DLLM_MASKED, dtype=np.uint8)

    plan = state.get_processing_plan(dllm_mask)
    assert plan.full_block is True
    assert (plan.target_start, plan.target_end) == (0, 8)
    assert plan.fill_kv_len == 32
    assert plan.attn_kv_len == 32
    assert plan.indices.tolist() == list(range(32))

    dllm_mask[:8] = consts.DLLM_UNMASKED
    plan = state.get_processing_plan(dllm_mask)
    assert plan.full_block is True
    assert (plan.target_start, plan.target_end) == (8, 16)
    assert plan.indices.tolist() == list(range(32))

    dllm_mask[8] = consts.DLLM_UNMASKED
    plan = state.get_processing_plan(dllm_mask)
    assert plan.full_block is False
    assert (plan.target_start, plan.target_end) == (8, 16)
    assert plan.fill_kv_len == 16
    assert plan.attn_kv_len == 32
    assert plan.indices.tolist() == list(range(8, 16))


def test_legacy_delayed_cache_plan_keeps_rightmost_fill_length():
    state = DelayedCacheState.new(block_length=8)
    state.needs_warmup = False
    state.uncached_positions[:] = False
    state.uncached_positions[[1, 4, 6]] = True

    plan = state.get_processing_plan()
    assert plan.full_block is False
    assert plan.indices.tolist() == [1, 4, 6]
    assert plan.fill_kv_len == 7
    assert plan.attn_kv_len == 7


def test_step_context_tracks_separate_attention_and_fill_kv_lengths():
    inputs = _make_model_inputs()
    model_config = _make_model_config()
    cache_config = CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0, num_gpu_blocks=1)
    build_ctx = BuildModelContext(dllm_config=DLLMConfig(block_length=32,
                                                         enable_delayed_cache=True,
                                                         enable_sub_block_cache_reuse=True),
                                  max_batch_size=1)

    context = StepContext.new(inputs, model_config, cache_config, build_ctx=build_ctx)

    assert context.input_ids.tolist() == [[8, 9, 10, 11, 12, 13, 14, 15]]
    assert context.q_seqlens.tolist() == [8]
    assert context.kv_seqlens.tolist() == [96]
    assert context.fill_kv_seqlens.tolist() == [80]
    assert context.attn_metadata.kv_seqlens.tolist() == [96]
    assert context.attn_metadata.fill_kv_seqlens.tolist() == [80]
    assert inputs.processing_attn_kv_lens.tolist() == [96]


def test_model_agent_only_unmasks_within_target_sub_block():
    cfg = DLLMConfig(block_length=32,
                     enable_delayed_cache=True,
                     enable_sub_block_cache_reuse=True,
                     sub_block_size=8)
    agent = DLLMModelAgentStrategy(cfg, dllm_mask_token=0)

    for proc_indices, proc_q_lens in (
        (torch.arange(32, dtype=torch.long), torch.tensor([32], dtype=torch.long)),
        (torch.arange(8, 16, dtype=torch.long), torch.tensor([8], dtype=torch.long)),
    ):
        inputs = _make_model_inputs(processing_indices=proc_indices, processing_q_lens=proc_q_lens)
        dllm_mask = torch.full((32,), consts.DLLM_MASKED, dtype=torch.uint8)
        cached = agent._temporarily_cache_unprocessed(dllm_mask, inputs)
        mask_view = dllm_mask.view(1, 32)
        assert torch.all(mask_view[:, :8] == consts.DLLM_CACHED)
        assert torch.all(mask_view[:, 8:16] == consts.DLLM_MASKED)
        assert torch.all(mask_view[:, 16:] == consts.DLLM_CACHED)
        assert int(cached.sum()) == 24


def test_attention_fill_uses_fill_kv_seqlens():
    pytest.importorskip('triton')
    from lmdeploy.pytorch.backends.cuda.attention.default import TritonAttentionImpl, TritonAttentionMetadata

    captured = {}
    impl = object.__new__(TritonAttentionImpl)
    impl.fill_kv_cache = lambda *args, **kwargs: captured.update(kwargs)

    key = torch.zeros((8, 1, 4), dtype=torch.float16)
    value = torch.zeros((8, 1, 4), dtype=torch.float16)
    metadata = TritonAttentionMetadata(
        is_decoding=True,
        block_offsets=torch.tensor([[0]], dtype=torch.int32),
        q_start_loc=torch.tensor([0], dtype=torch.int32),
        q_seqlens=torch.tensor([8], dtype=torch.int32),
        kv_seqlens=torch.tensor([96], dtype=torch.int32),
        fill_kv_seqlens=torch.tensor([80], dtype=torch.int32),
        processing_indices=torch.arange(8, dtype=torch.long),
        use_delayed_cache=True,
    )

    impl._fill_kv_cache_impl(key, value, key, value, metadata, max_q_seqlen=8)

    assert captured['kv_seq_length'].tolist() == [80]
    assert captured['processing_indices'].tolist() == list(range(8))
