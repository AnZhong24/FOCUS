# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.model_inputs import ModelInputs

from ..base.model_inputs import ModelInputsStrategy, make_dummy_inputs


def _next_power_of_two(val: int) -> int:
    """Return next power of two."""
    if val <= 1:
        return 1
    return 1 << (val - 1).bit_length()


def _build_ragged_metadata_for_warmup(
    batch_size: int,
    block_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    device: str,
):
    """Build ragged delayed-cache metadata for warmup."""
    kv_group_num = max(1, num_attention_heads // num_key_value_heads)
    max_q = block_size
    heads_per_req_max = kv_group_num * max_q
    # Use a reasonable default block_h for warmup
    block_h = max(16, min(64, _next_power_of_two(heads_per_req_max)))
    heads_per_req = block_size * kv_group_num
    tiles_per_seq = (heads_per_req + block_h - 1) // block_h
    # All sequences have the same number of tiles
    seq_tile_offsets = torch.arange(batch_size, dtype=torch.int32, device=device) * tiles_per_seq
    # tile_to_seq maps each tile to its sequence id
    tile_to_seq = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int32, device=device),
        tiles_per_seq,
    )
    return tile_to_seq, seq_tile_offsets


class DLLMModelInputsStrategy(ModelInputsStrategy):

    def __init__(
        self,
        block_size: int,
        enable_delayed_cache: bool = False,
        enable_focus: bool = False,
        num_attention_heads: int = 1,
        num_key_value_heads: int = 1,
    ):
        self.block_size = block_size
        self.enable_delayed_cache = enable_delayed_cache
        self.enable_focus = enable_focus
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

    def make_dummy(self,
                   batch_size: int,
                   is_decoding: bool,
                   device: str = 'cpu',
                   dummy_block_id: int = 0,
                   vocab_size: int = 1) -> ModelInputs:
        """Create dummy model inputs."""
        inputs = make_dummy_inputs(batch_size,
                                   max_q_seqlen=self.block_size,
                                   is_decoding=is_decoding,
                                   device=device,
                                   dummy_block_id=dummy_block_id,
                                   vocab_size=vocab_size)
        # For delayed cache warmup, set processing_indices and processing_q_lens
        # so that use_delayed_cache=True during warmup graph capture.
        if self.enable_delayed_cache and is_decoding:
            # processing_indices should be relative indices within each sequence,
            # repeated for each sequence in the batch. For warmup with full
            # block_size tokens per sequence, use [0, 1, ..., block_size-1] tiled.
            per_seq_indices = torch.arange(self.block_size, dtype=torch.long, device=device)
            inputs.processing_indices = per_seq_indices.repeat(batch_size)
            inputs.processing_q_lens = torch.full((batch_size,), self.block_size, dtype=torch.long, device=device)
            inputs.processing_max_q_len = self.block_size
            # Build ragged tile metadata for delayed cache
            tile_to_seq, seq_tile_offsets = _build_ragged_metadata_for_warmup(
                batch_size=batch_size,
                block_size=self.block_size,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                device=device,
            )
            inputs.ragged_tile_to_seq = tile_to_seq
            inputs.ragged_seq_tile_offsets = seq_tile_offsets
            # FOCUS-only warmup metadata. FOCUS runs (and evicts) tokens before
            # CUDA graph capture, so the warmup path also needs a valid (even if
            # degenerate) focus view to avoid hitting None attributes.
            if self.enable_focus:
                total_tokens = batch_size * self.block_size
                inputs.focus_block_progress = torch.zeros(batch_size, dtype=torch.int32).pin_memory()
                inputs.focus_avg_tokens = torch.full((batch_size, ), float(self.block_size), dtype=torch.float32).pin_memory()
                inputs.focus_mask_global_indices = torch.arange(total_tokens, dtype=torch.long, device=device)
                inputs.focus_mask_seq_offsets = torch.arange(0,
                                                            total_tokens + 1,
                                                            step=self.block_size,
                                                            dtype=torch.int32).pin_memory()
        return inputs
