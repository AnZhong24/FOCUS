# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager, get_step_ctx_manager
from lmdeploy.pytorch.nn import ApplyRotaryEmb, Attention, RMSNorm, SiluAndMul, build_rotary_embedding_from_config
from lmdeploy.pytorch.nn.linear import (build_down_linear, build_gateup_linear, build_o_proj, build_qkv_proj,
                                        build_rowwise_linear)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match attention heads."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class SDARAttention(nn.Module):
    """attention."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 focus_enabled: bool = False,
                 focus_max_batch_size: Optional[int] = 0,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
        num_replicate_kv_heads = getattr(config, 'num_replicate_key_value_heads', 1)
        # packed qkv
        # Qwen3 uses 'config.attention_bias = False' for q/k/o projections
        self.qkv_proj = build_qkv_proj(hidden_size,
                                       num_q_heads=num_heads,
                                       num_kv_heads=num_key_value_heads,
                                       head_size=head_dim,
                                       bias=config.attention_bias,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device,
                                       num_replicate_kv_heads=num_replicate_kv_heads)

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()
        dllm_block_length = config.dllm_block_length

        # attention
        self.attn_fwd = Attention(
            num_heads,
            head_dim,
            num_kv_heads=num_key_value_heads,
            v_head_size=head_dim,
            sliding_window=config.sliding_window,
            block_sparse_size=dllm_block_length,
        )
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = max(1, num_heads // num_key_value_heads)
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        # o_proj
        self.o_proj = build_o_proj(num_heads * head_dim,
                                   hidden_size,
                                   bias=config.attention_bias,
                                   quant_config=quantization_config,
                                   dtype=dtype,
                                   device=device,
                                   is_tp=True)

        # q, k norm
        self.q_norm = RMSNorm(head_dim, config.rms_norm_eps, dtype=dtype, device=device)
        self.k_norm = RMSNorm(head_dim, config.rms_norm_eps, dtype=dtype, device=device)
        # Pre-allocate focus buffers when FOCUS is enabled so later calls
        # can slice from a fixed (max_batch_size, dllm_block_length, ...) view.
        if focus_enabled:
            # q/k related buffers use the same dtype as attention inputs.
            self._focus_delta_buffer = torch.zeros((focus_max_batch_size, dllm_block_length),
                                                    dtype=dtype, 
                                                    device=device)
            self._focus_blockpos_buffer = torch.zeros((focus_max_batch_size, dllm_block_length), 
                                                       dtype=torch.long, 
                                                       device=device)
            self._focus_qpad_buffer = torch.zeros((focus_max_batch_size, dllm_block_length, num_heads, head_dim),
                                                    dtype=dtype, 
                                                    device=device)
            self._focus_kpad_buffer = torch.zeros((focus_max_batch_size, dllm_block_length, num_key_value_heads, head_dim),
                                                    dtype=dtype, 
                                                    device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        focus_mask = None
        context = self._get_context()
        focus_active = bool(context.focus_enabled() and context.is_decoding)
        if focus_active:
            ctx_meta = getattr(context, 'attn_metadata', None)
            if ctx_meta is not attn_metadata:
                attn_metadata = ctx_meta

        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.qkv_proj.split_qkv(qkv_states)

        # apply q, k norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # apply rotary embedding
        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        k_cache = past_key_value[0]
        v_cache = past_key_value[1]
        has_scales = len(past_key_value) > 2
        k_scales = None if not has_scales else past_key_value[2]
        v_scales = None if not has_scales else past_key_value[3]
        focus_fill_only = focus_active and self.layer_idx == 1
        if focus_active and self.layer_idx == 0:
            self._compute_focus_importance(context, query_states, key_states)
        elif focus_fill_only:
            # Preserve the original ragged view for KV fill before pruning.
            self.attn_fwd.forward_only_fill_kv(key_states, value_states, k_cache, v_cache, attn_metadata,
                                               k_scales_zeros=k_scales, v_scales_zeros=v_scales)
            query_states, key_states, value_states, hidden_states, focus_mask = \
                self._apply_focus_pruning(context, hidden_states, query_states, key_states, value_states)
            attn_metadata = context.attn_metadata

        # attention
        if focus_fill_only:
            attn_output = self.attn_fwd.forward_only_attention(
                query_states,
                k_cache,
                v_cache,
                attn_metadata,
                k_scales_zeros=k_scales,
                v_scales_zeros=v_scales,
                inplace=True,
            )
        else:
            attn_output = self.attn_fwd(
                query_states,
                key_states,
                value_states,
                k_cache,
                v_cache,
                attn_metadata,
                k_scales_zeros=k_scales,
                v_scales_zeros=v_scales,
                inplace=True,
            )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        # o proj
        attn_output = self.o_proj(attn_output)
        return attn_output, focus_mask

    def _get_context(self) -> StepContext:
        mgr = get_step_ctx_manager()
        return mgr.current_context()

    def _compute_focus_importance(self, context: StepContext, query_states: torch.Tensor, key_states: torch.Tensor):
        view = context.focus_view
        mask_indices = getattr(view, 'processing_mask_global_indices', None)
        mask_indptr = getattr(view, 'processing_mask_indptr', None)
        proc_view = context.processing_indices
        device = query_states.device
        mask_indices = mask_indices.to(device=device)
        max_mask_len = getattr(view, 'processing_mask_max_len', None)
        importance_flat = self._calc_focus_importance_ragged(query_states,
                                                             key_states,
                                                             mask_indices,
                                                             mask_indptr,
                                                             max_mask_len)
        num_tokens = proc_view.numel()
        importance = torch.zeros((num_tokens, ), dtype=query_states.dtype, device=device)
        importance.index_copy_(0, mask_indices, importance_flat)
        context.focus_first_layer_scores = importance

    def _calc_focus_importance(self, query_states: torch.Tensor, key_states: torch.Tensor) -> torch.Tensor:
        """Compute per-token importance used by FOCUS."""
        q = query_states.transpose(0, 1)  # [num_heads, seq_len, head_dim]
        k = key_states.transpose(0, 1)  # [num_kv_heads, seq_len, head_dim]
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=0)

        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        pooled_logits = F.max_pool1d(attn_logits, kernel_size=3, stride=1, padding=1)
        attn_weights = torch.softmax(pooled_logits, dim=-1)
        attn_sum = attn_weights.sum(dim=-2)
        importance = attn_sum.sum(dim=0)
        return importance.to(dtype=query_states.dtype)

    def _update_rotary_after_prune(self, context: StepContext, retain_mask: torch.Tensor):
        """Trim cached rotary embeddings to match the retained tokens."""
        rotary = getattr(context, 'rotary_pos_emb', None)
        cos, sin = rotary
        keep_idx = torch.arange(retain_mask.numel(), device=retain_mask.device)[retain_mask]
        cos = cos.index_select(0, keep_idx)
        sin = sin.index_select(0, keep_idx)
        context.rotary_pos_emb = (cos, sin)

    def _calc_focus_importance_ragged(self,
                                      query_states: torch.Tensor,
                                      key_states: torch.Tensor,
                                      mask_indices: torch.Tensor,
                                      mask_indptr: torch.Tensor,
                                      max_mask_len: int) -> torch.Tensor:
        """Vectorized importance computation over ragged sequences."""
        device = query_states.device
        # Gather masked tokens in ragged order.
        gathered_query = query_states.index_select(0, mask_indices)
        gathered_key = key_states.index_select(0, mask_indices)
        mask_offsets = mask_indptr.to(device=device, dtype=torch.long)
        lengths = mask_offsets[1:] - mask_offsets[:-1]
        num_seq = lengths.size(0)
        if max_mask_len <= 0:
            return query_states.new_zeros((0, ), dtype=query_states.dtype)
        seq_offsets = mask_offsets[:-1]
        seq_ids = torch.repeat_interleave(torch.arange(num_seq, device=device), lengths)
        rel_pos = torch.arange(mask_indices.numel(), device=device) - seq_offsets.repeat_interleave(lengths)
        # Pad sequences to batched tensor for attention computation.
        padded_q = self._get_focus_padding_buffer(
            '_focus_qpad_buffer',
            (num_seq, max_mask_len, gathered_query.size(1), gathered_query.size(2)),
        )
        padded_k = self._get_focus_padding_buffer(
            '_focus_kpad_buffer',
            (num_seq, max_mask_len, gathered_key.size(1), gathered_key.size(2)),
        )
        padded_q[seq_ids, rel_pos] = gathered_query
        padded_k[seq_ids, rel_pos] = gathered_key
        valid_mask = torch.arange(max_mask_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        q = padded_q.transpose(1, 2)  # (num_seq, num_heads, max_len, head_dim)
        k = padded_k.transpose(1, 2)  # (num_seq, num_kv_heads, max_len, head_dim)
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_logits = attn_logits.masked_fill(~valid_mask[:, None, None, :], float('-inf'))
        logits_flat = attn_logits.reshape(-1, max_mask_len)
        query_mask = valid_mask[:, None, :].expand(num_seq, self.num_attention_heads, max_mask_len).reshape(-1)
        key_mask_rows = valid_mask[:, None, None, :].expand(num_seq, self.num_attention_heads, max_mask_len, max_mask_len)
        key_mask_rows = key_mask_rows.reshape(-1, max_mask_len)
        row_probs = logits_flat.new_zeros((logits_flat.size(0), max_mask_len), dtype=query_states.dtype)
        valid_logits = logits_flat[query_mask]
        pooled = F.max_pool1d(valid_logits.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        valid_keys = key_mask_rows[query_mask]
        pooled = pooled.masked_fill(~valid_keys, float('-inf'))
        attn_weights = torch.softmax(pooled, dim=-1)
        row_probs[query_mask] = attn_weights.to(dtype=query_states.dtype)
        row_probs = row_probs.view(num_seq, self.num_attention_heads, max_mask_len, max_mask_len)
        importance = row_probs.sum(dim=2)
        importance = importance * valid_mask[:, None, :]
        importance = importance.sum(dim=1)
        return importance[valid_mask]

    def _get_focus_padding_buffer(self,
                                  attr_name: str,
                                  shape: Tuple[int, ...]) -> torch.Tensor:
        """Return a zeroed slice of a reusable buffer for padded tensors."""
        buf = getattr(self, attr_name, None)
        view = buf
        for dim, length in enumerate(shape):
            view = view.narrow(dim, 0, length)
        view.zero_()
        return view

    def _apply_focus_pruning(
        self,
        context: StepContext,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        prev_scores = getattr(context, 'focus_first_layer_scores', None)
        view = context.focus_view
        device = query_states.device
        proc_indices = context.processing_indices
        mask_globals_cpu = getattr(view, 'processing_mask_global_indices', None)
        orig_q_lens = context.q_seqlens.detach().clone()
        block_progress_view = view.block_progress
        mask_indptr = getattr(view, 'processing_mask_indptr', None)
        mask_indptr_dev = mask_indptr.to(device=device, dtype=torch.long)
        mask_lengths = (mask_indptr_dev[1:] - mask_indptr_dev[:-1])
        avg_tokens = view.avg_decoded_tokens.to(device=device)
        targets = self._compute_focus_targets(mask_lengths, avg_tokens, context.focus_params)
        should_prune = (targets > 0) & (mask_lengths > targets)

        retain_processing_mask = torch.ones_like(proc_indices, dtype=torch.bool, device=device)

        total_masked = getattr(view, 'processing_mask_total', None)
        should_prune_any = getattr(view, 'processing_mask_prunable', None)
        if total_masked > 0 and should_prune_any:
            mask_globals = mask_globals_cpu.to(device=device, dtype=torch.long)
            max_mask_len = getattr(view, 'processing_mask_max_len', None)
            mask_importance_flat = self._calc_focus_importance_ragged(query_states,
                                                                      key_states,
                                                                      mask_globals,
                                                                      mask_indptr,
                                                                      max_mask_len)
            num_seq = mask_lengths.size(0)
            seq_ids = torch.repeat_interleave(torch.arange(num_seq, device=device, dtype=torch.long), mask_lengths)
            seq_offsets = mask_indptr_dev[:-1]
            rel_pos = torch.arange(total_masked, device=device, dtype=torch.long)
            rel_pos = rel_pos - seq_offsets.repeat_interleave(mask_lengths)
            padded_shape = (num_seq, max_mask_len)
            prev_indices = mask_globals_cpu.to(prev_scores.device)
            prev_selected = prev_scores.index_select(0, prev_indices).to(device=device,
                                                                          dtype=mask_importance_flat.dtype)
            seq_delta_flat = mask_importance_flat - prev_selected
            padded_delta = self._get_focus_padding_buffer('_focus_delta_buffer', padded_shape)
            padded_delta[seq_ids, rel_pos] = seq_delta_flat
            valid_mask = torch.arange(max_mask_len, device=device, dtype=torch.long).unsqueeze(0)
            valid_mask = valid_mask < mask_lengths.unsqueeze(1)
            token_indices = mask_globals
            seq_block_positions_flat = proc_indices.index_select(0, token_indices)
            padded_block_positions = self._get_focus_padding_buffer('_focus_blockpos_buffer', padded_shape)
            padded_block_positions[seq_ids, rel_pos] = seq_block_positions_flat
            effective_targets = torch.where(should_prune, targets, torch.zeros_like(targets))
            selection_mask = self._select_focus_mask_batch(padded_delta, valid_mask, effective_targets)
            retain_mask = torch.where(should_prune.unsqueeze(1), selection_mask, valid_mask)
            block_progress = block_progress_view.to(device=device, dtype=torch.long)
            retain_mask = self._enforce_focus_rules_batch(padded_block_positions, block_progress, retain_mask,
                                                          valid_mask)
            retain_mask_flat = retain_mask[seq_ids, rel_pos]
            retain_processing_mask[token_indices] = retain_mask_flat

        context.focus_first_layer_scores = None

        retain_idx = torch.arange(retain_processing_mask.size(0), device=device)[retain_processing_mask]
        any_pruned = retain_idx.numel() != retain_processing_mask.size(0)
        if not any_pruned:
            context.update_focus_processed_mask()
            return query_states, key_states, value_states, hidden_states, None

        query_states = query_states.index_select(0, retain_idx)
        key_states = key_states.index_select(0, retain_idx)
        value_states = value_states.index_select(0, retain_idx)
        hidden_states = hidden_states[:, retain_processing_mask, :]

        lengths_device = orig_q_lens.device
        batch_size = orig_q_lens.size(0)
        new_q_lens = torch.zeros_like(orig_q_lens, device=lengths_device, dtype=orig_q_lens.dtype)
        mask_vals = retain_processing_mask.to(dtype=new_q_lens.dtype)
        seq_lengths = orig_q_lens.to(device=mask_vals.device, dtype=torch.long)
        token_seq_ids = torch.repeat_interleave(torch.arange(batch_size, device=mask_vals.device, dtype=torch.long),
                                                seq_lengths)
        seq_sums = torch.zeros(batch_size, dtype=new_q_lens.dtype, device=mask_vals.device)
        seq_sums.scatter_add_(0, token_seq_ids, mask_vals)
        new_q_lens.copy_(seq_sums.to(device=lengths_device, dtype=new_q_lens.dtype))
        new_q_lens = new_q_lens.to(device=context.q_seqlens.device, dtype=context.q_seqlens.dtype)
        new_proc_indices = proc_indices[retain_processing_mask]

        context.update_processing_view(new_proc_indices, new_q_lens)
        context.refresh_attention_metadata()
        context.update_focus_processed_mask()
        mask = retain_processing_mask.to(context.input_ids.device)
        context.input_ids = context.input_ids[:, mask]
        mask = retain_processing_mask.to(context.position_ids.device)
        context.position_ids = context.position_ids[:, mask]
        if context.attention_mask is not None:
            mask = retain_processing_mask.to(context.attention_mask.device)
            context.attention_mask = context.attention_mask[:, mask]
        if context.input_embeddings is not None:
            mask_gpu = retain_processing_mask.to(context.input_embeddings.device)
            context.input_embeddings = context.input_embeddings[:, mask_gpu, :]
        if context.input_embedding_indexing is not None:
            mask_cpu = retain_processing_mask.to(context.input_embedding_indexing.device)
            context.input_embedding_indexing = context.input_embedding_indexing[mask_cpu]

        self._update_rotary_after_prune(context, retain_processing_mask)
        return query_states, key_states, value_states, hidden_states, retain_processing_mask

    def _compute_focus_targets(self, mask_lengths: torch.Tensor, avg_tokens: torch.Tensor,
                               focus_params) -> torch.Tensor:
        avg_tokens = torch.maximum(avg_tokens, torch.ones_like(avg_tokens))
        retain = torch.ceil(avg_tokens * focus_params.focus_alpha).to(mask_lengths.dtype)
        retain = torch.clamp(retain, min=1)
        targets = torch.minimum(mask_lengths, retain)
        targets = torch.where(mask_lengths <= 0, torch.zeros_like(targets), targets)
        return targets

    def _build_ranked_selection(self, scores: torch.Tensor, valid_mask: torch.Tensor, targets: torch.Tensor,
                                descending: bool) -> torch.Tensor:
        device = scores.device
        fill_value = float('-inf') if descending else float('inf')
        scores = scores.masked_fill(~valid_mask, fill_value)
        order = torch.argsort(scores, dim=-1, descending=descending)
        rank_range = torch.arange(scores.size(-1), device=device, dtype=targets.dtype).unsqueeze(0).expand_as(order)
        rank_mask = rank_range < targets.unsqueeze(1)
        selection = torch.zeros(valid_mask.shape, device=device, dtype=torch.int64)
        selection.scatter_(1, order, rank_mask.to(dtype=selection.dtype))
        selection = selection.to(dtype=torch.bool)
        selection &= valid_mask
        return selection

    def _select_dynamic_mask(self, scores: torch.Tensor, valid_mask: torch.Tensor,
                             targets: torch.Tensor) -> torch.Tensor:
        base_selection = self._build_ranked_selection(scores, valid_mask, targets, descending=True)
        masked_scores = scores.masked_fill(~valid_mask, 0.0)
        counts = valid_mask.sum(dim=-1).clamp(min=1).to(masked_scores.dtype)
        mean = masked_scores.sum(dim=-1) / counts
        diff = (masked_scores - mean.unsqueeze(1)) * valid_mask
        variance = diff.pow(2).sum(dim=-1) / counts
        std = torch.sqrt(variance)
        threshold = mean + std
        candidate_mask = (scores >= threshold.unsqueeze(1)) & valid_mask
        candidate_counts = candidate_mask.sum(dim=-1).to(targets.dtype)
        positive_targets = targets > 0
        use_threshold = positive_targets & (candidate_counts >= targets)
        selection = torch.where(use_threshold.unsqueeze(1), candidate_mask, base_selection)
        selection &= valid_mask
        return selection

    def _select_focus_mask_batch(self,
                                 delta: torch.Tensor,
                                 valid_mask: torch.Tensor,
                                 targets: torch.Tensor) -> torch.Tensor:
        max_counts = valid_mask.sum(dim=-1).to(targets.dtype)
        targets = torch.clamp(targets, min=0)
        targets = torch.minimum(targets, max_counts)
        positive = targets > 0
        clamped = torch.where(positive, torch.clamp(targets, min=1), targets)
        return self._select_dynamic_mask(delta, valid_mask, clamped)

    def _enforce_focus_rules_batch(self, block_positions: torch.Tensor, block_progress: torch.Tensor,
                                   retain_mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        device = retain_mask.device
        block_progress = block_progress.to(device=device, dtype=torch.long)
        adjacency = (block_positions[:, 1:] - block_positions[:, :-1]) == 1
        adjacency = adjacency & valid_mask[:, 1:] & valid_mask[:, :-1]
        adjust = adjacency & retain_mask[:, 1:] & (~retain_mask[:, :-1])
        retain_mask[:, :-1] |= adjust
        retain_valid = retain_mask & valid_mask
        no_keep = (~retain_valid).all(dim=-1)
        if torch.any(no_keep):
            retain_mask[no_keep] = valid_mask[no_keep]
            retain_valid = retain_mask & valid_mask
        safe_positions = block_positions.masked_fill(~retain_valid, -1)
        rightmost = safe_positions.max(dim=-1).values
        evicted_before = (block_positions < rightmost.unsqueeze(1)) & (~retain_mask) & valid_mask
        progress = block_progress.unsqueeze(1).to(dtype=block_positions.dtype)
        is_unprocessed = block_positions > progress
        need_keep = is_unprocessed & evicted_before
        retain_mask |= need_keep
        return retain_mask


class SDARMLP(nn.Module):
    """mlp."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        # gate up
        self.gate_up_proj = build_gateup_linear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.down_proj = build_down_linear(config.intermediate_size,
                                           config.hidden_size,
                                           bias=False,
                                           quant_config=quantization_config,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=True)

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class SDARDecoderLayer(nn.Module):
    """Decode layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 focus_enabled: bool = False,
                 focus_max_batch_size: Optional[int] = 0,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.self_attn = SDARAttention(config,
                                       layer_idx=layer_idx,
                                       focus_enabled=focus_enabled,
                                       focus_max_batch_size=focus_max_batch_size,
                                       dtype=dtype,
                                       device=device)

        # build MLP
        self.mlp = SDARMLP(config, dtype=dtype, device=device)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps,
                                                quant_config=quantization_config,
                                                dtype=dtype,
                                                device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
    ):

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        hidden_states, focus_mask = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )
        if focus_mask is not None:
            residual = residual[:, focus_mask, :]

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class SDARModel(nn.Module):
    """model."""

    def __init__(self,
                 config: PretrainedConfig,
                 focus_enabled: bool = False,
                 focus_max_batch_size: Optional[int] = 0,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)

        # build all decode layers
        self.layers = nn.ModuleList([
            SDARDecoderLayer(config,
                             layer_idx,
                             focus_enabled=focus_enabled,
                             focus_max_batch_size=focus_max_batch_size,
                             dtype=dtype,
                             device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

        # build rotary embedding
        self.rotary_emb = build_rotary_embedding_from_config(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """Rewrite of forward."""

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # rotary embedding
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)
        context = get_step_ctx_manager().current_context()
        if context is not None:
            context.rotary_pos_emb = rotary_pos_emb

        # decoding
        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            if context is not None:
                rotary_override = getattr(context, 'rotary_pos_emb', None)
                if rotary_override is not None:
                    rotary_pos_emb = rotary_override
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embed_tokens


class SDARForCausalLM(nn.Module, CudaGraphMixin):
    """ModelForCausalLM."""

    packed_modules_mapping = {
        'qkv_proj': [
            'q_proj',
            'k_proj',
            'v_proj',
        ],
        'gate_up_proj': [
            'gate_proj',
            'up_proj',
        ],
    }

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        dllm_cfg = ctx_mgr.build_ctx.dllm_config
        if dllm_cfg is not None:
            config.dllm_block_length = dllm_cfg.block_length
        focus_enabled = bool(dllm_cfg is not None and getattr(dllm_cfg, 'enable_focus', False))
        focus_max_batch_size = getattr(ctx_mgr.build_ctx, 'max_batch_size', None)
        # build model
        self.model = SDARModel(config,
                               focus_enabled=focus_enabled,
                               focus_max_batch_size=focus_max_batch_size,
                               dtype=dtype,
                               device=device)
        # build lm_head
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        """Model forward, return logits."""
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.lm_head(hidden_states)

    def update_weights(self):
        """Update weights."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        # process vision embeddings
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:, vision_embedding_indexing, :] = vision_embeddings.to(inputs_embeds)

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""
        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)
