# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Iterable, List, Optional, Tuple
import weakref

import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager, get_step_ctx_manager
from lmdeploy.pytorch.nn import ApplyRotaryEmb, Attention, RMSNorm, SiluAndMul, build_rotary_embedding_from_config
from lmdeploy.pytorch.nn.linear import (build_down_linear, build_gateup_linear, build_o_proj, build_qkv_proj,
                                        build_rowwise_linear)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
from lmdeploy.pytorch.kernels.cuda.focus import (focus_compact_states, focus_compute_targets,
                                                 focus_importance_ragged, focus_select_and_enforce_ragged)

from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin


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
        self._keep_tokens_host: Optional[torch.Tensor] = None
        self._keep_tokens_event: Optional[torch.cuda.Event] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
        residual: Optional[torch.Tensor] = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        updated_residual = residual
        context = self._get_context()
        focus_active = bool(context.focus_enabled() and context.is_decoding)
        if focus_active:
            attn_metadata = context.attn_metadata

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
        if focus_active and self.layer_idx == 0 and context.focus_view.processing_mask_evictable:
            # torch.cuda.nvtx.range_push("self._compute_focus_importance")
            self._compute_focus_importance(context, query_states, key_states)
            # torch.cuda.nvtx.range_pop()
        elif focus_fill_only:
            if context.focus_view.processing_mask_evictable:
                # For better overlap.
                new_q_lens = torch.zeros_like(context.q_seqlens)
                # torch.cuda.nvtx.range_push("self._prepare_focus_eviction")
                retain_processing_mask = self._prepare_focus_eviction(context, query_states, key_states)
                # torch.cuda.nvtx.range_pop()
                # Preserve the original ragged view for KV fill before eviction.
                self.attn_fwd.forward_only_fill_kv(key_states, value_states, k_cache, v_cache, attn_metadata,
                                                   k_scales_zeros=k_scales, v_scales_zeros=v_scales)
                # torch.cuda.nvtx.range_push("self._apply_focus_eviction")
                (query_states, key_states, value_states, hidden_states, updated_residual,
                    new_proc_indices, new_q_lens, new_q_lens_host) = self._apply_focus_eviction(
                    context,
                    hidden_states,
                    query_states,
                    key_states,
                    value_states,
                    updated_residual,
                    retain_processing_mask,
                    new_q_lens,
                )
                context.update_processing_view(new_proc_indices, new_q_lens, new_q_lens_host)
                attn_metadata = context.attn_metadata
                # torch.cuda.nvtx.range_pop()
            else:
                context.update_focus_progress_only()
                self.attn_fwd.forward_only_fill_kv(key_states, value_states, k_cache, v_cache, attn_metadata,
                                                   k_scales_zeros=k_scales, v_scales_zeros=v_scales)

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
        return attn_output, updated_residual

    def _get_context(self) -> StepContext:
        mgr = get_step_ctx_manager()
        return mgr.current_context()

    def _get_keep_tokens_host(self) -> torch.Tensor:
        if self._keep_tokens_host is None:
            self._keep_tokens_host = torch.empty(1, dtype=torch.int32, device='cpu', pin_memory=True)
        return self._keep_tokens_host

    def _get_keep_tokens_event(self) -> torch.cuda.Event:
        if self._keep_tokens_event is None:
            self._keep_tokens_event = torch.cuda.Event()
        return self._keep_tokens_event

    def _compute_focus_importance(self, context: StepContext, query_states: torch.Tensor, key_states: torch.Tensor):
        view = context.focus_view
        mask_indices = view.processing_mask_global_indices
        mask_indptr = view.processing_mask_indptr
        max_mask_len = view.processing_mask_max_len
        importance_flat = focus_importance_ragged(query_states,
                                                  key_states,
                                                  mask_indices,
                                                  mask_indptr,
                                                  max_mask_len,
                                                  self.num_key_value_groups,
                                                  self.scale)
        num_tokens = context.processing_indices.numel()
        importance = torch.zeros((num_tokens, ), dtype=query_states.dtype, device=query_states.device)
        importance.index_copy_(0, mask_indices, importance_flat)
        context.focus_first_layer_scores = importance

    def _prepare_focus_eviction(
        self,
        context: StepContext,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> torch.Tensor:
        prev_scores = context.focus_first_layer_scores
        view = context.focus_view
        device = query_states.device
        proc_indices = context.processing_indices
        mask_globals = view.processing_mask_global_indices
        mask_indptr = view.processing_mask_indptr
        mask_lengths = view.processing_mask_lengths
        avg_tokens = view.avg_decoded_tokens
        targets = focus_compute_targets(mask_lengths, avg_tokens, context.focus_params.focus_alpha)
        should_evict = (targets > 0) & (mask_lengths > targets)

        retain_processing_mask = torch.ones_like(proc_indices, dtype=torch.bool, device=device)

        total_masked = view.processing_mask_total
        should_evict_any = view.processing_mask_evictable
        if total_masked > 0 and should_evict_any:
            max_mask_len = view.processing_mask_max_len
            mask_importance_flat = focus_importance_ragged(query_states,
                                                           key_states,
                                                           mask_globals,
                                                           mask_indptr,
                                                           max_mask_len,
                                                           self.num_key_value_groups,
                                                           self.scale)
            retain_mask_flat = focus_select_and_enforce_ragged(mask_importance_flat,
                                                               prev_scores,
                                                               mask_globals,
                                                               proc_indices,
                                                               mask_indptr,
                                                               targets,
                                                               should_evict,
                                                               view.block_progress,
                                                               max_mask_len)
            retain_processing_mask[mask_globals] = retain_mask_flat

        keep_tokens_host = self._get_keep_tokens_host()
        keep_tokens_dev = retain_processing_mask.sum(dtype=torch.int32)
        keep_tokens_host.copy_(keep_tokens_dev, non_blocking=True)
        keep_tokens_event = self._get_keep_tokens_event()
        keep_tokens_event.record()
        return retain_processing_mask

    def _apply_focus_eviction(
        self,
        context: StepContext,
        attn_output: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        retain_processing_mask: torch.Tensor,
        new_q_lens: torch.Tensor,
    ):
        keep_tokens_host = self._get_keep_tokens_host()
        keep_tokens_event = self._get_keep_tokens_event()
        if not keep_tokens_event.query():
            keep_tokens_event.synchronize()
        keep_tokens = keep_tokens_host.item()
        rotary_cos, rotary_sin = context.rotary_pos_emb
        (query_states, key_states, value_states, attn_output, new_input_ids, new_position_ids, new_q_lens,
         new_proc_indices, new_rotary, new_residual) = focus_compact_states(
            keep_tokens,
            retain_processing_mask,
            query_states,
            key_states,
            value_states,
            attn_output,
            context.input_ids,
            context.position_ids,
            context.processing_indices,
            context.q_seqlens,
            new_q_lens,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            residual_states=residual,
        )

        new_q_lens_host = context.prepare_processing_view(new_q_lens)
        context.input_ids = new_input_ids
        context.position_ids = new_position_ids
        if context.attention_mask is not None:
            context.attention_mask = context.attention_mask[:, retain_processing_mask]
        if context.input_embeddings is not None:
            context.input_embeddings = context.input_embeddings[:, retain_processing_mask, :]
        if context.input_embedding_indexing is not None:
            mask_cpu = retain_processing_mask.to(context.input_embedding_indexing.device, non_blocking=True)
            context.input_embedding_indexing = context.input_embedding_indexing[mask_cpu]

        context.rotary_pos_emb = new_rotary
        return query_states, key_states, value_states, attn_output, new_residual, new_proc_indices, new_q_lens, new_q_lens_host

    def forward_focus_qkv_and_evict(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
        residual: Optional[torch.Tensor] = None,
    ):
        """Forward through QKV, norms, rotary, and FOCUS eviction for layer 1.

        This is the prefix part that runs eagerly before CUDA graph capture.
        Returns compacted query_states, hidden_states, and residual for the suffix.
        """
        context = self._get_context()
        updated_residual = residual

        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.qkv_proj.split_qkv(qkv_states)

        # apply q, k norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # apply rotary embedding
        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        k_cache = past_key_value[0]
        v_cache = past_key_value[1]
        has_scales = len(past_key_value) > 2
        k_scales = None if not has_scales else past_key_value[2]
        v_scales = None if not has_scales else past_key_value[3]

        if not context.focus_view.processing_mask_evictable:
            context.update_focus_progress_only()
            self.attn_fwd.forward_only_fill_kv(key_states, value_states, k_cache, v_cache, attn_metadata,
                                               k_scales_zeros=k_scales, v_scales_zeros=v_scales)
            return query_states, hidden_states, updated_residual

        # Prepare eviction
        new_q_lens = torch.zeros_like(context.q_seqlens)
        # torch.cuda.nvtx.range_push("self._prepare_focus_eviction")
        retain_processing_mask = self._prepare_focus_eviction(context, query_states, key_states)
        # torch.cuda.nvtx.range_pop()

        # Fill KV cache before eviction
        self.attn_fwd.forward_only_fill_kv(key_states, value_states, k_cache, v_cache, attn_metadata,
                                           k_scales_zeros=k_scales, v_scales_zeros=v_scales)

        # Apply eviction
        # torch.cuda.nvtx.range_push("self._apply_focus_eviction")
        (query_states, key_states, value_states, hidden_states, updated_residual,
            new_proc_indices, new_q_lens, new_q_lens_host) = self._apply_focus_eviction(
            context,
            hidden_states,
            query_states,
            key_states,
            value_states,
            updated_residual,
            retain_processing_mask,
            new_q_lens,
        )
        context.update_processing_view(new_proc_indices, new_q_lens, new_q_lens_host)
        # torch.cuda.nvtx.range_pop()

        return query_states, hidden_states, updated_residual

    def forward_focus_attention(
        self,
        query_states: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
        residual: Optional[torch.Tensor] = None,
    ):
        """Forward through attention and o_proj for layer 1 after FOCUS eviction.

        This is the suffix part that can be captured in CUDA graph.
        """
        k_cache = past_key_value[0]
        v_cache = past_key_value[1]
        has_scales = len(past_key_value) > 2
        k_scales = None if not has_scales else past_key_value[2]
        v_scales = None if not has_scales else past_key_value[3]

        # attention
        attn_output = self.attn_fwd.forward_only_attention(
            query_states,
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
        return attn_output, residual


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
        hidden_states, residual = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
            residual=residual,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states, residual)
        return outputs

    def forward_focus_prefix(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: torch.Tensor,
        attn_metadata: Any = None,
    ):
        """Forward the prefix part of layer 1 with FOCUS.

        Does layernorm, QKV projection, and eviction. Returns compacted states for the suffix.
        This runs eagerly before CUDA graph capture.
        """
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

        query_states, hidden_states, residual = self.self_attn.forward_focus_qkv_and_evict(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
            residual=residual,
        )

        return query_states, hidden_states, residual

    def forward_focus_suffix(
        self,
        query_states: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: torch.Tensor,
        attn_metadata: Any = None,
    ):
        """Forward the suffix part of layer 1 with FOCUS.

        Does attention, o_proj, post-attention layernorm, and MLP.
        This can be captured in CUDA graph.
        """
        hidden_states, residual = self.self_attn.forward_focus_attention(
            query_states=query_states,
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
            residual=residual,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


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

    def forward_focus_prefix(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """Run the prefix part of the first two decoder layers eagerly.

        This helper is used by the CUDA graph runner when FOCUS is enabled.
        FOCUS evicts tokens inside layer 1, so we execute layer 0 fully and
        layer 1's QKV/eviction eagerly, then capture the remaining suffix
        (layer 1's attention + layers 2+) with fixed shapes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (hidden_states, residual, query_states)
            where query_states is the compacted query for layer 1's attention.
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # initial rotary embedding (pre-eviction)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)
        context = get_step_ctx_manager().current_context()
        if context is not None:
            context.rotary_pos_emb = rotary_pos_emb

        residual = None
        # Layer 0: full forward
        hidden_states, residual = self.layers[0](
            hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_values[0],
            residual=residual,
            attn_metadata=attn_metadata,
        )

        # Layer 1: only prefix (QKV projection, norms, rotary, eviction)
        # The attention and o_proj will be done in the suffix (CUDA graph)
        query_states, hidden_states, residual = self.layers[1].forward_focus_prefix(
            hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_values[1],
            residual=residual,
            attn_metadata=attn_metadata,
        )

        return hidden_states, residual, query_states

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
        self._focus_suffix_model: Optional[nn.Module] = None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        # context = get_step_ctx_manager().current_context()
        # if context.is_decoding:
        #     print("batch_size:", attn_metadata.q_seqlens.size(0), "num_tokens:", input_ids.size(1))
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

    def forward_focus_prefix(self, **kwargs):
        """Expose SDARModel.forward_focus_prefix."""
        return self.model.forward_focus_prefix(**kwargs)

    def get_focus_suffix_model(self):
        """Build or return a cudagraph-compatible suffix module.

        The suffix runs layer 1's attention/o_proj (using query_states from prefix),
        followed by decoder layers from layer 2 to the end. It inherits
        :class:`CudaGraphMixin` so the CUDA graph runner can allocate and fill
        static buffers.
        """
        if self._focus_suffix_model is None:
            self._focus_suffix_model = SDARPostFocusSuffix(self)
        return self._focus_suffix_model

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


class SDARPostFocusSuffix(nn.Module, CudaGraphMixin):
    """Suffix of SDAR used for CUDA graph after FOCUS eviction.

    This module takes compacted hidden_states/residual/query_states (after layer 1's
    QKV projection and eviction) and runs layer 1's attention + o_proj, followed by
    the remaining decoder layers and final norm. Input tensors are padded to
    ``graph_meta.max_tokens`` so CUDA graph shapes stay fixed.
    """

    def __init__(self, parent: SDARForCausalLM):
        super().__init__()
        # Keep a weak reference to avoid registering the parent as a submodule
        # (which would create a module cycle).
        self._parent_ref = weakref.ref(parent)
        self.ctx_mgr = parent.ctx_mgr
        self.config = parent.config

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        # FOCUS does not need input_ids in the graph; allocate only what the
        # suffix actually consumes to keep buffers light.
        max_batches = graph_meta.max_buffer_batch_size
        max_tokens = graph_meta.max_tokens
        num_blocks = graph_meta.num_blocks
        device = graph_meta.device

        input_buffers = dict()
        input_buffers['position_ids'] = torch.zeros((1, max_tokens), dtype=torch.int64, device=device)
        input_buffers['block_offsets'] = torch.zeros((max_batches, num_blocks), dtype=torch.int32, device=device)
        input_buffers['qkv_lens'] = torch.zeros(3, max_batches, dtype=torch.int32, device=device)

        input_buffers['q_start_loc'] = input_buffers['qkv_lens'][0]
        input_buffers['q_seqlens'] = input_buffers['qkv_lens'][1]
        input_buffers['kv_seqlens'] = input_buffers['qkv_lens'][2]
        input_buffers['local_adapter_ids'] = torch.zeros(max_batches, dtype=torch.int64, device=device)
        input_buffers['fill_seqlens'] = torch.zeros(max_batches, dtype=torch.int64, device=device)

        input_buffers['cu_seqlens'] = torch.zeros(2, max_batches + 1, dtype=torch.int32, device=device)
        input_buffers['cu_seqlens_q'] = input_buffers['cu_seqlens'][0]
        input_buffers['cu_seqlens_k'] = input_buffers['cu_seqlens'][1]

        input_buffers['processing_indices'] = torch.zeros(max_tokens, dtype=torch.long, device=device)
        max_tiles_per_seq = self._get_max_tiles_per_seq(graph_meta)
        graph_meta.max_tiles_per_seq = max_tiles_per_seq
        total_tiles = max_tiles_per_seq * max_batches
        input_buffers['seq_tile_offsets'] = torch.zeros(max_batches, dtype=torch.int32, device=device)
        input_buffers['tile_to_seq'] = torch.zeros(total_tiles, dtype=torch.int32, device=device)

        # Allocate buffers for prefix outputs.
        hs = kwargs.get('hidden_states', None)
        res = kwargs.get('residual', None)
        dtype = (hs.dtype if hs is not None else res.dtype if res is not None else getattr(self.config, 'torch_dtype',
                                                                                          torch.float16))
        hidden_size = self.config.hidden_size
        input_buffers['hidden_states'] = torch.zeros((1, max_tokens, hidden_size), dtype=dtype, device=device)
        input_buffers['residual'] = torch.zeros((1, max_tokens, hidden_size), dtype=dtype, device=device)

        # Allocate buffer for query_states (for layer 1's attention in CUDA graph).
        input_buffers['query_states'] = torch.zeros((max_tokens, self.config.num_attention_heads, self.config.head_dim),
                                                     dtype=dtype, device=device)

        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, hidden_states: torch.Tensor,
                               residual: torch.Tensor, query_states: torch.Tensor = None,
                               input_ids: torch.Tensor = None,
                               position_ids: torch.Tensor = None, past_key_values=None, attn_metadata=None,
                               inputs_embeds: torch.Tensor = None, **kwargs):
        """Fill static buffers for the FOCUS suffix capture."""
        block_offsets: torch.Tensor = attn_metadata.block_offsets
        q_start_loc: torch.Tensor = attn_metadata.q_start_loc
        q_seqlens: torch.Tensor = attn_metadata.q_seqlens
        kv_seqlens: torch.Tensor = attn_metadata.kv_seqlens
        input_buffers = graph_meta.input_buffers

        batch_size, num_blocks = block_offsets.size()
        num_tokens = hidden_states.size(1)
        # Fill metadata buffers (no input_ids copy needed for FOCUS).
        input_buffers['position_ids'][:, :num_tokens] = position_ids
        input_buffers['block_offsets'][:batch_size, :num_blocks] = block_offsets

        qkv = torch.stack((q_start_loc, q_seqlens, kv_seqlens))
        input_buffers['qkv_lens'].zero_()
        input_buffers['q_seqlens'].fill_(graph_meta.max_tokens // graph_meta.max_batchs)
        input_buffers['qkv_lens'][:, :batch_size] = qkv
        input_buffers['cu_seqlens_q'][1:batch_size + 1] = input_buffers['q_seqlens'][:batch_size].cumsum(0)
        input_buffers['cu_seqlens_k'][1:batch_size + 1] = input_buffers['kv_seqlens'][:batch_size].cumsum(0)
        input_buffers['fill_seqlens'][:batch_size] = q_seqlens

        new_inputs = dict(
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
        )

        # create inputs
        attn_metadata.block_offsets = input_buffers['block_offsets']
        attn_metadata.q_start_loc = input_buffers['q_start_loc']
        attn_metadata.q_seqlens = input_buffers['q_seqlens']
        attn_metadata.kv_seqlens = input_buffers['kv_seqlens']
        attn_metadata.cu_seqlens_q = input_buffers['cu_seqlens_q']
        attn_metadata.cu_seqlens_k = input_buffers['cu_seqlens_k']
        attn_metadata.fill_seqlens = input_buffers['fill_seqlens']

        proc_indices = attn_metadata.processing_indices
        proc_buf = input_buffers['processing_indices']
        proc_buf[:proc_indices.numel()] = proc_indices
        attn_metadata.processing_indices = proc_buf

        tile_to_seq = attn_metadata.tile_to_seq
        seq_tile_offsets = attn_metadata.seq_tile_offsets
        tile_buf = input_buffers['tile_to_seq']
        tile_buf.zero_()
        tile_buf[:tile_to_seq.numel()] = tile_to_seq
        offset_buf = input_buffers['seq_tile_offsets']
        offset_buf.zero_()
        offset_buf[:seq_tile_offsets.numel()] = seq_tile_offsets
        attn_metadata.tile_to_seq = tile_buf
        attn_metadata.seq_tile_offsets = offset_buf

        # Fill prefix outputs.
        hs_buf = input_buffers['hidden_states']
        hs_buf[:, :num_tokens] = hidden_states
        res_buf = input_buffers['residual']
        res_buf[:, :num_tokens] = residual
        # Fill query_states for layer 1's attention.
        qs_buf = input_buffers['query_states']
        qs_buf[:num_tokens] = query_states

        new_inputs['position_ids'] = input_buffers['position_ids']
        new_inputs['hidden_states'] = hs_buf
        new_inputs['residual'] = res_buf
        new_inputs['query_states'] = qs_buf
        new_inputs.update(kwargs)
        return new_inputs

    def forward(
        self,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        hidden_states: torch.Tensor = None,
        residual: torch.Tensor = None,
        query_states: torch.Tensor = None,
        **kwargs,
    ):
        """Run decoder layers from layer 1 onward.

        Runs layer 1's attention suffix (attention + o_proj) plus
        post-attention layernorm and MLP, then proceeds with layers 2+.
        """
        parent = self._parent_ref()
        # Recompute rotary embeddings for the (padded) post-eviction inputs.
        cos, sin = parent.model.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        # Layer 1: run the suffix part (attention + o_proj + post_attn_norm + MLP)
        past_key_value = past_key_values[1]
        hidden_states, residual = parent.model.layers[1].forward_focus_suffix(
            query_states=query_states,
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            residual=residual,
            attn_metadata=attn_metadata,
        )
        # Layers 2+
        for idx in range(2, len(parent.model.layers)):
            past_key_value = past_key_values[idx]
            hidden_states, residual = parent.model.layers[idx](
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )

        hidden_states, _ = parent.model.norm(hidden_states, residual)
        return hidden_states

    # The suffix module reuses parameters from the parent SDARForCausalLM;
    # it does not own weights and should not be passed to the weight loader.
