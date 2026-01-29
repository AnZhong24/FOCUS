import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class Context:
    is_decode: bool = False
    enable_token_eviction: bool = False
    mask_indices: Optional[torch.Tensor] = None
    processing_indices: Optional[torch.Tensor] = None
    unprocessed_positions: Optional[torch.Tensor] = None
    reference_indices: torch.Tensor = torch.tensor([])
    K: Optional[int] = None
    strategy: Optional[str] = None
    first_layer_importance: Optional[torch.Tensor] = None
    eviction_phase: bool = False


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(
    is_decode: bool,
    enable_token_eviction: bool = False,
    mask_indices: Optional[torch.Tensor] = None,
    processing_indices: Optional[torch.Tensor] = None,
    K: Optional[int] = None,
    strategy: Optional[str] = None,
    first_layer_importance: Optional[torch.Tensor] = None,
    eviction_phase: Optional[bool] = None,
    reference_indices: Optional[torch.Tensor] = None,
    unprocessed_positions: Optional[torch.Tensor] = None,
):
    """Update global context shared across SDAR focus utilities."""
    global _CONTEXT
    _CONTEXT.is_decode = is_decode
    _CONTEXT.enable_token_eviction = enable_token_eviction
    if mask_indices is not None:
        _CONTEXT.mask_indices = mask_indices
    if processing_indices is not None:
        _CONTEXT.processing_indices = processing_indices
    if K is not None:
        _CONTEXT.K = K
    if strategy is not None:
        _CONTEXT.strategy = strategy
    if first_layer_importance is not None:
        _CONTEXT.first_layer_importance = first_layer_importance
    if eviction_phase is not None:
        _CONTEXT.eviction_phase = eviction_phase
    if reference_indices is not None:
        _CONTEXT.reference_indices = reference_indices
    if unprocessed_positions is not None:
        _CONTEXT.unprocessed_positions = unprocessed_positions


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()


def same_seeds(seed: int):
    """Seed all RNGs for reproducible SDAR sampling."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply top-k filtering to logits."""
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Apply top-p (nucleus) filtering to logits."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool), -1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask_indices, float('-inf'))


def sample_with_temperature_topk_topp(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
):
    """Sample tokens with temperature, top-k, and top-p filtering."""
    orig_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)

    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)

    inf_mask = torch.isinf(logits)
    nan_mask = torch.isnan(logits)
    logits = torch.where(inf_mask, torch.full_like(logits, 1e10), logits)
    logits = torch.where(nan_mask, torch.full_like(logits, -1e10), logits)

    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    token_prob = torch.gather(probs, -1, token)

    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(block_length: int, steps: int) -> torch.Tensor:
    """Calculate the number of tokens to transfer at each diffusion step."""
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dimensions, used by rotary embeddings."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
):
    """Apply rotary position embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for grouped query attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def get_transfer_index(
    remasking_strategy: str,
    cur_x: torch.Tensor,
    x0: torch.Tensor,
    x0_p: torch.Tensor,
    mask_index: torch.Tensor,
    num_transfer_tokens_for_step: int,
    confidence_threshold: Optional[float] = None,
    eb_threshold: Optional[float] = None,
) -> torch.BoolTensor:
    """Compute boolean transfer_index for remasking strategies."""
    if remasking_strategy == 'sequential':
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        for j in range(cur_x.shape[0]):
            if mask_index[j].any():
                first_mask_index = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                end = first_mask_index + int(num_transfer_tokens_for_step)
                transfer_index[j, first_mask_index:end] = True
            else:
                raise ValueError("No mask tokens found in the current block.")
        return transfer_index

    if remasking_strategy == 'low_confidence_static':
        confidence = torch.where(mask_index, x0_p, -torch.inf)
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        for j in range(confidence.shape[0]):
            _, idx = torch.topk(confidence[j], int(num_transfer_tokens_for_step))
            transfer_index[j, idx] = True
        return transfer_index

    if remasking_strategy == 'low_confidence_dynamic':
        if confidence_threshold is None:
            raise ValueError("confidence_threshold is required for low_confidence_dynamic")
        confidence = torch.where(mask_index, x0_p, -torch.inf)
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        for j in range(confidence.shape[0]):
            high_conf_mask = confidence[j] > confidence_threshold
            num_high_confidence = int(high_conf_mask.sum().item())
            if num_high_confidence >= int(num_transfer_tokens_for_step):
                transfer_index[j] = high_conf_mask
            else:
                _, idx = torch.topk(confidence[j], int(num_transfer_tokens_for_step))
                transfer_index[j, idx] = True
        return transfer_index

    if remasking_strategy == 'entropy_bounded':
        if eb_threshold is None:
            raise ValueError("eb_threshold is required for entropy_bounded")
        eps = 1e-12
        probs_clamped = x0_p.clamp_min(eps)
        entropies = -(probs_clamped * probs_clamped.log()).sum(dim=-1)
        entropies = torch.where(mask_index, entropies, torch.inf)
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        ent_sorted, order = torch.sort(entropies, dim=1, descending=False)
        cumsum = torch.cumsum(ent_sorted, dim=1)
        for j in range(x0_p.shape[0]):
            k_tensor = torch.tensor(eb_threshold, device=x0_p.device)
            k = torch.searchsorted(cumsum[j], k_tensor, right=False).item()
            k = max(1, min(k, int(mask_index[j].sum().item())))
            selected_token_indices = order[j, :k]
            transfer_index[j, selected_token_indices] = True
        return transfer_index

    raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")


__all__ = [
    "Context",
    "get_context",
    "set_context",
    "reset_context",
    "same_seeds",
    "top_k_logits",
    "top_p_logits",
    "sample_with_temperature_topk_topp",
    "get_num_transfer_tokens",
    "rotate_half",
    "apply_rotary_pos_emb",
    "repeat_kv",
    "get_transfer_index",
]
