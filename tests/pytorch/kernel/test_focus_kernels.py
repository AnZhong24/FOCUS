import pytest
import torch
import torch.nn.functional as F

from lmdeploy.pytorch.kernels.cuda.focus import (_focus_compute_keep_offsets, focus_compact_states,
                                                 focus_compute_targets, focus_importance_ragged,
                                                 focus_select_and_enforce_ragged, focus_update_processing_metadata)


def _reference_ragged_importance(padded_q: torch.Tensor, padded_k: torch.Tensor, lengths: torch.Tensor,
                                 num_key_value_groups: int, scale: float) -> torch.Tensor:
    device = padded_q.device
    num_seq, max_len, _, _ = padded_q.shape
    valid_mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    q = padded_q.transpose(1, 2)
    k = padded_k.transpose(1, 2)
    if num_key_value_groups > 1:
        k = k.repeat_interleave(num_key_value_groups, dim=1)
    attn_logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    attn_logits = attn_logits.masked_fill(~valid_mask[:, None, None, :], float('-inf'))
    logits_flat = attn_logits.reshape(-1, max_len)
    query_mask = valid_mask[:, None, :].expand(num_seq, q.size(1), max_len).reshape(-1)
    key_mask_rows = valid_mask[:, None, None, :].expand(num_seq, q.size(1), max_len, max_len).reshape(-1, max_len)
    row_probs = logits_flat.new_zeros((logits_flat.size(0), max_len))
    if query_mask.any():
        valid_logits = logits_flat[query_mask]
        pooled = F.max_pool1d(valid_logits.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        valid_keys = key_mask_rows[query_mask]
        pooled = pooled.masked_fill(~valid_keys, float('-inf'))
        attn_weights = torch.softmax(pooled, dim=-1)
        row_probs[query_mask] = attn_weights.to(dtype=row_probs.dtype)
    row_probs = row_probs.view(num_seq, q.size(1), max_len, max_len)
    importance = row_probs.sum(dim=2)
    importance = importance * valid_mask[:, None, :]
    importance = importance.sum(dim=1)
    return importance[valid_mask]


def _reference_targets(mask_lengths: torch.Tensor, avg_tokens: torch.Tensor, alpha: float) -> torch.Tensor:
    avg_tokens = torch.maximum(avg_tokens, torch.ones_like(avg_tokens))
    retain = torch.ceil(avg_tokens * alpha).to(mask_lengths.dtype)
    retain = torch.clamp(retain, min=1)
    targets = torch.minimum(mask_lengths, retain)
    targets = torch.where(mask_lengths <= 0, torch.zeros_like(targets), targets)
    return targets


def _reference_ranked_selection(scores: torch.Tensor, valid_mask: torch.Tensor, targets: torch.Tensor,
                                descending: bool) -> torch.Tensor:
    fill_value = float('-inf') if descending else float('inf')
    masked = scores.masked_fill(~valid_mask, fill_value)
    order = torch.argsort(masked, dim=-1, descending=descending)
    rank_range = torch.arange(scores.size(-1), device=scores.device, dtype=targets.dtype).unsqueeze(0).expand_as(order)
    rank_mask = rank_range < targets.unsqueeze(1)
    selection = torch.zeros_like(scores, dtype=torch.bool)
    selection.scatter_(1, order, rank_mask)
    selection &= valid_mask
    return selection


def _reference_focus_select_mask(delta: torch.Tensor, valid_mask: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_counts = valid_mask.sum(dim=-1).to(targets.dtype)
    targets = torch.clamp(targets, min=0)
    targets = torch.minimum(targets, max_counts)
    positive = targets > 0
    clamped = torch.where(positive, torch.clamp(targets, min=1), targets)
    base_selection = _reference_ranked_selection(delta, valid_mask, clamped, True)
    masked_scores = delta.masked_fill(~valid_mask, 0.0)
    counts = valid_mask.sum(dim=-1).clamp(min=1).to(masked_scores.dtype)
    mean = masked_scores.sum(dim=-1) / counts
    diff = (delta - mean.unsqueeze(1)) * valid_mask
    variance = diff.pow(2).sum(dim=-1) / counts
    std = torch.sqrt(variance)
    threshold = mean + std
    candidate_mask = (delta >= threshold.unsqueeze(1)) & valid_mask
    candidate_counts = candidate_mask.sum(dim=-1).to(targets.dtype)
    positive_targets = clamped > 0
    use_threshold = positive_targets & (candidate_counts >= clamped)
    selection = torch.where(use_threshold.unsqueeze(1), candidate_mask, base_selection)
    selection &= valid_mask
    return selection


def _reference_focus_enforce_rules(block_positions: torch.Tensor, block_progress: torch.Tensor,
                                   retain_mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    retain = retain_mask.clone()
    adjacency = (block_positions[:, 1:] - block_positions[:, :-1]) == 1
    adjacency = adjacency & valid_mask[:, 1:] & valid_mask[:, :-1]
    adjust = adjacency & retain[:, 1:] & (~retain[:, :-1])
    retain[:, :-1] |= adjust
    retain_valid = retain & valid_mask
    no_keep = (~retain_valid).all(dim=-1)
    if torch.any(no_keep):
        retain[no_keep] = valid_mask[no_keep]
        retain_valid = retain & valid_mask
    safe_positions = block_positions.masked_fill(~retain_valid, -1)
    rightmost = safe_positions.max(dim=-1).values
    evicted_before = (block_positions < rightmost.unsqueeze(1)) & (~retain) & valid_mask
    progress = block_progress.to(dtype=block_positions.dtype).unsqueeze(1)
    is_unprocessed = block_positions > progress
    need_keep = is_unprocessed & evicted_before
    retain |= need_keep
    return retain


def _reference_focus_select_and_enforce(delta: torch.Tensor, valid_mask: torch.Tensor, targets: torch.Tensor,
                                        should_evict: torch.Tensor, block_positions: torch.Tensor,
                                        block_progress: torch.Tensor) -> torch.Tensor:
    effective_targets = torch.where(should_evict, targets, torch.zeros_like(targets))
    selection = _reference_focus_select_mask(delta, valid_mask, effective_targets)
    retain = torch.where(should_evict.unsqueeze(1), selection, valid_mask)
    return _reference_focus_enforce_rules(block_positions, block_progress, retain, valid_mask)


def _reference_focus_processing_update(new_q_lens: torch.Tensor, new_proc_indices: torch.Tensor,
                                       history_lengths: torch.Tensor, num_ignored_history: torch.Tensor,
                                       block_progress: torch.Tensor):
    q_cumsum = torch.cumsum(new_q_lens, dim=0)
    q_start_loc = q_cumsum - new_q_lens
    valid = new_q_lens > 0
    rightmost = new_proc_indices.new_full((new_q_lens.size(0), ), -1)
    if bool(valid.any().item()):
        end_offsets = q_start_loc + new_q_lens - 1
        safe_offsets = torch.where(valid, end_offsets, end_offsets.new_zeros(end_offsets.size()))
        gathered = new_proc_indices.index_select(0, safe_offsets)
        rightmost = torch.where(valid, gathered, rightmost)
    zeros = torch.zeros_like(rightmost)
    rightmost_plus_one = torch.where(rightmost < 0, zeros, rightmost + 1)
    kv_seqlens = history_lengths + rightmost_plus_one.to(history_lengths.dtype)
    kv_seqlens = kv_seqlens - num_ignored_history.to(history_lengths.dtype)
    cu_q = torch.zeros((new_q_lens.size(0) + 1, ), dtype=torch.int32, device=new_q_lens.device)
    if q_cumsum.numel() > 0:
        cu_q[1:] = q_cumsum.to(torch.int32)
    cu_k = torch.zeros_like(cu_q)
    if kv_seqlens.numel() > 0:
        cu_k[1:] = torch.cumsum(kv_seqlens, dim=0, dtype=torch.int32)
    new_progress = torch.maximum(block_progress, rightmost.to(block_progress.dtype))
    return q_start_loc, cu_q, kv_seqlens, cu_k, new_progress


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FOCUS Triton kernels require CUDA')
def test_focus_ragged_kernel_matches_reference():
    device = torch.device('cuda')
    torch.manual_seed(2)
    num_seq, max_len, num_heads, num_kv_heads, head_dim = 4, 9, 6, 3, 32
    lengths = torch.tensor([9, 5, 2, 0], device=device, dtype=torch.long)
    total_tokens = int(lengths.sum().item())
    query_states = torch.zeros(total_tokens, num_heads, head_dim, device=device, dtype=torch.float16)
    key_states = torch.zeros(total_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    padded_q = torch.zeros(num_seq, max_len, num_heads, head_dim, device=device, dtype=torch.float16)
    padded_k = torch.zeros(num_seq, max_len, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    mask_indices_parts = []
    offset = 0
    indptr_vals = [0]
    for seq_idx, length in enumerate(lengths.tolist()):
        if length > 0:
            seq_q = torch.randn(length, num_heads, head_dim, device=device, dtype=torch.float16)
            seq_k = torch.randn(length, num_kv_heads, head_dim, device=device, dtype=torch.float16)
            query_states[offset:offset + length] = seq_q
            key_states[offset:offset + length] = seq_k
            padded_q[seq_idx, :length] = seq_q
            padded_k[seq_idx, :length] = seq_k
            mask_indices_parts.append(torch.arange(offset, offset + length, device=device, dtype=torch.long))
            offset += length
        indptr_vals.append(offset)
    mask_indices = torch.cat(mask_indices_parts) if mask_indices_parts else torch.empty(0,
                                                                                         dtype=torch.long,
                                                                                         device=device)
    mask_indptr = torch.tensor(indptr_vals, device=device, dtype=torch.int32)
    num_groups = num_heads // num_kv_heads
    fused = focus_importance_ragged(query_states, key_states, mask_indices, mask_indptr, max_len, num_groups,
                                    head_dim**-0.5)
    reference = _reference_ragged_importance(padded_q, padded_k, lengths, num_groups, head_dim**-0.5)
    torch.testing.assert_close(fused.to(dtype=torch.float16), reference, rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FOCUS Triton kernels require CUDA')
def test_focus_target_kernel_matches_reference():
    device = torch.device('cuda')
    torch.manual_seed(42)
    mask_lengths = torch.randint(-2, 10, (32, ), device=device, dtype=torch.int32)
    avg_tokens = torch.rand(32, device=device, dtype=torch.float32) * 8
    alpha = 0.65
    fused = focus_compute_targets(mask_lengths, avg_tokens, alpha)
    reference = _reference_targets(mask_lengths, avg_tokens, alpha)
    torch.testing.assert_close(fused, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FOCUS Triton kernels require CUDA')
def test_focus_keep_offsets_kernel_matches_reference():
    device = torch.device('cuda')
    torch.manual_seed(3)
    lengths = [0, 1, 7, 64, 257]
    for total_tokens in lengths:
        retain_mask = torch.rand(total_tokens, device=device) > 0.45
        fused = _focus_compute_keep_offsets(retain_mask)
        if total_tokens == 0:
            assert fused.numel() == 0
            continue
        mask_i32 = retain_mask.to(dtype=torch.int32)
        reference = torch.cumsum(mask_i32, dim=0, dtype=torch.int32)
        reference -= mask_i32
        reference.masked_fill_(~retain_mask, -1)
        torch.testing.assert_close(fused, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FOCUS Triton kernels require CUDA')
def test_focus_update_processing_metadata_matches_reference():
    device = torch.device('cuda')
    torch.manual_seed(29)
    batch = 7
    new_q_lens = torch.randint(0, 6, (batch, ), device=device, dtype=torch.long)
    if not bool((new_q_lens > 0).any().item()):
        new_q_lens[0] = 1
    total_tokens = int(new_q_lens.sum().item())
    new_proc_indices = torch.randint(0, 32, (total_tokens, ), device=device, dtype=torch.long)
    history_lengths = torch.randint(0, 12, (batch, ), device=device, dtype=torch.long)
    num_ignored_history = torch.randint(0, 3, (batch, ), device=device, dtype=torch.long)
    block_progress = torch.randint(-1, 32, (batch, ), device=device, dtype=torch.int32)
    fused_block_progress = block_progress.clone()
    q_start_loc, cu_q, kv_seqlens, cu_k = focus_update_processing_metadata(new_q_lens.clone(),
                                                                           new_proc_indices.clone(),
                                                                           history_lengths.clone(),
                                                                           num_ignored_history.clone(),
                                                                           fused_block_progress)
    (ref_q_start, ref_cu_q, ref_kv, ref_cu_k,
     ref_progress) = _reference_focus_processing_update(new_q_lens.cpu(), new_proc_indices.cpu(),
                                                        history_lengths.cpu(), num_ignored_history.cpu(),
                                                        block_progress.cpu())
    torch.testing.assert_close(q_start_loc.cpu(), ref_q_start)
    torch.testing.assert_close(cu_q.cpu(), ref_cu_q)
    torch.testing.assert_close(kv_seqlens.cpu(), ref_kv)
    torch.testing.assert_close(cu_k.cpu(), ref_cu_k)
    torch.testing.assert_close(fused_block_progress.cpu(), ref_progress.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FOCUS Triton kernels require CUDA')
def test_focus_update_processing_metadata_empty_batch():
    device = torch.device('cuda')
    new_q_lens = torch.empty(0, device=device, dtype=torch.long)
    new_proc_indices = torch.empty(0, device=device, dtype=torch.long)
    history_lengths = torch.empty(0, device=device, dtype=torch.long)
    num_ignored_history = torch.empty(0, device=device, dtype=torch.long)
    block_progress = torch.empty(0, device=device, dtype=torch.int32)
    fused_block_progress = block_progress.clone()
    q_start_loc, cu_q, kv_seqlens, cu_k = focus_update_processing_metadata(new_q_lens,
                                                                           new_proc_indices,
                                                                           history_lengths,
                                                                           num_ignored_history,
                                                                           fused_block_progress)
    assert q_start_loc.numel() == 0
    assert kv_seqlens.numel() == 0
    torch.testing.assert_close(cu_q, torch.zeros(1, device=device, dtype=torch.int32))
    torch.testing.assert_close(cu_k, torch.zeros(1, device=device, dtype=torch.int32))
    assert fused_block_progress.numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FOCUS Triton kernels require CUDA')
def test_focus_compact_states_matches_reference_gathers():
    device = torch.device('cuda')
    torch.manual_seed(19)
    batch = 3
    q_lens = torch.tensor([5, 3, 4], device=device, dtype=torch.long)
    q_start = torch.cumsum(q_lens, dim=0) - q_lens
    total_tokens = int(q_lens.sum().item())
    num_heads, num_kv_heads, head_dim = 4, 2, 8
    hidden_batch, hidden_dim = 2, 16
    query_states = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float16)
    key_states = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    value_states = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    hidden_states = torch.randn(hidden_batch, total_tokens, hidden_dim, device=device, dtype=torch.float16)
    residual_states = torch.randn_like(hidden_states)
    proc_indices = (torch.arange(total_tokens, device=device, dtype=torch.long) * 3 + 7)
    input_ids = torch.randint(0, 32000, (batch, total_tokens), device=device, dtype=torch.int64)
    position_ids = torch.arange(total_tokens, device=device, dtype=torch.int64).expand(batch, -1).clone()
    retain_mask = torch.rand(total_tokens, device=device) > 0.4
    offset = 0
    for length in q_lens.tolist():
        if length == 0:
            continue
        seq_slice = retain_mask[offset:offset + length]
        if not bool(seq_slice.any().item()):
            seq_slice[0] = True
        offset += length
    if bool(retain_mask.all().item()):
        retain_mask[0] = False
    retain_idx = torch.nonzero(retain_mask, as_tuple=True)[0]
    keep_tokens = int(retain_idx.numel())
    rope_dim = 16
    rotary_cos = torch.randn(total_tokens, rope_dim, device=device, dtype=torch.float16)
    rotary_sin = torch.randn_like(rotary_cos)

    fused_q_lens_buffer = torch.zeros_like(q_lens)
    fused = focus_compact_states(keep_tokens,
                                 retain_mask,
                                 query_states,
                                 key_states,
                                 value_states,
                                 hidden_states,
                                 input_ids,
                                 position_ids,
                                 proc_indices,
                                 q_lens,
                                 fused_q_lens_buffer,
                                 rotary_cos=rotary_cos,
                                 rotary_sin=rotary_sin,
                                 residual_states=residual_states)
    (fused_q, fused_k, fused_v, fused_hidden, fused_input_ids, fused_position_ids, fused_q_lens, fused_proc,
     fused_rotary, fused_residual) = fused
    fused_cos, fused_sin = fused_rotary

    ref_q = query_states.index_select(0, retain_idx)
    ref_k = key_states.index_select(0, retain_idx)
    ref_v = value_states.index_select(0, retain_idx)
    ref_hidden = hidden_states[:, retain_mask, :]
    ref_residual = residual_states[:, retain_mask, :]
    ref_input_ids = input_ids[:, retain_mask]
    ref_position_ids = position_ids[:, retain_mask]
    mask_vals = retain_mask.to(dtype=q_lens.dtype)
    token_seq_ids = torch.repeat_interleave(torch.arange(batch, device=device, dtype=torch.long), q_lens)
    ref_q_lens = torch.zeros_like(q_lens)
    ref_q_lens.scatter_add_(0, token_seq_ids, mask_vals)
    ref_proc = proc_indices[retain_mask]
    ref_cos = rotary_cos.index_select(0, retain_idx)
    ref_sin = rotary_sin.index_select(0, retain_idx)

    torch.testing.assert_close(fused_q, ref_q)
    torch.testing.assert_close(fused_k, ref_k)
    torch.testing.assert_close(fused_v, ref_v)
    torch.testing.assert_close(fused_hidden, ref_hidden)
    torch.testing.assert_close(fused_residual, ref_residual)
    torch.testing.assert_close(fused_input_ids, ref_input_ids)
    torch.testing.assert_close(fused_position_ids, ref_position_ids)
    torch.testing.assert_close(fused_q_lens, ref_q_lens)
    torch.testing.assert_close(fused_proc, ref_proc)
    torch.testing.assert_close(fused_cos, ref_cos)
    torch.testing.assert_close(fused_sin, ref_sin)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FOCUS Triton kernels require CUDA')
def test_focus_select_and_enforce_ragged_matches_reference():
    device = torch.device('cuda')
    torch.manual_seed(11)
    num_seq, max_len, block_count = 4, 12, 24
    lengths = torch.randint(0, max_len + 1, (num_seq, ), device=device, dtype=torch.int32)
    mask_indptr = torch.zeros(num_seq + 1, dtype=torch.int32, device=device)
    mask_indptr[1:] = torch.cumsum(lengths, dim=0)
    total = int(mask_indptr[-1].item())
    delta = torch.randn(total, device=device)
    block_positions = torch.randint(0, block_count, (total, ), device=device, dtype=torch.long)
    block_progress = torch.randint(-1, block_count, (num_seq, ), device=device, dtype=torch.long)
    targets = torch.randint(0, max_len + 1, (num_seq, ), device=device, dtype=torch.int32)
    should_evict = torch.rand(num_seq, device=device) > 0.4
    base_offset = 3
    mask_indices = torch.arange(total, device=device, dtype=torch.long) + base_offset
    proc_length = total + base_offset
    proc_indices = torch.randint(0, block_count, (proc_length, ), device=device, dtype=torch.long)
    proc_indices[mask_indices] = block_positions
    prev_scores = torch.randn(proc_length, device=device)
    importance = delta + prev_scores.index_select(0, mask_indices)

    padded_delta = torch.zeros(num_seq, max_len, device=device)
    valid_mask = torch.zeros(num_seq, max_len, device=device, dtype=torch.bool)
    padded_positions = torch.full((num_seq, max_len), -1, device=device, dtype=torch.long)
    offset = 0
    for idx, length in enumerate(lengths.tolist()):
        if length <= 0:
            continue
        padded_delta[idx, :length] = delta[offset:offset + length]
        valid_mask[idx, :length] = True
        padded_positions[idx, :length] = block_positions[offset:offset + length]
        offset += length

    runtime_max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    fused = focus_select_and_enforce_ragged(importance.clone(),
                                            prev_scores.clone(),
                                            mask_indices.clone(),
                                            proc_indices.clone(),
                                            mask_indptr.clone(),
                                            targets.clone(),
                                            should_evict.clone(),
                                            block_progress.clone(),
                                            runtime_max_len)
    reference = _reference_focus_select_and_enforce(padded_delta.cpu(), valid_mask.cpu(), targets.cpu(),
                                                    should_evict.cpu(), padded_positions.cpu(),
                                                    block_progress.cpu())
    expected = []
    for idx, length in enumerate(lengths.tolist()):
        if length <= 0:
            continue
        expected.append(reference[idx, :length])
    if expected:
        expected_flat = torch.cat(expected).to(device=device)
        torch.testing.assert_close(fused.to(device=device, dtype=expected_flat.dtype), expected_flat)
    else:
        assert fused.numel() == 0
