import torch

import triton
import triton.language as tl


@triton.jit
def _focus_targets_kernel(mask_lengths,
                          avg_tokens,
                          targets,
                          alpha,
                          stride_ml: tl.constexpr,
                          stride_avg: tl.constexpr,
                          stride_out: tl.constexpr,
                          n_elements,
                          BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    lengths = tl.load(mask_lengths + offsets * stride_ml, mask=mask, other=0).to(tl.float32)
    avg = tl.load(avg_tokens + offsets * stride_avg, mask=mask, other=0.0)
    avg = tl.maximum(avg, 1.0)
    retain = tl.ceil(avg * alpha)
    retain = tl.maximum(retain, 1.0)
    retain = tl.minimum(lengths, retain)
    retain = tl.where(lengths <= 0, 0.0, retain)
    retain = retain.to(targets.dtype.element_ty)
    tl.store(targets + offsets * stride_out, retain, mask=mask)
    
    
@triton.jit
def _focus_importance_ragged_kernel(
    Q,
    K,
    INDICES,
    INDPTR,
    WORKSPACE,
    IMPORTANCE,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_ws,
    max_seq,
    head_dim,
    scale,
    rows_per_seq,
    BLOCK_D: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    num_key_value_groups: tl.constexpr,
):
    pid = tl.program_id(0)

    seq_idx = pid // rows_per_seq
    head_row = pid % rows_per_seq
    head_idx = head_row // max_seq
    query_idx = head_row % max_seq

    seq_start = tl.load(INDPTR + seq_idx).to(tl.int64)
    seq_end = tl.load(INDPTR + seq_idx + 1).to(tl.int64)
    seq_len = seq_end - seq_start
    if (seq_len <= 0) | (query_idx >= seq_len):
        return

    kv_head_idx = head_idx // num_key_value_groups

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    q_token_idx = tl.load(INDICES + seq_start + query_idx).to(tl.int64)
    q_ptr = Q + q_token_idx * stride_qt + head_idx * stride_qh
    q_vec = tl.load(q_ptr + offs_d * stride_qd, mask=mask_d, other=0.0).to(tl.float32)

    neg_inf = -float('inf')
    row_workspace_ptr = WORKSPACE + pid * stride_ws

    prev_score = neg_inf
    pos = 0
    cond = pos < seq_len
    key_idx = tl.load(INDICES + seq_start + pos, mask=cond, other=0).to(tl.int64)
    k_ptr = K + key_idx * stride_kt + kv_head_idx * stride_kh
    k_vec = tl.load(k_ptr + offs_d * stride_kd, mask=mask_d & cond, other=0.0).to(tl.float32)
    score = tl.sum(q_vec * k_vec, axis=0)
    curr_score = tl.where(cond, score * scale, neg_inf)

    pos = 1
    cond = pos < seq_len
    key_idx = tl.load(INDICES + seq_start + pos, mask=cond, other=0).to(tl.int64)
    k_ptr = K + key_idx * stride_kt + kv_head_idx * stride_kh
    k_vec = tl.load(k_ptr + offs_d * stride_kd, mask=mask_d & cond, other=0.0).to(tl.float32)
    score = tl.sum(q_vec * k_vec, axis=0)
    next_score = tl.where(cond, score * scale, neg_inf)

    for key_pos in tl.range(0, max_seq):
        key_valid = key_pos < seq_len
        next_valid = (key_pos + 1) < seq_len
        pooled = tl.maximum(prev_score, curr_score)
        pooled = tl.maximum(pooled, tl.where(next_valid, next_score, neg_inf))
        to_store = tl.where(key_valid, pooled, neg_inf)
        tl.store(row_workspace_ptr + key_pos, to_store)
        prev_score = tl.where(key_valid, curr_score, prev_score)
        curr_score = tl.where(next_valid, next_score, curr_score)
        future_idx = key_pos + 2
        cond_future = future_idx < seq_len
        key_idx = tl.load(INDICES + seq_start + future_idx, mask=cond_future, other=0).to(tl.int64)
        k_ptr = K + key_idx * stride_kt + kv_head_idx * stride_kh
        k_vec = tl.load(k_ptr + offs_d * stride_kd, mask=mask_d & cond_future, other=0.0).to(tl.float32)
        score = tl.sum(q_vec * k_vec, axis=0)
        next_score = tl.where(cond_future, score * scale, neg_inf)

    row_max = neg_inf
    offs_block = tl.arange(0, BLOCK_ROW)
    for start in tl.range(0, max_seq, BLOCK_ROW):
        block_offsets = start + offs_block
        mask = block_offsets < max_seq
        block_vals = tl.load(row_workspace_ptr + block_offsets, mask=mask, other=neg_inf)
        block_max = tl.max(block_vals, axis=0)
        row_max = tl.maximum(row_max, block_max)

    row_sum = tl.zeros([1], dtype=tl.float32)
    for start in tl.range(0, max_seq, BLOCK_ROW):
        block_offsets = start + offs_block
        mask = block_offsets < max_seq
        block_vals = tl.load(row_workspace_ptr + block_offsets, mask=mask, other=neg_inf)
        row_sum += tl.sum(tl.exp(block_vals - row_max), axis=0)

    row_sum = tl.where(row_sum > 0, row_sum, 1.0)
    inv_row_sum = 1.0 / row_sum
    importance_row_ptr = IMPORTANCE + seq_start
    for start in tl.range(0, max_seq, BLOCK_ROW):
        block_offsets = start + offs_block
        mask = block_offsets < seq_len
        block_vals = tl.load(row_workspace_ptr + block_offsets, mask=mask, other=neg_inf)
        weights = tl.exp(block_vals - row_max) * inv_row_sum
        tl.atomic_add(importance_row_ptr + block_offsets, weights, mask=mask)


@triton.jit
def _focus_enforce_rules_kernel(
    BLOCK_POS,
    BLOCK_UNPROCESSED,
    RETAIN,
    VALID,
    stride_bp,
    stride_bc,
    stride_up,
    stride_uc,
    stride_rb,
    stride_rc,
    stride_vb,
    stride_vc,
    width,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    in_bounds = offs < width

    pos_ptrs = BLOCK_POS + row * stride_bp + offs * stride_bc
    positions = tl.load(pos_ptrs, mask=in_bounds, other=-1).to(tl.int32)

    retain_ptrs = RETAIN + row * stride_rb + offs * stride_rc
    retain = tl.load(retain_ptrs, mask=in_bounds, other=0).to(tl.int1)

    valid_ptrs = VALID + row * stride_vb + offs * stride_vc
    valid = tl.load(valid_ptrs, mask=in_bounds, other=0).to(tl.int1)

    has_next = (offs + 1) < width
    next_pos = tl.load(pos_ptrs + stride_bc, mask=has_next, other=-2).to(tl.int32)
    next_valid = tl.load(valid_ptrs + stride_vc, mask=has_next, other=0).to(tl.int1)
    next_retain = tl.load(retain_ptrs + stride_rc, mask=has_next, other=0).to(tl.int1)

    adjacency = ((next_pos - positions) == 1)
    adjacency = adjacency & has_next & valid & next_valid
    adjust = adjacency & next_retain & (retain == 0)
    retain = retain | adjust

    retain_valid = retain & valid
    keep_count = tl.sum(retain_valid.to(tl.int32), axis=0)
    no_keep = keep_count == 0
    retain = tl.where(no_keep, valid, retain)
    retain_valid = retain & valid

    safe_positions = tl.where(retain_valid, positions, -1)
    rightmost = tl.max(safe_positions, axis=0)
    less_than_rightmost = positions < rightmost
    not_retain = retain == 0
    evicted_before = less_than_rightmost & not_retain & valid

    gather_indices = tl.where(positions >= 0, positions, 0)
    block_ptrs = BLOCK_UNPROCESSED + row * stride_up + gather_indices * stride_uc
    block_flags = tl.load(block_ptrs, mask=in_bounds, other=0).to(tl.int1)
    need_keep = block_flags & evicted_before
    retain = retain | need_keep

    tl.store(retain_ptrs, retain, mask=in_bounds)


@triton.jit
def _focus_select_enforce_ragged_kernel(
    IMPORTANCE,
    PREV_SCORES,
    MASK_GLOBALS,
    PROC_INDICES,
    INDPTR,
    TARGETS,
    SHOULD,
    BLOCK_UNPROCESSED,
    OUTPUT,
    stride_bp,
    stride_bc,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    start = tl.load(INDPTR + seq).to(tl.int64)
    end = tl.load(INDPTR + seq + 1).to(tl.int64)
    seq_len = end - start
    if seq_len <= 0:
        return
    offs = tl.arange(0, BLOCK)
    in_bounds = offs < seq_len

    mask_ptr = MASK_GLOBALS + start
    token_indices = tl.load(mask_ptr + offs, mask=in_bounds, other=0).to(tl.int64)
    curr_ptr = IMPORTANCE + start
    curr_scores = tl.load(curr_ptr + offs, mask=in_bounds, other=0.0).to(tl.float32)
    prev_scores = tl.load(PREV_SCORES + token_indices, mask=in_bounds, other=0.0).to(tl.float32)
    scores_row = curr_scores - prev_scores
    scores_rank = scores_row

    valid_row = in_bounds.to(tl.int1)
    valid_f32 = valid_row.to(tl.float32)
    counts = tl.sum(valid_f32, axis=0).to(tl.float32)

    target = tl.load(TARGETS + seq).to(tl.int32)
    should_prune = tl.load(SHOULD + seq).to(tl.int1)
    target = tl.where(should_prune, target, 0)
    target = tl.maximum(target, 0)
    max_counts = counts.to(tl.int32)
    target = tl.minimum(target, max_counts)
    positive = target > 0
    target_clamped = tl.where(positive, tl.maximum(target, 1), target)

    selected = tl.zeros([BLOCK], dtype=tl.int1)
    remaining = target_clamped
    filler = float('-inf')
    for _ in range(0, BLOCK):
        available = valid_row & (~selected) & in_bounds
        available_count = tl.sum(available.to(tl.int32), axis=0)
        work = (available_count > 0) & (remaining > 0)
        masked_scores = tl.where(available, scores_rank, filler)
        best_val = tl.max(masked_scores, axis=0)
        select_mask = available & (masked_scores == best_val)
        prefix = tl.cumsum(select_mask.to(tl.int32))
        take = select_mask & (prefix <= remaining) & (prefix > 0)
        take = tl.where(work, take, tl.zeros_like(take))
        selected = selected | take
        taken = tl.sum(take.to(tl.int32), axis=0)
        remaining = remaining - taken
        scores_rank = tl.where(take, filler, scores_rank)

    masked_scores = scores_row * valid_f32
    denom = tl.maximum(counts, 1.0)
    mean = tl.sum(masked_scores, axis=0) / denom
    diff = (scores_row - mean) * valid_f32
    variance = tl.sum(diff * diff, axis=0) / denom
    std = tl.sqrt(variance)
    threshold = mean + std
    candidate_mask = (scores_row >= threshold) & valid_row
    candidate_counts = tl.sum(candidate_mask.to(tl.int32), axis=0)
    use_threshold = (target_clamped > 0) & (candidate_counts >= target_clamped)
    selection = tl.where(use_threshold, candidate_mask, selected)
    selection = selection & valid_row & in_bounds

    base_retain = tl.where(should_prune, selection, (valid_row & in_bounds))
    retain_ptr = OUTPUT + start
    tl.store(retain_ptr + offs, base_retain, mask=in_bounds)

    retain = base_retain
    positions = tl.load(PROC_INDICES + token_indices, mask=in_bounds, other=-1).to(tl.int32)

    has_next = (offs + 1) < seq_len
    next_token_idx = tl.load(mask_ptr + offs + 1, mask=has_next, other=0).to(tl.int64)
    next_pos = tl.load(PROC_INDICES + next_token_idx, mask=has_next, other=-2).to(tl.int32)
    next_valid = valid_row & has_next
    next_retain = tl.load(retain_ptr + offs + 1, mask=has_next, other=0).to(tl.int1)

    adjacency = ((next_pos - positions) == 1)
    adjacency = adjacency & has_next & valid_row & next_valid
    adjust = adjacency & next_retain & (retain == 0)
    retain = retain | adjust

    retain_valid = retain & valid_row
    keep_count = tl.sum(retain_valid.to(tl.int32), axis=0)
    no_keep = keep_count == 0
    retain = tl.where(no_keep, valid_row & in_bounds, retain)
    retain_valid = retain & valid_row

    safe_positions = tl.where(retain_valid, positions, -1)
    rightmost = tl.max(safe_positions, axis=0)
    less_than_rightmost = positions < rightmost
    not_retain = retain == 0
    evicted_before = less_than_rightmost & not_retain & valid_row

    gather_indices = tl.where(positions >= 0, positions, 0)
    row_block_ptr = BLOCK_UNPROCESSED + seq * stride_bp
    block_ptrs = row_block_ptr + gather_indices * stride_bc
    block_flags = tl.load(block_ptrs, mask=in_bounds, other=0).to(tl.int1)
    need_keep = block_flags & evicted_before
    retain = retain | need_keep

    tl.store(retain_ptr + offs, retain, mask=in_bounds)


@triton.jit
def _focus_compact_states_kernel(
    RETAIN_IDX,
    PROC_IN,
    TOKEN_SEQ_IDS,
    Q,
    K,
    V,
    H,
    INPUT_IDS,
    POS_IDS,
    INPUT_IDS_OUT,
    POS_IDS_OUT,
    Q_OUT,
    K_OUT,
    V_OUT,
    H_OUT,
    PROC_OUT,
    NEW_Q_LENS,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_qot,
    stride_qoh,
    stride_qod,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_kot,
    stride_koh,
    stride_kod,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_vot,
    stride_voh,
    stride_vod,
    stride_hb,
    stride_ht,
    stride_hd,
    stride_hob,
    stride_hot,
    stride_hod,
    stride_ib,
    stride_it,
    stride_iob,
    stride_iot,
    stride_pb,
    stride_pt,
    stride_pob,
    stride_pot,
    n_keep,
    hidden_batch,
    input_batch,
    pos_batch,
    q_num_heads,
    q_head_dim,
    k_num_heads,
    k_head_dim,
    v_num_heads,
    v_head_dim,
    hidden_dim,
    BLOCK_Q_HEAD: tl.constexpr,
    BLOCK_Q_DIM: tl.constexpr,
    BLOCK_K_HEAD: tl.constexpr,
    BLOCK_V_HEAD: tl.constexpr,
    BLOCK_H_BATCH: tl.constexpr,
    BLOCK_H_DIM: tl.constexpr,
    BLOCK_ID_BATCH: tl.constexpr,
    BLOCK_POS_BATCH: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if token_idx >= n_keep:
        return

    src_idx = tl.load(RETAIN_IDX + token_idx).to(tl.int32)

    seq = tl.load(TOKEN_SEQ_IDS + src_idx).to(tl.int32)
    tl.atomic_add(NEW_Q_LENS + seq, 1)

    proc_val = tl.load(PROC_IN + src_idx)
    tl.store(PROC_OUT + token_idx, proc_val)

    offs_head = tl.arange(0, BLOCK_Q_HEAD)
    offs_dim = tl.arange(0, BLOCK_Q_DIM)
    q_ptr = Q + src_idx * stride_qt
    q_out_ptr = Q_OUT + token_idx * stride_qot
    for head_start in range(0, q_num_heads, BLOCK_Q_HEAD):
        head_offsets = head_start + offs_head
        head_mask = head_offsets < q_num_heads
        for dim_start in range(0, q_head_dim, BLOCK_Q_DIM):
            dim_offsets = dim_start + offs_dim
            dim_mask = dim_offsets < q_head_dim
            mask = head_mask[:, None] & dim_mask[None, :]
            vals = tl.load(q_ptr + head_offsets[:, None] * stride_qh + dim_offsets[None, :] * stride_qd,
                           mask=mask,
                           other=0.0)
            tl.store(q_out_ptr + head_offsets[:, None] * stride_qoh + dim_offsets[None, :] * stride_qod,
                     vals,
                     mask=mask)

    offs_k_head = tl.arange(0, BLOCK_K_HEAD)
    k_ptr = K + src_idx * stride_kt
    k_out_ptr = K_OUT + token_idx * stride_kot
    for head_start in range(0, k_num_heads, BLOCK_K_HEAD):
        head_offsets = head_start + offs_k_head
        head_mask = head_offsets < k_num_heads
        for dim_start in range(0, k_head_dim, BLOCK_Q_DIM):
            dim_offsets = dim_start + offs_dim
            dim_mask = dim_offsets < k_head_dim
            mask = head_mask[:, None] & dim_mask[None, :]
            vals = tl.load(k_ptr + head_offsets[:, None] * stride_kh + dim_offsets[None, :] * stride_kd,
                           mask=mask,
                           other=0.0)
            tl.store(k_out_ptr + head_offsets[:, None] * stride_koh + dim_offsets[None, :] * stride_kod,
                     vals,
                     mask=mask)

    offs_v_head = tl.arange(0, BLOCK_V_HEAD)
    v_ptr = V + src_idx * stride_vt
    v_out_ptr = V_OUT + token_idx * stride_vot
    for head_start in range(0, v_num_heads, BLOCK_V_HEAD):
        head_offsets = head_start + offs_v_head
        head_mask = head_offsets < v_num_heads
        for dim_start in range(0, v_head_dim, BLOCK_Q_DIM):
            dim_offsets = dim_start + offs_dim
            dim_mask = dim_offsets < v_head_dim
            mask = head_mask[:, None] & dim_mask[None, :]
            vals = tl.load(v_ptr + head_offsets[:, None] * stride_vh + dim_offsets[None, :] * stride_vd,
                           mask=mask,
                           other=0.0)
            tl.store(v_out_ptr + head_offsets[:, None] * stride_voh + dim_offsets[None, :] * stride_vod,
                     vals,
                     mask=mask)

    offs_hidden = tl.arange(0, BLOCK_H_DIM)
    offs_batch = tl.arange(0, BLOCK_H_BATCH)
    for batch_start in range(0, hidden_batch, BLOCK_H_BATCH):
        batch_offsets = batch_start + offs_batch
        batch_mask = batch_offsets < hidden_batch
        h_ptr = H + batch_offsets[:, None] * stride_hb + src_idx * stride_ht
        h_out_ptr = H_OUT + batch_offsets[:, None] * stride_hob + token_idx * stride_hot
        for dim_start in range(0, hidden_dim, BLOCK_H_DIM):
            dim_offsets = dim_start + offs_hidden
            dim_mask = dim_offsets < hidden_dim
            mask = batch_mask[:, None] & dim_mask[None, :]
            vals = tl.load(h_ptr + dim_offsets[None, :] * stride_hd, mask=mask, other=0.0)
            tl.store(h_out_ptr + dim_offsets[None, :] * stride_hod, vals, mask=mask)

    offs_ib = tl.arange(0, BLOCK_ID_BATCH)
    for batch_start in range(0, input_batch, BLOCK_ID_BATCH):
        batch_offsets = batch_start + offs_ib
        batch_mask = batch_offsets < input_batch
        inp_ptr = INPUT_IDS + batch_offsets * stride_ib + src_idx * stride_it
        inp_vals = tl.load(inp_ptr, mask=batch_mask, other=0)
        inp_out_ptr = INPUT_IDS_OUT + batch_offsets * stride_iob + token_idx * stride_iot
        tl.store(inp_out_ptr, inp_vals, mask=batch_mask)

    offs_pb = tl.arange(0, BLOCK_POS_BATCH)
    for batch_start in range(0, pos_batch, BLOCK_POS_BATCH):
        batch_offsets = batch_start + offs_pb
        batch_mask = batch_offsets < pos_batch
        pos_ptr = POS_IDS + batch_offsets * stride_pb + src_idx * stride_pt
        pos_vals = tl.load(pos_ptr, mask=batch_mask, other=0)
        pos_out_ptr = POS_IDS_OUT + batch_offsets * stride_pob + token_idx * stride_pot
        tl.store(pos_out_ptr, pos_vals, mask=batch_mask)


def focus_importance_ragged(query_states: torch.Tensor,
                            key_states: torch.Tensor,
                            mask_indices: torch.Tensor,
                            mask_indptr: torch.Tensor,
                            max_mask_len: int,
                            num_key_value_groups: int,
                            scale: float) -> torch.Tensor:
    """Compute importance scores directly over ragged sequences."""
    if max_mask_len <= 0:
        return query_states.new_zeros((0, ), dtype=query_states.dtype)
    device = query_states.device
    lengths = (mask_indptr[1:] - mask_indptr[:-1])
    num_seq = lengths.numel()
    num_heads = query_states.size(1)
    head_dim = query_states.size(2)
    rows_per_seq = num_heads * max_mask_len
    total_rows = rows_per_seq * num_seq
    total_tokens = mask_indices.size(0)
    importance = torch.zeros((total_tokens, ), dtype=torch.float32, device=device)
    workspace = torch.empty((total_rows, max_mask_len), dtype=torch.float32, device=device)
    stride_qt, stride_qh, stride_qd = query_states.stride()
    stride_kt, stride_kh, stride_kd = key_states.stride()
    stride_ws = workspace.stride(0)
    block_d = triton.next_power_of_2(head_dim)
    block_row = min(128, triton.next_power_of_2(max_mask_len))
    num_warps, num_stages = _pick_focus_kernel_meta(block_d, block_row)
    grid = (total_rows, )
    _focus_importance_ragged_kernel[grid](query_states,
                                          key_states,
                                          mask_indices,
                                          mask_indptr,
                                          workspace,
                                          importance,
                                          stride_qt,
                                          stride_qh,
                                          stride_qd,
                                          stride_kt,
                                          stride_kh,
                                          stride_kd,
                                          stride_ws,
                                          max_mask_len,
                                          head_dim,
                                          scale,
                                          rows_per_seq,
                                          BLOCK_D=block_d,
                                          BLOCK_ROW=block_row,
                                          num_key_value_groups=num_key_value_groups,
                                          num_warps=num_warps,
                                          num_stages=num_stages)
    return importance.to(dtype=query_states.dtype)


def focus_compute_targets(mask_lengths: torch.Tensor, avg_tokens: torch.Tensor, focus_alpha: float) -> torch.Tensor:
    """Compute focus targets via the Triton kernel."""
    targets = torch.empty_like(mask_lengths)
    n_elements = mask_lengths.numel()
    BLOCK = 128
    grid = (triton.cdiv(n_elements, BLOCK), )
    tgt_num_warps, tgt_num_stages = _pick_focus_target_meta(n_elements)
    _focus_targets_kernel[grid](mask_lengths,
                                avg_tokens,
                                targets,
                                focus_alpha,
                                stride_ml=mask_lengths.stride(0),
                                stride_avg=avg_tokens.stride(0),
                                stride_out=targets.stride(0),
                                n_elements=n_elements,
                                BLOCK=BLOCK,
                                num_warps=tgt_num_warps,
                                num_stages=tgt_num_stages)
    return targets


def focus_enforce_rules(block_positions: torch.Tensor, block_unprocessed: torch.Tensor,
                        retain_mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Fuse rule enforcement into a Triton kernel."""
    batch, width = block_positions.shape
    grid = (batch, )
    BLOCK = triton.next_power_of_2(max(1, width))
    num_warps, num_stages = _pick_focus_enforce_meta(width)
    _focus_enforce_rules_kernel[grid](block_positions,
                                      block_unprocessed,
                                      retain_mask,
                                      valid_mask,
                                      block_positions.stride(0),
                                      block_positions.stride(1),
                                      block_unprocessed.stride(0),
                                      block_unprocessed.stride(1),
                                      retain_mask.stride(0),
                                      retain_mask.stride(1),
                                      valid_mask.stride(0),
                                      valid_mask.stride(1),
                                      width,
                                      BLOCK=BLOCK,
                                      num_warps=num_warps,
                                      num_stages=num_stages)
    return retain_mask


def focus_select_and_enforce_ragged(importance: torch.Tensor,
                                    prev_scores: torch.Tensor,
                                    mask_indices: torch.Tensor,
                                    proc_indices: torch.Tensor,
                                    mask_indptr: torch.Tensor,
                                    targets: torch.Tensor,
                                    should_prune: torch.Tensor,
                                    block_unprocessed: torch.Tensor,
                                    max_len: int) -> torch.Tensor:
    """Ragged variant that also computes delta/block positions inside the Triton kernel."""
    num_seq = mask_indptr.numel() - 1
    device = importance.device
    targets_i32 = targets.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    should_prune_i32 = should_prune.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    output = torch.zeros_like(importance, dtype=torch.bool)
    block = triton.next_power_of_2(max_len)
    num_warps, num_stages = _pick_focus_select_enforce_meta(max_len)
    grid = (num_seq, )
    _focus_select_enforce_ragged_kernel[grid](importance,
                                              prev_scores,
                                              mask_indices,
                                              proc_indices,
                                              mask_indptr,
                                              targets_i32,
                                              should_prune_i32,
                                              block_unprocessed,
                                              output,
                                              block_unprocessed.stride(0),
                                              block_unprocessed.stride(1),
                                              BLOCK=block,
                                              num_warps=num_warps,
                                              num_stages=num_stages)
    return output


def focus_compact_states(retain_indices: torch.Tensor,
                         query_states: torch.Tensor,
                         key_states: torch.Tensor,
                         value_states: torch.Tensor,
                         hidden_states: torch.Tensor,
                         input_ids: torch.Tensor,
                         position_ids: torch.Tensor,
                         proc_indices: torch.Tensor,
                         orig_q_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                             torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse state compaction (q/k/v, hidden, indices, q_lens) into a Triton kernel."""
    device = query_states.device

    keep_tokens = retain_indices.numel()
    q_shape = query_states.shape
    k_shape = key_states.shape
    v_shape = value_states.shape
    hidden_shape = hidden_states.shape
    ids_shape = input_ids.shape
    pos_shape = position_ids.shape

    q_out = query_states.new_empty((keep_tokens, q_shape[1], q_shape[2]))
    k_out = key_states.new_empty((keep_tokens, k_shape[1], k_shape[2]))
    v_out = value_states.new_empty((keep_tokens, v_shape[1], v_shape[2]))
    hidden_out = hidden_states.new_empty((hidden_shape[0], keep_tokens, hidden_shape[2]))
    input_ids_out = input_ids.new_empty((ids_shape[0], keep_tokens))
    position_ids_out = position_ids.new_empty((pos_shape[0], keep_tokens))
    new_proc_indices = proc_indices.new_empty((keep_tokens, ))
    # q_out = torch.empty((keep_tokens, q_shape[1], q_shape[2]), dtype=query_states.dtype,
    #                      pin_memory=True).to(device=device, non_blocking=True)
    # k_out = torch.empty((keep_tokens, k_shape[1], k_shape[2]), dtype=key_states.dtype,
    #                      pin_memory=True).to(device=device, non_blocking=True)
    # v_out = torch.empty((keep_tokens, v_shape[1], v_shape[2]), dtype=value_states.dtype,
    #                      pin_memory=True).to(device=device, non_blocking=True)
    # hidden_out = torch.empty((hidden_shape[0], keep_tokens, hidden_shape[2]), dtype=hidden_states.dtype,
    #                           pin_memory=True).to(device=device, non_blocking=True)
    # input_ids_out = torch.empty((ids_shape[0], keep_tokens), dtype=input_ids.dtype,
    #                              pin_memory=True).to(device=device, non_blocking=True)
    # position_ids_out = torch.empty((pos_shape[0], keep_tokens), dtype=position_ids.dtype,
    #                                 pin_memory=True).to(device=device, non_blocking=True)
    # new_proc_indices = torch.empty((keep_tokens, ), dtype=proc_indices.dtype,
    #                                 pin_memory=True).to(device=device, non_blocking=True)

    orig_q_lens_i32 = orig_q_lens.to(dtype=torch.int32)
    seq_ids = torch.arange(orig_q_lens_i32.size(0), device=device, dtype=torch.int32)
    token_seq_ids = torch.repeat_interleave(seq_ids, orig_q_lens_i32, output_size=int(proc_indices.numel()))
    new_q_lens = torch.zeros_like(orig_q_lens_i32)

    stride_qt, stride_qh, stride_qd = query_states.stride()
    stride_qot, stride_qoh, stride_qod = q_out.stride()
    stride_kt, stride_kh, stride_kd = key_states.stride()
    stride_kot, stride_koh, stride_kod = k_out.stride()
    stride_vt, stride_vh, stride_vd = value_states.stride()
    stride_vot, stride_voh, stride_vod = v_out.stride()
    stride_hb, stride_ht, stride_hd = hidden_states.stride()
    stride_hob, stride_hot, stride_hod = hidden_out.stride()
    stride_ib, stride_it = input_ids.stride()
    stride_iob, stride_iot = input_ids_out.stride()
    stride_pb, stride_pt = position_ids.stride()
    stride_pob, stride_pot = position_ids_out.stride()

    grid = (keep_tokens, )
    num_warps, num_stages = _pick_focus_compact_meta(q_shape[1], q_shape[2], hidden_shape[0])
    _focus_compact_states_kernel[grid](retain_indices,
                                       proc_indices,
                                       token_seq_ids,
                                       query_states,
                                       key_states,
                                       value_states,
                                       hidden_states,
                                       input_ids,
                                       position_ids,
                                       input_ids_out,
                                       position_ids_out,
                                       q_out,
                                       k_out,
                                       v_out,
                                       hidden_out,
                                       new_proc_indices,
                                       new_q_lens,
                                       stride_qt,
                                       stride_qh,
                                       stride_qd,
                                       stride_qot,
                                       stride_qoh,
                                       stride_qod,
                                       stride_kt,
                                       stride_kh,
                                       stride_kd,
                                       stride_kot,
                                       stride_koh,
                                       stride_kod,
                                       stride_vt,
                                       stride_vh,
                                       stride_vd,
                                       stride_vot,
                                       stride_voh,
                                       stride_vod,
                                       stride_hb,
                                       stride_ht,
                                       stride_hd,
                                       stride_hob,
                                       stride_hot,
                                       stride_hod,
                                       stride_ib,
                                       stride_it,
                                       stride_iob,
                                       stride_iot,
                                       stride_pb,
                                       stride_pt,
                                       stride_pob,
                                       stride_pot,
                                       keep_tokens,
                                       hidden_shape[0],
                                       ids_shape[0],
                                       pos_shape[0],
                                       q_shape[1],
                                       q_shape[2],
                                       k_shape[1],
                                       k_shape[2],
                                       v_shape[1],
                                       v_shape[2],
                                       hidden_shape[2],
                                       BLOCK_Q_HEAD=4,
                                       BLOCK_Q_DIM=64,
                                       BLOCK_K_HEAD=4,
                                       BLOCK_V_HEAD=4,
                                       BLOCK_H_BATCH=4,
                                       BLOCK_H_DIM=64,
                                       BLOCK_ID_BATCH=4,
                                       BLOCK_POS_BATCH=4,
                                       num_warps=num_warps,
                                       num_stages=num_stages)
    return (q_out, k_out, v_out, hidden_out, input_ids_out, position_ids_out, new_q_lens.to(orig_q_lens.dtype),
            new_proc_indices)


def _pick_focus_enforce_meta(width: int) -> tuple[int, int]:
    major, _ = torch.cuda.get_device_capability()
    if major >= 9:
        if width >= 256:
            return 4, 3
        if width >= 128:
            return 4, 2
        if width >= 64:
            return 2, 2
        return 1, 2
    if major >= 8:
        if width >= 128:
            return 2, 2
        return 1, 2
    return 1, 1


def _pick_focus_select_enforce_meta(width: int) -> tuple[int, int]:
    """Scheduling heuristic for the fused selection+enforcement kernel."""
    major, _ = torch.cuda.get_device_capability()
    if major >= 9:
        if width >= 256:
            return 8, 3
        if width >= 128:
            return 4, 3
        if width >= 64:
            return 4, 2
        return 2, 2
    if major >= 8:
        if width >= 256:
            return 4, 2
        if width >= 128:
            return 2, 2
        return 1, 2
    return 1, 1


def _pick_focus_target_meta(n_elements: int) -> tuple[int, int]:
    """Heuristic num_warps/num_stages for the focus target kernel."""
    major, _ = torch.cuda.get_device_capability()
    if major >= 9:
        if n_elements >= 1 << 15:
            return 4, 3
        if n_elements >= 1 << 13:
            return 4, 2
        if n_elements >= 1 << 11:
            return 2, 2
        return 1, 2
    if major >= 8:
        if n_elements >= 1 << 14:
            return 4, 2
        if n_elements >= 1 << 12:
            return 2, 2
        return 1, 2
    return 1, 1


def _pick_focus_kernel_meta(block_d: int, block_row: int) -> tuple[int, int]:
    """Choose num_warps/num_stages based on tile shape and GPU architecture."""
    major, _ = torch.cuda.get_device_capability()
    area = block_d * block_row
    if major >= 9:
        if area >= 16384:
            return 8, 4
        if area >= 8192:
            return 8, 3
        if area >= 4096:
            return 4, 3
        return 2, 2
    if major >= 8:
        if area >= 16384:
            return 4, 3
        if area >= 8192:
            return 4, 2
        return 2, 2
    return 2, 2


def _pick_focus_compact_meta(num_heads: int, head_dim: int, hidden_batch: int) -> tuple[int, int]:
    """Simple heuristic for the compact kernel launch config."""
    major, _ = torch.cuda.get_device_capability()
    big_tensor = (num_heads * head_dim) >= 1024 or hidden_batch >= 4
    if major >= 9:
        return (4, 3) if big_tensor else (2, 2)
    if major >= 8:
        return (2, 2) if big_tensor else (1, 2)
    return 1, 1
