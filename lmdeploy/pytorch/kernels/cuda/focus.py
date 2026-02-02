# Copyright (c) 2026 SANDS Lab. All rights reserved.
import torch

import triton
import triton.language as tl
from typing import Tuple


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
def _focus_select_enforce_ragged_kernel(
    IMPORTANCE,
    PREV_SCORES,
    MASK_GLOBALS,
    PROC_INDICES,
    INDPTR,
    TARGETS,
    SHOULD,
    BLOCK_PROGRESS,
    OUTPUT,
    stride_prog,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    start = tl.load(INDPTR + seq).to(tl.int64)
    end = tl.load(INDPTR + seq + 1).to(tl.int64)
    seq_len = end - start
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
    should_evict = tl.load(SHOULD + seq).to(tl.int1)
    target = tl.where(should_evict, target, 0)
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

    base_retain = tl.where(should_evict, selection, (valid_row & in_bounds))
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

    progress = tl.load(BLOCK_PROGRESS + seq * stride_prog).to(tl.int32)
    is_unprocessed = positions > progress
    need_keep = is_unprocessed & evicted_before
    retain = retain | need_keep

    tl.store(retain_ptr + offs, retain, mask=in_bounds)


@triton.jit
def _focus_keep_offsets_kernel(RETAIN_MASK, KEEP_OFFSETS, n_tokens, num_blocks, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    carry = tl.zeros((), dtype=tl.int32)
    for block_idx in range(0, num_blocks):
        block_start = block_idx * BLOCK
        idx = block_start + offsets
        mask = idx < n_tokens
        keep = tl.load(RETAIN_MASK + idx, mask=mask, other=0).to(tl.int32)
        keep = keep * mask.to(tl.int32)
        inclusive = tl.cumsum(keep)
        exclusive = inclusive - keep + carry
        keep_mask = keep == 1
        out = tl.where(keep_mask, exclusive, -1)
        tl.store(KEEP_OFFSETS + idx, out, mask=mask)
        block_total = tl.sum(keep, axis=0)
        carry += block_total


@triton.jit
def _focus_processing_view_kernel(
    NEW_Q_LENS,
    PROC_INDICES,
    HISTORY_LENS,
    NUM_IGNORED,
    BLOCK_PROGRESS,
    Q_START_LOC,
    CU_Q,
    KV_SEQLENS,
    CU_K,
    batch_size,
    stride_q_lens,
    stride_proc,
    stride_history,
    stride_ignored,
    stride_progress,
    stride_q_start,
    stride_cuq,
    stride_kv,
    stride_cuk,
):
    zero32 = tl.zeros((), dtype=tl.int32)
    tl.store(CU_Q, zero32)
    tl.store(CU_K, zero32)
    q_prefix = tl.zeros((), dtype=tl.int64)
    kv_prefix = tl.zeros((), dtype=tl.int64)
    for seq in tl.range(0, batch_size):
        q_len = tl.load(NEW_Q_LENS + seq * stride_q_lens).to(tl.int64)
        tl.store(Q_START_LOC + seq * stride_q_start, q_prefix)
        q_prefix += q_len
        tl.store(CU_Q + (seq + 1) * stride_cuq, q_prefix.to(tl.int32))

        end_offset = q_prefix - 1
        proc_val = tl.load(PROC_INDICES + end_offset * stride_proc).to(tl.int64)
        rightmost = proc_val

        progress_val = tl.load(BLOCK_PROGRESS + seq * stride_progress)
        updated_progress = tl.maximum(progress_val, rightmost.to(progress_val.dtype))
        tl.store(BLOCK_PROGRESS + seq * stride_progress, updated_progress)

        plus_one = tl.where(rightmost < 0, tl.zeros((), dtype=tl.int64), rightmost + 1)
        history = tl.load(HISTORY_LENS + seq * stride_history).to(tl.int64)
        ignored = tl.load(NUM_IGNORED + seq * stride_ignored).to(tl.int64)
        kv_len = history + plus_one - ignored
        tl.store(KV_SEQLENS + seq * stride_kv, kv_len)

        kv_prefix += kv_len
        tl.store(CU_K + (seq + 1) * stride_cuk, kv_prefix.to(tl.int32))


def _focus_compute_keep_offsets(retain_mask: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    total_tokens = retain_mask.numel()
    keep_offsets = retain_mask.new_empty((total_tokens, ), dtype=torch.int32)

    num_blocks = triton.cdiv(total_tokens, block_size)
    grid = (1, )
    _focus_keep_offsets_kernel[grid](retain_mask,
                                     keep_offsets,
                                     total_tokens,
                                     num_blocks,
                                     BLOCK=block_size,
                                     num_warps=4,
                                     num_stages=1)
    return keep_offsets


@triton.jit
def _focus_compact_states_kernel(
    RETAIN_MASK,
    KEEP_OFFSETS,
    PROC_IN,
    ORIG_Q_LENS,
    Q,
    K,
    V,
    H,
    RESIDUAL,
    INPUT_IDS,
    POS_IDS,
    ROTARY_COS,
    ROTARY_SIN,
    INPUT_IDS_OUT,
    POS_IDS_OUT,
    ROTARY_COS_OUT,
    ROTARY_SIN_OUT,
    Q_OUT,
    K_OUT,
    V_OUT,
    H_OUT,
    RESIDUAL_OUT,
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
    stride_res_b,
    stride_res_t,
    stride_res_d,
    stride_res_ob,
    stride_res_ot,
    stride_res_od,
    stride_ib,
    stride_it,
    stride_iob,
    stride_iot,
    stride_pb,
    stride_pt,
    stride_pob,
    stride_pot,
    stride_cos_t,
    stride_cos_d,
    stride_sin_t,
    stride_sin_d,
    stride_cos_ot,
    stride_cos_od,
    stride_sin_ot,
    stride_sin_od,
    stride_seq,
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
    residual_batch,
    residual_dim,
    rotary_dim,
    num_seqs,
    BLOCK_Q_HEAD: tl.constexpr,
    BLOCK_Q_DIM: tl.constexpr,
    BLOCK_K_HEAD: tl.constexpr,
    BLOCK_V_HEAD: tl.constexpr,
    BLOCK_H_BATCH: tl.constexpr,
    BLOCK_H_DIM: tl.constexpr,
    BLOCK_ID_BATCH: tl.constexpr,
    BLOCK_POS_BATCH: tl.constexpr,
    BLOCK_ROT_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    keep = tl.load(RETAIN_MASK + token_idx).to(tl.int1)
    if keep == 0:
        return

    dst_idx = tl.load(KEEP_OFFSETS + token_idx).to(tl.int32)
    if dst_idx < 0:
        return

    src_idx = token_idx

    seq_val = tl.zeros((), dtype=tl.int32)
    seq_found = tl.zeros((), dtype=tl.int32)
    prefix = tl.zeros((), dtype=tl.int32)
    for seq_idx in tl.range(0, num_seqs):
        length = tl.load(ORIG_Q_LENS + seq_idx * stride_seq).to(tl.int32)
        prefix += length
        in_seq = (seq_found == 0) & (token_idx < prefix)
        seq_val = tl.where(in_seq, seq_idx, seq_val)
        seq_found = tl.where(in_seq, seq_found + 1, seq_found)
    seq = tl.where(seq_found == 1, seq_val, tl.zeros((), dtype=tl.int32))
    tl.atomic_add(NEW_Q_LENS + seq, 1)

    proc_val = tl.load(PROC_IN + src_idx)
    tl.store(PROC_OUT + dst_idx, proc_val)

    offs_head = tl.arange(0, BLOCK_Q_HEAD)
    offs_dim = tl.arange(0, BLOCK_Q_DIM)
    q_ptr = Q + src_idx * stride_qt
    q_out_ptr = Q_OUT + dst_idx * stride_qot
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
    k_out_ptr = K_OUT + dst_idx * stride_kot
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
    v_out_ptr = V_OUT + dst_idx * stride_vot
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
        h_out_ptr = H_OUT + batch_offsets[:, None] * stride_hob + dst_idx * stride_hot
        for dim_start in range(0, hidden_dim, BLOCK_H_DIM):
            dim_offsets = dim_start + offs_hidden
            dim_mask = dim_offsets < hidden_dim
            mask = batch_mask[:, None] & dim_mask[None, :]
            vals = tl.load(h_ptr + dim_offsets[None, :] * stride_hd, mask=mask, other=0.0)
            tl.store(h_out_ptr + dim_offsets[None, :] * stride_hod, vals, mask=mask)

    for batch_start in range(0, residual_batch, BLOCK_H_BATCH):
        batch_offsets = batch_start + offs_batch
        batch_mask = batch_offsets < residual_batch
        r_ptr = RESIDUAL + batch_offsets[:, None] * stride_res_b + src_idx * stride_res_t
        r_out_ptr = RESIDUAL_OUT + batch_offsets[:, None] * stride_res_ob + dst_idx * stride_res_ot
        for dim_start in range(0, residual_dim, BLOCK_H_DIM):
            dim_offsets = dim_start + offs_hidden
            dim_mask = dim_offsets < residual_dim
            mask = batch_mask[:, None] & dim_mask[None, :]
            vals = tl.load(r_ptr + dim_offsets[None, :] * stride_res_d, mask=mask, other=0.0)
            tl.store(r_out_ptr + dim_offsets[None, :] * stride_res_od, vals, mask=mask)

    offs_ib = tl.arange(0, BLOCK_ID_BATCH)
    for batch_start in range(0, input_batch, BLOCK_ID_BATCH):
        batch_offsets = batch_start + offs_ib
        batch_mask = batch_offsets < input_batch
        inp_ptr = INPUT_IDS + batch_offsets * stride_ib + src_idx * stride_it
        inp_vals = tl.load(inp_ptr, mask=batch_mask, other=0)
        inp_out_ptr = INPUT_IDS_OUT + batch_offsets * stride_iob + dst_idx * stride_iot
        tl.store(inp_out_ptr, inp_vals, mask=batch_mask)

    offs_pb = tl.arange(0, BLOCK_POS_BATCH)
    for batch_start in range(0, pos_batch, BLOCK_POS_BATCH):
        batch_offsets = batch_start + offs_pb
        batch_mask = batch_offsets < pos_batch
        pos_ptr = POS_IDS + batch_offsets * stride_pb + src_idx * stride_pt
        pos_vals = tl.load(pos_ptr, mask=batch_mask, other=0)
        pos_out_ptr = POS_IDS_OUT + batch_offsets * stride_pob + dst_idx * stride_pot
        tl.store(pos_out_ptr, pos_vals, mask=batch_mask)

    offs_rot = tl.arange(0, BLOCK_ROT_DIM)
    cos_ptr = ROTARY_COS + src_idx * stride_cos_t
    cos_out_ptr = ROTARY_COS_OUT + dst_idx * stride_cos_ot
    sin_ptr = ROTARY_SIN + src_idx * stride_sin_t
    sin_out_ptr = ROTARY_SIN_OUT + dst_idx * stride_sin_ot
    for rot_start in range(0, rotary_dim, BLOCK_ROT_DIM):
        rot_offsets = rot_start + offs_rot
        rot_mask = rot_offsets < rotary_dim
        cos_vals = tl.load(cos_ptr + rot_offsets * stride_cos_d, mask=rot_mask, other=0.0)
        sin_vals = tl.load(sin_ptr + rot_offsets * stride_sin_d, mask=rot_mask, other=0.0)
        tl.store(cos_out_ptr + rot_offsets * stride_cos_od, cos_vals, mask=rot_mask)
        tl.store(sin_out_ptr + rot_offsets * stride_sin_od, sin_vals, mask=rot_mask)


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
    block_row = triton.next_power_of_2(max(16, min(max_mask_len, 128)))
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
    BLOCK = max(32, min(128, triton.next_power_of_2(max(1, n_elements))))
    grid = (triton.cdiv(n_elements, BLOCK), )
    tgt_num_warps, tgt_num_stages = _pick_focus_target_meta(n_elements, BLOCK)
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


def focus_select_and_enforce_ragged(importance: torch.Tensor,
                                    prev_scores: torch.Tensor,
                                    mask_indices: torch.Tensor,
                                    proc_indices: torch.Tensor,
                                    mask_indptr: torch.Tensor,
                                    targets: torch.Tensor,
                                    should_evict: torch.Tensor,
                                    block_progress: torch.Tensor,
                                    max_len: int) -> torch.Tensor:
    """Ragged variant that also computes delta/block positions inside the Triton kernel."""
    num_seq = mask_indptr.numel() - 1
    output = torch.zeros_like(importance, dtype=torch.bool)
    block = triton.next_power_of_2(max_len)
    num_warps, num_stages = _pick_focus_select_enforce_meta(max_len)
    grid = (num_seq, )
    _focus_select_enforce_ragged_kernel[grid](importance,
                                              prev_scores,
                                              mask_indices,
                                              proc_indices,
                                              mask_indptr,
                                              targets,
                                              should_evict,
                                              block_progress,
                                              output,
                                              block_progress.stride(0),
                                              BLOCK=block,
                                              num_warps=num_warps,
                                              num_stages=num_stages)
    return output


def focus_compact_states(keep_tokens: int,
                         retain_mask: torch.Tensor,
                         query_states: torch.Tensor,
                         key_states: torch.Tensor,
                         value_states: torch.Tensor,
                         hidden_states: torch.Tensor,
                         input_ids: torch.Tensor,
                         position_ids: torch.Tensor,
                         proc_indices: torch.Tensor,
                         orig_q_lens: torch.Tensor,
                         new_q_lens: torch.Tensor,
                         rotary_cos: torch.Tensor,
                         rotary_sin: torch.Tensor,
                         residual_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                  torch.Tensor, torch.Tensor, torch.Tensor,
                                                                  torch.Tensor, torch.Tensor,
                                                                  Tuple[torch.Tensor, torch.Tensor],
                                                                  torch.Tensor]:
    """Fuse state compaction (q/k/v, hidden, indices, q_lens) into a Triton kernel.
    Rotary embeddings (cos/sin) are compacted alongside the other tensors when provided.
    """
    total_tokens = retain_mask.numel()
    keep_offsets = _focus_compute_keep_offsets(retain_mask)

    q_shape = query_states.shape
    k_shape = key_states.shape
    v_shape = value_states.shape
    hidden_shape = hidden_states.shape
    ids_shape = input_ids.shape
    pos_shape = position_ids.shape

    # Caller zeros new_q_lens ahead of time to hide the initialization cost.
    q_out = query_states.new_empty((keep_tokens, q_shape[1], q_shape[2]))
    k_out = key_states.new_empty((keep_tokens, k_shape[1], k_shape[2]))
    v_out = value_states.new_empty((keep_tokens, v_shape[1], v_shape[2]))
    hidden_out = hidden_states.new_empty((hidden_shape[0], keep_tokens, hidden_shape[2]))
    residual_shape = residual_states.shape
    residual_out = residual_states.new_empty((residual_shape[0], keep_tokens, residual_shape[2]))
    input_ids_out = input_ids.new_empty((ids_shape[0], keep_tokens))
    position_ids_out = position_ids.new_empty((pos_shape[0], keep_tokens))
    new_proc_indices = proc_indices.new_empty((keep_tokens, ))
    cos_out = rotary_cos.new_empty((keep_tokens, ) + rotary_cos.shape[1:])
    sin_out = torch.empty_like(cos_out)
    rotary_cos_view = rotary_cos.reshape(total_tokens, -1)
    rotary_sin_view = rotary_sin.reshape(total_tokens, -1)
    cos_out_view = cos_out.reshape(keep_tokens, -1)
    sin_out_view = sin_out.reshape(keep_tokens, -1)
    rotary_dim = rotary_cos_view.shape[1]
    residual_batch = residual_shape[0]
    residual_dim = residual_shape[2]

    stride_qt, stride_qh, stride_qd = query_states.stride()
    stride_qot, stride_qoh, stride_qod = q_out.stride()
    stride_kt, stride_kh, stride_kd = key_states.stride()
    stride_kot, stride_koh, stride_kod = k_out.stride()
    stride_vt, stride_vh, stride_vd = value_states.stride()
    stride_vot, stride_voh, stride_vod = v_out.stride()
    stride_hb, stride_ht, stride_hd = hidden_states.stride()
    stride_hob, stride_hot, stride_hod = hidden_out.stride()
    stride_res_b, stride_res_t, stride_res_d = residual_states.stride()
    stride_res_ob, stride_res_ot, stride_res_od = residual_out.stride()
    stride_ib, stride_it = input_ids.stride()
    stride_iob, stride_iot = input_ids_out.stride()
    stride_pb, stride_pt = position_ids.stride()
    stride_pob, stride_pot = position_ids_out.stride()
    stride_cos_t, stride_cos_d = rotary_cos_view.stride()
    stride_sin_t, stride_sin_d = rotary_sin_view.stride()
    stride_cos_ot, stride_cos_od = cos_out_view.stride()
    stride_sin_ot, stride_sin_od = sin_out_view.stride()

    grid = (total_tokens, )
    # Focus compaction runs every decode step; fixed launch params avoid repeated
    # device capability queries without impacting throughput on modern GPUs.
    num_warps, num_stages = 4, 3
    _focus_compact_states_kernel[grid](retain_mask,
                                        keep_offsets,
                                        proc_indices,
                                        orig_q_lens,
                                        query_states,
                                        key_states,
                                        value_states,
                                        hidden_states,
                                        residual_states,
                                        input_ids,
                                        position_ids,
                                        rotary_cos_view,
                                        rotary_sin_view,
                                        input_ids_out,
                                        position_ids_out,
                                        cos_out_view,
                                        sin_out_view,
                                        q_out,
                                        k_out,
                                        v_out,
                                        hidden_out,
                                        residual_out,
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
                                        stride_res_b,
                                        stride_res_t,
                                        stride_res_d,
                                        stride_res_ob,
                                        stride_res_ot,
                                        stride_res_od,
                                        stride_ib,
                                        stride_it,
                                        stride_iob,
                                        stride_iot,
                                        stride_pb,
                                        stride_pt,
                                        stride_pob,
                                        stride_pot,
                                        stride_cos_t,
                                        stride_cos_d,
                                        stride_sin_t,
                                        stride_sin_d,
                                        stride_cos_ot,
                                        stride_cos_od,
                                        stride_sin_ot,
                                        stride_sin_od,
                                        orig_q_lens.stride(0),
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
                                        residual_batch,
                                        residual_dim,
                                        rotary_dim,
                                        orig_q_lens.size(0),
                                        BLOCK_Q_HEAD=4,
                                        BLOCK_Q_DIM=64,
                                        BLOCK_K_HEAD=4,
                                        BLOCK_V_HEAD=4,
                                        BLOCK_H_BATCH=4,
                                        BLOCK_H_DIM=64,
                                        BLOCK_ID_BATCH=4,
                                        BLOCK_POS_BATCH=4,
                                        BLOCK_ROT_DIM=64,
                                        num_warps=num_warps,
                                        num_stages=num_stages)

    new_rotary = (cos_out, sin_out)

    return (q_out, k_out, v_out, hidden_out, input_ids_out, position_ids_out, new_q_lens,
            new_proc_indices, new_rotary, residual_out)


def focus_update_processing_metadata(
        new_q_lens: torch.Tensor, new_proc_indices: torch.Tensor, history_lengths: torch.Tensor,
        num_ignored_history: torch.Tensor,
        block_progress: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse prefix sums/metadata refresh for delayed cache updates."""
    batch_size = new_q_lens.size(0)
    q_start_loc = torch.empty_like(new_q_lens)
    kv_seqlens = history_lengths.new_empty((batch_size, ))
    cu_q = torch.empty((batch_size + 1, ), device=new_q_lens.device, dtype=torch.int32)
    cu_k = torch.empty_like(cu_q)
    grid = (1, )
    _focus_processing_view_kernel[grid](new_q_lens,
                                        new_proc_indices,
                                        history_lengths,
                                        num_ignored_history,
                                        block_progress,
                                        q_start_loc,
                                        cu_q,
                                        kv_seqlens,
                                        cu_k,
                                        batch_size,
                                        new_q_lens.stride(0),
                                        new_proc_indices.stride(0),
                                        history_lengths.stride(0),
                                        num_ignored_history.stride(0),
                                        block_progress.stride(0),
                                        q_start_loc.stride(0),
                                        cu_q.stride(0),
                                        kv_seqlens.stride(0),
                                        cu_k.stride(0),
                                        num_warps=1,
                                        num_stages=1)
    return q_start_loc, cu_q, kv_seqlens, cu_k


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


def _pick_focus_target_meta(n_elements: int, block: int) -> tuple[int, int]:
    """Heuristic num_warps/num_stages for the focus target kernel."""
    major, _ = torch.cuda.get_device_capability()
    if major >= 9:
        if block >= 128 and n_elements >= 256:
            return 4, 3
        if block >= 128:
            return 2, 2
        if block >= 64:
            return 2, 2
        return 1, 2
    if major >= 8:
        if block >= 128:
            return 2, 2
        if block >= 64:
            return 1, 2
        return 1, 1
    return 1, 1


def _pick_focus_kernel_meta(block_d: int, block_row: int) -> tuple[int, int]:
    """Choose num_warps/num_stages based on tile shape and GPU architecture."""
    major, _ = torch.cuda.get_device_capability()
    area = block_d * block_row
    if block_row <= 32 and block_d <= 128:
        if major >= 9:
            return 2, 2
        if major >= 8:
            return 2, 2
        return 1, 1
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
