import math

import pytest
import torch


NEG_INF = -1e4


def _naive_attention(batched_q, batched_kv, bias):
    batched_k, batched_v = batched_kv

    num_heads_q = batched_q.shape[2]
    num_heads_k = batched_k.shape[2]
    head_dim = batched_q.shape[-1]
    group = num_heads_q // num_heads_k

    q = batched_q.transpose(1, 2)
    k = batched_k.permute(0, 2, 3, 1)
    v = batched_v.transpose(1, 2)

    k = k.unsqueeze(2).expand(-1, -1, group, -1, -1).flatten(1, 2)
    v = v.unsqueeze(2).expand(-1, -1, group, -1, -1).flatten(1, 2)

    qk = torch.matmul(q, k) / math.sqrt(head_dim)
    attn_weight = qk + bias[:, None]
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    attn_weight = attn_weight.to(q.dtype)
    attn_output = torch.matmul(attn_weight, v)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


def _div_up(val, other):
    return (val + other - 1) // other


def _make_block_offsets(num_blocks_per_seq, batch_size, device):
    max_blocks = max(num_blocks_per_seq)
    block_ids = torch.arange(max_blocks, device=device, dtype=torch.long)
    base = torch.arange(batch_size, device=device, dtype=torch.long)
    block_offsets = base[:, None] + block_ids[None, :] * batch_size
    return block_offsets


def _scatter_history_cache(cache, history_states, block_offsets, block_size):
    """Populate cache with history tokens."""
    for seq_id, hist in enumerate(history_states):
        if hist.numel() == 0:
            continue
        block_ids = block_offsets[seq_id]
        cursor = 0
        block_idx = 0
        while cursor < hist.size(0):
            chunk = hist[cursor:cursor + block_size]
            block_off = int(block_ids[block_idx].item())
            cache[block_off, :chunk.size(0)] = chunk
            cursor += chunk.size(0)
            block_idx += 1


def _gather_ragged_rows(tensor, index_list):
    """Gather ragged rows from per-sequence tensors."""
    parts = []
    for seq_id, indices in enumerate(index_list):
        if indices.numel() == 0:
            continue
        parts.append(tensor[seq_id, indices])
    if not parts:
        return tensor.new_empty((0, tensor.size(2), tensor.size(3)))
    return torch.cat(parts, dim=0)


def _python_fill_sparse_reference(cache, states, block_offsets, kv_seqlens, q_start_loc, q_seqlens,
                                  processing_indices, block_size):
    """Python impl mirroring sparse fill kernel for validation."""
    cache = cache.clone()
    batch_size = q_seqlens.numel()
    for seq_id in range(batch_size):
        q_len = int(q_seqlens[seq_id].item())
        if q_len == 0:
            continue
        start = int(q_start_loc[seq_id].item())
        seq_indices = processing_indices[start:start + q_len]
        kv_len = int(kv_seqlens[seq_id].item())
        block_idx = (kv_len - 1) // block_size
        block_off = int(block_offsets[seq_id, block_idx].item())
        last_idx = int(seq_indices[-1].item())
        history_len = kv_len - (last_idx + 1)
        block_start = block_idx * block_size
        history_offset = max(history_len - block_start, 0)
        target_rows = (seq_indices + history_offset).long()
        cache[block_off].index_copy_(0, target_rows, states[start:start + q_len])
    return cache


def _ragged_attention_reference(block_q, k_cache, v_cache, block_offsets, kv_seqlens, q_seqlens, processing_indices,
                                history_lens, block_size):
    """Reference implementation for paged_attention_sparse outputs."""
    batch_size = len(processing_indices)
    q_lens = [int(idx.numel()) for idx in processing_indices]
    max_q_len = max(q_lens) if q_lens else 0
    max_kv_len = int(kv_seqlens.max().item()) if kv_seqlens.numel() > 0 else 0
    num_heads_q = block_q.size(2)
    num_heads_k = k_cache.size(2)
    head_dim = block_q.size(-1)
    head_dim_v = v_cache.size(3)

    batched_q = block_q.new_zeros((batch_size, max_q_len, num_heads_q, head_dim))
    batched_k = k_cache.new_zeros((batch_size, max_kv_len, num_heads_k, head_dim))
    batched_v = v_cache.new_zeros((batch_size, max_kv_len, num_heads_k, head_dim_v))
    bias = block_q.new_full((batch_size, max_q_len, max_kv_len), NEG_INF)

    for seq_id in range(batch_size):
        kv_len = int(kv_seqlens[seq_id].item())
        if kv_len > 0:
            rows_k = []
            rows_v = []
            for pos in range(kv_len):
                block_id = pos // block_size
                offset = pos % block_size
                block_off = int(block_offsets[seq_id, block_id].item())
                rows_k.append(k_cache[block_off, offset])
                rows_v.append(v_cache[block_off, offset])
            batched_k[seq_id, :kv_len] = torch.stack(rows_k)
            batched_v[seq_id, :kv_len] = torch.stack(rows_v)

        q_len = q_lens[seq_id]
        if q_len == 0:
            continue
        idx = processing_indices[seq_id]
        batched_q[seq_id, :q_len] = block_q[seq_id, idx]
        hist = int(history_lens[seq_id].item())
        for token_idx, proc_idx in enumerate(idx.tolist()):
            limit = hist + proc_idx + 1
            bias[seq_id, token_idx, :limit] = 0.0

    attn = _naive_attention(batched_q, (batched_k, batched_v), bias)
    outputs = []
    for seq_id, q_len in enumerate(q_lens):
        if q_len == 0:
            continue
        outputs.append(attn[seq_id, :q_len])
    if not outputs:
        return block_q.new_empty((0, num_heads_q, head_dim_v))
    return torch.cat(outputs, dim=0)


@pytest.fixture
def delayed_sparse_inputs():
    torch.manual_seed(3)
    device = 'cuda'
    dtype = torch.float16
    block_size = 16
    num_heads_q = 8
    num_heads_k = 4
    head_dim = 32
    head_dim_v = 48

    cases = [
        {
            'history': 0,
            'proc': torch.arange(block_size, device=device, dtype=torch.long),
        },
        {
            'history': block_size,
            'proc': torch.tensor([4, 5, 6, 10, 15], device=device, dtype=torch.long),
        },
        {
            'history': block_size * 2,
            'proc': torch.tensor([8, 9, 12], device=device, dtype=torch.long),
        },
        {
            'history': block_size * 3,
            'proc': torch.tensor([], device=device, dtype=torch.long),
        },
    ]

    history_lens = torch.tensor([c['history'] for c in cases], device=device, dtype=torch.long)
    processing_list = [c['proc'] for c in cases]
    kv_seqlens = []
    for hist, idx in zip(history_lens.tolist(), processing_list):
        if idx.numel() == 0:
            kv_seqlens.append(hist)
        else:
            kv_seqlens.append(hist + int(idx.max().item()) + 1)
    kv_seqlens = torch.tensor(kv_seqlens, device=device, dtype=torch.long)

    batch_size = len(cases)
    block_k = torch.randn(batch_size, block_size, num_heads_k, head_dim, dtype=dtype, device=device)
    block_v = torch.randn(batch_size, block_size, num_heads_k, head_dim_v, dtype=dtype, device=device)
    block_q = torch.randn(batch_size, block_size, num_heads_q, head_dim, dtype=dtype, device=device)

    history_k = [
        torch.randn(hist, num_heads_k, head_dim, dtype=dtype, device=device) for hist in history_lens.tolist()
    ]
    history_v = [
        torch.randn(hist, num_heads_k, head_dim_v, dtype=dtype, device=device) for hist in history_lens.tolist()
    ]

    k_states = _gather_ragged_rows(block_k, processing_list).contiguous()
    v_states = _gather_ragged_rows(block_v, processing_list).contiguous()
    q_tokens = _gather_ragged_rows(block_q, processing_list).contiguous()

    q_seqlens = torch.tensor([idx.numel() for idx in processing_list], device=device, dtype=torch.long)
    q_start_loc = q_seqlens.cumsum(0) - q_seqlens

    processing_flat = (torch.cat([idx for idx in processing_list if idx.numel() > 0])
                       if any(idx.numel() > 0 for idx in processing_list) else torch.empty(0, device=device,
                                                                                           dtype=torch.long))

    num_blocks_per_seq = [_div_up(hist, block_size) + 1 for hist in history_lens.tolist()]
    block_offsets = _make_block_offsets(num_blocks_per_seq, batch_size, device)
    max_blocks = block_offsets.size(1)
    cache_shape = (batch_size * max_blocks, block_size, num_heads_k, head_dim)
    k_cache = torch.randn(cache_shape, dtype=dtype, device=device)
    v_cache = torch.randn(batch_size * max_blocks, block_size, num_heads_k, head_dim_v, dtype=dtype, device=device)
    _scatter_history_cache(k_cache, history_k, block_offsets, block_size)
    _scatter_history_cache(v_cache, history_v, block_offsets, block_size)

    return {
        'block_size': block_size,
        'num_heads_q': num_heads_q,
        'num_heads_k': num_heads_k,
        'head_dim_v': head_dim_v,
        'history_lens': history_lens,
        'history_k': history_k,
        'history_v': history_v,
        'block_k': block_k,
        'block_v': block_v,
        'block_q': block_q,
        'processing_list': processing_list,
        'processing_indices': processing_flat,
        'q_seqlens': q_seqlens,
        'q_start_loc': q_start_loc,
        'kv_seqlens': kv_seqlens,
        'block_offsets': block_offsets,
        'k_states': k_states,
        'v_states': v_states,
        'q_tokens': q_tokens,
        'k_cache_base': k_cache,
        'v_cache_base': v_cache,
    }


@pytest.fixture
def dense_paged_attention_inputs():
    torch.manual_seed(7)
    device = 'cuda'
    dtype = torch.float16
    block_size = 16
    seq_len = block_size
    num_heads_q = 8
    num_heads_k = 4
    head_dim = 64
    head_dim_v = 80

    history_lens = torch.tensor([0, block_size // 2, block_size * 2], device=device, dtype=torch.long)
    batch_size = history_lens.numel()
    kv_seqlens = history_lens + seq_len

    block_q = torch.randn(batch_size, seq_len, num_heads_q, head_dim, dtype=dtype, device=device)
    q = block_q.reshape(batch_size * seq_len, num_heads_q, head_dim).contiguous()

    k_full = [torch.randn(int(kv_len.item()), num_heads_k, head_dim, dtype=dtype, device=device)
              for kv_len in kv_seqlens]
    v_full = [torch.randn(int(kv_len.item()), num_heads_k, head_dim_v, dtype=dtype, device=device)
              for kv_len in kv_seqlens]

    num_blocks_per_seq = [_div_up(int(kv_len.item()), block_size) for kv_len in kv_seqlens]
    block_offsets = _make_block_offsets(num_blocks_per_seq, batch_size, device)
    max_blocks = block_offsets.size(1)
    k_cache = torch.zeros(batch_size * max_blocks, block_size, num_heads_k, head_dim, dtype=dtype, device=device)
    v_cache = torch.zeros(batch_size * max_blocks, block_size, num_heads_k, head_dim_v, dtype=dtype, device=device)
    _scatter_history_cache(k_cache, k_full, block_offsets, block_size)
    _scatter_history_cache(v_cache, v_full, block_offsets, block_size)

    q_seqlens = torch.full((batch_size, ), seq_len, device=device, dtype=torch.long)
    q_start_loc = q_seqlens.cumsum(0) - q_seqlens
    block_indices = torch.arange(seq_len, device=device, dtype=torch.long)
    processing_indices = torch.empty((q.size(0), ), device=device, dtype=torch.long)
    for b in range(batch_size):
        start = int(q_start_loc[b].item())
        processing_indices[start:start + seq_len] = block_indices
    processing_indices = processing_indices.contiguous()

    return {
        'q': q,
        'k_cache': k_cache,
        'v_cache': v_cache,
        'kv_seqlens': kv_seqlens,
        'block_offsets': block_offsets,
        'q_seqlens': q_seqlens,
        'q_start_loc': q_start_loc,
        'processing_indices': processing_indices,
        'head_dim_v': head_dim_v,
    }


class TestDelayedCacheSparseKernels:

    def test_fill_kv_cache_sparse_scatter(self, delayed_sparse_inputs):
        from lmdeploy.pytorch.kernels.cuda import fill_kv_cache

        data = delayed_sparse_inputs
        if data['processing_indices'].numel() == 0:
            pytest.skip('No tokens to process for sparse fill.')

        k_cache = data['k_cache_base'].clone()
        v_cache = data['v_cache_base'].clone()
        max_q_len = int(data['q_seqlens'].max().item()) if data['q_seqlens'].numel() > 0 else 1

        fill_kv_cache(data['k_states'], data['v_states'], k_cache, v_cache, data['q_start_loc'], data['q_seqlens'],
                      data['kv_seqlens'], max_q_len, data['block_offsets'], processing_indices=data['processing_indices'])

        expected_k = _python_fill_sparse_reference(data['k_cache_base'], data['k_states'], data['block_offsets'],
                                                   data['kv_seqlens'], data['q_start_loc'], data['q_seqlens'],
                                                   data['processing_indices'], data['block_size'])
        expected_v = _python_fill_sparse_reference(data['v_cache_base'], data['v_states'], data['block_offsets'],
                                                   data['kv_seqlens'], data['q_start_loc'], data['q_seqlens'],
                                                   data['processing_indices'], data['block_size'])

        torch.testing.assert_close(k_cache, expected_k, atol=0, rtol=0)
        torch.testing.assert_close(v_cache, expected_v, atol=0, rtol=0)

    def test_sparse_kernel_matches_dense_kernel(self, dense_paged_attention_inputs):
        from lmdeploy.pytorch.kernels.cuda import paged_attention_fwd, paged_attention_sparse

        data = dense_paged_attention_inputs
        q = data['q']
        out_dense = torch.empty(q.size(0), q.size(1), data['head_dim_v'], dtype=q.dtype, device=q.device)
        paged_attention_fwd(q,
                            data['k_cache'],
                            data['v_cache'],
                            out_dense,
                            block_offsets=data['block_offsets'],
                            kv_seqlens=data['kv_seqlens'])

        out_sparse = torch.empty_like(out_dense)
        paged_attention_sparse(q,
                               data['k_cache'],
                               data['v_cache'],
                               out_sparse,
                               block_offsets=data['block_offsets'],
                               kv_seqlens=data['kv_seqlens'],
                               q_start_loc=data['q_start_loc'],
                               q_seqlens=data['q_seqlens'])
        torch.testing.assert_close(out_sparse, out_dense, atol=3e-3, rtol=3e-3)
