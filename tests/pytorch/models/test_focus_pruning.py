import torch
import pytest

from lmdeploy.pytorch.models.sdar import SDARAttention

_TEST_FOCUS_MAX_BATCH = 8
_TEST_FOCUS_BLOCK_LEN = 16


def _make_focus_attention(num_heads: int = 4, num_kv_heads: int = 2, head_dim: int = 8) -> SDARAttention:
    """Build a lightweight SDARAttention with only the fields required for FOCUS helpers."""
    module = SDARAttention.__new__(SDARAttention)
    module.num_attention_heads = num_heads
    module.num_key_value_heads = num_kv_heads
    module.num_key_value_groups = max(1, num_heads // num_kv_heads)
    module.head_dim = head_dim
    module.scale = head_dim**-0.5
    max_batch = _TEST_FOCUS_MAX_BATCH
    block_len = _TEST_FOCUS_BLOCK_LEN
    dtype = torch.float32
    module._focus_delta_buffer = torch.zeros((max_batch, block_len), dtype=dtype)
    module._focus_blockpos_buffer = torch.zeros((max_batch, block_len), dtype=torch.long)
    module._focus_qpad_buffer = torch.zeros((max_batch, block_len, num_heads, head_dim), dtype=dtype)
    module._focus_kpad_buffer = torch.zeros((max_batch, block_len, num_kv_heads, head_dim), dtype=dtype)
    return module


def _build_mask(seq_lengths, masked_counts):
    mask_indices_parts = []
    mask_indptr = [0]
    offset = 0
    for seq_len, mask_len in zip(seq_lengths, masked_counts):
        token_ids = torch.arange(offset, offset + seq_len)
        mask_indices_parts.append(token_ids[:mask_len])
        mask_indptr.append(mask_indptr[-1] + mask_len)
        offset += seq_len
    total_masked = mask_indptr[-1]
    mask_indices = torch.cat(mask_indices_parts) if total_masked > 0 else torch.empty(0, dtype=torch.long)
    mask_indptr = torch.tensor(mask_indptr, dtype=torch.int32)
    return mask_indices, mask_indptr


def _mask_max_len(mask_indptr: torch.Tensor) -> int:
    """Return the maximum ragged sequence length encoded by mask_indptr."""
    if mask_indptr.numel() <= 1:
        return 0
    lengths = mask_indptr[1:] - mask_indptr[:-1]
    return int(lengths.max().item()) if lengths.numel() > 0 else 0


def _reference_importance(attn: SDARAttention, query_states: torch.Tensor, key_states: torch.Tensor,
                          mask_indices: torch.Tensor, mask_indptr: torch.Tensor) -> torch.Tensor:
    """Compute per-sequence importance by calling the scalar helper sequentially."""
    outputs = []
    for i in range(mask_indptr.numel() - 1):
        seq_start = mask_indptr[i].item()
        seq_end = mask_indptr[i + 1].item()
        if seq_end == seq_start:
            continue
        seq_indices = mask_indices[seq_start:seq_end]
        seq_q = query_states.index_select(0, seq_indices)
        seq_k = key_states.index_select(0, seq_indices)
        outputs.append(attn._calc_focus_importance(seq_q, seq_k))
    if not outputs:
        return torch.empty(0, dtype=query_states.dtype)
    return torch.cat(outputs)


@pytest.mark.parametrize(
    ('seq_lengths', 'masked_counts'),
    [
        ([5], [3]),
        ([5, 4, 3], [3, 2, 3]),
        ([6, 1, 7, 3], [1, 0, 4, 2]),
        ([2, 2, 2, 2], [0, 1, 2, 0]),
    ],
)
def test_focus_importance_matches_iterative_reference(seq_lengths, masked_counts):
    """Vectorized ragged importance must match per-sequence computation in several scenarios."""
    torch.manual_seed(0)
    attn = _make_focus_attention()
    total_tokens = sum(seq_lengths)
    query_states = torch.randn(total_tokens, attn.num_attention_heads, attn.head_dim)
    key_states = torch.randn(total_tokens, attn.num_key_value_heads, attn.head_dim)
    mask_indices, mask_indptr = _build_mask(seq_lengths, masked_counts)
    max_mask_len = _mask_max_len(mask_indptr)

    batched_scores = attn._calc_focus_importance_ragged(query_states, key_states, mask_indices, mask_indptr,
                                                        max_mask_len)
    iterative_scores = _reference_importance(attn, query_states, key_states, mask_indices, mask_indptr)

    torch.testing.assert_close(batched_scores, iterative_scores)


def test_focus_importance_randomized_regression():
    """Stress test ragged importance across random ragged layouts."""
    attn = _make_focus_attention(num_heads=8, num_kv_heads=4, head_dim=16)
    for seed in range(5):
        torch.manual_seed(1234 + seed)
        num_seq = 4
        seq_lengths = torch.randint(1, 6, (num_seq, )).tolist()
        masked_counts = [torch.randint(0, seq_len + 1, (1, )).item() for seq_len in seq_lengths]
        total_tokens = sum(seq_lengths)
        query_states = torch.randn(total_tokens, attn.num_attention_heads, attn.head_dim)
        key_states = torch.randn(total_tokens, attn.num_key_value_heads, attn.head_dim)
        mask_indices, mask_indptr = _build_mask(seq_lengths, masked_counts)
        max_mask_len = _mask_max_len(mask_indptr)
        batched_scores = attn._calc_focus_importance_ragged(query_states, key_states, mask_indices, mask_indptr,
                                                            max_mask_len)
        iterative_scores = _reference_importance(attn, query_states, key_states, mask_indices, mask_indptr)
        torch.testing.assert_close(batched_scores, iterative_scores)


def _select_per_sequence(attn: SDARAttention, delta: torch.Tensor, valid_mask: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
    outputs = []
    for i in range(delta.size(0)):
        outputs.append(attn._select_focus_mask_batch(delta[i:i + 1], valid_mask[i:i + 1], targets[i:i + 1]))
    return torch.cat(outputs, dim=0)


def test_focus_mask_selection_matches_iterative_reference():
    """Selecting focus targets in batch must mirror iterating over each sequence."""
    torch.manual_seed(1)
    attn = _make_focus_attention()
    seq_lengths = torch.tensor([4, 2, 3])
    batch_size = seq_lengths.numel()
    max_len = int(seq_lengths.max().item())
    delta = torch.randn(batch_size, max_len)
    valid_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for idx, length in enumerate(seq_lengths.tolist()):
        valid_mask[idx, :length] = True
    targets = torch.tensor([2, 0, 1], dtype=torch.int64)

    batched = attn._select_focus_mask_batch(delta, valid_mask, targets)
    stacked = _select_per_sequence(attn, delta, valid_mask, targets)

    assert torch.equal(batched, stacked)


def test_focus_mask_selection_randomized_equivalence():
    """Random ragged layouts should still behave identically to sequential selection."""
    attn = _make_focus_attention()
    for seed in range(3):
        torch.manual_seed(2020 + seed)
        batch = 5
        max_len = 6
        valid_lengths = torch.randint(1, max_len + 1, (batch, ))
        valid_mask = torch.zeros(batch, max_len, dtype=torch.bool)
        for idx, length in enumerate(valid_lengths.tolist()):
            valid_mask[idx, :length] = True
        delta = torch.randn(batch, max_len)
        targets = torch.randint(-1, max_len + 1, (batch, ), dtype=torch.int64)
        batched = attn._select_focus_mask_batch(delta, valid_mask, targets)
        sequential = _select_per_sequence(attn, delta, valid_mask, targets)
        assert torch.equal(batched, sequential)


def test_focus_mask_selection_respects_minimum_targets():
    """Positive targets should retain at least the requested number of tokens per sequence."""
    torch.manual_seed(7)
    attn = _make_focus_attention()
    batch = 4
    max_len = 5
    valid_mask = torch.zeros(batch, max_len, dtype=torch.bool)
    valid_mask[0, :5] = True
    valid_mask[1, :3] = True
    valid_mask[2, :4] = True
    valid_mask[3, :2] = True
    delta = torch.randn(batch, max_len)
    targets = torch.tensor([4, 2, 6, -3], dtype=torch.int64)
    selection = attn._select_focus_mask_batch(delta, valid_mask, targets)
    counts = selection.sum(dim=1)
    max_counts = valid_mask.sum(dim=1)
    adjusted_targets = torch.minimum(torch.clamp(targets, min=0), max_counts)
    adjusted_targets = torch.where(adjusted_targets > 0, torch.clamp(adjusted_targets, min=1), adjusted_targets)
    assert torch.all(counts >= adjusted_targets)


def test_focus_rule_enforcement_matches_iterative_reference():
    """Vectorized enforcement should produce identical retain masks as iterating per sequence."""
    attn = _make_focus_attention()
    block_count = 12
    seq_lengths = [4, 3]
    max_len = max(seq_lengths)

    block_positions = torch.full((len(seq_lengths), max_len), -1, dtype=torch.long)
    valid_mask = torch.zeros_like(block_positions, dtype=torch.bool)
    retain_mask = torch.zeros_like(block_positions, dtype=torch.bool)

    block_positions[0, :4] = torch.tensor([2, 3, 4, 8])
    valid_mask[0, :4] = True
    retain_mask[0, :4] = torch.tensor([True, False, True, False])

    block_positions[1, :3] = torch.tensor([5, 6, 9])
    valid_mask[1, :3] = True
    retain_mask[1, :3] = torch.tensor([False, True, False])

    block_unprocessed = torch.zeros((len(seq_lengths), block_count), dtype=torch.bool)
    block_unprocessed[0, 3] = True
    block_unprocessed[0, 4] = True
    block_unprocessed[1, 5] = True
    block_unprocessed[1, 6] = True

    batched = attn._enforce_focus_rules_batch(block_positions.clone(), block_unprocessed.clone(),
                                              retain_mask.clone(), valid_mask.clone())

    per_sequence = []
    for i in range(len(seq_lengths)):
        per_sequence.append(
            attn._enforce_focus_rules_batch(block_positions[i:i + 1].clone(), block_unprocessed[i:i + 1].clone(),
                                            retain_mask[i:i + 1].clone(), valid_mask[i:i + 1].clone()))
    stacked = torch.cat(per_sequence, dim=0)

    assert torch.equal(batched, stacked)


def test_focus_rule_enforcement_preserves_unprocessed_tokens():
    """Tokens marked as unprocessed must be retained even if the selection mask drops them."""
    attn = _make_focus_attention()
    block_positions = torch.tensor([[1, 2, 3, -1], [4, 5, -1, -1]], dtype=torch.long)
    valid_mask = block_positions >= 0
    retain_mask = torch.zeros_like(block_positions, dtype=torch.bool)
    block_unprocessed = torch.zeros((2, 10), dtype=torch.bool)
    block_unprocessed[0, 2] = True
    block_unprocessed[1, 5] = True
    enforced = attn._enforce_focus_rules_batch(block_positions.clone(), block_unprocessed.clone(),
                                               retain_mask.clone(), valid_mask.clone())
    assert enforced[0, 1] and enforced[0, 2]
    assert enforced[1, 1]
