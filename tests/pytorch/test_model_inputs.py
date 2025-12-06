import random
from typing import Tuple

import pytest
import torch

from lmdeploy.pytorch.model_inputs import _gather_flat_tensor


def _baseline_gather(flat_tensor: torch.Tensor, full_lengths: torch.Tensor, gathered_lengths: torch.Tensor,
                     gathered_indices: torch.Tensor):
    offsets = torch.cumsum(full_lengths, dim=0) - full_lengths
    expanded_offsets = torch.repeat_interleave(offsets, gathered_lengths)
    absolute_indices = gathered_indices + expanded_offsets
    return flat_tensor.index_select(-1, absolute_indices)


def _optimized_gather(tensor: torch.Tensor, full_lengths: torch.Tensor, gathered_lengths: torch.Tensor,
                      gathered_indices: torch.Tensor):
    return _gather_flat_tensor(tensor, full_lengths, gathered_lengths, gathered_indices)


def _build_random_case(length_device: torch.device,
                       *,
                       force_zero_seq: bool = False,
                       max_batch: int = 6,
                       max_full: int = 64):
    batch = random.randint(1, max_batch)
    zero_idx = random.randrange(batch) if force_zero_seq else None
    full_lengths = []
    gathered_lengths = []
    gathered_chunks = []
    for idx in range(batch):
        seq_len = random.randint(1, max_full)
        full_lengths.append(seq_len)
        if force_zero_seq and idx == zero_idx:
            q_len = 0
        else:
            q_len = random.randint(0, seq_len)
        gathered_lengths.append(q_len)
        if q_len > 0:
            chunk = torch.randint(0, seq_len, (q_len, ), device=length_device, dtype=torch.long)
            gathered_chunks.append(chunk)
    full_lengths = torch.tensor(full_lengths, device=length_device, dtype=torch.long)
    gathered_lengths = torch.tensor(gathered_lengths, device=length_device, dtype=torch.long)
    if gathered_chunks:
        gathered_indices = torch.cat(gathered_chunks, dim=0)
    else:
        gathered_indices = torch.empty((0, ), device=length_device, dtype=torch.long)
    return full_lengths, gathered_lengths, gathered_indices


def _make_source_tensor(total: int, device: torch.device, dtype: torch.dtype, leading_shape: Tuple[int, ...]):
    shape = leading_shape + (total, )
    if dtype == torch.bool:
        return (torch.rand(shape, device=device) > 0.5)
    if dtype.is_floating_point:
        return torch.randn(shape, device=device, dtype=dtype)
    if dtype in (torch.int32, torch.int64):
        return torch.randint(-500, 500, shape, device=device, dtype=dtype)
    raise RuntimeError(f'Unsupported dtype {dtype}')


def _assert_match(tensor: torch.Tensor, full_lengths: torch.Tensor, gathered_lengths: torch.Tensor,
                  gathered_indices: torch.Tensor):
    baseline = _baseline_gather(tensor, full_lengths, gathered_lengths, gathered_indices)
    optimized = _optimized_gather(tensor, full_lengths, gathered_lengths, gathered_indices)
    assert torch.equal(baseline, optimized)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_gather_flat_tensor_matches_baseline(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')

    torch.manual_seed(0)
    random.seed(0)
    for _ in range(64):
        full_lengths, gathered_lengths, gathered_indices = _build_random_case(torch.device(device))
        total = int(full_lengths.sum().item())
        source = torch.arange(total, device=device, dtype=torch.long).unsqueeze(0)
        _assert_match(source, full_lengths, gathered_lengths, gathered_indices)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_gather_flat_tensor_handles_zero_length_sequences(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')

    random.seed(1)
    for _ in range(16):
        full_lengths, gathered_lengths, gathered_indices = _build_random_case(torch.device(device), force_zero_seq=True)
        assert (gathered_lengths == 0).any()
        total = int(full_lengths.sum().item())
        source = torch.arange(total, device=device, dtype=torch.long).unsqueeze(0)
        _assert_match(source, full_lengths, gathered_lengths, gathered_indices)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.int64, torch.bool])
@pytest.mark.parametrize('leading_shape', [(), (2, ), (2, 3)])
def test_gather_flat_tensor_multi_dimensional_inputs(device, dtype, leading_shape):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')

    random.seed(2)
    for _ in range(8):
        full_lengths, gathered_lengths, gathered_indices = _build_random_case(torch.device(device))
        total = int(full_lengths.sum().item())
        source = _make_source_tensor(total, torch.device(device), dtype, leading_shape)
        _assert_match(source, full_lengths, gathered_lengths, gathered_indices)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_gather_flat_tensor_noncontiguous_inputs(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')

    random.seed(3)
    for _ in range(16):
        full_lengths, gathered_lengths, gathered_indices = _build_random_case(torch.device(device))
        total = int(full_lengths.sum().item())
        leading = random.randint(2, 5)
        base = torch.arange(total * leading, device=device, dtype=torch.float32).view(total, leading)
        source = base.transpose(0, 1)
        assert not source.is_contiguous()
        _assert_match(source, full_lengths, gathered_lengths, gathered_indices)
