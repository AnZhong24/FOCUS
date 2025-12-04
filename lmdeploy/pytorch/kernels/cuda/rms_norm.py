# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor

from lmdeploy.pytorch.model_inputs import get_step_ctx_manager

from .utils import get_device_props


@triton.jit
def _compute_rms_norm(x, w, eps: tl.constexpr, N_COLS: tl.constexpr):
    """Compute rms norm."""
    xf = x.to(tl.float32)

    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    out = xf * tl.math.rsqrt(var + eps)
    out = w * out.to(x.dtype)
    return out


@triton.jit
def _flat_rms_norm_kernel(input, weight, output, seq_len, input_row_stride: tl.constexpr, eps: tl.constexpr,
                          N_COLS: tl.constexpr, BLOCK_N: tl.constexpr, NUM_STAGES: tl.constexpr):
    """RMS norm kernel for contiguous [rows, hidden] layout."""
    prog_id = tl.program_id(0)
    prog_stride = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N_COLS

    w = tl.load(weight + offsets, mask=mask)

    x_ptr = input + prog_id * input_row_stride + offsets
    out_ptr = output + prog_id * input_row_stride + offsets
    for _ in tl.range(prog_id, seq_len, prog_stride, num_stages=NUM_STAGES):
        x = tl.load(x_ptr, mask=mask)
        out = _compute_rms_norm(x, w, eps, N_COLS)
        tl.store(out_ptr, out, mask=mask)
        x_ptr += prog_stride * input_row_stride
        out_ptr += prog_stride * input_row_stride


@triton.jit
def _flat_add_rms_norm_kernel(input, weight, residual, output, out_residual, seq_len,
                              input_row_stride: tl.constexpr, residual_row_stride: tl.constexpr, eps: tl.constexpr,
                              N_COLS: tl.constexpr, BLOCK_N: tl.constexpr, NUM_STAGES: tl.constexpr):
    """RMS norm kernel with residual for contiguous layout."""
    prog_id = tl.program_id(0)
    prog_stride = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N_COLS

    w = tl.load(weight + offsets, mask=mask)

    x_ptr = input + prog_id * input_row_stride + offsets
    res_ptr = residual + prog_id * residual_row_stride + offsets
    out_res_ptr = out_residual + prog_id * residual_row_stride + offsets
    out_ptr = output + prog_id * input_row_stride + offsets
    for _ in tl.range(prog_id, seq_len, prog_stride, num_stages=NUM_STAGES):
        x = tl.load(x_ptr, mask=mask)
        res = tl.load(res_ptr, mask=mask)
        new_x = x + res
        tl.store(out_res_ptr, new_x, mask=mask)
        out = _compute_rms_norm(new_x, w, eps, N_COLS)
        tl.store(out_ptr, out, mask=mask)
        x_ptr += prog_stride * input_row_stride
        res_ptr += prog_stride * input_row_stride
        out_ptr += prog_stride * input_row_stride
        out_res_ptr += prog_stride * input_row_stride


@triton.jit
def add_rms_norm_kernel(input, weight, residual, output, out_residual, num_feats, num_groups, stride_ib, stride_ih,
                        stride_id: tl.constexpr, stride_rb, stride_rh, stride_rd: tl.constexpr, stride_ob, stride_oh,
                        stride_od: tl.constexpr, stride_rob, stride_roh, stride_rod: tl.constexpr,
                        has_residual: tl.constexpr, eps: tl.constexpr, N_COLS: tl.constexpr, BLOCK_N: tl.constexpr,
                        NUM_STAGES: tl.constexpr):
    """Rms norm kernel."""
    prog_id = tl.program_id(0)
    prog_stride = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N_COLS

    w = tl.load(weight + offsets, mask=mask)

    x_ptrs = input + offsets * stride_id
    res_ptrs = residual + offsets * stride_rd
    out_res_ptrs = out_residual + offsets * stride_rod
    out_ptrs = output + offsets * stride_od
    for idx in tl.range(prog_id, num_feats, prog_stride, num_stages=NUM_STAGES):
        batch_id = idx // num_groups
        head_id = idx % num_groups
        cur_x_ptrs = x_ptrs + batch_id * stride_ib + head_id * stride_ih
        cur_res_ptrs = res_ptrs + batch_id * stride_rb + head_id * stride_rh
        cur_out_ptrs = out_ptrs + batch_id * stride_ob + head_id * stride_oh
        cur_out_res_ptrs = out_res_ptrs + batch_id * stride_rob + head_id * stride_roh
        x = tl.load(cur_x_ptrs, mask=mask)
        if has_residual:
            res = tl.load(cur_res_ptrs, mask=mask)
            x += res
            tl.store(cur_out_res_ptrs, x, mask=mask)
        out = _compute_rms_norm(x, w, eps, N_COLS)
        tl.store(cur_out_ptrs, out, mask=mask)


def _unsqueeze_to_3d(tensor: Tensor) -> Tensor:
    """Unsqueeze tensor to 3d."""
    if tensor.dim() == 3:
        return tensor
    elif tensor.dim() == 2:
        return tensor.unsqueeze(0)
    elif tensor.dim() == 1:
        return tensor.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f'Unsupported tensor dim {tensor.dim()}')


def _squeeze_to_origin_dim(tensor: Tensor, origin_dim: int) -> Tensor:
    """Squeeze tensor to origin dim."""
    if origin_dim == 3:
        return tensor
    elif origin_dim == 2:
        return tensor.squeeze(0)
    elif origin_dim == 1:
        return tensor.squeeze(0).squeeze(0)
    else:
        raise ValueError(f'Unsupported origin dim {origin_dim}')


def _is_standard_3d_layout(tensor: Tensor) -> bool:
    """Check whether tensor uses the dense [batch, seq, hidden] layout."""
    stride0, stride1, stride2 = tensor.stride()
    size1, size2 = tensor.size(1), tensor.size(2)
    return stride2 == 1 and stride1 == size2 and stride0 == size1 * size2


def _can_use_strided_impl(hidden_states: Tensor, residual: Tensor, out: Tensor, out_residual: Tensor) -> bool:
    tensors = [hidden_states, residual, out, out_residual]
    return all(_is_standard_3d_layout(tensor) for tensor in tensors)


def _flatten_to_rows(tensor: Tensor, feat_size: int) -> Tensor:
    flat = tensor.reshape(-1, feat_size)
    if flat.stride(1) != 1:
        print("Met flat stride != 1")
        flat = flat.contiguous()
    return flat


def _prepare_output_buffer(tensor: Tensor, feat_size: int):
    flat = tensor.reshape(-1, feat_size)
    if flat.stride(1) == 1 and _share_underlying_storage(flat, tensor):
        return flat, None
    buf = tensor.new_empty(flat.shape)
    return buf, flat


def _tensor_data_ptr(tensor: Tensor) -> int:
    storage_fn = getattr(tensor, 'untyped_storage', None)
    if callable(storage_fn):
        return tensor.untyped_storage().data_ptr()
    else:
        print("Met not callable storage fn")
    return tensor.storage().data_ptr()


def _share_underlying_storage(a: Tensor, b: Tensor) -> bool:
    if a.numel() == 0 or b.numel() == 0:
        return False
    return _tensor_data_ptr(a) == _tensor_data_ptr(b)


def _is_delayed_cache_active() -> bool:
    mgr = get_step_ctx_manager()
    if mgr is None:
        return False
    ctx = mgr.current_context()
    if ctx is None:
        return False
    return bool(getattr(ctx, 'use_delayed_cache', False))


def rms_norm(hidden_states: Tensor,
             weight: Tensor,
             eps: float = 1e-6,
             residual: Tensor = None,
             out: Tensor = None,
             out_residual: Tensor = None):
    """Rms norm."""
    assert hidden_states.dim() <= 3
    assert weight.stride(-1) == 1
    feat_size = weight.shape[0]
    assert hidden_states.size(-1) == feat_size

    origin_dim = hidden_states.dim()
    if out is None:
        out = torch.empty_like(hidden_states)
    has_residual = residual is not None
    if has_residual:
        if out_residual is None:
            out_residual = torch.empty_like(residual)
    else:
        residual = hidden_states
        out_residual = out

    shape = hidden_states.shape
    assert residual.shape == shape
    assert out.shape == shape
    assert out_residual.shape == shape

    hidden_states = _unsqueeze_to_3d(hidden_states)
    residual = _unsqueeze_to_3d(residual)
    out = _unsqueeze_to_3d(out)
    out_residual = _unsqueeze_to_3d(out_residual)

    force_flat = _is_delayed_cache_active()
    use_strided_impl = (not force_flat) and _can_use_strided_impl(hidden_states, residual, out, out_residual)

    BLOCK_N = triton.next_power_of_2(feat_size)

    props = get_device_props(hidden_states.device.index)
    num_sm = props['multi_processor_count']
    warps_per_sm = props['warps_per_sm']
    blocks_per_sm = props['blocks_per_sm']

    if use_strided_impl:
        num_feats = hidden_states.numel() // hidden_states.size(-1)
        num_warps = min(triton.cdiv(BLOCK_N, 2048), 4)
        cta_per_sm = min(blocks_per_sm, warps_per_sm // num_warps)
        cta_per_device = num_sm * cta_per_sm
        num_stages = 1

        grid = (min(num_feats, cta_per_device), )
        add_rms_norm_kernel[grid](
            hidden_states,
            weight,
            residual,
            out,
            out_residual,
            num_feats=num_feats,
            num_groups=hidden_states.size(1),
            stride_ib=hidden_states.stride(0),
            stride_ih=hidden_states.stride(1),
            stride_id=hidden_states.stride(2),
            stride_rb=residual.stride(0),
            stride_rh=residual.stride(1),
            stride_rd=residual.stride(2),
            stride_ob=out.stride(0),
            stride_oh=out.stride(1),
            stride_od=out.stride(2),
            stride_rob=out_residual.stride(0),
            stride_roh=out_residual.stride(1),
            stride_rod=out_residual.stride(2),
            has_residual=has_residual,
            eps=eps,
            N_COLS=feat_size,
            BLOCK_N=BLOCK_N,
            NUM_STAGES=num_stages,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        hidden_flat = _flatten_to_rows(hidden_states, feat_size)
        out_buffer, out_view = _prepare_output_buffer(out, feat_size)

        seq_len = hidden_flat.shape[0]
        if seq_len > 0:
            num_warps = min(triton.cdiv(BLOCK_N, 128), 4)
            cta_per_sm = min(blocks_per_sm, warps_per_sm // num_warps)
            cta_per_device = num_sm * cta_per_sm
            num_stages = min(5, triton.cdiv(seq_len, cta_per_device))
            grid = (min(seq_len, cta_per_device), )

            if has_residual:
                residual_flat = _flatten_to_rows(residual, feat_size)
                out_res_buffer, out_res_view = _prepare_output_buffer(out_residual, feat_size)
                _flat_add_rms_norm_kernel[grid](
                    hidden_flat,
                    weight,
                    residual_flat,
                    out_buffer,
                    out_res_buffer,
                    seq_len=seq_len,
                    input_row_stride=hidden_flat.stride(0),
                    residual_row_stride=residual_flat.stride(0),
                    eps=eps,
                    N_COLS=feat_size,
                    BLOCK_N=BLOCK_N,
                    NUM_STAGES=num_stages,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
                if out_res_view is not None:
                    out_res_view.copy_(out_res_buffer)
            else:
                _flat_rms_norm_kernel[grid](
                    hidden_flat,
                    weight,
                    out_buffer,
                    seq_len=seq_len,
                    input_row_stride=hidden_flat.stride(0),
                    eps=eps,
                    N_COLS=feat_size,
                    BLOCK_N=BLOCK_N,
                    NUM_STAGES=num_stages,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

        if out_view is not None:
            out_view.copy_(out_buffer)

    out = _squeeze_to_origin_dim(out, origin_dim)
    out_residual = _squeeze_to_origin_dim(out_residual, origin_dim)
    if has_residual:
        return out, out_residual
    return out


if __name__ == '__main__':
    import time

    def torch_forward(hidden_states, weight, variance_epsilon=1e-6):
        """Pytorch forward."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        return weight * hidden_states.to(input_dtype)

    def test_rms_norm(bsz, ctx_len, feat_len, dtype):
        """Test rms norm."""
        input = torch.empty((bsz, ctx_len, feat_len), dtype=dtype, device='cuda').normal_(mean=0., std=0.5).contiguous()
        weight = torch.empty((feat_len), dtype=dtype, device='cuda').normal_(mean=0., std=0.5).contiguous()
        triton_output = rms_norm(hidden_states=input, weight=weight)
        torch_output = torch_forward(hidden_states=input, weight=weight)
        assert torch.allclose(torch_output, triton_output, atol=1e-2, rtol=0)

        N_REPEATS = 20

        t0 = time.time()
        for _ in range(N_REPEATS):
            torch_forward(hidden_states=input, weight=weight)

        t1 = time.time()
        for _ in range(N_REPEATS):
            rms_norm(hidden_states=input, weight=weight)
        t2 = time.time()

        torch_cost = (t1 - t0) / N_REPEATS * 1000
        triton_cost = (t2 - t1) / N_REPEATS * 1000
        print('input {} weight {} dtype {}\n  torch {:.3f} triton {:.3f} (ms)\n'.format(
            input.shape, weight.shape, dtype, torch_cost, triton_cost))

    test_rms_norm(1, 8128, 5120, torch.float16)
    test_rms_norm(1, 8128, 5120, torch.float32)
    test_rms_norm(1, 992, 128, torch.float16)
    test_rms_norm(1, 65537, 128, torch.float32)
