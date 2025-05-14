import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu


@triton.jit
def custom_forward(
    input__ptr,
    input_low_ptr,
    input_range_ptr,
    levels_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)[:]
    mask = tl.full([BLOCK_SIZE], True, tl.int1)
    d_offset = offset // 128256

    input_ = tl.load(input__ptr + offset, mask=mask).to(tl.float32)
    input_low = tl.load(input_low_ptr + d_offset, mask=mask, eviction_policy="evict_last").to(tl.float32)
    input_range = tl.load(input_range_ptr + d_offset, mask=mask, eviction_policy="evict_last").to(tl.float32)
    levels = tl.load(levels_ptr + (0))

    # Clip operation
    output_clip_ = triton_helpers.maximum(input_, input_low)
    input_high = input_low + input_range
    output_clip = triton_helpers.minimum(output_clip_, input_high)

    # Input low from output subtraction
    output_sub_1 = output_clip - input_low

    # Scale calculation
    ones = tl.full([1], 1, tl.int32)
    scale_ = ones / input_range
    scale = scale_ * levels

    # Output scaling
    output_scale = output_sub_1 * scale

    # Zero point calculation
    s_input_low = -input_low
    zero_point_ = s_input_low * scale
    zero_point = libdevice.nearbyint(zero_point_)

    # Zero point from output subtraction
    output_sub_2 = output_scale - zero_point

    # Output descaling
    output_ = libdevice.nearbyint(output_sub_2)
    output = output_ / scale

    tl.store(output_ptr + offset, output, None)


@triton.jit
def custom_backward(
    grad_output_ptr,
    input__ptr,
    input_low_ptr,
    input_range_ptr,
    levels_ptr,
    level_low_ptr,
    level_high_ptr,
    grad_input_ptr,
    grad_low_ptr,
    grad_range_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)[:]
    mask = tl.full([BLOCK_SIZE], True, tl.int1)
    d_offset = offset // 128256

    input_ = tl.load(input__ptr + offset, mask=mask).to(tl.float32)
    input_low = tl.load(input_low_ptr + d_offset, mask=mask, eviction_policy="evict_last").to(tl.float32)
    input_range = tl.load(input_range_ptr + d_offset, mask=mask, eviction_policy="evict_last").to(tl.float32)
    grad_output = tl.load(grad_output_ptr + (offset), mask=mask).to(tl.float32)
    levels = tl.load(levels_ptr + (0))
    level_low = tl.load(level_low_ptr + (0))
    level_high = tl.load(level_high_ptr + (0))

    # Mask high calculation
    input_high = input_low + input_range
    mask_hi_ = input_ > input_high
    mask_hi = mask_hi_.to(tl.float32)

    # Mask low calculation
    mask_lo_ = input_ < input_low
    mask_lo = mask_lo_.to(tl.float32)

    # Mask in calculation
    mask_c = 1.0
    mask_in_ = mask_c - mask_hi
    mask_in = mask_in_ - mask_lo

    # Output calculation
    #   Clip operation
    output_clip_ = triton_helpers.maximum(input_, input_low)
    output_clip = triton_helpers.minimum(output_clip_, input_high)

    #   Input low from output subtraction
    output_sub_1 = output_clip - input_low

    #   Scale calculation
    ones = tl.full([1], 1, tl.int32)
    scale_ = ones / input_range
    scale = scale_ * levels

    #   Output scaling
    output_scale = output_sub_1 * scale

    #   Zero point calculation
    s_input_low = -input_low
    zero_point_ = s_input_low * scale
    zero_point = libdevice.nearbyint(zero_point_)

    #   Zero point from output subtraction
    output_sub_2 = output_scale - zero_point

    #   Output descaling
    output_ = libdevice.nearbyint(output_sub_2)
    output = output_ / scale

    # Error calculation
    err_ = output - input_

    # Signed range calculation
    zeros = tl.full([1], 0, tl.int32)
    input_range_above_zero_ = zeros < input_range
    input_range_above_zero = input_range_above_zero_.to(tl.int8)
    input_range_below_zero_ = input_range < zeros
    input_range_below_zero = input_range_below_zero_.to(tl.int8)
    range_sign_ = input_range_above_zero - input_range_below_zero
    range_sign = range_sign_.to(input_range.dtype)

    # Reciprocal calculation
    reciprocal_ = input_range * range_sign
    reciprocal = ones / reciprocal_

    err = err_ * reciprocal

    # Range gradient calculation
    err_mask_in_ = err * mask_in
    level_low_level_high_div = level_low / level_high
    range_levels_ = range_sign * level_low_level_high_div
    range_mask_lo_ = range_levels_ * mask_lo
    err_range_ = err_mask_in_ + range_mask_lo_
    grad_range_ = err_range_ + mask_hi
    grad_range = grad_output * grad_range_

    # Input gradient calculation
    mask_hi_lo = mask_hi + mask_lo
    grad_input = grad_output * mask_in

    # Low gradient calculation
    grad_low = grad_output * mask_hi_lo

    tl.store(grad_input_ptr + (offset), grad_input, None)
    tl.store(grad_low_ptr + (d_offset), grad_low, None)
    tl.store(grad_range_ptr + (d_offset), grad_range, None)


def triton_forward(input_, input_low, input_range, levels):
    dtype = input_.dtype
    shape = tuple(input_.shape)
    device = input_.device
    empty_call = empty_strided_cuda if device.type == "cuda" else empty_strided_cpu

    levels = torch.tensor(levels, dtype=torch.float32).to(device)

    output = empty_call(shape, (shape[-1], 1), dtype)

    n_elements = input_.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    custom_forward[grid](input_, input_low, input_range, levels, output, n_elements, BLOCK_SIZE=512)

    return output


def triton_backward(
    grad_output,
    input_,
    input_low,
    input_range,
    levels,
    level_low,
    level_high,
    is_asymmetric=False,
):
    dtype = input_.dtype
    shape = tuple(input_.shape)
    device = input_.device
    empty_call = empty_strided_cuda if device.type == "cuda" else empty_strided_cpu

    levels = torch.tensor(levels, dtype=torch.float32).to(device)
    level_low = torch.tensor(level_low, dtype=torch.float32).to(device)
    level_high = torch.tensor(level_high, dtype=torch.float32).to(device)

    grad_input = empty_call(shape, (shape[-1], 1), dtype)
    grad_low = empty_call((shape[0], 1), (1, 1), dtype)
    grad_range = empty_call((shape[0], 1), (1, 1), dtype)

    n_elements = input_.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    custom_backward[grid](
        grad_output,
        input_,
        input_low,
        input_range,
        levels,
        level_low,
        level_high,
        grad_input,
        grad_low,
        grad_range,
        n_elements,
        BLOCK_SIZE=512,
    )

    return grad_input, grad_low, grad_range
