# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def custom_forward(
    input__ptr,
    input_low_ptr,
    input_range_ptr,
    levels,
    output_ptr,
    last_dim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)[:]
    mask = tl.full([BLOCK_SIZE], True, tl.int1)
    d_offset = offset // last_dim

    input_ = tl.load(input__ptr + offset, mask=mask).to(tl.float32)
    input_low = tl.load(input_low_ptr + d_offset, mask=mask, eviction_policy="evict_last").to(tl.float32)
    input_range = tl.load(input_range_ptr + d_offset, mask=mask, eviction_policy="evict_last").to(tl.float32)

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
    levels,
    level_low,
    level_high,
    grad_input_ptr,
    grad_low_ptr,
    grad_range_ptr,
    last_dim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)[:]
    mask = tl.full([BLOCK_SIZE], True, tl.int1)
    d_offset = offset // last_dim

    input_ = tl.load(input__ptr + offset, mask=mask).to(tl.float32)
    input_low = tl.load(input_low_ptr + d_offset, mask=mask, eviction_policy="evict_last").to(tl.float32)
    input_range = tl.load(input_range_ptr + d_offset, mask=mask, eviction_policy="evict_last").to(tl.float32)
    grad_output = tl.load(grad_output_ptr + (offset), mask=mask).to(tl.float32)

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
    shape = tuple(input_.shape)
    last_dim = shape[-1]

    output = torch.empty_like(input_)

    n_elements = input_.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    custom_forward[grid](input_, input_low, input_range, levels, output, last_dim, n_elements, BLOCK_SIZE=512)

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
    shape = tuple(input_.shape)
    last_dim = shape[-1]

    grad_input = torch.empty_like(input_)
    grad_low = torch.empty_like(input_low)
    grad_range = torch.empty_like(input_range)

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
        last_dim,
        n_elements,
        BLOCK_SIZE=512,
    )

    return grad_input, grad_low, grad_range
