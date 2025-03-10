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

from typing import List, Tuple

import torch

def fp32_accum_wrapper(func):
    def wrapper(tensor_to_sum, ret_tensor):
        half = tensor_to_sum.dtype == torch.float16
        if half:
            tensor_to_sum = tensor_to_sum.astype(torch.float32)
        retval = func(tensor_to_sum, ret_tensor)
        if half:
            retval = retval.astype(torch.float16)
        return retval

    return wrapper


@fp32_accum_wrapper
def sum_like(tensor_to_sum, ref_tensor):
    if ref_tensor.size == 1:
        return tensor_to_sum.sum()

    for dim, size in enumerate(ref_tensor.shape):
        if size == 1:
            tensor_to_sum = tensor_to_sum.sum(dim, keepdim=True)
    return tensor_to_sum


class ReferenceQuantize:
    def forward(
        self, input_: torch.Tensor, input_low: torch.Tensor, input_range: torch.Tensor, levels: int
    ) -> torch.Tensor:
        scale = (levels - 1) / input_range
        output = input_.clip(min=input_low, max=input_low + input_range)
        zero_point = (-input_low * scale).round()
        output -= input_low
        output *= scale
        output -= zero_point
        output = output.round()
        output = output / scale
        return output

    def backward(
        self,
        grad_output: torch.Tensor,
        input_: torch.Tensor,
        input_low: torch.Tensor,
        input_range: torch.Tensor,
        output: torch.Tensor,
        level_low: int,
        level_high: int,
        is_asymmetric: bool = False,
    ) -> List[torch.Tensor]:
        # is_asymmetric is unused, present only to correspond to the CPU signature of calling "backward"
        mask_hi = input_ > (input_low + input_range)
        mask_hi = mask_hi.to(input_.dtype)
        mask_lo = input_ < input_low
        mask_lo = mask_lo.to(input_.dtype)

        mask_in = 1 - mask_hi - mask_lo
        range_sign = torch.sign(input_range)
        err = (output - input_) * torch.reciprocal(input_range * range_sign)
        grad_range = grad_output * (err * mask_in + range_sign * (level_low / level_high) * mask_lo + mask_hi)
        grad_range = sum_like(grad_range, input_range)

        grad_input = grad_output * mask_in

        grad_low = grad_output * (mask_hi + mask_lo)
        grad_low = sum_like(grad_low, input_low)
        return [grad_input, grad_low, grad_range]

    def tune_range(
        self, input_low: torch.Tensor, input_range: torch.Tensor, levels: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_high = input_range + input_low
        input_low[input_low > 0] = 0
        input_high[input_high < 0] = 0
        n = levels - 1
        scale = n / (input_high - input_low)
        scale = scale.to(input_high.dtype)
        zp = torch.round(-input_low * scale)

        new_input_low = torch.where(zp < n, zp / (zp - n) * input_high, input_low)
        new_input_high = torch.where(zp > 0.0, (zp - n) / zp * input_low, input_high)

        range_1 = input_high - new_input_low
        range_2 = new_input_high - input_low

        mask = (range_1 > range_2).to(input_high.dtype)
        inv_mask = abs(1 - mask)

        new_input_low = mask * new_input_low + inv_mask * input_low
        new_input_range = inv_mask * new_input_high + mask * input_high - new_input_low

        return new_input_low, new_input_range


class ReferenceQuantizedFunctions:
    _executor = ReferenceQuantize()
    Quantize_forward = _executor.forward
    Quantize_backward = _executor.backward
