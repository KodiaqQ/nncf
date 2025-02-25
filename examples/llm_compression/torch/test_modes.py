import random
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

import nncf
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.quantization.quantize_functions import TuneRange
from nncf.torch.strip_tuned_lora_model import strip_tuned_lora_model


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(0)

MAIN_DIM = 8
LAST_DIM = 8
MAIN_WEIGHT = torch.rand(LAST_DIM, MAIN_DIM) - 0.5


class TestModel(nn.Module):
    INPUT_SIZE = [1, MAIN_DIM]

    def __init__(self, torch_dtype):
        super().__init__()
        self.linear = nn.Linear(MAIN_DIM, LAST_DIM, bias=False)
        self.linear.weight.data = MAIN_WEIGHT.to(torch_dtype)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.parametrize(
    ("mode", "torch_dtype"),
    (
        # (nncf.CompressWeightsMode.INT4_ASYM, torch.float32),
        # (nncf.CompressWeightsMode.INT4_ASYM, torch.float16),
        # (nncf.CompressWeightsMode.INT4_ASYM, torch.bfloat16),
        # (nncf.CompressWeightsMode.INT4_SYM, torch.float32),
        # (nncf.CompressWeightsMode.INT4_SYM, torch.float16),
        # (nncf.CompressWeightsMode.INT4_SYM, torch.bfloat16),
    ),
)
def test_lora_quantize(mode, torch_dtype):
    model = TestModel(torch_dtype=torch_dtype)
    with torch.no_grad():
        example = torch.ones(model.INPUT_SIZE).to(torch_dtype)
        dataset = [example]

        output = model(example)

        compressed_model = nncf.compress_weights(
            model,
            ratio=1,
            group_size=4,
            mode=mode,
            backup_mode=None,
            dataset=nncf.Dataset(dataset),
            all_layers=True,
        )

        strip_none = compressed_model(example)

        float_compressed_model = deepcopy(compressed_model)
        for _, quantizer in float_compressed_model._nncf.external_quantizers.items():
            layer = get_module_by_name(quantizer.module_name, float_compressed_model)
            FQ_W = quantizer.quantize(layer.weight)
            layer.weight = torch.nn.Parameter(FQ_W)
        float_compressed_model._nncf.external_quantizers = None
        ctx = float_compressed_model._nncf.get_tracing_context()
        ctx.disable_tracing()
        ctx._post_hooks = {}
        ctx._pre_hooks = {}

        strip_to_float = float_compressed_model(example)

        strip_compressed_model = deepcopy(compressed_model)
        strip_compressed_model = strip_tuned_lora_model(strip_compressed_model)

        strip_to_decompress = strip_compressed_model(example)

        print(f" dtype: {torch_dtype}")
        print(f" original:            {output}")
        print(f" strip_none:          {strip_none}")
        print(f" strip_to_float:      {strip_to_float}")
        print(f" strip_to_decompress: {strip_to_decompress}")
        # assert torch.allclose(output, compressed_output)
        assert torch.allclose(strip_none, strip_to_float)
        assert torch.allclose(strip_none, strip_to_decompress)


def common_q_dq(weight, num_bits, reduction_axes, asymmetric=False):
    if asymmetric:
        level_low = 0
        level_high = 2**num_bits - 1
        min_values = torch.amin(weight, axis=reduction_axes, keepdims=True)
        max_values = torch.amax(weight, axis=reduction_axes, keepdims=True)

        levels = level_high - level_low + 1
        scale = (max_values - min_values) / (levels - 1)

        # eps = torch.finfo(scale.dtype).eps

        # scale = torch.where(torch.abs(scale) < eps, eps, scale)
        zero_point = level_low - torch.round(min_values / scale)
        zero_point = torch.clip(zero_point, level_low, level_high)
    else:
        level_low = -(2 ** (num_bits - 1))
        level_high = 2 ** (num_bits - 1)

        w_abs_min = torch.abs(torch.amin(weight, axis=reduction_axes, keepdims=True))
        w_max = torch.amax(weight, axis=reduction_axes, keepdims=True)

        scale = torch.where(w_abs_min >= w_max, w_abs_min, -w_max)
        scale /= level_high

        # eps = torch.finfo(scale.dtype).eps
        # scale = torch.where(torch.abs(scale) < eps, eps, scale)

    compressed_weights = weight / scale
    if asymmetric:
        compressed_weights += zero_point
    compressed_weights = torch.round(compressed_weights)
    compressed_weights = torch.clip(compressed_weights, level_low, level_high)

    decompressed_weights = compressed_weights
    if asymmetric:
        decompressed_weights -= zero_point
    decompressed_weights = decompressed_weights * scale

    return decompressed_weights


def universal_q_dq(weight, num_bits, reduction_axes, asymmetric=False):
    if asymmetric:
        # eps = 1e-16
        levels = 2**num_bits
        level_high = levels - 1
        level_low = 0

        input_low = torch.amin(weight, reduction_axes, keepdim=True)
        input_high = torch.amax(weight, reduction_axes, keepdim=True)
        input_range = input_high - input_low

        # input_range = input_range - eps

        # End of quantizer calculation

        # input_range_safe = abs(input_range) + eps
        # input_low, input_range = TuneRange.apply(input_low, input_range_safe, levels)

        scale = (levels - 1) / input_range
        output = weight.clip(min=input_low, max=input_low + input_range)
        zero_point = (-input_low * scale).round()
        output -= input_low
        output *= scale
        output -= zero_point
        output = output.round()
        output = output / scale
    else:
        signed = True
        levels = 2**num_bits

        if signed:
            level_high = (levels // 2) - 1
            level_low = -(levels // 2)
            ll_lh = level_low / level_high
            input_low = torch.amin(weight, reduction_axes, keepdim=True)
            input_high = torch.amax(weight, reduction_axes, keepdim=True)
            w_abs_min = torch.abs(input_low)
            w_max = input_high
            scale = torch.where(w_abs_min >= w_max, w_abs_min, -w_max)
            # eps = 1e-16

            # scale = torch.where(torch.abs(scale) < eps, eps, scale)
            input_low = torch.where(scale > 0, -scale, -scale / ll_lh)
            input_range = torch.abs((2 + 1 / level_low) * scale)
            # scale = torch.where(torch.abs(scale) < eps, eps, scale)

            # End of quantizer calculation

            # scale_safe = torch.where(torch.abs(scale) < eps, eps, scale)
            # input_low = torch.where(scale_safe > 0, -scale_safe, -scale_safe / ll_lh)
            # input_range = torch.abs((2 + 1 / level_low) * scale_safe)
        else:
            level_high = levels - 1
            level_low = 0
            scale = input_high
            # scale = input_high - eps
            input_low = scale * ll_lh
            input_range = scale - input_low

            # End of quantizer calculation

            # scale_safe = abs(scale) + eps
            # input_low = scale_safe * ll_lh
            # input_range = scale_safe - input_low

        scale = (levels - 1) / input_range
        output = weight.clip(min=input_low, max=input_low + input_range)
        zero_point = (-input_low * scale).round()
        output -= input_low
        output *= scale
        output -= zero_point
        output = output.round()
        output = output / scale

    return output


def test_methods_equality():
    # weight = torch.Tensor([[-0.95, -0.7, -0.45, -0.2, -0.05, 0.3, 0.55, 0.8]]).to(torch.float32)
    weight = torch.randn(1, 8) - 0.5
    weight = weight.to(torch.float32)

    num_bits = 4
    reduction_axes = -1

    print(f"\nInitial weight:")
    print(f"    weight:      {weight}")

    common_q_dq_output_asym = common_q_dq(
        weight=weight, num_bits=num_bits, reduction_axes=reduction_axes, asymmetric=True
    )
    print(f"Common asymmetric output:")
    print(f"    q-dq weight: {common_q_dq_output_asym}")

    common_q_dq_output_sym = common_q_dq(
        weight=weight, num_bits=num_bits, reduction_axes=reduction_axes, asymmetric=False
    )
    print(f"Common symmetric output:")
    print(f"    q-dq weight: {common_q_dq_output_sym}")

    universal_q_dq_output_asym = universal_q_dq(
        weight=weight, num_bits=num_bits, reduction_axes=reduction_axes, asymmetric=True
    )
    print(f"Universal asymmetric output:")
    print(f"    q-dq weight: {universal_q_dq_output_asym}")

    universal_q_dq_output_sym = common_q_dq(
        weight=weight, num_bits=num_bits, reduction_axes=reduction_axes, asymmetric=False
    )
    print(f"Universal symmetric output:")
    print(f"    q-dq weight: {universal_q_dq_output_sym}")

    sym_close = torch.allclose(common_q_dq_output_sym, universal_q_dq_output_sym)
    asym_close = torch.allclose(common_q_dq_output_asym, universal_q_dq_output_asym)

    assert sym_close and asym_close
