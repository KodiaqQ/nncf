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


set_seed(11)

MAIN_DIM = 16
LAST_DIM = 2
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
        (nncf.CompressWeightsMode.INT4_ASYM, torch.float32),
        # (nncf.CompressWeightsMode.INT4_ASYM, torch.float16),
        # (nncf.CompressWeightsMode.INT4_ASYM, torch.bfloat16),
        # (nncf.CompressWeightsMode.INT4_SYM, torch.float32),
        # (nncf.CompressWeightsMode.INT4_SYM, torch.float16),
        # (nncf.CompressWeightsMode.INT4_SYM, torch.bfloat16),
    ),
)
def test_lora_quantize(mode, torch_dtype):
    pytest.skip()
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


def common_quantize_asymmetric(weight, num_bits, reduction_axes):
    level_low = 0
    level_high = 2**num_bits - 1
    min_values = torch.amin(weight, reduction_axes, keepdims=True)
    max_values = torch.amax(weight, reduction_axes, keepdims=True)

    levels = level_high - level_low + 1
    scale = ((max_values - min_values) / (levels - 1)).to(torch.float32)
    eps = torch.finfo(scale.dtype).eps
    scale = torch.where(torch.abs(scale) < eps, eps, scale)

    zero_point = level_low - torch.round(min_values / scale)
    zero_point = torch.clip(zero_point.to(torch.int32), level_low, level_high)

    q_weight = weight / scale
    q_weight = q_weight + zero_point.to(weight.dtype)
    q_weight = torch.round(q_weight)
    q_weight = torch.clip(q_weight, level_low, level_high).to(torch.uint8)

    return q_weight, scale, zero_point


def common_dequantize_asymetric(q_weight, scale, zero_point):
    weight = q_weight - zero_point
    weight = weight * scale
    return weight


def universal_quantize_asymmetric(weight, num_bits, reduction_axes):
    level_low = 0
    level_high = 2**num_bits - 1
    min_values = torch.amin(weight, reduction_axes, keepdims=True)
    max_values = torch.amax(weight, reduction_axes, keepdims=True)

    levels = level_high - level_low + 1
    eps = 1e-16
    input_range = max_values - min_values
    input_range = input_range - eps

    input_range_safe = torch.abs(input_range) + eps
    min_values, input_range = TuneRange.apply(min_values, input_range_safe, levels)

    scale = (levels - 1) / input_range
    zero_point = (-min_values * scale).round()

    q_weight = weight.clip(min=min_values, max=max_values)
    q_weight = q_weight - min_values
    q_weight = q_weight * scale
    q_weight = q_weight.round()

    return q_weight, scale, zero_point


def universal_dequantize_asymetric(q_weight, scale, zero_point):
    weight = q_weight - zero_point
    weight = weight / scale
    return weight


def test_methods_equality():

    weight = torch.Tensor(
        [
            [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75],
        ]
    ).to(torch.float32)

    print(f"\nInitial weight:")
    print(f"    weight: {weight}")

    num_bits = 4
    reduction_axes = 1

    common_quantized_weight, common_scale, common_zero_point = common_quantize_asymmetric(
        weight=weight,
        num_bits=num_bits,
        reduction_axes=reduction_axes,
    )

    common_dequantized_weight = common_dequantize_asymetric(common_quantized_weight, common_scale, common_zero_point)

    print(f"Common quantized asymmetric:")
    print(f"    weight: {common_quantized_weight}")
    print(f"    scale: {common_scale}")
    print(f"    zero point: {common_zero_point}")
    print(f"Common dequantized asymmetric:")
    print(f"    weight: {common_dequantized_weight}")

    universal_quantized_weight, universal_scale, universal_zero_point = universal_quantize_asymmetric(
        weight=weight,
        num_bits=num_bits,
        reduction_axes=reduction_axes,
    )

    universal_dequantized_weight = universal_dequantize_asymetric(
        universal_quantized_weight, universal_scale, universal_zero_point
    )

    print(f"Universal quantized asymmetric:")
    print(f"    weight: {universal_quantized_weight}")
    print(f"    scale: {universal_scale}")
    print(f"    zero point: {universal_zero_point}")
    print(f"Universal dequantized asymmetric:")
    print(f"    weight: {universal_dequantized_weight}")
