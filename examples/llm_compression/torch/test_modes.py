import random
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

import nncf
from nncf.torch.model_graph_manager import get_module_by_name
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

MAIN_DIM = 128
LAST_DIM = 128
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
    ("mode", "torch_dtype", "atol"),
    (
        (nncf.CompressWeightsMode.INT4_ASYM, torch.float32, 1e-04),
        (nncf.CompressWeightsMode.INT4_ASYM, torch.float16, 0.01),
        (nncf.CompressWeightsMode.INT4_ASYM, torch.bfloat16, 0.1),
        (nncf.CompressWeightsMode.INT4_SYM, torch.float32, 1e-04),
        (nncf.CompressWeightsMode.INT4_SYM, torch.float16, 0.01),
        (nncf.CompressWeightsMode.INT4_SYM, torch.bfloat16, 0.1),
    ),
)
def test_lora_quantize(mode, torch_dtype, atol):
    model = TestModel(torch_dtype=torch_dtype)
    with torch.no_grad():
        example = torch.ones(model.INPUT_SIZE).to(torch_dtype)
        dataset = [example]

        compressed_model = nncf.compress_weights(
            model,
            ratio=1,
            group_size=128,
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

        assert torch.allclose(strip_none, strip_to_float, atol=atol)
        assert torch.allclose(strip_none, strip_to_decompress, atol=atol)


def lowering_q_dq(weight, num_bits, reduction_axes, asymmetric=False):
    if asymmetric:
        level_low = 0
        level_high = 2**num_bits - 1

        min_values = torch.amin(weight, axis=reduction_axes, keepdims=True)
        max_values = torch.amax(weight, axis=reduction_axes, keepdims=True)

        scale = (max_values - min_values) / level_high

        zero_point = torch.round(-min_values / scale)

        compressed_weights = weight / scale
        compressed_weights = compressed_weights + zero_point
        compressed_weights = torch.round(compressed_weights)
        compressed_weights = torch.clip(compressed_weights, level_low, level_high)

        decompressed_weights = compressed_weights - zero_point
        decompressed_weights = decompressed_weights * scale
    else:
        level_low = -(2 ** (num_bits - 1))
        level_high = 2 ** (num_bits - 1)

        w_abs_min = torch.abs(torch.amin(weight, axis=reduction_axes, keepdims=True))
        w_max = torch.amax(weight, axis=reduction_axes, keepdims=True)

        scale = torch.where(w_abs_min >= w_max, w_abs_min, -w_max)
        scale /= level_high

        compressed_weights = weight / scale
        compressed_weights = torch.round(compressed_weights)
        compressed_weights = torch.clip(compressed_weights, level_low, level_high)
        decompressed_weights = compressed_weights * scale

    return decompressed_weights


def universal_q_dq(weight, num_bits, reduction_axes, asymmetric=False):
    if asymmetric:
        level_low = 0
        level_high = 2**num_bits - 1

        min_values = torch.amin(weight, reduction_axes, keepdim=True)
        max_values = torch.amax(weight, reduction_axes, keepdim=True)

        scale = level_high / (max_values - min_values)

        zero_point = torch.round(-min_values * scale)

        output = torch.clip(weight, min=min_values, max=max_values)
        output = output - min_values
        output = output * scale
        output = output - zero_point
        output = torch.round(output)
        output = output / scale
    else:
        levels = 2**num_bits
        level_high = (levels // 2) - 1
        level_low = -(levels // 2)
        ll_lh = level_low / level_high

        min_values = torch.amin(weight, reduction_axes, keepdim=True)
        max_values = torch.amax(weight, reduction_axes, keepdim=True)
        min_values = torch.abs(min_values)

        scale = torch.where(min_values >= max_values, min_values, -max_values)

        input_low = torch.where(scale > 0, -scale, -scale / ll_lh)
        input_range = torch.abs((2 + 1 / level_low) * scale)

        scale = (levels - 1) / input_range
        zero_point = torch.round(-input_low * scale)

        output = torch.clip(weight, min=input_low, max=input_low + input_range)
        output -= input_low
        output *= scale
        output -= zero_point
        output = torch.round(output)
        output = output / scale

    return output


@pytest.mark.parametrize(
    ("num_bits", "torch_dtype"),
    (
        (4, torch.float32),
        (4, torch.float16),
        (4, torch.bfloat16),
    ),
)
def test_methods_equality(num_bits, torch_dtype):
    weight = MAIN_WEIGHT.to(torch_dtype)
    reduction_axes = -1

    lowering_q_dq_output_asym = lowering_q_dq(
        weight=weight, num_bits=num_bits, reduction_axes=reduction_axes, asymmetric=True
    )
    print(f"Lowering asymmetric output:")
    print(f"    q-dq weight: {lowering_q_dq_output_asym}")

    lowering_q_dq_output_sym = lowering_q_dq(
        weight=weight, num_bits=num_bits, reduction_axes=reduction_axes, asymmetric=False
    )

    universal_q_dq_output_asym = universal_q_dq(
        weight=weight, num_bits=num_bits, reduction_axes=reduction_axes, asymmetric=True
    )
    print(f"Universal asymmetric output:")
    print(f"    q-dq weight: {universal_q_dq_output_asym}")

    universal_q_dq_output_sym = lowering_q_dq(
        weight=weight, num_bits=num_bits, reduction_axes=reduction_axes, asymmetric=False
    )

    sym_close = torch.allclose(lowering_q_dq_output_sym, universal_q_dq_output_sym)
    asym_close = torch.allclose(lowering_q_dq_output_asym, universal_q_dq_output_asym, atol=0.1)

    assert sym_close and asym_close
