import torch
from torch import nn
import nncf
from nncf.torch.model_graph_manager import get_module_by_name
import numpy as np
import random
from copy import deepcopy
from nncf.torch.strip_tuned_lora_model import strip_tuned_lora_model
import pytest

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(0)

MAIN_DIM = 4
LAST_DIM = 2
MAIN_WEIGHT = torch.rand(LAST_DIM, MAIN_DIM)


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
        (nncf.CompressWeightsMode.INT4_SYM, torch.float32),
        # (nncf.CompressWeightsMode.INT4_SYM, torch.float16),
        # (nncf.CompressWeightsMode.INT4_SYM, torch.bfloat16),
    )
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