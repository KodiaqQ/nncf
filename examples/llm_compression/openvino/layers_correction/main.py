# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeVar, List

import numpy as np
import openvino as ov
from datasets import load_dataset
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig

import nncf
from nncf.parameters import CompressWeightsMode
import sys
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters

DataItem = TypeVar("DataItem")
ModelInput = TypeVar("ModelInput")

def compress_model(
    ov_model: ov.Model, nncf_dataset: nncf.Dataset, mode: CompressWeightsMode, ratio: float, group_size: int, awq: bool, layers_to_correct_after: List[str]
) -> ov.Model:
    """
    Compress the given OpenVINO model using NNCF weight compression.

    :param ov_model: The original OpenVINO model to be compressed.
    :param nncf_dataset: A representative dataset for the weight compression algorithm.
    :param ratio: The ratio between baseline and backup precisions
    :param group_size: Number of weights (e.g. 128) in the channel dimension
        that share quantization parameters (scale).
    :param awq: Indicates whether use AWQ weights correction.
    :return: The OpenVINO model with compressed weights.
    """
    optimized_ov_model = nncf.compress_weights(
        ov_model,
        dataset=nncf_dataset,
        mode=mode,
        ratio=ratio,
        group_size=group_size,
        awq=awq,
        sensitivity_metric=nncf.parameters.SensitivityMetric.MAX_ACTIVATION_VARIANCE,
        advanced_parameters=AdvancedCompressionParameters(
            layers_to_correct_after=layers_to_correct_after
        )
    )
    return optimized_ov_model


def get_nncf_dataset(
    data_source: Iterable[DataItem], transform_func: Optional[Callable[[DataItem], ModelInput]] = None
) -> nncf.Dataset:
    """
    Create an NNCF dataset for the weight compression algorithm.

    :param data_source: The iterable object serving as the source of data items.
    :param transform_func: The transformation function applied to the data_source.
    :return: nncf_dataset: NNCF Dataset for the weight compression algorithm.
    """
    if data_source is None:
        return None
    if transform_func:
        return nncf.Dataset(data_source, transform_func)
    return nncf.Dataset(data_source)


def custom_transform_func(item, tokenizer, ov_model, config):
    input_dtypes = {inp.get_any_name(): inp.get_element_type() for inp in ov_model.inputs}
    input_names = [inp.get_any_name() for inp in ov_model.inputs]
    tokens = tokenizer(item["text"])
    input_ids = np.expand_dims(np.array(tokens["input_ids"]), 0)
    attention_mask = np.expand_dims(np.array(tokens["attention_mask"]), 0)
    position_ids = np.cumsum(attention_mask, axis=1) - 1
    position_ids[attention_mask == 0] = 1
    res = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    if "position_ids" in input_names:
        res["position_ids"] = position_ids.reshape(*attention_mask.shape)

    batch_size = input_ids.shape[0]

    if config.model_type == "bloom":
        batch_size *= config.num_attention_heads

    if "beam_idx" in input_names:
        res["beam_idx"] = np.arange(batch_size, dtype=int)

    def gen_pkv(num_heads, head_dim, num_layers):
        res = {}
        shape = (1, num_heads, 0, head_dim)
        for i in range(num_layers):
            key_name = f"past_key_values.{i}.key"
            val_name = f"past_key_values.{i}.value"
            res[key_name] = ov.Tensor(shape=shape, type=input_dtypes[key_name])
            res[val_name] = ov.Tensor(shape=shape, type=input_dtypes[val_name])
        return res
    
    state_inputs_len = len(input_names - res.keys()) // 2
    state_input_shape = ov_model.inputs[-1].partial_shape.get_min_shape()
    num_heads = state_input_shape[1]
    head_dim = state_input_shape[3]
    res.update(gen_pkv(num_heads, head_dim, state_inputs_len))
    return res


def main(float_model_path, output_model_path):
    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
    config = AutoConfig.from_pretrained(float_model_path, trust_remote_code=True)
    model = OVModelForCausalLM.from_pretrained(
        float_model_path,
        trust_remote_code=True,
        use_cache=True,
        ov_config=ov_config,
        load_in_8bit=False,
        config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(float_model_path, trust_remote_code=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")  # <YOUR_DATASET>
    dataset = dataset.filter(lambda example: len(example["text"]) > 128)
    transform_func = partial(custom_transform_func, tokenizer=tokenizer, ov_model=model.model, config=config)
    nncf_dataset = get_nncf_dataset(dataset, transform_func)

    optimized_model = compress_model(
        ov_model=model.model,
        nncf_dataset=nncf_dataset,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=1.0,
        group_size=128,
        awq=False,
        layers_to_correct_after=[
            # stable-zephyr-3b-dpo
            "__module.model.layers.0.mlp.down_proj/aten::linear/MatMul_349",
            "__module.model.layers.1.mlp.down_proj/aten::linear/MatMul_550",
            "__module.model.layers.2.mlp.down_proj/aten::linear/MatMul_751",
            "__module.model.layers.3.mlp.down_proj/aten::linear/MatMul_952",
            "__module.model.layers.4.mlp.down_proj/aten::linear/MatMul_1153",
            "__module.model.layers.5.mlp.down_proj/aten::linear/MatMul_1354",
            "__module.model.layers.6.mlp.down_proj/aten::linear/MatMul_1555",
            "__module.model.layers.7.mlp.down_proj/aten::linear/MatMul_1756",
            "__module.model.layers.8.mlp.down_proj/aten::linear/MatMul_1957",
            "__module.model.layers.9.mlp.down_proj/aten::linear/MatMul_2158",
            "__module.model.layers.10.mlp.down_proj/aten::linear/MatMul_2359",
            "__module.model.layers.11.mlp.down_proj/aten::linear/MatMul_2560",
            "__module.model.layers.12.mlp.down_proj/aten::linear/MatMul_2761",
            "__module.model.layers.13.mlp.down_proj/aten::linear/MatMul_2962",
            "__module.model.layers.14.mlp.down_proj/aten::linear/MatMul_3163",
            "__module.model.layers.15.mlp.down_proj/aten::linear/MatMul_3364",
            "__module.model.layers.16.mlp.down_proj/aten::linear/MatMul_3565",
            "__module.model.layers.17.mlp.down_proj/aten::linear/MatMul_3766",
            "__module.model.layers.18.mlp.down_proj/aten::linear/MatMul_3967",
            "__module.model.layers.19.mlp.down_proj/aten::linear/MatMul_4168",
            "__module.model.layers.20.mlp.down_proj/aten::linear/MatMul_4369",
            "__module.model.layers.21.mlp.down_proj/aten::linear/MatMul_4570",
            "__module.model.layers.22.mlp.down_proj/aten::linear/MatMul_4771",
            "__module.model.layers.23.mlp.down_proj/aten::linear/MatMul_4972",
            "__module.model.layers.24.mlp.down_proj/aten::linear/MatMul_5173",
            "__module.model.layers.25.mlp.down_proj/aten::linear/MatMul_5374",
            "__module.model.layers.26.mlp.down_proj/aten::linear/MatMul_5575",
            "__module.model.layers.27.mlp.down_proj/aten::linear/MatMul_5776",
            "__module.model.layers.28.mlp.down_proj/aten::linear/MatMul_5977",
            "__module.model.layers.29.mlp.down_proj/aten::linear/MatMul_6178",
            "__module.model.layers.30.mlp.down_proj/aten::linear/MatMul_6379",
            "__module.model.layers.31.mlp.down_proj/aten::linear/MatMul_6580",
            # "__module.model.norm/aten::layer_norm/Add"
        ]
    )
    optimized_model_path = Path(output_model_path)
    optimized_model_path.mkdir(exist_ok=True)
    model.config.save_pretrained(optimized_model_path.as_posix())
    ov.save_model(optimized_model, optimized_model_path.joinpath("openvino_model.xml"))

if __name__ == "__main__":
    try:
        # main.py stable-zephyr-3b-dpo/pytorch/dldt/FP16 stable-zephyr-3b-dpo/pytorch/dldt/custom_blocks_ends/
        main(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(e)
