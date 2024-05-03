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
            "__module.model.layers.0/aten::add/Add_352",
            "__module.model.layers.1/aten::add/Add_553",
            "__module.model.layers.2/aten::add/Add_754",
            "__module.model.layers.3/aten::add/Add_955",
            "__module.model.layers.4/aten::add/Add_1156",
            "__module.model.layers.5/aten::add/Add_1357",
            "__module.model.layers.6/aten::add/Add_1558",
            "__module.model.layers.7/aten::add/Add_1759",
            "__module.model.layers.8/aten::add/Add_1960",
            "__module.model.layers.9/aten::add/Add_2161",
            "__module.model.layers.10/aten::add/Add_2362",
            "__module.model.layers.11/aten::add/Add_2563",
            "__module.model.layers.12/aten::add/Add_2764",
            "__module.model.layers.13/aten::add/Add_2965",
            "__module.model.layers.14/aten::add/Add_3166",
            "__module.model.layers.15/aten::add/Add_3367",
            "__module.model.layers.16/aten::add/Add_3568",
            "__module.model.layers.17/aten::add/Add_3769",
            "__module.model.layers.18/aten::add/Add_3970",
            "__module.model.layers.19/aten::add/Add_4171",
            "__module.model.layers.20/aten::add/Add_4372",
            "__module.model.layers.21/aten::add/Add_4573",
            "__module.model.layers.22/aten::add/Add_4774",
            "__module.model.layers.23/aten::add/Add_4975",
            "__module.model.layers.24/aten::add/Add_5176",
            "__module.model.layers.25/aten::add/Add_5377",
            "__module.model.layers.26/aten::add/Add_5578",
            "__module.model.layers.27/aten::add/Add_5779",
            "__module.model.layers.28/aten::add/Add_5980",
            "__module.model.layers.29/aten::add/Add_6181",
            "__module.model.layers.30/aten::add/Add_6382",
            "__module.model.layers.31/aten::add/Add_6583",
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
