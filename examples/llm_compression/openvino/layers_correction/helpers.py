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

import subprocess
import sys
from typing import Callable, Iterable, Optional, TypeVar

import numpy as np
import openvino as ov

import nncf

DataItem = TypeVar("DataItem")
ModelInput = TypeVar("ModelInput")


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


def custom_transform_func(item, tokenizer, ov_model, config, data_name):
    input_dtypes = {inp.get_any_name(): inp.get_element_type() for inp in ov_model.inputs}
    input_names = [inp.get_any_name() for inp in ov_model.inputs]
    tokens = tokenizer(item[data_name])
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


def evaluate(experiment_path, model_path, task, limit):
    print("Validation")
    cmd_line = f"{sys.executable} -m lm_eval"
    cmd_line += f" --model_args pretrained={model_path},convert_tokenizer=true,trust_remote_code=true"
    cmd_line += f" --output_path {experiment_path}"
    cmd_line += " --device cpu"
    cmd_line += f" --task {task}"
    cmd_line += " --model openvino"
    cmd_line += " --batch_size 1"
    cmd_line += f" --limit {limit}"
    print(f"Running command: {cmd_line}")
    subprocess.run(cmd_line, check=True, shell=True, stdout=subprocess.DEVNULL)
