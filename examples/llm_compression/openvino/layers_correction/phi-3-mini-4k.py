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

import openvino as ov
from datasets import load_dataset
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig

import nncf
import sys
from examples.llm_compression.openvino.layers_correction.helpers import get_nncf_dataset
from examples.llm_compression.openvino.layers_correction.helpers import compress_model
from examples.llm_compression.openvino.layers_correction.helpers import custom_transform_func
from examples.llm_compression.openvino.layers_correction.helpers import evaluate

DATA_NAME = "question"
TASK = "gsm8k"
DATASET_NAME = "main"
SPLIT = "train[:1000]"
LIMIT = 100


def main(float_model_path, output_path, layers_to_correct = None, fast_correction = True):
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
    dataset = load_dataset(TASK, DATASET_NAME, split=SPLIT)
    dataset = dataset.filter(lambda example: len(example[DATA_NAME]) > 128)
    transform_func = partial(custom_transform_func, tokenizer=tokenizer, ov_model=model.model, config=config, data_name=DATA_NAME)
    nncf_dataset = get_nncf_dataset(dataset, transform_func)

    optimized_model = compress_model(
        ov_model=model.model,
        nncf_dataset=nncf_dataset,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=None,
        group_size=128,
        awq=False,
        layers_to_correct=layers_to_correct if layers_to_correct else [],
        fast_correction=fast_correction,
    )
    optimized_path = Path(output_path)
    optimized_path.mkdir(exist_ok=True)
    model.config.save_pretrained(optimized_path.as_posix())
    tokenizer.save_pretrained(optimized_path.as_posix())
    output_model_path = optimized_path.joinpath("openvino_model.xml")
    ov.save_model(optimized_model, output_model_path.as_posix())

    del model
    del optimized_model

    evaluate(optimized_path.as_posix(), optimized_path.as_posix(), TASK, LIMIT)

if __name__ == "__main__":
    input_dir = sys.argv[1]
    configurations = [
        {
            "name": "post_attention_layernorm_multiply_1_fast",
            "fast_correction": True,
            "layers": [
                "__module.model.layers.0.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.1.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.2.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.3.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.4.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.5.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.6.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.7.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.8.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.9.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.10.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.11.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.12.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.13.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.14.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.15.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.16.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.17.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.18.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.19.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.20.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.21.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.22.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.23.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.24.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.25.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.26.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.27.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.28.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.29.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.30.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.31.post_attention_layernorm/aten::mul/Multiply_1",
            ],
        },
        {
            "name": "post_attention_layernorm_multiply_1_slow",
            "fast_correction": False,
            "layers": [
                "__module.model.layers.0.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.1.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.2.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.3.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.4.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.5.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.6.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.7.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.8.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.9.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.10.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.11.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.12.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.13.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.14.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.15.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.16.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.17.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.18.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.19.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.20.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.21.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.22.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.23.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.24.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.25.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.26.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.27.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.28.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.29.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.30.post_attention_layernorm/aten::mul/Multiply_1",
                "__module.model.layers.31.post_attention_layernorm/aten::mul/Multiply_1",
            ],
        },
    ]
    for configuration in configurations:
        output_dir = Path(input_dir).parent.joinpath(str(LIMIT), configuration["name"]).as_posix()
        try:
            main(input_dir, output_dir, configuration["layers"], configuration["fast_correction"])
        except Exception as e:
            print(e)
