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

import datetime
import json
import sys
from functools import partial
from pathlib import Path

import openvino as ov
from datasets import load_dataset
from optimum.intel import OVModelForCausalLM
from transformers import AutoConfig
from transformers import AutoTokenizer

import nncf
from examples.llm_compression.openvino.layers_correction.helpers import custom_transform_func
from examples.llm_compression.openvino.layers_correction.helpers import evaluate
from examples.llm_compression.openvino.layers_correction.helpers import get_nncf_dataset
from nncf.common.logging.logger import nncf_logger
from nncf.common.logging.logger import set_log_file
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters


def main(float_model_path, output_path, experiment_config):
    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
    data_config = experiment_config["data"]
    config = AutoConfig.from_pretrained(float_model_path, trust_remote_code=True)
    model = OVModelForCausalLM.from_pretrained(
        float_model_path, trust_remote_code=True, use_cache=True, ov_config=ov_config, load_in_8bit=False, config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(float_model_path, trust_remote_code=True)
    dataset = load_dataset(data_config["task"], data_config["dataset_name"], split=data_config["split"])

    custom_dataset = []
    for item_id, data_item in enumerate(dataset):
        if item_id > experiment_config["subset_size"]:
            break
        question = data_item["question"]
        answer = data_item["answer"]
        answer_tokens = tokenizer.tokenize(answer)
        for i in range(1, len(answer_tokens) - 5):
            answer_part = tokenizer.convert_tokens_to_string(answer_tokens[:i])
            final_item = f"{question} + {answer_part}"
            custom_dataset.append({"modified": final_item})

    nncf_logger.info(f"Dataset len: {len(custom_dataset)}")
    # dataset = dataset.filter(lambda example: len(example[data_config["data_name"]]) > 128)
    transform_func = partial(
        custom_transform_func,
        tokenizer=tokenizer,
        ov_model=model.model,
        config=config,
        data_name=data_config["data_name"],
    )
    nncf_dataset = get_nncf_dataset(custom_dataset, transform_func)

    optimized_model = nncf.compress_weights(
        model.model,
        dataset=nncf_dataset,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=1.0,
        group_size=128,
        subset_size=len(custom_dataset),
        awq=False,
        sensitivity_metric=nncf.parameters.SensitivityMetric.MAX_ACTIVATION_VARIANCE,
        advanced_parameters=AdvancedCompressionParameters(
            layers_to_correct=experiment_config["layers_to_correct"],
            fast_correction=experiment_config["fast_correction"],
            correction_type=experiment_config["correction_type"],
            ignore_skip_connection=experiment_config["ignore_skip_connection"],
        ),
    )

    model.config.save_pretrained(output_path.as_posix())
    tokenizer.save_pretrained(output_path.as_posix())
    output_model_path = output_path.joinpath("openvino_model.xml")
    ov.save_model(optimized_model, output_model_path.as_posix())

    del model
    del optimized_model

    evaluate(output_path.as_posix(), output_path.as_posix(), data_config["task"], data_config["val_limit"])


if __name__ == "__main__":
    input_dir = sys.argv[1]
    configurations = [
        {
            "name": "End of model block, extended, each word, tokens, test data, pca",
            "fast_correction": True,
            "correction_type": "pca",
            "subset_size": 10,
            "ignore_skip_connection": False,
            "data": {
                "task": "gsm8k",
                "dataset_name": "main",
                "data_name": "modified",
                "split": "test",
                "val_limit": 100,
            },
            "layers_to_correct": [
                "__module.model.layers.\d+/aten::add/Add_1",
            ],
        },
        {
            "name": "End of the model, modified data, test data, pca",
            "fast_correction": True,
            "correction_type": "pca",
            "subset_size": 100,
            "ignore_skip_connection": False,
            "data": {
                "task": "gsm8k",
                "dataset_name": "main",
                "data_name": "modified",
                "split": "test",
                "val_limit": 100,
            },
            "layers_to_correct": [
                "__module.model.norm/aten::mul/Multiply_1",
            ],
        },
    ]
    for configuration in configurations:
        timestamp = datetime.datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        output_dir = Path(input_dir).parent.joinpath(f"run_{timestamp}")
        output_dir.mkdir(exist_ok=True)

        experiment_name = configuration["name"]
        nncf_logger.info(f"Started experiment: {experiment_name}")

        dump_path = output_dir.joinpath("experiment_config.json")
        log_path = output_dir.joinpath("experiment_log.txt")
        set_log_file(log_path)
        with open(dump_path.as_posix(), "w") as f:
            json.dump(configuration, f, sort_keys=True, indent=4)
            nncf_logger.info(f"Saved experiment configuration to {dump_path.as_posix()}")

        try:
            main(input_dir, output_dir, configuration)
        except Exception as e:
            print(e)
