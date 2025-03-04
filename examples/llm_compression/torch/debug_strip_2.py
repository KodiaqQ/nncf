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

import random
from pathlib import Path
from optimum.exporters.openvino.convert import export_from_model
from whowhatbench import TextEvaluator
from nncf.torch import load_from_config
from nncf.torch.model_graph_manager import get_module_by_name
from optimum.intel.openvino import OVModelForCausalLM
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from enum import Enum
from nncf.torch.strip_tuned_lora_model import strip_tuned_lora_model


class StripMode(Enum):
    NONE = "none"
    TO_FLOAT = "to_float"
    TO_DECOMPRESS = "to_decompress"
    TO_OV = "to_ov"

MODE = nncf.CompressWeightsMode.INT4_ASYM
BACKUP_MODE = nncf.BackupMode.INT8_ASYM

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
# MODEL_ID = 'HuggingFaceTB/SmolLM-1.7B-Instruct'
# MODEL_ID = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
GROUP_SIZE = 32
NUM_EVAL_SAMPLES = 1
MAIN_PATH = Path(__file__).parent.resolve()
WWB_REF = MAIN_PATH.joinpath('phi_chat_wwb.csv')
# WWB_REF = MAIN_PATH.joinpath('smollm_chat_wwb.csv')
# WWB_REF = MAIN_PATH.joinpath('ref_wwb.csv')
assert WWB_REF.exists()

def save_checkpoint(wrapped_model, ckpt_path='nncf_checkpoint.pth'):
    nncf_state_dict = wrapped_model.nncf.state_dict()
    nncf_config = wrapped_model.nncf.get_config()
    print(f"Saving ckpt to: {ckpt_path}")
    torch.save(
        {
            "nncf_state_dict": nncf_state_dict,
            "nncf_config": nncf_config,
        },
        ckpt_path,
    )

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def evaluate(tokenizer, model):
    wwb_eval = TextEvaluator(tokenizer=tokenizer, gt_data=WWB_REF, test_data=str(WWB_REF), use_chat_template=True, num_samples=NUM_EVAL_SAMPLES)
    _, all_metrics = wwb_eval.score(model)
    similarity = float(all_metrics["similarity"].iloc[0])
    print("    DEBUG:")
    print(f"     Strip: {strip_mode.value}")
    print(f"     Samples: {NUM_EVAL_SAMPLES}")
    print(f"     Similarity: {similarity}")
    return similarity

def main(strip_mode, eval = True, use_cuda = True, torch_dtype = torch.float32):
    set_seed(42)
    exp_name = MAIN_PATH.joinpath(f'phi_mode-{MODE.value}_backup-{BACKUP_MODE.value}_dtype-{torch_dtype}').as_posix()
    # exp_name = MAIN_PATH.joinpath(f'smollm_mode-{MODE.value}_backup-{BACKUP_MODE.value}_dtype-{torch_dtype}').as_posix()
    # exp_name = MAIN_PATH.joinpath(f'deep_mode-{MODE.value}_backup-{BACKUP_MODE.value}_dtype-{torch_dtype}').as_posix()
    ckpt_path = Path(exp_name + '.pth')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    tokenized_text = tokenizer("example" * 10, return_tensors="pt")
    labels = tokenized_text["input_ids"]
    attention_mask = tokenized_text["attention_mask"]
    input_ids = labels[:, :-1]
    labels = labels[:, 1:]
    position_ids = torch.cumsum(attention_mask, axis=1) - 1
    position_ids[attention_mask == 0] = 1
    if use_cuda:
        model = model.cuda()
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        position_ids = position_ids.cuda()
    dataset = [{"input_ids": input_ids, "attention_mask": attention_mask[:, :-1], "position_ids": position_ids[:, :-1]}]

    if ckpt_path.exists():
        nncf_ckpt = torch.load(ckpt_path)
        model = load_from_config(model, nncf_ckpt["nncf_config"], example_input=dataset[0])
        model.nncf.load_state_dict(nncf_ckpt["nncf_state_dict"])
        if use_cuda: model = model.cuda()
    else:
        model = nncf.compress_weights(
            model,
            ratio=1,
            group_size=GROUP_SIZE,
            mode=MODE,
            backup_mode=BACKUP_MODE,
            dataset=nncf.Dataset(dataset),
        )
        save_checkpoint(model, ckpt_path)

    result_path = Path(exp_name + f"_strip-{strip_mode.value}")
    result_path.mkdir(exist_ok=True, parents=True)
    result_path_ckpt = result_path.joinpath("nncf_checkpoint.pth").as_posix()

    if strip_mode == StripMode.NONE:
        save_checkpoint(model, result_path_ckpt)
    else:
        if strip_mode == StripMode.TO_FLOAT:
            for name, quantizer in model._nncf.external_quantizers.items():
                layer = get_module_by_name(quantizer.module_name, model)
                FQ_W = quantizer.quantize(layer.weight)
                layer.weight = torch.nn.Parameter(FQ_W)
            model._nncf.external_quantizers = None
            ctx = model._nncf.get_tracing_context()
            ctx.disable_tracing()
            ctx._post_hooks = {}
            ctx._pre_hooks = {}
            model.save_pretrained(result_path)
        elif strip_mode in [StripMode.TO_DECOMPRESS, StripMode.TO_OV]:
            model = strip_tuned_lora_model(model)

            if strip_mode == StripMode.TO_DECOMPRESS:
                save_checkpoint(model, result_path_ckpt)

            if strip_mode == StripMode.TO_OV:
                ov_dir = Path(exp_name + '_export')
                ov_dir.mkdir(exist_ok=True, parents=True)
                model = model.cpu()
                export_from_model(model, ov_dir, stateful=False)
                model = OVModelForCausalLM.from_pretrained(
                    model_id=ov_dir,
                    trust_remote_code=True,
                    load_in_8bit=False,
                    compile=True,
                    ov_config={"KV_CACHE_PRECISION": "f16", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"},
                )
                model.save_pretrained(result_path)
    tokenizer.save_pretrained(result_path)

    similarity = None
    if eval:
        similarity = evaluate(tokenizer, model)
    torch.cuda.empty_cache()
    return similarity

if __name__ == "__main__":
    results = {}
    for torch_dtype in [torch.float32]:
        for strip_mode in [StripMode.TO_OV]:
            similarity = main(strip_mode, eval=True, use_cuda=False, torch_dtype=torch_dtype)
            results[strip_mode] = similarity

        print("FINAL:")
        for strip_m, sim in results.items():
            print(f" Strip: {strip_m.value}")
            print(f" Samples: {NUM_EVAL_SAMPLES}")
            print(f" Similarity: {sim}")
        print(f"MODE: {MODE.value}")
        print(f"BACKUP MODE: {BACKUP_MODE.value}")
        print(f"TORCH DTYPE: {torch_dtype}")