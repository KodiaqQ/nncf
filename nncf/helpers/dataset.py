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

import torch
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from transformers.utils import logging

from nncf.common.logging.track_progress import track
from nncf.data.dataset import Dataset

logging.set_verbosity_error()

BASE_VOCAB_SIZE = 12000


def get_auto_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    transform_func: callable,
    seq_len: int = 32,
    dataset_size: int = 128,
) -> Dataset:
    """
    Generates dataset based on the model output.

    :param model: PreTrainedModel instance.
    :param tokenizer: PreTrainedTokenizer instance.
    :param transform_func: Callable transformation function.
    :param seq_len: Sequence length for generation.
    :param dataset_size: Size of the nncf.Dataset.
    :return: nncf.Dataset instance ready to use.
    """

    collected_data = []

    vocab_size_names = ["padded_vocab_size", "vocab_size"]
    vocab_size = BASE_VOCAB_SIZE
    for vocab_size_name in vocab_size_names:
        if hasattr(model.config, vocab_size_name):
            vocab_size = getattr(model.config, vocab_size_name)

    step_num = max(1, vocab_size // dataset_size)
    ids_counter = 0

    for _ in track(range(dataset_size), description="Collecting auto dataset"):
        # Creating the input for pre-generate step
        input_ids = torch.tensor([[ids_counter % vocab_size]])
        ids_counter += step_num

        # Collecting data from the pre & post generate steps
        outputs_prep = model.generate(input_ids, do_sample=False, max_length=seq_len // 2)
        outputs_post = model.generate(outputs_prep, do_sample=True, max_length=seq_len + seq_len // 2)
        gen_text = tokenizer.batch_decode(outputs_post[:, outputs_prep.shape[1] :], skip_special_tokens=True)

        collected_data.extend(gen_text)

    calibration_dataset = Dataset(collected_data, transform_func)
    return calibration_dataset
