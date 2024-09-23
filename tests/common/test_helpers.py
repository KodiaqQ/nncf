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

import pytest
from torch.random import manual_seed
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.data import generate_text_data
from tests.shared.helpers import load_json
from tests.shared.paths import TEST_ROOT

manual_seed(0)

BASE_TEST_MODEL_ID = "hf-internal-testing/tiny-random-gpt2"
GENERATED_TEXT_REF = TEST_ROOT / "common" / "data" / "ref_generated_data.json"


@pytest.mark.parametrize(
    "model, tokenizer, usage_error",
    [
        [None, None, True],
        [AutoModelForCausalLM.from_pretrained(BASE_TEST_MODEL_ID), None, True],
        [None, AutoTokenizer.from_pretrained(BASE_TEST_MODEL_ID), True],
        [
            AutoModelForCausalLM.from_pretrained(BASE_TEST_MODEL_ID),
            AutoTokenizer.from_pretrained(BASE_TEST_MODEL_ID),
            False,
        ],
    ],
)
def test_generate_text_data_usage(model, tokenizer, usage_error):
    try:
        generate_text_data(model, tokenizer, seq_len=2, dataset_size=1)
    except Exception as e:
        if usage_error:
            assert isinstance(e, nncf.ValidationError), "Expected exception."


def test_generate_text_data_functional():
    seq_len = 12
    max_seq_len = seq_len + seq_len // 2
    dataset_size = 6

    model = AutoModelForCausalLM.from_pretrained(BASE_TEST_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(BASE_TEST_MODEL_ID)

    generated_data = generate_text_data(
        model,
        tokenizer,
        seq_len=seq_len,
        dataset_size=dataset_size,
    )

    assert len(generated_data) == dataset_size
    generated_data = [tokenizer.encode(d) for d in generated_data]

    # Uncomment lines below to generate reference for new models.
    # from tests.shared.helpers import dump_to_json
    # dump_to_json(GENERATED_TEXT_REF, generated_data)

    reference_data = load_json(GENERATED_TEXT_REF)
    for ref_data, gen_data in zip(reference_data, generated_data):
        assert len(gen_data) <= max_seq_len
        assert ref_data == gen_data
