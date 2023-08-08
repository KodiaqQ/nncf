# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import Callable, Dict, TypeVar

import pytest

from nncf.common.factory import NNCFGraphFactory
from nncf.experimental.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.parameters import ModelType
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.smooth_quant.algorithm import SmoothQuant
from nncf.quantization.algorithms.smooth_quant.backend import SmoothQuantAlgoBackend
from tests.post_training.test_templates.helpers import LinearMultiShapeModel
from tests.post_training.test_templates.helpers import get_static_dataset

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


class TemplateTestSQAlgorithm:
    @staticmethod
    def fn_to_type(tensor) -> TTensor:
        return tensor

    @staticmethod
    @abstractmethod
    def get_transform_fn() -> Callable:
        """
        Get transformation function for dataset.
        """

    @staticmethod
    @abstractmethod
    def backend_specific_model(model: TModel, tmp_dir: str) -> TModel:
        """
        Return backend specific model.
        """

    @staticmethod
    @abstractmethod
    def check_scales(model: TModel, reference_values: Dict[str, TTensor]) -> None:
        """
        Checking scales from model with references.
        """

    @staticmethod
    @abstractmethod
    def get_backend() -> SmoothQuantAlgoBackend:
        """
        Returns backend-specific SmoothQuantAlgoBackend.
        """

    @staticmethod
    def get_quantization_algorithm():
        return PostTrainingQuantization(
            subset_size=1,
            model_type=ModelType.TRANSFORMER,
            advanced_parameters=AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.DISABLE, smooth_quant_alpha=0.95, inplace_statistics=False
            ),
        )

    @pytest.mark.parametrize(
        "model_cls, reference_values",
        (
            (
                LinearMultiShapeModel,
                {
                    "/Reshape_0_0/sq_multiply": [[[[1.0594617, 1.1019668, 1.2208323, 1.1003988]]]],
                    "/Split_1_0/sq_multiply": [[[[1.1276343, 0.7605822]]]],
                    "/Split_0_0/sq_multiply": [[[[0.32575992, 0.33121374]]]],
                    "/Reshape_1_0_0/sq_multiply": [
                        [
                            [
                                0.3251956,
                                0.3326432,
                                1.5490624,
                                0.7233769,
                                0.3689916,
                                0.4845651,
                                1.2022541,
                                1.3118246,
                            ]
                        ]
                    ],
                    "/Reshape_1_0_1/sq_multiply": [[[0.4699388], [0.3369332], [0.3674589]]],
                    "/Reshape_2_0_0/sq_multiply": [[0.1242606]],
                    "/ReduceMax_0_0/sq_multiply": [
                        [0.0944255, 0.0853033, 0.7187095, 0.3429819, 0.1422914, 0.2127623, 0.4640060, 0.7210725]
                    ],
                },
            ),
        ),
    )
    def test_smooth_quant_algo(self, model_cls, reference_values, tmpdir):
        model = self.backend_specific_model(model_cls(), tmpdir)
        dataset = get_static_dataset(model_cls.INPUT_SIZE, self.get_transform_fn(), self.fn_to_type)

        quantization_algorithm = self.get_quantization_algorithm()
        graph = NNCFGraphFactory.create(model)
        quantized_model = quantization_algorithm.apply(model, graph, dataset=dataset)

        self.check_scales(quantized_model, reference_values)

    # pylint:disable=protected-access
    def test_get_abs_max_channel_collector(self):
        backend = self.get_backend()
        reduction_shape = (3, 2, 1)
        samples = 1

        for inplace_type in [False, True]:
            backend_tensor_collector = backend.get_abs_max_channel_collector(
                num_samples=samples,
                stats_reduction_shape=reduction_shape,
                inplace=inplace_type,
                branch_key="test_branch",
            )

            for aggregator in backend_tensor_collector.aggregators.values():
                assert isinstance(aggregator, MaxAggregator)

            for reducer in backend_tensor_collector.reducers:
                assert isinstance(reducer, AbsMaxReducer)
                assert reducer.inplace == inplace_type
                assert reducer._reduction_shape == reduction_shape

    @pytest.mark.parametrize(
        "model_cls, references",
        (
            (
                LinearMultiShapeModel,
                [
                    ("/MatMul_1", 0),
                    ("/MatMul", 0),
                    ("/linear_2/MatMul", 0),
                    ("/linear_1/MatMul", 0),
                    ("/MatMul_2", 0),
                    ("/MatMul_4", 1),
                    ("55", 1),
                    ("41", 0),
                    ("19", 1),
                    ("24", 0),
                ],
            ),
        ),
    )
    # pylint:disable=protected-access
    def test__get_nodes_to_smooth_data(self, model_cls, references, tmpdir):
        model = self.backend_specific_model(model_cls(), tmpdir)
        nncf_graph = NNCFGraphFactory.create(model)

        algo = SmoothQuant()
        algo._set_backend_entity(model)
        smooth_data = algo._get_nodes_to_smooth_data(nncf_graph)
        smooth_data = {d["node_to_smooth"].node_name: d["input_act_port"] for d in smooth_data}

        for ref_node_name, ref_port_id in references:
            assert ref_node_name in smooth_data
            assert smooth_data[ref_node_name] == ref_port_id
