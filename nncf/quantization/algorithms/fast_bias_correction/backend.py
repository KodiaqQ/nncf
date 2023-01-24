"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from abc import ABC
from abc import abstractmethod
from typing import List, Optional, Tuple, TypeVar

import numpy as np

from nncf.common.graph import NNCFGraph, NNCFNode
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.utils.registry import Registry

TModel = TypeVar('TModel')
OutputType = TypeVar('OutputType')
ALGO_BACKENDS = Registry('algo_backends')


class FastBiasCorrectionAlgoBackend(ABC):

    @property
    @abstractmethod
    def operation_metatypes(self):
        """
        Property for the backend-specific metatypes.
        """

    @property
    @abstractmethod
    def channel_axis_by_types(self):
        """
        Property for the backend-specific info about channels placement in the layout.
        """

    @property
    @abstractmethod
    def tensor_processor(self):
        """
        Returns backend-specific instance of the NNCFCollectorTensorProcessor.
        """

    @staticmethod
    @abstractmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> TargetPoint:
        """
        Returns backend-specific target point.

        :param target_type: Type of the location that should be modified.
        :param target_node_name: Name of the located node.
        :param port_id: Port ID of the tensor for the statistics distribution.
        :return: Backend-specific TargetPoint.
        """

    @staticmethod
    @abstractmethod
    def create_bias_correction_command(bias_node: NNCFNode, bias_value: np.ndarray) -> TransformationCommand:
        """
        Creates backend-specific command to update bias value.

        :param bias_node: The node for which bias should be updated.
        :param bias_value: New value for the bias.
        :return: Backend-specific command to update bias value.
        """

    @staticmethod
    @abstractmethod
    def model_extraction_command(inputs: List[str], outputs: List[str]) -> TransformationCommand:
        """
        Returns backend-specific bias correction.

        :param inputs: List of the input names for sub-model beggining.
        :param outputs: List of the output names for sub-model end.
        :return: Backend-specific TransformationCommand for the model extraction.
        """

    @staticmethod
    @abstractmethod
    def mean_statistic_collector(reduction_shape: ReductionShape,
                                 num_samples: Optional[int] = None,
                                 window_size: Optional[int] = None) -> TensorStatisticCollectorBase:
        """
        Returns backend-specific mean statistic collector.

        :param reduction_shape: Channel axes for the statistics aggregation.
        :param num_samples: Maximum number of samples to collect.
        :param window_size: The maximum size of the samples queue.
        :return: Backend-specific TensorStatisticCollectorBase for the statistics calculation.
        """

    @staticmethod
    @abstractmethod
    def get_input_output_names(biased_node: NNCFNode) -> Tuple[List[str], List[str]]:
        """
        Returns tuple of the input & output names respectively.

        :param biased_node: NNCFNode affected by bias.
        :return: Tuple of the input & output names.
        """

    @staticmethod
    @abstractmethod
    def create_blob(shape: Tuple[int], data: List[float]) -> np.ndarray:
        """
        Creates the backend-specific (because of layout) blob.

        :param shape: Shape of the blob.
        :param data: Data to fill the blob.
        :return: np.ndarray blob.
        """

    @staticmethod
    @abstractmethod
    def get_bias_value(bias_node: NNCFNode, model: TModel) -> np.ndarray:
        """
        Returns bias value in the NumPy format of provided node.

        :param bias_node: Node of NNCFGraph with bias value.
        :param model: Backend-specific model for the initializer finding.
        :return: Bias value in the NumPy format.
        """

    @staticmethod
    @abstractmethod
    def get_activation_port_ids_for_bias_node(model: TModel, biased_node: NNCFNode) -> Tuple[int, int]:
        """
        Returns Input Port ID and Output Port ID corresponding to activation input and output edges for
        the node.
        Supports only nodes that could have bias value.

        :param model: Backend-specific model.
        :param biased_node: NNCFNode affected by bias.
        """

    @staticmethod
    @abstractmethod
    def is_quantized_weights(biased_node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        """
        Checks whether the node is quantized or not, based on a backend-specific way.

        :param biased_node: NNCFNode affected by bias.
        :param nncf_graph: NNCFGraph instance.
        :return: boolean indicating whether the node has a quantized weights or not.
        """

    @staticmethod
    @abstractmethod
    def process_model_output(raw_data: OutputType, output_name: str) -> NNCFTensor:
        """
        Returns backend-specific processed output from the model.

        :param raw_data: Backend-specific output from the model.
        :param output_name: Name of the output layer or tensor name.
        :return: Processed output as NNCFTensor.
        """

    @staticmethod
    @abstractmethod
    def is_node_with_bias(biased_node: NNCFNode) -> bool:
        """
        Checks whether the node has a bias or not.

        :param biased_node: NNCFNode affected by bias.
        :return: Boolean indicating whether the node has a bias or not.
        """

    @staticmethod
    @abstractmethod
    def get_bias_node(biased_node: NNCFNode, nncf_graph: NNCFGraph) -> Optional[NNCFNode]:
        """
        Returns node that contains bias, if it exists.

        :param biased_node: NNCFNode affected by bias.
        :param nncf_graph: NNCFGraph instance.
        :return: Optional NNCFNode with potential bias.
        """
