"""
 Copyright (c) 2022 Intel Corporation
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
from typing import List

import numpy as np
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import TensorType
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.registry import Registry

ALGO_BACKENDS = Registry('algo_backends')


class FBCAlgoBackend(ABC):

    @property
    @abstractmethod
    def operation_metatypes(self):
        """ Property for the backend-specific metatypes """

    @property
    @abstractmethod
    def layers_with_bias_metatypes(self):
        """ Property for the backend-specific metatypes with bias """

    @staticmethod
    @abstractmethod
    def target_point(target_type: TargetType, target_node_name: str, edge_name: str = None) -> TargetPoint:
        """ Returns backend-specific target point """

    @staticmethod
    @abstractmethod
    def bias_correction_command(target_point: TargetPoint,
                                bias_value: np.ndarray,
                                threshold: float) -> TransformationCommand:
        """ Returns backend-specific bias correction command """

    @staticmethod
    @abstractmethod
    def model_extraction_command(inputs: List[str], outputs: List[str]) -> TransformationCommand:
        """ Returns backend-specific bias correction command """

    @staticmethod
    @abstractmethod
    def mean_statistic_collector(reduction_shape: ReductionShape,
                                 num_samples: int = None,
                                 window_size: int = None) -> TensorStatisticCollectorBase:
        """ Returns backend-specific mean statistic collector """

    @staticmethod
    @abstractmethod
    def nncf_tensor(tensor: TensorType) -> NNCFTensor:
        """ Returns backend-specific NNCFTensor """