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

from typing import Type

import numpy as np
import pytest
import torch

from nncf.common.tensor import NNCFTensor
from nncf.common.tensor_statistics.statistics import MeanTensorStatistic
from nncf.common.tensor_statistics.statistics import MedianMADTensorStatistic
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.tensor_statistics.statistics import PercentileTensorStatistic
from nncf.common.tensor_statistics.statistics import RawTensorStatistic
from nncf.torch.tensor import PTNNCFTensor
from nncf.torch.tensor_statistics.statistics import PTMeanTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTMedianMADTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTMinMaxTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTPercentileTensorStatistic
from tests.common.experimental.test_statistic_collector import TemplateTestStatisticCollector


class TestPTStatisticCollector(TemplateTestStatisticCollector):
    def get_nncf_tensor(self, value: np.ndarray) -> NNCFTensor:
        return PTNNCFTensor(torch.tensor(value))

    @pytest.fixture
    def min_max_statistic_cls(self) -> Type[MinMaxTensorStatistic]:
        return PTMinMaxTensorStatistic

    @pytest.fixture
    def mean_statistic_cls(self) -> Type[MeanTensorStatistic]:
        return PTMeanTensorStatistic

    @pytest.fixture
    def median_mad_statistic_cls(self) -> Type[MedianMADTensorStatistic]:
        return PTMedianMADTensorStatistic

    @pytest.fixture
    def percentile_statistic_cls(self) -> Type[PercentileTensorStatistic]:
        return PTPercentileTensorStatistic

    @pytest.fixture
    def raw_statistic_cls(self) -> Type[RawTensorStatistic]:
        raise NotImplementedError()

    @pytest.mark.skip
    def test_raw_max_stat_building(self, raw_statistic_cls: RawTensorStatistic):
        pass
