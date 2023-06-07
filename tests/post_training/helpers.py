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

from ast import Tuple
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from tests.torch.helpers import create_bn
from tests.torch.helpers import create_conv
from tests.torch.helpers import set_torch_seed


class StaticDatasetMock(Dataset):
    def __init__(self, input_size: Tuple, fn_to_type: Callable = None):
        super().__init__()
        self._len = 1
        self._input_size = input_size
        self._fn_to_type = fn_to_type

    def __getitem__(self, index):
        np.random.seed(0)
        data = np.random.rand(*tuple(self._input_size)).astype(np.float32)
        if self._fn_to_type:
            data = self._fn_to_type(data)
        return data, 0

    def __len__(self) -> int:
        return self._len


class ConvTestModel(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 2, 2, -1, -2)
        self.conv.weight.data = torch.Tensor([[[[0.1, -2.0], [1.0, 0.1]]], [[[0.1, 2.0], [-1.0, 0.1]]]])
        self.conv.bias.data = torch.Tensor([0.1, 1.0])

    def forward(self, x):
        return self.conv(x)


class ConvBNTestModel(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 2, 2, bias=False)
        self.conv.weight.data = torch.Tensor([[[[0.1, -2.0], [1.0, 0.1]]], [[[0.1, 2.0], [-1.0, 0.1]]]])
        self.bn = create_bn(2)
        self.bn.bias.data = torch.Tensor([0.1, 1.0])
        self.bn.weight.data = torch.Tensor([0.2, 2.0])

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class FCTestModel(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.fc.weight.data = torch.Tensor([[0.1, 0.2, 0.3, 0.2], [0.3, -0.1, 0.2, 0.4]])
        self.fc.bias.data = torch.Tensor([1.0, 1.1])

    def forward(self, x):
        x = self.fc(x)
        return x


class MultipleConvTestModel(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self):
        super().__init__()
        with set_torch_seed():
            self.conv_1 = self._build_conv(1, 2, 2)
            self.conv_2 = self._build_conv(2, 3, 2)
            self.conv_3 = self._build_conv(1, 2, 3)
            self.conv_4 = self._build_conv(2, 3, 1)
            self.conv_5 = self._build_conv(3, 2, 2)

    def _build_conv(self, in_channels=1, out_channels=2, kernel_size=2):
        conv = create_conv(in_channels, out_channels, kernel_size)
        conv.weight.data = torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        conv.bias.data = torch.randn([out_channels])
        return conv

    def forward(self, x):
        x_1 = self.conv_1(x)
        x_1 = self.conv_2(F.relu(x_1))
        x_2 = self.conv_3(x)
        x_2 = self.conv_4(F.relu(x_2))
        x_1_2 = torch.concat([x_1, x_2])
        return self.conv_5(F.relu(x_1_2))
