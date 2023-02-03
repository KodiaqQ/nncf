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
from typing import Dict
from nncf.common.utils.backend import BackendType
from nncf.parameters import TargetDevice
from nncf.common.graph.patterns.patterns import HWFusedPatterns
from nncf.common.graph.patterns.patterns import PatternNames


class PatternsManager:

    def get_backend_patterns_map(self, backend: BackendType) -> Dict[PatternNames, callable]:
        """
        Returns the backend-specific map from the Refgistry.

        :param backend: BackendType instance.
        :return: Dictionary with the PatternNames instance as keys and callable as value.
        """
        if backend == BackendType.ONNX:
            from nncf.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS
            return ONNX_HW_FUSED_PATTERNS.registry_dict
        if backend == BackendType.OPENVINO:
            from nncf.experimental.openvino_native.hardware.fused_patterns import OPENVINO_HW_FUSED_PATTERNS
            return OPENVINO_HW_FUSED_PATTERNS.registry_dict
        raise ValueError(f'Hardware-fused patterns not implemented for {backend} backend.')

    def get_patterns(self, backend: BackendType, device: TargetDevice) -> HWFusedPatterns:
        """
        Returns the backend- & device-specific HWFusedPatterns instance.

        :param backend: BackendType instance.
        :param device: TargetDevice instance.
        :return: Completed HWFusedPatterns value based on the backend & device.
        """
        backend_registry_map = self.get_backend_patterns_map(backend)
        hw_fused_patterns = HWFusedPatterns()

        for pattern_desc, pattern in backend_registry_map.items():
            pattern_desc_devices = pattern_desc.value.devices
            if pattern() is None:
                continue
            if pattern_desc_devices is None or device in pattern_desc_devices:
                hw_fused_patterns.register(pattern(), pattern_desc.value.name)
        return hw_fused_patterns
