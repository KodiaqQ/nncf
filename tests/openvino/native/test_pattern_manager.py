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
from nncf.common.utils.backend import BackendType
from nncf.common.graph.patterns import PatternNames
from nncf.common.graph.patterns.manager import PatternsManager

IGNORED_PATTERN_REASONS = {
     PatternNames.SWISH_WITH_SIGMOID: 'Swish exists in the OpenVINO as layer.',
     PatternNames.SWISH_WITH_HARD_SIGMOID: 'Swish exists in the OpenVINO as layer.',
     PatternNames.INPUT_MULTIPLY_SUBTRACT: 'Not relevant for OpenVINO.',
     PatternNames.ACTIVATIONS_SCALE_SHIFT: 'Not relevant for OpenVINO.',
     PatternNames.ARITHMETIC_ACTIVATIONS_BATCH_NORM: 'Not relevant for OpenVINO.',
     PatternNames.ARITHMETIC_ACTIVATIONS_SCALE_SHIFT: 'Not relevant for OpenVINO.',
     PatternNames.ARITHMETIC_BATCH_NORM: 'Not relevant for OpenVINO.',
     PatternNames.ARITHMETIC_BATCH_NORM_ACTIVATIONS: 'Not relevant for OpenVINO.',
     PatternNames.ARITHMETIC_SCALE_SHIFT: 'Not relevant for OpenVINO.',
     PatternNames.ARITHMETIC_SCALE_SHIFT_ACTIVATIONS: 'Not relevant for OpenVINO.',
     PatternNames.BATCH_NORM_SCALE_SHIFT_ACTIVATIONS: 'Not relevant for OpenVINO.',
     PatternNames.LINEAR_ACTIVATIONS_BATCH_NORM: 'Not relevant for OpenVINO.',
     PatternNames.LINEAR_ACTIVATIONS_SCALE_SHIFT: 'Not relevant for OpenVINO.',
     PatternNames.LINEAR_BATCH_NORM: 'Not relevant for OpenVINO.',
     PatternNames.LINEAR_BATCH_NORM_ACTIVATIONS: 'Not relevant for OpenVINO.',
     PatternNames.LINEAR_BATCH_NORM_SCALE_SHIFT_ACTIVATIONS: 'Not relevant for OpenVINO.',
     PatternNames.LINEAR_SCALE_SHIFT_ACTIVATIONS: 'Not relevant for OpenVINO.',
     PatternNames.SCALE_SHIFT_ACTIVATIONS: 'Not relevant for OpenVINO.',
}

def test_pattern_manager():
     manager = PatternsManager()
     backend_patterns = manager.get_backend_patterns_map(BackendType.OPENVINO)
     
     all_base_apatterns = PatternNames
     for base_pattern in all_base_apatterns:
          if base_pattern in IGNORED_PATTERN_REASONS:
               ignore_reason = IGNORED_PATTERN_REASONS[base_pattern]
               print(f'{base_pattern.value.name} is ignored. Reason: {ignore_reason}')
               continue
          assert base_pattern in backend_patterns
