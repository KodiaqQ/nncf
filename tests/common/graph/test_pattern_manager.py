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
import pytest
from nncf.common.utils.backend import BackendType
from nncf.common.graph.patterns import PatternNames
from nncf.common.graph.patterns.manager import PatternsManager

IGNORED_PATTERN_REASONS = {
    BackendType.OPENVINO: {
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
    },
    BackendType.ONNX: {
        PatternNames.ADD_SCALE_SHIFT_OUTPUT: 'Not relevant for ONNX.',
        PatternNames.BATCH_INDEX: 'Not relevant for ONNX.',
        PatternNames.MVN_SCALE_SHIFT: 'Not relevant for ONNX.',
        PatternNames.NORMALIZE_L2_MULTIPLY: 'Not relevant for ONNX.',
        PatternNames.LINEAR_WITH_BIAS: 'Linear layers contains biases in ONNX.',
        PatternNames.SE_BLOCK: 'Not relevant for ONNX.',
        PatternNames.STABLE_DIFFUSION: 'Not relevant for ONNX.',
        PatternNames.SOFTMAX_DIV: 'Not relevant for ONNX.',
        PatternNames.SOFTMAX_RESHAPE_MATMUL: 'Not relevant for ONNX.',
        PatternNames.SOFTMAX_RESHAPE_TRANSPOSE_MATMUL: 'Not relevant for ONNX.',
        PatternNames.SOFTMAX_RESHAPE_TRANSPOSE_GATHER_MATMUL: 'Not relevant for ONNX.',
        PatternNames.EQUAL_LOGICALNOT: 'Not relevant for ONNX.',
        PatternNames.FC_BN_HSWISH_ACTIVATION: 'Not relevant for ONNX.',
        PatternNames.HSWISH_ACTIVATION: 'Not relevant for ONNX.',
        PatternNames.HSWISH_ACTIVATION_V2: 'Not relevant for ONNX.',
        PatternNames.HSWISH_ACTIVATION_WITHOUT_DENOMINATOR: 'Not relevant for ONNX.',
        PatternNames.SOFTMAX: 'Not relevant for ONNX.',
        PatternNames.INPUT_CONVERT_TRANSPOSE_PROCESSING: 'Not relevant for ONNX.',
        PatternNames.INPUT_CONVERT_TRANSPOSE_REVERSE_ADD: 'Not relevant for ONNX.',
        PatternNames.INPUT_CONVERT_TRANSPOSE_REVERSE_SCALE_SHIFT: 'Not relevant for ONNX.',
        PatternNames.INPUT_CONVERT_TRANSPOSE_SCALE_SHIFT: 'Not relevant for ONNX.',
        PatternNames.INPUT_REVERSE_ADD: 'Not relevant for ONNX.',
        PatternNames.INPUT_REVERSE_SCALE_SHIFT: 'Not relevant for ONNX.',
        PatternNames.INPUT_TRANSPOSE_PROCESSING: 'Not relevant for ONNX.',
        PatternNames.INPUT_TRANSPOSE_REVERSE_ADD: 'Not relevant for ONNX.',
        PatternNames.INPUT_TRANSPOSE_SCALE_SHIFT: 'Not relevant for ONNX.',
        PatternNames.LINEAR_ARITHMETIC_ACTIVATIONS: 'Not relevant for ONNX.',
        PatternNames.HSWISH_ACTIVATION_CLAMP_MULTIPLY: 'Not relevant for ONNX.',
        PatternNames.LINEAR_BIASED_SCALE_SHIFT: 'Not relevant for ONNX.',
        PatternNames.LINEAR_ACTIVATION_SCALE_SHIFT: 'Not relevant for ONNX.',
        PatternNames.LINEAR_BIASED_ACTIVATION_SCALE_SHIFT: 'Not relevant for ONNX.',
        PatternNames.LINEAR_ELEMENTWISE: 'Not relevant for ONNX.',
        PatternNames.LINEAR_BIASED_ELEMENTWISE: 'Not relevant for ONNX.',
        PatternNames.LINEAR_ACTIVATION_ELEMENTWISE: 'Not relevant for ONNX.',
        PatternNames.LINEAR_BIASED_ACTIVATION_ELEMENTWISE: 'Not relevant for ONNX.',
    }
}


@pytest.mark.parametrize('backend', [BackendType.OPENVINO, BackendType.ONNX])
def test_pattern_manager(backend):
    manager = PatternsManager()
    backend_patterns = manager.get_backend_patterns_map(backend)
    ignored_reasons = IGNORED_PATTERN_REASONS[backend]

    all_base_apatterns = PatternNames
    for base_pattern in all_base_apatterns:
        if base_pattern in ignored_reasons:
            ignore_reason = ignored_reasons[base_pattern]
            print(f'{base_pattern.value.name} is ignored. Reason: {ignore_reason}')
            continue
        assert base_pattern in backend_patterns