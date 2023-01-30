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

from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.patterns import GraphPattern

from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXSigmoidMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXHardSigmoidMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXAddLayerMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXSubMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXMulLayerMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXBatchNormMetatype
from nncf.onnx.hardware.pattern_operations import LINEAR_OPERATIONS
from nncf.onnx.hardware.pattern_operations import ATOMIC_ACTIVATIONS_OPERATIONS


def create_activations() -> GraphPattern:
    swish = create_swish_activation()
    activations = GraphPattern()
    activations.add_node(**ATOMIC_ACTIVATIONS_OPERATIONS)
    activations.add_pattern_alternative(swish)
    return activations


def create_swish_activation() -> GraphPattern:
    pattern = GraphPattern()

    input_pattern_node_1 = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: '*INPUT_NODE*',
           GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})
    sigmoid_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SIGMOID',
                                         GraphPattern.METATYPE_ATTR: ONNXSigmoidMetatype})
    mul_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: ONNXMulLayerMetatype})

    pattern.add_edge(input_pattern_node_1, sigmoid_node_1)
    pattern.add_edge(input_pattern_node_1, mul_node_1)
    pattern.add_edge(sigmoid_node_1, mul_node_1)

    input_pattern_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: '*INPUT_NODE*',
                                               GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})
    sigmoid_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'HARDSIGMOID',
                                         GraphPattern.METATYPE_ATTR: ONNXHardSigmoidMetatype})
    mul_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: ONNXMulLayerMetatype})

    pattern.add_edge(input_pattern_node_2, sigmoid_node_2)
    pattern.add_edge(input_pattern_node_2, mul_node_2)
    pattern.add_edge(sigmoid_node_2, mul_node_2)

    return pattern


def create_input_preprocessing_pattern() -> GraphPattern:
    pattern = GraphPattern()

    model_input_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                             GraphPattern.METATYPE_ATTR: InputNoopMetatype})
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: ONNXAddLayerMetatype})
    mul_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: ONNXMulLayerMetatype})

    pattern.add_edge(model_input_node_1, add_node_1)
    pattern.add_edge(add_node_1, mul_node_1)

    model_input_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                             GraphPattern.METATYPE_ATTR: InputNoopMetatype})
    mul_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: ONNXMulLayerMetatype})
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: ONNXAddLayerMetatype})

    pattern.add_edge(model_input_node_2, mul_node_2)
    pattern.add_edge(mul_node_2, add_node_2)

    model_input_node_3 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                             GraphPattern.METATYPE_ATTR: InputNoopMetatype})
    add_node_3 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: ONNXAddLayerMetatype})

    pattern.add_edge(model_input_node_3, add_node_3)

    model_input_node_4 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                             GraphPattern.METATYPE_ATTR: InputNoopMetatype})
    mul_node_4 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: ONNXMulLayerMetatype})

    pattern.add_edge(model_input_node_4, mul_node_4)

    return pattern


def create_decomposed_batch_norm() -> GraphPattern:
    pattern = GraphPattern()

    model_input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: '*INPUT_NODE*',
                                           GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})
    mul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                   GraphPattern.METATYPE_ATTR: ONNXMulLayerMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: ONNXAddLayerMetatype})

    pattern.add_edge(model_input_node, mul_node)
    pattern.add_edge(mul_node, add_node)

    return pattern


def create_scale_shift() -> GraphPattern:
    pattern = GraphPattern()

    model_input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: '*INPUT_NODE*',
                                           GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})
    mul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                   GraphPattern.METATYPE_ATTR: ONNXMulLayerMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: [ONNXAddLayerMetatype, ONNXSubMetatype]})

    pattern.add_edge(model_input_node, mul_node)
    pattern.add_edge(mul_node, add_node)
    return pattern


def create_linear_bn_scale_shift_activation() -> GraphPattern:
    #          Linear
    #            |
    #        BatchNorm
    #            |
    #        Scale_Shift
    #            |
    #        Activation
    pattern = GraphPattern()
    linear_node = pattern.add_node(**LINEAR_OPERATIONS)
    bn_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'BATCH_NORM',
                                  GraphPattern.METATYPE_ATTR: ONNXBatchNormMetatype})
    scale_shift = create_scale_shift()
    activations = create_activations()
    pattern.add_edge(linear_node, bn_node)
    pattern.join_patterns(scale_shift)
    pattern.join_patterns(activations)
    return pattern


def create_bn_scale_shift_activation() -> GraphPattern:
    #        BatchNorm
    #            |
    #        Scale_Shift
    #            |
    #        Activation
    pattern = GraphPattern()
    _ = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'BATCH_NORM',
                            GraphPattern.METATYPE_ATTR: ONNXBatchNormMetatype})
    scale_shift = create_scale_shift()
    activations = create_activations()
    pattern.join_patterns(scale_shift)
    pattern.join_patterns(activations)
    return pattern
