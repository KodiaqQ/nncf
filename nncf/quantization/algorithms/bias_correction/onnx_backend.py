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

from typing import Dict, Tuple, List, Optional

import onnx
import numpy as np

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.backend import BackendType
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFGraph
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionTransposeMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDepthwiseConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXMatMulMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDequantizeLinearMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXQuantizeLinearMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXOpMetatype
from nncf.onnx.graph.model_transformer import ONNXModelTransformer
from nncf.onnx.graph.transformations.commands import ONNXBiasCorrectionCommand
from nncf.onnx.graph.transformations.commands import ONNXModelExtractionCommand
from nncf.onnx.graph.transformations.commands import ONNXQDQNodeRemovingCommand
from nncf.onnx.graph.transformations.commands import ONNXOutputInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.onnx.statistics.collectors import ONNXMeanStatisticCollector
from nncf.onnx.statistics.collectors import ONNXBatchStatisticCollector
from nncf.onnx.statistics.collectors import ONNXNNCFCollectorTensorProcessor
from nncf.onnx.tensor import ONNXNNCFTensor
from nncf.quantization.algorithms.bias_correction.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.bias_correction.backend import BiasCorrectionAlgoBackend
from nncf.onnx.graph.onnx_graph import ONNXGraph
from nncf.onnx.graph.node_utils import get_bias_value
from nncf.onnx.graph.node_utils import is_node_with_bias
from nncf.onnx.graph.transformations.command_creation import create_bias_correction_command


#pylint:disable=too-many-public-methods
@ALGO_BACKENDS.register(BackendType.ONNX)
class ONNXBiasCorrectionAlgoBackend(BiasCorrectionAlgoBackend):

    @property
    def channel_axis_by_types(self) -> Dict[ONNXOpMetatype, int]:
        return {
            ONNXConvolutionMetatype: 1,
            ONNXMatMulMetatype: -1,
            ONNXConvolutionTransposeMetatype: 1,
            ONNXDepthwiseConvolutionMetatype: 1
        }

    @property
    def tensor_processor(self) -> ONNXNNCFCollectorTensorProcessor:
        return ONNXNNCFCollectorTensorProcessor()

    @property
    def quantizer_types(self) -> List[OperatorMetatype]:
        return [ONNXQuantizeLinearMetatype, ONNXDequantizeLinearMetatype]

    @staticmethod
    def target_point(target_type: TargetType,
                     target_node_name: str,
                     port_id: int) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def create_bias_correction_command(node: NNCFNode,
                                       bias_value: np.ndarray,
                                       nncf_graph: NNCFGraph) -> ONNXBiasCorrectionCommand:
        return create_bias_correction_command(node, bias_value)

    @staticmethod
    def output_insertion_command(target_point: ONNXTargetPoint) -> ONNXOutputInsertionCommand:
        return ONNXOutputInsertionCommand(target_point)

    @staticmethod
    def node_removing_command(target_point: ONNXTargetPoint) -> ONNXQDQNodeRemovingCommand:
        return ONNXQDQNodeRemovingCommand(target_point)

    @staticmethod
    def mean_statistic_collector(reduction_shape: ReductionShape,
                                 num_samples: Optional[int] = None,
                                 window_size: Optional[int] = None) -> ONNXMeanStatisticCollector:
        return ONNXMeanStatisticCollector(reduction_shape,  num_samples, window_size)

    @staticmethod
    def batch_statistic_collector(num_samples: int = None) -> ONNXMeanStatisticCollector:
        return ONNXBatchStatisticCollector(num_samples)

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: str) -> ONNXNNCFTensor:
        return ONNXNNCFTensor(raw_data[output_name])

    @staticmethod
    def get_activation_port_ids_for_bias_node(node: NNCFNode) -> Tuple[int, int]:
        return 0, 0

    @staticmethod
    def get_bias_value(node: NNCFNode, model: onnx.ModelProto, nncf_graph: NNCFGraph) -> np.ndarray:
        return get_bias_value(node, model)

    @staticmethod
    def get_bias_port_id(node: NNCFNode, model: onnx.ModelProto) -> int:
        onnx_graph = ONNXGraph(model)
        onnx_node = onnx_graph.get_node_by_name(node.node_name)
        return onnx_graph.get_bias_tensor_port_id(onnx_node)

    @staticmethod
    def get_input_name(model: onnx.ModelProto, node_name: str) -> str:
        onnx_graph = ONNXGraph(model)
        node = onnx_graph.get_node_by_name(node_name)
        return node.input[0]

    @staticmethod
    def get_output_name(model: onnx.ModelProto, node_name: str) -> List[str]:
        onnx_graph = ONNXGraph(model)
        node = onnx_graph.get_node_by_name(node_name)
        return node.output[0]

    @staticmethod
    def extract_model(model: onnx.ModelProto,
                      input_node_names: List[str],
                      output_node_names: List[str]) -> onnx.ModelProto:
        onnx_graph = ONNXGraph(model)

        input_tensor_names = []
        for input_node_name in input_node_names:
            input_onnx_node = onnx_graph.get_node_by_name(input_node_name)
            input_tensor_names.append(input_onnx_node.input[0])

        output_tensor_names = []
        for output_node_name in output_node_names:
            output_onnx_node = onnx_graph.get_node_by_name(output_node_name)
            output_tensor_names.append(output_onnx_node.output[0])

        if not output_node_names:
            output_tensor_names = [n.name for n in onnx_graph.get_model_outputs()]

        transformation_layout = TransformationLayout()
        model_transformer = ONNXModelTransformer(model)
        transformation_layout.register(ONNXModelExtractionCommand(set(input_tensor_names), set(output_tensor_names)))
        return model_transformer.transform(transformation_layout)

    @staticmethod
    def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        input_nodes = nncf_graph.get_previous_nodes(node)
        weight_port_id = node.metatype.weight_definitions.weight_port_id
        weight_node = input_nodes[weight_port_id]
        return weight_node.metatype == ONNXDequantizeLinearMetatype

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_node_with_bias(node)
