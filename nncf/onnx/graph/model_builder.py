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


import nncf
from nncf.common.graph import NNCFNode
import onnx
from nncf.onnx.graph.onnx_helper import get_name_to_node_map
from nncf.onnx.graph.onnx_helper import get_edge_shape
from nncf.onnx.graph.onnx_helper import get_edge_info_mapping
from nncf.onnx.graph.onnx_helper import get_parents_node_mapping
from nncf.onnx.graph.onnx_helper import get_parent
from nncf.onnx.graph.onnx_helper import get_tensor
from nncf.onnx.graph.onnx_helper import has_tensor
from nncf.onnx.graph.metatypes import onnx_metatypes as om


@staticmethod
def build_for_fast_bc(model: onnx.ModelProto, node: NNCFNode, act_port_id: int, weight_port_id: int, out_port_id: int = 0) -> onnx.ModelProto:
    """
    Builds submodel for the FastBiasCorrection algorithm.

    The submodel consists of the biased layer (but without bias), weight quantizer-dequantizer and weights:
                 Constant
                    |
                 Quantizer
                    |
    Input  Dequantizer
        \          /
        Convolution
            |
          Output
    
    :param model: onnx.ModelProto instance as the reference.
    :param node: NNCFNode with the layer-related information.
    :param act_port_id: Activation port ID.
    :param weight_port_id: Weight port ID.
    :param out_port_id: Output port ID.
    :return: onnx.ModelProto subgraph.
    """
    # Create nodes mapping
    name_to_node_mapping = get_name_to_node_map(model)
    name_to_edge_mapping = get_edge_info_mapping(model)
    node_to_parents_mapping = get_parents_node_mapping(model)

    node_name = node.node_name
    original_node = name_to_node_mapping[node_name]
    act_edge = name_to_edge_mapping[original_node.input[act_port_id]]
    out_edge = name_to_edge_mapping[original_node.output[out_port_id]]

    original_weight_dequantizer = get_parent(original_node, weight_port_id, node_to_parents_mapping)
    original_weight_quantizer = get_parent(original_weight_dequantizer, 0, node_to_parents_mapping)

    # Build subgraph
    initializers_list = set()
    input_name = f"Input_{node_name}"
    input_node = onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, get_edge_shape(act_edge))

    output_name = f"Output_{node_name}"
    output_node = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, get_edge_shape(out_edge))

    weight_quantizer = onnx.helper.make_node(
        name=original_weight_quantizer.name,
        op_type="QuantizeLinear",
        inputs=original_weight_quantizer.input,
        outputs=original_weight_quantizer.output,
        **{attr.name: attr.i for attr in original_weight_quantizer.attribute}
    )
    initializers_list.update([t for t in original_weight_quantizer.input if has_tensor(model, t)])

    weight_dequantizer = onnx.helper.make_node(
        name=original_weight_dequantizer.name,
        op_type="DequantizeLinear",
        inputs=original_weight_dequantizer.input,
        outputs=original_weight_dequantizer.output,
        **{attr.name: attr.i for attr in original_weight_dequantizer.attribute}
    )
    initializers_list.update([t for t in original_weight_dequantizer.input if has_tensor(model, t)])

    if node.metatype == om.ONNXConvolutionMetatype:
        main_node_inputs = list(original_weight_dequantizer.output)
        node_params = {attr.name: attr for attr in original_node.attribute}
        main_node_params = {
            "dilations": node_params["dilations"].ints,
            "kernel_shape": node_params["kernel_shape"].ints,
            "pads": node_params["pads"].ints,
            "strides": node_params["strides"].ints,
            "group": node_params["group"].i,
        }
        main_node = onnx.helper.make_node(
            name=original_node.name,
            op_type="Conv",
            inputs=[input_name] + main_node_inputs,
            outputs=[output_name],
            **main_node_params
        )
    else:
        raise nncf.ModuleNotFoundError(f"Not found node type: {node.metatype.name}!")

    graph_def = onnx.helper.make_graph(
        nodes=[weight_quantizer, weight_dequantizer, main_node],
        name=f"fbc_{node_name}_model",
        inputs=[input_node],
        outputs=[output_node],
        initializer=[get_tensor(model, t) for t in initializers_list],
    )

    builded_model = onnx.helper.make_model(graph_def)
    onnx.checker.check_model(builded_model)
    return builded_model
