
import torch

import nncf
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.model_graph_manager import get_const_node
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import split_const_name
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.layers import INT4AsymmetricWeightsDecompressor, INT8AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT4SymmetricWeightsDecompressor, INT8SymmetricWeightsDecompressor
from nncf.torch.quantization.quantize_functions import TuneRange

def strip_tuned_lora_model(model: NNCFNetwork) -> NNCFNetwork:
    layout = model.nncf.transformation_layout()
    model = model.nncf.get_clean_shallow_copy()
    graph = model.nncf.get_graph()
    transformation_layout = TransformationLayout()
    t = layout.transformations

    for command in t:
        quantizer_module = command.fn
        if isinstance(quantizer_module, AsymmetricQuantizer):
            input_range_safe = abs(quantizer_module.input_range) + quantizer_module.eps
            input_low, input_range = TuneRange.apply(
                quantizer_module.input_low, input_range_safe, quantizer_module.levels
            )
            # TODO: bfloat16 calculation in training. should take dtype somwhere.
            # input_low, input_range = quantizer_module.input_low, quantizer_module.input_range
            assert len(command.target_points) == 1
            tp = command.target_points[0]

            node_with_weight = graph.get_node_by_name(tp.target_node_name)

            weight_node = get_const_node(node_with_weight, tp.input_port_id, graph)
            weight_name = weight_node.layer_attributes.name
            module_name, weight_attr_name = split_const_name(weight_name)
            module = get_module_by_name(module_name, model)
            w = getattr(module, weight_attr_name)
            if w is None or not isinstance(w, torch.nn.Parameter):
                raise nncf.InternalError(f"Could not find a torch.nn.Parameter in the model by name {weight_name}.")

            input_low = input_low.to(w.dtype)
            input_range = input_range.to(w.dtype)

            input_ = w + quantizer_module._lora_B @ quantizer_module._lora_A
            input_ = input_.reshape(quantizer_module._qspec.weight_shape)
            input_ = input_.to(w.dtype)

            scale = input_range / (quantizer_module.levels - 1)
            scale = torch.where(torch.abs(scale) < quantizer_module.eps, quantizer_module.eps, scale)
            zero_point = quantizer_module.level_low - (input_low / scale).round()
            zero_point = zero_point.clip(min=quantizer_module.level_low, max=quantizer_module.level_high)

            output = input_ / scale
            output = output + zero_point
            output = output.round()
            output = output.clip(min=quantizer_module.level_low, max=quantizer_module.level_high)
            output = output.to(torch.uint8)

            if quantizer_module.num_bits == 8:
                decompressor = INT8AsymmetricWeightsDecompressor(
                    scale=scale,
                    zero_point=zero_point.to(torch.uint8),
                    result_dtype=w.dtype
                )
            else:
                decompressor = INT4AsymmetricWeightsDecompressor(
                    scale=scale,
                    zero_point=zero_point.to(torch.uint8),
                    compressed_weight_shape=output.shape,
                    result_shape=w.shape,
                    result_dtype=w.dtype,
                )
            packed_tensor = decompressor.pack_weight(output)

            # tmp = decompressor(packed_tensor)

            # sets compressed tensor
            compressed_parameter = torch.nn.Parameter(packed_tensor, requires_grad=False)
            setattr(module, weight_attr_name, compressed_parameter)

            consumer_nodes = graph.get_next_nodes(weight_node)
            if len(consumer_nodes) > 1:
                for c_node in consumer_nodes:
                    c_module = model.get_module_by_scope(Scope.from_str(c_node.layer_name))
                    for name, param in c_module.named_parameters(recurse=False, remove_duplicate=False):
                        if id(param) == id(w):
                            setattr(c_module, name, compressed_parameter)

            # registry weight decompression module in the model
            decompressor_name = f"weights_decompressor_{weight_node.node_name.replace('.', '_')}"

            # inserts the weight decompressor into the model as the post hook on the model weight
            transformation_layout.register(
                PTSharedFnInsertionCommand(
                    [PTTargetPoint(TargetType.OPERATOR_POST_HOOK, target_node_name=weight_node.node_name)],
                    decompressor,
                    decompressor_name,
                )
            )

        elif isinstance(quantizer_module, SymmetricQuantizer):
            assert len(command.target_points) == 1
            tp = command.target_points[0]

            node_with_weight = graph.get_node_by_name(tp.target_node_name)

            weight_node = get_const_node(node_with_weight, tp.input_port_id, graph)
            weight_name = weight_node.layer_attributes.name
            module_name, weight_attr_name = split_const_name(weight_name)
            module = get_module_by_name(module_name, model)
            w = getattr(module, weight_attr_name)
            if w is None or not isinstance(w, torch.nn.Parameter):
                raise nncf.InternalError(f"Could not find a torch.nn.Parameter in the model by name {weight_name}.")

            max_value = quantizer_module.scale
            max_value = max_value.to(w.dtype)
            input_ = w + quantizer_module._lora_B @ quantizer_module._lora_A
            input_ = input_.to(w.dtype)
            input_ = input_.reshape(quantizer_module._qspec.weight_shape)

            level_high = 2 ** (quantizer_module.num_bits - 1)
            scale = max_value / level_high
            scale = torch.where(torch.abs(scale) < quantizer_module.eps, quantizer_module.eps, scale)
            output = input_ / scale
            output = output.round()
            output = output.clip(min=quantizer_module.level_low, max=quantizer_module.level_high)
            output = output.to(torch.int8)

            if quantizer_module.num_bits == 8:
                decompressor = INT8SymmetricWeightsDecompressor(
                    scale=scale,
                    result_dtype=w.dtype
                )
            else:
                decompressor = INT4SymmetricWeightsDecompressor(
                    scale=scale,
                    compressed_weight_shape=output.shape,
                    result_shape=w.shape,
                    result_dtype=w.dtype,
                )

            packed_tensor = decompressor.pack_weight(output)

            # tmp = decompressor(packed_tensor)

            # sets compressed tensor
            compressed_parameter = torch.nn.Parameter(packed_tensor, requires_grad=False)
            setattr(module, weight_attr_name, compressed_parameter)

            consumer_nodes = graph.get_next_nodes(weight_node)
            if len(consumer_nodes) > 1:
                for c_node in consumer_nodes:
                    c_module = model.get_module_by_scope(Scope.from_str(c_node.layer_name))
                    for name, param in c_module.named_parameters(recurse=False, remove_duplicate=False):
                        if id(param) == id(w):
                            setattr(c_module, name, compressed_parameter)

            # registry weight decompression module in the model
            decompressor_name = f"weights_decompressor_{weight_node.node_name.replace('.', '_')}"

            # inserts the weight decompressor into the model as the post hook on the model weight
            transformation_layout.register(
                PTSharedFnInsertionCommand(
                    [PTTargetPoint(TargetType.OPERATOR_POST_HOOK, target_node_name=weight_node.node_name)],
                    decompressor,
                    decompressor_name,
                )
            )

    return PTModelTransformer(model).transform(transformation_layout)