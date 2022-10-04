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

from typing import Dict, Tuple
from typing import List
from typing import TypeVar
from typing import Union

import numpy as np
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.tensor import NNCFTensor
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.utils.backend import BackendType, get_backend
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.experimental.post_training.algorithms import AlgorithmParameters
from nncf.experimental.post_training.algorithms.algorithm import Algorithm
from nncf.experimental.post_training.algorithms.algorithm import PostTrainingAlgorithms
from nncf.experimental.post_training.algorithms.fast_bias_correction.backend import ALGO_BACKENDS
from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.factories import NNCFGraphFactory
from nncf.experimental.post_training.statistics.statistic_point import StatisticPoint
from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer

ModelType = TypeVar('ModelType')


class FastBiasCorrectionParameters(AlgorithmParameters):
    """
    Base class of FastBiasCorrection parameters.
    """

    def __init__(self, number_samples: int = 100, threshold: float = 2.0) -> None:
        """
        Initial method for the FastBiasCorrectionParameters

        :param number_samples: number of the samples for the statistics collection
        :param threshold: threshold that regulates the application of the shift (or not)
        """
        self.number_samples = number_samples
        self.threshold = threshold

    def to_json(self) -> Dict[str, Union[str, float, int]]:
        """
        Serialize all FastBiasCorrection parameters to JSON.
        """


# pylint: disable = too-many-function-args
class FastBiasCorrection(Algorithm):

    def __init__(self, parameters: FastBiasCorrectionParameters):
        super().__init__()
        self.number_samples = parameters.number_samples
        self.threshold = parameters.threshold
        self.nncf_graph = None
        self._backend_entity = None

    @property
    def available_backends(self) -> List[BackendType]:
        return ALGO_BACKENDS.registry_dict

    def _set_backend_entity(self, model: ModelType) -> None:
        """
        Factory method for creating the needed backend based on the model

        :param model: backend-specific input model
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.ONNX:
            from nncf.experimental.post_training.algorithms.fast_bias_correction.onnx_backend import ONNXFBCAlgoBackend
            self._backend_entity = ONNXFBCAlgoBackend()
        else:
            raise RuntimeError('Cannot return backend-specific entity'
                               'because {} is not supported!'.format(model_backend))

    def _apply(self, model: ModelType, engine: Engine,
               statistic_points: StatisticPointsContainer) -> ModelType:

        self._set_backend_entity(model)

        transformation_layout = TransformationLayout()
        nncf_graph = NNCFGraphFactory.create(model)

        layers_with_bias_types = self._backend_entity.layers_with_bias_metatypes
        biased_nodes = nncf_graph.get_nodes_by_metatypes(layers_with_bias_types)

        for node in biased_nodes:
            node_name = node.node_name

            if not self._is_node_with_bias(node):
                nncf_logger.debug('Skipping node {} because there is no bias'.format(node_name))
                continue
            if not self._is_quantized_weights(node, nncf_graph):
                nncf_logger.debug('Skipping node {} because weights was not quantized'.format(node_name))
                continue

            input_fp, input_shape = self._get_fp_inputs(statistic_points, node_name)
            output_fp = self._get_fp_outputs(statistic_points, node_name)

            input_name = node.layer_attributes.input_tensor_names[0]
            output_name = node.layer_attributes.output_tensor_names[0]

            extracted_model = self._extract_submodel([input_name], [output_name])

            channel_axis = self._backend_entity.channel_axis_by_types[node.node_type]
            input_blob = self._create_input_blob(input_shape,
                                                 input_fp,
                                                 input_name)
            bias_shift = self._get_bias_shift(
                engine=engine,
                model=extracted_model,
                input_blob=input_blob,
                channel_axis=channel_axis,
                output_fp=output_fp,
                output_name=output_name)

            target_point = self._backend_entity.target_point(TargetType.LAYER,
                                                             node.node_name)
            bias_correction_command = self._backend_entity.bias_correction_command(target_point,
                                                                                   bias_shift,
                                                                                   self.threshold)
            transformation_layout.register(bias_correction_command)

        quantized_model = self._model_transformer.transform(transformation_layout)
        return quantized_model

    def _get_fp_inputs(self, statistic_points: StatisticPointsContainer, node_name: str) -> Tuple[List, List]:
        """
        Makes out per-layer needed data from the floating-point collected statistics

        :param statistic_points: filled StatisticPointsContainer
        :param node_name: name of the current layer
        :return: collected mean tensor data and shape for the further bias calculation
        """
        def input_filter_func(point):
            return PostTrainingAlgorithms.FastBiasCorrection in point.algorithm_to_tensor_collectors and \
                point.target_point.type == TargetType.PRE_LAYER_OPERATION

        input_fp = []
        input_shape = []
        for tensor_collector in statistic_points.iter_through_algorithm_tensor_collectors_in_target_node(
                node_name,
                input_filter_func,
                PostTrainingAlgorithms.FastBiasCorrection):
            input_fp.extend(tensor_collector.get_statistics().mean_values)
            input_shape.extend(tensor_collector.get_statistics().shape)
        return input_fp, input_shape

    def _get_fp_outputs(self, statistic_points: StatisticPointsContainer, node_name: str) -> List[np.ndarray]:
        """
        Makes out per-layer needed data from the floating-point collected statistics

        :param statistic_points: filled StatisticPointsContainer
        :param node_name: name of the current layer
        :return: collected mean tensor data for the further bias calculation
        """
        def output_filter_func(point):
            return PostTrainingAlgorithms.FastBiasCorrection in point.algorithm_to_tensor_collectors and \
                point.target_point.type == TargetType.POST_LAYER_OPERATION

        output_fp = []
        for tensor_collector in statistic_points.iter_through_algorithm_tensor_collectors_in_target_node(
                node_name,
                output_filter_func,
                PostTrainingAlgorithms.FastBiasCorrection):
            output_fp.extend(tensor_collector.get_statistics().mean_values)
        return output_fp

    def _extract_submodel(self, input_names: List[str], output_names: List[str]) -> ModelType:
        """
        Extracts sub-model from the original based on the input & output tensor names

        :param input_names: list of the input names
        :param output_names: list of the output names
        :return: backend-specific sub-model
        """
        model_extraction_command = self._backend_entity.model_extraction_command(input_names,
                                                                                 output_names)
        me_transformation_layout = TransformationLayout()
        me_transformation_layout.register(model_extraction_command)
        extracted_model = self._model_transformer.transform(me_transformation_layout)
        return extracted_model

    def _add_statistic_point(self, container: StatisticPointsContainer, point: TargetPoint, axis: int) -> None:
        """
        Adds specific statistic point

        :param container: StatisticPointsContainer
        :param point: TargetPoint for statistic collection
        :param axis: channel axis for the statistics calculation
        """
        stat_collector = self._backend_entity.mean_statistic_collector(reduction_shape=axis,
                                                                       num_samples=self.number_samples)
        container.add_statistic_point(StatisticPoint(target_point=point,
                                                     tensor_collector=stat_collector,
                                                     algorithm=PostTrainingAlgorithms.FastBiasCorrection))

    def _create_input_blob(self,
                           input_shape: Tuple[int],
                           input_fp: List[np.ndarray],
                           input_name: str) -> Dict[str, NNCFTensor]:
        """
        Creates input blob for the bias shift calculation

        :param input_shape: input shape for the blob
        :param input_fp: input data for the blob
        :param input_name: name for the output dict
        :return: dictionary of the blob by input name
        """
        input_blob = np.zeros(input_shape)
        for i, value in enumerate(input_fp):
            input_blob[:, i] = value
        input_blob = input_blob.astype(np.float32)

        input_data = self._backend_entity.nncf_tensor(input_blob)
        input_data = {input_name: input_data}
        return input_data

    def _get_bias_shift(self,
                        engine: Engine,
                        model: ModelType,
                        input_blob: Dict[str, NNCFTensor],
                        channel_axis: Tuple[int],
                        output_fp: List[np.ndarray],
                        output_name: str) -> np.ndarray:
        """
        Calculates bias shift for the further corretion

        :param engine: backend-specific engine instance for the model execution
        :param model: backend-specific sub-model for the execution
        :param input_blob: input data for the execution
        :param channel_axis: channel axes for the raw data aggregation
        :param output_fp: output data for the shift calculation
        :param output_name: name of the output tensor for the data collection
        :return: calculated bias shift
        """
        engine.set_model(model)
        q_outputs = engine.infer(input_blob)
        q_outputs = q_outputs[output_name]
        q_outputs = self._backend_entity.tensor_processor.mean_per_channel(q_outputs, channel_axis).tensor
        bias_shift = np.array(output_fp) - q_outputs
        return bias_shift

    @staticmethod
    def _is_node_with_bias(node: NNCFNode) -> bool:
        """
        Checks whether the node has a bias or not

        :param node: NNCFNode with the attributes
        :return: boolean indicating whether the node has a bias or not
        """
        # Should be updated to take account of backend specifics
        return len(node.layer_attributes.input_tensor_names) > 2

    @staticmethod
    def _is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        """
        Checks whether the node is quantised or not

        :param node: NNCFNode with the attributes
        :param nncf_graph: NNCFGraph for the traverce
        :return: boolean indicating whether the node has a quantized weights or not
        """
        # Should be updated to take account of backend specifics
        input_nodes = nncf_graph.get_previous_nodes(node)
        weight_dequantizer = input_nodes[1]
        return weight_dequantizer.node_type == 'DequantizeLinear'

    def get_statistic_points(self, model: ModelType) -> StatisticPointsContainer:
        self._set_backend_entity(model)
        nncf_graph = NNCFGraphFactory.create(model) if self.nncf_graph is None else self.nncf_graph
        layers_with_bias_types = self._backend_entity.layers_with_bias_metatypes
        biased_nodes = nncf_graph.get_nodes_by_metatypes(layers_with_bias_types)

        statistic_container = StatisticPointsContainer()

        for node in biased_nodes:
            if not self._is_node_with_bias(node):
                continue
            edge_name = node.layer_attributes.input_tensor_names[0]
            pre_layer_statistic_point = self._backend_entity.target_point(TargetType.PRE_LAYER_OPERATION,
                                                                          node.node_name,
                                                                          edge_name)
            post_layer_statistic_point = self._backend_entity.target_point(TargetType.POST_LAYER_OPERATION,
                                                                           node.node_name)
            channel_axis = self._backend_entity.channel_axis_by_types[node.node_type]

            self._add_statistic_point(statistic_container,
                                      pre_layer_statistic_point,
                                      channel_axis)
            self._add_statistic_point(statistic_container,
                                      post_layer_statistic_point,
                                      channel_axis)

        return statistic_container
