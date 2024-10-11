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

from copy import deepcopy
from typing import Callable, List, Optional, TypeVar

import numpy as np

from nncf import Dataset
from nncf import ModelType
from nncf import QuantizationMode
from nncf import TargetDevice
from nncf.common.factory import NNCFGraphFactory
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.statistics.collectors import get_raw_stat_collector
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.bias_correction.algorithm import BIAS_CORRECTION_THRESHOLD
from nncf.quantization.algorithms.bias_correction.algorithm import BiasCorrection
from nncf.quantization.algorithms.channel_alignment.algorithm import ChannelAlignment
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FAST_BIAS_CORRECTION_THRESHOLD
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FastBiasCorrection
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.smooth_quant.algorithm import SmoothQuant
from nncf.scopes import IgnoredScope
from nncf.tensor import functions as fns

TModel = TypeVar("TModel")
TPass = Callable[[TModel], TModel]

ALPHAS_GRID = [
    0.0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
]
# ALPHAS_GRID = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


class SmoothQuantTuner(Algorithm):
    def __init__(
        self,
        mode: Optional[QuantizationMode] = None,
        preset: Optional[QuantizationPreset] = None,
        target_device: TargetDevice = TargetDevice.ANY,
        subset_size: int = 300,
        fast_bias_correction: bool = True,
        model_type: Optional[ModelType] = None,
        ignored_scope: Optional[IgnoredScope] = None,
        advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
    ):
        self._algorithm_key = f"SQ_{hash(self)}_TUNE"
        self._alphas_grid = ALPHAS_GRID
        self._subset_size = subset_size

        self._smooth_quant = None
        self._channel_alignment = None
        self._min_max = None
        self._bias_correction = None

        if model_type == ModelType.TRANSFORMER:
            self._smooth_quant = SmoothQuant(subset_size, advanced_parameters.inplace_statistics)

        if not advanced_parameters.disable_channel_alignment:
            self._channel_alignment = ChannelAlignment(subset_size, advanced_parameters.inplace_statistics)

        self._min_max = MinMaxQuantization(
            mode=mode,
            preset=preset,
            target_device=target_device,
            subset_size=subset_size,
            model_type=model_type,
            ignored_scope=ignored_scope,
            overflow_fix=advanced_parameters.overflow_fix,
            quantize_outputs=advanced_parameters.quantize_outputs,
            inplace_statistics=advanced_parameters.inplace_statistics,
            batchwise_statistics=advanced_parameters.batchwise_statistics,
            activations_quantization_params=advanced_parameters.activations_quantization_params,
            weights_quantization_params=advanced_parameters.weights_quantization_params,
            activations_range_estimator_params=advanced_parameters.activations_range_estimator_params,
            weights_range_estimator_params=advanced_parameters.weights_range_estimator_params,
            backend_params=advanced_parameters.backend_params,
        )

        if not advanced_parameters.disable_bias_correction:
            bias_correction_params = advanced_parameters.bias_correction_params
            if fast_bias_correction:
                threshold = FAST_BIAS_CORRECTION_THRESHOLD
                bias_correction_subset_size = subset_size
                bias_correction_cls = FastBiasCorrection
            else:
                threshold = BIAS_CORRECTION_THRESHOLD
                bias_correction_subset_size = max(int(subset_size * 0.2), 1)
                bias_correction_cls = BiasCorrection

            if bias_correction_params.threshold is not None:
                threshold = bias_correction_params.threshold

            self._bias_correction = bias_correction_cls(
                bias_correction_subset_size,
                threshold,
                bias_correction_params.apply_for_all_nodes,
                advanced_parameters.inplace_statistics,
                advanced_parameters.backend_params,
            )

    @property
    def available_backends(self) -> List[BackendType]:
        return self._smooth_quant.available_backends

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        return StatisticPointsContainer()

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        sq_statistic_points = self._smooth_quant.get_statistic_points(model, graph)
        sq_statistics = self.collect_statistics(sq_statistic_points, model, graph, dataset)

        output_port_id = 0

        alpha_map = self._smooth_quant._get_alpha_map()
        nodes_to_smooth = self._smooth_quant._get_nodes_to_smooth_data(graph, alpha_map)
        node_groups = self._smooth_quant._group_nodes_by_source(nodes_to_smooth, graph)
        group_losses = {gid: np.inf for gid in node_groups}

        nodes_list = list(sq_statistic_points.keys())
        nodes_alpha = {n: 0 for n in nodes_list}

        fp_container = self.get_container_for_nodes(nodes_list, self._algorithm_key, output_port_id, self._subset_size)

        nncf_logger.info(">>> Collecting FP statistics for tune")
        fp_post_layer_statistics = self.collect_statistics(fp_container, model, graph, dataset)

        nncf_logger.info(">>> Applying tuning from space:")
        nncf_logger.info(self._alphas_grid)

        for alpha in self._alphas_grid:
            nncf_logger.info(f"Tuning with {alpha} alpha")
            modified_model = deepcopy(model)
            self._smooth_quant._alpha_map["matmul"] = alpha

            modified_nncf_graph = NNCFGraphFactory.create(modified_model)
            modified_model = self._smooth_quant.apply(modified_model, modified_nncf_graph, sq_statistics)

            modified_nncf_graph = NNCFGraphFactory.create(modified_model)
            min_max_statistic_points = self._min_max.get_statistic_points(modified_model, modified_nncf_graph)

            nncf_logger.info(">>> Collecting statistics for MinMax")
            min_max_statistics = self.collect_statistics(
                min_max_statistic_points, modified_model, modified_nncf_graph, dataset
            )
            modified_model = self._min_max.apply(modified_model, modified_nncf_graph, min_max_statistics)
            del min_max_statistic_points, min_max_statistics
            modified_nncf_graph = NNCFGraphFactory.create(modified_model)

            int_container = self.get_container_for_nodes(
                nodes_list, self._algorithm_key, output_port_id, self._subset_size
            )
            nncf_logger.info(">>> Collecting INT statistics for tune")
            int_post_layer_statistics = self.collect_statistics(
                int_container, modified_model, modified_nncf_graph, dataset
            )

            for group_id, nodes in node_groups.items():
                current_loss = 0
                for node_to_smooth in nodes:
                    node_name = node_to_smooth.node_name
                    fp_statistics = self.get_statistics_for_node(
                        fp_post_layer_statistics, node_name, self._algorithm_key, output_port_id
                    )[0]
                    int_statistics = self.get_statistics_for_node(
                        int_post_layer_statistics, node_name, self._algorithm_key, output_port_id
                    )[0]

                    for fp_stat, int_stat in zip(fp_statistics, int_statistics):
                        current_loss += fns.sum(fns.abs(fp_stat - int_stat) ** 2)

                if current_loss < group_losses[group_id]:
                    group_losses[group_id] = current_loss
                    for node in nodes:
                        nodes_alpha[node.node_name] = alpha

            del int_post_layer_statistics

        nncf_logger.info(">>> Final quantization")
        self._smooth_quant._alpha_map_by_layers = nodes_alpha
        nncf_logger.info(self._smooth_quant._alpha_map_by_layers)
        print(self._smooth_quant._alpha_map_by_layers)
        model = self._smooth_quant.apply(model, graph, sq_statistics)
        del sq_statistic_points, sq_statistics
        graph = NNCFGraphFactory.create(model)

        if self._channel_alignment:
            ch_al_statistic_points = self._channel_alignment.get_statistic_points(model, graph)
            ch_al_statistics = self.collect_statistics(ch_al_statistic_points, model, graph, dataset)
            model = self._channel_alignment(model, graph, ch_al_statistics)
            del ch_al_statistic_points, ch_al_statistics
            graph = NNCFGraphFactory.create(model)

        min_max_statistic_points = self._min_max.get_statistic_points(model, graph)
        min_max_statistics = self.collect_statistics(min_max_statistic_points, model, graph, dataset)

        if self._bias_correction:
            fast_bc_statistics_points = self._bias_correction.get_statistic_points(model, graph)
            fast_bc_statistics = self.collect_statistics(fast_bc_statistics_points, model, graph, dataset)

        model = self._min_max.apply(model, graph, min_max_statistics)

        if self._bias_correction:
            graph = NNCFGraphFactory.create(model)
            model = self._bias_correction.apply(model, graph, fast_bc_statistics)

        return model

    @staticmethod
    def collect_statistics(
        container,
        model,
        graph,
        dataset,
    ):
        statistics_aggregator = StatisticsAggregatorFactory.create(model, dataset)
        statistics_aggregator.register_statistic_points(container)
        statistics_aggregator.collect_statistics(model, graph)
        return statistics_aggregator.statistic_points

    @staticmethod
    def get_statistics_for_node(statistic_points, node_name, key, output_port_id):
        def filter_func(point: StatisticPoint) -> bool:
            return (
                key in point.algorithm_to_tensor_collectors
                and point.target_point.type == TargetType.POST_LAYER_OPERATION
                and point.target_point.port_id == output_port_id
            )

        statistics_for_node = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(
            node_name,
            filter_func,
            key,
        ):
            statistic = tensor_collector.get_statistics().values
            statistics_for_node.append(statistic)
        return statistics_for_node

    @staticmethod
    def get_container_for_nodes(node_names, key, output_port_id, subset_size):
        container = StatisticPointsContainer()
        for node_name in node_names:
            target_point = OVTargetPoint(TargetType.POST_LAYER_OPERATION, node_name, output_port_id)
            stat_collector = get_raw_stat_collector(subset_size)
            container.add_statistic_point(
                StatisticPoint(
                    target_point=target_point,
                    tensor_collector=stat_collector,
                    algorithm=key,
                )
            )
        return container
