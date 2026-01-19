# GNNIntegration.py
import torch
import torch.nn as nn
from logger import debug, debug_print
from GraphBuilder import global_graph_builder
from GNNModel import global_gnn_model, GNNDQN
from Parameters import USE_GNN_ENHANCEMENT

class GNNIntegrationManager:

    def __init__(self, use_gnn=True):
        self.use_gnn = use_gnn and USE_GNN_ENHANCEMENT
        self.gnn_model = global_gnn_model if self.use_gnn else None
        self.graph_builder = global_graph_builder
        self.enhanced_dqns = {}

        debug(f"GNN Integration Manager initialized: use_gnn={self.use_gnn}")

    def enhance_dqn_with_gnn(self, original_dqn):
        if not self.use_gnn or self.gnn_model is None:
            return original_dqn

        if original_dqn.dqn_id not in self.enhanced_dqns:
            gnn_dqn = GNNDQN(self.gnn_model, original_dqn)
            self.enhanced_dqns[original_dqn.dqn_id] = gnn_dqn
            debug(f"Enhanced DQN {original_dqn.dqn_id} with GNN")

        return self.enhanced_dqns[original_dqn.dqn_id]

    def build_and_process_graph(self, dqn_list, vehicle_list, epoch):
        if not self.use_gnn:
            return None

        try:
            # 构建动态图
            graph_data = self.graph_builder.build_dynamic_graph(dqn_list, vehicle_list, epoch)

            # 可视化图结构（调试用）
            if DEBUG_MODE:
                self.graph_builder.visualize_graph_simple(graph_data)

            return graph_data

        except Exception as e:
            debug(f"Error in graph building: {e}")
            return None

    def get_gnn_enhanced_q_values(self, graph_data, dqn_id=None):
        if not self.use_gnn or graph_data is None:
            return None

        try:
            device = next(self.gnn_model.parameters()).device

            # 确保所有数据都在正确设备上
            graph_data['node_features']['features'] = graph_data['node_features']['features'].to(device)
            graph_data['node_features']['types'] = graph_data['node_features']['types'].to(device)

            for edge_type in self.gnn_model.edge_types:
                if graph_data['edge_features'][edge_type] is not None:
                    graph_data['edge_features'][edge_type]['edge_index'] = \
                        graph_data['edge_features'][edge_type]['edge_index'].to(device)
                    graph_data['edge_features'][edge_type]['edge_weights'] = \
                        graph_data['edge_features'][edge_type]['edge_weights'].to(device)

            # 获取GNN Q值
            with torch.no_grad():
                if dqn_id is not None:
                    q_values = self.gnn_model(graph_data, dqn_id)
                else:
                    q_values = self.gnn_model(graph_data)

            return q_values

        except Exception as e:
            debug(f"Error in GNN Q value calculation: {e}")
            return None


# 全局GNN集成管理器
global_gnn_manager = GNNIntegrationManager()