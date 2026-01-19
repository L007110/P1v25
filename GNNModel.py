# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from logger import debug, debug_print, set_debug_mode
from Parameters import *
import Parameters


class EnhancedHeteroGNN(nn.Module):
    def __init__(self, node_feature_dim=9, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.2):
        super(EnhancedHeteroGNN, self).__init__()

        # 动态读取当前架构模式
        self.arch_type = getattr(Parameters, 'GNN_ARCH', 'HYBRID')
        debug_print(f"Initializing GNN Model with Architecture: {self.arch_type}")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        from GraphBuilder import global_graph_builder
        self.edge_feature_dim = global_graph_builder.comm_edge_feature_dim
        self.edge_types = ['communication', 'interference', 'proximity']

        # 1. 节点嵌入
        # 将节点类型（0或1）映射为向量
        self.node_type_embedding = nn.Embedding(2, hidden_dim // 4)

        # 计算每一层的输入维度
        # Layer 0 输入: 原始特征 + 类型嵌入
        initial_input_dim = node_feature_dim + (hidden_dim // 4)

        # 2. 定义图卷积层 (ModuleDict 嵌套 ModuleList)
        self.edge_type_layers = nn.ModuleDict()

        for edge_type in self.edge_types:
            layers = nn.ModuleList()
            for i in range(num_layers):
                # 第一层输入是原始特征，后续层输入是 Hidden Dim
                curr_in = initial_input_dim if i == 0 else hidden_dim

                # GAT 多头处理：如果不是最后一层，输出需要除以头数以便拼接
                # 这里简化处理：我们让 GAT 输出拼接后等于 hidden_dim
                curr_out = hidden_dim // num_heads if (self.arch_type != "GCN" and i < num_layers - 1) else hidden_dim

                if self.arch_type == "GCN":
                    layers.append(GCNConv(curr_in, hidden_dim))
                else:
                    # GAT / HYBRID
                    heads = num_heads if i < num_layers - 1 else 1
                    concat = True if i < num_layers - 1 else False
                    layers.append(GATConv(curr_in, curr_out, heads=heads, dropout=dropout,
                                          edge_dim=self.edge_feature_dim, concat=concat))
            self.edge_type_layers[edge_type] = layers

        # 3. HYBRID 专属组件: 边门控
        if self.arch_type == "HYBRID":
            self.edge_type_gates = nn.Parameter(torch.zeros(len(self.edge_types)))

        # 4. 边类型融合权重
        self.edge_type_attention = nn.Parameter(torch.ones(len(self.edge_types)))

        # 5. [关键修复] 残差投影层
        # 只有第一层维数改变时需要线性映射，后续层直接相加 (Identity)
        self.residual_projections = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # Layer 0: 维度从 Input -> Hidden
                self.residual_projections.append(nn.Linear(initial_input_dim, hidden_dim))
            else:
                # Layer > 0: 维度保持 Hidden，直接使用 Identity (也就是 x = x)
                self.residual_projections.append(nn.Identity())

        # 6. [关键修复] 层归一化
        # 防止深层网络中的梯度消失或爆炸，特别是在残差连接后
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # 7. 输出层
        self.attn_pool_linear = nn.Linear(hidden_dim, 1)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, RL_N_ACTIONS)
        )

    def forward(self, graph_data, dqn_id=None):
        node_features = graph_data['node_features']['features']
        node_types = graph_data['node_features']['types']
        edge_features = graph_data['edge_features']

        # 1. 构建初始特征 X
        type_embedding = self.node_type_embedding(node_types)
        x = torch.cat([node_features, type_embedding], dim=1)  # Shape: [N, initial_input_dim]

        # 准备边权重 (Hybrid 门控)
        edge_gates = torch.sigmoid(self.edge_type_gates) if self.arch_type == "HYBRID" else None
        edge_type_weights = F.softmax(self.edge_type_attention, dim=0)

        # 2. [重构] 逐层前向传播 (Layer-wise Forward)
        # 这是一个大循环：Layer 0 -> Layer 1 -> ...
        for i in range(self.num_layers):
            x_in = x  # 保存当前层的输入，用于残差连接

            # --- 分支聚合：计算每种边类型的 GNN 输出 ---
            layer_outputs = []

            for type_idx, edge_type in enumerate(self.edge_types):
                # 获取当前层的 GNN 模块
                gnn_layer = self.edge_type_layers[edge_type][i]

                # 获取边数据
                if edge_features[edge_type] is None:
                    # 如果某种边不存在 (例如没有干扰边)，补零
                    layer_outputs.append(torch.zeros(x.size(0), self.hidden_dim, device=x.device))
                    continue

                edge_index = edge_features[edge_type]['edge_index']
                edge_attr = edge_features[edge_type]['edge_attr']

                # 处理边特征 (Hybrid 门控)
                if self.arch_type == "HYBRID":
                    gated_attr = edge_attr * edge_gates[type_idx]
                else:
                    gated_attr = edge_attr

                # 前向计算
                if self.arch_type == "GCN":
                    out = gnn_layer(x, edge_index)
                else:
                    out = gnn_layer(x, edge_index, edge_attr=gated_attr)

                # 收集结果
                if self.arch_type == "GCN":
                    layer_outputs.append(out)
                else:
                    # GAT/Hybrid 使用可学习的类型权重进行加权
                    layer_outputs.append(out * edge_type_weights[type_idx])

            # --- 融合：将不同边类型的结果相加 ---
            if layer_outputs:
                x_combined = torch.sum(torch.stack(layer_outputs), dim=0)
            else:
                x_combined = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

            # --- [核心] 残差连接 (Residual Connection) ---
            # 公式: x_new = x_combined + Projection(x_in)
            # 即使 x_combined 是 0 (孤岛节点)，信息也能通过 x_res 传下去
            x_res = self.residual_projections[i](x_in)
            x = x_combined + x_res

            # --- [核心] 归一化与激活 ---
            x = self.layer_norms[i](x)  # LayerNorm
            x = F.relu(x)  # ReLU
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 3. 输出提取 (保持不变)
        if dqn_id is not None:
            q_values = self._extract_local_features(x, graph_data, dqn_id)
        else:
            q_values = self._extract_global_features(x, graph_data)

        aux_info = self.edge_type_attention if self.arch_type == "HYBRID" else None
        return q_values, aux_info

    def _extract_local_features(self, node_embeddings, graph_data, dqn_id):
        # 提取指定 RSU 的 Q 值 (保持逻辑不变)
        nodes = graph_data['nodes']
        target_rsu_index = -1
        for i, rsu_node in enumerate(nodes['rsu_nodes']):
            if rsu_node['original_id'] == dqn_id:
                target_rsu_index = i
                break
        if target_rsu_index == -1:
            return torch.zeros(RL_N_ACTIONS, device=node_embeddings.device)

        rsu_embedding = node_embeddings[target_rsu_index]
        vehicle_embeddings = []

        # 寻找与该 RSU 通信的车辆
        for vehicle_node in nodes['vehicle_nodes']:
            # 这里简化逻辑：在 GraphBuilder 中 RSU 和服务的车之间有 communication 边
            # 遍历边列表寻找 target
            found = False
            if 'communication' in graph_data['edges']:
                for edge in graph_data['edges']['communication']:
                    if (edge['source'] == f"rsu_{dqn_id}" and edge['target'] == vehicle_node['id']):
                        vehicle_index = len(nodes['rsu_nodes']) + nodes['vehicle_nodes'].index(vehicle_node)
                        vehicle_embeddings.append(node_embeddings[vehicle_index])
                        found = True
                        break
            if not found:
                pass  # 没有找到服务的车

        if vehicle_embeddings:
            vehicle_stack = torch.stack(vehicle_embeddings)
            # 使用 Attention Pooling 聚合车辆特征
            attn_scores = self.attn_pool_linear(vehicle_stack)
            attn_weights = F.softmax(attn_scores, dim=0)
            vehicle_embedding = torch.mm(attn_weights.t(), vehicle_stack).squeeze(0)
        else:
            vehicle_embedding = torch.zeros_like(rsu_embedding)

        combined_features = torch.cat([rsu_embedding, vehicle_embedding], dim=0)
        q_values = self.output_layer(combined_features)
        return q_values

    def _extract_global_features(self, node_embeddings, graph_data):
        nodes = graph_data['nodes']
        num_rsus = len(nodes['rsu_nodes'])
        all_q_values = []
        for dqn_id in range(1, num_rsus + 1):
            q_value = self._extract_local_features(node_embeddings, graph_data, dqn_id)
            all_q_values.append(q_value)
        if all_q_values:
            return torch.stack(all_q_values, dim=0)
        else:
            return torch.zeros(0, RL_N_ACTIONS, device=node_embeddings.device)

    def get_attention_weights(self, graph_data):
        attention_info = {
            'edge_type_weights': F.softmax(self.edge_type_attention, dim=0).detach().cpu().numpy(),
            'edge_types': self.edge_types
        }
        return attention_info


# 全局模型初始化 (保持不变)
global_gnn_model = EnhancedHeteroGNN(
    node_feature_dim=12,
    hidden_dim=64,
    num_heads=4,
    num_layers=2,
    dropout=0.2
)

global_target_gnn_model = EnhancedHeteroGNN(
    node_feature_dim=12,
    hidden_dim=64,
    num_heads=4,
    num_layers=2,
    dropout=0.2
)


def update_target_gnn():
    global_target_gnn_model.load_state_dict(global_gnn_model.state_dict())
    global_target_gnn_model.eval()
    debug(f"Global Target GNN ({getattr(Parameters, 'GNN_ARCH', 'Hybrid')}) updated")


def update_target_gnn_soft(tau):
    try:
        with torch.no_grad():
            for target_param, online_param in zip(global_target_gnn_model.parameters(), global_gnn_model.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
    except Exception as e:
        debug(f"Error during GNN soft update: {e}")


# 初始化并同步
update_target_gnn()
debug_print(f"Global GNN ({getattr(Parameters, 'GNN_ARCH', 'Hybrid')}) initialized and synced.")

if __name__ == "__main__":
    set_debug_mode(True)
    debug_print("GNNModel.py (Fixed Residual Version) loaded.")