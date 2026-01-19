# -*- coding: utf-8 -*-
import random
import torch
from collections import deque, namedtuple
from copy import deepcopy
from logger import debug

# GNN的经验元组，存储一个完整的系统转换
GNNExperience = namedtuple('GNNExperience',
                           ['graph_t', 'actions_t', 'rewards_t', 'graph_t1'])


class GNNReplayBuffer:
    """
    为 GNN-DRL 准备的经验回放缓冲区。
    它存储的是完整的图（Graph）转换，而不是单个智能体的状态。
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        debug(f"GNNReplayBuffer initialized with capacity {capacity}")

    def __len__(self):
        return len(self.buffer)

    def _graphs_to_device(self, graph_data, device):
        """辅助函数：将图数据字典（的张量）移动到指定设备"""
        if graph_data is None:
            return None

        # 深度复制以避免修改缓冲区中的原始数据
        graph_data_copy = deepcopy(graph_data)

        try:
            graph_data_copy['node_features']['features'] = graph_data_copy['node_features']['features'].to(device)
            graph_data_copy['node_features']['types'] = graph_data_copy['node_features']['types'].to(device)

            # 假设 gnn_model.edge_types 是可访问的，或者硬编码
            edge_types = ['communication', 'interference', 'proximity']
            for edge_type in edge_types:
                if graph_data_copy['edge_features'][edge_type] is not None:
                    graph_data_copy['edge_features'][edge_type]['edge_index'] = \
                        graph_data_copy['edge_features'][edge_type]['edge_index'].to(device)
                    graph_data_copy['edge_features'][edge_type]['edge_attr'] = \
                        graph_data_copy['edge_features'][edge_type]['edge_attr'].to(device)
            return graph_data_copy
        except Exception as e:
            debug(f"Error moving graph to {device}: {e}")
            return None

    def add(self, graph_t, actions_t, rewards_t, graph_t1):
        """
        添加一个完整的系统转换经验。

        Args:
            graph_t (dict): t 时刻的图数据 (来自 GraphBuilder)
            actions_t (dict): {dqn_id: action_index} 的字典
            rewards_t (dict): {dqn_id: reward} 的字典
            graph_t1 (dict): t+1 时刻的图数据
        """
        if graph_t is None or graph_t1 is None:
            debug("GNNReplayBuffer: Skipping add due to None graph")
            return

        # 1. 将所有图数据中的张量移到 CPU 存储，节省 GPU 显存
        graph_t_cpu = self._graphs_to_device(graph_t, 'cpu')
        graph_t1_cpu = self._graphs_to_device(graph_t1, 'cpu')

        # 2. 确保 actions 和 rewards 也是纯 Python 类型
        actions_t_cpu = deepcopy(actions_t)
        rewards_t_cpu = deepcopy(rewards_t)

        # 3. 创建经验元组
        experience = GNNExperience(
            graph_t=graph_t_cpu,
            actions_t=actions_t_cpu,
            rewards_t=rewards_t_cpu,
            graph_t1=graph_t1_cpu
        )

        # 4. 存入缓冲区
        self.buffer.append(experience)
        # debug(f"GNN Experience added. Buffer size: {len(self.buffer)}") # (信息量太大，建议注释掉)

    def sample(self, batch_size, device):
        """
        从缓冲区中采样一个批次，并自动将图数据移到目标设备。

        Args:
            batch_size (int): 批次大小
            device (torch.device): 目标设备 (e.g., 'cuda')

        Returns:
            list[GNNExperience]: 包含 (graph_t, actions_t, rewards_t, graph_t1) 的列表
                                 其中 graph_t 和 graph_t1 已经
                                 被移到了目标 device。
        """
        if len(self.buffer) < batch_size:
            return None

        # 1. 随机采样
        sampled_experiences = random.sample(self.buffer, batch_size)

        # 2. 将采样到的数据的图张量移到目标设备 (e.g., 'cuda')
        batch_on_device = []
        for exp in sampled_experiences:
            graph_t_dev = self._graphs_to_device(exp.graph_t, device)
            graph_t1_dev = self._graphs_to_device(exp.graph_t1, device)

            if graph_t_dev and graph_t1_dev:
                batch_on_device.append(GNNExperience(
                    graph_t=graph_t_dev,
                    actions_t=exp.actions_t,  # actions 和 rewards 不需要
                    rewards_t=exp.rewards_t,  # 移到GPU
                    graph_t1=graph_t1_dev
                ))

        return batch_on_device