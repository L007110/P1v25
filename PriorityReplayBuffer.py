# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from collections import namedtuple
from logger import debug, debug_print

# 经验元组
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])


class PriorityReplayBuffer:
    """
    优先级经验回放缓冲区 - 基础版本
    基于TD误差优先级采样
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # 优先级程度
        self.beta = beta  # 重要性采样权重
        self.beta_increment = beta_increment
        self.max_priority = 1.0  # 初始最大优先级

        # 经验存储
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

        debug(f"PriorityReplayBuffer initialized: capacity={capacity}")

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, done, priority=None):
        """
        添加经验到缓冲区
        """
        experience = Experience(state, action, reward, next_state, done)

        if priority is None:
            # 新经验获得当前最大优先级
            if self.size > 0:
                priority = np.max(self.priorities[:self.size])
            else:
                priority = self.max_priority

        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

        # 更新最大优先级
        if priority > self.max_priority:
            self.max_priority = priority

        debug(f"Experience added. Buffer size: {self.size}, Priority: {priority:.4f}")

    def sample(self, batch_size):
        """
        基于优先级采样批次
        """
        if self.size < batch_size:
            debug(f"Not enough experiences: {self.size} < {batch_size}")
            return None, None, None

        try:
            # 计算采样概率
            priorities = self.priorities[:self.size]
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()

            # 采样索引
            indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)

            # 计算重要性采样权重
            weights = (self.size * probabilities[indices]) ** (-self.beta)
            weights /= weights.max()  # 归一化

            # 更新beta
            self.beta = min(1.0, self.beta + self.beta_increment)

            # 获取批次数据
            batch = [self.buffer[idx] for idx in indices]

            debug(f"PER sampling: {batch_size} experiences, avg_weight: {np.mean(weights):.3f}")

            return batch, indices, weights

        except Exception as e:
            debug(f"Error in PER sampling: {e}")
            # 降级到均匀采样
            indices = np.random.choice(self.size, batch_size, replace=False)
            batch = [self.buffer[idx] for idx in indices]
            weights = np.ones(batch_size, dtype=np.float32)
            return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        """
        更新经验的TD误差优先级
        """
        try:
            # 添加小常数避免零优先级
            priorities = np.abs(td_errors) + 1e-6

            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority

                # 更新最大优先级
                if priority > self.max_priority:
                    self.max_priority = priority

            debug(f"Updated priorities for {len(indices)} experiences. Max: {self.max_priority:.4f}")

        except Exception as e:
            debug(f"Error updating priorities: {e}")

    def get_statistics(self):
        """
        获取缓冲区统计信息
        """
        if self.size == 0:
            return {
                'buffer_size': 0,
                'avg_priority': 0,
                'max_priority': self.max_priority
            }

        avg_priority = np.mean(self.priorities[:self.size])

        return {
            'buffer_size': self.size,
            'avg_priority': avg_priority,
            'max_priority': self.max_priority,
            'beta': self.beta
        }


# 全局PER缓冲区实例
global_per_buffer = None


def initialize_global_per(capacity=10000):
    """
    初始化全局PER缓冲区
    """
    global global_per_buffer
    global_per_buffer = PriorityReplayBuffer(capacity)
    return global_per_buffer