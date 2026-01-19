# -*- coding: utf-8 -*-
import numpy as np
import torch
from logger import debug, debug_print
import time  # <--- 确保这一行存在


def choose_action(dqn, action_space, device):
    # 确保状态是正确格式的tensor
    state_tensor = torch.tensor(dqn.curr_state).float().to(device)

    # 如果状态是1D，确保DQN能正确处理
    if state_tensor.dim() == 1:
        state_tensor = state_tensor.unsqueeze(0)  # 添加批次维度

    # 开始计时
    start_time = time.perf_counter()

    actions_tensor = dqn(state_tensor)

    end_time = time.perf_counter()
    dqn.last_decision_time = (end_time - start_time) * 1000  # 转换为毫秒

    # 如果输出是2D（批次），取第一个元素
    if actions_tensor.dim() == 2 and actions_tensor.size(0) == 1:
        actions_tensor = actions_tensor.squeeze(0)

    # if np.random.uniform() > dqn.epsilon:
    if np.random.uniform() < dqn.epsilon:
        debug(f"Random action for exploration")
        action_index = np.random.randint(0, len(action_space))
        dqn.action = action_space[action_index]
        dqn.q_estimate = actions_tensor[action_index]
    else:
        debug(f"Action chosen by DQN for exploitation")
        dqn.action = action_space[actions_tensor.argmax()]
        dqn.q_estimate = actions_tensor.max()


# --- MODIFIED: 在文件末尾添加这个新函数 ---

def choose_action_from_tensor(dqn, actions_tensor, action_space, device):
    """
    一个辅助函数，用于GNN模型。
    它接收已经计算好的Q值张量，并执行探索/利用。
    """
    # 确保 actions_tensor 是 1D
    if actions_tensor.dim() == 2 and actions_tensor.size(0) == 1:
        actions_tensor = actions_tensor.squeeze(0)

    # --- ADDED: 开始计时 (与 choose_action 保持一致) ---
    # (GNN 的 'decision' 已在 Main.py 中计时, 这里只记录探索开销)
    start_time_explore = time.time()

    if np.random.uniform() < dqn.epsilon:
        debug(f"GNN Random action for exploration")
        action_index = np.random.randint(0, len(action_space))
        dqn.action = action_space[action_index]
        dqn.q_estimate = actions_tensor[action_index]
    else:
        debug(f"GNN Action chosen by DQN for exploitation")
        dqn.action = action_space[actions_tensor.argmax()]
        dqn.q_estimate = actions_tensor.max()

    # --- ADDED: 结束计时 ---
    end_time_explore = time.time()
    # (累加探索时间到 GNN 的总决策时间上)
    dqn.last_decision_time += (end_time_explore - start_time_explore) * 1000