# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import time
import pandas as pd
from ActionChooser import choose_action, choose_action_from_tensor
from logger import global_logger, debug_print, debug, set_debug_mode
from Parameters import *
from Topology import formulate_global_list_dqn, vehicle_movement
from Classes import Vehicle
from Parameters import USE_PRIORITY_REPLAY, PER_BATCH_SIZE
from Parameters import TARGET_UPDATE_FREQUENCY
from Parameters import (
    N_V2I_LINKS, V2I_TX_POWER, V2I_LINK_POSITIONS, SYSTEM_BANDWIDTH,
    TRANSMITTDE_POWER, USE_UMI_NLOS_MODEL,
    RL_N_STATES_BASE, RL_N_STATES_CSI
)
from GraphBuilder import global_graph_builder
from GNNModel import (
    global_gnn_model, global_target_gnn_model,
    update_target_gnn, update_target_gnn_soft
)
from GNNReplayBuffer import GNNReplayBuffer
from Parameters import (
    GNN_REPLAY_CAPACITY, GNN_BATCH_SIZE,
    GNN_TRAIN_START_SIZE, GNN_SOFT_UPDATE_TAU
)
import torch.optim as optim
import Parameters
import argparse
import random


# ==============================================================================
# 辅助函数与模块初始化
# ==============================================================================

def move_graph_to_device(graph_data, device):
    """辅助函数：将图数据字典移动到指定设备"""
    try:
        graph_data['node_features']['features'] = graph_data['node_features']['features'].to(device)
        graph_data['node_features']['types'] = graph_data['node_features']['types'].to(device)

        for edge_type in global_gnn_model.edge_types:
            if graph_data['edge_features'][edge_type] is not None:
                graph_data['edge_features'][edge_type]['edge_index'] = \
                    graph_data['edge_features'][edge_type]['edge_index'].to(device)
                graph_data['edge_features'][edge_type]['edge_attr'] = \
                    graph_data['edge_features'][edge_type]['edge_attr'].to(device)
    except Exception as e:
        debug(f"Error moving graph to device: {e}")
    return graph_data


if USE_UMI_NLOS_MODEL:
    from ChannelModel import global_channel_model
    from NewRewardCalculator import new_reward_calculator

    debug_print("Main.py: Using NewRewardCalculator with UMi NLOS model")
else:
    debug_print("Main.py: Using original RewardCalculator")


def calculate_mean_metrics(dqn_list):
    """安全计算平均指标 (包含 P95 延迟)"""
    delays = []
    snrs = []
    v2v_successes = []
    v2v_delay_ok = []
    v2v_snr_ok = []
    # --- [NEW] 新增统计容器 ---
    feasible_success_counts = 0  # 可行且成功的次数
    total_feasible_counts = 0  # 物理可行的总次数
    # -------------------------

    debug("=== Calculating Mean Metrics ===")

    for dqn in dqn_list:
        if hasattr(dqn, 'delay_list') and dqn.delay_list:
            valid_delays = [d for d in dqn.delay_list
                            if d is not None and not np.isnan(d) and d > 0]
            if valid_delays:
                recent_delays = valid_delays[-min(20, len(valid_delays)):]
                delays.extend(recent_delays)

        if hasattr(dqn, 'snr_list') and dqn.snr_list:
            valid_snrs = [s for s in dqn.snr_list
                          if s is not None and not np.isnan(s) and not np.isinf(s)]
            if valid_snrs:
                recent_snrs = valid_snrs[-min(20, len(valid_snrs)):]
                snrs.extend(recent_snrs)

        if hasattr(dqn, 'v2v_success_list') and dqn.v2v_success_list:
            recent_successes = dqn.v2v_success_list[-min(20, len(dqn.v2v_success_list)):]
            v2v_successes.extend(recent_successes)

            # --- [NEW] 统计可行成功率 ---
            # 获取对应的 feasible_list
            if hasattr(dqn, 'feasible_list') and dqn.feasible_list:
                # 取出相同长度的最近数据
                limit = len(recent_successes)
                recent_feasible = dqn.feasible_list[-limit:]

                # 遍历配对数据
                for s, f in zip(recent_successes, recent_feasible):
                    if f == 1:  # 只有当物理可行时，才计入分母
                        total_feasible_counts += 1
                        if s == 1:  # 分子：可行且成功
                            feasible_success_counts += 1
            # ---------------------------

        if hasattr(dqn, 'v2v_delay_ok_list') and dqn.v2v_delay_ok_list:
            v2v_delay_ok.extend(dqn.v2v_delay_ok_list[-min(20, len(dqn.v2v_delay_ok_list)):])
        if hasattr(dqn, 'v2v_snr_ok_list') and dqn.v2v_snr_ok_list:
            v2v_snr_ok.extend(dqn.v2v_snr_ok_list[-min(20, len(dqn.v2v_snr_ok_list)):])

    mean_delay = np.mean(delays) if delays else 1.0
    p95_delay = np.percentile(delays, 95) if delays else 1.0

    mean_snr_linear = np.mean(snrs) if snrs else 1.0
    if mean_snr_linear > 0:
        mean_snr_db = 10 * np.log10(mean_snr_linear)
    else:
        mean_snr_db = -100

    v2v_success_rate = np.mean(v2v_successes) if v2v_successes else 0.0
    v2v_delay_only_rate = np.mean(v2v_delay_ok) if v2v_delay_ok else 0.0
    v2v_snr_only_rate = np.mean(v2v_snr_ok) if v2v_snr_ok else 0.0

    # --- [NEW] 计算最终的可行成功率 ---
    if total_feasible_counts > 0:
        feasible_v2v_success_rate = feasible_success_counts / total_feasible_counts
    else:
        # 如果没有任何可行链路，可以设为 0 或者与原始成功率一致（视具体定义，这里设为0更严谨）
        feasible_v2v_success_rate = 0.0

    debug(
        f"Metrics: Raw Success={v2v_success_rate:.3f}, Feasible Success={feasible_v2v_success_rate:.3f} (N={total_feasible_counts})")
    # 注意：返回值最后增加了一项
    return mean_delay, p95_delay, mean_snr_db, v2v_success_rate, v2v_delay_only_rate, v2v_snr_only_rate, feasible_v2v_success_rate


def initialize_enhanced_training():
    """初始化增强训练组件"""
    from PriorityReplayBuffer import initialize_global_per
    from Parameters import USE_PRIORITY_REPLAY, PER_CAPACITY

    if USE_PRIORITY_REPLAY:
        global_per_buffer = initialize_global_per(PER_CAPACITY)
        from logger import debug_print
        debug_print("Priority Experience Replay initialized")
        return global_per_buffer
    else:
        from logger import debug_print
        debug_print("Using standard experience replay")
        return None


def enhanced_training_step(dqn, per_buffer, device):
    """PER增强训练步骤 - 使用目标网络"""
    try:
        batch, indices, weights = per_buffer.sample(PER_BATCH_SIZE)
        if batch is None:
            traditional_training_step(dqn, device)
            return

        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(device)
        states = torch.FloatTensor(np.array([exp.state for exp in batch])).to(device)
        actions = torch.LongTensor([exp.action for exp in batch]).to(device)
        next_states = torch.FloatTensor(np.array([exp.next_state for exp in batch])).to(device)
        weights = torch.FloatTensor(weights).to(device)

        with torch.no_grad():
            next_q_values_online = dqn(next_states)
            best_action_indices = next_q_values_online.argmax(dim=1, keepdim=True)
            next_q_values_target = dqn.target_network(next_states)
            next_q_for_target = next_q_values_target.gather(1, best_action_indices).squeeze(1)
            target_q_values = rewards + RL_GAMMA * next_q_for_target

        current_q_values = dqn(states)
        current_action_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        td_errors = (target_q_values - current_action_q_values).abs().detach().cpu().numpy()
        dqn.loss = (weights * torch.nn.functional.mse_loss(current_action_q_values, target_q_values.detach(),
                                                           reduction='none')).mean()

        per_buffer.update_priorities(indices, td_errors)

        dqn.optimizer.zero_grad()
        dqn.loss.backward()
        torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=1.0)
        dqn.optimizer.step()

        if not FLAG_ADAPTIVE_EPSILON_ADJUSTMENT and dqn.epsilon > RL_EPSILON_MIN:
            dqn.epsilon *= RL_EPSILON_DECAY

    except Exception as e:
        debug(f"Error in enhanced training step (DDQN): {e}")
        traditional_training_step(dqn, device)


def traditional_training_step(dqn, device):
    """标准训练步骤"""
    try:
        curr_state_tensor = torch.tensor(dqn.curr_state).float().to(device)
        next_state_tensor = torch.tensor(dqn.next_state).float().to(device)

        if curr_state_tensor.dim() == 1: curr_state_tensor = curr_state_tensor.unsqueeze(0)
        if next_state_tensor.dim() == 1: next_state_tensor = next_state_tensor.unsqueeze(0)

        with torch.no_grad():
            next_q_values_online = dqn(next_state_tensor)
            if next_q_values_online.dim() == 2:
                best_action_indices = next_q_values_online.argmax(dim=1, keepdim=True)
            else:
                best_action_indices = next_q_values_online.argmax(dim=0, keepdim=True).unsqueeze(0)

            next_q_values_target = dqn.target_network(next_state_tensor)
            if next_q_values_target.dim() == 1: next_q_values_target = next_q_values_target.unsqueeze(0)
            next_q_for_target = next_q_values_target.gather(1, best_action_indices).squeeze()

            reward_tensor = torch.tensor(dqn.reward, dtype=torch.float32, device=device)
            dqn.q_target = reward_tensor + RL_GAMMA * next_q_for_target

        curr_q_values = dqn(curr_state_tensor)
        if curr_q_values.dim() == 1: curr_q_values = curr_q_values.unsqueeze(0)

        action_index = RL_ACTION_SPACE.index(dqn.action) if dqn.action in RL_ACTION_SPACE else 0
        action_index_tensor = torch.tensor([[action_index]], dtype=torch.long, device=device)
        dqn.q_estimate = curr_q_values.gather(1, action_index_tensor).squeeze()

        dqn.loss = torch.nn.functional.mse_loss(dqn.q_estimate, dqn.q_target.detach())

        dqn.optimizer.zero_grad()
        dqn.loss.backward()
        dqn.optimizer.step()

        if not FLAG_ADAPTIVE_EPSILON_ADJUSTMENT and dqn.epsilon > RL_EPSILON_MIN:
            dqn.epsilon *= RL_EPSILON_DECAY

    except Exception as e:
        debug(f"Error in traditional training step (DDQN): {e}")
        dqn.loss = torch.tensor(1.0, requires_grad=True, device=device)


# ==============================================================================
# 核心 RL 循环 (包含 物理状态同步修复)
# ==============================================================================

def rl(mean_loss_across_epochs=None, gnn_optimizer=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch = 1
    global_vehicle_id = 0
    overall_vehicle_list = []

    global_per_buffer = None
    global_gnn_buffer = None

    if USE_PRIORITY_REPLAY:
        global_per_buffer = initialize_enhanced_training()

    if USE_GNN_ENHANCEMENT:
        debug_print("Starting GNN-DRL training (Dueling-Double-DQN w/ GNN)")
        global_gnn_buffer = GNNReplayBuffer(capacity=GNN_REPLAY_CAPACITY)
    else:
        debug_print("Starting No-GNN training (Dueling-Double-DQN w/ PER)")
        if USE_PRIORITY_REPLAY:
            global_per_buffer = initialize_enhanced_training()

    for dqn in global_dqn_list:
        if not hasattr(dqn, 'delay_list'): dqn.delay_list = []
        if not hasattr(dqn, 'snr_list'): dqn.snr_list = []
        if not hasattr(dqn, 'v2v_success_list'): dqn.v2v_success_list = []
        if not hasattr(dqn, 'v2v_delay_ok_list'): dqn.v2v_delay_ok_list = []
        if not hasattr(dqn, 'v2v_snr_ok_list'): dqn.v2v_snr_ok_list = []
        if not hasattr(dqn, 'prev_v2i_interference'): dqn.prev_v2i_interference = 0.0

    graph_data_t = None
    max_epochs = Parameters.RL_N_EPOCHS if hasattr(Parameters, 'RL_N_EPOCHS') else 1500

    while epoch <= max_epochs:
        # 动态密度调度器 (Dynamic Density Scheduler)
        # 每 50 个 Epoch 随机切换一次密度(不要随机，次数太少）
        # if epoch % 50 == 0:
        #
        #     # 1. 随机选择一个新的密度
        #     new_target = np.random.choice(DENSITY_LEVELS)
        #
        #     # 2. 更新全局目标 (告诉环境我们要多少车)
        #     Parameters.TRAINING_VEHICLE_TARGET = new_target
        #
        #     print(f"\n" + "=" * 50)
        #     print(f"[Dynamic Density] Epoch {epoch}: Switching target to {new_target} Vehicles!")
        #     print("=" * 50 + "\n")
        #
        #     # 3. 【关键】强制裁剪多余车辆 (Pruning)
        #     # 如果从 100 辆切到 20 辆，必须立刻删掉 80 辆，否则模型会面对错误的密度
        #     if len(overall_vehicle_list) > new_target:
        #         # 随机保留 new_target 辆
        #         overall_vehicle_list = random.sample(overall_vehicle_list, new_target)
        #         print(f"   -> Pruned excess vehicles. Current count: {len(overall_vehicle_list)}")
        # 补充：为了防止初始车辆数 > 目标数（比如从N=100的检查点加载跑N=80），在每轮开始前做一个简单的检查即可
        if len(overall_vehicle_list) > Parameters.TRAINING_VEHICLE_TARGET:
            overall_vehicle_list = random.sample(overall_vehicle_list, Parameters.TRAINING_VEHICLE_TARGET)
        # 步骤 1: 车辆移动
        global_vehicle_id, overall_vehicle_list = vehicle_movement(
            global_vehicle_id,
            overall_vehicle_list,
            target_count=Parameters.TRAINING_VEHICLE_TARGET
        )

        loss_list_per_epoch = []
        mean_loss = 0.0
        cumulative_reward_per_epoch = 0.0
        v2i_sum_capacity_mbps = 0.0
        epoch_breakdown_stats = {'norm_snr': [], 'norm_delay': [], 'norm_v2i': [], 'norm_power': [], 'raw_v2i': [],
                                 'total_reward': []}

        if len(loss_list_per_epoch) > 0 and mean_loss_across_epochs is not None and len(mean_loss_across_epochs) > 10:
            debug_print(f"Epoch {epoch} Prev mean loss {mean_loss} Vehicle count {len(overall_vehicle_list)}")
        else:
            debug_print(f"Epoch {epoch}")

        # 干扰列表初始化
        active_v2v_interferers = []

        # 步骤 2: 构建图
        graph_data_t_plus_1 = None
        if USE_GNN_ENHANCEMENT:
            global_gnn_model.train()
            try:
                graph_data_t_plus_1 = global_graph_builder.build_dynamic_graph(global_dqn_list, overall_vehicle_list,
                                                                               epoch)
            except Exception as e:
                debug(f"GNN S_t+1 graph build/forward pass failed: {e}")

        # 步骤 3: GNN 训练
        if USE_GNN_ENHANCEMENT and global_gnn_buffer is not None:
            if epoch % 10 == 0:
                print(
                    f"[DEBUG Epoch {epoch}] Buffer Size: {len(global_gnn_buffer)} / Start Size: {GNN_TRAIN_START_SIZE}")

        if (USE_GNN_ENHANCEMENT and global_gnn_buffer is not None and len(global_gnn_buffer) >= GNN_TRAIN_START_SIZE):
            batch = global_gnn_buffer.sample(GNN_BATCH_SIZE, device)
            if batch is not None:
                total_gnn_loss = torch.tensor(0.0, device=device)
                agents_trained = 0
                entropy_loss = torch.tensor(0.0, device=device)

                for experience in batch:
                    graph_t_dev, actions_t, rewards_t, graph_t1_dev = experience
                    all_q_values_t, aux_info_t = global_gnn_model(graph_t_dev)

                    with torch.no_grad():
                        all_q_values_t1_online, _ = global_gnn_model(graph_t1_dev)
                        all_q_values_t1_target, _ = global_target_gnn_model(graph_t1_dev)

                    current_arch = getattr(Parameters, 'GNN_ARCH', 'HYBRID')
                    if current_arch == "HYBRID" and aux_info_t is not None:
                        P = F.softmax(aux_info_t, dim=0)
                        entropy = -torch.sum(P * torch.log(P + 1e-9))
                        entropy_loss += entropy

                    for dqn_id_str, action_index in actions_t.items():
                        try:
                            dqn_id = int(dqn_id_str)
                            dqn_id_index = dqn_id - 1
                            q_estimate = all_q_values_t[dqn_id_index, action_index]
                            with torch.no_grad():
                                reward = rewards_t[dqn_id_str]
                                best_action_t1 = torch.argmax(all_q_values_t1_online[dqn_id_index])
                                q_target_next = all_q_values_t1_target[dqn_id_index, best_action_t1]
                                q_target = reward + RL_GAMMA * q_target_next
                            loss = torch.nn.functional.mse_loss(q_estimate, q_target.detach())
                            total_gnn_loss += loss
                            agents_trained += 1
                            for dqn in global_dqn_list:
                                if dqn.dqn_id == dqn_id:
                                    dqn.loss = loss.item()
                                    loss_list_per_epoch.append(dqn.loss)
                                    break
                        except Exception:
                            pass

                if agents_trained > 0:
                    gnn_optimizer.zero_grad()
                    mean_batch_loss_td = total_gnn_loss / agents_trained
                    mean_entropy = entropy_loss / GNN_BATCH_SIZE
                    final_loss = mean_batch_loss_td - LAMBDA_ENTROPY * mean_entropy
                    final_loss.backward()
                    torch.nn.utils.clip_grad_norm_(global_gnn_model.parameters(), max_norm=1.0)
                    gnn_optimizer.step()
                    update_target_gnn_soft(GNN_SOFT_UPDATE_TAU)
                    for dqn in global_dqn_list:
                        if not FLAG_ADAPTIVE_EPSILON_ADJUSTMENT and dqn.epsilon > RL_EPSILON_MIN:
                            dqn.epsilon *= RL_EPSILON_DECAY

        # 在智能体决策前，强制所有车辆“静默”。只有稍后被服务的车辆会被赋予功率。
        for vehicle in overall_vehicle_list:
            vehicle.power_W = 0.0
            # 同时更新一下 tx_pos 为当前位置，确保位置也是最新的
            vehicle.tx_pos = vehicle.curr_loc


        # ==================================================================
        # 步骤 4: 动作选择 A_t+1 (【移除】此处原有的干扰列表构建代码)
        # ==================================================================
        current_actions_t = {}
        current_rewards_t = {}

        for dqn in global_dqn_list:
            dqn.vehicle_exist_curr = False
            base_state = []
            dqn.vehicle_in_dqn_range_by_distance = []

            for vehicle in overall_vehicle_list:
                if (dqn.start[0] <= vehicle.curr_loc[0] <= dqn.end[0] and
                        dqn.start[1] <= vehicle.curr_loc[1] <= dqn.end[1]):
                    dqn.vehicle_exist_curr = True
                    vehicle.distance_to_bs = new_reward_calculator.channel_model.calculate_3d_distance(
                        (dqn.bs_loc[0], dqn.bs_loc[1]), vehicle.curr_loc)
                    dqn.vehicle_in_dqn_range_by_distance.append(vehicle)

            dqn.vehicle_in_dqn_range_by_distance.sort(key=lambda x: x.distance_to_bs, reverse=False)

            if dqn.vehicle_exist_curr:
                # 状态构建
                iState = 0
                for iVehicle in range(min(RL_N_STATES_BASE // 4, len(dqn.vehicle_in_dqn_range_by_distance))):
                    base_state.append(dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_loc[0])
                    base_state.append(dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_loc[1])
                    base_state.append(dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_dir[0])
                    base_state.append(dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_dir[1])
                    iState += 4
                if len(base_state) < RL_N_STATES_BASE:
                    base_state.extend([0.0] * (RL_N_STATES_BASE - len(base_state)))
                else:
                    base_state = base_state[:RL_N_STATES_BASE]

                if USE_UMI_NLOS_MODEL and hasattr(dqn, 'update_csi_states'):
                    dqn.update_csi_states(dqn.vehicle_in_dqn_range_by_distance, is_current=True)

                interf_val = dqn.prev_v2i_interference
                interf_log = np.log10(interf_val + 1e-20)
                interf_norm = (interf_log + 20) / 14.0

                dir_x, dir_y = 0.0, 0.0
                if V2I_LINK_POSITIONS and dqn.vehicle_in_dqn_range_by_distance:
                    target_rx = V2I_LINK_POSITIONS[0]['rx']
                    curr_pos = dqn.vehicle_in_dqn_range_by_distance[0].curr_loc
                    dx, dy = target_rx[0] - curr_pos[0], target_rx[1] - curr_pos[1]
                    d = np.sqrt(dx ** 2 + dy ** 2) + 1e-9
                    dir_x, dir_y = dx / d, dy / d

                v2i_state = [interf_norm, dir_x, dir_y]
                dqn.curr_state = base_state + dqn.csi_states_curr + v2i_state

                # 动作选择
                if USE_GNN_ENHANCEMENT:
                    try:
                        global_gnn_model.eval()
                        graph_data_local = global_graph_builder.build_spatial_subgraph(dqn, global_dqn_list,
                                                                                       overall_vehicle_list, epoch)
                        graph_data_local = move_graph_to_device(graph_data_local, device)
                        with torch.no_grad():
                            actions_tensor, _ = global_gnn_model(graph_data_local, dqn_id=dqn.dqn_id)
                        global_gnn_model.train()
                        choose_action_from_tensor(dqn, actions_tensor, RL_ACTION_SPACE, device)
                    except Exception as e:
                        debug(f"!!! GNN action selection failed: {e}")
                        global_gnn_model.train()
                        choose_action(dqn, RL_ACTION_SPACE, device)
                else:
                    choose_action(dqn, RL_ACTION_SPACE, device)
            else:
                dqn.curr_state = [0.0] * RL_N_STATES
                dqn.action = None

        # ==================================================================
        # [修改] 步骤 4.5: 物理状态同步 (Phase 2 Sync)
        # ==================================================================
        # 1. 根据 Agent 的动作，更新“被服务车辆”的 power_W
        for dqn in global_dqn_list:
            # 只有当：1.车在范围内 且 2.智能体选择了动作 时，才计算功率
            if dqn.vehicle_exist_curr and dqn.action is not None:
                # 解析动作
                beam_count = dqn.action[0] + 1
                horizontal_dir = dqn.action[1]
                vertical_dir = dqn.action[2]
                power_ratio = (dqn.action[3] + 1) / 10.0

                # 计算增益 (调用 reward_calculator 的辅助函数)
                directional_gain = new_reward_calculator._calculate_directional_gain(horizontal_dir,
                                                                                             vertical_dir)

                # 计算总功率 (Watts)
                total_power_W = TRANSMITTDE_POWER * power_ratio * beam_count * directional_gain * Parameters.GAIN_ANTENNA_T

                # 赋值给对应的车辆
                if dqn.vehicle_in_dqn_range_by_distance:
                    serving_vehicle = dqn.vehicle_in_dqn_range_by_distance[0]
                    serving_vehicle.power_W = total_power_W
                    serving_vehicle.tx_pos = serving_vehicle.curr_loc  # 确保位置同步

        # 2. 构建【真实】的干扰列表 (用于后续 V2I 和 V2V 的干扰计算)
        # 这一步非常关键：只收集 power_W > 0 的车作为干扰源
        active_v2v_interferers = []
        for vehicle in overall_vehicle_list:
            if hasattr(vehicle, 'power_W') and vehicle.power_W > 0:
                active_v2v_interferers.append({
                    'tx_pos': vehicle.curr_loc,  # 使用真实物理位置
                    'power_W': vehicle.power_W  # 使用刚才计算出的真实功率
                })

        # ==================================================================
        # 步骤 5: V2I 容量计算
        # ==================================================================
        total_v2i_capacity_bps = 0.0
        if USE_UMI_NLOS_MODEL:
            for link in V2I_LINK_POSITIONS:
                v2i_tx_pos = link['tx']
                v2i_rx_pos = link['rx']
                v2i_dist = global_channel_model.calculate_3d_distance(v2i_tx_pos, v2i_rx_pos)
                _, _, v2i_signal_power_W = global_channel_model.calculate_snr(V2I_TX_POWER, v2i_dist,
                                                                              bandwidth=SYSTEM_BANDWIDTH)

                total_interference_W = 0.0
                for interferer in active_v2v_interferers:
                    interf_dist = global_channel_model.calculate_3d_distance(interferer['tx_pos'], v2i_rx_pos)
                    pl_db, _, _ = global_channel_model.calculate_path_loss(interf_dist)
                    pl_linear = 10 ** (-pl_db / 10)
                    total_interference_W += interferer['power_W'] * pl_linear

                noise_power_W = global_channel_model._calculate_noise_power(SYSTEM_BANDWIDTH)
                v2i_sinr_linear = v2i_signal_power_W / (total_interference_W + noise_power_W)
                total_v2i_capacity_bps += SYSTEM_BANDWIDTH * np.log2(1 + v2i_sinr_linear)
            v2i_sum_capacity_mbps = total_v2i_capacity_bps / 1e6

        # ==================================================================
        # 步骤 6: 奖励结算与经验存储
        # ==================================================================
        for dqn in global_dqn_list:
            dqn.vehicle_exist_next = False
            base_state_next = []

            # 检查 Next 车辆
            if not USE_GNN_ENHANCEMENT:
                dqn.vehicle_in_dqn_range_by_distance = []
                for vehicle in overall_vehicle_list:
                    if (dqn.start[0] <= vehicle.curr_loc[0] <= dqn.end[0] and dqn.start[1] <= vehicle.curr_loc[1] <=
                            dqn.end[1]):
                        dqn.vehicle_exist_next = True
                        vehicle.distance_to_bs = new_reward_calculator.channel_model.calculate_3d_distance(
                            (dqn.bs_loc[0], dqn.bs_loc[1]), vehicle.curr_loc)
                        dqn.vehicle_in_dqn_range_by_distance.append(vehicle)
                dqn.vehicle_in_dqn_range_by_distance.sort(key=lambda x: x.distance_to_bs, reverse=False)

            if dqn.vehicle_exist_curr:
                # 【关键】传入 active_v2v_interferers
                dqn.reward, breakdown = new_reward_calculator.calculate_complete_reward(
                    dqn, dqn.vehicle_in_dqn_range_by_distance, dqn.action, active_v2v_interferers
                )

                if breakdown:
                    for k, v in breakdown.items():
                        if k in epoch_breakdown_stats: epoch_breakdown_stats[k].append(v)
                cumulative_reward_per_epoch += dqn.reward

                if USE_GNN_ENHANCEMENT and dqn.action is not None:
                    current_actions_t[str(dqn.dqn_id)] = RL_ACTION_SPACE.index(dqn.action)
                    current_rewards_t[str(dqn.dqn_id)] = dqn.reward

                # Next State 构建
                if dqn.vehicle_exist_next or dqn.vehicle_exist_curr:
                    iState = 0
                    for iVehicle in range(min(RL_N_STATES_BASE // 4, len(dqn.vehicle_in_dqn_range_by_distance))):
                        base_state_next.append(dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_loc[0])
                        base_state_next.append(dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_loc[1])
                        base_state_next.append(dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_dir[0])
                        base_state_next.append(dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_dir[1])
                        iState += 4
                    if len(base_state_next) < RL_N_STATES_BASE:
                        base_state_next.extend([0.0] * (RL_N_STATES_BASE - len(base_state_next)))
                    else:
                        base_state_next = base_state_next[:RL_N_STATES_BASE]

                    if USE_UMI_NLOS_MODEL and hasattr(dqn, 'update_csi_states'):
                        dqn.update_csi_states(dqn.vehicle_in_dqn_range_by_distance, is_current=False)

                    # Next State V2I Interference (Use current power as estimate)
                    v2i_interf_next = 0.0
                    if dqn.vehicle_in_dqn_range_by_distance:
                        my_power = dqn.vehicle_in_dqn_range_by_distance[0].power_W
                        my_pos = dqn.vehicle_in_dqn_range_by_distance[0].curr_loc
                        for link in V2I_LINK_POSITIONS:
                            d = new_reward_calculator.channel_model.calculate_3d_distance(my_pos, link['rx'])
                            pl, _, _ = new_reward_calculator.channel_model.calculate_path_loss(d)
                            v2i_interf_next += my_power * (10 ** (-pl / 10))
                    dqn.prev_v2i_interference = v2i_interf_next

                    interf_log = np.log10(dqn.prev_v2i_interference + 1e-20)
                    interf_norm = (interf_log + 20) / 14.0

                    dir_x, dir_y = 0.0, 0.0
                    if V2I_LINK_POSITIONS and dqn.vehicle_in_dqn_range_by_distance:
                        target_rx = V2I_LINK_POSITIONS[0]['rx']
                        curr_pos = dqn.vehicle_in_dqn_range_by_distance[0].curr_loc
                        dx, dy = target_rx[0] - curr_pos[0], target_rx[1] - curr_pos[1]
                        d = np.sqrt(dx ** 2 + dy ** 2) + 1e-9
                        dir_x, dir_y = dx / d, dy / d

                    v2i_state = [interf_norm, dir_x, dir_y]
                    dqn.next_state = base_state_next + dqn.csi_states_next + v2i_state

                    if global_per_buffer is not None:
                        action_index = RL_ACTION_SPACE.index(dqn.action) if dqn.action in RL_ACTION_SPACE else 0
                        global_per_buffer.add(state=dqn.curr_state, action=action_index, reward=dqn.reward,
                                              next_state=dqn.next_state, done=False)

                    if global_per_buffer is not None and len(global_per_buffer) >= PER_BATCH_SIZE:
                        enhanced_training_step(dqn, global_per_buffer, device)
                    elif not USE_GNN_ENHANCEMENT:
                        traditional_training_step(dqn, device)

                    if hasattr(dqn, 'loss'):
                        # 如果是 Tensor，取 item()；如果是 float，直接用
                        if isinstance(dqn.loss, torch.Tensor):
                            loss_list_per_epoch.append(dqn.loss.item())
                        else:
                            loss_list_per_epoch.append(float(dqn.loss))

                    else:
                        # [修复] 统一使用 Tensor 以保持一致性，或者在读取时做兼容（上面的代码已经做了兼容）
                        # 这里为了保险，我们赋值为 Tensor，并带上 device
                        dqn.loss = torch.tensor(0.0, device=device)
                        dqn.reward = 0.0
                        if not USE_GNN_ENHANCEMENT:
                            new_reward_calculator._record_communication_metrics(dqn, 1.0, -100.0)

        # 步骤 7: GNN Buffer Add
        if USE_GNN_ENHANCEMENT and global_gnn_buffer is not None and graph_data_t is not None:
            if graph_data_t_plus_1 is not None and current_actions_t:
                global_gnn_buffer.add(graph_t=graph_data_t, actions_t=current_actions_t, rewards_t=current_rewards_t,
                                      graph_t1=graph_data_t_plus_1)
        if USE_GNN_ENHANCEMENT:
            graph_data_t = graph_data_t_plus_1

        # 步骤 8: 日志
        # --- [MODIFIED] 接收 feasible_v2v_success_rate
        mean_delay, p95_delay, mean_snr_db, v2v_success_rate, v2v_delay_only_rate, v2v_snr_only_rate, feasible_v2v_success_rate = calculate_mean_metrics(
            global_dqn_list)
        avg_breakdown = {k: (np.mean(v) if v else 0.0) for k, v in epoch_breakdown_stats.items()}

        debug_print(
            f"  [Reward Analysis] Norm Scores (0-1): SNR={avg_breakdown.get('norm_snr', 0):.3f}, Delay={avg_breakdown.get('norm_delay', 0):.3f}, V2I={avg_breakdown.get('norm_v2i', 0):.3f}, Power={avg_breakdown.get('norm_power', 0):.3f}")
        debug_print(f"  [Raw Metrics] V2I Penalty Power={avg_breakdown.get('raw_v2i', 0):.2e} W")

        if len(loss_list_per_epoch) > 0: mean_loss = np.mean(loss_list_per_epoch)
        # --- [MODIFIED] 传给 logger (我们马上会去修改 logger)
        global_logger.log_epoch(epoch, cumulative_reward_per_epoch, mean_loss, mean_delay, p95_delay, mean_snr_db,
                                len(overall_vehicle_list), v2v_success_rate, v2i_sum_capacity_mbps,
                                v2v_delay_only_rate, v2v_snr_only_rate,
                                feasible_v2v_success_rate=feasible_v2v_success_rate)  # 新增参数

        if epoch % TARGET_UPDATE_FREQUENCY == 0:
            if USE_GNN_ENHANCEMENT:
                update_target_gnn()
            else:
                for dqn in global_dqn_list: dqn.update_target_network()

        if epoch == max_epochs:
            global_logger.log_convergence(epoch, mean_loss)
            try:
                if USE_GNN_ENHANCEMENT:
                    torch.save(global_gnn_model.state_dict(), Parameters.MODEL_PATH_GNN)
                else:
                    path = MODEL_PATH_NO_GNN if USE_DUELING_DQN else MODEL_PATH_DQN
                    save_data = {f'dqn_{dqn.dqn_id}': dqn.state_dict() for dqn in global_dqn_list}
                    torch.save(save_data, path)
            except Exception:
                pass
            global_logger.save_metrics_to_csv()
            break

        epoch += 1
    global_logger.save_metrics_to_csv()
    return {'reward': cumulative_reward_per_epoch, 'v2v_success': v2v_success_rate,
            'v2i_capacity': v2i_sum_capacity_mbps, 'delay': mean_delay, 'snr': mean_snr_db}


def run_training(device):
    debug_print(f"--- Run Config: GNN={Parameters.USE_GNN_ENHANCEMENT}, Arch={Parameters.GNN_ARCH} ---")

    # 初始化 DQN 列表
    formulate_global_list_dqn(global_dqn_list, device)

    gnn_optimizer = None
    if Parameters.USE_GNN_ENHANCEMENT:
        import GNNModel
        GNNModel.global_gnn_model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64, num_heads=4,
                                                               num_layers=2, dropout=0.2).to(device)
        GNNModel.global_target_gnn_model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64, num_heads=4,
                                                                      num_layers=2, dropout=0.2).to(device)
        GNNModel.global_target_gnn_model.load_state_dict(GNNModel.global_gnn_model.state_dict())
        global global_gnn_model, global_target_gnn_model
        global_gnn_model = GNNModel.global_gnn_model
        global_target_gnn_model = GNNModel.global_target_gnn_model
        gnn_optimizer = optim.Adam(global_gnn_model.parameters(), lr=RL_ALPHA_GNN)

    return rl(gnn_optimizer=gnn_optimizer, device=device)

def test():
    debug_print("========== STARTING SCALABILITY TEST MODE (Strict Alignment) ==========")
    set_debug_mode(False)

    test_scenarios = {
        "GNN-DRL": {"model_path": MODEL_PATH_GNN, "use_gnn": True},
    }

    # 1. 初始化结果容器 (之前漏了这里)
    results = []

    for model_name, config in test_scenarios.items():
        debug_print(f"--- Testing Model: {model_name} ---")
        Parameters.USE_GNN_ENHANCEMENT = config["use_gnn"]
        Parameters.USE_DUELING_DQN = True
        is_gnn_model = Parameters.USE_GNN_ENHANCEMENT

        formulate_global_list_dqn(global_dqn_list, device)

        try:
            if is_gnn_model:
                global_gnn_model.load_state_dict(torch.load(config["model_path"], map_location=device))
                global_gnn_model.to(device)
                global_gnn_model.eval()
            else:
                checkpoint = torch.load(config["model_path"], map_location=device)
                for dqn in global_dqn_list:
                    dqn.load_state_dict(checkpoint[f'dqn_{dqn.dqn_id}'])
                    dqn.eval()
        except Exception as e:
            print(f"❌ Load failed: {e}")
            continue

        for vehicle_count in TEST_VEHICLE_COUNTS:
            debug_print(f"  Testing with {vehicle_count} vehicles...")

            # 清空统计容器
            episode_v2v_success_rates = []
            episode_feasible_rates = []
            episode_v2i_capacities = []

            global_vehicle_id = 0
            overall_vehicle_list = []

            # Reset DQN History
            for dqn in global_dqn_list:
                dqn.delay_list = []
                dqn.snr_list = []
                dqn.v2v_success_list = []
                dqn.prev_v2i_interference = 0.0
                dqn.prev_snr = 0.0

            # Warm up
            print(f"    >>> Warming up environment (2000 steps) to distribute vehicles...")
            for _ in range(2000):
                global_vehicle_id, overall_vehicle_list = vehicle_movement(
                    global_vehicle_id, overall_vehicle_list, target_count=vehicle_count
                )

            for i_episode in range(TEST_EPISODES_PER_COUNT):
                # [Step 1] Move
                global_vehicle_id, overall_vehicle_list = vehicle_movement(
                    global_vehicle_id, overall_vehicle_list, target_count=vehicle_count
                )

                # [Step 2] Power Reset
                for vehicle in overall_vehicle_list:
                    vehicle.power_W = 0.0
                    vehicle.tx_pos = vehicle.curr_loc

                # [Step 3] Action Selection (与 rl() 严格一致)
                for dqn in global_dqn_list:
                    dqn.vehicle_exist_curr = False
                    dqn.vehicle_in_dqn_range_by_distance = []

                    for vehicle in overall_vehicle_list:
                        if (dqn.start[0] <= vehicle.curr_loc[0] <= dqn.end[0] and
                                dqn.start[1] <= vehicle.curr_loc[1] <= dqn.end[1]):
                            dqn.vehicle_exist_curr = True
                            vehicle.distance_to_bs = new_reward_calculator.channel_model.calculate_3d_distance(
                                (dqn.bs_loc[0], dqn.bs_loc[1]), vehicle.curr_loc)
                            dqn.vehicle_in_dqn_range_by_distance.append(vehicle)

                    dqn.vehicle_in_dqn_range_by_distance.sort(key=lambda x: x.distance_to_bs)

                    if dqn.vehicle_exist_curr:
                        # State Construction
                        base_state = []
                        for iVehicle in range(min(RL_N_STATES_BASE // 4, len(dqn.vehicle_in_dqn_range_by_distance))):
                            v = dqn.vehicle_in_dqn_range_by_distance[iVehicle]
                            base_state.extend([v.curr_loc[0], v.curr_loc[1], v.curr_dir[0], v.curr_dir[1]])
                        if len(base_state) < RL_N_STATES_BASE:
                            base_state.extend([0.0] * (RL_N_STATES_BASE - len(base_state)))
                        else:
                            base_state = base_state[:RL_N_STATES_BASE]

                        if USE_UMI_NLOS_MODEL:
                            dqn.update_csi_states(dqn.vehicle_in_dqn_range_by_distance, is_current=True)

                        interf_log = np.log10(dqn.prev_v2i_interference + 1e-20)
                        interf_norm = (interf_log + 20) / 14.0

                        dir_x, dir_y = 0.0, 0.0
                        if V2I_LINK_POSITIONS and dqn.vehicle_in_dqn_range_by_distance:
                            target_rx = V2I_LINK_POSITIONS[0]['rx']
                            curr_pos = dqn.vehicle_in_dqn_range_by_distance[0].curr_loc
                            dx, dy = target_rx[0] - curr_pos[0], target_rx[1] - curr_pos[1]
                            d = np.sqrt(dx ** 2 + dy ** 2) + 1e-9
                            dir_x, dir_y = dx / d, dy / d

                        v2i_state = [interf_norm, dir_x, dir_y]
                        dqn.curr_state = base_state + dqn.csi_states_curr + v2i_state
                        dqn.epsilon = 0.0

                        # --- GNN Logic (Local Spatial Subgraph - Training Consistent) ---
                        if is_gnn_model:
                            try:
                                # 1. 使用 build_dynamic_graph (全局图)，与 ReplayBuffer 中的训练数据保持一致
                                # 注意：这里我们传入 i_episode 作为 epoch 参数
                                graph_data_global = global_graph_builder.build_dynamic_graph(
                                    global_dqn_list, overall_vehicle_list, i_episode
                                )
                                graph_data_global = move_graph_to_device(graph_data_global, device)

                                with torch.no_grad():
                                    # 2. 传入全局图进行推理
                                    actions_tensor, _ = global_gnn_model(graph_data_global, dqn_id=dqn.dqn_id)

                                choose_action_from_tensor(dqn, actions_tensor, RL_ACTION_SPACE, device)
                            except Exception as e:
                                print(f"GNN Inference Error: {e}")  # 打印错误以便调试
                                choose_action(dqn, RL_ACTION_SPACE, device)
                    else:
                        dqn.action = None

                # [Step 4] Physics Sync
                for dqn in global_dqn_list:
                    if dqn.vehicle_exist_curr and dqn.action is not None:
                        beam_count = dqn.action[0] + 1
                        h_dir, v_dir = dqn.action[1], dqn.action[2]
                        power_ratio = (dqn.action[3] + 1) / 10.0
                        dir_gain = new_reward_calculator._calculate_directional_gain(h_dir, v_dir)
                        total_power = Parameters.TRANSMITTDE_POWER * power_ratio * beam_count * dir_gain * Parameters.GAIN_ANTENNA_T

                        if dqn.vehicle_in_dqn_range_by_distance:
                            dqn.vehicle_in_dqn_range_by_distance[0].power_W = total_power
                            dqn.vehicle_in_dqn_range_by_distance[0].tx_pos = dqn.vehicle_in_dqn_range_by_distance[
                                0].curr_loc

                # [Step 5] Interferers
                active_v2v_interferers = []
                for vehicle in overall_vehicle_list:
                    if vehicle.power_W > 0:
                        active_v2v_interferers.append({'tx_pos': vehicle.curr_loc, 'power_W': vehicle.power_W})

                # [Step 6] Reward & Update (关键修正区)
                # 计算 V2I (每帧都算)
                total_v2i_capacity_bps = 0.0
                if USE_UMI_NLOS_MODEL:
                    for link in V2I_LINK_POSITIONS:
                        v2i_sig = global_channel_model.calculate_snr(V2I_TX_POWER,
                                                                     global_channel_model.calculate_3d_distance(
                                                                         link['tx'], link['rx']),
                                                                     bandwidth=SYSTEM_BANDWIDTH)[2]
                        total_interf = sum([interf['power_W'] * (10 ** (-global_channel_model.calculate_path_loss(
                            global_channel_model.calculate_3d_distance(interf['tx_pos'], link['rx']))[0] / 10)) for
                                            interf in active_v2v_interferers])
                        total_v2i_capacity_bps += SYSTEM_BANDWIDTH * np.log2(1 + v2i_sig / (
                                    total_interf + global_channel_model._calculate_noise_power(SYSTEM_BANDWIDTH)))
                episode_v2i_capacities.append(total_v2i_capacity_bps / 1e6)

                # 计算 V2V
                for dqn in global_dqn_list:
                    if dqn.vehicle_exist_curr:
                        # 1. 只有车存在时，才计算并记录指标！
                        new_reward_calculator.calculate_complete_reward(
                            dqn, dqn.vehicle_in_dqn_range_by_distance, dqn.action, active_v2v_interferers
                        )

                        # 2. 更新 State (为下一帧准备)
                        v2i_interf_next = 0.0
                        if dqn.vehicle_in_dqn_range_by_distance and dqn.vehicle_in_dqn_range_by_distance[0].power_W > 0:
                            my_pos = dqn.vehicle_in_dqn_range_by_distance[0].curr_loc
                            my_pwr = dqn.vehicle_in_dqn_range_by_distance[0].power_W
                            for link in V2I_LINK_POSITIONS:
                                pl = global_channel_model.calculate_path_loss(
                                    global_channel_model.calculate_3d_distance(my_pos, link['rx']))[0]
                                v2i_interf_next += my_pwr * (10 ** (-pl / 10))
                        dqn.prev_v2i_interference = v2i_interf_next
                    else:
                        # ⚠️⚠️⚠️【关键修改】⚠️⚠️⚠️
                        # 彻底删除原来的 else 分支中的记录代码。
                        # 不要在这里调用 record_communication_metrics。
                        # 仅仅重置干扰状态即可。
                        dqn.prev_v2i_interference = 0.0

            # Statistics (计算平均值时，现在只基于真实发生的通信事件)
            _, _, _, v2v_success_rate, _, _, feasible_rate = calculate_mean_metrics(global_dqn_list)

            if i_episode > 10:
                episode_v2v_success_rates.append(v2v_success_rate)
                episode_feasible_rates.append(feasible_rate)
                episode_v2i_capacities.append(total_v2i_capacity_bps / 1e6)
            print(f"    -> Count {vehicle_count}: Success={v2v_success_rate:.2%}, Feasible={feasible_rate:.2%}")

            results.append({
                "model": model_name, "vehicle_count": vehicle_count,
                "v2v_success_rate": np.mean(episode_v2v_success_rates),
                "feasible_v2v_success_rate": np.mean(episode_feasible_rates),
                "v2i_sum_capacity_mbps": np.mean(episode_v2i_capacities),
            })

    # 保存
    df = pd.DataFrame(results)
    df.to_csv(f"{global_logger.log_dir}/scalability{Parameters.ABLATION_SUFFIX}.csv", index=False)
    print("Test completed.")

if __name__ == "__main__":
    # 1. 基础设置
    set_debug_mode(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_print(f"Current device is: {device}")

    # 2. 参数定义
    parser = argparse.ArgumentParser(description="V2V/V2I DRL Training Script")

    # --- 基础训练参数 ---
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    parser.add_argument("--epochs", type=int, default=1500, help="Number of training epochs")
    parser.add_argument("--run_mode", type=str, default="TRAIN", choices=["TRAIN", "TEST"], help="Execution mode")

    # --- 环境/物理参数 (Multipliers) ---
    parser.add_argument("--snr_mul", type=float, default=1.0, help="SNR Multiplier (Interference)")
    parser.add_argument("--v2i_mul", type=float, default=1.0, help="V2I Weight Multiplier")
    parser.add_argument("--delay_mul", type=float, default=1.0, help="Delay Weight Multiplier")
    parser.add_argument("--power_mul", type=float, default=1.0, help="Power Weight Multiplier")

    # --- 实验关键参数 (Scalability & Ablation) ---
    # [修改点 1] 车辆密度 (Scalability)
    parser.add_argument('--vehicle_count', type=int, default=None,
                        help='Override vehicle count for scalability test (e.g., 20, 40, 60)')

    # [修改点 2] 模型架构开关 (Ablation)
    # 使用字符串解析，防止 bool 类型转换的坑 (命令行传 "False" 会被解析为 True)
    parser.add_argument("--use_gnn", type=str, default="True", choices=["True", "False"],
                        help="Enable/Disable GNN module")

    parser.add_argument('--dueling', type=str, default="True", choices=["True", "False"],
                        help='Enable Dueling DQN mechanism')

    parser.add_argument('--gnn_arch', type=str, default="HYBRID", choices=["HYBRID", "GAT", "GCN"],
                        help='GNN Architecture Type')

    # 解析参数
    args, unknown = parser.parse_known_args()

    # ==========================================
    # 3. 参数应用与覆盖 (Parameter Overrides)
    # ==========================================

    # 3.1 基础参数映射
    Parameters.RANDOM_SEED = args.seed
    Parameters.SNR_MULTIPLIER = args.snr_mul
    Parameters.V2I_MULTIPLIER = args.v2i_mul
    Parameters.DELAY_MULTIPLIER = args.delay_mul
    Parameters.POWER_MULTIPLIER = args.power_mul
    Parameters.RUN_MODE = args.run_mode
    Parameters.GNN_ARCH = args.gnn_arch

    # 更新文件后缀，防止结果覆盖
    Parameters.ABLATION_SUFFIX = f"_Veh{args.vehicle_count if args.vehicle_count else 'Def'}_{args.gnn_arch}"
    Parameters.MODEL_PATH_GNN = f"model_{args.gnn_arch}.pt"

    # 3.2 [关键] 覆盖车辆数量 (Scalability Experiment)
    if args.vehicle_count is not None:
        print(f"!!! OVERRIDING VEHICLE COUNT: {args.vehicle_count} !!!")

        # 1. 训练目标强制设为 80
        Parameters.TRAINING_VEHICLE_TARGET = args.vehicle_count
        Parameters.NUM_VEHICLES = args.vehicle_count
        Parameters.ROBUSTNESS_FIXED_VEHICLE_COUNT = args.vehicle_count

        # 2. 测试列表强制锁定为 80
        if hasattr(Parameters, 'TEST_VEHICLE_COUNTS'):
            Parameters.TEST_VEHICLE_COUNTS = [args.vehicle_count]

    # 3.3 [关键] 处理 Boolean 类型的字符串转换
    use_gnn_flag = (args.use_gnn.lower() == "true")

    # 处理 Dueling DQN 开关
    if args.dueling.lower() == "false":
        Parameters.USE_DUELING_DQN = False
    else:
        Parameters.USE_DUELING_DQN = True

    # 3.4 [核心修复] 覆盖所有可能的 Epoch 变量名
    if args.epochs:
        print(f"\n[System] !!! OVERRIDING MAX EPOCHS: {args.epochs} (via Command Line) !!!\n")
        # 把所有可能用到的变量名全改了，宁可杀错不可放过
        Parameters.MAX_EPOCHS = args.epochs
        Parameters.TRAINING_EPOCHS = args.epochs
        Parameters.RL_N_EPOCHS = args.epochs

    # 打印最终配置以供检查
    print("=" * 30)
    print(f"RUN CONFIGURATION:")
    print(f"  > Mode: {Parameters.RUN_MODE}")
    print(f"  > GNN Enabled: {use_gnn_flag}")
    print(f"  > GNN Arch: {Parameters.GNN_ARCH}")
    print(f"  > Dueling DQN: {Parameters.USE_DUELING_DQN}")
    print(f"  > Vehicle Count: {getattr(Parameters, 'NUM_VEHICLES', 'Default/Test Loop')}")
    print(f"  > SNR Multiplier: {Parameters.SNR_MULTIPLIER}")
    print("=" * 30)

    # 4. 设置随机种子
    random.seed(Parameters.RANDOM_SEED)
    np.random.seed(Parameters.RANDOM_SEED)
    torch.manual_seed(Parameters.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Parameters.RANDOM_SEED)

    # 5. 执行主逻辑
    if Parameters.RUN_MODE == "TRAIN":
        run_training(device)
    elif Parameters.RUN_MODE == "TEST":
        test()