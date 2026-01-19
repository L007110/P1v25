# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
from Parameters import (
    RL_N_STATES, RL_N_HIDDEN, RL_N_ACTIONS, SCENE_SCALE_X, SCENE_SCALE_Y,
    VEHICLE_SAFETY_DISTANCE, DIRECTION_H_RIGHT, DIRECTION_H_STEADY, DIRECTION_H_LEFT,
    DIRECTION_V_UP, DIRECTION_V_STEADY, DIRECTION_V_DOWN, BOUNDARY_POSITION_LIST,
    VEHICLE_OCCUR_PROB, USE_UMI_NLOS_MODEL, ANTENNA_HEIGHT_BS, VEHICLE_SPEED_KMH,
    TRAINING_VEHICLE_TARGET
)
import Parameters
#from Classes import Vehicle # 只导入 Vehicle
from logger import debug, debug_print


def formulate_global_list_dqn(dqn_list, device):
    """
    创建全局DQN列表 - 支持双头DQN和传统DQN，并正确初始化目标网络
    """
    # <<< 在函数内部导入 DQN 类，避免循环导入问题 >>>
    from Classes import DQN, DuelingDQN

    # --- 新增：在这里根据 Parameters 的 *当前* 状态动态计算 RL_N_HIDDEN ---
    local_rl_n_hidden = 0
    if Parameters.USE_DUELING_DQN:
        DQNClass = DuelingDQN
        local_rl_n_hidden = RL_N_ACTIONS * 3  # 双头DQN的隐藏层大小
        debug_print("Creating Dueling DQN instances with value-advantage architecture...")
    else:
        DQNClass = DQN
        local_rl_n_hidden = RL_N_ACTIONS * 2  # 标准DQN的隐藏层大小
        debug_print("Creating traditional DQN instances...")

    # --- 确保 local_rl_n_hidden 被设置 ---
    if local_rl_n_hidden == 0:
        debug_print("!!! 错误: local_rl_n_hidden 未被设置!")
        return

    dqn_list.clear()

    # --- 使用循环创建 DQN 实例 ---
    coords = {
        1: (0, SCENE_SCALE_Y / 3, SCENE_SCALE_X / 3, SCENE_SCALE_Y / 3),
        2: (SCENE_SCALE_X / 3, 0, SCENE_SCALE_X / 3, SCENE_SCALE_Y / 3),
        3: (SCENE_SCALE_X / 3, SCENE_SCALE_Y / 3, SCENE_SCALE_X / 3, 2 * SCENE_SCALE_Y / 3),
        4: (SCENE_SCALE_X / 3, 2 * SCENE_SCALE_Y / 3, SCENE_SCALE_X / 3, SCENE_SCALE_Y),
        5: (SCENE_SCALE_X / 3, 2 * SCENE_SCALE_Y / 3, 2 * SCENE_SCALE_X / 3, 2 * SCENE_SCALE_Y / 3),
        6: (2 * SCENE_SCALE_X / 3, SCENE_SCALE_Y / 3, 2 * SCENE_SCALE_X / 3, 2 * SCENE_SCALE_Y / 3),
        7: (2 * SCENE_SCALE_X / 3, 2 * SCENE_SCALE_Y / 3, SCENE_SCALE_X, 2 * SCENE_SCALE_Y / 3),
        8: (2 * SCENE_SCALE_X / 3, 2 * SCENE_SCALE_Y / 3, 2 * SCENE_SCALE_X / 3, SCENE_SCALE_Y),
        9: (2 * SCENE_SCALE_X / 3, SCENE_SCALE_Y / 3, SCENE_SCALE_X, SCENE_SCALE_Y / 3),
        10: (2 * SCENE_SCALE_X / 3, 0, 2 * SCENE_SCALE_X / 3, SCENE_SCALE_Y / 3),
    }

    for i in range(1, 11):
        start_x, start_y, end_x, end_y = coords[i]

        # 1. 创建在线网络 (使用 local_rl_n_hidden)
        dqn_eval = DQNClass(
            RL_N_STATES, local_rl_n_hidden, RL_N_ACTIONS, dqn_id=i,
            start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y,
        ).to(device)

        # 2. 创建目标网络 (结构相同，但不参与训练)
        dqn_target = DQNClass(
             RL_N_STATES, local_rl_n_hidden, RL_N_ACTIONS, dqn_id=i,
             start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y,
        ).to(device)

        # 3. 将在线网络的初始权重复制到目标网络
        dqn_target.load_state_dict(dqn_eval.state_dict())
        dqn_target.eval() # <<< 设置目标网络为评估模式，禁用 dropout 等

        # 4. 将目标网络赋给在线网络的属性 (普通属性赋值，不会注册为子模块)
        dqn_eval.target_network = dqn_target

        dqn_list.append(dqn_eval) # 只将在线网络添加到全局列表

    debug_print(f"Successfully created {len(dqn_list)} {DQNClass.__name__} instances with target networks")

    # 显示网络架构信息
    if dqn_list:
        sample_dqn = dqn_list[0]
        debug_print(f"Network architecture: {type(sample_dqn).__name__}")
        debug_print(f"Input dim: {RL_N_STATES}, Hidden dim: {local_rl_n_hidden}, Output dim: {RL_N_ACTIONS}")
        if Parameters.USE_DUELING_DQN:
            debug_print("Value-Advantage streams enabled")


def vehicle_movement(vehicle_id, vehicle_list, target_count=None, speed_kmh=VEHICLE_SPEED_KMH):
    """
    车辆移动更新 - 修复版 (v2)
    - 训练和测试都使用 target_count 来维持车辆密度。
    - 修复了内部路段的出生点方向。
    """
    from Classes import Vehicle

    # --- ADDED: 根据传入参数计算 m/s 速度 ---
    speed_m3s = speed_kmh * 1000 / 3600
    debug(f"Vehicle movement step with speed: {speed_kmh} km/h ({speed_m3s:.2f} m/s)")
    # --- ADDED END ---

    # 1. 更新当前车辆位置 (直接移动，不再检查碰撞)
    if len(vehicle_list) > 0:
        debug(f"Moving {len(vehicle_list)} vehicles...")
        for vehicle in vehicle_list:
            vehicle.move(speed_m3s)
        debug(f"Vehicle location update succeed.")

        # 2. 将超出边界的车辆移除
        original_count = len(vehicle_list)
        vehicle_list = [v for v in vehicle_list if
                        0 <= v.curr_loc[0] <= SCENE_SCALE_X and
                        0 <= v.curr_loc[1] <= SCENE_SCALE_Y]
        removed_count = original_count - len(vehicle_list)
        if removed_count > 0:
            debug(f"Removed {removed_count} vehicles out of boundary.")

    # 3. 车辆生成逻辑

    # --- MODIFIED: 统一训练和测试的车辆生成逻辑 ---

    # 确定目标车辆数 (如果未提供, 使用训练目标)
    effective_target_count = target_count if target_count is not None else TRAINING_VEHICLE_TARGET

    current_count = len(vehicle_list)
    if current_count < effective_target_count:
        # 随机决定这一轮要不要生车 (避免每轮都生)
        if np.random.uniform() <= VEHICLE_OCCUR_PROB:
            # 计算需要多少车
            needed = effective_target_count - current_count
            # 随机决定生几辆 (最多生 5 辆)
            num_to_spawn = np.random.randint(1, min(needed, 5) + 1)

            for _ in range(num_to_spawn):
                boundary_position = BOUNDARY_POSITION_LIST[np.random.randint(0, len(BOUNDARY_POSITION_LIST))]
                horizontal, vertical = 0, 0

                # --- MODIFIED: 包含所有出生点的转向逻辑 ---
                if boundary_position == (0, SCENE_SCALE_Y / 3):  # 1
                    horizontal, vertical = DIRECTION_H_RIGHT, DIRECTION_V_STEADY
                elif boundary_position == (SCENE_SCALE_X / 3, 0):  # 2
                    horizontal, vertical = DIRECTION_H_STEADY, DIRECTION_V_UP
                elif boundary_position == (SCENE_SCALE_X / 3, SCENE_SCALE_Y):  # 4
                    horizontal, vertical = DIRECTION_H_STEADY, DIRECTION_V_DOWN
                elif boundary_position == (SCENE_SCALE_X, 2 * SCENE_SCALE_Y / 3):  # 7
                    horizontal, vertical = DIRECTION_H_LEFT, DIRECTION_V_STEADY
                elif boundary_position == (2 * SCENE_SCALE_X / 3, SCENE_SCALE_Y):  # 8
                    horizontal, vertical = DIRECTION_H_STEADY, DIRECTION_V_DOWN
                elif boundary_position == (SCENE_SCALE_X, SCENE_SCALE_Y / 3):  # 9
                    horizontal, vertical = DIRECTION_H_LEFT, DIRECTION_V_STEADY
                elif boundary_position == (2 * SCENE_SCALE_X / 3, 0):  # 10
                    horizontal, vertical = DIRECTION_H_STEADY, DIRECTION_V_UP

                # (新加的出生点)
                elif boundary_position == (SCENE_SCALE_X / 3, SCENE_SCALE_Y / 3 + 1):  # 进入 DQN 3
                    horizontal, vertical = DIRECTION_H_STEADY, DIRECTION_V_UP
                elif boundary_position == (SCENE_SCALE_X / 3 + 1, 2 * SCENE_SCALE_Y / 3):  # 进入 DQN 5
                    horizontal, vertical = DIRECTION_H_RIGHT, DIRECTION_V_STEADY
                elif boundary_position == (2 * SCENE_SCALE_X / 3, SCENE_SCALE_Y / 3 + 1):  # 进入 DQN 6
                    horizontal, vertical = DIRECTION_H_STEADY, DIRECTION_V_UP

                else:
                    debug_print(f"Error: invalid boundary_position {boundary_position}")
                    quit()

                vehicle_id += 1
                vehicle = Vehicle(
                    vehicle_id,
                    boundary_position[0],
                    boundary_position[1],
                    horizontal,
                    vertical,
                )
                vehicle.next_loc = (
                    vehicle.curr_loc[0] + vehicle.curr_dir[0] * speed_m3s,
                    vehicle.curr_loc[1] + vehicle.curr_dir[1] * speed_m3s,
                )
                vehicle_list.append(vehicle)
            debug(f"Spawned {num_to_spawn} vehicles. Total: {len(vehicle_list)}")

    return vehicle_id, vehicle_list