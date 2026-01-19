# -*- coding: utf-8 -*-
import numpy as np
import torch
from Parameters import RL_N_STATES_CSI, RL_TAU
from copy import deepcopy
from logger import debug, debug_print
from Parameters import (
    RL_ALPHA, RL_EPSILON, SCENE_SCALE_X, SCENE_SCALE_Y, VEHICLE_SPEED_M3S,
    CROSS_POSITION_LIST, DIRECTION_H_LEFT, DIRECTION_H_STEADY, DIRECTION_H_RIGHT,
    DIRECTION_V_UP, DIRECTION_V_STEADY, DIRECTION_V_DOWN, USE_UMI_NLOS_MODEL,VEHICLE_CAPACITY_PER_LANE
)


class BaseDQN(torch.nn.Module):
    """
    DQN 和 DuelingDQN 的共享基类。
    包含所有共享的属性、CSI状态更新 和 目标网络更新逻辑。
    """

    def __init__(self, dqn_id, start_x, start_y, end_x, end_y):
        super(BaseDQN, self).__init__()

        # --- 共享的DQN属性 ---
        self.dqn_id = dqn_id
        self.start = (start_x, start_y)
        self.end = (end_x, end_y)
        self.bs_loc = (min(start_x, end_x) + abs(start_x - end_x) / 2,
                       min(start_y, end_y) + abs(start_y - end_y) / 2)

        self.vehicle_exist_curr = False
        self.vehicle_exist_next = False
        self.curr_state = []
        self.next_state = []
        self.action = None
        self.reward = 0.0
        self.q_estimate = 0.0
        self.q_target = 0.0
        self.loss = 0.0
        self.loss_list = []
        self.epsilon = RL_EPSILON
        self.prev_loss = 0.0
        self.prev_snr = 0.0
        self.prev_delay = 0.0
        self.last_decision_time = 0.0  # DuelingDQN 和 ActionChooser 都使用

        self.csi_states_curr = []
        self.csi_states_next = []
        self.csi_states_history = []
        self.gnn_enhanced = False
        self.graph_features = None
        self.vehicle_in_dqn_range_by_distance = []
        self.delay_list = []
        self.snr_list = []
        self.vehicle_count_list = []

        # 目标网络相关
        self.target_network = None
        self.target_update_counter = 0

        # --- [NEW] 在这里显式添加这两个列表 ---
        self.v2v_success_list = []  # 显式初始化成功率列表
        self.feasible_list = []  # 新增：显式初始化物理可行性列表
        # ------------------------------------

        debug(f"BaseDQN {self.dqn_id} initialized from {self.start} to {self.end}")

    def update_csi_states(self, vehicles, is_current=True):
        """(共享) CSI状态更新"""
        if USE_UMI_NLOS_MODEL:
            from NewRewardCalculator import new_reward_calculator

            csi_states = []
            for vehicle in vehicles[:min(len(vehicles), VEHICLE_CAPACITY_PER_LANE)]:
                csi_state = new_reward_calculator.get_csi_for_state(vehicle, self)
                csi_states.extend(csi_state)

            target_length = RL_N_STATES_CSI
            if len(csi_states) < target_length:
                csi_states.extend([0.0] * (target_length - len(csi_states)))
            else:
                csi_states = csi_states[:target_length]

            if is_current:
                self.csi_states_curr = csi_states
            else:
                self.csi_states_next = csi_states

    def update_target_network(self):
        """
        执行软更新: theta_target = tau * theta_online + (1 - tau) * theta_target
        这比硬更新 (Hard Update) 更稳定，符合 DDPG/MADDPG/SAC 等现代 DRL 的标准。
        """
        if self.target_network is None:
            debug(f"Warning: Target network not initialized for DQN {self.dqn_id}")
            return

        try:
            # 获取参数字典
            target_state_dict = self.target_network.state_dict()
            online_state_dict = self.state_dict()

            # 1. 过滤掉 target_network 自身的参数（如果有嵌套，防御性编程）
            # 在某些实现中，target_network 可能是 online_network 的一个子属性，这会导致递归
            filtered_online_dict = {
                k: v for k, v in online_state_dict.items()
                if not k.startswith('target_network.')
            }

            # 2. 执行软更新
            for key in target_state_dict:
                if key in filtered_online_dict:
                    # Soft Update 公式
                    target_param = target_state_dict[key]
                    online_param = filtered_online_dict[key]

                    # In-place 更新以节省内存
                    target_state_dict[key] = RL_TAU * online_param + (1.0 - RL_TAU) * target_param

            # 3. 加载回目标网络
            self.target_network.load_state_dict(target_state_dict)

        except Exception as e:
            print(f"!!! Error during Soft Update for DQN {self.dqn_id}: {e}")

    def forward(self, x):
        """
        子类必须实现此方法。
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def __repr__(self):
        return (
            f"DQN {self.dqn_id} from {self.start} to {self.end}, bs_loc {self.bs_loc}"
        )


class DQN(BaseDQN):
    def __init__(self, n_states, n_hidden, n_actions, dqn_id, start_x, start_y, end_x, end_y):
        # 1. 初始化所有基类属性 (self.dqn_id, self.epsilon, etc.)
        super(DQN, self).__init__(dqn_id, start_x, start_y, end_x, end_y)

        # 2. 定义该类特有的网络层
        self.ln = torch.nn.LayerNorm(n_states)
        self.fc1 = torch.nn.Linear(n_states, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_actions)

        # 3. 定义优化器 (必须在定义网络层之后)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=RL_ALPHA)

        debug(f"Standard DQN {self.dqn_id} created.")

    def forward(self, x):
        # x = self.ln(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        actions_tensor = self.fc2(x)
        return actions_tensor


class DuelingDQN(BaseDQN):
    """
    双头DQN网络 (Dueling DQN)
    分离状态价值(Value)和动作优势(Advantage)
    """

    def __init__(self, n_states, n_hidden, n_actions, dqn_id, start_x, start_y, end_x, end_y):
        # 1. 初始化所有基类属性
        super(DuelingDQN, self).__init__(dqn_id, start_x, start_y, end_x, end_y)

        # 2. 定义该类特有的网络层
        # --- 共享特征层 ---
        self.feature_layer = torch.nn.Sequential(
            torch.nn.Linear(n_states, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden // 2),
            torch.nn.ReLU()
        )
        # --- 价值流 ---
        self.value_stream = torch.nn.Sequential(
            torch.nn.Linear(n_hidden // 2, n_hidden // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden // 4, 1)
        )
        # --- 优势流 ---
        self.advantage_stream = torch.nn.Sequential(
            torch.nn.Linear(n_hidden // 2, n_hidden // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden // 4, n_actions)
        )

        # 3. 定义优化器 (必须在定义网络层之后)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=RL_ALPHA)

        debug(f"DuelingDQN {self.dqn_id} created with value-advantage architecture")



    def forward(self, x):
        """
        双头DQN前向传播
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        """
        # 确保输入是tensor
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        # 处理单样本情况 - 添加批次维度
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [state_dim] -> [1, state_dim]

        # 共享特征提取
        features = self.feature_layer(x)

        # 价值流
        value = self.value_stream(features)  # [batch_size, 1]

        # 优势流
        advantages = self.advantage_stream(features)  # [batch_size, num_actions]

        # 组合Q值: Q = V + (A - mean(A))
        # 确保维度匹配
        if value.dim() == 1:
            value = value.unsqueeze(1)  # 确保value是2D

        # 计算优势的均值，保持正确维度
        advantages_mean = advantages.mean(dim=1, keepdim=True)  # [batch_size, 1]

        # 组合Q值
        q_values = value + (advantages - advantages_mean)  # [batch_size, num_actions]

        # 如果是单样本，返回1D张量以保持兼容性
        if q_values.size(0) == 1:
            q_values = q_values.squeeze(0)

        return q_values

    def get_value_advantage(self, x):
        """
        分别获取状态价值和动作优势（用于分析）
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        # 处理单样本情况
        if x.dim() == 1:
            x = x.unsqueeze(0)

        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # 返回时保持原始维度
        if value.size(0) == 1:
            value = value.squeeze(0)
            advantages = advantages.squeeze(0)

        return value, advantages

    def __repr__(self):
        return f"DuelingDQN {self.dqn_id} from {self.start} to {self.end}"


class Vehicle:
    def __init__(self, index, x, y, horizontal, vertical):
        self.first_occur = True
        self.id = index
        self.curr_loc = (x, y)
        self.curr_dir = (horizontal, vertical)

        debug(f"Vehicle {self.id} created at {self.curr_loc} with direction {self.curr_dir}")

        self.next_loc = (
            self.curr_loc[0] + self.curr_dir[0] * VEHICLE_SPEED_M3S,
            self.curr_loc[1] + self.curr_dir[1] * VEHICLE_SPEED_M3S,
        )
        self.distance_to_bs = None
        self.communication_metrics = {
            'snr_history': [],
            'delay_history': [],
            'throughput_history': []
        }

        self.power_W = 0.0  # 初始发射功率为 0
        self.tx_pos = self.curr_loc  # 初始发射位置为出生点

    def move(self, speed_m3s=VEHICLE_SPEED_M3S):
        self.first_occur = False
        flag_turned = False
        curr_loc_for_debug = deepcopy(self.curr_loc)  # 备份移动前的位置

        # --- 修复 1: 浮点数容差 ---
        PROXIMITY_TOLERANCE = 1.0  # 1.0 米的容差
        # --- 修复 1 结束 ---

        for cross_position in CROSS_POSITION_LIST:  # 判断交叉路口转向

            is_on_horizontal_road = abs(self.curr_loc[1] - cross_position[1]) < PROXIMITY_TOLERANCE
            is_on_vertical_road = abs(self.curr_loc[0] - cross_position[0]) < PROXIMITY_TOLERANCE

            if is_on_horizontal_road and not flag_turned:  # --- 修正: 增加 not flag_turned 避免重复转向 ---
                if self.curr_dir[0] == DIRECTION_H_RIGHT:  # 当前为向右移动
                    if (
                            self.curr_loc[0] < cross_position[0]
                            and abs(self.curr_loc[0] - cross_position[0])
                            <= speed_m3s
                    ):
                        flag_turned = True  # 标记已转向
                        if (
                                self.curr_loc[0] < SCENE_SCALE_X / 3
                        ):
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_V_UP, DIRECTION_V_DOWN]
                            )
                            if turn_direction == DIRECTION_V_UP:  # 左转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                            elif turn_direction == DIRECTION_V_DOWN:  # 右转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                        elif (
                                SCENE_SCALE_X / 3 < self.curr_loc[0] < 2 * SCENE_SCALE_X / 3
                        ):
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_V_UP, DIRECTION_V_STEADY, DIRECTION_V_DOWN]
                            )
                            if turn_direction == DIRECTION_V_UP:  # 左转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                            elif turn_direction == DIRECTION_V_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_RIGHT, DIRECTION_V_STEADY)
                            elif turn_direction == DIRECTION_V_DOWN:  # 右转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                        residue_distance = speed_m3s - abs(
                            self.curr_loc[0] - cross_position[0]
                        )
                        self.curr_loc = (
                            cross_position[0] + residue_distance * self.curr_dir[0],
                            cross_position[1] + residue_distance * self.curr_dir[1],
                        )
                elif self.curr_dir[0] == DIRECTION_H_LEFT:  # 当前为向左移动
                    if (
                            self.curr_loc[0] > cross_position[0]
                            and abs(self.curr_loc[0] - cross_position[0])
                            <= speed_m3s
                    ):
                        flag_turned = True

                        is_road_7 = abs(self.curr_loc[1] - 2 * SCENE_SCALE_Y / 3) < PROXIMITY_TOLERANCE
                        is_road_9 = abs(self.curr_loc[1] - SCENE_SCALE_Y / 3) < PROXIMITY_TOLERANCE

                        if (
                                SCENE_SCALE_X / 3 < self.curr_loc[0] < 2 * SCENE_SCALE_X / 3
                        ):
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_V_DOWN, DIRECTION_V_UP]
                            )
                            if turn_direction == DIRECTION_V_DOWN:  # 左转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                            elif turn_direction == DIRECTION_V_UP:  # 右转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                        elif (
                                is_road_7
                        ):
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_V_DOWN, DIRECTION_V_STEADY, DIRECTION_V_UP]
                            )
                            if turn_direction == DIRECTION_V_DOWN:  # 左转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                            elif turn_direction == DIRECTION_V_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_LEFT, DIRECTION_V_STEADY)
                            elif turn_direction == DIRECTION_V_UP:  # 右转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                        elif (
                                is_road_9
                        ):
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_V_DOWN, DIRECTION_V_UP]
                            )
                            if turn_direction == DIRECTION_V_DOWN:  # 左转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                            elif turn_direction == DIRECTION_V_UP:  # 右转
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                        residue_distance = speed_m3s - abs(
                            self.curr_loc[0] - cross_position[0]
                        )
                        self.curr_loc = (
                            cross_position[0] + residue_distance * self.curr_dir[0],
                            cross_position[1] + residue_distance * self.curr_dir[1],
                        )

            elif is_on_vertical_road and not flag_turned:  # --- 修正: 增加 not flag_turned 避免重复转向 ---
                if self.curr_dir[1] == DIRECTION_V_UP:  # 当前为向上移动
                    if (
                            self.curr_loc[1] < cross_position[1]
                            and abs(self.curr_loc[1] - cross_position[1])
                            <= speed_m3s
                    ):
                        flag_turned = True

                        is_road_2_or_3 = abs(self.curr_loc[0] - SCENE_SCALE_X / 3) < PROXIMITY_TOLERANCE
                        is_road_10_or_6 = abs(self.curr_loc[0] - 2 * SCENE_SCALE_X / 3) < PROXIMITY_TOLERANCE

                        if (
                                self.curr_loc[1] < SCENE_SCALE_Y / 3
                                and is_road_2_or_3
                        ):
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_H_LEFT, DIRECTION_H_STEADY]
                            )
                            if turn_direction == DIRECTION_H_LEFT:  # 左转
                                self.curr_dir = (DIRECTION_H_LEFT, DIRECTION_V_STEADY)
                            elif turn_direction == DIRECTION_H_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                        elif (
                                self.curr_loc[1] < SCENE_SCALE_Y / 3
                                and is_road_10_or_6
                        ):
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_H_STEADY, DIRECTION_H_RIGHT]
                            )
                            if turn_direction == DIRECTION_H_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                            elif turn_direction == DIRECTION_H_RIGHT:  # 右转
                                self.curr_dir = (DIRECTION_H_RIGHT, DIRECTION_V_STEADY)
                        elif (
                                SCENE_SCALE_X / 3 < self.curr_loc[1] < 2 * SCENE_SCALE_X / 3
                                and is_road_2_or_3
                        ):
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_H_STEADY, DIRECTION_H_RIGHT]
                            )
                            if turn_direction == DIRECTION_H_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                            elif turn_direction == DIRECTION_H_RIGHT:  # 右转
                                self.curr_dir = (DIRECTION_H_RIGHT, DIRECTION_V_STEADY)
                        elif (
                                SCENE_SCALE_X / 3 < self.curr_loc[1] < 2 * SCENE_SCALE_X / 3
                                and is_road_10_or_6
                        ):
                            turn_direction = np.random.default_rng().choice(
                                [
                                    DIRECTION_H_LEFT,
                                    DIRECTION_H_STEADY,
                                    DIRECTION_H_RIGHT,
                                ]
                            )
                            if turn_direction == DIRECTION_H_LEFT:  # 左转
                                self.curr_dir = (DIRECTION_H_LEFT, DIRECTION_V_STEADY)
                            elif turn_direction == DIRECTION_H_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_UP)
                            elif turn_direction == DIRECTION_H_RIGHT:  # 右转
                                self.curr_dir = (DIRECTION_H_RIGHT, DIRECTION_V_STEADY)
                        residue_distance = speed_m3s - abs(
                            self.curr_loc[1] - cross_position[1]
                        )
                        self.curr_loc = (
                            cross_position[0] + residue_distance * self.curr_dir[0],
                            cross_position[1] + residue_distance * self.curr_dir[1],
                        )
                elif self.curr_dir[1] == DIRECTION_V_DOWN:  # 当前为向下移动
                    if (
                            self.curr_loc[1] > cross_position[1]
                            and abs(self.curr_loc[1] - cross_position[1])
                            <= speed_m3s
                    ):
                        flag_turned = True

                        is_road_4 = abs(self.curr_loc[0] - SCENE_SCALE_X / 3) < PROXIMITY_TOLERANCE
                        is_road_8 = abs(self.curr_loc[0] - 2 * SCENE_SCALE_X / 3) < PROXIMITY_TOLERANCE

                        if (
                                self.curr_loc[1] > 2 * SCENE_SCALE_Y / 3
                                and is_road_4
                        ):
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_H_RIGHT, DIRECTION_H_STEADY]
                            )
                            if turn_direction == DIRECTION_H_RIGHT:  # 左转
                                self.curr_dir = (DIRECTION_H_RIGHT, DIRECTION_V_STEADY)
                            elif turn_direction == DIRECTION_H_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                        elif (
                                self.curr_loc[1] > 2 * SCENE_SCALE_Y / 3
                                and is_road_8
                        ):
                            turn_direction = np.random.default_rng().choice(
                                [
                                    DIRECTION_H_RIGHT,
                                    DIRECTION_H_STEADY,
                                    DIRECTION_H_LEFT,
                                ]
                            )
                            if turn_direction == DIRECTION_H_RIGHT:  # 左转
                                self.curr_dir = (DIRECTION_H_RIGHT, DIRECTION_V_STEADY)
                            elif turn_direction == DIRECTION_H_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                            elif turn_direction == DIRECTION_H_LEFT:  # 右转
                                self.curr_dir = (DIRECTION_H_LEFT, DIRECTION_V_STEADY)
                        elif (
                                SCENE_SCALE_X / 3 < self.curr_loc[1] < 2 * SCENE_SCALE_X / 3
                                and is_road_4
                        ):
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_H_STEADY, DIRECTION_H_LEFT]
                            )
                            if turn_direction == DIRECTION_H_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                            elif turn_direction == DIRECTION_H_LEFT:  # 右转
                                self.curr_dir = (DIRECTION_H_LEFT, DIRECTION_V_STEADY)
                        elif (
                                SCENE_SCALE_X / 3 < self.curr_loc[1] < 2 * SCENE_SCALE_X / 3
                                and is_road_8
                        ):
                            turn_direction = np.random.default_rng().choice(
                                [DIRECTION_H_RIGHT, DIRECTION_H_STEADY]
                            )
                            if turn_direction == DIRECTION_H_RIGHT:  # 左转
                                self.curr_dir = (DIRECTION_H_RIGHT, DIRECTION_V_STEADY)
                            elif turn_direction == DIRECTION_H_STEADY:  # 直行
                                self.curr_dir = (DIRECTION_H_STEADY, DIRECTION_V_DOWN)
                        residue_distance = speed_m3s - abs(
                            self.curr_loc[1] - cross_position[1]
                        )
                        self.curr_loc = (
                            cross_position[0] + residue_distance * self.curr_dir[0],
                            cross_position[1] + residue_distance * self.curr_dir[1],
                        )

        # --- 修复 2: 将 'if not flag_turned' 移到 'for' 循环之外 ---
        if not flag_turned:  # 如果未发生转向, 则直接基于速度更新位置
            self.curr_loc = (
                self.curr_loc[0] + self.curr_dir[0] * speed_m3s,
                self.curr_loc[1] + self.curr_dir[1] * speed_m3s,
            )

        self.next_loc = (
            self.curr_loc[0] + self.curr_dir[0] * speed_m3s,
            self.curr_loc[1] + self.curr_dir[1] * speed_m3s,
        )  # 基于速度计算的下一步位置

        debug(
            f"Vehicle {self.id} moved from {curr_loc_for_debug} to {self.curr_loc} at speed {speed_m3s:.2f} m/s")


    def record_communication_metrics(self, delay, snr, throughput=None):
        """安全记录通信指标"""
        if delay is not None and not np.isnan(delay) and delay > 0:
            self.communication_metrics['delay_history'].append(delay)
        else:
            self.communication_metrics['delay_history'].append(1.0)

        if snr is not None and not np.isnan(snr) and snr > 0 and not np.isinf(snr):
            self.communication_metrics['snr_history'].append(snr)
        else:
            self.communication_metrics['snr_history'].append(0.0)

        if throughput:
            self.communication_metrics['throughput_history'].append(throughput)