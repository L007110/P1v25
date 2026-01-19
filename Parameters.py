# -*- coding: utf-8 -*-
import torch
import numpy as np
import itertools

from sympy import false

from logger import debug, debug_print

# 泛化性/鲁棒性测试配置
# 1. 鲁棒性 vs. 速度
TEST_SPEEDS_KMH = [30, 45, 60, 75, 90, 105, 120]
# 2. 鲁棒性 vs. 负载
# TEST_PAYLOADS_BYTES = [1060, 2*1060, 4*1060, 6*1060]
# 3. 运行这些测试时的固定车辆数
ROBUSTNESS_FIXED_VEHICLE_COUNT = 60 # 选择一个有代表性的密度
# 4. 每个速度跑多少轮
ROBUSTNESS_EPISODES_PER_SETTING = 100

RUN_MODE = "TRAIN"  # 设为 "TRAIN" 进行训练
# RUN_MODE = "TEST"     # 设为 "TEST" 进行可扩展性测试

USE_DUELING_DQN = True # 启用双头架构
# === Model Architecture Ablation ===
# "HYBRID" : P1v18 当前的最优模型 (GAT + Gating + Entropy)
# "GAT"    : 标准 GAT (无 Gating，无 Entropy 正则化) -> 测试稳定性
# "GCN"    : 标准 GCN (无注意力机制) -> 测试特征提取上限
GNN_ARCH = "HYBRID"

# GNN
USE_GNN_ENHANCEMENT = True # GNN增强开关
if USE_GNN_ENHANCEMENT:
    USE_PRIORITY_REPLAY = False # GNN 模式下禁用
    debug_print("GNN 模式激活：将使用 GNN 经验回放缓冲区 (GNNReplayBuffer) 训练。")
else:
    USE_PRIORITY_REPLAY = False  # No-GNN 基线模式下启用优先级经验回放(目前先关掉，变成消融实验）
GNN_REPLAY_CAPACITY = 2000   # GNN 经验缓冲区的容量 (图很占内存, 设小一点)
GNN_BATCH_SIZE = 64         # GNN 训练的批次大小
GNN_TRAIN_START_SIZE = 100   # 缓冲区中至少有多少经验才开始训练 GNN
GNN_SOFT_UPDATE_TAU = 0.005  # GNN 目标网络软更新的 TAU
GNN_OUTPUT_DIM = 64         # GNN输出维度
ATTENTION_HEADS = 8        # 注意力头数
# 定义在测试/推理时，GNN 构建子图的空间半径 (米)
# 500米 意味着它会考虑自己和周围约 500米 内的车辆和RSU
GNN_INFERENCE_RADIUS = 500.0

# 测试用的车辆数量列表
TEST_VEHICLE_COUNTS = [20, 40, 60, 80, 100, 120]
# TEST_VEHICLE_COUNTS = [80]
# 每个车辆数测试多少个 Epochs
TEST_EPISODES_PER_COUNT = 100

# 1. 归一化滑动窗口 (样本数)
# 【修改说明】虽然 NewRewardCalculator 改用了固定边界，保留此参数以兼容旧代码或其他用途
REWARD_RUNNING_WINDOW = 2000

# 2. 奖励乘数 (Grid Search Multipliers) - 默认值设为 1.0 (基准)
SNR_MULTIPLIER = 0.5
V2I_MULTIPLIER = 0.1
DELAY_MULTIPLIER = 1.0
POWER_MULTIPLIER = 1.0

# 3. 3GPP QoS Requirement (新增)
# Reference: 3GPP TS 22.186 (Service requirements for enhanced V2X scenarios)
# Target: High Data Rate Service (e.g., Sensor Sharing) -> Spectral Efficiency target
V2I_CAPACITY_THRESHOLD = 1.0  # bps/Hz

# 4. 消融实验兼容层 (Ablation Flags)
try:
    # 尝试读取 P1v16 的配置
    if 'REWARD_ABLATION_MODE' in globals():
        _legacy_mode = globals()['REWARD_ABLATION_MODE'].lower()
        if _legacy_mode == "no_v2i_penalty": REWARD_ABLATION = "no_v2i"
        elif _legacy_mode == "no_delay_reward": REWARD_ABLATION = "no_delay"
        elif _legacy_mode == "only_sinr": REWARD_ABLATION = "only_snr"
        else: REWARD_ABLATION = "none"
    else:
        REWARD_ABLATION = "none"
except:
    REWARD_ABLATION = "none"

# 5. 自动生成后缀 (用于文件名)
# 注意：这只是默认值，Main.py 运行时会根据 CLI 参数重新生成它
ABLATION_SUFFIX = f"_P1v21_s{SNR_MULTIPLIER}_v{V2I_MULTIPLIER}_d{DELAY_MULTIPLIER}"

# 6. 模型路径 (自动适配)
MODEL_PATH_GNN = f"model_GAT{ABLATION_SUFFIX}.pth"
MODEL_PATH_NO_GNN = "model_NoGNN_Baseline_v2.pth"
MODEL_PATH_DQN = "model_Standard_DQN.pth"

# 7. 随机种子
RANDOM_SEED = 11


# 全局列表
global_dqn_list = []

# 强化学习超参数
RL_ALPHA = 0.0001
RL_ALPHA_GNN = 0.0001
RL_EPSILON = 0.9
RL_EPSILON_MIN = 0.01
RL_EPSILON_MAX = 0.99
RL_EPSILON_DECAY = 0.995
RL_GAMMA = 0.95
LAMBDA_ENTROPY = 0.0005
RL_TAU = 0.005
RL_N_EPOCHS = 300

# 防止惩罚项过大导致梯度爆炸的截断阈值
REWARD_CLIP_MIN = -5.0
REWARD_CLIP_MAX = 5.0

# V2V 可靠性参数
V2V_PACKET_SIZE_BYTES = 1060  # V2V BSM消息大小 (字节)
V2V_PACKET_SIZE_BITS = V2V_PACKET_SIZE_BYTES * 8
V2V_CHANNEL_BANDWIDTH = 40e6 #28GHz 毫米波拥有巨大的频谱资源。将 400MHz 总带宽划分为 40MHz 的子信道是合理的（相比 20MHz 更能体现毫米波的高吞吐优势）。
V2V_DELAY_THRESHOLD = 0.01          # (10ms)
V2V_MIN_SNR_DB = 6.0               # (6dB)

# 信道模型选择标志
USE_UMI_NLOS_MODEL = True  # True: 使用新UMi NLOS模型, False: 使用旧模型

# 功能标志位
FLAG_ADAPTIVE_EPSILON_ADJUSTMENT = False
FLAG_EMA_LOSS = True
LOS = False  # 改为False，使用NLOS模型
NLOSS = True  # 改为True，使用NLOS模型

# UMi NLOS 信道参数
# 毫米波频段参数
CENTER_FREQUENCY = 28e9  # 载波频率 28 GHz
ANTENNA_HEIGHT_BS = 10  # RSU天线高度 10m (微基站)
ANTENNA_HEIGHT_UE = 1.5  # 车辆天线高度 1.5m

# 3GPP UMi NLOS 路径损耗模型参数
PATH_LOSS_A = 35.3  # 距离系数
PATH_LOSS_B = 22.4  # 常量项
PATH_LOSS_C = 21.3  # 频率系数
SHADOWING_STD = 7.0  # 阴影衰落标准差 (dB)

# 系统带宽 (毫米波典型带宽)
SYSTEM_BANDWIDTH = 400e6  # 系统带宽 400 MHz

# 噪声参数
NOISE_POWER_DENSITY = -174  # 热噪声功率谱密度 (dBm/Hz)
BOLTZMANN_CONSTANT = 1.38e-23  # 玻尔兹曼常数
NOISE_TEMPERATURE = 290  # 噪声温度 (K)

ATTENTION_MECHANISMS = {
    'multi_head': True,
    'hierarchical': True,
    'temporal': True,
    'spatial_temporal': True,
    'graph_aware': True
}

ATTENTION_DROPOUT = 0.1
TEMPORAL_SEQ_LEN = 5  # 时序注意力序列长度

# 场景参数
SCENE_SCALE_X = 1200
SCENE_SCALE_Y = 1200
VEHICLE_SAFETY_DISTANCE = 50
VEHICLE_CAPACITY_PER_LANE = int((SCENE_SCALE_X / 3) // VEHICLE_SAFETY_DISTANCE) + 1

# 状态空间大小
# 位置(x,y) + 方向(水平,垂直) = 4个维度
# CSI状态: 距离 + 路径损耗 + 阴影衰落 + 当前SNR + 历史SNR = 5个维度
RL_N_STATES_BASE = int(VEHICLE_CAPACITY_PER_LANE * 4)  # 基础状态
RL_N_STATES_CSI = int(VEHICLE_CAPACITY_PER_LANE * 5)  # CSI状态
RL_N_STATES_V2I = 3 # 历史V2I干扰状态
RL_N_STATES = RL_N_STATES_BASE + RL_N_STATES_CSI + RL_N_STATES_V2I  # 总状态维度


# 动作空间
def formulate_action_space():
    action_space = []
    for params in itertools.product(range(5), range(3), range(3), range(10)):
        action_space.append(list(params))
    return action_space


RL_ACTION_SPACE = formulate_action_space()
RL_N_ACTIONS = len(RL_ACTION_SPACE)

# 基站和车辆参数
BASE_STATION_HEIGHT = 10  # 更新为UMi模型中的10m

DIRECTION_H_RIGHT = 1
DIRECTION_H_STEADY = 0
DIRECTION_H_LEFT = -1
DIRECTION_V_UP = 1
DIRECTION_V_STEADY = 0
DIRECTION_V_DOWN = -1

TRAINING_VEHICLE_TARGET = 80 # 训练时维持的车辆总数

BOUNDARY_POSITION_LIST = [
    (0, SCENE_SCALE_Y / 3),
    (SCENE_SCALE_X / 3, 0),
    (SCENE_SCALE_X / 3, SCENE_SCALE_Y),
    (SCENE_SCALE_X, 2 * SCENE_SCALE_Y / 3),
    (2 * SCENE_SCALE_X / 3, SCENE_SCALE_Y),
    (SCENE_SCALE_X, SCENE_SCALE_Y / 3),
    (2 * SCENE_SCALE_X / 3, 0),

    # 为DQN 3, 5, 6 添加强制出生点
    (SCENE_SCALE_X / 3, SCENE_SCALE_Y / 3 + 1),
    (SCENE_SCALE_X / 3 + 1, 2 * SCENE_SCALE_Y / 3),
    (2 * SCENE_SCALE_X / 3, SCENE_SCALE_Y / 3 + 1),

    # 增加 DQN 1 (y=400) 的出生概率
    (0, SCENE_SCALE_Y / 3),
    (0, SCENE_SCALE_Y / 3),
    # 增加 DQN 5 (y=800) 的出生概率
    (SCENE_SCALE_X / 3 + 1, 2 * SCENE_SCALE_Y / 3),
    (SCENE_SCALE_X / 3 + 1, 2 * SCENE_SCALE_Y / 3),

]

CROSS_POSITION_LIST = [
    (SCENE_SCALE_X / 3, SCENE_SCALE_Y / 3),
    (SCENE_SCALE_X / 3, 2 * SCENE_SCALE_Y / 3),
    (2 * SCENE_SCALE_X / 3, 2 * SCENE_SCALE_Y / 3),
    (2 * SCENE_SCALE_X / 3, SCENE_SCALE_Y / 3),
]

VEHICLE_OCCUR_PROB = 0.5
VEHICLE_SPEED_KMH = 60
VEHICLE_SPEED_M3S = VEHICLE_SPEED_KMH * 1000 / 3600


# 添加缺失的通信参数
GAIN_ANTENNA_T = 1.0  # 发射天线增益
GAIN_ANTENNA_b = 1.0  # 接收天线增益
BANDWIDTH = 10e6  # 带宽
SPEED_C = 3e8  # 光速
SIGNAL_FREQUENCY = 28e9  # 信号频率
CARRIER_FREQUENCY = 28e9  # 载波频率

# 原有通信参数
# 以下参数将被新的UMi NLOS模型替代
CARRIER_FREQUENCY_DEPRECATED = 28e9  # 使用 CENTER_FREQUENCY 替代
BASE_STATION_HEIGHT_DEPRECATED = 20  # 使用 ANTENNA_HEIGHT_BS 替代
BANDWIDTH_DEPRECATED = 10e6  # 使用 SYSTEM_BANDWIDTH 替代
TRANSMITTDE_POWER = 3  # 保持，但将在新模型中使用


# 双头DQN网络结构参数
DUELING_HIDDEN_RATIO = 0.5  # 隐藏层比例
RL_N_HIDDEN = RL_N_ACTIONS * 3  # 默认值 (将被 Topology.py 动态覆盖)

# 优先级经验回放参数
PER_CAPACITY = 10000  # 经验回放缓冲区容量
PER_ALPHA = 0.6       # 优先级程度 (0=均匀, 1=完全优先级)
PER_BETA = 0.4        # 重要性采样权重
PER_BETA_INCREMENT = 0.001  # beta增量
PER_BATCH_SIZE = 32   # 训练批次大小

# 分布式PER参数
TARGET_UPDATE_FREQUENCY = 100 # 目标网络更新频率 (多少个 epoch 更新一次)


# V2I 链路模拟参数
# (假设有固定4个的 V2I 链路在场景中被干扰)
N_V2I_LINKS = 4              # 假设有 4 个 V2I 链路
V2I_TX_POWER = 0.2           # V2I 用户的固定发射功率 (23 dBm)
# V2I 链路的 (发射机, 接收机) 位置坐标
V2I_LINK_POSITIONS = [
    {'tx': (200, 200), 'rx': (200, 250)}, # V2I 链路 1
    {'tx': (200, 1000), 'rx': (200, 1050)}, # V2I 链路 2
    {'tx': (1000, 200), 'rx': (1000, 250)}, # V2I 链路 3
    {'tx': (1000, 1000), 'rx': (1000, 1050)}  # V2I 链路 4
]

# 更新参数打印函数
def print_parameters():
    debug_print("######## 参数 begin ########")
    debug_print("=== 强化学习参数 ===")
    debug_print(f"RL_ALPHA: {RL_ALPHA}")
    debug_print(f"RL_EPSILON: {RL_EPSILON}")
    debug_print(f"RL_GAMMA: {RL_GAMMA}")

    debug_print("=== 架构增强参数 ===")
    debug_print(f"USE_DUELING_DQN: {USE_DUELING_DQN}")
    debug_print(f"USE_PRIORITY_REPLAY: {USE_PRIORITY_REPLAY}")
    if USE_DUELING_DQN:
        debug_print(f"DUELING_HIDDEN_RATIO: {DUELING_HIDDEN_RATIO}")
    if USE_PRIORITY_REPLAY:
        debug_print(f"PER_CAPACITY: {PER_CAPACITY}")
        debug_print(f"PER_ALPHA: {PER_ALPHA}, PER_BETA: {PER_BETA}")

    debug_print("=== UMi NLOS 信道参数 ===")
    debug_print(f"CENTER_FREQUENCY: {CENTER_FREQUENCY / 1e9} GHz")
    debug_print(f"SYSTEM_BANDWIDTH: {SYSTEM_BANDWIDTH / 1e6} MHz")

    debug_print("=== 场景参数 ===")
    debug_print(f"SCENE_SCALE_X: {SCENE_SCALE_X}")
    debug_print(f"SCENE_SCALE_Y: {SCENE_SCALE_Y}")
    debug_print(f"RL_N_STATES: {RL_N_STATES} (Base: {RL_N_STATES_BASE} + CSI: {RL_N_STATES_CSI})")
    debug_print(f"RL_N_ACTIONS: {RL_N_ACTIONS}")
    debug_print(f"RL_N_HIDDEN: {RL_N_HIDDEN}")

    debug_print("=== GNN增强参数 ===")
    debug_print(f"USE_GNN_ENHANCEMENT: {USE_GNN_ENHANCEMENT}")

    debug_print(f"TARGET_UPDATE_FREQUENCY: {TARGET_UPDATE_FREQUENCY}")
    debug_print("######## 参数 end ########")