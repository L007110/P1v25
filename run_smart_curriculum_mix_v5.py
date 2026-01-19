import torch
import torch.optim as optim
import os
import shutil
import numpy as np
import time
import gc
import sys
import math
import copy
import random

# 引入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Parameters
import Main
import GNNModel
import Topology
from logger import global_logger
from Topology import formulate_global_list_dqn


# =======================================================================
# 🔧 1. 修复版：混合密度拦截器 (Stable Density Mixer)
# =======================================================================
class VehicleDensityMixer:
    def __init__(self, original_func):
        self.original_func = original_func
        self.current_high_level_n = 20
        # [修改点 1] 提高混合比例，从 0.2 改为 0.5
        # 让模型有一半的时间在复习低密度，保持"野性"
        self.mix_ratio = 0.5
        self.low_density_candidates = [20]  # 初始值
        self.active = True
        self.episode_length = 50
        self.step_counter = 0
        self.current_target = 20

    def set_level(self, n):
        self.current_high_level_n = n

        # [修改点 2] 动态更新候选池
        # 当 n=100 时，candidates 自动变成 [20, 40, 60, 80]
        # 确保 N=60, N=80 这种"过渡区"也能被复习到
        if n > 20:
            self.low_density_candidates = [i for i in range(20, n, 20)]
        else:
            self.low_density_candidates = [20]

        self._refresh_target()

    def _refresh_target(self):
        # 如果是第一阶段，直接用当前密度
        if self.current_high_level_n <= 20:
            self.current_target = self.current_high_level_n
            return

        # [修改点 3] 只要 active 就进行混合采样
        if self.active and random.random() < self.mix_ratio:
            # 从 [20, 40, 60, ..., n-20] 中随机选一个复习
            self.current_target = random.choice(self.low_density_candidates)
        else:
            # 训练当前的高难度等级
            self.current_target = self.current_high_level_n


    def __call__(self, vehicle_id, vehicle_list, target_count=None, speed_kmh=60):
        if self.step_counter % self.episode_length == 0:
            self._refresh_target()

        self.step_counter += 1
        real_target = self.current_target

        if len(vehicle_list) > real_target:
            vehicle_list = vehicle_list[:real_target]

        return self.original_func(vehicle_id, vehicle_list, target_count=real_target, speed_kmh=speed_kmh)


# 🔥 安装拦截器
print("🛠️ 正在安装车辆密度拦截器 (V4 Full Curriculum)...")
original_movement_func = Topology.vehicle_movement
density_mixer = VehicleDensityMixer(original_movement_func)
Topology.vehicle_movement = density_mixer
print("✅ 拦截器安装完成！")


# =======================================================================
# 🔧 2. 修复版：缓冲区持久化
# =======================================================================
class PersistentBufferWrapper:
    _instance_store = []

    @classmethod
    def save_buffer(cls, buffer_instance):
        if buffer_instance is not None and len(buffer_instance) > 0:
            cls._instance_store = [buffer_instance.buffer]
            print(f"   💾 [Buffer] 已保存本关卡 {len(buffer_instance)} 条经验")


class PatchedGNNReplayBuffer(Main.GNNReplayBuffer):
    current_instance = None

    def __init__(self, capacity):
        super().__init__(capacity)
        PatchedGNNReplayBuffer.current_instance = self

        if PersistentBufferWrapper._instance_store:
            old_data = PersistentBufferWrapper._instance_store[0]
            inherit_ratio = 0.5
            inherit_size = int(len(old_data) * inherit_ratio)

            if inherit_size > 0:
                injected_data = random.sample(old_data, inherit_size)
                self.buffer = copy.deepcopy(injected_data)
                print(f"   🔄 [Buffer] 软继承: 抽取上一关 {inherit_size} 条经验")


Main.GNNReplayBuffer = PatchedGNNReplayBuffer

# =======================================================================
# 3. 完整课程配置 (从幼儿园到大学)
# =======================================================================

LEVEL_CONFIGS = {
    # 阶段一：基础瞄准 (无干扰/低干扰)
    20: (0.0005, 400, 0.5),  # 幼儿园：大探索，学瞄准
    40: (0.0005, 250, 0.4),  # 小学：稍微有点车，巩固瞄准

    # 阶段二：进阶抗干扰 (开启 Mix 回顾)
    60: (0.0004, 300, 0.3),  # 中学：开始面对拥堵
    80: (0.0004, 300, 0.2),  # 大学：复杂环境

    # 阶段三：专家模式 (高强度)
    100: (0.0003, 300, 0.15),
    120: (0.0003, 300, 0.1),
    140: (0.0002, 300, 0.1)
}

CURRICULUM_LEVELS = sorted(LEVEL_CONFIGS.keys())
FINAL_EPSILON = 0.01
FINAL_MODEL_NAME = "model_Universal_Final_V5.pt"


# =======================================================================
# 4. 主流程
# =======================================================================

def calculate_decay(start_eps, end_eps, total_epochs):
    target_step = int(total_epochs * 0.80)
    if target_step <= 0: return 0.9
    return math.pow(end_eps / start_eps, 1.0 / target_step)


def run_full_curriculum_v5():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 70}")
    print(f"🚀 启动全流程课程学习 V5 (Full Curriculum)")
    print(f"✨ 课程表: {CURRICULUM_LEVELS}")
    print(f"✨ 策略: 先学瞄准(20/40)，再学抗干扰(60+)，全程防遗忘")
    print(f"📍 设备: {device}")
    print(f"{'=' * 70}\n")

    Parameters.USE_GNN_ENHANCEMENT = True
    Parameters.GNN_ARCH = "HYBRID"
    Parameters.SCENE_SCALE_X = 1200
    Parameters.SCENE_SCALE_Y = 1200

    last_passed_model_path = None
    current_level_idx = 0

    while current_level_idx < len(CURRICULUM_LEVELS):
        n_vehicles = CURRICULUM_LEVELS[current_level_idx]
        current_lr, total_epochs, start_epsilon = LEVEL_CONFIGS[n_vehicles]

        density_mixer.set_level(n_vehicles)

        decay_rate = calculate_decay(start_epsilon, FINAL_EPSILON, total_epochs)

        print(f"\n" + "=" * 60)
        print(f"🔥 [LEVEL {current_level_idx + 1}] 当前关卡 N={n_vehicles}")
        print(f"🎲 Epsilon: {start_epsilon} -> {FINAL_EPSILON}")
        print("=" * 60)

        # 环境准备
        gc.collect()
        torch.cuda.empty_cache()

        Parameters.TRAINING_VEHICLE_TARGET = n_vehicles
        Parameters.NUM_VEHICLES = n_vehicles
        Parameters.RL_N_EPOCHS = total_epochs
        Parameters.ABLATION_SUFFIX = f"_V5_N{n_vehicles}"

        # global_logger._init_metrics_storage()
        formulate_global_list_dqn(Parameters.global_dqn_list, device)

        for dqn in Parameters.global_dqn_list:
            dqn.epsilon = start_epsilon

        # 模型加载
        GNNModel.global_gnn_model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64).to(device)
        GNNModel.global_target_gnn_model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64).to(device)

        if last_passed_model_path and os.path.exists(last_passed_model_path):
            print(f"   📥 继承上一关权重: {last_passed_model_path}")
            checkpoint = torch.load(last_passed_model_path, map_location=device)
            GNNModel.global_gnn_model.load_state_dict(checkpoint)
            GNNModel.global_target_gnn_model.load_state_dict(checkpoint)
        else:
            print("   🌱 [Cold Start] 从零开始初始化 (N=20)")
            GNNModel.update_target_gnn()

        gnn_optimizer = optim.Adam(GNNModel.global_gnn_model.parameters(), lr=current_lr)

        try:
            Main.rl(gnn_optimizer=gnn_optimizer, device=device)

            if hasattr(PatchedGNNReplayBuffer, 'current_instance'):
                active_buf = PatchedGNNReplayBuffer.current_instance
                PersistentBufferWrapper.save_buffer(active_buf)

            save_name = f"checkpoint_v5_passed_N{n_vehicles}.pt"
            torch.save(GNNModel.global_gnn_model.state_dict(), save_name)
            last_passed_model_path = save_name
            current_level_idx += 1

        except Exception as e:
            print(f"   ❌ 训练中断: {e}")
            import traceback
            traceback.print_exc()
            return

    print("\n" + "=" * 70)
    print("🏆 全流程训练完成！")
    if last_passed_model_path:
        shutil.copy(last_passed_model_path, FINAL_MODEL_NAME)
        print(f"💾 最终通用模型: {FINAL_MODEL_NAME}")
    print("=" * 70)


if __name__ == "__main__":
    run_full_curriculum_v5()