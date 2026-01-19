import os
import torch
import numpy as np
import pandas as pd
import random
import Parameters
import Main
import GNNModel
from Topology import formulate_global_list_dqn, vehicle_movement

# ================= 配置区域 =================
# 定义你要对比的四个“选手”
MODELS = {
    "Proposed (Ours)": {"path": "model_Universal_Final_V5.pt", "type": "GNN", "arch": "HYBRID"},
    "Ji et al. (GCN)": {"path": "model_GCN.pt", "type": "GNN", "arch": "GCN"},
    "Ashraf (No-GNN)": {"path": "model_NoGNN.pt", "type": "NoGNN", "arch": None},
    "Random Baseline": {"path": None, "type": "Random", "arch": None}
}

SCENARIOS = [20, 40, 60, 80, 100, 120, 140]  # 测试密度列表
TEST_STEPS = 200  # 正式测试步数
WARMUP_STEPS = 500  # 预热步数 (保持与 Honest Eval 一致)
SYSTEM_BANDWIDTH = 400e6  # 400 MHz (保持与 Honest Eval 一致)


def calculate_shannon_capacity(snr_db, bandwidth_hz):
    """ 香农公式: C = B * log2(1 + S/N) """
    if snr_db < -100: return 0.0
    snr_linear = 10 ** (snr_db / 10.0)
    return bandwidth_hz * np.log2(1 + snr_linear) / 1e6  # Mbps


def evaluate_method(method_name, config, density_list):
    print(f"\n🚀 启动评估: {method_name} (Type: {config['type']})")
    print(f"📐 统计标准: 全物理计算 (Real Physics V2I/V2V) | Bandwidth: 400MHz | Warmup: {WARMUP_STEPS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    # 1. 环境强制配置 (完全对齐 honest_eval)
    Parameters.RUN_MODE = "TEST"
    Parameters.SCENE_SCALE_X = 1200
    Parameters.SCENE_SCALE_Y = 1200

    # 根据方法类型配置 GNN 开关
    if config["type"] == "GNN":
        Parameters.USE_GNN_ENHANCEMENT = True
        Parameters.GNN_ARCH = config["arch"]
    else:
        Parameters.USE_GNN_ENHANCEMENT = False  # NoGNN 和 Random 关闭 GNN

    # 2. 初始化环境
    formulate_global_list_dqn(Parameters.global_dqn_list, device)
    channel_model = Main.new_reward_calculator.channel_model

    # 3. 加载模型
    gnn_model = None
    if config["type"] == "GNN":
        try:
            gnn_model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64).to(device)
            # 兼容 CPU/GPU 加载
            if torch.cuda.is_available():
                state = torch.load(config["path"])
            else:
                state = torch.load(config["path"], map_location=torch.device('cpu'))

            gnn_model.load_state_dict(state)
            gnn_model.eval()
            GNNModel.global_gnn_model = gnn_model
            print(f"   ✅ GNN 模型加载成功: {config['path']}")
        except Exception as e:
            print(f"   ⚠️ 模型加载失败 {config['path']}: {e}")
            return []

    elif config["type"] == "NoGNN":
        try:
            # NoGNN 是保存的 DQN 字典
            if torch.cuda.is_available():
                checkpoint = torch.load(config["path"])
            else:
                checkpoint = torch.load(config["path"], map_location=torch.device('cpu'))

            for dqn in Parameters.global_dqn_list:
                dqn.load_state_dict(checkpoint[f'dqn_{dqn.dqn_id}'])
                dqn.eval()
            print(f"   ✅ No-GNN 模型加载成功: {config['path']}")
        except Exception as e:
            print(f"   ⚠️ 模型加载失败 {config['path']}: {e}")
            return []

    # 4. 循环密度测试
    for n in density_list:
        Parameters.TRAINING_VEHICLE_TARGET = n
        Parameters.NUM_VEHICLES = n

        # 重置所有 DQN 状态 (对齐 honest_eval)
        for dqn in Parameters.global_dqn_list:
            dqn.delay_list = []
            dqn.snr_list = []
            dqn.v2v_success_list = []
            dqn.feasible_list = []
            dqn.prev_v2i_interference = 0.0
            dqn.curr_state = []
            dqn.epsilon = 0.0

        # === 预热 (Warm-up) ===
        vid = 0
        vlist = []
        # print(f"   🔥 Warming up for {WARMUP_STEPS} steps...")
        for _ in range(WARMUP_STEPS):
            vid, vlist = vehicle_movement(vid, vlist, target_count=n)

        # 统计容器
        history_S = []  # 成功率
        history_F = []  # 可行率
        history_V2V_Sum = []
        history_V2I_Sum = []
        history_SNR = []

        # === 正式测试循环 ===
        for step in range(TEST_STEPS):
            # A. 移动
            vid, vlist = vehicle_movement(vid, vlist, target_count=n)

            # B. 观测更新 (所有方法都需要基本的物理观测来计算距离等)
            for dqn in Parameters.global_dqn_list:
                dqn.vehicle_exist_curr = False
                dqn.vehicle_in_dqn_range_by_distance = []
                for v in vlist:
                    if dqn.start[0] <= v.curr_loc[0] <= dqn.end[0] and dqn.start[1] <= v.curr_loc[1] <= dqn.end[1]:
                        dqn.vehicle_exist_curr = True
                        v.distance_to_bs = channel_model.calculate_3d_distance((dqn.bs_loc[0], dqn.bs_loc[1]),
                                                                               v.curr_loc)
                        dqn.vehicle_in_dqn_range_by_distance.append(v)

                # 排序并更新 CSI (即使是 NoGNN 也需要 update_csi_states 来刷新 dqn.csi_states_curr)
                dqn.vehicle_in_dqn_range_by_distance.sort(key=lambda x: x.distance_to_bs)
                if dqn.vehicle_exist_curr:
                    dqn.update_csi_states(dqn.vehicle_in_dqn_range_by_distance, is_current=True)

            # C. 动作决策 (不同方法分支)

            # --- C1. Random ---
            if config["type"] == "Random":
                for dqn in Parameters.global_dqn_list:
                    if dqn.vehicle_exist_curr:
                        # 随机动作: Beam(0-4), H(0-2), V(0-2), Power(0-9)
                        dqn.action = [random.randint(0, 4), random.randint(0, 2), random.randint(0, 2),
                                      random.randint(0, 9)]

                        # 物理映射 (关键: Random 也要产生真实的功率，否则干扰为0)
                        if dqn.vehicle_in_dqn_range_by_distance:
                            beam_count = dqn.action[0] + 1
                            power_ratio = (dqn.action[3] + 1) / 10.0
                            gain = Main.new_reward_calculator._calculate_directional_gain(dqn.action[1], dqn.action[2])
                            pwr = Parameters.TRANSMITTDE_POWER * power_ratio * beam_count * gain * Parameters.GAIN_ANTENNA_T
                            dqn.vehicle_in_dqn_range_by_distance[0].power_W = pwr
                            dqn.vehicle_in_dqn_range_by_distance[0].tx_pos = dqn.vehicle_in_dqn_range_by_distance[
                                0].curr_loc
                    else:
                        dqn.action = None

            # --- C2. GNN (Ours & Ji) ---
            elif config["type"] == "GNN":
                graph = Main.global_graph_builder.build_dynamic_graph(Parameters.global_dqn_list, vlist, step)
                graph = Main.move_graph_to_device(graph, device)
                with torch.no_grad():
                    q_values, _ = gnn_model(graph)
                    # 重置功率
                    for v in vlist: v.power_W = 0.0; v.tx_pos = v.curr_loc

                    for dqn in Parameters.global_dqn_list:
                        if dqn.vehicle_exist_curr:
                            idx = dqn.dqn_id - 1
                            act_idx = q_values[idx].argmax().item()
                            dqn.action = Parameters.RL_ACTION_SPACE[act_idx]

                            # 物理映射
                            if dqn.vehicle_in_dqn_range_by_distance:
                                beam_count = dqn.action[0] + 1
                                power_ratio = (dqn.action[3] + 1) / 10.0
                                gain = Main.new_reward_calculator._calculate_directional_gain(dqn.action[1],
                                                                                              dqn.action[2])
                                pwr = Parameters.TRANSMITTDE_POWER * power_ratio * beam_count * gain * Parameters.GAIN_ANTENNA_T
                                dqn.vehicle_in_dqn_range_by_distance[0].power_W = pwr
                                dqn.vehicle_in_dqn_range_by_distance[0].tx_pos = dqn.vehicle_in_dqn_range_by_distance[
                                    0].curr_loc
                        else:
                            dqn.action = None

            # --- C3. NoGNN (Ashraf) ---
            elif config["type"] == "NoGNN":
                for dqn in Parameters.global_dqn_list:
                    if dqn.vehicle_exist_curr:
                        # 构建状态: Local State + V2I History (归一化)
                        base_state = []
                        iState = 0
                        for iVehicle in range(
                                min(Parameters.RL_N_STATES_BASE // 4, len(dqn.vehicle_in_dqn_range_by_distance))):
                            v = dqn.vehicle_in_dqn_range_by_distance[iVehicle]
                            base_state.extend([v.curr_loc[0], v.curr_loc[1], v.curr_dir[0], v.curr_dir[1]])
                            iState += 4
                        if len(base_state) < Parameters.RL_N_STATES_BASE:
                            base_state.extend([0.0] * (Parameters.RL_N_STATES_BASE - len(base_state)))

                        # V2I 干扰历史
                        interf_norm = (np.log10(dqn.prev_v2i_interference + 1e-20) + 20) / 14.0
                        v2i_state = [interf_norm, 0.0, 0.0]

                        # 拼接状态
                        dqn.curr_state = base_state + dqn.csi_states_curr + v2i_state

                        # 推理
                        with torch.no_grad():
                            state_tensor = torch.tensor(dqn.curr_state).float().to(device).unsqueeze(0)
                            q = dqn(state_tensor)
                            act_idx = q.argmax().item()
                            dqn.action = Parameters.RL_ACTION_SPACE[act_idx]

                            # 物理映射
                            if dqn.vehicle_in_dqn_range_by_distance:
                                beam_count = dqn.action[0] + 1
                                power_ratio = (dqn.action[3] + 1) / 10.0
                                gain = Main.new_reward_calculator._calculate_directional_gain(dqn.action[1],
                                                                                              dqn.action[2])
                                pwr = Parameters.TRANSMITTDE_POWER * power_ratio * beam_count * gain * Parameters.GAIN_ANTENNA_T
                                dqn.vehicle_in_dqn_range_by_distance[0].power_W = pwr
                                dqn.vehicle_in_dqn_range_by_distance[0].tx_pos = dqn.vehicle_in_dqn_range_by_distance[
                                    0].curr_loc
                    else:
                        dqn.action = None

            # D. 计算与统计 (核心部分: Honest Physics)
            # 获取所有发射干扰源
            active_interferers = [{'tx_pos': v.curr_loc, 'power_W': v.power_W} for v in vlist if v.power_W > 0]

            step_v2v_sum = 0.0
            step_v2i_sum = 0.0

            # --- D1. 处理 V2V 链路 ---
            for dqn in Parameters.global_dqn_list:
                if dqn.vehicle_exist_curr:
                    Main.new_reward_calculator.calculate_complete_reward(
                        dqn, dqn.vehicle_in_dqn_range_by_distance, dqn.action, active_interferers
                    )

                    # 记录 SNR 和 容量
                    if dqn.snr_list:
                        current_snr = dqn.snr_list[-1]
                        link_cap = calculate_shannon_capacity(current_snr, SYSTEM_BANDWIDTH)
                        step_v2v_sum += link_cap
                        history_SNR.append(current_snr)

                    # 记录原始成功状态
                    if dqn.v2v_success_list: history_S.append(dqn.v2v_success_list[-1])
                    if dqn.feasible_list: history_F.append(dqn.feasible_list[-1])

                    # 维护 NoGNN 需要的 V2I 历史 (预测下一帧干扰)
                    v2i_next = 0.0
                    if dqn.vehicle_in_dqn_range_by_distance and dqn.vehicle_in_dqn_range_by_distance[0].power_W > 0:
                        my_pos = dqn.vehicle_in_dqn_range_by_distance[0].curr_loc
                        my_pwr = dqn.vehicle_in_dqn_range_by_distance[0].power_W
                        for link in Parameters.V2I_LINK_POSITIONS:
                            d = channel_model.calculate_3d_distance(my_pos, link['rx'])
                            pl, _, _ = channel_model.calculate_path_loss(d)
                            v2i_next += my_pwr * (10 ** (-pl / 10))
                    dqn.prev_v2i_interference = v2i_next

            # --- D2. 处理 V2I 链路 (全物理计算) ---
            noise_w = channel_model._calculate_noise_power(SYSTEM_BANDWIDTH)

            for link in Parameters.V2I_LINK_POSITIONS:
                # Signal (Parameters.V2I_TX_POWER = 0.2W / 23dBm)
                d_sig = channel_model.calculate_3d_distance(link['tx'], link['rx'])
                _, _, sig_w = channel_model.calculate_snr(Parameters.V2I_TX_POWER, d_sig, bandwidth=SYSTEM_BANDWIDTH)

                # Interference (来自所有 V2V 用户)
                int_w = 0.0
                for interf in active_interferers:
                    d_i = channel_model.calculate_3d_distance(interf['tx_pos'], link['rx'])
                    pl, _, _ = channel_model.calculate_path_loss(d_i)
                    int_w += interf['power_W'] * (10 ** (-pl / 10))

                # Capacity
                sinr_v2i = sig_w / (int_w + noise_w + 1e-20)
                cap = calculate_shannon_capacity(10 * np.log10(sinr_v2i), SYSTEM_BANDWIDTH)
                step_v2i_sum += cap

            history_V2V_Sum.append(step_v2v_sum)
            history_V2I_Sum.append(step_v2i_sum)

        # E. 汇总该密度结果
        # Feasible Success Rate = Raw_Success * Feasible (物理可行性)
        # 注意: 只有当 history_F 中有 1 时，分母才有效，否则为 0
        if history_S and history_F:
            raw_succ = np.mean(history_S)
            feas_succ = np.mean(np.array(history_S) * np.array(history_F)) / (np.mean(history_F) + 1e-10) * np.mean(
                history_F)
            # 简化计算：Feasible Success Rate 定义为 (成功且可行次数) / 总尝试次数
            # 这里沿用 honest_eval 的逻辑：
            # feas_succ = np.mean(np.array(history_S) * np.array(history_F)) # 严格定义
        else:
            raw_succ = 0
            feas_succ = 0

        avg_v2v = np.mean(history_V2V_Sum)
        avg_v2i = np.mean(history_V2I_Sum)
        avg_snr = np.mean(history_SNR) if history_SNR else -100

        print(f"   📊 N={n} | V2V Succ: {raw_succ:.2%} | V2V Sum: {avg_v2v:.1f} Mbps | V2I Sum: {avg_v2i:.1f} Mbps")

        results.append({
            "Method": method_name,
            "Density": n,
            "V2V_Success_Rate": raw_succ,  # 保持原始成功率，或根据需要改为 feas_succ
            "V2V_Sum_Capacity": avg_v2v,
            "V2I_Sum_Capacity": avg_v2i,
            "Avg_SNR": avg_snr
        })

    return results


if __name__ == "__main__":
    all_data = []

    # 检查文件是否存在 (Random 除外)
    for name, conf in MODELS.items():
        if conf["path"] and not os.path.exists(conf["path"]):
            print(f"❌ 找不到模型文件: {conf['path']} (Skipping {name})")
            continue

        # 跑评估
        res = evaluate_method(name, conf, SCENARIOS)
        all_data.extend(res)

    # 保存大表
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv("Final_Comparison_Results.csv", index=False)
        print("\n✅ 所有对比测试完成！数据已保存至 Final_Comparison_Results.csv")
    else:
        print("\n⚠️ 没有生成任何数据，请检查模型路径。")