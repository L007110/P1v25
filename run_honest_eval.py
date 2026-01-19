import os
import torch
import numpy as np
import pandas as pd
import Parameters
import Main
import GNNModel
from Topology import formulate_global_list_dqn


def calculate_shannon_capacity(snr_db, bandwidth_hz):
    """
    香农公式: C = B * log2(1 + S/N)
    """
    if snr_db < -100: return 0.0
    snr_linear = 10 ** (snr_db / 10.0)
    return bandwidth_hz * np.log2(1 + snr_linear) / 1e6  # Mbps


def run_honest_evaluation(model_path, target_density_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动多维诚实评估 (Mode: Inference Only)")
    print(f"📂 模型: {model_path}")
    print(f"📐 统计定义: 全物理计算 (Real Physics V2I/V2V)")

    # 1. 强制参数覆盖
    Parameters.RUN_MODE = "TEST"
    Parameters.USE_GNN_ENHANCEMENT = True
    Parameters.GNN_ARCH = "HYBRID"
    Parameters.SCENE_SCALE_X = 1200
    Parameters.SCENE_SCALE_Y = 1200

    # 【关键】带宽修正为 400 MHz (Parameters.SYSTEM_BANDWIDTH)
    SYSTEM_BANDWIDTH = 400e6

    # 2. 初始化环境
    formulate_global_list_dqn(Parameters.global_dqn_list, device)

    # 3. 加载模型
    model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64).to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()

    GNNModel.global_gnn_model = model
    GNNModel.global_target_gnn_model = model

    results = []

    # 获取信道模型引用，方便调用
    channel_model = Main.new_reward_calculator.channel_model

    # 4. 循环测试不同的密度 N
    for n in target_density_list:
        print(f"\n⚡ 正在测试密度 N={n} ...")

        Parameters.TRAINING_VEHICLE_TARGET = n
        Parameters.NUM_VEHICLES = n

        # 重置所有 DQN 状态
        for dqn in Parameters.global_dqn_list:
            dqn.delay_list = []
            dqn.snr_list = []
            dqn.v2v_success_list = []
            dqn.feasible_list = []
            dqn.prev_v2i_interference = 0.0
            dqn.curr_state = []
            dqn.epsilon = 0.0

        # 预热 (Warm-up)
        from Topology import vehicle_movement
        vid = 0
        vlist = []
        for _ in range(500):
            vid, vlist = vehicle_movement(vid, vlist, target_count=n)

        # 数据容器
        history_S = []
        history_F = []
        history_SNR = []
        history_Link_Cap = []
        history_V2V_Sum = []
        history_V2I_Sum = []

        test_steps = 200

        for step in range(test_steps):
            # A. 移动
            vid, vlist = vehicle_movement(vid, vlist, target_count=n)

            # B. 状态更新
            for dqn in Parameters.global_dqn_list:
                dqn.vehicle_exist_curr = False
                dqn.vehicle_in_dqn_range_by_distance = []
                for vehicle in vlist:
                    if (dqn.start[0] <= vehicle.curr_loc[0] <= dqn.end[0] and
                            dqn.start[1] <= vehicle.curr_loc[1] <= dqn.end[1]):
                        dqn.vehicle_exist_curr = True
                        vehicle.distance_to_bs = channel_model.calculate_3d_distance(
                            (dqn.bs_loc[0], dqn.bs_loc[1]), vehicle.curr_loc)
                        dqn.vehicle_in_dqn_range_by_distance.append(vehicle)
                dqn.vehicle_in_dqn_range_by_distance.sort(key=lambda x: x.distance_to_bs)
                if dqn.vehicle_exist_curr:
                    dqn.update_csi_states(dqn.vehicle_in_dqn_range_by_distance, is_current=True)

            # C. 推理 (GNN)
            graph = Main.global_graph_builder.build_dynamic_graph(Parameters.global_dqn_list, vlist, step)
            graph = Main.move_graph_to_device(graph, device)

            with torch.no_grad():
                q_values, _ = model(graph)
                # 重置功率
                for v in vlist: v.power_W = 0.0; v.tx_pos = v.curr_loc

                for dqn in Parameters.global_dqn_list:
                    if dqn.vehicle_exist_curr:
                        dqn_idx = dqn.dqn_id - 1
                        action_idx = q_values[dqn_idx].argmax().item()
                        dqn.action = Parameters.RL_ACTION_SPACE[action_idx]

                        # 设置功率
                        beam_count = dqn.action[0] + 1
                        power_ratio = (dqn.action[3] + 1) / 10.0
                        gain = Main.new_reward_calculator._calculate_directional_gain(dqn.action[1], dqn.action[2])
                        pwr = Parameters.TRANSMITTDE_POWER * power_ratio * beam_count * gain * Parameters.GAIN_ANTENNA_T
                        if dqn.vehicle_in_dqn_range_by_distance:
                            dqn.vehicle_in_dqn_range_by_distance[0].power_W = pwr
                    else:
                        dqn.action = None

            # D. 计算与统计 (核心部分)
            active_interferers = [{'tx_pos': v.curr_loc, 'power_W': v.power_W} for v in vlist if v.power_W > 0]

            step_v2v_sum = 0.0
            step_v2i_sum = 0.0

            # --- 1. 处理 V2V 链路 ---
            for dqn in Parameters.global_dqn_list:
                if dqn.vehicle_exist_curr:
                    Main.new_reward_calculator.calculate_complete_reward(
                        dqn, dqn.vehicle_in_dqn_range_by_distance, dqn.action, active_interferers
                    )

                    if dqn.snr_list:
                        current_snr = dqn.snr_list[-1]
                        link_cap = calculate_shannon_capacity(current_snr, SYSTEM_BANDWIDTH)

                        history_SNR.append(current_snr)
                        history_Link_Cap.append(link_cap)
                        step_v2v_sum += link_cap

                    if dqn.v2v_success_list: history_S.append(dqn.v2v_success_list[-1])
                    if dqn.feasible_list: history_F.append(dqn.feasible_list[-1])

                    # 更新 V2I 干扰状态 (供 Agent 观测用)
                    v2i_interf_next = 0.0
                    if dqn.vehicle_in_dqn_range_by_distance and dqn.vehicle_in_dqn_range_by_distance[0].power_W > 0:
                        my_pos = dqn.vehicle_in_dqn_range_by_distance[0].curr_loc
                        my_pwr = dqn.vehicle_in_dqn_range_by_distance[0].power_W
                        for link in Parameters.V2I_LINK_POSITIONS:
                            d_temp = channel_model.calculate_3d_distance(my_pos, link['rx'])
                            pl_temp, _, _ = channel_model.calculate_path_loss(d_temp)
                            v2i_interf_next += my_pwr * (10 ** (-pl_temp / 10))
                    dqn.prev_v2i_interference = v2i_interf_next

            # --- 2. 处理 V2I 链路 (【修正】全物理计算) ---
            # 不再使用假设值，而是直接读取 Parameters.V2I_TX_POWER (0.2W)

            # 计算底噪
            noise_w = channel_model._calculate_noise_power(SYSTEM_BANDWIDTH)

            # 遍历每一条真实的 V2I 链路
            for link in Parameters.V2I_LINK_POSITIONS:
                # 2.1 计算 V2I 信号强度 (Signal)
                # 根据发射点(link['tx'])和接收点(link['rx'])计算距离
                d_sig = channel_model.calculate_3d_distance(link['tx'], link['rx'])

                # 使用 ChannelModel 计算接收功率
                # 注意：calculate_snr 内部已经包含了 Path Loss 计算
                # 参数: TxPower=0.2W, Distance=d_sig
                _, _, v2i_sig_w = channel_model.calculate_snr(Parameters.V2I_TX_POWER, d_sig,
                                                              bandwidth=SYSTEM_BANDWIDTH)

                # 2.2 计算 V2I 干扰强度 (Interference)
                total_interf_w = 0.0
                for interf in active_interferers:
                    d_i = channel_model.calculate_3d_distance(interf['tx_pos'], link['rx'])
                    pl_i, _, _ = channel_model.calculate_path_loss(d_i)
                    total_interf_w += interf['power_W'] * (10 ** (-pl_i / 10))

                # 2.3 计算 SINR
                sinr_v2i = v2i_sig_w / (noise_w + total_interf_w + 1e-20)

                # 2.4 计算容量
                v2i_link_cap = calculate_shannon_capacity(10 * np.log10(sinr_v2i), SYSTEM_BANDWIDTH)
                step_v2i_sum += v2i_link_cap

            # 记录这一帧的总量
            history_V2V_Sum.append(step_v2v_sum)
            history_V2I_Sum.append(step_v2i_sum)

        # ==========================================
        # 📊 最终统计
        # ==========================================
        avg_v2v_sum = np.mean(history_V2V_Sum)
        avg_v2i_sum = np.mean(history_V2I_Sum)
        total_system = avg_v2v_sum + avg_v2i_sum

        avg_snr = np.mean(history_SNR) if history_SNR else -100
        avg_link_rate = np.mean(history_Link_Cap) if history_Link_Cap else 0

        raw_succ = np.mean(history_S) if history_S else 0
        feas_succ = np.mean(np.array(history_S) * np.array(history_F)) / np.mean(history_F) if np.sum(
            history_F) > 0 else 0

        print(f"   📊 N={n} Result:")
        print(f"      [Reliability] Feasible: {feas_succ * 100:.2f}% | Raw: {raw_succ * 100:.2f}%")
        print(f"      [Micro QoS  ] Avg SNR: {avg_snr:.2f} dB | Avg Link Rate: {avg_link_rate:.2f} Mbps")
        print(f"      [Macro Sum  ] V2V Sum: {avg_v2v_sum:.2f} Mbps | V2I Sum: {avg_v2i_sum:.2f} Mbps")

        results.append({
            'density': n,
            'raw_success_rate': raw_succ,
            'feasible_success_rate': feas_succ,
            'v2v_sum_mbps': avg_v2v_sum,
            'v2i_sum_mbps': avg_v2i_sum,
            'system_sum_rate_mbps': total_system,
            'avg_snr_db': avg_snr,
            'avg_v2v_capacity_mbps': avg_link_rate,
            'avg_v2i_capacity_mbps': avg_v2i_sum / 4.0
        })

    df = pd.DataFrame(results)
    df.to_csv("honest_test_results_qos.csv", index=False)
    print("\n✅ 数据已保存: honest_test_results_qos.csv")


if __name__ == "__main__":
    MODEL = "model_NoCL_Baseline_N140.pt"
    if not os.path.exists(MODEL): MODEL = "checkpoint_v5_passed_N140.pt"
    SCENARIOS = [20, 40, 60, 80, 100, 120, 140]
    run_honest_evaluation(MODEL, SCENARIOS)