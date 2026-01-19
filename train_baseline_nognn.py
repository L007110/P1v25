# train_baseline_nognn_FIXED.py
import sys
import torch
import Parameters  # 1. 先导入 Parameters

# ==========================================
# 🛑 强制关闭 GNN (必须在 import Main 之前!)
# ==========================================
print("⚡ FORCE DISABLING GNN MODE...")
Parameters.USE_GNN_ENHANCEMENT = False
Parameters.GNN_ARCH = "NONE"  # 双重保险
# ==========================================

import Main  # 2. 现在才导入 Main (此时 Main 看到的是 False)
import GNNModel
from Topology import formulate_global_list_dqn


def train_nognn_baseline():
    print("🚀 Training Baseline: No-GNN (Ashraf / Pure DRL)")

    # 二次确认
    if Parameters.USE_GNN_ENHANCEMENT:
        raise ValueError("❌ 严重错误: GNN 仍然处于开启状态！请检查导入顺序。")
    else:
        print("✅ 检测通过: GNN 已成功关闭 (USE_GNN_ENHANCEMENT = False)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 其他配置 ===
    Parameters.USE_DUELING_DQN = True
    Parameters.TRAINING_VEHICLE_TARGET = 80
    Parameters.NUM_VEHICLES = 80
    Parameters.RL_N_EPOCHS = 1000
    Parameters.ABLATION_SUFFIX = "_NoGNN_Baseline"

    # === 初始化 ===
    formulate_global_list_dqn(Parameters.global_dqn_list, device)

    # === 训练 ===
    # 这里的 rl() 现在应该读取到 False
    Main.rl(device=device)

    # === 保存 ===
    save_data = {f'dqn_{dqn.dqn_id}': dqn.state_dict() for dqn in Parameters.global_dqn_list}
    torch.save(save_data, "model_NoGNN.pt")
    print("✅ model_NoGNN.pt saved.")


if __name__ == "__main__":
    train_nognn_baseline()