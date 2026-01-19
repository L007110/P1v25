# train_baseline_gcn.py
import torch
import torch.optim as optim
import Parameters
import Main
import GNNModel
from Topology import formulate_global_list_dqn


def train_gcn_baseline():
    print("🚀 Training Baseline: GCN (Proxy for Ji et al.)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 配置 ===
    Parameters.USE_GNN_ENHANCEMENT = True
    Parameters.GNN_ARCH = "GCN"  # <--- 关键修改
    Parameters.TRAINING_VEHICLE_TARGET = 80  # 固定密度
    Parameters.NUM_VEHICLES = 80
    Parameters.RL_N_EPOCHS = 1000  # 足够收敛即可
    Parameters.ABLATION_SUFFIX = "_GCN_Baseline"

    # === 初始化 ===
    formulate_global_list_dqn(Parameters.global_dqn_list, device)

    # 重新初始化 GNN 模型为 GCN 架构
    GNNModel.global_gnn_model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64).to(device)
    GNNModel.global_target_gnn_model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64).to(device)
    GNNModel.update_target_gnn()

    optimizer = optim.Adam(GNNModel.global_gnn_model.parameters(), lr=0.0003)

    # === 训练 ===
    Main.rl(gnn_optimizer=optimizer, device=device)

    # === 保存 ===
    torch.save(GNNModel.global_gnn_model.state_dict(), "model_GCN.pt")
    print("✅ model_GCN.pt saved.")


if __name__ == "__main__":
    train_gcn_baseline()