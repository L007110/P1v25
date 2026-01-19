import torch
import torch.optim as optim
import Parameters
import Main
import GNNModel
from Topology import formulate_global_list_dqn
from logger import global_logger


def run_no_cl_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 No-CL 基准训练 (Hard Mode Direct Training)")
    print(f"📍 目标密度: N=140 (直接训练，无课程)")
    print(f"⏱️ 总 Epochs: 2150 (与 CL 保持一致)")

    # 1. 设置参数
    Parameters.USE_GNN_ENHANCEMENT = True
    Parameters.GNN_ARCH = "HYBRID"
    Parameters.SCENE_SCALE_X = 1200
    Parameters.SCENE_SCALE_Y = 1200

    # 关键：直接设定为最难难度，且不更改
    Parameters.TRAINING_VEHICLE_TARGET = 140
    Parameters.NUM_VEHICLES = 140

    # 总 Epochs 等于 CL 累加的总和
    Parameters.RL_N_EPOCHS = 2150

    # 为了区分日志
    Parameters.ABLATION_SUFFIX = "_NoCL_Baseline_N140"

    # 2. 初始化环境
    formulate_global_list_dqn(Parameters.global_dqn_list, device)

    # 3. 初始化模型
    GNNModel.global_gnn_model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64).to(device)
    GNNModel.global_target_gnn_model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64).to(device)
    GNNModel.update_target_gnn()

    # 4. 优化器 (使用一个折中的学习率，或者 CL 最后阶段的学习率)
    gnn_optimizer = optim.Adam(GNNModel.global_gnn_model.parameters(), lr=0.0003)

    # 5. 开始训练
    # 注意：这里直接调用 Main.rl，不要挂载 run_smart_curriculum... 中的 density_mixer
    try:
        Main.rl(gnn_optimizer=gnn_optimizer, device=device)

        # 保存模型
        save_name = "model_NoCL_Baseline_N140.pt"
        torch.save(GNNModel.global_gnn_model.state_dict(), save_name)
        print(f"✅ No-CL 模型已保存: {save_name}")

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_no_cl_baseline()