# -*- coding: utf-8 -*-
import logging
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class TrainingLogger:

    def __init__(self, log_dir="training_results"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self._setup_logging()
        self._init_metrics_storage()

        self.logger.info("TrainingLogger initialized")
        self.logger.info(f"Log directory: {os.path.abspath(log_dir)}")

    def _setup_logging(self):
        self.logger = logging.getLogger('RL_Training')
        self.logger.setLevel(logging.INFO)

        # 防止重复添加handler
        if self.logger.handlers:
            self.logger.handlers.clear()

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S'
        )

        # 文件handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(f'{self.log_dir}/training_{timestamp}.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)


    def _init_metrics_storage(self):
        """初始化指标存储结构"""
        self.metrics = {
            'epoch': [],
            'cumulative_reward': [],
            'mean_loss': [],
            'mean_delay': [],
            'p95_delay': [],
            'mean_snr': [],
            'vehicle_count': [],
            'v2v_success_rate': [],
            # --- [NEW] ---
            'feasible_v2v_success_rate': [],
            # -------------
            'v2i_sum_capacity': [],
            'timestamp': [],
            'v2v_delay_only_rate': [],
            'v2v_snr_only_rate': []
        }

        self.dqn_metrics = defaultdict(lambda: {
            'loss': [],
            'reward': [],
            'epsilon': [],
            'vehicle_count': [],
            'snr': [],
            'delay': []
        })

        self.training_stats = {
            'start_time': datetime.now(),
            'total_epochs': 0,
            'best_reward': -float('inf'),
            'best_epoch': 0,
            'convergence_epoch': None
        }



    def _convert_tensor_to_float(self, value):
        """安全转换PyTorch张量为float"""
        if hasattr(value, 'detach'):
            # 如果是PyTorch张量
            return value.detach().item() if value.numel() == 1 else float(value.detach().mean())
        elif hasattr(value, 'numpy'):
            # 如果是TensorFlow张量或其他
            return float(value)
        else:
            # 如果是普通数值
            return float(value)

    def _convert_tensor_to_int(self, value):
        """安全转换PyTorch张量为int"""
        if hasattr(value, 'detach'):
            # 如果是PyTorch张量
            return int(value.detach().item() if value.numel() == 1 else int(value.detach().mean()))
        elif hasattr(value, 'numpy'):
            # 如果是TensorFlow张量或其他
            return int(value)
        else:
            # 如果是普通数值
            return int(value)

    def log_epoch(self, epoch, cumulative_reward, mean_loss, mean_delay, p95_delay,
                  mean_snr, vehicle_count, v2v_success_rate, v2i_sum_capacity,
                  v2v_delay_only_rate=None, v2v_snr_only_rate=None,
                  feasible_v2v_success_rate=None):  # 新增参数，默认 None 兼容旧代码
        """记录每个epoch的全局指标 """
        # 安全转换所有值为正确的类型
        cumulative_reward_float = self._convert_tensor_to_float(cumulative_reward)
        mean_loss_float = self._convert_tensor_to_float(mean_loss)
        mean_delay_float = self._convert_tensor_to_float(mean_delay)
        p95_delay_float = self._convert_tensor_to_float(p95_delay)
        mean_snr_float = self._convert_tensor_to_float(mean_snr)
        vehicle_count_int = self._convert_tensor_to_int(vehicle_count)
        v2v_success_rate_float = self._convert_tensor_to_float(v2v_success_rate)
        v2i_sum_capacity_float = self._convert_tensor_to_float(v2i_sum_capacity)

        v2v_delay_only_rate_float = self._convert_tensor_to_float(
            v2v_delay_only_rate if v2v_delay_only_rate is not None else 0.0)
        v2v_snr_only_rate_float = self._convert_tensor_to_float(
            v2v_snr_only_rate if v2v_snr_only_rate is not None else 0.0)

        # --- [NEW] 处理新参数 ---
        feasible_rate_float = self._convert_tensor_to_float(
            feasible_v2v_success_rate if feasible_v2v_success_rate is not None else 0.0)
        self.metrics['feasible_v2v_success_rate'].append(feasible_rate_float)
        # -----------------------

        self.metrics['epoch'].append(epoch)
        self.metrics['cumulative_reward'].append(cumulative_reward_float)
        self.metrics['mean_loss'].append(mean_loss_float)
        self.metrics['mean_delay'].append(mean_delay_float)
        self.metrics['p95_delay'].append(p95_delay_float)
        self.metrics['mean_snr'].append(mean_snr_float)
        self.metrics['vehicle_count'].append(vehicle_count_int)
        self.metrics['v2v_success_rate'].append(v2v_success_rate_float)
        self.metrics['v2i_sum_capacity'].append(v2i_sum_capacity_float)
        self.metrics['timestamp'].append(datetime.now())
        self.metrics['v2v_delay_only_rate'].append(v2v_delay_only_rate_float)
        self.metrics['v2v_snr_only_rate'].append(v2v_snr_only_rate_float)

        # 更新训练统计
        self.training_stats['total_epochs'] = epoch
        if cumulative_reward_float > self.training_stats['best_reward']:
            self.training_stats['best_reward'] = cumulative_reward_float
            self.training_stats['best_epoch'] = epoch

        self.logger.info(
            f"Epoch {epoch:4d} | "
            f"Loss: {mean_loss_float:7.4f} | "
            f"Reward: {cumulative_reward_float:8.3f} | "
            f"V2V Success: {v2v_success_rate_float:6.2%} (Feasible: {feasible_rate_float:6.2%}) | " # 打印日志
            f"V2I Cap: {v2i_sum_capacity_float:7.2f} Mbps | "
            f"Delay (Avg/P95): {mean_delay_float * 1000:6.2f}ms / {p95_delay_float * 1000:6.2f}ms | "  
            f"SNR: {mean_snr_float:6.2f}dB | "
            f"Vehicles: {vehicle_count_int:3d}"
        )


    def log_dqn_performance(self, dqn_id, metrics_dict):
        """记录单个DQN的性能指标 - 修复版"""
        # (此函数保持不变)
        safe_metrics = {}
        for metric_name, value in metrics_dict.items():
            if metric_name in ['vehicle_count']:
                safe_metrics[metric_name] = self._convert_tensor_to_int(value)
            else:
                safe_metrics[metric_name] = self._convert_tensor_to_float(value)

        for metric_name, value in safe_metrics.items():
            if metric_name in self.dqn_metrics[dqn_id]:
                self.dqn_metrics[dqn_id][metric_name].append(value)

        vehicle_count = safe_metrics.get('vehicle_count', 0)
        if isinstance(vehicle_count, (int, np.integer)):
            vehicle_format = f"{vehicle_count:2d}"
        else:
            vehicle_format = f"{int(vehicle_count):2d}"

        self.logger.debug(
            f"DQN {dqn_id:2d} | "
            f"Loss: {safe_metrics.get('loss', 0):7.4f} | "
            f"Reward: {safe_metrics.get('reward', 0):8.3f} | "
            f"Epsilon: {safe_metrics.get('epsilon', 0):5.3f} | "
            f"Vehicles: {vehicle_format}"
        )

    def log_convergence(self, epoch, final_loss):
        """记录收敛信息"""
        final_loss_float = self._convert_tensor_to_float(final_loss)
        self.training_stats['convergence_epoch'] = epoch
        self.training_stats['final_loss'] = final_loss_float

        self.logger.info(f" CONVERGENCE ACHIEVED at epoch {epoch} with final loss {final_loss_float:.6f}")

    def _ensure_consistent_array_lengths(self):
        """确保所有数组长度一致 - 修复长度不一致问题"""
        max_length = max(len(arr) for arr in self.metrics.values() if arr)

        for key in self.metrics:
            current_length = len(self.metrics[key])
            if current_length < max_length:
                fill_value = self.metrics[key][-1] if self.metrics[key] else 0
                self.metrics[key].extend([fill_value] * (max_length - current_length))
                self.logger.warning(f"Fixed array length for {key}: {current_length} -> {max_length}")

    def save_metrics_to_csv(self):
        """将指标保存为CSV文件 """
        try:
            self._ensure_consistent_array_lengths()

            global_df = pd.DataFrame(self.metrics)
            global_csv_path = f"{self.log_dir}/global_metrics.csv"
            global_df.to_csv(global_csv_path, index=False)
            self.logger.info(f"Global metrics saved to {global_csv_path}")

            dqn_count = 0
            for dqn_id, metrics in self.dqn_metrics.items():
                if metrics and metrics['loss']:  # 只保存有数据的DQN
                    # 确保DQN指标数组长度一致
                    max_dqn_length = max(len(arr) for arr in metrics.values() if arr)
                    for key in metrics:
                        if len(metrics[key]) < max_dqn_length:
                            fill_value = metrics[key][-1] if metrics[key] else 0
                            metrics[key].extend([fill_value] * (max_dqn_length - len(metrics[key])))

                    dqn_df = pd.DataFrame(metrics)
                    dqn_df['epoch'] = range(1, len(dqn_df) + 1)
                    dqn_csv_path = f"{self.log_dir}/dqn_{dqn_id}_metrics.csv"
                    dqn_df.to_csv(dqn_csv_path, index=False)
                    dqn_count += 1

            self.logger.info(f"Saved metrics for {dqn_count} DQNs to CSV files")

        except Exception as e:
            self.logger.error(f"Error saving CSV: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def generate_plots(self):
        """生成性能图表"""
        try:
            self._ensure_consistent_array_lengths()
            plt.style.use('default')
            sns.set_palette("husl")

            fig, axes = plt.subplots(4, 2, figsize=(15, 18))
            fig.suptitle('Training Performance Metrics', fontsize=16, fontweight='bold')

            if not self.metrics['epoch']:
                self.logger.warning("No data available for plotting")
                return

            epochs = np.array(self.metrics['epoch'])
            mean_loss = np.array(self.metrics['mean_loss'])
            cumulative_reward = np.array(self.metrics['cumulative_reward'])
            mean_delay = np.array(self.metrics['mean_delay'])
            p95_delay = np.array(self.metrics['p95_delay'])  # <--- 新增
            mean_snr = np.array(self.metrics['mean_snr'])
            vehicle_count = np.array(self.metrics['vehicle_count'])
            v2v_success_rate = np.array(self.metrics['v2v_success_rate'])
            v2i_sum_capacity = np.array(self.metrics['v2i_sum_capacity'])

            # 1. 损失曲线 (axes[0, 0])
            axes[0, 0].plot(epochs, mean_loss, 'b-', linewidth=2)
            axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            if len(mean_loss) > 0 and max(mean_loss) > min(mean_loss) * 10:
                axes[0, 0].set_yscale('log')

            # 2. 奖励曲线 (axes[0, 1])
            axes[0, 1].plot(epochs, cumulative_reward, 'g-', linewidth=2)
            axes[0, 1].set_title('Cumulative Reward', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 延迟曲线 (axes[1, 0]) - 更改为毫秒
            axes[1, 0].plot(epochs, mean_delay * 1000, 'r-', linewidth=2)
            axes[1, 0].set_title('Mean Delay (ms)', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Delay (ms)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim(bottom=0)

            # 4. SNR曲线 (axes[1, 1])
            axes[1, 1].plot(epochs, mean_snr, 'm-', linewidth=2)
            axes[1, 1].set_title('Mean SNR', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('SNR (dB)')
            axes[1, 1].grid(True, alpha=0.3)

            # 5. 车辆数量 (axes[2, 0])
            axes[2, 0].plot(epochs, vehicle_count, 'c-', linewidth=2)
            axes[2, 0].set_title('Vehicle Count', fontsize=14, fontweight='bold')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Number of Vehicles')
            axes[2, 0].grid(True, alpha=0.3)

            # 6. P95 延迟曲线 (替换 DQN Loss)
            axes[2, 1].plot(epochs, p95_delay * 1000, 'y-', linewidth=2)
            axes[2, 1].set_title('V2V P95 Latency (ms)', fontsize=14, fontweight='bold')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('P95 Delay (ms)')
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].set_ylim(bottom=0)


            # 7. V2V 成功概率 (axes[3, 0])
            axes[3, 0].plot(epochs, v2v_success_rate * 100, 'g-', label='Raw Success', linewidth=2)
            # --- [NEW] 绘制可行成功率 ---
            if 'feasible_v2v_success_rate' in self.metrics and self.metrics['feasible_v2v_success_rate']:
                feasible_rate = self._safe_numpy_conversion(self.metrics['feasible_v2v_success_rate'])
                # 使用红色虚线，代表“理论上限”或“修正后性能”
                axes[3, 0].plot(epochs, feasible_rate * 100, 'r--', label='Physically Feasible', linewidth=2)
            # --------------------------
            axes[3, 0].set_title('V2V Success Rate', fontsize=14, fontweight='bold')
            axes[3, 0].set_xlabel('Epoch')
            axes[3, 0].set_ylabel('Success Rate (%)')
            axes[3, 0].grid(True, alpha=0.3)
            axes[3, 0].set_ylim(0, 105)
            axes[3, 0].legend(loc='lower right')  # 显示图例

            # 8. V2I 和容量 (axes[3, 1])
            axes[3, 1].plot(epochs, v2i_sum_capacity, 'k-', linewidth=2)
            axes[3, 1].set_title('V2I Sum Capacity', fontsize=14, fontweight='bold')
            axes[3, 1].set_xlabel('Epoch')
            axes[3, 1].set_ylabel('Capacity (Mbps)')
            axes[3, 1].grid(True, alpha=0.3)
            axes[3, 1].set_ylim(bottom=0)

            plt.tight_layout()
            plot_path = f"{self.log_dir}/training_metrics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Performance plots saved to {plot_path}")

        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")
            import traceback
            self.logger.error(traceback.format_exc())



    def _safe_numpy_conversion(self, data):
        """安全转换为numpy数组"""
        if isinstance(data, list):
            return np.array(data)
        elif hasattr(data, 'detach'):
            return data.detach().numpy()
        elif hasattr(data, 'numpy'):
            return data.numpy()
        else:
            return np.array([float(x) for x in data])

    def _generate_report_content(self):
        """生成报告内容"""
        if not self.metrics['epoch']:
            return "# 训练报告\n\n暂无训练数据"

        try:
            mean_loss = self._safe_numpy_conversion(self.metrics['mean_loss'])
            cumulative_reward = self._safe_numpy_conversion(self.metrics['cumulative_reward'])
            mean_delay = self._safe_numpy_conversion(self.metrics['mean_delay'])
            p95_delay = self._safe_numpy_conversion(self.metrics['p95_delay'])
            mean_snr = self._safe_numpy_conversion(self.metrics['mean_snr'])
            vehicle_count = self._safe_numpy_conversion(self.metrics['vehicle_count'])
            v2v_success = self._safe_numpy_conversion(self.metrics['v2v_success_rate'])
            v2i_capacity = self._safe_numpy_conversion(self.metrics['v2i_sum_capacity'])

            content = f"""
# 强化学习训练报告

## 训练概览

- **训练开始时间**: {self.training_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}
- **总训练轮次**: {self.training_stats['total_epochs']}
- **最佳奖励**: {self.training_stats['best_reward']:.4f} (Epoch {self.training_stats['best_epoch']})
- **收敛轮次**: {self.training_stats.get('convergence_epoch', 'N/A')}
- **最终损失**: {mean_loss[-1]:.6f}

## 性能指标汇总

| 指标 | 平均值 | 最小值 | 最大值 | 最终值 | 标准差 |
|------|--------|--------|--------|---------|--------|
| 损失 | {np.mean(mean_loss):.4f} | {np.min(mean_loss):.4f} | {np.max(mean_loss):.4f} | {mean_loss[-1]:.4f} | {np.std(mean_loss):.4f} |
| 奖励 | {np.mean(cumulative_reward):.4f} | {np.min(cumulative_reward):.4f} | {np.max(cumulative_reward):.4f} | {cumulative_reward[-1]:.4f} | {np.std(cumulative_reward):.4f} |
| 平均延迟(ms) | {np.mean(mean_delay * 1000):.2f} | {np.min(mean_delay * 1000):.2f} | {np.max(mean_delay * 1000):.2f} | {mean_delay[-1] * 1000:.2f} | {np.std(mean_delay * 1000):.2f} |
| P95 延迟(ms) | {np.mean(p95_delay * 1000):.2f} | {np.min(p95_delay * 1000):.2f} | {np.max(p95_delay * 1000):.2f} | {p95_delay[-1] * 1000:.2f} | {np.std(p95_delay * 1000):.2f} |
| SNR(dB) | {np.mean(mean_snr):.2f} | {np.min(mean_snr):.2f} | {np.max(mean_snr):.2f} | {mean_snr[-1]:.2f} | {np.std(mean_snr):.2f} |
| 车辆数 | {np.mean(vehicle_count):.1f} | {np.min(vehicle_count):.0f} | {np.max(vehicle_count):.0f} | {vehicle_count[-1]:.0f} | {np.std(vehicle_count):.1f} |
| V2V成功率 | {np.mean(v2v_success):.2%} | {np.min(v2v_success):.2%} | {np.max(v2v_success):.2%} | {v2v_success[-1]:.2%} | {np.std(v2v_success):.3f} |
| V2I容量(Mbps) | {np.mean(v2i_capacity):.2f} | {np.min(v2i_capacity):.2f} | {np.max(v2i_capacity):.2f} | {v2i_capacity[-1]:.2f} | {np.std(v2i_capacity):.2f} |
## 训练曲线



## DQN性能分析

"""
        # --- END MODIFIED ---
        except Exception as e:
            content = f"# 训练报告\n\n错误生成统计信息: {e}\n\n"

            for dqn_id in sorted(self.dqn_metrics.keys()):
                metrics = self.dqn_metrics[dqn_id]
                if metrics and metrics['loss']:
                    try:
                        dqn_loss = self._safe_numpy_conversion(metrics['loss'])
                        dqn_reward = self._safe_numpy_conversion(metrics['reward'])
                        dqn_vehicle_count = self._safe_numpy_conversion(metrics['vehicle_count'])
                        dqn_snr = self._safe_numpy_conversion(metrics['snr'])
                        dqn_delay = self._safe_numpy_conversion(metrics['delay'])

                        # 修复epsilon值的处理
                        epsilon_value = 0.0
                        if metrics['epsilon']:
                            epsilon_value = metrics['epsilon'][-1] if isinstance(metrics['epsilon'][-1],
                                                                                 (int, float)) else 0.0

                        content += f"""
        ### DQN {dqn_id}
        - **平均损失**: {np.mean(dqn_loss):.4f}
        - **最终损失**: {dqn_loss[-1]:.4f}
        - **平均奖励**: {np.mean(dqn_reward):.3f}
        - **服务车辆数**: {np.mean(dqn_vehicle_count):.1f}
        - **最终ε值**: {epsilon_value:.3f}
        - **平均SNR**: {np.mean(dqn_snr):.2f} dB
        - **平均延迟**: {np.mean(dqn_delay):.6f} s

        """
                    except Exception as e:
                        content += f"""
        ### DQN {dqn_id}
        - **错误**: 无法计算统计信息 ({str(e)})

        """

            content += """
        ## 收敛分析

        """

            if self.training_stats.get('convergence_epoch'):
                content += f"- 模型在 **Epoch {self.training_stats['convergence_epoch']}** 收敛\n"
                content += f"- 最终损失: {self.training_stats['final_loss']:.6f}\n"
            else:
                content += "- 模型未完全收敛，建议继续训练或调整超参数\n"

            content += f"""
        ## 建议

        基于当前训练结果，建议：
        1. {'继续优化奖励函数' if self.metrics['cumulative_reward'] and self.metrics['cumulative_reward'][-1] < 0 else '奖励函数设计良好'}
        2. {'调整学习率或探索策略' if self.metrics['mean_loss'] and np.std(np.array(self.metrics['mean_loss'])) > 1.0 else '训练过程稳定'}
        3. {'检查信道模型参数' if self.metrics['mean_snr'] and np.mean(np.array(self.metrics['mean_snr'])) < 10 else 'SNR性能良好'}

        ---

        *报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
        """

        return content


    def finalize(self):
        """完成训练，保存所有结果"""
        try:
            self.save_metrics_to_csv()
            self.generate_plots()
            self._generate_report_content()

            training_time = (datetime.now() - self.training_stats['start_time']).total_seconds()
            self.logger.info(f"Training completed in {training_time:.2f} seconds. All results saved to {self.log_dir}")

        except Exception as e:
            self.logger.error(f"Error during finalization: {e}")
            import traceback
            self.logger.error(traceback.format_exc())


def debug_print(msg):
    global_logger.logger.info(msg)


def debug(msg):
    global_logger.logger.debug(msg)


def set_debug_mode(mode):
    if mode:
        global_logger.logger.setLevel(logging.DEBUG)
    else:
        global_logger.logger.setLevel(logging.INFO)


# 全局日志实例
global_logger = TrainingLogger()