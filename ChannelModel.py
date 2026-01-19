# -*- coding: utf-8 -*-
import numpy as np
import torch
from logger import debug, debug_print
from Parameters import (
    CENTER_FREQUENCY, ANTENNA_HEIGHT_BS, ANTENNA_HEIGHT_UE,
    PATH_LOSS_A, PATH_LOSS_B, PATH_LOSS_C, SHADOWING_STD,
    SYSTEM_BANDWIDTH, NOISE_POWER_DENSITY, BOLTZMANN_CONSTANT,
    NOISE_TEMPERATURE
)

class UMiNLOSChannel:

    def __init__(self):
        # 毫米波频段参数
        self.center_frequency = CENTER_FREQUENCY
        self.antenna_height_bs = ANTENNA_HEIGHT_BS
        self.antenna_height_ue = ANTENNA_HEIGHT_UE

        # 3GPP UMi NLOS 路径损耗模型参数
        self.path_loss_A = PATH_LOSS_A
        self.path_loss_B = PATH_LOSS_B
        self.path_loss_C = PATH_LOSS_C
        self.shadowing_std = SHADOWING_STD

        # 系统带宽 (毫米波典型带宽)
        self.system_bandwidth = SYSTEM_BANDWIDTH

        # 噪声参数
        self.noise_power_density = NOISE_POWER_DENSITY
        self.boltzmann_constant = BOLTZMANN_CONSTANT
        self.temperature = NOISE_TEMPERATURE

        debug("UMiNLOSChannel initialized with 28GHz UMi NLOS model")

    def _calculate_noise_power(self, bandwidth):
        """计算指定带宽下的噪声功率"""
        # 方法1: 使用玻尔兹曼常数计算
        noise_power_linear = (self.boltzmann_constant * self.temperature *
                              bandwidth)

        # 方法2的日志记录也应使用 bandwidth
        noise_power_dbm = (self.noise_power_density +
                           10 * np.log10(bandwidth))
        noise_power_linear_alt = 10 ** ((noise_power_dbm - 30) / 10)

        debug(f"Noise power for {bandwidth/1e6}MHz: {noise_power_linear:.2e} W")
        return noise_power_linear

    def calculate_3d_distance(self, pos_tx, pos_rx):
        """
        计算包含天线高度差的3D距离

        Args:
            pos_tx: 发射机位置 (x, y)
            pos_rx: 接收机位置 (x, y)

        Returns:
            distance_3d: 3D距离 (m)
        """
        dx = pos_tx[0] - pos_rx[0]
        dy = pos_tx[1] - pos_rx[1]
        d_2d = np.sqrt(dx ** 2 + dy ** 2)
        d_3d = np.sqrt(d_2d ** 2 + (self.antenna_height_bs - self.antenna_height_ue) ** 2)

        debug(f"2D distance: {d_2d:.2f}m, 3D distance: {d_3d:.2f}m")
        return d_3d

    def calculate_path_loss(self, distance_3d):
        """
        计算28GHz毫米波频段的3GPP UMi NLOS路径损耗

        Args:
            distance_3d: 3D距离 (m)

        Returns:
            total_pl_db: 总路径损耗 (dB)
            pl_deterministic: 确定性路径损耗 (dB)
            shadowing: 阴影衰落分量 (dB)
        """
        if distance_3d <= 0:
            raise ValueError("Distance must be positive")

        fc_ghz = self.center_frequency / 1e9  # 转换为GHz
        h_ut = self.antenna_height_ue

        # 计算确定性路径损耗 (公式: PL = 35.3*log10(d_3d) + 22.4 + 21.3*log10(fc) - 0.3*(h_UT - 1.5))
        pl_deterministic = (self.path_loss_A * np.log10(distance_3d) +
                            self.path_loss_B +
                            self.path_loss_C * np.log10(fc_ghz) -
                            0.3 * (h_ut - 1.5))

        # 生成阴影衰落 (对数正态分布)
        shadowing = np.random.normal(0, self.shadowing_std)

        total_pl_db = pl_deterministic + shadowing

        debug(f"Path loss - Deterministic: {pl_deterministic:.2f}dB, "
              f"Shadowing: {shadowing:.2f}dB, Total: {total_pl_db:.2f}dB")

        return total_pl_db, pl_deterministic, shadowing

    def calculate_snr(self, tx_power, distance_3d, beamforming_gain=0, bandwidth=None):
        """
        计算接收信噪比 (SNR)

        Args:
            tx_power: 发射功率 (W)
            distance_3d: 3D距离 (m)
            beamforming_gain: 波束成形增益 (dB)
            bandwidth: (可选) 计算SNR所用的带宽 (Hz)
        """

        # --- MODIFIED: 动态计算噪声功率 ---
        if bandwidth is None:
            bandwidth = self.system_bandwidth  # 默认使用 V2I 的 400MHz

        noise_power = self._calculate_noise_power(bandwidth)
        # 计算总路径损耗
        total_pl_db, _, shadowing = self.calculate_path_loss(distance_3d)
        total_pl_linear = 10 ** (-total_pl_db / 10)  # 转换为线性值

        # 考虑波束成形增益
        effective_pl_linear = total_pl_linear * 10 ** (-beamforming_gain / 10)

        # 计算接收功率
        received_power = tx_power * effective_pl_linear

        # 计算SNR
        snr_linear = received_power / noise_power

        # <<< --- 添加数值稳定性处理 --- >>>
        epsilon = 1e-20  # 定义一个极小的正数
        snr_linear = max(snr_linear, epsilon)  # 确保 snr_linear 至少为 epsilon

        # 现在 snr_linear 保证大于 0
        snr_db = 10 * np.log10(snr_linear)  # 可以直接计算 log10
        # snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -float('inf')

        debug(f"SNR calc (BW={bandwidth / 1e6}MHz) - TxPwr: {tx_power}W, "
              f"RxPwr: {received_power:.2e}W, NoisePwr: {noise_power:.2e}W, SNR: {snr_db:.2f}dB")

        return snr_db, snr_linear, received_power

    def get_channel_state_info(self, pos_tx, pos_rx, tx_power, beamforming_gain=0, bandwidth=None):
        """
        获取完整的信道状态信息 (CSI)
        """
        if bandwidth is None:
            bandwidth = self.system_bandwidth  # 默认

        distance_3d = self.calculate_3d_distance(pos_tx, pos_rx)
        total_pl_db, pl_deterministic, shadowing = self.calculate_path_loss(distance_3d)

        snr_db, snr_linear, received_power = self.calculate_snr(
            tx_power, distance_3d, beamforming_gain, bandwidth=bandwidth)

        csi_info = {
            'distance_3d': distance_3d,
            'path_loss_total_db': total_pl_db,
            'path_loss_deterministic_db': pl_deterministic,
            'shadowing_db': shadowing,
            'snr_db': snr_db,
            'snr_linear': snr_linear,
            'received_power': received_power,
            'is_los': False,  # UMi NLOS 模型
            'frequency': self.center_frequency
        }

        return csi_info


# 全局信道模型实例
global_channel_model = UMiNLOSChannel()


def test_channel_model():
    """测试信道模型"""
    debug_print("Testing UMi NLOS Channel Model...")

    # 测试用例
    test_cases = [
        ((0, 0), (100, 0)),  # 100m水平距离
        ((0, 0), (500, 0)),  # 500m水平距离
        ((0, 0), (0, 100)),  # 100m垂直距离
    ]

    for pos_tx, pos_rx in test_cases:
        debug_print(f"\nTesting TX {pos_tx} -> RX {pos_rx}:")
        csi = global_channel_model.get_channel_state_info(
            pos_tx, pos_rx, tx_power=1.0)  # 1W发射功率

        for key, value in csi.items():
            debug_print(f"  {key}: {value}")


if __name__ == "__main__":
    set_debug_mode(True)
    test_channel_model()