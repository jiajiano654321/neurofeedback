"""唯一常量出处。任何信号处理/流程参数只在这里定义。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- 采样与分段 ---
SFREQ = 250.0
SEGMENT_SEC = 0.2
SEGMENT_SAMPLES = 50  # 250 Hz × 0.2 s

# --- theta / epoch（IT2 离线） ---
THETA_BAND_HZ = (4.0, 8.0)
EPOCH_SEC = 1.5
MULTITAPER_NW = 2.0
FILTER_BAND_HZ = (1.0, 30.0)  # 全程零相位带通（离线 theta 管线 + preflight 单位诊断共用）

# --- 被试名单 ---
RECOMMENDED_SUBS = ["001", "002", "006", "014", "015", "018", "019"]
EXTREME_SUB = "012"  # 留作极端测试，本期不跑

# --- 通道（名字大小写以 channels.tsv 为准） ---
EOG_PROXY_CHANNELS = ("FP1", "FP2")  # IT3 才用，先登记

# --- 速率显示颜色阈值（data_elapsed / wall_elapsed） ---
RATE_GREEN = 0.98
RATE_YELLOW = 0.95

# --- preflight 单位硬检查：1-30Hz 带通后 Fz std 的允许范围（µV） ---
UNIT_STD_RANGE_UV = (5.0, 300.0)


@dataclass(frozen=True)
class Target:
    """训练靶点。C1：direction 必填、无默认，构造时不写就报错。"""

    channel: str
    band_hz: tuple[float, float]
    direction: str  # 必须显式传 "up" 或 "down"，无默认值

    def __post_init__(self):
        if not isinstance(self.direction, str):
            raise ValueError("direction 必须是字符串 'up'/'down'，不许留空")
        if self.direction not in ("up", "down"):
            raise ValueError(f"direction 只能是 'up'/'down'，收到 {self.direction!r}")


# 本期靶点：theta 4-8 Hz @ Fz，方向上调（显式写出，C1）
TARGET = Target(channel="Fz", band_hz=THETA_BAND_HZ, direction="up")


def data_root() -> Path:
    """ds007169 根目录。默认项目内 ds007169/，可用 BCI_DATA_ROOT 覆盖。"""
    env = os.environ.get("BCI_DATA_ROOT")
    if env:
        return Path(env)
    # src/bci_sys/config.py → parents[2] = 仓库根
    return Path(__file__).resolve().parents[2] / "ds007169"
