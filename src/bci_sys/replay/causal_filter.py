"""流式因果 1–30Hz 带通（实时路径专用，C3）。

【严禁 filtfilt / 零相位】零相位滤波要用到未来样本，只能出现在 offline/。
本模块只用单向 scipy.signal.lfilter：4 阶 Butterworth 高通（1Hz）串 4 阶
低通（30Hz），每节、每通道各自携带 zi 跨块延续。初始 zi=0，因此
50 点一块的流式输出与整段一次性同参数单向 lfilter 逐点一致（测试锁定）。
零初始状态造成的启动瞬态由 QC 校准时剔除前 10s 解决。
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .. import config


def design_filter_coeffs(
    sfreq: float = config.SFREQ,
    band_hz: tuple[float, float] = config.FILTER_BAND_HZ,
):
    """返回 ((b_hp, a_hp), (b_lp, a_lp))。频带冻结 1–30Hz。"""
    lo, hi = band_hz
    b_hp, a_hp = signal.butter(4, lo / (sfreq / 2.0), btype="high")
    b_lp, a_lp = signal.butter(4, hi / (sfreq / 2.0), btype="low")
    return (b_hp, a_hp), (b_lp, a_lp)


def bandpass_oneshot(
    x: np.ndarray,
    sfreq: float = config.SFREQ,
    band_hz: tuple[float, float] = config.FILTER_BAND_HZ,
) -> np.ndarray:
    """整段一次性单向带通（零初始 zi）。仅作流式实现的参照基准/非实时用途。"""
    (b_hp, a_hp), (b_lp, a_lp) = design_filter_coeffs(sfreq, band_hz)
    y = signal.lfilter(b_hp, a_hp, np.asarray(x, dtype=float))
    return signal.lfilter(b_lp, a_lp, y)


class _CausalSection:
    """单节单向 lfilter，zi 跨块延续。"""

    def __init__(self, b: np.ndarray, a: np.ndarray):
        self.b, self.a = b, a
        self.zi = np.zeros(max(len(a), len(b)) - 1)

    def push(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = signal.lfilter(
            self.b, self.a, np.asarray(x, dtype=float), zi=self.zi
        )
        return y


class CausalBandpass:
    """逐块推进的因果带通：先高通后低通串联，每通道独立状态。

    用法：
        bp = CausalBandpass(("Fz", "FP1", "FP2"))
        filtered = bp.push_block({"Fz": 块, "FP1": 块, "FP2": 块})
    """

    def __init__(
        self,
        channels: tuple[str, ...] | list[str],
        sfreq: float = config.SFREQ,
        band_hz: tuple[float, float] = config.FILTER_BAND_HZ,
    ):
        self.channels = tuple(channels)
        (b_hp, a_hp), (b_lp, a_lp) = design_filter_coeffs(sfreq, band_hz)
        self._hp = {c: _CausalSection(b_hp, a_hp) for c in self.channels}
        self._lp = {c: _CausalSection(b_lp, a_lp) for c in self.channels}

    def push_block(self, block: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """所有通道块长必须一致；返回同结构的滤波后块（单位与输入相同，µV）。"""
        sizes = {np.asarray(block[c]).size for c in self.channels}
        if len(sizes) != 1:
            raise ValueError(f"各通道块长不一致: {sizes}")
        out: dict[str, np.ndarray] = {}
        for c in self.channels:
            y = self._hp[c].push(block[c])
            out[c] = self._lp[c].push(y)
        return out
