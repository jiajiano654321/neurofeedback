"""离线 theta 管线（C3 离线路径：允许零相位滤波，禁止与 replay 共用处理逻辑）。

【硬约束】处理顺序固定：读全程 → 全程零相位滤波 → 再切 epoch → 谱估计。
1-30Hz FIR 长 825 点(3.3s) > epoch 375 点(1.5s)，先切后滤会补零废掉 theta 估计。
"""
from __future__ import annotations

import numpy as np

from .. import config


def filter_whole(x: np.ndarray, sfreq: float = config.SFREQ) -> np.ndarray:
    """全程 1-30Hz 零相位 FIR 带通。必须在切 epoch 之前对整段信号调用。"""
    import mne

    lo, hi = config.FILTER_BAND_HZ
    return mne.filter.filter_data(
        x.astype(np.float64), sfreq, lo, hi, method="fir", verbose="ERROR"
    )


def slice_epochs(
    x_filtered: np.ndarray,
    onsets_sec: np.ndarray,
    sfreq: float = config.SFREQ,
    epoch_sec: float = config.EPOCH_SEC,
) -> np.ndarray:
    """刺激 onset 起 [0, epoch_sec) 窗口切 epoch；超出信号末尾的不完整窗口丢弃。"""
    n = int(round(epoch_sec * sfreq))
    out = []
    for t in onsets_sec:
        i = int(round(float(t) * sfreq))
        if i + n <= x_filtered.size:
            out.append(x_filtered[i : i + n])
    return np.asarray(out, dtype=np.float64)


def theta_power_uV2(
    epochs: np.ndarray,
    sfreq: float = config.SFREQ,
    band: tuple[float, float] = config.THETA_BAND_HZ,
    nw: float = config.MULTITAPER_NW,
) -> np.ndarray:
    """每 epoch 的 theta(4-8Hz) 功率，单位 µV²（输入 µV）。multitaper，NW=2。"""
    from mne.time_frequency import psd_array_multitaper

    epochs = np.atleast_2d(epochs)
    t_epoch = epochs.shape[1] / sfreq
    # mne 没有 NW 参数，要的是 bandwidth（全带宽，Hz）。
    # mne 内部：half_nbw(NW) = bandwidth * n_times / (2*sfreq) = bandwidth*T/2
    # → bandwidth = 2*NW/T。NW=2、T=1.5s → bandwidth = 2.667 Hz（不是 NW/T=1.333）。
    bandwidth = 2.0 * nw / t_epoch
    psd, freqs = psd_array_multitaper(
        epochs, sfreq, fmin=band[0], fmax=band[1],
        bandwidth=bandwidth, normalization="full", verbose="ERROR",
    )
    return np.trapezoid(psd, freqs, axis=1)
