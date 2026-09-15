"""θ 反馈指标纯逻辑（无 I/O、无墙钟；C3：只允许单向 lfilter，严禁 filtfilt）。

- 因果 4–8Hz：4 阶 Butterworth hp(4) 串 lp(8)（与 IT3 滤波器同构）；
- 校准窗以 zi=0 一次性滤波并取回末端 zf，运行期逐块续接
  （流式 θ 与整段一次性单向滤波逐点相等，测试锁定）；
- 段功率＝段内 50 点平方均值（µV²；mne 输出已是 µV，严禁 ×1e6）；
- W=5【仅有效段】尾窗均值：无效/非有限段不入窗、输出 None（C4/C5）；
- S3 P20 一次性冻结（numpy 2.0 method="linear"）；direction 显式无默认（C1）。
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy import signal

from .. import config
from . import causal_filter


class FreezeFailed(Exception):
    """S3 冻结条件不满足（成熟有效段不足 / 阈值非有限），硬停不进 S4（C7）。"""


def segment_powers(
    theta_uV: np.ndarray, seg_samples: int = config.SEGMENT_SAMPLES
) -> np.ndarray:
    """整段 θ 序列切成 50 点块，逐块 mean(x²)；末尾不足一块的尾巴不参与。"""
    x = np.asarray(theta_uV, dtype=float)
    n = x.size // seg_samples
    r = x[: n * seg_samples].reshape(n, seg_samples)
    return np.mean(r * r, axis=1)


class ValidOnlySmoother:
    """最近 W 个【有效且有限】段功率的尾窗均值。

    无效段/非有限功率：不入窗、不动窗、返回 None（反馈暂停，C5）；
    窗内有效段不足 W 个（未成熟）同样返回 None。
    """

    def __init__(self, win: int = config.THETA_SMOOTH_SEGS):
        win = int(win)
        if win < 1:
            raise ValueError(f"win 必须 >=1，收到 {win}")
        self.win = win
        self._q: deque[float] = deque(maxlen=win)

    def update(self, power_uV2: float, valid: bool) -> float | None:
        if not valid or not np.isfinite(power_uV2):
            return None
        self._q.append(float(power_uV2))
        if len(self._q) < self.win:
            return None
        return float(np.mean(self._q))


class HitDecider:
    """smoothed 与冻结阈值比较。C1：direction 必填，只允许 'up'/'down'，无默认。"""

    def __init__(self, threshold_uV2: float, direction: str):
        if not isinstance(direction, str) or direction not in ("up", "down"):
            raise ValueError(f"direction 只能是 'up'/'down'，收到 {direction!r}")
        thr = float(threshold_uV2)
        if not np.isfinite(thr):
            raise ValueError("阈值非有限值，无法构造达标判定")
        self.threshold_uV2 = thr
        self.direction = direction

    def hit(self, smoothed_uV2: float) -> bool:
        """调用契约：s 必须有限；smoothed 为 None（未成熟/无效段，C5）时调用方
        必须直接跳过本方法——该段不产生达标判定，而不是把 None 传进来。"""
        if self.direction == "up":
            return bool(smoothed_uV2 >= self.threshold_uV2)
        return bool(smoothed_uV2 <= self.threshold_uV2)


def calibrate_theta(
    cal_clean_uV: np.ndarray,
    cal_valid: np.ndarray,
    *,
    sfreq: float = config.SFREQ,
    band_hz: tuple[float, float] = config.THETA_BAND_HZ,
    seg_samples: int = config.SEGMENT_SAMPLES,
    warmup_sec: float = config.QC_FILTER_WARMUP_SEC,
    win: int = config.THETA_SMOOTH_SEGS,
    quantile: float = config.THETA_QUANTILE,
    min_mature_segs: int = config.THETA_MIN_CALIB_SEGS,
) -> dict:
    """冻结瞬间对 S3 校准序列一次性重建 θ 链路（spec §4.2 步骤 3–6）。

    cal_clean_uV: 30000 点 Fz_clean（1–30Hz 因果滤波并去眼电后）；
    cal_valid: 600 个逐段 QC 有效性（与运行期同一 L1/L3 规则重算）。
    以 zi=0 单向 4–8Hz 滤波，取回 hp/lp 两节末端 zf 供运行期续接。

    返回 dict：theta_thr_uV2 / n_thr_segs / calib_hit_rate /
    cal_powers(600,) / cal_smoothed(list[float|None], 600) / zf=(zf_hp,zf_lp)。
    成熟有效段不足 min_mature_segs 或阈值非有限 → FreezeFailed。
    """
    x = np.asarray(cal_clean_uV, dtype=float)
    valid = np.asarray(cal_valid, dtype=bool)
    n_seg = x.size // seg_samples
    if valid.shape != (n_seg,):
        raise ValueError(f"cal_valid 形状 {valid.shape} 与段数 {n_seg} 不符")

    (b_hp, a_hp), (b_lp, a_lp) = causal_filter.design_filter_coeffs(sfreq, band_hz)
    zi_hp = np.zeros(max(len(a_hp), len(b_hp)) - 1)
    zi_lp = np.zeros(max(len(a_lp), len(b_lp)) - 1)
    theta, zf_hp = signal.lfilter(b_hp, a_hp, x, zi=zi_hp)
    theta, zf_lp = signal.lfilter(b_lp, a_lp, theta, zi=zi_lp)

    powers = segment_powers(theta, seg_samples)
    smoother = ValidOnlySmoother(win)
    cal_smoothed: list[float | None] = []
    mature: list[float] = []
    for k in range(n_seg):
        s = smoother.update(float(powers[k]), bool(valid[k]))
        t1 = (k + 1) * seg_samples / sfreq
        if s is None or t1 <= warmup_sec:
            cal_smoothed.append(None)
        else:
            cal_smoothed.append(s)
            mature.append(s)

    if len(mature) < int(min_mature_segs):
        raise FreezeFailed(
            f"S3 成熟有效段 {len(mature)} < 最小要求 {min_mature_segs}，冻结失败"
        )
    E = np.asarray(mature, dtype=float)
    thr = float(np.quantile(E, quantile, method="linear"))
    if not np.isfinite(thr):
        raise FreezeFailed("S3 θ 阈值非有限值，冻结失败")
    return {
        "theta_thr_uV2": thr,
        "n_thr_segs": len(mature),
        "calib_hit_rate": float(np.mean(E >= thr)),
        "cal_powers": powers,
        "cal_smoothed": cal_smoothed,
        "zf": (np.asarray(zf_hp), np.asarray(zf_lp)),
    }


class ThetaStream:
    """运行期逐段 4–8Hz 因果滤波 + 段功率，zi 跨段延续。

    zf 取 calibrate_theta 返回的校准末端两节状态时，逐块流式输出与对
    clean[0:840s] 以 zi=0 整段一次性单向滤波的对应样本逐点相等。
    """

    def __init__(
        self,
        zf: tuple[np.ndarray, np.ndarray] | None = None,
        *,
        sfreq: float = config.SFREQ,
        band_hz: tuple[float, float] = config.THETA_BAND_HZ,
    ):
        zi = (None, None) if zf is None else zf
        self._f = causal_filter.CausalChannel(sfreq=sfreq, band_hz=band_hz, zi=zi)

    def push_segment(self, clean_uV: np.ndarray) -> tuple[np.ndarray, float]:
        """返回 (本段 θ 样本, 段功率 µV²)。"""
        y = self._f.push(np.asarray(clean_uV, dtype=float))
        return y, float(np.mean(y * y))

    def states(self) -> tuple[np.ndarray, np.ndarray]:
        return self._f.states()
