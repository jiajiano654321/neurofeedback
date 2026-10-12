"""θ（theta）反馈指标——「这一段该不该给奖励」的全部纯逻辑。

它在 QC 之后接手，每 200ms 段的处理链：

    Fz_clean（QC 扣完眼电的 1–30Hz 信号）
        │  再串一节因果 4–8Hz 带通（θ 频带；结构与 causal_filter 同构，仍严禁 filtfilt）
        ▼
      段功率 = 段内 50 点平方均值 mean(θ²)，单位 µV²
        │  （mne 读出的数已经是 µV，严禁再 ×1e6——错乘不报错，功率虚大 10¹² 倍）
        ▼
      ValidOnlySmoother：最近 5 个【有效段】功率的均值 = smoothed
        │  坏段/NaN 段不入窗、不输出（反馈暂停，C4/C5）
        ▼
      HitDecider：smoothed ≥ S3 冻结阈值？（direction="up"，C1 显式无默认）
        │
        ▼
      hit=True → 进度条涨；hit=False → 不涨；None（坏段）→ 进度条冻结

两个时刻，两套用法：
    S3 校准结束（一次性）：calibrate_theta() —— 对 120s 校准序列整段单向
        4–8Hz 滤波、重算 600 段功率与平滑，取 P20 分位冻成阈值，并导出滤波器
        末端状态 zf。
    S4 运行期（逐段）：ThetaStream 从 zf 续接，逐段滤波出功率；
        ValidOnlySmoother + HitDecider 逐段判 hit。
    「整段滤到 120s 再续着逐段滤」与「0→840s 整段一次性滤」逐点相等（测试锁定）。

本模块与 qc.py 同级：纯逻辑、无 I/O、无墙钟、无随机，是 A4 字节确定性的一环。
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy import signal

from .. import config
from . import causal_filter


class FreezeFailed(Exception):
    """S3 冻结条件不满足（成熟有效段不足 50 / 阈值算出 NaN）。

    硬停：会话进 aborted(freeze_failed)，不许带着不可信的阈值进 S4（C7）。
    """


def segment_powers(
    theta_uV: np.ndarray, seg_samples: int = config.SEGMENT_SAMPLES
) -> np.ndarray:
    """把一整段 θ 序列按 50 点切块，逐块算功率 mean(x²)（µV²）。

    末尾不足 50 点的尾巴不参与（与分段器「残尾不发段」口径一致）。
    """
    x = np.asarray(theta_uV, dtype=float)
    n = x.size // seg_samples
    r = x[: n * seg_samples].reshape(n, seg_samples)
    return np.mean(r * r, axis=1)


class ValidOnlySmoother:
    """最近 W 个【有效且数值有限】段功率的尾窗均值（默认 W=5，即 1 秒）。

    为什么要平滑：单段 200ms 的功率抖动极大，直接拿去比阈值会让进度条疯狂
    横跳；取 1 秒尾窗均值后反馈才稳定。W=5 是实测选型（更长的窗会被慢漂移
    主导，block 间达标率摆幅变大）。

    为什么只让有效段进窗：伪迹段的功率是被污染的数，让它进反馈指标等于
    间接奖励眨眼，违反 C4。所以坏段的待遇是「窗不动、输出 None」，前端仪表
    保持上一值并显示暂停横幅（C5：不奖也不罚）。
    窗里有效段还不满 W 个（开机未成熟）时同样输出 None。
    """

    def __init__(self, win: int = config.THETA_SMOOTH_SEGS):
        win = int(win)
        if win < 1:
            raise ValueError(f"win 必须 >=1，收到 {win}")
        self.win = win
        self._q: deque[float] = deque(maxlen=win)

    def update(self, power_uV2: float, valid: bool) -> float | None:
        """喂入本段功率与 QC 结论，返回平滑值；无效/未成熟返回 None。"""
        if not valid or not np.isfinite(power_uV2):
            return None
        self._q.append(float(power_uV2))
        if len(self._q) < self.win:
            return None
        return float(np.mean(self._q))


class HitDecider:
    """把平滑功率与冻结阈值比大小，决定本段是否「达标」。

    C1：direction 必填且只接受 'up'/'down'，没有默认值。训练方向在本项目
    历史上被搞反过两次——反了不报错、不崩溃、数据看着完全正常，所以用构造
    期硬校验兜底。本期靶点 direction='up'：θ 功率越高越好，hit = 功率 ≥ 阈值。
    """

    def __init__(self, threshold_uV2: float, direction: str):
        if not isinstance(direction, str) or direction not in ("up", "down"):
            raise ValueError(f"direction 只能是 'up'/'down'，收到 {direction!r}")
        thr = float(threshold_uV2)
        if not np.isfinite(thr):
            raise ValueError("阈值非有限值，无法构造达标判定")
        self.threshold_uV2 = thr
        self.direction = direction

    def hit(self, smoothed_uV2: float) -> bool:
        """调用契约：入参必须是有限数。smoothed 为 None（未成熟/坏段）时由
        调用方直接跳过本方法——坏段的语义是「不产生达标判定」，不是把 None
        拿来比大小。"""
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
    """S3 冻结瞬间一次性做的事：从 120s 校准数据算出 θ 阈值（spec §4.2 步骤 3–6）。

    入参：
      cal_clean_uV : 30000 点 Fz_clean——已经过 1–30Hz 因果带通并扣完眼电；
      cal_valid   : 600 段的 QC 有效性（用与运行期完全相同的 L1/L3 规则重算，
                     保证阈值建立时对「好坏段」的口径和 S4 运行期一致）。

    步骤：
      1. zi=0 对校准序列整段单向 4–8Hz 滤波（高通串低通），取回两节末端 zf，
         供运行期 ThetaStream 逐点续接；
      2. 600 段功率 → W=5 仅有效段平滑；
      3. 成熟段集合 E = t1>10s（滤掉滤波启动期）且平滑已成熟（窗满 5 段）；
      4. 阈值 = E 的 P20 分位（numpy 2.0 线性插值）。P20 的含义：约 80% 的
         校准段功率在阈值之上——即 S4 开始时反馈进度条天然有 ~80% 时间在涨，
         这是「让人有东西可学」的初始难度，不是合格线；THETA_TARGET_RATE
         只记录不参与控制。阈值运行时现算，sub-001 实测 16.77µV² 仅为参照，
         代码里没有这个数。

    返回：theta_thr_uV2 / n_thr_segs（参与定阈的成熟段数）/ calib_hit_rate
    （校准段达标率，纯描述）/ cal_powers(600,) / cal_smoothed(600 个
    float|None，写 calib_seg 行用）/ zf=(zf_hp, zf_lp)。

    成熟有效段不足 min_mature_segs（默认 50）或阈值非有限 → FreezeFailed。
    """
    x = np.asarray(cal_clean_uV, dtype=float)
    valid = np.asarray(cal_valid, dtype=bool)
    n_seg = x.size // seg_samples
    if valid.shape != (n_seg,):
        raise ValueError(f"cal_valid 形状 {valid.shape} 与段数 {n_seg} 不符")

    # 第 1 步：校准窗整段单向滤波，zi=0，滤完取出两节末端状态。
    (b_hp, a_hp), (b_lp, a_lp) = causal_filter.design_filter_coeffs(sfreq, band_hz)
    zi_hp = np.zeros(max(len(a_hp), len(b_hp)) - 1)
    zi_lp = np.zeros(max(len(a_lp), len(b_lp)) - 1)
    theta, zf_hp = signal.lfilter(b_hp, a_hp, x, zi=zi_hp)
    theta, zf_lp = signal.lfilter(b_lp, a_lp, theta, zi=zi_lp)

    # 第 2–3 步：逐段功率 + 仅有效段平滑；同时收集成熟段集合 E。
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

    # 第 4 步：样本太少不许冻结（阈值毫无代表性），硬停（C7）。
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
    """运行期逐段 4–8Hz 因果滤波 + 段功率；滤波器状态跨段延续。

    用 calibrate_theta 返回的 zf 构造时，本流式输出与「对 clean[0:840s] 整段
    一次性单向滤波」的对应样本逐点相等——同一份 θ 信号，只是分 4200 次喂入。
    跨 block、跨 30s 休息状态都保持不变（休息期没有数据进来，自然挂起）。
    """

    def __init__(
        self,
        zf: tuple[np.ndarray, np.ndarray] | None = None,
        *,
        sfreq: float = config.SFREQ,
        band_hz: tuple[float, float] = config.THETA_BAND_HZ,
    ):
        # zf=None 时两节零初始启动（测试用；生产路径总是注入校准 zf）。
        zi = (None, None) if zf is None else zf
        self._f = causal_filter.CausalChannel(sfreq=sfreq, band_hz=band_hz, zi=zi)

    def push_segment(self, clean_uV: np.ndarray) -> tuple[np.ndarray, float]:
        """喂入一段 50 点 Fz_clean，返回 (本段 θ 样本, 本段功率 µV²)。"""
        y = self._f.push(np.asarray(clean_uV, dtype=float))
        return y, float(np.mean(y * y))

    def states(self) -> tuple[np.ndarray, np.ndarray]:
        """导出两节当前状态（测试/诊断用）。"""
        return self._f.states()
