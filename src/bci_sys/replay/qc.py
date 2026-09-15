"""QC 纯逻辑（无 I/O、无墙钟）：p2p、冻结 EOG 回归、L1/L3 判定、校准状态机。

输入均为实时因果路径已产出的数据（C3）；无效段只输出标签与原因，
不删除任何数据（C5）。
"""
from __future__ import annotations

import numpy as np

from .. import config


def p2p_uv(x) -> float:
    """段内峰峰值（µV）。"""
    x = np.asarray(x, dtype=float)
    return float(x.max() - x.min())


def fit_eog_coeffs(fz_f, fp1_f, fp2_f, start_sample: int) -> tuple[float, float]:
    """在 [start_sample, ] 区间对【去均值】三路做无截距 OLS：

        Fz_centered = b1 * FP1_centered + b2 * FP2_centered

    返回 (b1, b2)。以下情况均抛 ValueError：
    - 三路长度不一致
    - start_sample 越界
    - 校准窗内含非有限输入（NaN / Inf）
    - 设计矩阵零秩（两路 EOG 校准窗内全恒定，去均值后全零）
    - 回归系数非有限（NaN / Inf）
    """
    fz = np.asarray(fz_f, dtype=float)
    fp1 = np.asarray(fp1_f, dtype=float)
    fp2 = np.asarray(fp2_f, dtype=float)
    if not (fz.size == fp1.size == fp2.size):
        raise ValueError(f"三路长度不一致: {fz.size}, {fp1.size}, {fp2.size}")
    if start_sample < 0 or start_sample >= fz.size:
        raise ValueError(f"warmup 起点越界: {start_sample} / {fz.size}")

    y0 = fz[start_sample:]
    X0 = np.column_stack([fp1[start_sample:], fp2[start_sample:]])
    # 先在原始切片上拦非有限：若先做 inf−mean(inf) 去均值会先抛 RuntimeWarning。
    # 也防住「y 全 NaN、X 全零」时 lstsq 静默返回有限 [0,0] 绕过后续检查。
    if not (np.all(np.isfinite(y0)) and np.all(np.isfinite(X0))):
        raise ValueError("EOG 回归输入含非有限值（NaN/Inf），校准失败")
    y = y0 - y0.mean()
    X = X0 - X0.mean(axis=0, keepdims=True)
    # 零秩（两路去均值后全 0＝全程恒定）时 lstsq 静默返回有限 [0,0]，会伪装成
    # 「冻结成功但完全不去眼电」，必须硬拦（C7）。秩 1（两通道共线、或一路恒定）
    # 仍给出唯一有效的去眼电线性组合（实测最小范数解的残差相同），可接受。
    if np.linalg.matrix_rank(X) == 0:
        raise ValueError("EOG 回归设计矩阵零秩（FP1/FP2 校准窗内恒定？），校准失败")
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    if not np.all(np.isfinite(b)):
        raise ValueError("EOG 回归系数非有限值（NaN/Inf），校准失败")
    return float(b[0]), float(b[1])


def segment_verdict(
    fz_f, fp1_f, fp2_f, b_fp1: float, b_fp2: float,
    l1_thr: float, l3_thr: float,
) -> dict:
    """冻结后单段 L1/L3 裁决（纯函数；QcJudge 运行期与 IT4 校准重建共用，保证同源）。

    返回 fz_clean_p2p / eog_p2p / valid / reasons（不含 phase）。
    """
    fz = np.asarray(fz_f, dtype=float)
    fp1 = np.asarray(fp1_f, dtype=float)
    fp2 = np.asarray(fp2_f, dtype=float)

    clean = fz - b_fp1 * fp1 - b_fp2 * fp2
    clean_p2p = p2p_uv(clean)
    # 两路取大必须用 np.maximum：内建 max 遇 NaN 有顺序依赖（见 QcJudge 旧注）。
    eog_p2p = float(np.maximum(p2p_uv(fp1), p2p_uv(fp2)))
    # 用 not(<=) 而非 (>)：NaN 段也判命中，绝不把坏段静默标成有效。
    hit_l1 = not (clean_p2p <= l1_thr)
    hit_l3 = not (eog_p2p <= l3_thr)
    reasons = []
    if hit_l1:
        reasons.append("L1")
    if hit_l3:
        reasons.append("L3")
    return {
        "fz_clean_p2p": clean_p2p,
        "eog_p2p": eog_p2p,
        "valid": not (hit_l1 or hit_l3),
        "reasons": reasons,
    }


class QcJudge:
    """校准→运行状态机。逐段喂入【已因果带通】的三路等长块。

    - 未冻结：累积滤波后样本；段段返回 phase="calibration"（valid=None）。
      累积到 calibration_sec 对应样本数的那一段，用 [warmup_sec, calibration_sec]
      区间一次性估计并冻结 b，在该段返回值里附带 calib 记录（仅一次）。
    - 冻结后：残差 L1 + EOG L3 裁决，返回 phase="running"。
    - 数据短于校准窗：永不冻结，调用方循环正常结束（短回放/测试场景）。
    """

    def __init__(
        self,
        calibration_sec: float = config.QC_CALIBRATION_SEC,
        warmup_sec: float = config.QC_FILTER_WARMUP_SEC,
        sfreq: float = config.SFREQ,
        l1_thr: float = config.QC_L1_P2P_UV,
        l3_thr: float = config.QC_L3_EOG_P2P_UV,
        fz_name: str = config.TARGET.channel,
        eog_names: tuple[str, str] = config.EOG_PROXY_CHANNELS,
    ):
        sfreq = float(sfreq)
        self.calibration_sec = float(calibration_sec)
        self.warmup_sec = float(warmup_sec)
        self.l1_thr = float(l1_thr)
        self.l3_thr = float(l3_thr)
        self.fz_name = fz_name
        self.eog_names = tuple(eog_names)

        self.calib_samples = int(round(self.calibration_sec * sfreq))
        self.warmup_samples = int(round(self.warmup_sec * sfreq))
        self._names = (self.fz_name, *self.eog_names)
        self._buf = {c: [] for c in self._names}
        self._n_accum = 0
        self.b_fp1: float | None = None
        self.b_fp2: float | None = None
        self.frozen = False
        self.calib: dict | None = None

    def _calibration_verdict(self) -> dict:
        return {
            "phase": "calibration",
            "fz_clean_p2p": None,
            "eog_p2p": None,
            "valid": None,
            "reasons": [],
        }

    def push(self, t1: float, filtered: dict[str, np.ndarray]) -> dict:
        """t1=本段结束的数据时间（秒）；filtered 含 Fz/FP1/FP2 三路本段样本。

        前置条件：各通道块等长、非空、一维（上游 CausalBandpass / iter_segments 保证），
        judge 不重复校验形状。
        """
        if self.frozen:
            return self._running_verdict(filtered)

        block_len = np.asarray(filtered[self.fz_name]).size
        for c in self._names:
            self._buf[c].append(np.asarray(filtered[c], dtype=float))
        self._n_accum += block_len

        verdict = self._calibration_verdict()
        if self._n_accum >= self.calib_samples:
            self._freeze(t1)
            verdict["calib"] = self.calib
        return verdict

    def _freeze(self, t1: float) -> None:
        arrs = {
            c: np.concatenate(self._buf[c])[: self.calib_samples]
            for c in self._names
        }
        b1, b2 = fit_eog_coeffs(
            arrs[self.fz_name],
            arrs[self.eog_names[0]],
            arrs[self.eog_names[1]],
            self.warmup_samples,
        )
        self.b_fp1, self.b_fp2 = b1, b2
        self.frozen = True
        self.calib = {
            "type": "calib",
            "t_sec": round(float(t1), 6),
            "b_fp1": b1,
            "b_fp2": b2,
            "fit_window_sec": [self.warmup_sec, self.calibration_sec],
            "n_fit_samples": self.calib_samples - self.warmup_samples,
        }
        self._buf = {}  # 校准原始缓冲不再需要

    def _running_verdict(self, filtered: dict[str, np.ndarray]) -> dict:
        return {
            "phase": "running",
            **segment_verdict(
                filtered[self.fz_name],
                filtered[self.eog_names[0]],
                filtered[self.eog_names[1]],
                self.b_fp1, self.b_fp2, self.l1_thr, self.l3_thr,
            ),
        }
