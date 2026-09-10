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

    返回 (b1, b2)。长度不一致、起点越界或系数非有限（NaN/Inf）→ ValueError。
    """
    fz = np.asarray(fz_f, dtype=float)
    fp1 = np.asarray(fp1_f, dtype=float)
    fp2 = np.asarray(fp2_f, dtype=float)
    if not (fz.size == fp1.size == fp2.size):
        raise ValueError(f"三路长度不一致: {fz.size}, {fp1.size}, {fp2.size}")
    if start_sample < 0 or start_sample >= fz.size:
        raise ValueError(f"warmup 起点越界: {start_sample} / {fz.size}")

    y = fz[start_sample:] - fz[start_sample:].mean()
    X = np.column_stack([fp1[start_sample:], fp2[start_sample:]])
    X = X - X.mean(axis=0, keepdims=True)
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(X))):
        raise ValueError("EOG 回归输入含非有限值（NaN/Inf），校准失败")
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    if not np.all(np.isfinite(b)):
        raise ValueError("EOG 回归系数非有限值（NaN/Inf），校准失败")
    return float(b[0]), float(b[1])
