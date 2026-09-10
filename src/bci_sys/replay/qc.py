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
