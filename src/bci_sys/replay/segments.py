"""因果分段器：只向前切，绝不碰未来样本（C3 实时路径）。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config


@dataclass(frozen=True)
class Segment:
    seg: int
    t0: float  # 数据时间（秒）
    t1: float
    samples: np.ndarray  # 本段 50 个采样点


def _available_samples(n_samples: int, sfreq: float, duration_sec: float) -> int:
    """参与切分的可用样本数。负 duration 抛 ValueError，零/None 表示全量。"""
    if duration_sec is not None and duration_sec < 0:
        raise ValueError(f"duration_sec 不能为负: {duration_sec}")
    if duration_sec and duration_sec > 0:
        return min(n_samples, int(duration_sec * sfreq))
    return n_samples


def n_planned_segments(
    n_samples: int,
    seg_samples: int,
    duration_sec: float = 0.0,
    sfreq: float = config.SFREQ,
) -> int:
    n_avail = _available_samples(n_samples, sfreq, duration_sec)
    return int(n_avail // seg_samples)


def tail_seconds(
    n_samples: int,
    sfreq: float = config.SFREQ,
    seg_samples: int = config.SEGMENT_SAMPLES,
    duration_sec: float = 0.0,
) -> float:
    """已纳入切分的样本中不足一段的尾巴（秒）。

    全量：源文件末尾残段。
    限时：在段边界停止 → 0.0；duration 落在段中间 → 该残段时长。
    """
    n_avail = _available_samples(n_samples, sfreq, duration_sec)
    return float((n_avail % seg_samples) / sfreq)


def iter_segments(
    fz_uV: np.ndarray,
    sfreq: float = config.SFREQ,
    seg_samples: int = config.SEGMENT_SAMPLES,
    duration_sec: float = 0.0,
):
    """逐段产出 Segment。duration_sec=0 表示全量；>0 时在段边界截断。"""
    n_emit = n_planned_segments(fz_uV.size, seg_samples, duration_sec, sfreq=sfreq)
    for k in range(n_emit):
        s = k * seg_samples
        e = s + seg_samples
        yield Segment(seg=k, t0=s / sfreq, t1=e / sfreq, samples=fz_uV[s:e])


def segment_events(events: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    """onset 落在 [t0, t1) 的事件（左闭右开，避免边界事件被两段重复计数）。"""
    return events[(events["onset"] >= t0) & (events["onset"] < t1)]
