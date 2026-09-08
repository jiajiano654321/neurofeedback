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


def n_planned_segments(n_samples: int, seg_samples: int, duration_sec: float) -> int:
    n_full = n_samples // seg_samples
    if duration_sec and duration_sec > 0:
        n_dur = int(duration_sec * config.SFREQ) // seg_samples
        return int(min(n_full, n_dur))
    return int(n_full)


def tail_seconds(
    n_samples: int,
    sfreq: float = config.SFREQ,
    seg_samples: int = config.SEGMENT_SAMPLES,
    duration_sec: float = 0.0,
) -> float:
    """不足一段的尾巴时长（秒）。限时模式下按截断后的时长算。"""
    n_emit = n_planned_segments(n_samples, seg_samples, duration_sec)
    return n_samples / sfreq - n_emit * seg_samples / sfreq


def iter_segments(
    fz_uV: np.ndarray,
    sfreq: float = config.SFREQ,
    seg_samples: int = config.SEGMENT_SAMPLES,
    duration_sec: float = 0.0,
):
    """逐段产出 Segment。duration_sec=0 表示全量；>0 时在段边界截断。"""
    n_emit = n_planned_segments(fz_uV.size, seg_samples, duration_sec)
    for k in range(n_emit):
        s = k * seg_samples
        e = s + seg_samples
        yield Segment(seg=k, t0=s / sfreq, t1=e / sfreq, samples=fz_uV[s:e])


def segment_events(events: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    """onset 落在 [t0, t1) 的事件（左闭右开，避免边界事件被两段重复计数）。"""
    return events[(events["onset"] >= t0) & (events["onset"] < t1)]
