"""纯 I/O：读 BrainVision 与 events.tsv、通道查找。
本模块不含信号处理流程逻辑。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _eeg_dir(subj: str, root: Path | str | None = None) -> Path:
    base = Path(root) if root is not None else config.data_root()
    return base / f"sub-{subj}" / "eeg"


def vhdr_path(subj: str, root: Path | str | None = None) -> Path:
    return _eeg_dir(subj, root) / f"sub-{subj}_task-nback_eeg.vhdr"


def events_path(subj: str, root: Path | str | None = None) -> Path:
    return _eeg_dir(subj, root) / f"sub-{subj}_task-nback_events.tsv"


def channel_index(raw, name: str) -> int:
    """按名字显式查找通道下标；找不到直接抛错（通道映射安全靠它，不靠位置假设）。"""
    names = list(raw.ch_names)
    if name not in names:
        raise KeyError(f"通道 {name!r} 不存在；可用通道：{names}")
    return names.index(name)


def load_raw(subj: str, root: Path | str | None = None):
    import mne

    p = vhdr_path(subj, root)
    if not p.exists():
        raise FileNotFoundError(f"找不到 BrainVision 文件：{p}")
    return mne.io.read_raw_brainvision(str(p), preload=True, verbose="ERROR")


def load_fz_uV(subj: str, root: Path | str | None = None) -> np.ndarray:
    """返回 Fz 单通道全程数据，单位 µV。

    【单位陷阱】本数据集由 pybv 0.7.6 写出，mne 输出数值经实测已是 µV 量级
    （1-30Hz 带通后 std ≈ 42 µV，符合正常 EEG）。
    因此【严禁再乘 1e6】——错乘会让 theta 功率大 10^12 倍但单调性不变、
    IT3 的 L1 全部超标，且全程不报错。
    信号带 ~1.44e6 的直流偏移属正常，由后续带通滤波去除。
    """
    raw = load_raw(subj, root)
    idx = channel_index(raw, config.TARGET.channel)
    return raw.get_data()[idx]


def load_events(subj: str, root: Path | str | None = None) -> pd.DataFrame:
    p = events_path(subj, root)
    if not p.exists():
        raise FileNotFoundError(f"找不到 events.tsv：{p}")
    return pd.read_csv(p, sep="\t")


def real_trial_events(events: pd.DataFrame) -> pd.DataFrame:
    """非练习 n-back 试次。

    pandas 把 istutorial 的 "true" 解析为 True、"n/a" 解析为 NaN；
    真实试次 istutorial 为 NaN（!= True 成立），练习试次为 True（被排除）。
    nback_level 为 1-4 的行才是有效试次（dropped_samples 等事件为 NaN）。
    """
    tr = events[
        (events["istutorial"] != True)  # noqa: E712 —— NaN != True 为 True，正是我们要的
        & (events["nback_level"].isin([1, 2, 3, 4]))
    ].copy()
    tr["nback_level"] = tr["nback_level"].astype(int)
    return tr.sort_values("onset").reset_index(drop=True)
