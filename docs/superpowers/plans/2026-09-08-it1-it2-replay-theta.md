# IT1 replay_monitor + IT2 theta_probe 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现两个可运行程序——IT1 `replay-monitor`（1x 节奏回放 ds007169 到终端 + 确定性 JSONL）和 IT2 `theta-probe`（离线 Fz theta 按 n-back 负荷分组，单调上升硬门控）。

**Architecture:** 单包双入口（src layout），`bci_sys/replay/`（实时路径）与 `bci_sys/offline/`（离线路径）物理隔离、互不 import，共享 `config.py`（常量）与 `io_bids.py`（纯 I/O），AST 测试强制边界。所有信号处理参数集中在 config；靶点 `direction="up"` 必填无默认（C1）。

**Tech Stack:** Python 3.11、uv、mne（读 BrainVision + multitaper PSD）、numpy、scipy、pandas、matplotlib(Agg)、pytest。

**依据 spec：** [docs/superpowers/specs/2026-09-08-it1-it2-replay-theta-design.md](../specs/2026-09-08-it1-it2-replay-theta-design.md)

**关键事实（已实测，违反即错）：**
- mne 读出的数值**直接就是 µV，严禁乘 1e6**；信号带 ~1.44e6 直流偏移，滤波后 std 应在 5–300 µV。
- 1–30 Hz FIR 长 825 点（3.3 s）> epoch 375 点（1.5 s）：**必须先全程滤波再切 epoch**。
- 非练习试次 400 个（1/2/3/4-back 各 100），过滤条件 `istutorial != True 且 nback_level ∈ {1,2,3,4}`（pandas 把 "true" 解析为 True、"n/a" 解析为 NaN）。
- 全量 958.508 s → 4792 个整段 + 0.108 s 尾巴。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `pyproject.toml` | uv 项目、依赖、两个 console script、pytest 配置 |
| `src/bci_sys/__init__.py` | 包标记 |
| `src/bci_sys/config.py` | 全部常量；`Target`（direction 必填）；`data_root()` |
| `src/bci_sys/io_bids.py` | 纯 I/O：路径、`load_fz_uV`、`load_events`、`channel_index`、`preflight`（含单位硬检查） |
| `src/bci_sys/gitmeta.py` | `code_version()`：git hash + dirty（D2） |
| `src/bci_sys/replay/__init__.py` | 包标记 |
| `src/bci_sys/replay/segments.py` | 因果分段器 + 事件对齐 |
| `src/bci_sys/replay/pace.py` | 1x 节拍器（时钟/sleep 可注入） |
| `src/bci_sys/replay/monitor.py` | 回放主循环、JSONL/run.json 落盘、终端渲染、速率颜色 |
| `src/bci_sys/replay/cli.py` | `replay-monitor` 入口 |
| `src/bci_sys/offline/__init__.py` | 包标记 |
| `src/bci_sys/offline/theta.py` | 全程滤波 → 切 epoch → multitaper theta 功率 |
| `src/bci_sys/offline/probe.py` | 负荷分组、中位数、单调门控、CSV/PNG 产物、--all |
| `src/bci_sys/offline/cli.py` | `theta-probe` 入口 |
| `tests/conftest.py` | `data_root` fixture（数据缺失则 skip） |
| `tests/test_*.py` | 各模块测试 |

---

## Task 1: 项目脚手架

**Files:**
- Create: `pyproject.toml`
- Create: `src/bci_sys/__init__.py`
- Create: `src/bci_sys/replay/__init__.py`
- Create: `src/bci_sys/offline/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: 创建 `pyproject.toml`**

```toml
[project]
name = "bci-sys"
version = "0.1.0"
description = "神经反馈训练原型：回放模拟实时（IT1）+ 离线 theta 检验（IT2）"
requires-python = ">=3.11"
dependencies = [
    "mne>=1.6",
    "numpy>=2.0",
    "scipy",
    "pandas",
    "matplotlib",
]

[dependency-groups]
dev = ["pytest"]

[project.scripts]
replay-monitor = "bci_sys.replay.cli:main"
theta-probe = "bci_sys.offline.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/bci_sys"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: 创建包目录标记文件**

`src/bci_sys/__init__.py`、`src/bci_sys/replay/__init__.py`、`src/bci_sys/offline/__init__.py` 均为空文件。

- [ ] **Step 3: 创建 `tests/conftest.py`**

```python
import pytest

from bci_sys import config


@pytest.fixture
def data_root():
    """返回数据根目录；ds007169 不在则 skip（数据 gitignore，不进仓库）。"""
    p = config.data_root()
    if not (p / "sub-001" / "eeg").exists():
        pytest.skip("ds007169 data not available")
    return p
```

- [ ] **Step 4: 创建 `tests/test_smoke.py`**

```python
def test_imports():
    import bci_sys
    import bci_sys.replay
    import bci_sys.offline
    assert bci_sys is not None
```

- [ ] **Step 5: 同步依赖并跑测试**

Run: `uv sync && uv run pytest -v`
Expected: 1 passed（uv 会自动建虚拟环境并安装 mne/numpy/scipy/pandas/matplotlib/pytest）。

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src tests
git commit -m "chore: 项目脚手架（uv + src layout + 两个 console script）"
```

---

## Task 2: `config.py` —— 常量与靶点（C1）

**Files:**
- Create: `src/bci_sys/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写失败测试 `tests/test_config.py`**

```python
import os

import pytest

from bci_sys import config


def test_target_direction_is_required_and_up():
    # C1：direction 必填、无默认；本项目靶点显式上调
    with pytest.raises(TypeError):
        config.Target(channel="Fz", band_hz=(4.0, 8.0))  # 缺 direction
    with pytest.raises(ValueError):
        config.Target(channel="Fz", band_hz=(4.0, 8.0), direction="sideways")
    t = config.Target(channel="Fz", band_hz=(4.0, 8.0), direction="up")
    assert t.channel == "Fz"
    assert t.direction == "up"


def test_module_target_constant():
    assert config.TARGET.channel == "Fz"
    assert config.TARGET.direction == "up"
    assert config.TARGET.band_hz == (4.0, 8.0)


def test_segment_constants():
    assert config.SFREQ == 250.0
    assert config.SEGMENT_SEC == 0.2
    assert config.SEGMENT_SAMPLES == 50


def test_data_root_default_and_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("BCI_DATA_ROOT", raising=False)
    assert config.data_root().name == "ds007169"
    monkeypatch.setenv("BCI_DATA_ROOT", str(tmp_path))
    assert config.data_root() == tmp_path


def test_eog_proxy_channel_names_uppercase():
    # 通道名以 channels.tsv 为准：FP1/FP2 大写
    assert config.EOG_PROXY_CHANNELS == ("FP1", "FP2")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL —— `ModuleNotFoundError: bci_sys.config`。

- [ ] **Step 3: 实现 `src/bci_sys/config.py`**

```python
"""唯一常量出处。任何信号处理/流程参数只在这里定义。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- 采样与分段 ---
SFREQ = 250.0
SEGMENT_SEC = 0.2
SEGMENT_SAMPLES = 50  # 250 Hz × 0.2 s

# --- theta / epoch（IT2 离线） ---
THETA_BAND_HZ = (4.0, 8.0)
EPOCH_SEC = 1.5
MULTITAPER_NW = 2.0

# --- 被试名单 ---
RECOMMENDED_SUBS = ["001", "002", "006", "014", "015", "018", "019"]
EXTREME_SUB = "012"  # 留作极端测试，本期不跑

# --- 通道（名字大小写以 channels.tsv 为准） ---
EOG_PROXY_CHANNELS = ("FP1", "FP2")  # IT3 才用，先登记

# --- 速率显示颜色阈值（data_elapsed / wall_elapsed） ---
RATE_GREEN = 0.98
RATE_YELLOW = 0.95

# --- preflight 单位硬检查：1-30Hz 带通后 Fz std 的允许范围（µV） ---
UNIT_STD_RANGE_UV = (5.0, 300.0)


@dataclass(frozen=True)
class Target:
    """训练靶点。C1：direction 必填、无默认，构造时不写就报错。"""

    channel: str
    band_hz: tuple[float, float]
    direction: str  # 必须显式传 "up" 或 "down"，无默认值

    def __post_init__(self):
        if not isinstance(self.direction, str):
            raise ValueError("direction 必须是字符串 'up'/'down'，不许留空")
        if self.direction not in ("up", "down"):
            raise ValueError(f"direction 只能是 'up'/'down'，收到 {self.direction!r}")


# 本期靶点：theta 4-8 Hz @ Fz，方向上调（显式写出，C1）
TARGET = Target(channel="Fz", band_hz=THETA_BAND_HZ, direction="up")


def data_root() -> Path:
    """ds007169 根目录。默认项目内 ds007169/，可用 BCI_DATA_ROOT 覆盖。"""
    env = os.environ.get("BCI_DATA_ROOT")
    if env:
        return Path(env)
    # src/bci_sys/config.py → parents[2] = 仓库根
    return Path(__file__).resolve().parents[2] / "ds007169"
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: 5 passed。

- [ ] **Step 5: Commit**

```bash
git add src/bci_sys/config.py tests/test_config.py
git commit -m "feat: config.py 常量与 Target（direction 必填，C1）"
```

---

## Task 3: `io_bids` —— 路径、events、通道查找

**Files:**
- Create: `src/bci_sys/io_bids.py`
- Test: `tests/test_io_bids.py`

- [ ] **Step 1: 写失败测试 `tests/test_io_bids.py`**

```python
import pandas as pd
import pytest

from bci_sys import config, io_bids


class _FakeRaw:
    """最小 mne Raw 替身：只需要 ch_names。"""

    def __init__(self, ch_names):
        self.ch_names = ch_names


def test_channel_index_finds_fz():
    raw = _FakeRaw(["Pz", "Fz", "Cz"])
    assert io_bids.channel_index(raw, "Fz") == 1


def test_channel_index_raises_on_missing():
    raw = _FakeRaw(["Pz", "Cz"])
    with pytest.raises(KeyError, match="Fz"):
        io_bids.channel_index(raw, "Fz")


def test_paths_point_to_bids_files(data_root):
    vhdr = io_bids.vhdr_path("001", data_root)
    assert vhdr.name == "sub-001_task-nback_eeg.vhdr"
    assert vhdr.exists()
    assert io_bids.events_path("001", data_root).exists()


def test_load_events_and_trial_filter(data_root):
    ev = io_bids.load_events("001", data_root)
    assert "nback_level" in ev.columns
    assert "istutorial" in ev.columns
    tr = io_bids.real_trial_events(ev)
    # 非练习试次 400 个，1/2/3/4-back 各 100
    assert len(tr) == 400
    counts = tr["nback_level"].value_counts().sort_index()
    assert list(counts.index) == [1, 2, 3, 4]
    assert (counts == 100).all()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_io_bids.py -v`
Expected: FAIL —— `ModuleNotFoundError: bci_sys.io_bids`。

- [ ] **Step 3: 实现 `src/bci_sys/io_bids.py`（本任务只写路径/events/通道部分）**

```python
"""纯 I/O：读 BrainVision 与 events.tsv、通道查找、preflight。
本模块不含任何信号处理流程逻辑（preflight 里的带通是一次性单位诊断，不是管线处理）。
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_io_bids.py -v`
Expected: 4 passed（后两个需要真实数据；数据在项目内 ds007169/）。

- [ ] **Step 5: Commit**

```bash
git add src/bci_sys/io_bids.py tests/test_io_bids.py
git commit -m "feat: io_bids 路径/events 读取/通道按名查找（含单位陷阱 docstring）"
```

---

## Task 4: `preflight` —— S1 质检与单位硬门

**Files:**
- Modify: `src/bci_sys/io_bids.py`
- Test: `tests/test_preflight.py`

- [ ] **Step 1: 写失败测试 `tests/test_preflight.py`**

```python
import numpy as np
import pytest

from bci_sys import io_bids


def test_unit_std_check_passes_normal_eeg():
    rng = np.random.default_rng(0)
    t = np.arange(250 * 30) / 250.0
    # 30 µV 量级的 6 Hz 振荡 + 噪声 → 带通后 std 在正常范围
    x = 30.0 * np.sin(2 * np.pi * 6 * t) + 5.0 * rng.standard_normal(t.size)
    io_bids.check_fz_unit_std(x, sfreq=250.0)  # 不抛错即通过


def test_unit_std_check_catches_1e6_mistake():
    rng = np.random.default_rng(1)
    t = np.arange(250 * 30) / 250.0
    x = 30.0 * np.sin(2 * np.pi * 6 * t) + 5.0 * rng.standard_normal(t.size)
    with pytest.raises(ValueError, match="单位换算"):
        io_bids.check_fz_unit_std(x * 1e6, sfreq=250.0)  # 模拟误乘 1e6


def test_unit_std_check_catches_flat_signal():
    x = np.zeros(250 * 30)
    with pytest.raises(ValueError, match="单位换算"):
        io_bids.check_fz_unit_std(x, sfreq=250.0)


def test_preflight_passes_on_sub001(data_root):
    io_bids.preflight("001", data_root)  # 不抛错即通过


def test_preflight_fails_on_missing_subject(data_root, tmp_path):
    with pytest.raises(FileNotFoundError):
        io_bids.preflight("999", data_root)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_preflight.py -v`
Expected: FAIL —— `AttributeError: module 'bci_sys.io_bids' has no attribute 'check_fz_unit_std'`。

- [ ] **Step 3: 在 `src/bci_sys/io_bids.py` 末尾追加实现**

```python
def check_fz_unit_std(fz_uV: np.ndarray, sfreq: float = config.SFREQ) -> None:
    """单位硬检查：1-30Hz 带通后 Fz std 必须落在正常 EEG 范围。

    这是唯一能挡住『误乘 1e6 / 单位搞错』这类不报错错误的闸门。
    注意：此处带通仅为启动诊断，不属于实时/离线信号处理管线。
    """
    import mne

    lo, hi = config.UNIT_STD_RANGE_UV
    filt = mne.filter.filter_data(
        fz_uV.astype(np.float64), sfreq, 1.0, 30.0, method="fir", verbose="ERROR"
    )
    std = float(np.std(filt))
    if not (lo < std < hi):
        raise ValueError(
            f"Fz 带通后 std={std:.1f} µV，超出正常 EEG 范围 {lo}-{hi} µV。"
            f"极可能是单位换算错误（检查是否误乘 1e6，或 mne 输出被错误换算）。"
        )


def preflight(subj: str, root: Path | str | None = None) -> None:
    """S1 质检（回放期简化版）。任一不过：抛异常（硬停，C7），不写任何文件。"""
    # 1. 文件可读
    raw = load_raw(subj, root)
    ev_path = events_path(subj, root)
    if not ev_path.exists():
        raise FileNotFoundError(f"events.tsv 缺失：{ev_path}")

    # 2. 采样率
    sfreq = float(raw.info["sfreq"])
    if abs(sfreq - config.SFREQ) > 1e-6:
        raise ValueError(f"采样率 {sfreq} != 设计值 {config.SFREQ}")

    # 3. 通道：mne 侧 Fz 存在（不存在即抛 KeyError）；EEG 通道数以 channels.tsv 为准
    #    （mne 读 vhdr 可能把 24 导全标成 eeg，不能信它的类型计数）
    channel_index(raw, config.TARGET.channel)
    import pandas as pd

    ch_tsv = _eeg_dir(subj, root) / f"sub-{subj}_task-nback_channels.tsv"
    if not ch_tsv.exists():
        raise FileNotFoundError(f"channels.tsv 缺失：{ch_tsv}")
    chans = pd.read_csv(ch_tsv, sep="\t")
    n_eeg = int((chans["type"] == "EEG").sum())
    if n_eeg != 19:
        raise ValueError(f"channels.tsv 中 EEG 通道数 {n_eeg} != 19")
    if config.TARGET.channel not in set(chans["name"]):
        raise KeyError(f"channels.tsv 中无通道 {config.TARGET.channel}")

    # 4. events 含 nback_level 列
    ev = pd.read_csv(ev_path, sep="\t", nrows=1)
    if "nback_level" not in ev.columns:
        raise ValueError("events.tsv 缺少 nback_level 列")

    # 5. 单位硬检查
    idx = channel_index(raw, config.TARGET.channel)
    fz = raw.get_data()[idx]
    check_fz_unit_std(fz, sfreq)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_preflight.py -v`
Expected: 5 passed。

- [ ] **Step 5: Commit**

```bash
git add src/bci_sys/io_bids.py tests/test_preflight.py
git commit -m "feat: preflight S1 质检 + 单位硬门（std 5-300µV，挡住误乘 1e6）"
```

---

## Task 5: `replay/segments.py` —— 因果分段器

**Files:**
- Create: `src/bci_sys/replay/segments.py`
- Test: `tests/test_segments.py`

- [ ] **Step 1: 写失败测试 `tests/test_segments.py`**

```python
import numpy as np
import pandas as pd

from bci_sys import config
from bci_sys.replay import segments


def test_segmenter_basic_shape_and_times():
    x = np.arange(150, dtype=float)  # 3 整段
    segs = list(segments.iter_segments(x, sfreq=250.0, seg_samples=50))
    assert len(segs) == 3
    assert segs[0].seg == 0
    assert np.allclose(segs[0].samples, np.arange(50))
    assert segs[0].t0 == 0.0
    assert abs(segs[0].t1 - 0.2) < 1e-9
    assert np.allclose(segs[2].samples, np.arange(100, 150))


def test_segmenter_tail_not_emitted():
    x = np.arange(130, dtype=float)  # 2 整段 + 30 点尾巴
    segs = list(segments.iter_segments(x, sfreq=250.0, seg_samples=50))
    assert len(segs) == 2
    # 尾巴 = 130/250 - 2*0.2 = 0.12 s
    assert abs(segments.tail_seconds(x.size, sfreq=250.0, seg_samples=50) - 0.12) < 1e-9


def test_segmenter_duration_limit_at_segment_boundary():
    x = np.arange(250, dtype=float)  # 5 段 = 1.0 s
    segs = list(segments.iter_segments(x, sfreq=250.0, seg_samples=50, duration_sec=0.4))
    assert len(segs) == 2  # 0.4 s = 2 段，段边界截断
    assert segs[-1].seg == 1


def test_segmenter_duration_zero_means_full():
    x = np.arange(250, dtype=float)
    assert len(list(segments.iter_segments(x, sfreq=250.0, seg_samples=50, duration_sec=0.0))) == 5


def test_segment_events_half_open_interval():
    ev = pd.DataFrame({"onset": [0.0, 0.2, 0.25, 0.4], "trial_type": ["a", "b", "c", "d"]})
    # 段 [0.2, 0.4)：含 0.2、0.25，不含 0.0、0.4（左闭右开，边界不重复计）
    got = segments.segment_events(ev, t0=0.2, t1=0.4)
    assert list(got["trial_type"]) == ["b", "c"]


def test_sub001_full_run_is_4792_segments(data_root):
    from bci_sys import io_bids

    fz = io_bids.load_fz_uV("001", data_root)
    segs = list(
        segments.iter_segments(fz, sfreq=config.SFREQ, seg_samples=config.SEGMENT_SAMPLES)
    )
    assert len(segs) == 4792
    assert all(s.samples.size == 50 for s in segs)
    assert abs(segments.tail_seconds(fz.size) - 0.108) < 0.01
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_segments.py -v`
Expected: FAIL —— `ModuleNotFoundError: bci_sys.replay.segments`。

- [ ] **Step 3: 实现 `src/bci_sys/replay/segments.py`**

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_segments.py -v`
Expected: 6 passed。

- [ ] **Step 5: Commit**

```bash
git add src/bci_sys/replay/segments.py tests/test_segments.py
git commit -m "feat: 因果分段器 + 事件左闭右开对齐（4792 段/0.108s 尾）"
```

---

## Task 6: `replay/pace.py` —— 1x 节拍器

**Files:**
- Create: `src/bci_sys/replay/pace.py`
- Test: `tests/test_pace.py`

- [ ] **Step 1: 写失败测试 `tests/test_pace.py`**

```python
from bci_sys.replay import pace


class FakeClock:
    """假时钟：sleep 多久时钟就走多久，确定性测试用（不走真睡眠）。"""

    def __init__(self):
        self.t = 0.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, dt):
        self.slept.append(dt)
        self.t += dt


def test_pacer_sleeps_until_segment_deadline():
    clk = FakeClock()
    p = pace.Pacer(segment_sec=0.2, sleep_fn=clk.sleep, clock=clk.now)
    p.start()
    p.wait_after_segment(0)  # 第 0 段结束：应睡到 0.2
    assert abs(clk.slept[-1] - 0.2) < 1e-9
    # 模拟处理耗了 0.05s（时钟已随 sleep 走过 0.2，再手动加 0.05）
    clk.t += 0.05
    p.wait_after_segment(1)  # 目标 0.4，当前 0.25 → 睡 0.15
    assert abs(clk.slept[-1] - 0.15) < 1e-9


def test_pacer_does_not_sleep_when_behind():
    clk = FakeClock()
    p = pace.Pacer(segment_sec=0.2, sleep_fn=clk.sleep, clock=clk.now)
    p.start()
    clk.t += 1.0  # 已经落后 1 秒
    p.wait_after_segment(0)
    assert clk.slept == []  # 落后时不睡负时间


def test_rate_ratio():
    clk = FakeClock()
    p = pace.Pacer(segment_sec=0.2, sleep_fn=clk.sleep, clock=clk.now)
    p.start()
    p.wait_after_segment(0)  # 墙钟走 0.2，数据走 0.2 → 1.0x
    assert abs(p.rate_ratio(data_elapsed=0.2) - 1.0) < 1e-9


def test_rate_color_thresholds():
    assert pace.rate_color(0.99) == pace.GREEN
    assert pace.rate_color(0.98) == pace.GREEN
    assert pace.rate_color(0.97) == pace.YELLOW
    assert pace.rate_color(0.95) == pace.YELLOW
    assert pace.rate_color(0.94) == pace.RED
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_pace.py -v`
Expected: FAIL —— `ModuleNotFoundError: bci_sys.replay.pace`。

- [ ] **Step 3: 实现 `src/bci_sys/replay/pace.py`**

```python
"""1x 节拍器。deadline 锚定起点、自校正，不累加漂移。时钟/sleep 可注入（测试用假时钟）。"""
from __future__ import annotations

import time

from .. import config

GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
RESET = "\x1b[0m"


def rate_color(ratio: float) -> str:
    if ratio >= config.RATE_GREEN:
        return GREEN
    if ratio >= config.RATE_YELLOW:
        return YELLOW
    return RED


class Pacer:
    def __init__(self, segment_sec: float = config.SEGMENT_SEC, sleep_fn=time.sleep,
                 clock=time.perf_counter):
        self.segment_sec = segment_sec
        self._sleep = sleep_fn
        self._clock = clock
        self._wall0 = None

    def start(self):
        self._wall0 = self._clock()

    def wait_after_segment(self, seg_index: int) -> None:
        """第 seg_index（0 起）段处理完后调用：睡到 wall0 + (k+1)*段长。"""
        target = self._wall0 + (seg_index + 1) * self.segment_sec
        dt = target - self._clock()
        if dt > 0:
            self._sleep(dt)

    def rate_ratio(self, data_elapsed: float) -> float:
        wall = self.wall_elapsed()
        if wall <= 0:
            return 1.0
        return data_elapsed / wall

    def wall_elapsed(self) -> float:
        return self._clock() - self._wall0
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_pace.py -v`
Expected: 4 passed。

- [ ] **Step 5: Commit**

```bash
git add src/bci_sys/replay/pace.py tests/test_pace.py
git commit -m "feat: 1x 节拍器（锚定起点自校正）+ 速率颜色阈值"
```

---

## Task 7: `gitmeta.py` —— 代码版本留痕（D2）

**Files:**
- Create: `src/bci_sys/gitmeta.py`
- Test: `tests/test_gitmeta.py`

- [ ] **Step 1: 写失败测试 `tests/test_gitmeta.py`**

```python
import re

from bci_sys import gitmeta


def test_code_version_is_hash_or_unknown():
    v = gitmeta.code_version()
    # 仓库内：7+ 位 hash，可能带 -dirty；仓库外：'unknown'
    assert v == "unknown" or re.fullmatch(r"[0-9a-f]{7,40}(-dirty)?", v)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_gitmeta.py -v`
Expected: FAIL —— `ModuleNotFoundError: bci_sys.gitmeta`。

- [ ] **Step 3: 实现 `src/bci_sys/gitmeta.py`**

```python
"""记录代码版本（D2）：git 短 hash + dirty 标记。非 git 环境返回 'unknown'。"""
from __future__ import annotations

import subprocess


def code_version() -> str:
    try:
        h = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_gitmeta.py -v`
Expected: 1 passed。

- [ ] **Step 5: Commit**

```bash
git add src/bci_sys/gitmeta.py tests/test_gitmeta.py
git commit -m "feat: code_version() 记录 git hash+dirty（D2）"
```

---

## Task 8: `replay/monitor.py` —— 回放主循环与确定性落盘

**Files:**
- Create: `src/bci_sys/replay/monitor.py`
- Test: `tests/test_monitor.py`

- [ ] **Step 1: 写失败测试 `tests/test_monitor.py`**

```python
import json
from pathlib import Path

from bci_sys.replay import monitor, pace


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt  # 不走真睡眠


def _read_jsonl(p: Path):
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def test_replay_writes_meta_seg_event_summary(data_root, tmp_path):
    clk = FakeClock()
    status = monitor.run_replay(
        "001", duration_sec=2.0, root=data_root, out_root=tmp_path,
        run_id="fixed", pacer=pace.Pacer(sleep_fn=clk.sleep, clock=clk.now),
    )
    assert status == "ok"
    run_dir = tmp_path / "replay" / "sub-001" / "fixed"
    rows = _read_jsonl(run_dir / "replay.jsonl")
    kinds = [r["type"] for r in rows]
    assert kinds[0] == "meta"
    assert kinds[-1] == "summary"
    assert kinds.count("seg") == 10  # 2.0 s = 10 段
    # meta 不含墙钟；含代码版本（D2）
    meta = rows[0]
    assert meta["sfreq"] == 250.0 and meta["fz_channel"] == "Fz"
    assert meta["n_segments_planned"] == 10
    assert "code_version" in meta and meta["code_version"] != "unknown"
    # 段行结构
    seg = next(r for r in rows if r["type"] == "seg")
    assert seg["seg"] == 0 and len(seg["fz_uV"]) == 50
    # 事件行：前 2 秒内有 dropped_samples（sub-001 首个事件在 0.17s）
    evs = [r for r in rows if r["type"] == "event"]
    assert any(e["kind"] == "dropped" for e in evs)
    # summary 只含数据侧量，无墙钟
    summ = rows[-1]
    assert summ["n_segments"] == 10
    assert "wall_elapsed_s" not in summ and "rate_ratio" not in summ
    assert summ["n_dropped"] >= 1


def test_replay_jsonl_byte_deterministic(data_root, tmp_path):
    """A4：同参数两次回放，replay.jsonl 字节级一致。"""
    outs = []
    for rid in ["a", "b"]:
        clk = FakeClock()
        monitor.run_replay(
            "001", duration_sec=2.0, root=data_root, out_root=tmp_path / rid,
            run_id="fixed", pacer=pace.Pacer(sleep_fn=clk.sleep, clock=clk.now),
        )
        outs.append((tmp_path / rid / "replay" / "sub-001" / "fixed" / "replay.jsonl").read_bytes())
    assert outs[0] == outs[1]


def test_run_json_contains_wall_and_status(data_root, tmp_path):
    clk = FakeClock()
    monitor.run_replay(
        "001", duration_sec=1.0, root=data_root, out_root=tmp_path,
        run_id="r", pacer=pace.Pacer(sleep_fn=clk.sleep, clock=clk.now),
    )
    run = json.loads((tmp_path / "replay" / "sub-001" / "r" / "run.json").read_text())
    assert run["status"] == "ok"
    assert "wall_elapsed_s" in run and "rate_ratio" in run and "wall_started_iso" in run


def test_abort_writes_run_json_and_no_summary(data_root, tmp_path):
    """Ctrl+C：run.json=aborted，已写内容保留，不写 summary 行（D3 精神）。"""

    class AbortClock:
        def __init__(self):
            self.t = 0.0

        def now(self):
            return self.t

        def sleep(self, dt):
            self.t += dt
            if self.t >= 0.6:  # 第 3 段结束时（墙钟 0.6s）中止
                raise KeyboardInterrupt

    clk = AbortClock()
    status = monitor.run_replay(
        "001", duration_sec=2.0, root=data_root, out_root=tmp_path,
        run_id="x", pacer=pace.Pacer(sleep_fn=clk.sleep, clock=clk.now),
    )
    assert status == "aborted"
    run_dir = tmp_path / "replay" / "sub-001" / "x"
    rows = _read_jsonl(run_dir / "replay.jsonl")
    assert rows[-1]["type"] != "summary"
    run = json.loads((run_dir / "run.json").read_text())
    assert run["status"] == "aborted"


def test_preflight_failure_writes_nothing(data_root, tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        monitor.run_replay("999", root=data_root, out_root=tmp_path, run_id="z")
    assert not (tmp_path / "replay").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_monitor.py -v`
Expected: FAIL —— `ModuleNotFoundError: bci_sys.replay.monitor`。

- [ ] **Step 3: 实现 `src/bci_sys/replay/monitor.py`**

```python
"""回放主循环：preflight → 分段 1x 回放 → 终端显示 + JSONL/run.json 落盘。

确定性（A4）：replay.jsonl 只含数据侧量，两次同参运行字节级一致；
墙钟量隔离到 run.json。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .. import config, gitmeta, io_bids
from . import pace, segments

SCHEMA_VERSION = "1.0"


def _jsonable(v):
    """pandas/NA → JSON 可序列化。"""
    if v is None or pd.isna(v):
        return None
    if isinstance(v, (int, float, str, bool)):
        return v
    return str(v)


def _event_kind(row) -> str:
    if row["trial_type"] == "dropped_samples":
        return "dropped"
    nb = row.get("nback_level")
    if pd.notna(nb) and int(nb) in (1, 2, 3, 4):
        return "trial"
    return "marker"


def _event_line(t: float, row) -> dict:
    nb = row.get("nback_level")
    return {
        "type": "event",
        "t": float(t),
        "trial_type": _jsonable(row["trial_type"]),
        "nback_level": int(nb) if pd.notna(nb) else None,
        "istutorial": bool(row["istutorial"]) if pd.notna(row.get("istutorial")) else None,
        "kind": _event_kind(row),
    }


def format_seg_line(seg: int, t1: float, p2p: float, ratio: float, marks: str) -> str:
    color = pace.rate_color(ratio)
    return (
        f"{seg:05d} │ t={t1:6.1f}s │ Fz p2p {p2p:6.1f}µV │ "
        f"rate {color}{ratio:.2f}x{pace.RESET} │ {marks}"
    )


def _event_marks(evs: pd.DataFrame) -> str:
    parts = []
    for _, r in evs.iterrows():
        if _event_kind(r) == "dropped":
            parts.append(f"{pace.RED}✗ dropped{pace.RESET}")
        elif _event_kind(r) == "trial":
            parts.append(f"◆ {int(r['nback_level'])}-back L{int(r['nback_level'])}")
    return " ".join(parts)


def run_replay(
    subj: str,
    duration_sec: float = 0.0,
    root: Path | str | None = None,
    out_root: Path | str = "outputs",
    run_id: str | None = None,
    pacer: pace.Pacer | None = None,
) -> str:
    """跑一次回放。返回状态 'ok' / 'aborted'。preflight 失败直接抛异常（不写文件）。"""
    io_bids.preflight(subj, root)  # 硬停，不写任何文件

    fz = io_bids.load_fz_uV(subj, root)
    events = io_bids.load_events(subj, root)
    n_planned = segments.n_planned_segments(fz.size, config.SEGMENT_SAMPLES, duration_sec)

    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_root) / "replay" / f"sub-{subj}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pacer = pacer or pace.Pacer()
    wall_start = datetime.now().isoformat(timespec="seconds")
    pacer.start()

    n_events = 0
    n_dropped = 0
    jsonl_path = run_dir / "replay.jsonl"
    status = "ok"
    last_ratio = 1.0
    n_emitted = 0

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "type": "meta",
            "schema_version": SCHEMA_VERSION,
            "code_version": gitmeta.code_version(),
            "subj": subj,
            "data_file": io_bids.vhdr_path(subj, root).name,
            "sfreq": config.SFREQ,
            "segment_sec": config.SEGMENT_SEC,
            "fz_channel": config.TARGET.channel,
            "n_segments_planned": n_planned,
        }, ensure_ascii=False) + "\n")

        try:
            for seg in segments.iter_segments(
                fz, config.SFREQ, config.SEGMENT_SAMPLES, duration_sec
            ):
                f.write(json.dumps({
                    "type": "seg",
                    "seg": seg.seg,
                    "t0": seg.t0,
                    "t1": seg.t1,
                    "fz_uV": [float(v) for v in seg.samples],
                }, ensure_ascii=False) + "\n")

                evs = segments.segment_events(events, seg.t0, seg.t1)
                for _, row in evs.iterrows():
                    line = _event_line(row["onset"], row)
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    n_events += 1
                    if line["kind"] == "dropped":
                        n_dropped += 1

                p2p = float(seg.samples.max() - seg.samples.min())
                ratio = pacer.rate_ratio(seg.t1)
                last_ratio = ratio
                print(format_seg_line(seg.seg, seg.t1, p2p, ratio, _event_marks(evs)))

                pacer.wait_after_segment(seg.seg)
                n_emitted += 1
        except KeyboardInterrupt:
            status = "aborted"

        if status == "ok":
            f.write(json.dumps({
                "type": "summary",
                "n_segments": n_emitted,
                "data_elapsed_s": n_emitted * config.SEGMENT_SEC,
                "tail_s": segments.tail_seconds(fz.size, duration_sec=duration_sec),
                "n_events": n_events,
                "n_dropped": n_dropped,
            }, ensure_ascii=False) + "\n")

    run_json = {
        "subj": subj,
        "status": status,
        "wall_started_iso": wall_start,
        "wall_elapsed_s": round(pacer.wall_elapsed(), 3),
        "rate_ratio": round(last_ratio, 4),
        "n_segments": n_emitted,
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return status
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_monitor.py -v`
Expected: 5 passed。

- [ ] **Step 5: Commit**

```bash
git add src/bci_sys/replay/monitor.py tests/test_monitor.py
git commit -m "feat: 回放主循环 + 确定性 JSONL（字节一致）+ run.json + 中止处理"
```

---

## Task 9: `replay/cli.py` —— replay-monitor 入口

**Files:**
- Create: `src/bci_sys/replay/cli.py`
- Test: `tests/test_replay_cli.py`

- [ ] **Step 1: 写失败测试 `tests/test_replay_cli.py`**

```python
import json
import subprocess
import sys


def test_cli_runs_limited_replay(data_root, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.replay.cli",
         "--subj", "001", "--duration-sec", "2",
         "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "rate" in r.stdout
    run_dir = tmp_path / "replay" / "sub-001"
    assert run_dir.exists()
    js = list(run_dir.rglob("replay.jsonl"))
    assert len(js) == 1
    rows = [json.loads(l) for l in js[0].read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["type"] == "summary" and rows[-1]["n_segments"] == 10


def test_cli_bad_subject_exits_nonzero_and_writes_nothing(data_root, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.replay.cli",
         "--subj", "999", "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "PREFLIGHT" in r.stderr or "preflight" in r.stderr.lower()
    assert not (tmp_path / "replay").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_replay_cli.py -v`
Expected: FAIL —— `No module named bci_sys.replay.cli`。

- [ ] **Step 3: 实现 `src/bci_sys/replay/cli.py`**

```python
"""replay-monitor 命令行入口。"""
from __future__ import annotations

import argparse
import sys

from . import monitor


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="replay-monitor", description="1x 回放 ds007169")
    ap.add_argument("--subj", required=True, help="被试编号，如 001")
    ap.add_argument("--duration-sec", type=float, default=0.0,
                    help="回放时长（秒）；0=全量（默认）")
    ap.add_argument("--data-root", default=None, help="ds007169 根目录（默认项目内）")
    ap.add_argument("--out-dir", default="outputs", help="产物根目录")
    args = ap.parse_args(argv)

    try:
        status = monitor.run_replay(
            args.subj, duration_sec=args.duration_sec,
            root=args.data_root, out_root=args.out_dir,
        )
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"PREFLIGHT FAIL: {e}", file=sys.stderr)
        return 2

    if status == "aborted":
        print("\n[中止] 已写 run.json(status=aborted)，本次数据标记为中止。", file=sys.stderr)
        return 130
    print(f"[完成] 产物在 {args.out_dir}/replay/sub-{args.subj}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_replay_cli.py -v`
Expected: 2 passed。

- [ ] **Step 5: 手工冒烟（看真实终端输出）**

Run: `uv run replay-monitor --subj 001 --duration-sec 5`
Expected: 终端 5 行/秒滚动 25 行，rate 显示绿色 1.00x；`outputs/replay/sub-001/<时间戳>/` 下有 replay.jsonl 与 run.json。

- [ ] **Step 6: Commit**

```bash
git add src/bci_sys/replay/cli.py tests/test_replay_cli.py
git commit -m "feat: replay-monitor CLI（--duration-sec 0=全量，preflight 失败非零退出）"
```

---

## Task 10: C3 物理隔离边界测试

**Files:**
- Create: `tests/test_boundary.py`

- [ ] **Step 1: 写测试 `tests/test_boundary.py`**

```python
import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "bci_sys"


def _imports_of(pkg: pathlib.Path) -> set[str]:
    mods = set()
    for f in pkg.rglob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module)
            elif isinstance(n, ast.Import):
                mods.update(a.name for a in n.names)
    return mods


def test_replay_does_not_import_offline():
    """C3：实时路径不得 import 离线路径。"""
    mods = _imports_of(SRC / "replay")
    assert not any(m == "bci_sys.offline" or m.startswith("bci_sys.offline.")
                   for m in mods), f"replay 违规 import offline: {mods}"


def test_offline_does_not_import_replay():
    """C3：离线路径不得 import 实时路径。"""
    mods = _imports_of(SRC / "offline")
    assert not any(m == "bci_sys.replay" or m.startswith("bci_sys.replay.")
                   for m in mods), f"offline 违规 import replay: {mods}"
```

- [ ] **Step 2: 跑测试确认通过（此刻 offline 还是空包，应通过）**

Run: `uv run pytest tests/test_boundary.py -v`
Expected: 2 passed。后续 Task 11-14 写 offline 代码时，若违规 import 这条测试会立刻变红。

- [ ] **Step 3: Commit**

```bash
git add tests/test_boundary.py
git commit -m "test: C3 物理隔离边界（AST 扫描 replay/offline 互不 import）"
```

---

## Task 11: `offline/theta.py` —— 全程滤波 → epoch → multitaper

**Files:**
- Create: `src/bci_sys/offline/theta.py`
- Test: `tests/test_theta.py`

- [ ] **Step 1: 写失败测试 `tests/test_theta.py`**

```python
import numpy as np

from bci_sys import config
from bci_sys.offline import theta


def _sine(freq, amp=10.0, sec=30.0, sfreq=250.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * sfreq)) / sfreq
    return amp * np.sin(2 * np.pi * freq * t) + 0.5 * rng.standard_normal(t.size)


def test_theta_power_6hz_far_exceeds_20hz():
    sfreq = 250.0
    eps6 = np.stack([_sine(6.0)[:375] for _ in range(5)])
    eps20 = np.stack([_sine(20.0)[:375] for _ in range(5)])
    p6 = theta.theta_power_uV2(eps6, sfreq).mean()
    p20 = theta.theta_power_uV2(eps20, sfreq).mean()
    assert p6 > 50 * p20


def test_theta_power_matches_theoretical_amp_squared_half():
    # 纯正弦 A*sin(2πft) 全带功率 ≈ A²/2；theta 带含 6Hz，应接近 A²/2
    sfreq = 250.0
    t = np.arange(375) / sfreq
    epoch = 10.0 * np.sin(2 * np.pi * 6 * t)
    p = theta.theta_power_uV2(epoch[None, :], sfreq)[0]
    assert abs(p - (10.0 ** 2) / 2) / (10.0 ** 2 / 2) < 0.25  # 容差 25%


def test_slice_epochs_shape_and_no_overlap_counts():
    # 4 个 onset，间隔 1.7s，窗口 1.5s → 4 个 375 点 epoch
    sfreq = 250.0
    x = np.zeros(int(20 * sfreq))
    onsets = np.array([1.0, 2.7, 4.4, 6.1])
    eps = theta.slice_epochs(x, onsets, sfreq)
    assert eps.shape == (4, 375)


def test_slice_epochs_drops_incomplete_tail():
    sfreq = 250.0
    x = np.zeros(int(10.0 * sfreq))  # 10 秒信号
    onsets = np.array([1.0, 9.5])    # 9.5s 起的窗口(到 11.0s)超出末尾 → 丢弃
    eps = theta.slice_epochs(x, onsets, sfreq)
    assert eps.shape == (1, 375)


def test_pipeline_on_sub001_produces_400_epochs(data_root):
    from bci_sys import io_bids

    fz = io_bids.load_fz_uV("001", data_root)
    ev = io_bids.load_events("001", data_root)
    tr = io_bids.real_trial_events(ev)
    ff = theta.filter_whole(fz)
    eps = theta.slice_epochs(ff, tr["onset"].values)
    powers = theta.theta_power_uV2(eps)
    assert eps.shape[1] == 375
    assert 395 <= eps.shape[0] <= 400  # 末尾不完整 epoch 可能被丢
    assert powers.shape[0] == eps.shape[0]
    assert np.all(np.isfinite(powers)) and np.all(powers > 0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_theta.py -v`
Expected: FAIL —— `ModuleNotFoundError: bci_sys.offline.theta`。

- [ ] **Step 3: 实现 `src/bci_sys/offline/theta.py`**

```python
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

    return mne.filter.filter_data(
        x.astype(np.float64), sfreq, 1.0, 30.0, method="fir", verbose="ERROR"
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
    bandwidth = nw / t_epoch  # NW = bandwidth * T
    psd, freqs = psd_array_multitaper(
        epochs, sfreq, fmin=band[0], fmax=band[1],
        bandwidth=bandwidth, normalization="full", verbose="ERROR",
    )
    return np.trapezoid(psd, freqs, axis=1)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_theta.py -v`
Expected: 5 passed。

- [ ] **Step 5: Commit**

```bash
git add src/bci_sys/offline/theta.py tests/test_theta.py
git commit -m "feat: 离线 theta 管线（全程滤波→epoch→multitaper，顺序硬约束）"
```

---

## Task 12: `offline/probe.py` —— 分组、门控、CSV/PNG

**Files:**
- Create: `src/bci_sys/offline/probe.py`
- Test: `tests/test_probe.py`

- [ ] **Step 1: 写失败测试 `tests/test_probe.py`**

```python
import pandas as pd

from bci_sys.offline import probe


def test_gate_passes_strictly_monotonic():
    assert probe.gate_passes({1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}) is True


def test_gate_fails_flat_and_nonmonotonic():
    assert probe.gate_passes({1: 1.0, 2: 1.0, 3: 2.0, 4: 3.0}) is False  # 相等不算
    assert probe.gate_passes({1: 1.0, 2: 3.0, 3: 2.0, 4: 4.0}) is False


def test_median_by_load():
    df = pd.DataFrame({"load": [1, 1, 2, 2], "theta_uV2": [1.0, 3.0, 10.0, 20.0]})
    med = probe.median_by_load(df)
    assert med[1] == 2.0 and med[2] == 15.0


def test_run_probe_sub001_passes_and_writes_outputs(data_root, tmp_path):
    ok = probe.run_probe("001", root=data_root, out_dir=tmp_path)
    assert ok is True  # 硬门控：1→4-back 单调上升
    d = tmp_path / "theta" / "001"
    csv = d / "theta_by_load.csv"
    png = d / "theta_by_load.png"
    assert csv.exists() and png.exists()
    out = pd.read_csv(csv)
    assert list(out["load"]) == [1, 2, 3, 4]
    assert list(out.columns) == ["load", "n_epochs", "median_theta_uV2"]
    assert (out["n_epochs"] > 90).all()


def test_run_all_writes_summary(data_root, tmp_path):
    results = probe.run_all(root=data_root, out_dir=tmp_path, subs=["001", "002"])
    assert len(results) == 2
    assert (tmp_path / "theta" / "all" / "summary.csv").exists()
    assert (tmp_path / "theta" / "all" / "summary.png").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_probe.py -v`
Expected: FAIL —— `ModuleNotFoundError: bci_sys.offline.probe`。

- [ ] **Step 3: 实现 `src/bci_sys/offline/probe.py`**

```python
"""按 n-back 负荷分组、中位数、单调门控（IT2 硬判据）、出 CSV/PNG。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config, io_bids
from . import theta


def median_by_load(df: pd.DataFrame) -> dict[int, float]:
    return {int(k): float(v) for k, v in df.groupby("load")["theta_uV2"].median().items()}


def gate_passes(medians: dict[int, float]) -> bool:
    """硬门控：median(L1) < L2 < L3 < L4 严格成立。"""
    vals = [medians[i] for i in (1, 2, 3, 4)]
    return all(vals[i] < vals[i + 1] for i in range(3))


def _compute(subj: str, root=None) -> pd.DataFrame:
    """返回每试次 DataFrame: load, theta_uV2（已全程滤波→epoch→谱）。"""
    fz = io_bids.load_fz_uV(subj, root)
    ev = io_bids.load_events(subj, root)
    tr = io_bids.real_trial_events(ev)
    fz_f = theta.filter_whole(fz)  # 先全程滤波
    eps = theta.slice_epochs(fz_f, tr["onset"].values)  # 再切 epoch
    powers = theta.theta_power_uV2(eps)
    # 末尾不完整 epoch 可能被丢，截取与 eps 等长的试次表（按 onset 排序，丢的总是最后几个）
    n = powers.shape[0]
    return pd.DataFrame({"load": tr["nback_level"].values[:n], "theta_uV2": powers})


def run_probe(subj: str, root=None, out_dir="outputs") -> bool:
    """单被试：出 CSV+PNG，返回门控是否通过。"""
    io_bids.preflight(subj, root)
    df = _compute(subj, root)
    med = median_by_load(df)
    n_ep = {int(k): int(v) for k, v in df.groupby("load").size().items()}

    out = Path(out_dir) / "theta" / subj
    out.mkdir(parents=True, exist_ok=True)
    csv = pd.DataFrame(
        {"load": [1, 2, 3, 4],
         "n_epochs": [n_ep.get(i, 0) for i in (1, 2, 3, 4)],
         "median_theta_uV2": [med[i] for i in (1, 2, 3, 4)]}
    )
    csv.to_csv(out / "theta_by_load.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([1, 2, 3, 4], [med[i] for i in (1, 2, 3, 4)], "o-")
    ax.set(xlabel="n-back level", ylabel="median Fz theta power (µV²)",
           title=f"sub-{subj} theta by n-back load")
    ax.set_xticks([1, 2, 3, 4])
    fig.tight_layout()
    fig.savefig(out / "theta_by_load.png", dpi=120)
    plt.close(fig)
    return gate_passes(med)


def run_all(root=None, out_dir="outputs", subs=None) -> list[dict]:
    """多被试汇总（参考用，不作硬门控）。"""
    subs = subs or config.RECOMMENDED_SUBS
    rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in subs:
        df = _compute(s, root)
        med = median_by_load(df)
        ok = gate_passes(med)
        rows.append({"subj": s, "m1": med[1], "m2": med[2], "m3": med[3],
                     "m4": med[4], "pass": ok})
        ax.plot([1, 2, 3, 4], [med[i] for i in (1, 2, 3, 4)], "o-", label=f"sub-{s}{'' if ok else ' FAIL'}")
    out = Path(out_dir) / "theta" / "all"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    ax.set(xlabel="n-back level", ylabel="median Fz theta power (µV²)",
           title="theta by load — recommended subjects")
    ax.set_xticks([1, 2, 3, 4])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=120)
    plt.close(fig)
    return rows
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_probe.py -v`
Expected: 5 passed。**`test_run_probe_sub001_passes...` 就是 IT2 验收判据**——若它失败，说明 theta 计算或通道映射有错，不许进 IT3。

- [ ] **Step 5: Commit**

```bash
git add src/bci_sys/offline/probe.py tests/test_probe.py
git commit -m "feat: theta 负荷分组 + 单调硬门控 + CSV/PNG（sub-001 门控 PASS）"
```

---

## Task 13: `offline/cli.py` —— theta-probe 入口

**Files:**
- Create: `src/bci_sys/offline/cli.py`
- Test: `tests/test_theta_cli.py`

- [ ] **Step 1: 写失败测试 `tests/test_theta_cli.py`**

```python
import subprocess
import sys


def test_theta_cli_sub001_pass_exit0(data_root, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.offline.cli",
         "--subj", "001", "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "PASS" in r.stdout
    assert (tmp_path / "theta" / "001" / "theta_by_load.png").exists()


def test_theta_cli_all_always_exit0(data_root, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.offline.cli",
         "--all", "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0  # --all 纯参考，退出码恒 0
    assert (tmp_path / "theta" / "all" / "summary.csv").exists()


def test_theta_cli_requires_subj_or_all(data_root, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.offline.cli",
         "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_theta_cli.py -v`
Expected: FAIL —— `No module named bci_sys.offline.cli`。

- [ ] **Step 3: 实现 `src/bci_sys/offline/cli.py`**

```python
"""theta-probe 命令行入口。"""
from __future__ import annotations

import argparse
import sys

from .. import config
from . import probe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="theta-probe", description="离线 Fz theta 负荷检验")
    ap.add_argument("--subj", help="单被试编号，如 001（硬门控，失败退出码 1）")
    ap.add_argument("--all", action="store_true", help="跑 7 个推荐被试汇总（参考，退出码恒 0）")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args(argv)

    if not args.subj and not args.all:
        ap.error("必须指定 --subj 或 --all")

    try:
        if args.all:
            rows = probe.run_all(root=args.data_root, out_dir=args.out_dir)
            for r in rows:
                print(f"sub-{r['subj']}: {'PASS' if r['pass'] else 'FAIL'} "
                      f"({r['m1']:.1f} < {r['m2']:.1f} < {r['m3']:.1f} < {r['m4']:.1f})"
                      if r['pass'] else
                      f"sub-{r['subj']}: FAIL ({r['m1']:.1f}, {r['m2']:.1f}, {r['m3']:.1f}, {r['m4']:.1f})")
            print(f"[完成] 汇总在 {args.out_dir}/theta/all/（参考，不作硬门控）")
            return 0

        ok = probe.run_probe(args.subj, root=args.data_root, out_dir=args.out_dir)
        if ok:
            print(f"PASS: sub-{args.subj} theta 1→4-back 单调上升")
            print(f"[完成] 产物在 {args.out_dir}/theta/{args.subj}/")
            return 0
        print(f"FAIL: sub-{args.subj} theta 不单调——检查 theta 计算或通道映射，不许进 IT3")
        return 1
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"PREFLIGHT FAIL: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_theta_cli.py -v`
Expected: 3 passed。

- [ ] **Step 5: Commit**

```bash
git add src/bci_sys/offline/cli.py tests/test_theta_cli.py
git commit -m "feat: theta-probe CLI（--subj 硬门控退出码 1 / --all 恒 0）"
```

---

## Task 14: 全量测试与验收产物（evidence/）

**Files:**
- Create: `evidence/`（截图/PNG）

- [ ] **Step 1: 跑完整测试套件**

Run: `uv run pytest -v`
Expected: 全部 passed（约 40 个测试；需要 ds007169 的测试在数据就位时跑，数据缺失自动 skip）。

- [ ] **Step 2: IT1 验收——限时回放截图**

Run: `uv run replay-monitor --subj 001 --duration-sec 60`
- 终端滚动 60 秒，rate 稳定绿色 1.00x，能看到 `✗ dropped` 与 `◆ N-back` 标记。
- **手工截终端图**（含速率比），存为 `evidence/IT1-replay-速率比.png`。
- 核对：`outputs/replay/sub-001/<ts>/replay.jsonl` 段数 = 300；`run.json` status=ok。

- [ ] **Step 3: IT1 确定性人工核对（A4）**

Run:
```bash
uv run replay-monitor --subj 001 --duration-sec 30 --out-dir outputs/det1
uv run replay-monitor --subj 001 --duration-sec 30 --out-dir outputs/det2
diff <(cat outputs/det1/replay/sub-001/*/replay.jsonl) <(cat outputs/det2/replay/sub-001/*/replay.jsonl) && echo IDENTICAL
```
Expected: 输出 `IDENTICAL`。

- [ ] **Step 4: IT2 验收——单调曲线 PNG**

Run: `uv run theta-probe --subj 001`
Expected: 打印 `PASS`，退出码 0。
复制产物：`cp outputs/theta/001/theta_by_load.png evidence/IT2-theta-单调曲线.png`

- [ ] **Step 5: IT2 多被试汇总（参考）**

Run: `uv run theta-probe --all`
Expected: 7 人逐行 PASS/FAIL，生成 `outputs/theta/all/summary.png` 与 `summary.csv`。把汇总图也存 `evidence/IT2-theta-多被试汇总.png`。

- [ ] **Step 6: 提交验收产物**

```bash
git add evidence/
git commit -m "evidence: IT1 速率比截图 + IT2 theta 单调曲线/多被试汇总（IT1+IT2 验收）"
```

- [ ] **Step 7: 推送（本机需绕过 gitclone 镜像）**

```bash
git -c url.https://github.com/.insteadOf=https://github.com/ push
```
Expected: 推送到 https://github.com/jiajiano654321/neurofeedback 的 main。

---

## 完成判据对照

| spec 要求 | 落地任务 |
|---|---|
| 单包双入口、replay/offline 物理隔离 | Task 1（结构）、Task 10（AST 边界测试） |
| C1 direction 必填无默认 | Task 2 |
| 单位陷阱：不乘 1e6 + preflight std 硬门 | Task 3（docstring）、Task 4（硬检查） |
| 通道按名查找 | Task 3 |
| 因果分段、4792 段、0.108s 尾、段边界截断 | Task 5 |
| 1x 节拍、速率颜色 | Task 6 |
| D2 代码版本 | Task 7 |
| JSONL schema（meta/seg/event/summary）、事件左闭右开 | Task 8、Task 5 |
| A4 字节级确定性、run.json 隔离墙钟 | Task 8 |
| 中止 aborted、preflight 失败不写文件 | Task 8、Task 9 |
| 滤波顺序硬约束（全程滤波→epoch） | Task 11 |
| multitaper theta、400 epoch、A²/2 合成校验 | Task 11 |
| 单调硬门控、CSV/PNG、--all | Task 12 |
| CLI 退出码（门控失败 1、--all 恒 0、preflight 2） | Task 9、Task 13 |
| evidence 截图/PNG | Task 14 |
