# IT3 `qc_monitor` 实现计划（TDD）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 1x 回放的实时因果路径上新增 `qc-monitor`：流式因果 1–30Hz 带通 → 120s 校准冻结 EOG 回归系数 → 每 200ms 段 L1（残差 p2p>100µV）/L3（FP1/FP2 任一 p2p>200µV）裁决，无效段标红记因，产出 qc.jsonl / run.json / 有效率时间轴 PNG。

**Architecture:** 全部新代码在 `src/bci_sys/replay/`（实时侧，严禁 import offline、严禁 filtfilt）。分层：`causal_filter.py`（逐块携带 zi 的单向 lfilter）→ `qc.py`（纯逻辑：OLS + 状态机，无 I/O）→ `qc_timeline.py`（事后绘图，读 jsonl，不回流裁决）→ `qc_monitor.py`（编排/终端/落盘）→ `qc_cli.py`（退出码 0/2/130）。共享层只改 `config.py`、`io_bids.py`。

**Tech Stack:** Python 3.12 / scipy.signal（butter + lfilter，已是依赖）/ numpy / matplotlib Agg / pytest；分支 `feature/it3-qc-monitor`（已存在，spec 已提交）。

**设计依据：** `docs/superpowers/specs/2026-09-09-it3-qc-monitor-design.md`（已批准）。冻结参数不许运行期修改（C2）；实时不得用未来数据（C3）；无效只标记不删除（C5）；硬停不提示（C7）；qc.jsonl 字节级确定性、墙钟只进 run.json（A4）。

**现状（已核实，勿重复探查）：**
- 当前 57 个测试全绿；`tests/test_boundary.py` 用 rglob 扫描整个 `replay/` 与 `offline/` 目录，**新增文件自动被边界测试覆盖，无需改该文件**。
- mne 读出的 ds007169 数值**已是 µV，严禁 ×1e6**（io_bids docstring 有单位陷阱说明）。
- `segments.iter_segments(fz, sfreq, seg_samples, duration_sec)` 产 `Segment(seg,t0,t1,samples)`；段 k 的 t1 = `(k+1)*50/250`，第 600 段（seg=599）t1 恰为 120.0（整除精确）。
- FakeClock/FakePacer 范式见 `tests/test_monitor.py`：`pace.Pacer(sleep_fn=clk.sleep, clock=clk.now)`，假时钟让 130s 回放瞬时完成。
- CLI 子进程测试范式见 `tests/test_replay_cli.py`：`[sys.executable, "-m", "bci_sys.replay.qc_cli", ...]`，必须 `encoding="utf-8"`，模块文件需 `if __name__ == "__main__"` 守卫。
- 出图用英文标签 + `matplotlib.use("Agg")`，范式见 `src/bci_sys/offline/probe.py`（µ 在 DejaVu Sans 中可显示，不要用中文标题）。
- scipy 已在 pyproject 依赖中，无需新增依赖。

---

## Task 1: config.py 增加 IT3 冻结常量

**Files:**
- Modify: `src/bci_sys/config.py`（在 `UNIT_STD_RANGE_UV` 之后追加）
- Test: `tests/test_config.py`（追加测试）

- [ ] **Step 1: 先写失败测试**

在 `tests/test_config.py` 末尾追加：

```python
def test_qc_frozen_constants():
    # IT3 冻结参数（S2 语义，运行期无修改入口）
    assert config.QC_CALIBRATION_SEC == 120.0
    assert config.QC_FILTER_WARMUP_SEC == 10.0
    assert config.QC_L1_P2P_UV == 100.0
    assert config.QC_L3_EOG_P2P_UV == 200.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py::test_qc_frozen_constants -v`
Expected: FAIL，`AttributeError: module 'bci_sys.config' has no attribute 'QC_CALIBRATION_SEC'`

- [ ] **Step 3: 实现**

在 `src/bci_sys/config.py` 的 `UNIT_STD_RANGE_UV` 行之后追加：

```python
# --- IT3 实时 QC（冻结，C2：运行期无修改入口） ---
QC_CALIBRATION_SEC = 120.0      # S3 校准窗长度
QC_FILTER_WARMUP_SEC = 10.0     # 估 EOG 回归系数时剔除的滤波器启动样本
QC_L1_P2P_UV = 100.0            # L1：去眼电后 Fz 残差 200ms 峰峰值上限
QC_L3_EOG_P2P_UV = 200.0        # L3：FP1/FP2 任一路 200ms 峰峰值上限（否决项）
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS（含原有 5 个测试）

- [ ] **Step 5: 提交**

```bash
git add src/bci_sys/config.py tests/test_config.py
git commit -m "feat(qc): IT3 冻结常量（校准120s/预热10s/L1=100/L3=200µV）"
```

---

## Task 2: io_bids 多通道加载 + preflight 强制 FP1/FP2

**Files:**
- Modify: `src/bci_sys/io_bids.py`
- Test: `tests/test_io_bids.py`（追加真数据测试）、`tests/test_preflight.py`（追加缺通道测试）

- [ ] **Step 1: 先写失败测试**

在 `tests/test_io_bids.py` 末尾追加（若该文件尚无 `import numpy`，在文件头补 `import numpy as np`）：

```python
def test_load_channels_uv_returns_named_channels(data_root):
    ch = io_bids.load_channels_uV("001", ["Fz", "FP1", "FP2"], data_root)
    assert set(ch.keys()) == {"Fz", "FP1", "FP2"}
    # Fz 必须与既有单通道取数完全一致（同一 µV 取数路径，禁止二次换算）
    fz = io_bids.load_fz_uV("001", data_root)
    np.testing.assert_array_equal(ch["Fz"], fz)
    assert ch["FP1"].size == ch["FP2"].size == fz.size
```

在 `tests/test_preflight.py` 末尾追加两个缺通道测试：

```python
def test_preflight_fails_when_raw_missing_fp2(tmp_path, monkeypatch):
    """mne 侧缺 FP2：channel_index 直接 KeyError（硬停）。"""
    from types import SimpleNamespace

    rng = np.random.default_rng(2)
    n = 250 * 30
    names = ["Fz", "FP1"] + [f"E{i:02d}" for i in range(17)]  # 19 导，无 FP2
    data = np.stack(
        [30.0 * np.sin(2 * np.pi * 6 * np.arange(n) / 250.0) + 5.0 * rng.standard_normal(n)]
        + [rng.standard_normal(n) for _ in names[1:]]
    )
    fake_raw = SimpleNamespace(
        ch_names=names, info={"sfreq": 250.0}, get_data=lambda: data
    )
    monkeypatch.setattr(io_bids, "load_raw", lambda subj, root=None: fake_raw)

    eeg_dir = tmp_path / "sub-001" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-001_task-nback_events.tsv").write_text("onset\tnback_level\n", encoding="utf-8")

    with pytest.raises(KeyError, match="FP2"):
        io_bids.preflight("001", tmp_path)


def test_preflight_fails_when_channels_tsv_missing_fp2(tmp_path, monkeypatch):
    """mne 侧有 FP2 但 channels.tsv 没有：按名检查仍 KeyError。"""
    from types import SimpleNamespace
    import pandas as pd

    rng = np.random.default_rng(3)
    n = 250 * 30
    base = ["Fz", "FP1"] + [f"E{i:02d}" for i in range(17)]  # tsv 19 导无 FP2
    names = base + ["FP2"]
    data = np.stack(
        [30.0 * np.sin(2 * np.pi * 6 * np.arange(n) / 250.0) + 5.0 * rng.standard_normal(n)]
        + [rng.standard_normal(n) for _ in names[1:]]
    )
    fake_raw = SimpleNamespace(
        ch_names=names, info={"sfreq": 250.0}, get_data=lambda: data
    )
    monkeypatch.setattr(io_bids, "load_raw", lambda subj, root=None: fake_raw)

    eeg_dir = tmp_path / "sub-001" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-001_task-nback_events.tsv").write_text("onset\tnback_level\n", encoding="utf-8")
    pd.DataFrame({"name": base, "type": ["EEG"] * 19}).to_csv(
        eeg_dir / "sub-001_task-nback_channels.tsv", sep="\t", index=False
    )

    with pytest.raises(KeyError, match="FP2"):
        io_bids.preflight("001", tmp_path)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_io_bids.py::test_load_channels_uv_returns_named_channels tests/test_preflight.py::test_preflight_fails_when_raw_missing_fp2 tests/test_preflight.py::test_preflight_fails_when_channels_tsv_missing_fp2 -v`
Expected: FAIL，`AttributeError: module 'bci_sys.io_bids' has no attribute 'load_channels_uV'`；两个 preflight 测试因未检查 EOG 而“假通过”（DID NOT RAISE / 失败）。

- [ ] **Step 3: 实现 load_channels_uV**

在 `src/bci_sys/io_bids.py` 的 `load_fz_uV` 函数之后插入：

```python
def load_channels_uV(
    subj: str, names: list[str], root: Path | str | None = None
) -> dict[str, np.ndarray]:
    """按名返回多通道全程数据，单位 µV。

    【单位陷阱】同 load_fz_uV：mne 输出经实测已是 µV，【严禁再乘 1e6】。
    缺任一名直接 KeyError（通道映射安全靠显式按名查找，不靠位置假设）。
    """
    raw = load_raw(subj, root)
    data = raw.get_data()
    return {name: data[channel_index(raw, name)] for name in names}
```

- [ ] **Step 4: preflight 增加 FP1/FP2 按名检查**

在 `src/bci_sys/io_bids.py` 的 `preflight` 中，把第 3 步通道检查段：

```python
    # 3. 通道：mne 侧 Fz 存在（不存在即抛 KeyError）；EEG 通道数以 channels.tsv 为准
    #    （mne 读 vhdr 可能把 24 导全标成 eeg，不能信它的类型计数）
    idx = channel_index(raw, config.TARGET.channel)
    ch_tsv = _eeg_dir(subj, root) / f"sub-{subj}_task-nback_channels.tsv"
    if not ch_tsv.exists():
        raise FileNotFoundError(f"channels.tsv 缺失：{ch_tsv}")
    chans = pd.read_csv(ch_tsv, sep="\t")
    n_eeg = int((chans["type"] == "EEG").sum())
    if n_eeg != 19:
        raise ValueError(f"channels.tsv 中 EEG 通道数 {n_eeg} != 19")
    if config.TARGET.channel not in set(chans["name"]):
        raise KeyError(f"channels.tsv 中无通道 {config.TARGET.channel}")
```

替换为：

```python
    # 3. 通道：Fz + EOG 代理 FP1/FP2 必须在 mne 侧与 channels.tsv 双侧按名存在
    #    （不存在即抛 KeyError）；EEG 通道数以 channels.tsv 为准
    #    （mne 读 vhdr 可能把 24 导全标成 eeg，不能信它的类型计数）
    required = [config.TARGET.channel, *config.EOG_PROXY_CHANNELS]
    for name in required:
        channel_index(raw, name)
    idx = channel_index(raw, config.TARGET.channel)
    ch_tsv = _eeg_dir(subj, root) / f"sub-{subj}_task-nback_channels.tsv"
    if not ch_tsv.exists():
        raise FileNotFoundError(f"channels.tsv 缺失：{ch_tsv}")
    chans = pd.read_csv(ch_tsv, sep="\t")
    n_eeg = int((chans["type"] == "EEG").sum())
    if n_eeg != 19:
        raise ValueError(f"channels.tsv 中 EEG 通道数 {n_eeg} != 19")
    ch_names = set(chans["name"])
    for name in required:
        if name not in ch_names:
            raise KeyError(f"channels.tsv 中无通道 {name}")
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_io_bids.py tests/test_preflight.py -v`
Expected: PASS（含原有 `test_preflight_passes_on_sub001`——实测 sub-001 确有 FP1/FP2）

- [ ] **Step 6: 提交**

```bash
git add src/bci_sys/io_bids.py tests/test_io_bids.py tests/test_preflight.py
git commit -m "feat(qc): load_channels_uV 多通道加载，preflight 强制 FP1/FP2"
```

---

## Task 3: 流式因果 1–30Hz 带通 causal_filter.py

**Files:**
- Create: `src/bci_sys/replay/causal_filter.py`
- Test: `tests/test_causal_filter.py`

- [ ] **Step 1: 先写失败测试**

创建 `tests/test_causal_filter.py`：

```python
"""流式因果带通：分块==整段单向 lfilter；追加未来数据不得改变过去（C3）。"""
import pathlib

import numpy as np

from bci_sys.replay import causal_filter as cf


def test_chunked_streaming_equals_oneshot_lfilter():
    rng = np.random.default_rng(0)
    # 含大直流偏移（实测 ~1.44e6 µV），验证启动后直流被高通干掉
    x = 1.44e6 + 20.0 * rng.standard_normal(250 * 5)

    stream = cf.CausalBandpass(["Fz"])
    ys = [
        stream.push_block({"Fz": x[s : s + 50]})["Fz"]
        for s in range(0, x.size, 50)
    ]
    y_chunk = np.concatenate(ys)

    np.testing.assert_allclose(y_chunk, cf.bandpass_oneshot(x), atol=1e-10)


def test_appending_future_data_does_not_change_past():
    """因果性（守 C3）：先喂 10 段记录输出；喂 20 段重跑，前 10 段逐点不变。"""
    rng = np.random.default_rng(3)
    x_short = rng.standard_normal(50 * 10) * 20.0
    x_long = np.concatenate([x_short, rng.standard_normal(50 * 10) * 20.0])

    f1 = cf.CausalBandpass(["c"])
    out_short = np.concatenate([
        f1.push_block({"c": x_short[s : s + 50]})["c"] for s in range(0, 500, 50)
    ])
    f2 = cf.CausalBandpass(["c"])
    out_long = np.concatenate([
        f2.push_block({"c": x_long[s : s + 50]})["c"] for s in range(0, 1000, 50)
    ])
    np.testing.assert_allclose(out_short, out_long[:500], atol=1e-12)


def test_multichannel_states_are_independent():
    rng = np.random.default_rng(4)
    xs = {c: rng.standard_normal(50 * 4) * (10 + i * 5)
          for i, c in enumerate(["Fz", "FP1", "FP2"])}
    stream = cf.CausalBandpass(["Fz", "FP1", "FP2"])
    out = {c: [] for c in xs}
    for s in range(0, 200, 50):
        block = stream.push_block({c: xs[c][s : s + 50] for c in xs})
        for c in xs:
            out[c].append(block[c])
    for c in xs:
        np.testing.assert_allclose(
            np.concatenate(out[c]), cf.bandpass_oneshot(xs[c]), atol=1e-10
        )


def test_dc_offset_removed_after_startup():
    # 稳态（跳过启动段）后输出应在零附近，不再含 1.44e6 直流
    x = np.full(250 * 5, 1.44e6)
    y = cf.bandpass_oneshot(x)
    assert abs(y[250 * 2 :]).max() < 1e-3


def test_module_never_calls_filtfilt():
    # 只禁止【调用】零相位滤波；docstring 里的警示文字不算违规
    import ast

    tree = ast.parse(pathlib.Path(cf.__file__).read_text(encoding="utf-8"))
    attrs = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    assert "filtfilt" not in attrs
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_causal_filter.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'bci_sys.replay.causal_filter'`

- [ ] **Step 3: 实现**

创建 `src/bci_sys/replay/causal_filter.py`：

```python
"""流式因果 1–30Hz 带通（实时路径专用，C3）。

【严禁 filtfilt / 零相位】零相位滤波要用到未来样本，只能出现在 offline/。
本模块只用单向 scipy.signal.lfilter：4 阶 Butterworth 高通（1Hz）串 4 阶
低通（30Hz），每节、每通道各自携带 zi 跨块延续。初始 zi=0，因此
50 点一块的流式输出与整段一次性同参数单向 lfilter 逐点一致（测试锁定）。
零初始状态造成的启动瞬态由 QC 校准时剔除前 10s 解决。
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .. import config


def design_filter_coeffs(
    sfreq: float = config.SFREQ,
    band_hz: tuple[float, float] = config.FILTER_BAND_HZ,
):
    """返回 ((b_hp, a_hp), (b_lp, a_lp))。频带冻结 1–30Hz。"""
    lo, hi = band_hz
    b_hp, a_hp = signal.butter(4, lo / (sfreq / 2.0), btype="high")
    b_lp, a_lp = signal.butter(4, hi / (sfreq / 2.0), btype="low")
    return (b_hp, a_hp), (b_lp, a_lp)


def bandpass_oneshot(
    x: np.ndarray,
    sfreq: float = config.SFREQ,
    band_hz: tuple[float, float] = config.FILTER_BAND_HZ,
) -> np.ndarray:
    """整段一次性单向带通（零初始 zi）。仅作流式实现的参照基准/非实时用途。"""
    (b_hp, a_hp), (b_lp, a_lp) = design_filter_coeffs(sfreq, band_hz)
    y = signal.lfilter(b_hp, a_hp, np.asarray(x, dtype=float))
    return signal.lfilter(b_lp, a_lp, y)


class _CausalSection:
    """单节单向 lfilter，zi 跨块延续。"""

    def __init__(self, b: np.ndarray, a: np.ndarray):
        self.b, self.a = b, a
        self.zi = np.zeros(max(len(a), len(b)) - 1)

    def push(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = signal.lfilter(
            self.b, self.a, np.asarray(x, dtype=float), zi=self.zi
        )
        return y


class CausalBandpass:
    """逐块推进的因果带通：先高通后低通串联，每通道独立状态。

    用法：
        bp = CausalBandpass(("Fz", "FP1", "FP2"))
        filtered = bp.push_block({"Fz": 块, "FP1": 块, "FP2": 块})
    """

    def __init__(
        self,
        channels: tuple[str, ...] | list[str],
        sfreq: float = config.SFREQ,
        band_hz: tuple[float, float] = config.FILTER_BAND_HZ,
    ):
        self.channels = tuple(channels)
        (b_hp, a_hp), (b_lp, a_lp) = design_filter_coeffs(sfreq, band_hz)
        self._hp = {c: _CausalSection(b_hp, a_hp) for c in self.channels}
        self._lp = {c: _CausalSection(b_lp, a_lp) for c in self.channels}

    def push_block(self, block: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """所有通道块长必须一致；返回同结构的滤波后块（单位与输入相同，µV）。"""
        sizes = {np.asarray(block[c]).size for c in self.channels}
        if len(sizes) != 1:
            raise ValueError(f"各通道块长不一致: {sizes}")
        out: dict[str, np.ndarray] = {}
        for c in self.channels:
            y = self._hp[c].push(block[c])
            out[c] = self._lp[c].push(y)
        return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_causal_filter.py -v`
Expected: PASS（5 个）

- [ ] **Step 5: 提交**

```bash
git add src/bci_sys/replay/causal_filter.py tests/test_causal_filter.py
git commit -m "feat(qc): 流式因果 1-30Hz 带通（逐块 zi，分块等价整段 lfilter）"
```

---

## Task 4: qc.py 纯逻辑 —— p2p 与冻结 EOG 回归（OLS）

**Files:**
- Create: `src/bci_sys/replay/qc.py`
- Test: `tests/test_qc.py`

- [ ] **Step 1: 先写失败测试**

创建 `tests/test_qc.py`：

```python
"""QC 纯逻辑：OLS 系数回收、warmup 剔除、非有限值硬错。"""
import numpy as np
import pytest

from bci_sys.replay import qc


def test_p2p_basic():
    assert qc.p2p_uv(np.array([3.0, -7.0, 10.0, 2.0])) == 17.0


def test_ols_recovers_known_coefficients():
    # 合成模型 Fz = 0.30*FP1 + 0.50*FP2 + 独立残差；120s @250Hz
    rng = np.random.default_rng(0)
    n = 250 * 120
    fp1 = rng.standard_normal(n) * 30.0
    fp2 = rng.standard_normal(n) * 25.0
    fz = 0.30 * fp1 + 0.50 * fp2 + rng.standard_normal(n) * 10.0
    b1, b2 = qc.fit_eog_coeffs(fz, fp1, fp2, start_sample=250 * 10)
    assert abs(b1 - 0.30) < 0.02
    assert abs(b2 - 0.50) < 0.02


def test_ols_warmup_exclusion_is_necessary_and_sufficient():
    """前 10s（2500 点）放【相关的】启动瞬态：含进去估错，剔除后估对。

    注意常数污染会被去均值消掉、不影响斜率，所以启动瞬态必须是
    Fz 与 FP1 相关、但比例错误（-3.0）的共同阶跃，才考得出 warmup 的作用。
    """
    rng = np.random.default_rng(1)
    n = 250 * 120
    fp1 = np.zeros(n)
    fp1[2500:] = rng.standard_normal(n - 2500) * 30.0
    fz = 0.40 * fp1 + rng.standard_normal(n) * 5.0
    fp1[:2500] = 5000.0     # 启动期共同大阶跃
    fz[:2500] = -15000.0    # Fz/FP1 比例 -3.0（错误系数，主导 OLS）
    b_bad, _ = qc.fit_eog_coeffs(fz, fp1, fp1 * 0, start_sample=0)
    b_good, _ = qc.fit_eog_coeffs(fz, fp1, fp1 * 0, start_sample=2500)
    assert abs(b_bad - 0.40) > 0.1
    assert abs(b_good - 0.40) < 0.02


def test_ols_rejects_nonfinite_and_bad_arguments():
    n = 1000
    fz = np.full(n, np.nan)
    with pytest.raises(ValueError):
        qc.fit_eog_coeffs(fz, np.zeros(n), np.zeros(n), start_sample=0)
    with pytest.raises(ValueError):
        qc.fit_eog_coeffs(np.zeros(n), np.zeros(n - 1), np.zeros(n), 0)
    with pytest.raises(ValueError):
        qc.fit_eog_coeffs(np.zeros(n), np.zeros(n), np.zeros(n), n + 1)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_qc.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'bci_sys.replay.qc'`

- [ ] **Step 3: 实现**

创建 `src/bci_sys/replay/qc.py`：

```python
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
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    if not np.all(np.isfinite(b)):
        raise ValueError("EOG 回归系数非有限值（NaN/Inf），校准失败")
    return float(b[0]), float(b[1])
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_qc.py -v`
Expected: PASS（4 个）

- [ ] **Step 5: 提交**

```bash
git add src/bci_sys/replay/qc.py tests/test_qc.py
git commit -m "feat(qc): p2p 与去均值无截距 OLS（warmup 剔除启动瞬态）"
```

---

## Task 5: qc.py 校准→运行状态机 QcJudge

**Files:**
- Modify: `src/bci_sys/replay/qc.py`（追加 QcJudge）
- Test: `tests/test_qc.py`（追加测试）

- [ ] **Step 1: 先写失败测试**

在 `tests/test_qc.py` 末尾追加：

```python
# ---------- QcJudge 状态机 ----------

def _blocks(fz, fp1, fp2, seg=50):
    for s in range(0, fz.size, seg):
        yield {"Fz": fz[s:s + seg], "FP1": fp1[s:s + seg], "FP2": fp2[s:s + seg]}


def _synthetic_120s(b1=0.25, b2=0.38, seed=5):
    rng = np.random.default_rng(seed)
    n = 250 * 120
    fp1 = rng.standard_normal(n) * 30.0
    fp2 = rng.standard_normal(n) * 25.0
    fz = b1 * fp1 + b2 * fp2 + rng.standard_normal(n) * 10.0
    return fz, fp1, fp2


def test_judge_calibration_then_freeze_then_running():
    fz, fp1, fp2 = _synthetic_120s()
    judge = qc.QcJudge()
    blocks = list(_blocks(fz, fp1, fp2))
    assert len(blocks) == 600

    calib_seen = 0
    for k, blk in enumerate(blocks):
        t1 = (k + 1) * 0.2
        v = judge.push(t1, blk)
        assert v["phase"] == "calibration"
        assert v["valid"] is None and v["reasons"] == []
        assert v["fz_clean_p2p"] is None and v["eog_p2p"] is None
        if v.get("calib") is not None:
            calib_seen += 1
            c = v["calib"]
            assert c["t_sec"] == 120.0
            assert c["fit_window_sec"] == [10.0, 120.0]
            assert c["n_fit_samples"] == 27500
            assert abs(c["b_fp1"] - 0.25) < 0.05
            assert abs(c["b_fp2"] - 0.38) < 0.05
    assert calib_seen == 1  # 恰在边界冻结一次
    assert judge.frozen

    # 第 601 段起进入运行期
    v = judge.push(120.2, {"Fz": np.zeros(50), "FP1": np.zeros(50), "FP2": np.zeros(50)})
    assert v["phase"] == "running" and v["valid"] is True and v["reasons"] == []
    assert v["fz_clean_p2p"] == 0.0 and v["eog_p2p"] == 0.0


def test_judge_running_verdicts_l1_l3_both():
    fz, fp1, fp2 = _synthetic_120s()
    judge = qc.QcJudge()
    for k, blk in enumerate(_blocks(fz, fp1, fp2)):
        judge.push((k + 1) * 0.2, blk)

    rng = np.random.default_rng(6)

    # L1：残差 p2p=101，EOG 两路干净
    fz_bad = np.linspace(0.0, 101.0, 50)
    v = judge.push(120.2, {"Fz": fz_bad, "FP1": np.zeros(50), "FP2": np.zeros(50)})
    assert v["valid"] is False and v["reasons"] == ["L1"]
    assert v["fz_clean_p2p"] == pytest.approx(101.0)
    assert v["eog_p2p"] == 0.0

    # L3：FP2 p2p=201，Fz 干净
    fp2_bad = np.linspace(0.0, 201.0, 50)
    v = judge.push(120.4, {"Fz": np.zeros(50), "FP1": np.zeros(50), "FP2": fp2_bad})
    assert v["valid"] is False and v["reasons"] == ["L3"]
    assert v["eog_p2p"] == pytest.approx(201.0)

    # L1 与 L3 同时命中 → 原因合并全记录（C5）
    fp1_bad = rng.standard_normal(50) * 200.0
    fp1_bad -= fp1_bad.min()  # p2p 远超 200
    v = judge.push(120.6, {"Fz": fz_bad, "FP1": fp1_bad, "FP2": np.zeros(50)})
    assert v["valid"] is False and set(v["reasons"] ) == {"L1", "L3"}

    # 阈值边界：恰好 100 / 200 不算超标（严格 >）
    v = judge.push(120.8, {
        "Fz": np.linspace(0.0, 100.0, 50),
        "FP1": np.linspace(0.0, 200.0, 50), "FP2": np.zeros(50),
    })
    assert v["valid"] is True


def test_judge_short_run_never_freezes():
    # 数据短于 120s：全程 calibration，无 calib，无异常
    fz, fp1, fp2 = _synthetic_120s()
    judge = qc.QcJudge()
    for k in range(10):
        s = k * 50
        v = judge.push((k + 1) * 0.2,
                       {"Fz": fz[s:s+50], "FP1": fp1[s:s+50], "FP2": fp2[s:s+50]})
        assert v["phase"] == "calibration" and v.get("calib") is None
    assert not judge.frozen


def test_judge_freeze_raises_on_nonfinite_b():
    # 用小校准窗加速：0.4s = 100 点，warmup=0；输入 NaN → 冻结时 ValueError
    judge = qc.QcJudge(calibration_sec=0.4, warmup_sec=0.0)
    nan_block = {"Fz": np.full(50, np.nan), "FP1": np.zeros(50), "FP2": np.zeros(50)}
    judge.push(0.2, nan_block)
    with pytest.raises(ValueError, match="非有限"):
        judge.push(0.4, nan_block)


def test_judge_warmup_excludes_startup_transient():
    """judge 级：短窗合成，启动样本污染时 warmup 剔除后仍回收正确系数。"""
    rng = np.random.default_rng(7)
    n = 250  # 1.0s 校准
    fp1 = np.zeros(n)
    fp1[125:] = rng.standard_normal(125) * 30.0
    fz = 0.40 * fp1 + rng.standard_normal(n) * 2.0
    fp1[:125] = 5000.0    # 启动期共同大阶跃（比例错误的相关瞬态）
    fz[:125] = -15000.0
    judge = qc.QcJudge(calibration_sec=1.0, warmup_sec=0.5)
    blocks = list(_blocks(fz, fp1, np.zeros(n)))
    v = None
    for k, blk in enumerate(blocks):
        v = judge.push((k + 1) * 0.2, blk)
    assert judge.frozen
    assert abs(v["calib"]["b_fp1"] - 0.40) < 0.05
    assert v["calib"]["n_fit_samples"] == 125
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_qc.py -v`
Expected: FAIL，`AttributeError: module 'bci_sys.replay.qc' has no attribute 'QcJudge'`

- [ ] **Step 3: 实现 QcJudge**

在 `src/bci_sys/replay/qc.py` 末尾追加（文件头无需改 import，numpy/config 已有）：

```python
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
        self.calibration_sec = float(calibration_sec)
        self.warmup_sec = float(warmup_sec)
        self.sfreq = float(sfreq)
        self.l1_thr = float(l1_thr)
        self.l3_thr = float(l3_thr)
        self.fz_name = fz_name
        self.eog_names = tuple(eog_names)

        self.calib_samples = int(round(self.calibration_sec * self.sfreq))
        self.warmup_samples = int(round(self.warmup_sec * self.sfreq))
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
        """t1=本段结束的数据时间（秒）；filtered 含 Fz/FP1/FP2 三路本段样本。"""
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
        arrs = {c: np.concatenate(self._buf[c]) for c in self._names}
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
        fz = np.asarray(filtered[self.fz_name], dtype=float)
        fp1 = np.asarray(filtered[self.eog_names[0]], dtype=float)
        fp2 = np.asarray(filtered[self.eog_names[1]], dtype=float)

        clean = fz - self.b_fp1 * fp1 - self.b_fp2 * fp2
        clean_p2p = p2p_uv(clean)
        eog_p2p = max(p2p_uv(fp1), p2p_uv(fp2))  # 两路取大
        hit_l1 = clean_p2p > self.l1_thr
        hit_l3 = eog_p2p > self.l3_thr
        reasons = []
        if hit_l1:
            reasons.append("L1")
        if hit_l3:
            reasons.append("L3")
        return {
            "phase": "running",
            "fz_clean_p2p": clean_p2p,
            "eog_p2p": eog_p2p,
            "valid": not (hit_l1 or hit_l3),
            "reasons": reasons,
        }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_qc.py -v`
Expected: PASS（10 个）

- [ ] **Step 5: 提交**

```bash
git add src/bci_sys/replay/qc.py tests/test_qc.py
git commit -m "feat(qc): QcJudge 校准冻结/运行裁决状态机（L1 残差门 + L3 EOG 否决）"
```

---

## Task 6: 事后有效率时间轴绘图 qc_timeline.py

**Files:**
- Create: `src/bci_sys/replay/qc_timeline.py`
- Test: `tests/test_qc_timeline.py`

说明：绘图只读已落盘的 qc.jsonl 标签，在回放结束后运行，不向因果裁决回流任何信息（C3）。英文标签（沿用 offline/probe.py 字体策略）。

- [ ] **Step 1: 先写失败测试**

创建 `tests/test_qc_timeline.py`：

```python
"""事后时间轴图：读合成 qc.jsonl 标签出 PNG；不参与裁决。"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bci_sys.replay import qc_timeline


def _write_jsonl(path, seg_specs):
    # seg_specs: [(t1, phase, valid, reasons)]
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "meta", "subj": "001"}) + "\n")
        for seg, (t1, phase, valid, reasons) in enumerate(seg_specs):
            f.write(json.dumps({
                "type": "seg", "seg": seg, "t0": t1 - 0.2, "t1": t1,
                "phase": phase, "fz_clean_p2p": None, "eog_p2p": None,
                "valid": valid, "reasons": reasons,
            }) + "\n")
        f.write(json.dumps({"type": "summary", "n_running": 1}) + "\n")


def test_timeline_png_from_mixed_labels(tmp_path):
    specs = []
    for k in range(600):  # 0–120s 校准
        specs.append(((k + 1) * 0.2, "calibration", None, []))
    for k in range(600, 750):  # 120–150s：交替有效/无效
        t1 = (k + 1) * 0.2
        if k % 4 == 0:
            specs.append((t1, "running", False, ["L3"]))
        elif k % 4 == 1:
            specs.append((t1, "running", False, ["L1", "L3"]))
        else:
            specs.append((t1, "running", True, []))
    jsonl = tmp_path / "qc.jsonl"
    png = tmp_path / "valid_rate_timeline.png"
    _write_jsonl(jsonl, specs)

    qc_timeline.plot_timeline(jsonl, png, subj="001")

    assert png.exists() and png.stat().st_size > 1000
    img = plt.imread(png)
    assert img.ndim == 3 and img.shape[2] in (3, 4)


def test_timeline_all_calibration_still_renders(tmp_path):
    specs = [((k + 1) * 0.2, "calibration", None, []) for k in range(10)]
    jsonl = tmp_path / "qc.jsonl"
    png = tmp_path / "valid_rate_timeline.png"
    _write_jsonl(jsonl, specs)
    qc_timeline.plot_timeline(jsonl, png, subj="001")  # 不抛错即通过
    assert png.exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_qc_timeline.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'bci_sys.replay.qc_timeline'`

- [ ] **Step 3: 实现**

创建 `src/bci_sys/replay/qc_timeline.py`：

```python
"""事后绘图：读 qc.jsonl 的因果标签画有效率时间轴（IT3 产物）。

只在回放结束后运行；输入是已经产生的标签，输出是 PNG，
不向 qc.py 的任何裁决回流信息（C3：事后处理与实时因果判定隔离）。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

BIN_SEC = 30.0
COLORS = {
    "calibration": "#bdbdbd",  # 灰
    "valid": "#2ca02c",        # 绿
    "L1": "#ff7f0e",           # 橙：仅 L1
    "L3": "#d62728",           # 红：仅 L3
    "both": "#7b1fa2",         # 紫：L1+L3
}


def load_seg_rows(qc_jsonl_path: Path | str) -> list[dict]:
    rows = []
    for line in Path(qc_jsonl_path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("type") == "seg":
            rows.append(rec)
    return rows


def _row_color(row: dict) -> str:
    if row["phase"] == "calibration":
        return COLORS["calibration"]
    if row["valid"]:
        return COLORS["valid"]
    has_l1 = "L1" in row["reasons"]
    has_l3 = "L3" in row["reasons"]
    if has_l1 and has_l3:
        return COLORS["both"]
    return COLORS["L1"] if has_l1 else COLORS["L3"]


def _bin_valid_rates(rows: list[dict], bin_sec: float = BIN_SEC):
    """仅运行期段按 t1 落 30s 箱；返回 (箱中心秒, 有效率)。校准箱自然不出现。"""
    run = [r for r in rows if r["phase"] == "running"]
    if not run:
        return np.array([]), np.array([])
    t1 = np.array([r["t1"] for r in run], dtype=float)
    valid = np.array([1.0 if r["valid"] else 0.0 for r in run])
    b = np.floor((t1 - 1e-9) / bin_sec).astype(int)
    xs, ys = [], []
    for k in range(b.min(), b.max() + 1):
        m = b == k
        if m.any():
            xs.append((k + 0.5) * bin_sec)
            ys.append(float(valid[m].mean()))
    return np.array(xs), np.array(ys)


def plot_timeline(qc_jsonl_path: Path | str, png_path: Path | str, subj: str = "") -> None:
    rows = load_seg_rows(qc_jsonl_path)
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(10, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 5]},
    )

    # 上轨：逐段裁决色带
    t_max = rows[-1]["t1"] if rows else 1.0
    rgb = np.array([[mcolors.to_rgb(_row_color(r)) for r in rows]])
    ax0.imshow(rgb, aspect="auto", extent=[0.0, t_max, 0.0, 1.0])
    ax0.set_yticks([])
    ax0.set_ylabel("verdict")

    # 下轨：运行期 30s 分箱有效率
    bx, by = _bin_valid_rates(rows)
    if bx.size:
        ax1.plot(bx, by, "o-", ms=4, color="#1f77b4")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("valid rate (30s bin)")
    ax1.set_xlabel("time (s)")
    ax1.grid(alpha=0.3)

    run = [r for r in rows if r["phase"] == "running"]
    title = f"sub-{subj} QC valid-rate timeline" if subj else "QC valid-rate timeline"
    if run:
        overall = sum(1 for r in run if r["valid"]) / len(run)
        title += f" — overall {overall:.1%} (running only, calibration excluded)"
    fig.suptitle(title, fontsize=11)

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=COLORS["calibration"], label="calibration"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=COLORS["valid"], label="valid"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=COLORS["L1"], label="invalid L1"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=COLORS["L3"], label="invalid L3"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=COLORS["both"], label="invalid L1+L3"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=5, fontsize=7, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_qc_timeline.py -v`
Expected: PASS（2 个）

- [ ] **Step 5: 提交**

```bash
git add src/bci_sys/replay/qc_timeline.py tests/test_qc_timeline.py
git commit -m "feat(qc): 事后有效率时间轴图（裁决色带+30s 分箱，不回流因果路径）"
```

---

## Task 7: qc_monitor 编排、终端标色与落盘（含真数据结构验收）

**Files:**
- Create: `src/bci_sys/replay/qc_monitor.py`
- Test: `tests/test_qc_monitor.py`

- [ ] **Step 1: 先写失败测试**

创建 `tests/test_qc_monitor.py`：

```python
import json
from pathlib import Path

from bci_sys.replay import pace, qc_monitor


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class AbortClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt
        if self.t >= 0.6:  # 第 3 段结束时中止
            raise KeyboardInterrupt


def _read_jsonl(p: Path):
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def _run_dir(tmp_path, rid="fixed"):
    return tmp_path / "qc" / "sub-001" / rid


def test_qc_short_run_meta_seg_summary_schema(data_root, tmp_path):
    clk = FakeClock()
    status = qc_monitor.run_qc(
        "001", duration_sec=2.0, root=data_root, out_root=tmp_path,
        run_id="fixed", pacer=pace.Pacer(sleep_fn=clk.sleep, clock=clk.now),
    )
    assert status == "ok"
    rows = _read_jsonl(_run_dir(tmp_path) / "qc.jsonl")
    kinds = [r["type"] for r in rows]
    assert kinds[0] == "meta" and kinds[-1] == "summary"
    assert kinds.count("seg") == 10
    assert "calib" not in kinds  # 2s 短回放不到校准边界

    meta = rows[0]
    assert meta["schema_version"] == "1.0"
    assert meta["subj"] == "001" and meta["sfreq"] == 250.0
    assert meta["segment_sec"] == 0.2 and meta["n_segments_planned"] == 10
    assert meta["channels"] == {"Fz": "Fz", "FP1": "FP1", "FP2": "FP2"}
    assert meta["filter_band_hz"] == [1.0, 30.0]
    assert meta["calibration_sec"] == 120.0 and meta["filter_warmup_sec"] == 10.0
    assert meta["l1_p2p_uv"] == 100.0 and meta["l3_eog_p2p_uv"] == 200.0
    assert "code_version" in meta and meta["code_version"] != "unknown"

    for r in rows[1:-1]:
        assert r["phase"] == "calibration"
        assert r["valid"] is None and r["reasons"] == []
        assert r["fz_clean_p2p"] is None and r["eog_p2p"] is None

    summ = rows[-1]
    assert summ["n_running"] == 0 and summ["n_valid"] == 0 and summ["n_invalid"] == 0
    assert summ["n_l1"] == 0 and summ["n_l3"] == 0 and summ["n_l1_and_l3"] == 0
    assert summ["valid_rate"] is None  # 运行期 0 段 → null，不许 ZeroDivision

    run = json.loads((_run_dir(tmp_path) / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "ok"
    assert {"wall_started_iso", "wall_elapsed_s", "rate_ratio", "n_segments"} <= run.keys()
    assert (_run_dir(tmp_path) / "valid_rate_timeline.png").exists()


def test_qc_jsonl_byte_deterministic_and_wall_free(data_root, tmp_path):
    outs = []
    for rid in ("a", "b"):
        clk = FakeClock()
        qc_monitor.run_qc(
            "001", duration_sec=2.0, root=data_root, out_root=tmp_path / rid,
            run_id="fixed", pacer=pace.Pacer(sleep_fn=clk.sleep, clock=clk.now),
        )
        p = tmp_path / rid / "qc" / "sub-001" / "fixed" / "qc.jsonl"
        outs.append(p.read_bytes())
        text = p.read_text(encoding="utf-8")
        assert "wall" not in text and "rate_ratio" not in text  # A4：墙钟只进 run.json
    assert outs[0] == outs[1]


def test_qc_abort_writes_run_json_no_summary(data_root, tmp_path):
    clk = AbortClock()
    status = qc_monitor.run_qc(
        "001", duration_sec=2.0, root=data_root, out_root=tmp_path,
        run_id="x", pacer=pace.Pacer(sleep_fn=clk.sleep, clock=clk.now),
    )
    assert status == "aborted"
    rd = _run_dir(tmp_path, "x")
    rows = _read_jsonl(rd / "qc.jsonl")
    assert rows[-1]["type"] != "summary"
    run = json.loads((rd / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "aborted"
    assert not (rd / "valid_rate_timeline.png").exists()  # 仅 ok 才出图


def test_qc_preflight_failure_writes_nothing(data_root, tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        qc_monitor.run_qc("999", root=data_root, out_root=tmp_path, run_id="z")
    assert not (tmp_path / "qc").exists()


def test_qc_real_data_130s_freeze_and_counters(data_root, tmp_path):
    """真数据结构验收（不设有效率断言）：冻结 b 有限且 0.1–0.7；计数自洽。"""
    clk = FakeClock()  # 假时钟：130s 数据瞬时跑完
    status = qc_monitor.run_qc(
        "001", duration_sec=130.0, root=data_root, out_root=tmp_path,
        run_id="real", pacer=pace.Pacer(sleep_fn=clk.sleep, clock=clk.now),
    )
    assert status == "ok"
    rd = _run_dir(tmp_path, "real")
    rows = _read_jsonl(rd / "qc.jsonl")

    calibs = [r for r in rows if r["type"] == "calib"]
    assert len(calibs) == 1
    c = calibs[0]
    assert c["t_sec"] == 120.0 and c["n_fit_samples"] == 27500
    for key in ("b_fp1", "b_fp2"):
        assert c[key] == c[key]  # 非 NaN
        assert 0.1 <= c[key] <= 0.7  # 实测合理区间（结构验收，非疗效结论）

    segs = [r for r in rows if r["type"] == "seg"]
    assert len(segs) == 650
    assert sum(r["phase"] == "calibration" for r in segs) == 600
    running = [r for r in segs if r["phase"] == "running"]
    assert len(running) == 50
    assert all(r["valid"] in (True, False) for r in running)
    assert all(set(r["reasons"]) <= {"L1", "L3"} for r in running)

    summ = rows[-1]
    assert summ["type"] == "summary"
    assert summ["n_running"] == 50
    assert summ["n_valid"] + summ["n_invalid"] == 50
    assert summ["n_l1"] == sum("L1" in r["reasons"] for r in running)
    assert summ["n_l3"] == sum("L3" in r["reasons"] for r in running)
    assert summ["n_l1_and_l3"] == sum(
        {"L1", "L3"} <= set(r["reasons"]) for r in running
    )
    assert 0.0 <= summ["valid_rate"] <= 1.0
    assert (rd / "valid_rate_timeline.png").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_qc_monitor.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'bci_sys.replay.qc_monitor'`

- [ ] **Step 3: 实现**

创建 `src/bci_sys/replay/qc_monitor.py`：

```python
"""QC 回放主循环（IT3）：preflight → 三路流式因果带通 → 120s 校准冻结 →
逐段 L1/L3 裁决 → 终端标色 + qc.jsonl / run.json / 有效率时间轴 PNG。

A4：qc.jsonl 只含数据侧量，同参两次运行字节级一致；墙钟只进 run.json。
C5：无效段只打标并记录原因，绝不删除；反馈暂停是 IT4 的事。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .. import config, gitmeta, io_bids
from . import causal_filter, pace, qc, qc_timeline, segments

SCHEMA_VERSION = "1.0"

GRAY = "\x1b[90m"
FZ = config.TARGET.channel
EOGS = config.EOG_PROXY_CHANNELS
ALL_CH = (FZ, *EOGS)


def format_seg_line(
    seg: int, t1: float, clean_p2p, eog_p2p, ratio: float, verdict: dict
) -> str:
    color = pace.rate_color(ratio)
    if verdict["phase"] == "calibration":
        clean_s = eog_s = "   —"
        mark = f"{GRAY}CAL{pace.RESET}"
    else:
        clean_s = f"{clean_p2p:6.1f}"
        eog_s = f"{eog_p2p:6.1f}"
        if verdict["valid"]:
            mark = f"{pace.GREEN}✓{pace.RESET}"
        else:
            mark = f"{pace.RED}✗ {' '.join(verdict['reasons'])}{pace.RESET}"
    return (
        f"{seg:05d} │ t={t1:6.1f}s │ Fz_clean p2p {clean_s}µV │ "
        f"EOG p2p {eog_s}µV │ rate {color}{ratio:.2f}x{pace.RESET} │ {mark}"
    )


def run_qc(
    subj: str,
    duration_sec: float = 0.0,
    root: Path | str | None = None,
    out_root: Path | str = "outputs",
    run_id: str | None = None,
    pacer: pace.Pacer | None = None,
) -> str:
    """跑一次 QC 回放。返回 'ok' / 'aborted'。preflight 失败直接抛异常（不写文件，C7）。"""
    io_bids.preflight(subj, root)  # 硬停，不写任何文件

    chans = io_bids.load_channels_uV(subj, list(ALL_CH), root)
    fz_all = chans[FZ]
    n_planned = segments.n_planned_segments(
        fz_all.size, config.SEGMENT_SAMPLES, duration_sec
    )

    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_root) / "qc" / f"sub-{subj}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pacer = pacer or pace.Pacer()
    wall_start = datetime.now().isoformat(timespec="seconds")
    pacer.start()

    band = causal_filter.CausalBandpass(ALL_CH)
    judge = qc.QcJudge()

    print(
        f"[qc-monitor] sub-{subj} | 因果带通 "
        f"{config.FILTER_BAND_HZ[0]:g}–{config.FILTER_BAND_HZ[1]:g}Hz | "
        f"校准 {config.QC_CALIBRATION_SEC:g}s（预热剔除 {config.QC_FILTER_WARMUP_SEC:g}s）| "
        f"L1 残差 p2p>{config.QC_L1_P2P_UV:g}µV | "
        f"L3 EOG p2p>{config.QC_L3_EOG_P2P_UV:g}µV（{EOGS[0]}/{EOGS[1]} 任一）| "
        f"段 {config.SEGMENT_SEC:g}s"
    )

    counters = {
        "n_running": 0, "n_valid": 0, "n_invalid": 0,
        "n_l1": 0, "n_l3": 0, "n_l1_and_l3": 0,
    }
    status = "ok"
    last_ratio = 1.0
    n_emitted = 0
    jsonl_path = run_dir / "qc.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "type": "meta",
            "schema_version": SCHEMA_VERSION,
            "code_version": gitmeta.code_version(),
            "subj": subj,
            "sfreq": config.SFREQ,
            "segment_sec": config.SEGMENT_SEC,
            "channels": {FZ: FZ, EOGS[0]: EOGS[0], EOGS[1]: EOGS[1]},
            "filter_band_hz": list(config.FILTER_BAND_HZ),
            "calibration_sec": config.QC_CALIBRATION_SEC,
            "filter_warmup_sec": config.QC_FILTER_WARMUP_SEC,
            "l1_p2p_uv": config.QC_L1_P2P_UV,
            "l3_eog_p2p_uv": config.QC_L3_EOG_P2P_UV,
            "n_segments_planned": n_planned,
        }, ensure_ascii=False) + "\n")

        try:
            for seg in segments.iter_segments(
                fz_all, config.SFREQ, config.SEGMENT_SAMPLES, duration_sec
            ):
                s = seg.seg * config.SEGMENT_SAMPLES
                e = s + config.SEGMENT_SAMPLES
                filtered = band.push_block({c: chans[c][s:e] for c in ALL_CH})
                verdict = judge.push(seg.t1, filtered)

                if verdict.get("calib") is not None:
                    f.write(json.dumps(verdict["calib"], ensure_ascii=False) + "\n")
                    print(
                        f"[校准] b_{EOGS[0]}={judge.b_fp1:.3f} "
                        f"b_{EOGS[1]}={judge.b_fp2:.3f} (frozen)"
                    )

                f.write(json.dumps({
                    "type": "seg",
                    "seg": seg.seg,
                    "t0": seg.t0,
                    "t1": seg.t1,
                    "phase": verdict["phase"],
                    "fz_clean_p2p": verdict["fz_clean_p2p"],
                    "eog_p2p": verdict["eog_p2p"],
                    "valid": verdict["valid"],
                    "reasons": verdict["reasons"],
                }, ensure_ascii=False) + "\n")

                if verdict["phase"] == "running":
                    counters["n_running"] += 1
                    hit_l1 = "L1" in verdict["reasons"]
                    hit_l3 = "L3" in verdict["reasons"]
                    if verdict["valid"]:
                        counters["n_valid"] += 1
                    else:
                        counters["n_invalid"] += 1
                    if hit_l1:
                        counters["n_l1"] += 1
                    if hit_l3:
                        counters["n_l3"] += 1
                    if hit_l1 and hit_l3:
                        counters["n_l1_and_l3"] += 1

                ratio = pacer.rate_ratio(seg.t1)
                last_ratio = ratio
                print(format_seg_line(
                    seg.seg, seg.t1,
                    verdict["fz_clean_p2p"], verdict["eog_p2p"],
                    ratio, verdict,
                ))

                pacer.wait_after_segment(seg.seg)
                n_emitted += 1
        except KeyboardInterrupt:
            status = "aborted"

        if status == "ok":
            valid_rate = (
                None if counters["n_running"] == 0
                else round(counters["n_valid"] / counters["n_running"], 6)
            )
            f.write(json.dumps({
                "type": "summary",
                **counters,
                "valid_rate": valid_rate,
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

    if status == "ok":
        qc_timeline.plot_timeline(
            jsonl_path, run_dir / "valid_rate_timeline.png", subj=subj
        )
    return status
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_qc_monitor.py -v`
Expected: PASS（5 个；最后一个为真数据结构验收，假时钟瞬时完成，不对有效率做断言）

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/test_monitor.py tests/test_boundary.py tests/test_qc.py tests/test_qc_monitor.py -v`
Expected: PASS——边界 AST 测试自动把 `causal_filter.py`/`qc.py`/`qc_timeline.py`/`qc_monitor.py` 纳入扫描，确认无 offline import。

- [ ] **Step 6: 提交**

```bash
git add src/bci_sys/replay/qc_monitor.py tests/test_qc_monitor.py
git commit -m "feat(qc): qc_monitor 编排（终端✓/✗/CAL、qc.jsonl/run.json/时间轴、A4）"
```

---

## Task 8: qc-monitor 命令行入口 + console script

**Files:**
- Create: `src/bci_sys/replay/qc_cli.py`
- Modify: `pyproject.toml`（`[project.scripts]` 加一行）
- Test: `tests/test_qc_cli.py`

- [ ] **Step 1: 先写失败测试**

创建 `tests/test_qc_cli.py`（子进程必须 `encoding="utf-8"`，范式同 test_replay_cli.py）：

```python
import json
import subprocess
import sys


def test_cli_runs_limited_qc(data_root, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.replay.qc_cli",
         "--subj", "001", "--duration-sec", "2",
         "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    assert "CAL" in r.stdout
    run_dir = tmp_path / "qc" / "sub-001"
    assert run_dir.exists()
    js = list(run_dir.rglob("qc.jsonl"))
    assert len(js) == 1
    rows = [json.loads(l) for l in js[0].read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["type"] == "summary" and rows[-1]["n_running"] == 0


def test_cli_bad_subject_exits_2_and_writes_nothing(data_root, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.replay.qc_cli",
         "--subj", "999", "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 2
    assert "PREFLIGHT" in r.stderr or "preflight" in r.stderr.lower()
    assert not (tmp_path / "qc").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_qc_cli.py -v`
Expected: FAIL，`No module named bci_sys.replay.qc_cli`（returncode != 0）

- [ ] **Step 3: 实现 CLI**

创建 `src/bci_sys/replay/qc_cli.py`：

```python
"""qc-monitor 命令行入口。退出码：ok=0；preflight/数据错误=2；中止=130。"""
from __future__ import annotations

import argparse
import sys

from . import qc_monitor


def main(argv=None) -> int:
    # Windows 控制台 GBK 下 µ/✓/✗ 需要 UTF-8（与 replay CLI 同）
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        prog="qc-monitor",
        description="1x 回放 + 实时三级质量判定（L1 残差 / L3 EOG 否决）",
    )
    ap.add_argument("--subj", required=True, help="被试编号，如 001")
    ap.add_argument("--duration-sec", type=float, default=0.0,
                    help="回放时长（秒）；0=全量（默认）")
    ap.add_argument("--data-root", default=None, help="ds007169 根目录（默认项目内）")
    ap.add_argument("--out-dir", default="outputs", help="产物根目录")
    args = ap.parse_args(argv)

    try:
        status = qc_monitor.run_qc(
            args.subj, duration_sec=args.duration_sec,
            root=args.data_root, out_root=args.out_dir,
        )
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"PREFLIGHT FAIL: {e}", file=sys.stderr)
        return 2

    if status == "aborted":
        print("\n[中止] 已写 run.json(status=aborted)，本次数据标记为中止。", file=sys.stderr)
        return 130
    print(f"[完成] 产物在 {args.out_dir}/qc/sub-{args.subj}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 注册 console script**

在 `pyproject.toml` 的 `[project.scripts]` 段把：

```toml
[project.scripts]
replay-monitor = "bci_sys.replay.cli:main"
theta-probe = "bci_sys.offline.cli:main"
```

替换为：

```toml
[project.scripts]
replay-monitor = "bci_sys.replay.cli:main"
qc-monitor = "bci_sys.replay.qc_cli:main"
theta-probe = "bci_sys.offline.cli:main"
```

- [ ] **Step 5: 跑测试确认通过，并冒烟验证安装后的入口**

Run: `uv run pytest tests/test_qc_cli.py -v`
Expected: PASS（2 个）

Run: `uv run qc-monitor --subj 001 --duration-sec 1 --out-dir outputs`
Expected: 退出码 0；输出含参数回显行与 5 行灰色 `CAL`；`outputs/qc/sub-001/<时间戳>/` 下有 qc.jsonl、run.json、valid_rate_timeline.png。

- [ ] **Step 6: 提交**

```bash
git add src/bci_sys/replay/qc_cli.py pyproject.toml tests/test_qc_cli.py
git commit -m "feat(qc): qc-monitor CLI（退出码 0/2/130）与 console script"
```

---

## Task 9: 全量回归与收尾

**Files:** 无新增（仅验证；如有修复则改对应文件）

- [ ] **Step 1: 全量测试**

Run: `uv run pytest -v`
Expected: 全部 PASS。基线 57 个 + 本计划新增：config 1、io/preflight 3、causal_filter 5、qc 10、qc_timeline 2、qc_monitor 5、qc_cli 2 = 28 个，合计约 85 个（若实际计数有出入，以全绿为准，不允许 skip 之外的失败）。

- [ ] **Step 2: 确认边界隔离覆盖新文件**

Run: `uv run pytest tests/test_boundary.py -v`
Expected: 2 个 PASS。该测试 rglob 整个 `replay/` 目录，新模块自动受约束；若失败，说明新代码违规 import 了 `bci_sys.offline*`，必须立即移除（只允许共享 `config.py`/`io_bids.py`）。

- [ ] **Step 3: 手工全量 1x 回放留证（约 16 分钟，可后台）**

Run: `uv run qc-monitor --subj 001`

Expected:
- 前 120s（600 段）灰色 `CAL`；t=120.0 打印冻结行 `[校准] b_FP1≈0.25 b_FP2≈0.38 (frozen)`（以实际数字为准，记录到 evidence，禁止为好看调阈值）；
- 之后每段绿 `✓` 或红 `✗ L1` / `✗ L3` / `✗ L1 L3`；rate 稳态绿 1.00x；
- 产物 `outputs/qc/sub-001/<run_id>/`：qc.jsonl、run.json、valid_rate_timeline.png；
- 再跑一次（可 `--duration-sec 300` 加速比对），两次 qc.jsonl `diff` 字节一致。

- [ ] **Step 4: 提交任何收尾改动**

```bash
git status --short
# 若有改动：
git add -A
git commit -m "chore(qc): IT3 全量回归收尾"
```

---

## 实现约束清单（每个任务都适用）

1. **C2 冻结**：阈值/窗长只从 `config.py` 读，不加任何运行期修改入口（CLI 无阈值参数）。
2. **C3 因果/隔离**：实时新代码只在 `replay/`；滤波只用单向 `lfilter`，禁止 filtfilt；不 import `bci_sys.offline*`；绘图只读已落盘标签。
3. **C5 只标不删**：无效段保留全部段记录，`reasons` 写全（L1/L3 可并存）。
4. **C7 硬停**：preflight/系数非有限直接抛异常；CLI 转退出码 2，不出现"是否继续"提示。
5. **A4**：qc.jsonl 禁墙钟、禁 run_id 时间戳污染（测试固定 run_id）；两次同参运行字节一致；墙钟只进 run.json；PNG 不参与字节比对。
6. **单位**：mne 数值已是 µV，禁止 ×1e6。
7. 不顺手调阈值、不做 IT4+ 的事（theta/反馈/任务编排一律不碰）。
