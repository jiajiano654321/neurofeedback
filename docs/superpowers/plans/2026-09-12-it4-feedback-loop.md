# IT4 feedback_loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) steps for tracking.

**Goal:** 在 1x 回放的实时因果路径上实现 `feedback-loop`：S3 一次性冻结 EOG 系数与 θ 阈值 → Web 单页八相位显示 → 6×120s block 逐段反馈与 QC 暂停 → 确定性 jsonl + 网页回放。

**Architecture:** 单进程：uvicorn/FastAPI 在主线程；POST /api/start 后同步引擎在单个 daemon 线程跑实时循环（风格与 IT3 `run_qc` 同构），事件经有界队列（256、丢最旧、带 seq）→ asyncio 桥接协程 → 单个 WebSocket。全部新代码在 `src/bci_sys/replay/`，与 offline/ 物理隔离（AST 守卫自动覆盖新文件）。

**Tech Stack:** Python 3.12, uv, numpy 2.0（`np.quantile(..., method="linear")`）, scipy.signal（单向 lfilter；replay/ 内严禁 filtfilt）, FastAPI + uvicorn + WebSocket, httpx + wsproto（dev，TestClient WebSocket 用 wsproto）, 无构建静态 HTML/JS（无 CDN、无外部 URL）。

**Spec:** `docs/superpowers/specs/2026-09-12-it4-feedback-loop-design.md`（已批准；本计划唯一需求来源）。

---

## 0. 硬性约定（每个任务都适用）

- **实测数字不硬编码进断言**（项目记忆：单位/点数/参数名等硬编码值必须先跑真数据验证）。sub-001 实测 θ_thr≈16.74 µV²、n(E)≈491、block 达标率 0.76/0.88/0.80/0.83/0.81/0.88 来自**校准窗全程预热**的探针；本期运行期平滑器是**冷启动**（冻结后空 deque），block 0 前几段会与探针不同。测试只做宽结构断言（finite、范围、自洽），具体数值由 T9 在真实引擎输出上填写到 evidence。
- **C1**：`direction` 必填、无默认，构造即校验。
- **C2**：所有参数只在 [config.py](src/bci_sys/config.py)；运行期/前端无修改入口。
- **C3**：新代码全部在 `src/bci_sys/replay/`；**禁止 import `bci_sys.offline.*`、禁止 filtfilt**（[test_boundary.py](tests/test_boundary.py) 自动覆盖新文件）；replay/ 只共享 `config.py`、`io_bids.py`。
- **C4/C5**：去眼电 + QC 在 θ 之前；无效段只打标、反馈暂停（smoothed/hit 落 null，分子分母都不动），绝不删除。
- **C7**：错误硬拦，不提供「继续/忽略/跳过」入口。
- **A4**：feedback.jsonl 两次运行字节一致；墙钟、线程、队列、seq、dropped、ISO 时间戳一律不进 jsonl（只进 run.json）。
- **单位**：mne 输出已是 µV，**严禁 ×1e6**；θ 功率单位 µV²。
- 每个任务结束：`uv run pytest -q` 全绿才 commit；commit message 用 `type(scope): 中文描述`，trailer 固定 `Co-Authored-By: Claude <noreply@anthropic.com>`。
- **不要 add/commit** [src/bci_sys/offline/probe.py](src/bci_sys/offline/probe.py) 和 [src/bci_sys/offline/theta.py](src/bci_sys/offline/theta.py)（外部未提交改动）；每个任务只 stage 本任务文件。
- Windows 控制台：CLI 入口开头 UTF-8 reconfigure（沿 IT3）；jsonl `newline="\n"`、`encoding="utf-8"`、逐行 flush。
- 本计划 9 个实现任务 + 1 个验收任务；命名/签名在任务间严格一致（见各任务代码）。

## 0.1 实测前提与预期值（来自 spec §2，仅供 T9 evidence 与人工验收参照，不进代码断言）

| 量 | 实测值 | 备注 |
|---|---|---|
| b_FP1 / b_FP2 | 0.248183 / 0.378130 | 与 IT3 冻结一致，宽断言 0.1–0.7 |
| θ_thr（W5 仅有效、P20、t1>10 且成熟） | 约 16.7 µV² | 运行时现算；测试不断言具体值 |
| n(E) | 约 491（探针 t1>11 口径）；引擎 t1>10 口径会略多 | 断言 ≥450 |
| block 无效段数 | 43/41/57/71/85/108 | 与 θ 平滑无关，确定性，T9 核对 |
| 会话墙钟 | ≈840+150+3=993s + 确认等待 | FakeClock 测试瞬时 |
| 校准缓冲 | 3×30000×8B=720,000B≈0.72MB | <10MB（spec §4.1） |

## 0.2 File Structure

| 文件 | 责任 | 任务 |
|---|---|---|
| `pyproject.toml` | fastapi/uvicorn[standard] 依赖、httpx/wsproto dev、feedback-loop script | T1 |
| `src/bci_sys/config.py` | IT4 冻结常量（spec §5.3） | T1 |
| `src/bci_sys/replay/causal_filter.py` | 新增 `CausalChannel`（可注入/导出 zi）；`CausalBandpass` 行为保持 | T2 |
| `src/bci_sys/replay/theta_metric.py` | 4–8Hz 校准一次性滤波+zf、段功率、仅有效段平滑、P20 冻结、HitDecider（C1）、ThetaStream | T3 |
| `src/bci_sys/replay/qc.py` | 抽 `segment_verdict` 纯函数（行为保持），校准重建复用 | T4 |
| `src/bci_sys/replay/feedback.py` | 纯会话状态机：相位/计数/rest/wall_cap/终态 | T4 |
| `src/bci_sys/replay/feedback_engine.py` | EventBus、run_feedback 引擎、FeedbackRunner 线程壳、build_params/initial_snapshot | T5 |
| `src/bci_sys/replay/webapp.py` | FastAPI 路由、WS live 桥接、replay 播放器（offsets/reducer/seek）、白名单 | T6 |
| `src/bci_sys/web/index.html`、`app.js` | 无构建单页八相位（布局 A）；webapp.py 以 `parent.parent/"web"` 定位 | T7 |
| `src/bci_sys/replay/feedback_cli.py` | argparse、preflight、uvicorn、live/replay、退出码 0/2/130/3/1 | T8 |
| `evidence/README.md` | IT4 节（命令/数值/A4/replay/截图清单） | T9 |
| `tests/test_it4_config.py`、`test_it4_causal_channel.py`、`test_theta_metric.py`、`test_feedback_session.py`、`test_feedback_engine.py`、`test_webapp.py`、`test_feedback_cli.py` | 各任务测试 | T1–T8 |

基线：92 passed（T0 前先确认）。每个任务都是 TDD：先写失败测试 → 跑确认失败 → 实现 → 跑绿 → commit。

---

### Task 0: 建分支、装依赖

**Files:**
- Modify: git 分支、`pyproject.toml`/`uv.lock`（由 uv 命令改）

- [ ] **Step 1: 确认基线与工作区**

Run:
```bash
cd e:/trae_projects/bci-sys && git status --short && uv run pytest -q
```
Expected: 只有 `src/bci_sys/offline/probe.py`、`src/bci_sys/offline/theta.py` 两个 M（不要碰它们）；`92 passed`。

- [ ] **Step 2: 建特性分支**

```bash
git checkout -b feature/it4-feedback-loop
```

- [ ] **Step 3: 加依赖（uv 改 pyproject + lock）**

```bash
uv add "fastapi>=0.115" "uvicorn[standard]>=0.30"
uv add --dev "httpx>=0.27" "wsproto>=1.2"
```

- [ ] **Step 4: 验证导入**

Run: `uv run python -c "import fastapi, uvicorn, httpx, wsproto; print('deps ok')"`
Expected: `deps ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(it4): 加 fastapi/uvicorn 运行依赖与 httpx/wsproto 测试依赖

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1: config 冻结常量

**Files:**
- Modify: [src/bci_sys/config.py](src/bci_sys/config.py)（在 IT3 常量块之后追加，不改既有值）
- Test: `tests/test_it4_config.py`（新建）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_it4_config.py`：

```python
"""IT4 冻结常量（spec §5.3，C2：运行期无修改入口）。"""

from bci_sys import config


def test_it4_frozen_constants():
    assert config.THETA_SMOOTH_SEGS == 5
    assert config.THETA_QUANTILE == 0.20
    assert config.THETA_TARGET_RATE == 0.80
    assert config.THETA_MIN_CALIB_SEGS == 50
    assert config.FEEDBACK_N_BLOCKS == 6
    assert config.FEEDBACK_BLOCK_SEC == 120.0
    assert config.FEEDBACK_REST_SEC == 30.0
    assert config.FEEDBACK_FREEZE_CARD_SEC == 3.0
    assert config.SESSION_WALL_CAP_SEC == 1500.0


def test_it4_reused_constants_unchanged():
    # 复用项不得在 IT4 被顺手改掉
    assert config.THETA_BAND_HZ == (4.0, 8.0)
    assert config.QC_CALIBRATION_SEC == 120.0
    assert config.QC_FILTER_WARMUP_SEC == 10.0
    assert config.QC_L1_P2P_UV == 100.0
    assert config.QC_L3_EOG_P2P_UV == 200.0
    assert config.TARGET.channel == "Fz"
    assert config.TARGET.direction == "up"
    assert config.SEGMENT_SAMPLES == 50 and config.SFREQ == 250.0


def test_it4_segment_counts():
    cal = int(round(config.QC_CALIBRATION_SEC / config.SEGMENT_SEC))
    per_block = int(round(config.FEEDBACK_BLOCK_SEC / config.SEGMENT_SEC))
    assert cal == 600
    assert per_block == 600
    assert config.FEEDBACK_N_BLOCKS * per_block == 3600
    assert config.QC_CALIBRATION_SEC + config.FEEDBACK_N_BLOCKS * config.FEEDBACK_BLOCK_SEC == 840.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_it4_config.py -q`
Expected: FAIL（`AttributeError: module 'bci_sys.config' has no attribute 'THETA_SMOOTH_SEGS'`）

- [ ] **Step 3: 实现**

在 [src/bci_sys/config.py](src/bci_sys/config.py) 的 IT3 常量块（`QC_L3_EOG_P2P_UV` 那行）之后、`@dataclass(frozen=True) class Target` 之前插入：

```python
# --- IT4 反馈会话（冻结，C2：运行期无修改入口） ---
THETA_SMOOTH_SEGS = 5        # 仅有效段尾窗（1s）
THETA_QUANTILE = 0.20        # S3 阈值分位（numpy linear）
THETA_TARGET_RATE = 0.80     # 设计目标达标率（=1−分位）；仅冻结时使用，不参与运行期控制
THETA_MIN_CALIB_SEGS = 50    # 冻结所需最小合格成熟校准段
FEEDBACK_N_BLOCKS = 6
FEEDBACK_BLOCK_SEC = 120.0
FEEDBACK_REST_SEC = 30.0
FEEDBACK_FREEZE_CARD_SEC = 3.0
SESSION_WALL_CAP_SEC = 1500.0
```

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run: `uv run pytest tests/test_it4_config.py -q && uv run pytest -q`
Expected: 新测试 PASS；总计 `95 passed`（92+3）。

- [ ] **Step 5: Commit**

```bash
git add src/bci_sys/config.py tests/test_it4_config.py
git commit -m "feat(it4): config 冻结 θ 平滑/分位/会话结构常量（C2）

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: causal_filter 抽出 CausalChannel（zi 注入/导出，行为保持）

θ 链路需要：校准窗对 cal_clean 以 zi=0 一次性 4–8Hz 滤波并取回末端 zf，运行期再用该 zf 初始化逐块滤波器（spec §4.2-3/7）。做法：扩 `_CausalSection` 支持可选 zi，新增公开 `CausalChannel`，`CausalBandpass` 改为持有它（**对外行为逐点不变**）。

**Files:**
- Modify: [src/bci_sys/replay/causal_filter.py](src/bci_sys/replay/causal_filter.py)
- Modify: [tests/test_causal_filter.py](tests/test_causal_filter.py)（仅一处私有属性路径随重构更新）
- Test: `tests/test_it4_causal_channel.py`（新建）

- [ ] **Step 1: 写新测试**

创建 `tests/test_it4_causal_channel.py`：

```python
"""CausalChannel：单通道两节单向滤波 + zi 注入/导出（IT4 θ 链路续接）。"""

import numpy as np
import pytest

from bci_sys.replay import causal_filter as cf


def test_channel_chunked_equals_oneshot_default_band():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50 * 8) * 20.0
    ch = cf.CausalChannel()
    ys = [ch.push(x[s:s + 50]) for s in range(0, x.size, 50)]
    np.testing.assert_allclose(np.concatenate(ys), cf.bandpass_oneshot(x), atol=1e-10)


def test_channel_theta_band_chunked_equals_oneshot():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(50 * 8) * 20.0
    ch = cf.CausalChannel(band_hz=(4.0, 8.0))
    ys = [ch.push(x[s:s + 50]) for s in range(0, x.size, 50)]
    np.testing.assert_allclose(
        np.concatenate(ys), cf.bandpass_oneshot(x, band_hz=(4.0, 8.0)), atol=1e-10
    )


def test_zi_injection_makes_streaming_continue_oneshot_exactly():
    """spec §4.2-7：前段 zi=0 一次性滤波取 zf，后段用 zf 续接，
    拼接结果与整段 zi=0 一次性单向滤波逐点相等。"""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(30000 + 50 * 6) * 20.0  # 120s 校准 + 6 段
    head, tail = x[:30000], x[30000:]

    ch_head = cf.CausalChannel(band_hz=(4.0, 8.0))
    y_head = ch_head.push(head)
    zf_hp, zf_lp = ch_head.states()

    ch_tail = cf.CausalChannel(band_hz=(4.0, 8.0), zi=(zf_hp, zf_lp))
    y_tail = np.concatenate([ch_tail.push(tail[s:s + 50]) for s in range(0, tail.size, 50)])

    ref = cf.bandpass_oneshot(x, band_hz=(4.0, 8.0))
    np.testing.assert_array_equal(y_head, ref[:30000])
    np.testing.assert_array_equal(y_tail, ref[30000:])


def test_bad_zi_length_raises_before_state_change():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        cf.CausalChannel(zi=(np.zeros(2), np.zeros(4)))
    ch = cf.CausalChannel()
    good = rng.standard_normal(50)
    ch.push(good)
    with pytest.raises(ValueError):
        # 推一段后用错误长度替换：构造即失败，不影响已存在的滤波器
        cf.CausalChannel(zi=(np.zeros(2), np.zeros(4)))


def test_empty_segment_preserves_zi():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(50 * 4) * 20.0
    ch = cf.CausalChannel()
    z_before = ch.states()[0].copy()
    out_empty = ch.push(np.empty(0))
    assert out_empty.size == 0
    np.testing.assert_array_equal(ch.states()[0], z_before)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_it4_causal_channel.py -q`
Expected: FAIL（`AttributeError: module 'bci_sys.replay.causal_filter' has no attribute 'CausalChannel'`）

- [ ] **Step 3: 实现**

把 [src/bci_sys/replay/causal_filter.py](src/bci_sys/replay/causal_filter.py) 整体替换为下面内容（docstring 保留原口径；唯一行为差异是新增类与可选参数）：

```python
"""流式因果带通（实时路径专用，C3）。

【严禁 filtfilt / 零相位】零相位滤波要用到未来样本，只能出现在 offline/。
本模块只用单向 scipy.signal.lfilter：4 阶 Butterworth 高通串 4 阶
低通，每节、每通道各自携带 zi 跨块延续。初始 zi=0，因此
50 点一块的流式输出与整段一次性同参数单向 lfilter 逐点一致（测试锁定）。
零初始状态造成的启动瞬态由 QC 校准时剔除前 10s 解决。

CausalChannel 支持注入/导出两节 zi：IT4 θ 链路在校准窗一次性滤波后
用末端 zf 续接运行期逐块滤波（与整段一次性单向滤波逐点相等）。
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .. import config


def design_filter_coeffs(
    sfreq: float = config.SFREQ,
    band_hz: tuple[float, float] = config.FILTER_BAND_HZ,
):
    """返回 ((b_hp, a_hp), (b_lp, a_lp))。"""
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
    """单节单向 lfilter，zi 跨块延续；zi 可在构造时注入（拷贝，防外部别名）。"""

    def __init__(self, b: np.ndarray, a: np.ndarray, zi: np.ndarray | None = None):
        self.b, self.a = b, a
        n = max(len(a), len(b)) - 1
        if zi is None:
            self.zi = np.zeros(n)
        else:
            z = np.asarray(zi, dtype=float)
            if z.shape != (n,):
                raise ValueError(f"zi 长度 {z.shape} != 要求 ({n},)")
            self.zi = z.copy()

    def push(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = signal.lfilter(
            self.b, self.a, np.asarray(x, dtype=float), zi=self.zi
        )
        return y


class CausalChannel:
    """单通道「高通串低通」两节单向滤波；可注入/导出两节 zi。

    zi=(zi_hp, zi_lp)，任一元素为 None 表示该节零初始。
    空块原样返回且不触碰 zi（scipy lfilter 对空输入会返回错误 zf）。
    """

    def __init__(
        self,
        sfreq: float = config.SFREQ,
        band_hz: tuple[float, float] = config.FILTER_BAND_HZ,
        zi: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
    ):
        (b_hp, a_hp), (b_lp, a_lp) = design_filter_coeffs(sfreq, band_hz)
        zi_hp, zi_lp = zi
        self.hp = _CausalSection(b_hp, a_hp, zi=zi_hp)
        self.lp = _CausalSection(b_lp, a_lp, zi=zi_lp)

    def push(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"通道块必须一维，收到 ndim={arr.ndim}")
        if arr.size == 0:
            return np.empty(0, dtype=float)
        return self.lp.push(self.hp.push(arr))

    def states(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (zf_hp, zf_lp) 拷贝。"""
        return self.hp.zi.copy(), self.lp.zi.copy()


class CausalBandpass:
    """逐块推进的因果带通：每通道一个独立的 CausalChannel。

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
        self._ch = {
            c: CausalChannel(sfreq=sfreq, band_hz=band_hz) for c in self.channels
        }

    def push_block(self, block: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """所有通道块必须等长且一维；返回同结构滤波后块（单位与输入相同，µV）。

        空块直接原样返回、不触碰 zi（实测 scipy lfilter 对空输入会返回错误的
        非零 zf 并永久污染 IIR 状态）；非法形状在推进任何状态【之前】报错，
        保证调用失败不污染滤波器（分块==整段不变量不被边界输入击穿）。
        """
        arrays = {c: np.asarray(block[c], dtype=float) for c in self.channels}
        sizes = {a.size for a in arrays.values()}
        ndims = {a.ndim for a in arrays.values()}
        if len(sizes) != 1 or ndims != {1}:
            raise ValueError(f"各通道块必须等长且一维: sizes={sizes}, ndims={ndims}")
        if sizes == {0}:
            return {c: np.empty(0, dtype=float) for c in self.channels}
        return {c: self._ch[c].push(arrays[c]) for c in self.channels}
```

- [ ] **Step 4: 更新旧测试的私有属性路径**

[tests/test_causal_filter.py](tests/test_causal_filter.py) 的 `test_2d_or_mismatched_block_raises_before_state_change` 原来检查 `f._hp["Fz"].zi`，重构后路径变为 `f._ch["Fz"].hp.zi`。把该测试整体替换为：

```python
def test_2d_or_mismatched_block_raises_before_state_change():
    rng = np.random.default_rng(9)
    f = cf.CausalBandpass(["Fz", "FP1"])
    good = rng.standard_normal(50)
    f.push_block({"Fz": good, "FP1": good})  # 先推进一个正常块建立状态
    hp_zi_before = f._ch["Fz"].hp.zi.copy()
    with pytest.raises(ValueError):
        f.push_block({"Fz": good, "FP1": good.reshape(25, 2)})  # 等 size 但二维
    with pytest.raises(ValueError):
        f.push_block({"Fz": good, "FP1": good[:10]})            # 长度不等
    np.testing.assert_array_equal(f._ch["Fz"].hp.zi, hp_zi_before)  # 状态未被污染
```

- [ ] **Step 5: 跑全部测试**

Run: `uv run pytest tests/test_it4_causal_channel.py tests/test_causal_filter.py -q && uv run pytest -q`
Expected: 新测试全 PASS、旧滤波器测试全 PASS；总计 `100 passed`（95+5）。

- [ ] **Step 6: Commit**

```bash
git add src/bci_sys/replay/causal_filter.py tests/test_it4_causal_channel.py tests/test_causal_filter.py
git commit -m "refactor(it4): CausalChannel 支持 zi 注入/导出，CausalBandpass 行为保持

Co-Authored-By: Claude <noreply@anthropic.com>"
```
---

### Task 3: theta_metric（4–8Hz 校准重建/zf 续接/段功率/仅有效段平滑/P20 冻结）

纯逻辑、无 I/O、无墙钟。对应 spec §4.2 步骤 3–6 与 §4.3-2/3/4。

**Files:**
- Create: `src/bci_sys/replay/theta_metric.py`
- Test: `tests/test_theta_metric.py`（新建）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_theta_metric.py`：

```python
"""θ 指标纯逻辑：段功率、仅有效段平滑、P20 冻结、direction 显式、zf 续接。"""

import numpy as np
import pytest

from bci_sys import config
from bci_sys.replay import causal_filter, theta_metric as tm


def test_segment_powers_known_values():
    # 两段：常数 2（平方均值 4）、[3,4]（平方均值 12.5），尾巴 3 点丢弃
    x = np.array([2, 2, 2, 3, 4, 0, 0], dtype=float)
    p = tm.segment_powers(x, seg_samples=3)
    np.testing.assert_allclose(p, [4.0, 25.0 / 3.0])


def test_smoother_valid_only_and_invalid_holds_window():
    sm = tm.ValidOnlySmoother(win=3)
    assert sm.update(10.0, True) is None      # 未满
    assert sm.update(20.0, True) is None
    assert sm.update(30.0, True) == 20.0
    assert sm.update(999.0, False) is None    # 无效：不入窗、返回 None
    assert sm.update(float("nan"), True) is None  # 非有限：同样不进窗
    assert sm.update(40.0, True) == 30.0      # 窗=[20,30,40]，最旧 10 已弹


def test_smoother_rejects_win_below_one():
    with pytest.raises(ValueError):
        tm.ValidOnlySmoother(win=0)


def test_hit_decider_direction_required_and_validated():
    with pytest.raises(TypeError):
        tm.HitDecider(5.0)                    # C1：无默认，缺参即 TypeError
    with pytest.raises(ValueError):
        tm.HitDecider(5.0, "sideways")
    with pytest.raises(ValueError):
        tm.HitDecider(float("nan"), "up")
    up = tm.HitDecider(5.0, "up")
    assert up.hit(5.0) and up.hit(6.0) and not up.hit(4.99)
    down = tm.HitDecider(5.0, "down")
    assert down.hit(5.0) and not down.hit(5.01)


def _sine_clean(n_seg=600, freq=5.0, amp=10.0):
    # 5Hz @250Hz：每 50 点恰好 1 个整周期；tile 一张周期表保证各段输入
    # 【比特一致】（用全局角 np.sin(2π·5·n/250) 时大角度自变量带来 ~1e-10 抖动）。
    # 但 4Hz 高通极点 |z|≈0.962、IIR 内部状态尺度大，稳态轨道仍带 ~1e-12
    # 输出级舍入噪声，段功率不可能逐比特相等——用例只锁"相对一致的平台"。
    per = int(round(config.SFREQ / freq))  # 5Hz → 50 点/周期
    one = amp * np.sin(2 * np.pi * freq * np.arange(per) / config.SFREQ)
    return np.tile(one, n_seg)


def test_calibrate_constant_amplitude_locks_linear_quantile():
    # 恒定幅值 5Hz 正弦 → 稳态段功率平台（IIR 舍入带造成 ~1e-12 相对展宽，
    # 非比特相等）；成熟期固定：前 10s=50 段剔除，窗 5 段成熟。
    x = _sine_clean()
    valid = np.ones(600, dtype=bool)
    fr = tm.calibrate_theta(x, valid)
    assert fr["n_thr_segs"] == 550
    assert fr["cal_powers"].shape == (600,)
    # 前 10s 与未成熟位置 smoothed 为 null；之后全部有限
    assert fr["cal_smoothed"][49] is None
    assert all(v is not None for v in fr["cal_smoothed"][50:])
    assert np.isfinite(fr["theta_thr_uV2"])
    assert fr["theta_thr_uV2"] > 0
    mature = np.asarray(fr["cal_smoothed"][50:], dtype=float)
    # 平台锁：550 个成熟值相对展宽 < 1e-9（实测 ~2e-12），阈值即平台水平
    assert np.ptp(mature) / float(np.median(mature)) < 1e-9
    assert fr["theta_thr_uV2"] == pytest.approx(float(np.median(mature)), rel=1e-9)
    # method="linear" 在相邻次序统计量间插值：即便 ulp 级展宽也无 ties，
    # 达标率由分位秩决定 ≈0.80（不会是 1）——真实数据上的 P20 语义（spec §9）
    assert fr["calib_hit_rate"] == pytest.approx(
        float(np.mean(mature >= fr["theta_thr_uV2"])), rel=1e-12)
    assert 0.79 <= fr["calib_hit_rate"] <= 0.81
    zf_hp, zf_lp = fr["zf"]
    assert zf_hp.shape == (4,) and zf_lp.shape == (4,)  # 4 阶节状态长度 4


def test_calibrate_p20_linear_interpolation_locked():
    # 600 段 5Hz 正弦（50 点=整周期，tile 周期表、段边界精确过零）：
    # 前 50 段（t1<=10s）幅值 2（剔除），seg50–159 低幅 1（110 段），
    # seg160–599 高幅 3（440 段）。
    # 4Hz 高通极点 |z|≈0.962（每段仅衰减 0.146），阶跃后段功率需 ~9 段才沉入
    # 1e-8 带，叠加 W=5 窗历史混合（4 段）：实测低侧 13 个、高侧 14 个过渡值。
    per = 50
    table = np.sin(2 * np.pi * 5.0 * np.arange(per) / config.SFREQ)
    amps = np.array(
        [2.0 if j < 50 else (1.0 if j < 160 else 3.0) for j in range(600)])
    x = np.tile(table, 600) * np.repeat(amps, per)
    fr = tm.calibrate_theta(x, np.ones(600, dtype=bool))
    E = np.asarray([s for s in fr["cal_smoothed"] if s is not None])
    assert E.size == 550  # t1>10s 且窗成熟
    # 核心锁定：阈值必须按 numpy 2.0 method="linear" 从同一份成熟值算出
    assert fr["theta_thr_uV2"] == float(
        np.quantile(E, 0.20, method="linear"))
    # n=550 时 P20 线性位次 (550-1)*.2=109.8：阈值严格夹在 srt[109] 与
    # srt[110] 之间（linear 插值；nearest/更低/更高分位算法都会落一侧）
    srt = np.sort(E)
    assert srt[109] < fr["theta_thr_uV2"] < srt[110]
    # 440/550≈0.80 在高幅平台 → P20 阈值给出 ≈0.80 校准达标率（spec §9 构造分布）
    assert 0.77 <= fr["calib_hit_rate"] <= 0.83
    assert fr["calib_hit_rate"] == pytest.approx(
        float(np.mean(E >= fr["theta_thr_uV2"])), rel=1e-12)
    # 两平台各去掉实测十余个过渡值后严格相等（断言留裕度：>=90 / >=420）
    low_med = float(np.median(E[:120]))
    high_med = float(np.median(E[-440:]))
    assert int(np.count_nonzero(np.abs(E - low_med) < 1e-8)) >= 90
    assert int(np.count_nonzero(np.abs(E - high_med) < 1e-8)) >= 420


def test_calibrate_too_few_mature_segments_raises():
    x = _sine_clean()
    valid = np.zeros(600, dtype=bool)
    valid[50:90] = True  # t1>10s 的有效段仅 40 < 50
    with pytest.raises(tm.FreezeFailed):
        tm.calibrate_theta(x, valid, min_mature_segs=50)


def test_calibrate_validity_shape_mismatch_raises():
    with pytest.raises(ValueError):
        tm.calibrate_theta(_sine_clean(), np.ones(599, dtype=bool))


def test_running_stream_with_zf_equals_one_shot_one_way_exactly():
    """spec §4.2-7 不变量：校准 zf 续接的逐块流式 θ == 整段 zi=0 单向滤波。"""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(30000 + 50 * 12) * 20.0  # 120s + 12 运行段
    head, tail = x[:30000], x[30000:]
    fr = tm.calibrate_theta(head, np.ones(600, dtype=bool))

    stream = tm.ThetaStream(zf=fr["zf"])
    ys = []
    powers = []
    for s in range(0, tail.size, 50):
        y, p = stream.push_segment(tail[s:s + 50])
        ys.append(y)
        powers.append(p)
    y_stream = np.concatenate(ys)

    ref = causal_filter.bandpass_oneshot(x, band_hz=config.THETA_BAND_HZ)
    np.testing.assert_array_equal(y_stream, ref[30000:])
    # 段功率与参考序列重算一致
    ref_p = tm.segment_powers(ref[30000:])
    np.testing.assert_allclose(np.array(powers), ref_p, rtol=1e-12)


def test_theta_stream_default_zi_is_zero():
    rng = np.random.default_rng(8)
    x = rng.standard_normal(50 * 3) * 20.0
    stream = tm.ThetaStream()
    ys = [stream.push_segment(x[s:s + 50])[0] for s in range(0, x.size, 50)]
    # 默认 zi=None 等价零初始态：逐块拼接与一次性单向滤波逐点相等（实测 max diff 0.0）
    np.testing.assert_array_equal(
        np.concatenate(ys),
        causal_filter.bandpass_oneshot(x, band_hz=config.THETA_BAND_HZ),
    )
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_theta_metric.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'bci_sys.replay.theta_metric'`）

- [ ] **Step 3: 实现**

创建 `src/bci_sys/replay/theta_metric.py`：

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_theta_metric.py -q`
Expected: 10 passed。若 P20 用例在严格插值断言（`srt[109] < thr < srt[110]`）或平台计数上失败，**不得**改断言语义；先打印 `np.sort(E)` 确认阶跃过渡段是否超出预期的各 4 个（核心锁定是 `theta_thr_uV2 == np.quantile(E, .2, method="linear")` 与 ≈0.80 构造达标率）。

- [ ] **Step 5: 全量回归 + AST 守卫**

Run: `uv run pytest -q`
Expected: `110 passed`（100+10）；`tests/test_boundary.py` 自动覆盖新文件（无 offline import、无 filtfilt）。

- [ ] **Step 6: Commit**

```bash
git add src/bci_sys/replay/theta_metric.py tests/test_theta_metric.py
git commit -m "feat(it4): theta_metric——4–8Hz 校准冻结/zf 续接/仅有效段平滑/P20（C1/C4/C5）

Co-Authored-By: Claude <noreply@anthropic.com>"
```
---

### Task 4: qc.segment_verdict 纯函数 + feedback 会话状态机

冻结瞬间要在**校准缓冲**上用与运行期完全相同的 L1/L3 规则重算 600 段有效性（spec §4.2-4）。把 `QcJudge._running_verdict` 的计算体抽成公开纯函数（行为保持），再写纯会话状态机（spec §4.3–4.5、§5）。

**Files:**
- Modify: [src/bci_sys/replay/qc.py](src/bci_sys/replay/qc.py)
- Create: `src/bci_sys/replay/feedback.py`
- Test: `tests/test_feedback_session.py`（新建，含 segment_verdict 一致性用例）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_feedback_session.py`：

```python
"""IT4 会话状态机 + qc.segment_verdict 复用（spec §4.3–4.5、§5）。"""

import numpy as np
import pytest

from bci_sys import config
from bci_sys.replay import causal_filter, feedback, qc, theta_metric as tm


def test_segment_verdict_matches_frozen_judge(data_root):
    """校准重建用的纯函数与冻结后 QcJudge 的逐段裁决必须同源同结果。"""
    chans = {}
    import bci_sys.io_bids as io_bids
    raw = io_bids.load_channels_uV("001", ["Fz", "FP1", "FP2"], data_root)
    n = int(config.QC_CALIBRATION_SEC * config.SFREQ) + 50 * 4
    bp = causal_filter.CausalBandpass(("Fz", "FP1", "FP2"))
    filtered = bp.push_block({c: raw[c][:n] for c in ("Fz", "FP1", "FP2")})
    judge = qc.QcJudge()
    for k in range(n // 50):
        sl = slice(k * 50, (k + 1) * 50)
        v = judge.push((k + 1) * 0.2, {c: filtered[c][sl] for c in ("Fz", "FP1", "FP2")})
    assert judge.frozen
    for k in range(600, n // 50):
        sl = slice(k * 50, (k + 1) * 50)
        got = qc.segment_verdict(
            filtered["Fz"][sl], filtered["FP1"][sl], filtered["FP2"][sl],
            judge.b_fp1, judge.b_fp2, config.QC_L1_P2P_UV, config.QC_L3_EOG_P2P_UV,
        )
        exp = judge.push((k + 1) * 0.2, {c: filtered[c][sl] for c in ("Fz", "FP1", "FP2")})
        assert got["valid"] == exp["valid"]
        assert got["reasons"] == exp["reasons"]
        # 同源纯函数 + 同一份输入：必须位级相等，approx 会掩盖未来两条路径的漂移
        assert got["fz_clean_p2p"] == exp["fz_clean_p2p"]
        assert got["eog_p2p"] == exp["eog_p2p"]


def _freeze_info(n=550, thr=16.7, rate=0.8, b1=0.248, b2=0.378):
    return {"n_thr_segs": n, "theta_thr_uV2": thr, "calib_hit_rate": rate,
            "b_fp1": b1, "b_fp2": b2}


def test_freeze_note_and_block_lifecycle():
    s = feedback.FeedbackSession()
    assert s.phase == "calibration"
    s.note_freeze(_freeze_info())
    assert s.phase == "freeze_card" and s.freeze_info["theta_thr_uV2"] == 16.7

    row = s.begin_block(120.0)
    assert row == {"type": "phase", "phase": "block_start", "block_idx": 0, "t1": 120.0}
    for j in range(600):
        s.record_segment(valid=(j % 4 != 0), power_finite=True, hit=(j % 2 == 0))
    # valid=450（j%4∈{1,2,3}）；命中=有效且偶数=j%4==2 → 150
    summ = s.end_block_if_due(240.0)
    assert summ["type"] == "block_summary" and summ["block_idx"] == 0
    assert summ["n_valid"] == 450 and summ["n_invalid"] == 150
    assert summ["n_hit"] == 150
    assert summ["hit_rate"] == round(150 / 450, 6)
    assert s.blocks_completed == 1 and s.block_rates[-1] == summ["hit_rate"]


def test_rest_does_not_advance_any_state():
    s = feedback.FeedbackSession()
    s.note_freeze(_freeze_info())
    s.begin_block(120.0)
    for _ in range(600):
        s.record_segment(valid=True, power_finite=True, hit=True)
    s.end_block_if_due(240.0)
    r1 = s.begin_rest(30.0)
    assert r1 == {"type": "phase", "phase": "rest_start", "rest_idx": 0, "rest_sec": 30.0}
    counts = (s.total.n_valid, s.total.n_hit, s.blocks_completed)
    r2 = s.end_rest(30.0)
    assert r2 == {"type": "phase", "phase": "rest_end", "rest_idx": 0, "rest_sec": 30.0}
    assert (s.total.n_valid, s.total.n_hit, s.blocks_completed) == counts
    row = s.begin_block(240.0)
    assert row["block_idx"] == 1 and s.seg_in_block == 0


def test_counters_frozen_on_invalid_segment_c5():
    s = feedback.FeedbackSession()
    s.note_freeze(_freeze_info())
    s.begin_block(120.0)
    s.record_segment(valid=True, power_finite=True, hit=True)
    s.record_segment(valid=False, power_finite=True, hit=None)
    s.record_segment(valid=True, power_finite=False, hit=None)  # 实测不可达守卫：不动计数
    assert s.total.n_valid == 1 and s.total.n_hit == 1 and s.total.n_invalid == 1
    assert s.seg_in_block == 3


def test_block_rate_zero_division_returns_zero():
    s = feedback.FeedbackSession()
    s.note_freeze(_freeze_info())
    s.begin_block(120.0)
    for _ in range(600):
        s.record_segment(valid=False, power_finite=True, hit=None)
    summ = s.end_block_if_due(240.0)
    assert summ["n_valid"] == 0 and summ["hit_rate"] == 0.0


def test_full_six_blocks_final_totals():
    s = feedback.FeedbackSession()
    s.note_freeze(_freeze_info())
    for b in range(6):
        s.begin_block(120.0 + b * 120.0)
        for j in range(600):
            # j=6（6%3==0）无效段恰为应命中位 → 锁住 C5"无效段不计 hit"
            s.record_segment(valid=(j != 6), power_finite=True, hit=(j % 3 == 0 and j != 6))
        s.end_block_if_due(240.0 + b * 120.0)
        if b < 5:
            s.begin_rest(30.0)
            s.end_rest(30.0)
    final = s.finish()
    assert final["type"] == "final" and final["status"] == "ok"
    assert final["n_valid"] == 6 * 599 and final["n_invalid"] == 6
    assert final["n_hit"] == sum(1 for j in range(600) if j % 3 == 0 and j != 6) * 6
    assert len(final["block_rates"]) == 6
    assert final["overall_hit_rate"] == round(final["n_hit"] / final["n_valid"], 6)
    assert s.phase == "finished"


def test_freeze_with_too_few_segments_raises():
    s = feedback.FeedbackSession()
    with pytest.raises(tm.FreezeFailed):
        s.note_freeze(_freeze_info(n=49))
    assert s.phase == "calibration"  # 未进入冻结卡


def test_terminate_rows_and_reason_whitelist():
    s = feedback.FeedbackSession()
    s.note_freeze(_freeze_info())
    s.begin_block(120.0)
    s.record_segment(valid=True, power_finite=True, hit=False)
    row = s.terminate("user_abort", k=600, t1=120.2)
    assert row["type"] == "aborted" and row["status"] == "aborted"
    assert row["reason"] == "user_abort" and row["k"] == 600
    assert row["blocks_completed"] == 0 and s.phase == "aborted"
    with pytest.raises(ValueError):
        feedback.FeedbackSession().terminate("nope", k=0, t1=0.0)


def test_wall_cap_terminate():
    s = feedback.FeedbackSession(wall_cap_sec=100.0)
    assert s.wall_cap_terminate(99.9, k=10, t1=2.2) is None
    row = s.wall_cap_terminate(100.1, k=10, t1=2.2)
    assert row["reason"] == "wall_cap"
    # 已终态后不再出第二行
    assert s.wall_cap_terminate(999.0, k=11, t1=2.4) is None


def test_engine_error_row_carries_error_name():
    s = feedback.FeedbackSession()
    row = s.terminate("engine_error", k=3, t1=0.8, error="RuntimeError")
    assert row["error"] == "RuntimeError"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_feedback_session.py -q`
Expected: FAIL（`ImportError: cannot import name 'segment_verdict'` / `No module named 'bci_sys.replay.feedback'`）

- [ ] **Step 3: 抽 qc.segment_verdict（行为保持）**

在 [src/bci_sys/replay/qc.py](src/bci_sys/replay/qc.py) 中，把 `_running_verdict` 之前插入模块级纯函数：

```python
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
```

然后把 `QcJudge._running_verdict` 整体替换为：

```python
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
```

- [ ] **Step 4: 实现 feedback.py**

创建 `src/bci_sys/replay/feedback.py`：

```python
"""IT4 反馈会话状态机（纯逻辑：无 I/O、无线程、无真实时钟来源）。

相位（spec §5.1）：
    calibration → freeze_card → block ↔ rest ×6 → finished
异常终态 aborted：user_abort / freeze_failed / incomplete_data / wall_cap / engine_error。
相位推进权只在引擎同步循环（按已发射段硬推进），页面无继续/延长/跳过控件（C7）。
rest/freeze_card 只走墙钟、不推进任何数据侧状态（§4.4）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import config
from .theta_metric import FreezeFailed


@dataclass
class _Counts:
    n_valid: int = 0
    n_invalid: int = 0
    n_hit: int = 0


_ABORT_REASONS = (
    "user_abort", "freeze_failed", "incomplete_data", "wall_cap", "engine_error",
)


class FeedbackSession:
    """计数语义（spec §4.3/4.5，C5）：
    - valid 且 θ 有限：n_valid/n_hit 才动；无效段：只动 n_invalid；
    - valid 但 θ 非有限（实测不可达的守卫分支）：三者都不动；
    - hit_rate 分母只含 valid 段；n_valid=0 → 0.0，不除零。
    """

    def __init__(
        self,
        *,
        calib_segs: int | None = None,
        n_blocks: int | None = None,
        block_segs: int | None = None,
        wall_cap_sec: float | None = None,
        min_calib_segs: int | None = None,
    ):
        # 默认值在构造时从 config 读（不用 def-time 默认参数），便于测试 monkeypatch
        self.calib_segs = int(round(config.QC_CALIBRATION_SEC / config.SEGMENT_SEC)) \
            if calib_segs is None else calib_segs
        per_block = int(round(config.FEEDBACK_BLOCK_SEC / config.SEGMENT_SEC))
        self.n_blocks = config.FEEDBACK_N_BLOCKS if n_blocks is None else n_blocks
        self.block_segs = per_block if block_segs is None else block_segs
        self.wall_cap_sec = config.SESSION_WALL_CAP_SEC if wall_cap_sec is None else wall_cap_sec
        self.min_calib_segs = (
            config.THETA_MIN_CALIB_SEGS if min_calib_segs is None else min_calib_segs
        )

        self.phase = "calibration"
        self.status: str | None = None   # None=运行中；"ok" / "aborted"
        self.reason: str | None = None
        self.block_idx = -1
        self.seg_in_block = 0
        self.blocks_completed = 0
        self.block_rates: list[float] = []
        self.summaries: list[dict] = []
        self.freeze_info: dict | None = None
        self._block_t1_start: float | None = None
        self._cur = _Counts()
        self.total = _Counts()

    # ---- 冻结 ----
    def note_freeze(self, theta_freeze: dict) -> None:
        n = int(theta_freeze["n_thr_segs"])
        if n < self.min_calib_segs:
            raise FreezeFailed(
                f"S3 成熟有效段 {n} < 最小要求 {self.min_calib_segs}，冻结失败"
            )
        thr = float(theta_freeze["theta_thr_uV2"])
        if not math.isfinite(thr):
            raise FreezeFailed("S3 θ 阈值非有限值，冻结失败")
        self.freeze_info = {
            "b_fp1": float(theta_freeze["b_fp1"]),
            "b_fp2": float(theta_freeze["b_fp2"]),
            "theta_thr_uV2": thr,
            "n_thr_segs": n,
            "calib_hit_rate": float(theta_freeze["calib_hit_rate"]),
        }
        self.phase = "freeze_card"

    # ---- block / rest ----
    def begin_block(self, t1: float) -> dict:
        self.block_idx += 1
        if not (0 <= self.block_idx < self.n_blocks):
            raise RuntimeError(f"block 越界: {self.block_idx}")
        self._cur = _Counts()  # 幂等：正常路径 end_block 已清零，防未完结块计数泄漏
        self.seg_in_block = 0
        self._block_t1_start = float(t1)
        self.phase = "block"
        return {
            "type": "phase", "phase": "block_start",
            "block_idx": self.block_idx, "t1": round(float(t1), 6),
        }

    def record_segment(self, *, valid: bool, power_finite: bool, hit: bool | None) -> None:
        if valid and power_finite:
            self._cur.n_valid += 1
            self.total.n_valid += 1
            if hit:
                self._cur.n_hit += 1
                self.total.n_hit += 1
        elif not valid:
            self._cur.n_invalid += 1
            self.total.n_invalid += 1
        self.seg_in_block += 1

    @staticmethod
    def _rate(c: _Counts) -> float:
        return round(c.n_hit / c.n_valid, 6) if c.n_valid else 0.0

    def end_block_if_due(self, t1_end: float) -> dict | None:
        if self.seg_in_block < self.block_segs:
            return None
        row = {
            "type": "block_summary",
            "block_idx": self.block_idx,
            "t1_start": round(float(self._block_t1_start), 6),
            "t1_end": round(float(t1_end), 6),
            "n_valid": self._cur.n_valid,
            "n_invalid": self._cur.n_invalid,
            "n_hit": self._cur.n_hit,
            "hit_rate": self._rate(self._cur),
        }
        self.summaries.append(row)
        self.block_rates.append(row["hit_rate"])
        self.blocks_completed += 1
        self._cur = _Counts()
        self.seg_in_block = 0
        return row

    def begin_rest(self, rest_sec: float) -> dict:
        rest_idx = self.blocks_completed - 1
        if not (0 <= rest_idx < self.n_blocks - 1):
            raise RuntimeError(f"rest 越界: {rest_idx}")
        self.phase = "rest"
        return {
            "type": "phase", "phase": "rest_start",
            "rest_idx": rest_idx, "rest_sec": float(rest_sec),
        }

    def end_rest(self, rest_sec: float) -> dict:
        rest_idx = self.blocks_completed - 1
        return {
            "type": "phase", "phase": "rest_end",
            "rest_idx": rest_idx, "rest_sec": float(rest_sec),
        }

    def finish(self) -> dict:
        if self.status is not None:
            raise RuntimeError(f"会话已终态（status={self.status}），不能重复 finish")
        if self.blocks_completed != self.n_blocks:
            raise RuntimeError(
                f"未完成全部 block 不能 finish：{self.blocks_completed}/{self.n_blocks}"
            )
        self.phase = "finished"
        self.status = "ok"
        return {
            "type": "final",
            "status": "ok",
            "n_valid": self.total.n_valid,
            "n_invalid": self.total.n_invalid,
            "n_hit": self.total.n_hit,
            "overall_hit_rate": self._rate(self.total),
            "block_rates": list(self.block_rates),
        }

    # ---- 异常终态 ----
    def terminate(self, reason: str, *, k: int, t1: float, error: str | None = None) -> dict:
        if reason not in _ABORT_REASONS:
            raise ValueError(f"未知 abort reason: {reason!r}")
        self.phase = "aborted"
        self.status = "aborted"
        self.reason = reason
        row = {
            "type": "aborted",
            "status": "aborted",
            "reason": reason,
            "k": int(k),
            "t1": round(float(t1), 6),
            "blocks_completed": self.blocks_completed,
        }
        if error is not None:
            row["error"] = str(error)
        return row

    def wall_cap_terminate(self, wall_elapsed: float, *, k: int, t1: float) -> dict | None:
        if self.status is not None:
            return None
        if wall_elapsed > self.wall_cap_sec:
            return self.terminate("wall_cap", k=k, t1=t1)
        return None
```

- [ ] **Step 5: 跑测试 + 全量回归**

Run: `uv run pytest tests/test_feedback_session.py tests/test_qc.py -q && uv run pytest -q`
Expected: 新测试全 PASS、既有 qc 测试全 PASS（纯函数抽取行为保持）；总计 `120 passed`（110+10；`test_segment_verdict_matches_frozen_judge` 用 data_root，无数据时 skip 不计入）。

- [ ] **Step 6: Commit**

```bash
git add src/bci_sys/replay/qc.py src/bci_sys/replay/feedback.py tests/test_feedback_session.py
git commit -m "feat(it4): segment_verdict 纯函数 + 反馈会话状态机（C5 计数/相位/wall_cap）

Co-Authored-By: Claude <noreply@anthropic.com>"
```
---

### Task 5: feedback_engine（引擎线程/EventBus/jsonl/run.json/终态）

spec §3 进程模型、§4 全流程、§5 终态、§7 落盘与 A4。风格与 IT3 `run_qc` 同构（preflight→load→print→mkdir；计数在等待前落盘；run.json 总写）。

**Files:**
- Create: `src/bci_sys/replay/feedback_engine.py`
- Test: `tests/test_feedback_engine.py`（新建）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_feedback_engine.py`：

```python
"""IT4 引擎：schema/顺序、A4 字节一致、abort/incomplete/freeze_failed/engine_error、
帧与 jsonl 同行一致、独立重算、内存上限、LF/非有限守卫。"""
import hashlib
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from bci_sys import config, io_bids
from bci_sys.replay import feedback_engine as eng
from bci_sys.replay import theta_metric as tm, qc, causal_filter


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class GateClock:
    """t 进入 (lo, hi) 墙钟窗后阻塞，直到 abort 置位（测中止落点）。"""
    def __init__(self, abort, lo, hi=1e9):
        self.t = 0.0
        self.abort = abort
        self.lo, self.hi = lo, hi

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt
        if self.lo < self.t < self.hi:
            self.abort.wait(5.0)


def _rows(rd: Path):
    return [json.loads(x) for x in (rd / "feedback.jsonl").read_text("utf-8").splitlines()]


def _run(data_root, tmp, rid="fixed", **kw):
    clk = FakeClock()
    out = eng.run_feedback(
        subj="001", root=data_root, out_root=tmp, run_id=rid,
        sleep_fn=clk.sleep, clock=clk.now, **kw)
    rd = tmp / "feedback" / "sub-001" / rid
    return out, rd, clk


def test_full_run_schema_order_and_counts(data_root, tmp_path):
    out, rd, clk = _run(data_root, tmp_path)
    assert out["status"] == "ok" and out["reason"] is None
    rows = _rows(rd)
    kinds = [r["type"] for r in rows]
    assert kinds.count("meta") == 1 and kinds[0] == "meta"
    assert kinds.count("calib_seg") == 600
    assert kinds.count("freeze") == 1
    assert kinds.count("seg") == 3600
    assert kinds.count("block_summary") == 6
    assert kinds.count("final") == 1 and kinds[-1] == "final"
    phases = [r for r in rows if r["type"] == "phase"]
    assert sum(r["phase"] == "block_start" for r in phases) == 6
    assert sum(r["phase"] == "rest_start" for r in phases) == 5
    assert sum(r["phase"] == "rest_end" for r in phases) == 5
    assert len(phases) == 16
    # calib_seg 必须整体在 freeze 之前；freeze 后立刻是 block_start 0
    fi = kinds.index("freeze")
    assert kinds[1:fi] == ["calib_seg"] * 600
    assert rows[fi + 1] == {"type": "phase", "phase": "block_start",
                            "block_idx": 0, "t1": 120.0}
    # 每块：600 seg 后一条 summary；rest 夹在块间且只带 rest_idx/rest_sec
    si = fi + 2
    for b in range(6):
        assert all(r["block_idx"] == b for r in rows[si:si + 600])
        assert [r["seg_in_block"] for r in rows[si:si + 600]] == list(range(600))
        sm = rows[si + 600]
        assert sm["type"] == "block_summary" and sm["block_idx"] == b
        assert sm["t1_start"] == 120.0 + 120 * b and sm["t1_end"] == 240.0 + 120 * b
        assert sm["n_valid"] + sm["n_invalid"] == 600
        assert sm["hit_rate"] == (round(sm["n_hit"] / sm["n_valid"], 6) if sm["n_valid"] else 0.0)
        si += 601
        if b < 5:
            rs, re_ = rows[si], rows[si + 1]
            assert rs["phase"] == "rest_start" and re_["phase"] == "rest_end"
            assert rs["rest_idx"] == b and re_["rest_idx"] == b
            assert rs["rest_sec"] == 30.0 and re_["rest_sec"] == 30.0
            assert "wall" not in rs and "t1" not in rs
            assert rows[si + 2]["phase"] == "block_start" and rows[si + 2]["block_idx"] == b + 1
            si += 3


def test_meta_freeze_seg_row_field_sets(data_root, tmp_path):
    _, rd, _ = _run(data_root, tmp_path)
    rows = _rows(rd)
    meta = rows[0]
    assert meta["schema_version"] == "1.0"
    assert meta["data_vhdr"] == "sub-001_task-nback_eeg.vhdr"
    assert meta["target"] == {"channel": "Fz", "band_hz": [4.0, 8.0], "direction": "up"}
    assert set(meta["params"]) == {
        "sfreq", "segment_sec", "qc_calibration_sec", "qc_filter_warmup_sec",
        "l1_p2p_uv", "l3_eog_p2p_uv", "theta_band_hz", "theta_smooth_segs",
        "theta_quantile", "theta_target_rate", "theta_min_calib_segs",
        "n_blocks", "block_sec", "rest_sec", "freeze_card_sec", "session_wall_cap_sec"}
    assert meta["params"]["theta_smooth_segs"] == 5
    fz = rows[601]
    assert set(fz) == {"type", "t_sec", "b_fp1", "b_fp2", "theta_thr_uV2",
                       "smooth_win_seg", "quantile", "target_rate", "direction",
                       "n_thr_segs", "calib_hit_rate", "n_fit_samples", "fit_window_sec"}
    assert fz["t_sec"] == 120.0 and fz["n_fit_samples"] == 27500
    assert fz["fit_window_sec"] == [10.0, 120.0] and fz["direction"] == "up"
    assert fz["quantile"] == 0.20 and fz["target_rate"] == 0.80
    assert 0.1 <= fz["b_fp1"] <= 0.7 and 0.1 <= fz["b_fp2"] <= 0.7
    assert fz["n_thr_segs"] >= 450 and math.isfinite(fz["theta_thr_uV2"])
    seg = next(r for r in rows if r["type"] == "seg")
    assert set(seg) == {"type", "k", "t1", "block_idx", "seg_in_block",
                        "theta_pow_uV2", "smoothed_uV2", "hit",
                        "fz_clean_p2p_uv", "eog_p2p_uv", "valid", "reasons"}
    assert seg["k"] == 600 and seg["t1"] == 120.2
    # 实测无效段分布（只依赖 QC，与 θ 平滑无关；spec §2.6）
    summaries = [r for r in rows if r["type"] == "block_summary"]
    assert [s["n_invalid"] for s in summaries] == [43, 41, 57, 71, 85, 108]
    for s in summaries:
        assert 0.5 < s["hit_rate"] < 0.99  # 宽结构断言；具体数值进 evidence
    final = rows[-1]
    assert final["status"] == "ok" and len(final["block_rates"]) == 6
    assert final["n_valid"] == sum(s["n_valid"] for s in summaries)
    assert final["n_hit"] == sum(s["n_hit"] for s in summaries)


def test_run_json_matches_rows_and_wall_separated(data_root, tmp_path):
    _, rd, clk = _run(data_root, tmp_path)
    run = json.loads((rd / "run.json").read_text("utf-8"))
    rows = _rows(rd)
    assert run["status"] == "ok" and run["mode"] == "live"
    assert run["n_calib_seg"] == 600 and run["n_running_seg"] == 3600
    assert run["data_elapsed_s"] == 840.0
    assert run["wall_elapsed_s"] > run["data_elapsed_s"]  # 含 150s rest + 3s 卡
    assert 0.8 < run["rate_ratio"] < 0.9
    assert len(run["block_rates"]) == 6 and "git_commit" in run
    raw = (rd / "feedback.jsonl").read_bytes()
    assert b"\r" not in raw  # LF 锁定
    # 墙钟/线程/队列量只进 run.json，绝不进 jsonl（A4）。
    # 注意不能按子串 b"wall" 扫——冻结参数名 session_wall_cap_sec 合法含 "wall"。
    forbidden = {"wall_elapsed_s", "wall_started_iso", "rate_ratio", "seq", "dropped"}
    assert all(forbidden.isdisjoint(r) for r in rows)
    assert not any(k.endswith("_iso") for r in rows for k in r)


def test_a4_byte_identical_two_runs(data_root, tmp_path):
    bs = []
    for rid in ("a", "b"):
        clk = FakeClock()
        eng.run_feedback(subj="001", root=data_root, out_root=tmp_path / rid,
                         run_id="fixed", sleep_fn=clk.sleep, clock=clk.now)
        bs.append((tmp_path / rid / "feedback" / "sub-001" / "fixed" /
                   "feedback.jsonl").read_bytes())
    assert bs[0] == bs[1]
    assert hashlib.sha256(bs[0]).hexdigest() == hashlib.sha256(bs[1]).hexdigest()


def test_bus_frames_match_jsonl_rows_and_ticks_live_only(data_root, tmp_path):
    clk = FakeClock()
    bus = eng.EventBus(maxsize=100000)
    eng.run_feedback(subj="001", root=data_root, out_root=tmp_path, run_id="fixed",
                     sleep_fn=clk.sleep, clock=clk.now, bus=bus)
    frames = []
    while True:
        try:
            frames.append(bus.q.get_nowait())
        except Exception:
            break
    assert [f["msg"]["type"] for f in frames[:600]] == ["calib_tick"] * 600
    assert frames[0]["msg"] == {"type": "calib_tick", "k": 0, "t1": 0.2}
    assert all(f["seq"] == i + 1 for i, f in enumerate(frames))  # seq 连续
    rows = _rows(tmp_path / "feedback" / "sub-001" / "fixed")
    # meta 与 600 calib_seg 不走 live WS；其余帧 msg 与 jsonl 同行逐字相等
    persisted = rows[601:]
    assert [f["msg"] for f in frames[600:]] == persisted


def test_abort_preset_before_start(data_root, tmp_path):
    ev = threading.Event()
    ev.set()
    clk = FakeClock()
    out = eng.run_feedback(subj="001", root=data_root, out_root=tmp_path,
                           run_id="a", sleep_fn=clk.sleep, clock=clk.now, abort_event=ev)
    assert out["status"] == "aborted" and out["reason"] == "user_abort"
    rd = tmp_path / "feedback" / "sub-001" / "a"
    rows = _rows(rd)
    assert rows[-1]["reason"] == "user_abort"
    assert not any(r["type"] in ("calib_seg", "freeze", "seg") for r in rows)
    run = json.loads((rd / "run.json").read_text("utf-8"))
    assert run["status"] == "aborted" and run["n_calib_seg"] == 0 and run["n_running_seg"] == 0


def test_abort_during_first_rest(data_root, tmp_path):
    # GateClock 与 setter 必须操作 runner 自己的 abort Event（引擎只认它）；
    # setter 轮询里 time.sleep 让出 GIL，否则忙等会饿死引擎线程。
    runner = eng.FeedbackRunner(subj="001", root=data_root, out_root=tmp_path, run_id="r")
    gate = GateClock(runner.abort, lo=243.5)  # block0 后 30s rest 窗内
    # 先让引擎进 rest 再置 abort
    def setter():
        while runner.bus.snapshot is None or runner.bus.snapshot.get("phase") != "rest":
            if runner.outcome is not None:
                return
            time.sleep(0.001)
        runner.abort.set()
    threading.Thread(target=setter, daemon=True).start()
    runner.start(sleep_fn=gate.sleep, clock=gate.now)
    runner.join(10.0)
    assert not runner.alive
    assert runner.outcome["reason"] == "user_abort"
    rows = _rows(tmp_path / "feedback" / "sub-001" / "r")
    assert sum(r["type"] == "seg" for r in rows) == 600
    assert rows[-1]["blocks_completed"] == 1
    assert any(r["type"] == "phase" and r["phase"] == "rest_start" for r in rows)
    assert not any(r["type"] == "phase" and r["phase"] == "rest_end" for r in rows)


def test_incomplete_data_mid_block(data_root, tmp_path, monkeypatch):
    # 截断到 1500 段（300s）：block0 完整、block1 中途耗尽
    orig = io_bids.load_channels_uV
    def truncated(subj, names, root=None):
        d = orig(subj, names, root)
        return {c: a[:1500 * 50] for c, a in d.items()}
    monkeypatch.setattr(eng.io_bids, "load_channels_uV", truncated)
    clk = FakeClock()
    out = eng.run_feedback(subj="001", root=data_root, out_root=tmp_path,
                           run_id="i", sleep_fn=clk.sleep, clock=clk.now)
    assert out["reason"] == "incomplete_data"
    rows = _rows(tmp_path / "feedback" / "sub-001" / "i")
    assert rows[-1]["blocks_completed"] == 1 and rows[-1]["k"] == 1499
    run = json.loads((tmp_path / "feedback" / "sub-001" / "i" / "run.json").read_text("utf-8"))
    assert run["status"] == "aborted" and run["reason"] == "incomplete_data"
    assert run["n_running_seg"] == sum(r["type"] == "seg" for r in rows)


def test_freeze_failed_writes_aborted_no_calib_rows(data_root, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise tm.FreezeFailed("x")
    monkeypatch.setattr(tm, "calibrate_theta", boom)
    clk = FakeClock()
    out = eng.run_feedback(subj="001", root=data_root, out_root=tmp_path,
                           run_id="ff", sleep_fn=clk.sleep, clock=clk.now)
    assert out["reason"] == "freeze_failed"
    rows = _rows(tmp_path / "feedback" / "sub-001" / "ff")
    assert rows[-1]["type"] == "aborted" and rows[-1]["reason"] == "freeze_failed"
    assert not any(r["type"] == "calib_seg" for r in rows)


def test_engine_error_unexpected_exception(data_root, tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(qc, "p2p_uv", boom)
    clk = FakeClock()
    out = eng.run_feedback(subj="001", root=data_root, out_root=tmp_path,
                           run_id="ee", sleep_fn=clk.sleep, clock=clk.now)
    assert out["status"] == "aborted" and out["reason"] == "engine_error"
    rows = _rows(tmp_path / "feedback" / "sub-001" / "ee")
    assert rows[-1]["error"] == "RuntimeError"
    run = json.loads((tmp_path / "feedback" / "sub-001" / "ee" / "run.json").read_text("utf-8"))
    assert run["reason"] == "engine_error"


def test_preflight_failure_writes_nothing(data_root, tmp_path):
    with pytest.raises(FileNotFoundError):
        eng.run_feedback(subj="999", root=data_root, out_root=tmp_path, run_id="z")
    assert not (tmp_path / "feedback").exists()


def test_preloaded_chans_bypasses_preflight_and_runs_ok(data_root, tmp_path, monkeypatch):
    """CLI 接缝：chans 预载后引擎不再碰 preflight/load（CLI 在 uvicorn 前完成它们）。"""
    raw = io_bids.load_channels_uV("001", list(eng.ALL_CH), data_root)
    short = {c: a[:4200 * 50] for c, a in raw.items()}  # 恰好 4200 段=600+3600

    def boom(*a, **k):
        raise AssertionError("预载 chans 时不应再调 preflight")

    monkeypatch.setattr(eng.io_bids, "preflight", boom)
    clk = FakeClock()
    out = eng.run_feedback(subj="001", root=data_root, out_root=tmp_path, run_id="p",
                           sleep_fn=clk.sleep, clock=clk.now, chans=short)
    assert out["status"] == "ok"
    assert len(_rows(tmp_path / "feedback" / "sub-001" / "p")) == 4225


def test_finite_or_none_guard():
    assert eng._finite_or_none(float("nan")) is None
    assert eng._finite_or_none(float("inf")) is None
    assert eng._finite_or_none(None) is None
    assert eng._finite_or_none(1.5) == 1.5


def test_calibration_buffer_footprint_under_10mb():
    # spec §4.1：三路 30000 float64 校准缓冲 = 720,000B ≈ 0.72MB，上限 < 10MB
    n_bytes = 3 * int(config.QC_CALIBRATION_SEC * config.SFREQ) * 8
    assert n_bytes == 720_000
    assert n_bytes < 10_000_000


def test_independent_recompute_matches_jsonl(data_root, tmp_path):
    """spec §7.3：只读 jsonl+meta 参数+原始数据，用公开原语重算阈值与各块结果。"""
    _, rd, _ = _run(data_root, tmp_path)
    rows = _rows(rd)
    meta = rows[0]
    freeze = next(r for r in rows if r["type"] == "freeze")
    names = [meta["target"]["channel"], *config.EOG_PROXY_CHANNELS]
    raw = io_bids.load_channels_uV("001", names, data_root)
    ff = {c: causal_filter.bandpass_oneshot(raw[c]) for c in names}
    n_cal = int(config.QC_CALIBRATION_SEC * config.SFREQ)
    b1, b2 = freeze["b_fp1"], freeze["b_fp2"]
    cal_clean = ff["Fz"][:n_cal] - b1 * ff["FP1"][:n_cal] - b2 * ff["FP2"][:n_cal]
    cal_valid = []
    for j in range(600):
        sl = slice(j * 50, j * 50 + 50)
        cal_valid.append(qc.segment_verdict(
            ff["Fz"][:n_cal][sl], ff["FP1"][:n_cal][sl], ff["FP2"][:n_cal][sl],
            b1, b2, config.QC_L1_P2P_UV, config.QC_L3_EOG_P2P_UV)["valid"])
    fr = tm.calibrate_theta(cal_clean, np.asarray(cal_valid, dtype=bool))
    assert fr["theta_thr_uV2"] == pytest.approx(freeze["theta_thr_uV2"], abs=1e-10)

    stream = tm.ThetaStream(zf=fr["zf"])
    smoother = tm.ValidOnlySmoother()
    decider = tm.HitDecider(fr["theta_thr_uV2"], config.TARGET.direction)
    got_rates = []
    seg_rows = [r for r in rows if r["type"] == "seg"]
    for b in range(6):
        nv = nh = 0
        for j in range(600):
            k = b * 600 + j
            s0, s1 = n_cal + k * 50, n_cal + (k + 1) * 50
            clean = ff["Fz"][s0:s1] - b1 * ff["FP1"][s0:s1] - b2 * ff["FP2"][s0:s1]
            _, p = stream.push_segment(clean)
            v = qc.segment_verdict(ff["Fz"][s0:s1], ff["FP1"][s0:s1], ff["FP2"][s0:s1],
                                   b1, b2, config.QC_L1_P2P_UV, config.QC_L3_EOG_P2P_UV)
            sm = smoother.update(p, v["valid"])
            hit = None if not v["valid"] or sm is None else decider.hit(sm)
            row = seg_rows[k]
            assert row["theta_pow_uV2"] == pytest.approx(p, rel=1e-12)
            assert row["smoothed_uV2"] == (None if sm is None else pytest.approx(sm, rel=1e-12))
            assert row["hit"] == hit
            if v["valid"]:
                nv += 1
                if hit:
                    nh += 1
        got_rates.append(round(nh / nv, 6))
    assert got_rates == [r["hit_rate"] for r in rows if r["type"] == "block_summary"]
    assert got_rates == rows[-1]["block_rates"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_feedback_engine.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'bci_sys.replay.feedback_engine'`）

- [ ] **Step 3: 实现**

创建 `src/bci_sys/replay/feedback_engine.py`：

```python
"""IT4 反馈会话引擎（后台 daemon 线程内的同步实时循环；spec §3–§5、§7）。

preflight → 读 Fz/FP1/FP2 → 三路 1–30Hz 因果带通 → S3 校准
（冻结 b 与 θ 阈值）→ 冻结卡 3s → 6×(120s block + 30s rest) →
feedback.jsonl / run.json。事件帧经 EventBus（有界 256、丢最旧、带 seq）
交给 Web 层；引擎永不阻塞在 UI 上。

A4：jsonl 只含数据侧量，两次运行字节一致；墙钟/线程/队列量只进 run.json。
C5：无效段 smoothed/hit=null、计数不动；数据只标记不删除。
"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import config, gitmeta, io_bids
from . import causal_filter, feedback, qc, theta_metric

SCHEMA_VERSION = "1.0"
FZ = config.TARGET.channel
EOGS = config.EOG_PROXY_CHANNELS
ALL_CH = (FZ, *EOGS)
SEG = config.SEGMENT_SAMPLES
CALIB_SEGS = int(round(config.QC_CALIBRATION_SEC / config.SEGMENT_SEC))
BLOCK_SEGS = int(round(config.FEEDBACK_BLOCK_SEC / config.SEGMENT_SEC))


def _finite_or_none(v):
    """非有限值（NaN/Inf）落盘为 null：裸 NaN/Infinity 不是合法 JSON token。"""
    if v is None:
        return None
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def build_params() -> dict:
    """meta.params 与初始 snapshot.params 共用同一份冻结常量（C2），字段顺序即落盘顺序。"""
    return {
        "sfreq": config.SFREQ,
        "segment_sec": config.SEGMENT_SEC,
        "qc_calibration_sec": config.QC_CALIBRATION_SEC,
        "qc_filter_warmup_sec": config.QC_FILTER_WARMUP_SEC,
        "l1_p2p_uv": config.QC_L1_P2P_UV,
        "l3_eog_p2p_uv": config.QC_L3_EOG_P2P_UV,
        "theta_band_hz": list(config.THETA_BAND_HZ),
        "theta_smooth_segs": config.THETA_SMOOTH_SEGS,
        "theta_quantile": config.THETA_QUANTILE,
        "theta_target_rate": config.THETA_TARGET_RATE,
        "theta_min_calib_segs": config.THETA_MIN_CALIB_SEGS,
        "n_blocks": config.FEEDBACK_N_BLOCKS,
        "block_sec": config.FEEDBACK_BLOCK_SEC,
        "rest_sec": config.FEEDBACK_REST_SEC,
        "freeze_card_sec": config.FEEDBACK_FREEZE_CARD_SEC,
        "session_wall_cap_sec": config.SESSION_WALL_CAP_SEC,
    }


def initial_snapshot() -> dict:
    """WS 首帧 snapshot（live）；引擎每发一帧用最新值整体重建。"""
    return {
        "type": "snapshot",
        "mode": "live",
        "phase": "calibration",
        "block_idx": None,
        "seg_in_phase": 0,
        "k": None,
        "t1": None,
        "b_fp1": None,
        "b_fp2": None,
        "theta_thr": None,
        "params": build_params(),
        "n_valid": 0,
        "n_hit": 0,
        "block_rates": [],
        "last_smoothed": None,
        "last_verdict": None,
        "status": None,
        "reason": None,
    }


class EventBus:
    """引擎→Web 的有界单向通道（线程安全）。

    put_nowait 永不阻塞引擎：满则丢最旧帧并累加 dropped；每帧带单调 seq，
    客户端断档后凭 snapshot 重同步。snapshot 由引擎每帧重建、随帧携带。
    """

    def __init__(self, maxsize: int = 256):
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.seq = 0
        self.snapshot: dict | None = None
        self._lock = threading.Lock()

    def publish(self, msg: dict, snapshot: dict) -> None:
        with self._lock:
            self.seq += 1
            snap = dict(snapshot)
            env = {"seq": self.seq, "msg": msg}
            try:
                self.q.put_nowait(env)
            except queue.Full:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
                self.dropped += 1
                try:
                    self.q.put_nowait(env)
                except queue.Full:
                    pass
            snap["seq"] = self.seq
            snap["dropped"] = self.dropped
            self.snapshot = snap


def _interruptible_wait(seconds: float, abort_event, sleep_fn, step: float = 0.2) -> bool:
    """片长 ≤step 的可中止等待（中止响应 ≤200ms，spec §3）。True=走完，False=被中止。"""
    slept = 0.0
    while slept + 1e-9 < seconds:
        if abort_event is not None and abort_event.is_set():
            return False
        dt = min(step, seconds - slept)
        sleep_fn(dt)
        slept += dt
    return not (abort_event is not None and abort_event.is_set())


def _compute_freeze(judge: qc.QcJudge, cal_buf: dict) -> dict:
    """spec §4.2 步骤 1–6：b 取自 QcJudge 同次冻结；重建 clean、重算 600 段
    有效性（同一 L1/L3 规则）、一次性 4–8Hz 取 zf、P20 阈值。"""
    cal = judge.calib
    b1, b2 = cal["b_fp1"], cal["b_fp2"]
    n = judge.calib_samples
    arrs = {c: np.concatenate(cal_buf[c])[:n] for c in ALL_CH}
    cal_valid = []
    for j in range(CALIB_SEGS):
        sl = slice(j * SEG, j * SEG + SEG)
        cal_valid.append(qc.segment_verdict(
            arrs[FZ][sl], arrs[EOGS[0]][sl], arrs[EOGS[1]][sl],
            b1, b2, config.QC_L1_P2P_UV, config.QC_L3_EOG_P2P_UV)["valid"])
    clean = arrs[FZ] - b1 * arrs[EOGS[0]] - b2 * arrs[EOGS[1]]
    out = theta_metric.calibrate_theta(clean, np.asarray(cal_valid, dtype=bool))
    out["b_fp1"] = b1
    out["b_fp2"] = b2
    out["cal_valid"] = cal_valid
    out["n_fit_samples"] = cal["n_fit_samples"]
    out["fit_window_sec"] = list(cal["fit_window_sec"])
    return out


class _Stop(Exception):
    """引擎内部控制流：终态行已写，跳出 block 循环。"""


def run_feedback(
    *,
    subj: str,
    root=None,
    out_root: str | Path = "outputs",
    run_id: str | None = None,
    abort_event=None,
    bus: EventBus | None = None,
    sleep_fn=time.sleep,
    clock=time.perf_counter,
    chans: dict | None = None,
) -> dict:
    """跑一次 live 反馈会话。返回 {status, reason, run_dir}。

    preflight 失败直接抛异常（不建目录，CLI 映射 rc=2）。
    chans=None 时自己 preflight+读盘；CLI 在 uvicorn 启动前已完成这两步，
    会把读好的三路数组作为 chans 传入（保证服务器起来后不再可能因数据问题 rc=2）。
    """
    if chans is None:
        io_bids.preflight(subj, root)  # 硬停，不写任何文件
        chans = io_bids.load_channels_uV(subj, list(ALL_CH), root)
    n_avail = chans[FZ].size // SEG

    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    print(
        f"[feedback-loop] sub-{subj} | S3 校准 {config.QC_CALIBRATION_SEC:g}s | "
        f"S4 {config.FEEDBACK_N_BLOCKS}×{config.FEEDBACK_BLOCK_SEC:g}s "
        f"rest {config.FEEDBACK_REST_SEC:g}s | θ {config.THETA_BAND_HZ[0]:g}–"
        f"{config.THETA_BAND_HZ[1]:g}Hz W={config.THETA_SMOOTH_SEGS} "
        f"P{config.THETA_QUANTILE*100:g} direction={config.TARGET.direction}"
    )
    run_dir = Path(out_root) / "feedback" / f"sub-{subj}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    wall0 = clock()
    wall_iso = datetime.now().isoformat(timespec="seconds")
    bus = bus or EventBus()
    band = causal_filter.CausalBandpass(ALL_CH)
    judge = qc.QcJudge()
    session = feedback.FeedbackSession()
    run_smoother = theta_metric.ValidOnlySmoother()
    theta_stream = None
    decider = None
    cal_buf: dict | None = {c: [] for c in ALL_CH}

    params = build_params()
    snap = initial_snapshot()
    snap["params"] = params
    last_k, last_t1 = 0, 0.0
    n_cal_rows, n_run_rows = 0, 0
    pause_total = 0.0
    terminal_row: dict | None = None

    vhdr = io_bids.vhdr_path(subj, root).name
    jsonl_path = run_dir / "feedback.jsonl"

    def wall_elapsed() -> float:
        return clock() - wall0

    def wait_data(k: int) -> None:
        """数据钟 1x 节拍；rest/冻结卡的墙钟通过 pause_total 平移 deadline。"""
        target = wall0 + (k + 1) * config.SEGMENT_SEC + pause_total
        dt = target - clock()
        if dt > 0:
            sleep_fn(dt)

    def aborted() -> bool:
        return abort_event is not None and abort_event.is_set()

    with open(jsonl_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({
            "type": "meta",
            "schema_version": SCHEMA_VERSION,
            "subj": subj,
            "data_vhdr": vhdr,
            "git_commit": gitmeta.code_version(),
            "target": {
                "channel": config.TARGET.channel,
                "band_hz": list(config.TARGET.band_hz),
                "direction": config.TARGET.direction,
            },
            "params": params,
        }, ensure_ascii=False) + "\n")
        f.flush()

        try:
            # ---------- S3 校准（k=0…599） ----------
            k = 0
            while k < CALIB_SEGS:
                if k >= n_avail:
                    terminal_row = session.terminate("incomplete_data", k=last_k, t1=last_t1)
                    break
                if aborted():
                    terminal_row = session.terminate("user_abort", k=last_k, t1=last_t1)
                    break
                s, e = k * SEG, k * SEG + SEG
                t1 = round(e / config.SFREQ, 6)
                filtered = band.push_block({c: chans[c][s:e] for c in ALL_CH})
                try:
                    verdict = judge.push(t1, filtered)
                except ValueError as exc:  # EOG 拟合失败：冻结失败硬停
                    terminal_row = session.terminate(
                        "freeze_failed", k=k, t1=t1, error=str(exc))
                    break
                if cal_buf is not None:
                    for c in ALL_CH:
                        cal_buf[c].append(filtered[c])
                last_k, last_t1 = k, t1
                bus.publish({"type": "calib_tick", "k": k, "t1": t1},
                            {**snap, "phase": "calibration", "seg_in_phase": k + 1,
                             "k": k, "t1": t1})
                wait_data(k)
                cap = session.wall_cap_terminate(wall_elapsed(), k=k, t1=t1)
                if cap is not None:
                    terminal_row = cap
                    break
                k += 1

                if verdict.get("calib") is not None:
                    try:
                        fr = _compute_freeze(judge, cal_buf)
                        # note_freeze 二次守门（成熟段 <50 / 阈值非有限）
                        session.note_freeze(fr)
                    except theta_metric.FreezeFailed as exc:
                        terminal_row = session.terminate(
                            "freeze_failed", k=k - 1, t1=t1, error=str(exc))
                        break
                    b1, b2 = fr["b_fp1"], fr["b_fp2"]
                    for j in range(CALIB_SEGS):
                        f.write(json.dumps({
                            "type": "calib_seg",
                            "k": j,
                            "t1": round((j + 1) * SEG / config.SFREQ, 6),
                            "theta_pow_uV2": _finite_or_none(float(fr["cal_powers"][j])),
                            "valid": bool(fr["cal_valid"][j]),
                            "smoothed_uV2": _finite_or_none(fr["cal_smoothed"][j]),
                        }, ensure_ascii=False) + "\n")
                    n_cal_rows = CALIB_SEGS
                    freeze_row = {
                        "type": "freeze",
                        "t_sec": round(config.QC_CALIBRATION_SEC, 6),
                        "b_fp1": b1,
                        "b_fp2": b2,
                        "theta_thr_uV2": fr["theta_thr_uV2"],
                        "smooth_win_seg": config.THETA_SMOOTH_SEGS,
                        "quantile": config.THETA_QUANTILE,
                        "target_rate": config.THETA_TARGET_RATE,
                        "direction": config.TARGET.direction,
                        "n_thr_segs": fr["n_thr_segs"],
                        "calib_hit_rate": fr["calib_hit_rate"],
                        "n_fit_samples": fr["n_fit_samples"],
                        "fit_window_sec": fr["fit_window_sec"],
                    }
                    f.write(json.dumps(freeze_row, ensure_ascii=False) + "\n")
                    f.flush()
                    cal_buf = None  # 释放 ~0.72MB（720KB）校准缓冲
                    theta_stream = theta_metric.ThetaStream(zf=fr["zf"])
                    decider = theta_metric.HitDecider(
                        fr["theta_thr_uV2"], config.TARGET.direction)
                    snap.update({
                        "phase": "freeze_card", "seg_in_phase": 0,
                        "k": k - 1, "t1": config.QC_CALIBRATION_SEC,
                        "b_fp1": b1, "b_fp2": b2,
                        "theta_thr": fr["theta_thr_uV2"],
                    })
                    bus.publish(freeze_row, snap)
                    print(f"[冻结] b_{EOGS[0]}={b1:.4f} b_{EOGS[1]}={b2:.4f} "
                          f"θ_thr={fr['theta_thr_uV2']:.2f}µV² "
                          f"n={fr['n_thr_segs']} hit_rate={fr['calib_hit_rate']:.3f}")
                    if not _interruptible_wait(
                        config.FEEDBACK_FREEZE_CARD_SEC, abort_event, sleep_fn):
                        terminal_row = session.terminate(
                            "user_abort", k=k - 1, t1=config.QC_CALIBRATION_SEC)
                        break
                    pause_total += config.FEEDBACK_FREEZE_CARD_SEC
                    cap = session.wall_cap_terminate(wall_elapsed(), k=k - 1,
                                                     t1=config.QC_CALIBRATION_SEC)
                    if cap is not None:
                        terminal_row = cap
                        break
                    bs_row = session.begin_block(config.QC_CALIBRATION_SEC)
                    f.write(json.dumps(bs_row, ensure_ascii=False) + "\n")
                    f.flush()
                    snap.update({"phase": "block", "block_idx": 0, "seg_in_phase": 0,
                                 "t1": config.QC_CALIBRATION_SEC})
                    bus.publish(bs_row, snap)
                    break

            # ---------- S4：6×(block + rest) ----------
            if terminal_row is None and theta_stream is not None:
                try:
                    for bi in range(config.FEEDBACK_N_BLOCKS):
                        if bi > 0:
                            rs = session.begin_rest(config.FEEDBACK_REST_SEC)
                            f.write(json.dumps(rs, ensure_ascii=False) + "\n")
                            f.flush()
                            snap.update({"phase": "rest", "seg_in_phase": 0})
                            bus.publish(rs, snap)
                            if not _interruptible_wait(
                                config.FEEDBACK_REST_SEC, abort_event, sleep_fn):
                                terminal_row = session.terminate(
                                    "user_abort", k=last_k, t1=last_t1)
                                raise _Stop
                            pause_total += config.FEEDBACK_REST_SEC
                            cap = session.wall_cap_terminate(
                                wall_elapsed(), k=last_k, t1=last_t1)
                            if cap is not None:
                                terminal_row = cap
                                raise _Stop
                            re_ = session.end_rest(config.FEEDBACK_REST_SEC)
                            f.write(json.dumps(re_, ensure_ascii=False) + "\n")
                            f.flush()
                            bus.publish(re_, snap)
                            t1b = round(config.QC_CALIBRATION_SEC + bi * config.FEEDBACK_BLOCK_SEC, 6)
                            bs = session.begin_block(t1b)
                            f.write(json.dumps(bs, ensure_ascii=False) + "\n")
                            f.flush()
                            snap.update({"phase": "block", "block_idx": bi,
                                         "seg_in_phase": 0, "t1": t1b})
                            bus.publish(bs, snap)

                        for j in range(BLOCK_SEGS):
                            k = CALIB_SEGS + bi * BLOCK_SEGS + j
                            if k >= n_avail:
                                terminal_row = session.terminate(
                                    "incomplete_data", k=last_k, t1=last_t1)
                                raise _Stop
                            if aborted():
                                terminal_row = session.terminate(
                                    "user_abort", k=last_k, t1=last_t1)
                                raise _Stop
                            s0, s1 = k * SEG, k * SEG + SEG
                            t1 = round(s1 / config.SFREQ, 6)
                            filtered = band.push_block(
                                {c: chans[c][s0:s1] for c in ALL_CH})
                            v = judge.push(t1, filtered)
                            clean50 = (filtered[FZ] - b1 * filtered[EOGS[0]]
                                       - b2 * filtered[EOGS[1]])
                            _, powv = theta_stream.push_segment(clean50)
                            finite = math.isfinite(powv)
                            sm = run_smoother.update(powv, bool(v["valid"]))
                            hit = None if (not v["valid"] or sm is None) \
                                else bool(decider.hit(sm))
                            session.record_segment(
                                valid=v["valid"], power_finite=finite, hit=hit)
                            last_k, last_t1 = k, t1
                            seg_row = {
                                "type": "seg",
                                "k": k,
                                "t1": t1,
                                "block_idx": bi,
                                "seg_in_block": j,
                                "theta_pow_uV2": _finite_or_none(powv),
                                "smoothed_uV2": _finite_or_none(sm),
                                "hit": hit,
                                "fz_clean_p2p_uv": _finite_or_none(v["fz_clean_p2p"]),
                                "eog_p2p_uv": _finite_or_none(v["eog_p2p"]),
                                "valid": v["valid"],
                                "reasons": list(v["reasons"]),
                            }
                            f.write(json.dumps(seg_row, ensure_ascii=False) + "\n")
                            f.flush()
                            n_run_rows += 1  # 计数紧跟落盘、先于等待（Ctrl-C 落点一致性）
                            snap.update({
                                "phase": "block", "block_idx": bi,
                                "seg_in_phase": j + 1, "k": k, "t1": t1,
                                "n_valid": session.total.n_valid,
                                "n_hit": session.total.n_hit,
                                "block_rates": list(session.block_rates),
                                "last_smoothed": _finite_or_none(sm),
                                "last_verdict": {
                                    "valid": v["valid"],
                                    "reasons": list(v["reasons"]),
                                    "fz_clean_p2p_uv": _finite_or_none(v["fz_clean_p2p"]),
                                    "eog_p2p_uv": _finite_or_none(v["eog_p2p"]),
                                },
                            })
                            bus.publish(seg_row, snap)
                            wait_data(k)
                            cap = session.wall_cap_terminate(wall_elapsed(), k=k, t1=t1)
                            if cap is not None:
                                terminal_row = cap
                                raise _Stop

                        t1_end = round(
                            config.QC_CALIBRATION_SEC + (bi + 1) * config.FEEDBACK_BLOCK_SEC, 6)
                        summ = session.end_block_if_due(t1_end)
                        f.write(json.dumps(summ, ensure_ascii=False) + "\n")
                        f.flush()
                        snap.update({"block_rates": list(session.block_rates)})
                        bus.publish(summ, snap)
                        print(f"[block {bi + 1}/6] hit_rate={summ['hit_rate']:.3f} "
                              f"valid={summ['n_valid']} invalid={summ['n_invalid']}")

                    final_row = session.finish()
                    f.write(json.dumps(final_row, ensure_ascii=False) + "\n")
                    f.flush()
                    snap.update({"phase": "finished", "status": "ok",
                                 "n_valid": session.total.n_valid,
                                 "n_hit": session.total.n_hit,
                                 "block_rates": list(session.block_rates)})
                    bus.publish(final_row, snap)
                except _Stop:
                    pass
        except Exception as exc:  # 未预期异常：engine_error 硬停，traceback 只进终端
            traceback.print_exc()
            terminal_row = session.terminate(
                "engine_error", k=last_k, t1=last_t1, error=type(exc).__name__)

        if terminal_row is not None:
            f.write(json.dumps(terminal_row, ensure_ascii=False) + "\n")
            f.flush()
            snap.update({"phase": "aborted", "status": "aborted",
                         "reason": terminal_row["reason"],
                         "k": terminal_row["k"], "t1": terminal_row["t1"]})
            bus.publish(terminal_row, snap)

    wall = wall_elapsed()
    run_json = {
        "subj": subj,
        "status": session.status,
        "mode": "live",
        "wall_started_iso": wall_iso,
        "wall_elapsed_s": round(wall, 3),
        "data_elapsed_s": round(last_t1, 3),
        "rate_ratio": round(last_t1 / wall, 4) if wall > 0 else 1.0,
        "n_calib_seg": n_cal_rows,
        "n_running_seg": n_run_rows,
        "block_rates": list(session.block_rates),
        "git_commit": gitmeta.code_version(),
    }
    if session.reason is not None:
        run_json["reason"] = session.reason
    (run_dir / "run.json").write_text(
        json.dumps(run_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": session.status, "reason": session.reason, "run_dir": str(run_dir)}


class FeedbackRunner:
    """Web 层持有的引擎线程壳：POST /api/start 建一个，daemon 线程。"""

    def __init__(self, *, subj: str, root=None, out_root="outputs",
                 run_id: str | None = None, chans: dict | None = None):
        self.subj = subj
        self.root = root
        self.chans = chans
        self.out_root = out_root
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = str(
            Path(out_root) / "feedback" / f"sub-{subj}" / self.run_id)
        self.bus = EventBus()
        self.bus.snapshot = initial_snapshot()
        self.abort = threading.Event()
        self.outcome: dict | None = None
        self._sleep_fn = None
        self._clock = None
        self._t: threading.Thread | None = None

    def start(self, *, sleep_fn=None, clock=None) -> None:
        # 只在显式传参时覆盖：CLI 测试可能已在构造后预置快放时钟，
        # webapp 以无参 start() 调用，不能把预置值冲掉
        if sleep_fn is not None:
            self._sleep_fn = sleep_fn
        if clock is not None:
            self._clock = clock
        self._t = threading.Thread(target=self._target, name="feedback-engine", daemon=True)
        self._t.start()

    def _target(self) -> None:
        kwargs = {}
        if self._sleep_fn is not None:
            kwargs["sleep_fn"] = self._sleep_fn
        if self._clock is not None:
            kwargs["clock"] = self._clock
        try:
            self.outcome = run_feedback(
                subj=self.subj, root=self.root, out_root=self.out_root,
                run_id=self.run_id, abort_event=self.abort, bus=self.bus,
                chans=self.chans, **kwargs)
        except Exception:
            traceback.print_exc()
            self.outcome = {"status": "aborted", "reason": "engine_error",
                            "run_dir": self.run_dir}

    def join(self, timeout=None) -> None:
        if self._t is not None:
            self._t.join(timeout)

    @property
    def alive(self) -> bool:
        return self._t is not None and self._t.is_alive()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_feedback_engine.py -q`
Expected: 15 passed（无数据时相关用例 skip）。若 `test_full_run...` 的无效段分布与 `[43,41,57,71,85,108]` 不符，**不要放宽断言**，报告实际值并停下排查（该分布只依赖 QC，spec §2.6 已实测）。

- [ ] **Step 5: 全量回归**

Run: `uv run pytest -q`
Expected: 全绿（120 + 15 = `135 passed`，skip 取决于数据）。

- [ ] **Step 6: Commit**

```bash
git add src/bci_sys/replay/feedback_engine.py tests/test_feedback_engine.py
git commit -m "feat(it4): feedback 引擎——EventBus/冻结协议/6block 循环/jsonl+run.json/A4

Co-Authored-By: Claude <noreply@anthropic.com>"
```
---

### Task 6: webapp（FastAPI：静态/WS 桥接/start/abort/replay 白名单）

spec §6。**只绑 127.0.0.1**（uvicorn 绑定在 T8 CLI；app 本身不绑）。引擎线程与 FastAPI
之间只经 T5 的 EventBus；WS 断开不碰引擎；replay 不起引擎、不写文件。

**Files:**
- Create: `src/bci_sys/replay/webapp.py`
- Test: `tests/test_webapp.py`（新建）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_webapp.py`：

```python
"""IT4 Web 层（spec §6）：静态页、单连接 WS+snapshot、start/abort、429/409、
replay 白名单与帧字段一致、reducer/seek、断 WS 引擎继续。"""
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from bci_sys.replay import feedback_engine as eng
from bci_sys.replay import webapp


class FakeRunner:
    """测试桩：与 FeedbackRunner 同形；脚本线程按开关发帧，abort 后收尾。"""

    def __init__(self, subj="001", root=None, out_root="outputs", run_id=None, chans=None):
        self.subj, self.root, self.out_root = subj, root, out_root
        self.run_id = run_id or "fake"
        self.run_dir = str(Path(out_root) / "feedback" / f"sub-{subj}" / self.run_id)
        self.bus = eng.EventBus()
        self.bus.snapshot = eng.initial_snapshot()
        self.abort = threading.Event()
        self.outcome = None
        self._go = threading.Event()
        self._done = threading.Event()

    def start(self, **kw):
        threading.Thread(target=self._script, daemon=True).start()

    def join(self, timeout=None):
        self._done.wait(timeout)

    @property
    def alive(self):
        return self.outcome is None

    def _pub(self, i):
        row = {"type": "seg", "k": 600 + i, "t1": round(120.2 + 0.2 * i, 6),
               "block_idx": 0, "seg_in_block": i, "theta_pow_uV2": 1.0 + i,
               "smoothed_uV2": 2.0, "hit": i % 2 == 0, "fz_clean_p2p_uv": 5.0,
               "eog_p2p_uv": 6.0, "valid": True, "reasons": []}
        snap = {**eng.initial_snapshot(), "phase": "block", "block_idx": 0,
                "seg_in_phase": i + 1, "k": 600 + i, "t1": row["t1"],
                "n_valid": i + 1, "n_hit": (i + 2) // 2}
        self.bus.publish(row, snap)

    def _script(self):
        self._go.wait(5.0)
        for i in range(30):
            if self.abort.is_set():
                break
            self._pub(i)
            time.sleep(0.005)
        if self.abort.is_set():
            self.outcome = {"status": "aborted", "reason": "user_abort",
                            "run_dir": self.run_dir}
        self._done.set()

    def release(self):
        self._go.set()


@pytest.fixture
def client():
    runner_holder = {}

    def factory(*a, **k):
        runner_holder["r"] = FakeRunner(*a, **k)
        return runner_holder["r"]

    app = webapp.create_app(runner_factory=factory)
    c = TestClient(app)
    yield c, runner_holder


def test_static_index_and_app_js_served_no_external_urls():
    app = webapp.create_app()
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200 and "<html" in r.text.lower()
        rj = c.get("/app.js")
        assert rj.status_code == 200
        body = r.text + rj.text
        assert "http://" not in body and "https://" not in body  # 无外链/CDN


def test_run_state_before_start(client):
    c, _ = client
    r = c.get("/api/run")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "live" and body["status"] is None and body["run_dir"] is None
    # 确认页所需静态信息在引擎启动前即可回显（纯字符串拼接，不读盘）
    assert body["subj"] == "001"
    assert body["data_vhdr"] == "sub-001_task-nback_eeg.vhdr"
    assert body["target"] == {"channel": "Fz", "band_hz": [4.0, 8.0], "direction": "up"}


def test_start_creates_runner_and_second_start_is_409(client):
    c, holder = client
    r = c.post("/api/start")
    assert r.status_code == 200 and r.json()["run_dir"].endswith("sub-001/fake")
    holder["r"].release()
    assert c.post("/api/start").status_code == 409
    holder["r"].join(5)


def test_abort_202_while_running_409_before_start_and_after_terminal(client):
    c, holder = client
    assert c.post("/api/abort").status_code == 409  # 引擎不存在
    c.post("/api/start")
    holder["r"].release()
    assert c.post("/api/abort").status_code == 202
    holder["r"].join(5)
    assert not holder["r"].alive  # join 后脚本线程必须已收尾
    assert c.get("/api/run").json()["status"] == "aborted"
    assert c.post("/api/abort").status_code == 409  # 已终态


def test_ws_snapshot_first_single_connection_429(client):
    c, holder = client
    with c.websocket_connect("/ws") as ws:
        env = ws.receive_json()
        assert env["msg"]["type"] == "snapshot"
        assert env["msg"]["mode"] == "live"
        assert env["msg"]["params"]["n_blocks"] == 6
        with pytest.raises(WebSocketDisconnect) as ei:
            with c.websocket_connect("/ws") as ws2:
                ws2.receive_json()
        assert ei.value.code == 429
        # 第一连接仍可用：start 后能逐帧收到
        c.post("/api/start")
        holder["r"].release()
        f = ws.receive_json()
        assert f["msg"]["type"] == "seg" and f["seq"] == 1
    holder["r"].join(5)


def test_ws_disconnect_does_not_stop_engine_and_reconnect_resyncs(client):
    c, holder = client
    c.post("/api/start")
    runner = holder["r"]
    with c.websocket_connect("/ws") as ws:
        ws.receive_json()  # snapshot
        runner.release()
        ws.receive_json()  # 至少一帧
    # 断开后引擎照跑：等脚本发完 30 帧
    for _ in range(100):
        if runner.bus.seq >= 30:
            break
        time.sleep(0.02)
    assert runner.bus.seq == 30
    with c.websocket_connect("/ws") as ws2:
        env = ws2.receive_json()  # 重连 snapshot 直接追平到最新
        assert env["seq"] == 30
        assert env["msg"]["k"] == 629 and env["msg"]["n_valid"] == 30


def _real_run_jsonl(data_root, tmp_path, rid="fixed"):
    class Clk:
        t = 0.0
        def now(self): return self.t
        def sleep(self, dt): self.t += dt
    clk = Clk()
    eng.run_feedback(subj="001", root=data_root, out_root=tmp_path, run_id=rid,
                     sleep_fn=clk.sleep, clock=clk.now)
    rd = tmp_path / "feedback" / "sub-001" / rid
    return rd / "feedback.jsonl"


def test_replay_payload_matches_jsonl_and_offsets_monotonic(data_root, tmp_path):
    jf = _real_run_jsonl(data_root, tmp_path)
    app = webapp.create_app(mode="replay", replay_file=jf)
    with TestClient(app) as c:
        r = c.get(f"/api/replay?file={jf.name}")
        assert r.status_code == 200
        pl = r.json()
        rows = [json.loads(x) for x in jf.read_text("utf-8").splitlines()]
        assert pl["meta"] == rows[0]
        assert pl["frames"] == rows[1:]  # 数据帧与 jsonl 一字不差
        assert [f["type"] for f in pl["frames"][:600]] == ["calib_seg"] * 600
        offs = pl["offsets_ms"]
        assert all(offs[i + 1] >= offs[i] for i in range(len(offs) - 1))
        # 关键墙钟点：冻结 120s；block1 起点 数据240s+卡3s+rest30s=273s；rest 整 30s
        fi = next(i for i, f in enumerate(pl["frames"]) if f["type"] == "freeze")
        assert offs[fi] == 120_000
        bs1 = next(i for i, f in enumerate(pl["frames"])
                   if f["type"] == "phase" and f["phase"] == "block_start"
                   and f["block_idx"] == 1)
        # 数据 240s + 冻结卡 3s + 一个 rest 30s = 273s 墙钟
        assert offs[bs1] == 273_000
        rs = next(i for i, f in enumerate(pl["frames"])
                  if f["type"] == "phase" and f["phase"] == "rest_start")
        re_ = next(i for i, f in enumerate(pl["frames"])
                   if f["type"] == "phase" and f["phase"] == "rest_end")
        assert offs[re_] - offs[rs] == 30_000
        # 白名单：别的文件名 / 目录穿越一律 400
        assert c.get("/api/replay?file=other.jsonl").status_code == 400
        assert c.get("/api/replay?file=..%2f..%2fother.jsonl").status_code == 400
        # /api/run 在 replay 下回显白名单文件名与数据文件
        run_info = c.get("/api/run").json()
        assert run_info["mode"] == "replay"
        assert run_info["replay_file"] == jf.name
        assert run_info["data_vhdr"] == "sub-001_task-nback_eeg.vhdr"
        # replay 下写操作与 live WS 全拒
        assert c.post("/api/start").status_code == 403
        assert c.post("/api/abort").status_code == 403
        with pytest.raises(WebSocketDisconnect) as ei:
            with c.websocket_connect("/ws") as ws:
                ws.receive_json()
        assert ei.value.code == 1008


def test_reducer_counts_only_valid_finite_segments_like_engine():
    """reducer 中间态计数口径必须与 FeedbackSession.record_segment 一致：
    valid 但功率非有限（jsonl 中 theta_pow_uV2=null）的段不进 n_valid/n_hit。"""
    meta = {"params": eng.build_params()}

    def seg(k, *, valid, powv, hit):
        return {"type": "seg", "k": k, "t1": round(120.2 + 0.2 * k, 6),
                "block_idx": 0, "seg_in_block": k, "theta_pow_uV2": powv,
                "smoothed_uV2": None if powv is None else 2.0, "hit": hit,
                "fz_clean_p2p_uv": 5.0, "eog_p2p_uv": 6.0,
                "valid": valid, "reasons": [] if valid else ["L1"]}

    frames = [
        {"type": "phase", "phase": "block_start", "block_idx": 0, "t1": 123.0},
        seg(0, valid=True, powv=1.0, hit=True),    # 计 n_valid+n_hit
        seg(1, valid=True, powv=None, hit=None),   # 有效但非有限：两计数都不动
        seg(2, valid=False, powv=None, hit=None),  # 无效段：两计数都不动
    ]
    snap = webapp.reduce_snapshot(meta, frames, len(frames))
    assert snap["n_valid"] == 1 and snap["n_hit"] == 1


def test_replay_reducer_finished_snapshot_and_seek(data_root, tmp_path):
    jf = _real_run_jsonl(data_root, tmp_path, rid="fixed2")
    rows = [json.loads(x) for x in jf.read_text("utf-8").splitlines()]
    pl = webapp.load_replay(jf)
    end = webapp.reduce_snapshot(pl["meta"], pl["frames"], len(pl["frames"]))
    assert end["phase"] == "finished" and end["status"] == "ok"
    assert len(end["block_rates"]) == 6
    assert end["n_valid"] == rows[-1]["n_valid"] and end["n_hit"] == rows[-1]["n_hit"]
    # seek：拖到墙钟 274s（block1 起点 273s 之后 1s）→ 落在 block_start 1 边界
    idx = webapp.boundary_before(pl["frames"], pl["offsets_ms"], 274_000)
    snap = webapp.reduce_snapshot(pl["meta"], pl["frames"], idx + 1)
    assert snap["phase"] == "block" and snap["block_idx"] == 1
    # seek 到 10s → freeze 之前的校准态（无更早相位边界 → idx=-1，零帧重建）
    idx0 = webapp.boundary_before(pl["frames"], pl["offsets_ms"], 10_000)
    assert idx0 == -1
    snap0 = webapp.reduce_snapshot(pl["meta"], pl["frames"], idx0 + 1)
    assert snap0["phase"] == "calibration" and snap0["k"] is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_webapp.py -q`
Expected: FAIL（`ModuleNotFoundError: webapp`；静态文件 404）

- [ ] **Step 3: 实现**

创建 `src/bci_sys/replay/webapp.py`：

```python
"""IT4 Web 层（spec §6）：FastAPI app 工厂。

- live：POST /api/start 起引擎 daemon 线程；WS /ws 连接先发 snapshot 再桥接
  EventBus；单连接（第二连接 429）；WS 断开不中止引擎；关页不中止。
- replay：不起引擎、不写文件；GET /api/replay?file= 只允许启动时白名单解析出的
  单一 feedback.jsonl，返回 {meta, frames, offsets_ms, boundaries}，同一前端渲染；
  live 专有 WS/POST 在 replay 模式拒绝（1008/403）。
- 静态资源无外链；只服务 127.0.0.1（绑定在 CLI，见 feedback_cli）。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .. import config, io_bids
from . import feedback_engine as eng

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# 纯墙钟时长（数据钟暂停段），ms 形式供 offsets 计算；单一来源 config，禁止再硬编码
_CARD_MS = int(config.FEEDBACK_FREEZE_CARD_SEC * 1000)
_REST_MS = int(config.FEEDBACK_REST_SEC * 1000)


# ---------------- replay：jsonl → 同协议帧 + 墙钟偏移（只算不落盘） ----------------

def load_replay(path: str | Path) -> dict:
    """读一次 feedback.jsonl，拆 meta/frames，并给每帧算 1x 重放墙钟偏移（ms）。

    数据钟：seg/calib 段 0.2s；冻结卡 3s、每个 rest 30s 为纯墙钟（数据钟暂停）。
    offsets_ms 是播放节奏用的表现层量，绝不回写（A4）。
    """
    rows = [json.loads(line) for line in Path(path).read_text("utf-8").splitlines()]
    if not rows or rows[0].get("type") != "meta":
        raise ValueError(f"非法 feedback.jsonl：首行不是 meta（{path}）")
    meta, frames = rows[0], rows[1:]
    offsets_ms: list[int] = []
    carry = 0
    for f in frames:
        t = f.get("type")
        if t == "calib_seg":
            wall = int((f["k"] + 1) * config.SEGMENT_SEC * 1000)
        elif t == "freeze":
            wall = int(f["t_sec"] * 1000)
        elif t == "phase" and f["phase"] == "block_start":
            wall = int(f["t1"] * 1000) + _CARD_MS + f["block_idx"] * _REST_MS
        elif t == "seg":
            wall = int(f["t1"] * 1000) + _CARD_MS + f["block_idx"] * _REST_MS
        elif t == "block_summary":
            wall = int(f["t1_end"] * 1000) + _CARD_MS + f["block_idx"] * _REST_MS
        elif t == "phase" and f["phase"] == "rest_start":
            wall = carry  # 紧贴上一块 summary 时刻
        elif t == "phase" and f["phase"] == "rest_end":
            wall = carry + _REST_MS
        else:  # final / aborted：停在上一帧时刻
            wall = carry
        offsets_ms.append(wall)
        carry = wall
    boundaries = [
        i for i, f in enumerate(frames)
        if f.get("type") in ("freeze", "final")
        or (f.get("type") == "phase" and f["phase"] in ("block_start", "rest_start"))
    ]
    return {"file": Path(path).name, "meta": meta, "frames": frames,
            "offsets_ms": offsets_ms, "boundaries": boundaries}


def reduce_snapshot(meta: dict, frames: list[dict], upto: int) -> dict:
    """从头回放 frames[:upto] 重建 snapshot（replay 首帧与 seek 后追平共用；
    与引擎 snapshot 同字段语义，replay 模式 status 由 final/aborted 定）。"""
    snap = {
        "type": "snapshot", "mode": "replay", "phase": "calibration",
        "block_idx": None, "seg_in_phase": 0, "k": None, "t1": None,
        "b_fp1": None, "b_fp2": None, "theta_thr": None,
        "params": meta["params"], "n_valid": 0, "n_hit": 0, "block_rates": [],
        "last_smoothed": None, "last_verdict": None,
        "status": None, "reason": None,
    }
    for f in frames[:upto]:
        t = f.get("type")
        if t in ("calib_seg", "calib_tick"):
            snap.update(phase="calibration", seg_in_phase=f["k"] + 1,
                        k=f["k"], t1=f["t1"])
        elif t == "freeze":
            snap.update(phase="freeze_card", seg_in_phase=0,
                        k=None, t1=f["t_sec"], b_fp1=f["b_fp1"],
                        b_fp2=f["b_fp2"], theta_thr=f["theta_thr_uV2"])
        elif t == "phase" and f["phase"] == "block_start":
            snap.update(phase="block", block_idx=f["block_idx"],
                        seg_in_phase=0, t1=f["t1"])
        elif t == "seg":
            snap.update(phase="block", block_idx=f["block_idx"],
                        seg_in_phase=f["seg_in_block"] + 1, k=f["k"], t1=f["t1"],
                        last_smoothed=f["smoothed_uV2"],
                        last_verdict={"valid": f["valid"], "reasons": f["reasons"],
                                      "fz_clean_p2p_uv": f["fz_clean_p2p_uv"],
                                      "eog_p2p_uv": f["eog_p2p_uv"]})
            # 口径与 FeedbackSession.record_segment 一致：valid 且功率有限才进 n_valid
            # （seg 行 theta_pow_uV2 非有限时落盘 null，既不进分母也不进分子）
            if f["valid"] and f["theta_pow_uV2"] is not None:
                snap["n_valid"] += 1
                if f["hit"]:
                    snap["n_hit"] += 1
        elif t == "block_summary":
            snap["block_rates"].append(f["hit_rate"])
        elif t == "phase" and f["phase"] == "rest_start":
            snap.update(phase="rest", seg_in_phase=0)
        elif t == "phase" and f["phase"] == "rest_end":
            snap["phase"] = "rest"
        elif t == "final":
            snap.update(phase="finished", status="ok",
                        n_valid=f["n_valid"], n_hit=f["n_hit"],
                        block_rates=list(f["block_rates"]))
        elif t == "aborted":
            snap.update(phase="aborted", status="aborted", reason=f["reason"],
                        k=f["k"], t1=f["t1"])
    return snap


def boundary_before(frames: list[dict], offsets_ms: list[int], wall_ms: int) -> int:
    """seek：wall_ms 处或之前最近的相位边界帧下标（freeze / block_start /
    rest_start / final；无更早边界时返回 -1，调用方从校准态起播）。"""
    idx = -1
    for i, f in enumerate(frames):
        if offsets_ms[i] > wall_ms:
            break
        if (f.get("type") in ("freeze", "final")
                or (f.get("type") == "phase" and f["phase"]
                    in ("block_start", "rest_start"))):
            idx = i
    return idx


def _default_runner_factory(subj, root, out_root):
    return eng.FeedbackRunner(subj=subj, root=root, out_root=out_root)


def create_app(
    *,
    mode: str = "live",
    subj: str = "001",
    data_root=None,
    out_root: str = "outputs",
    runner_factory=None,
    replay_file: str | Path | None = None,
) -> FastAPI:
    if mode not in ("live", "replay"):
        raise ValueError(f"mode 只能是 live/replay，收到 {mode!r}")
    if mode == "replay" and replay_file is None:
        raise ValueError("replay 模式必须给 replay_file")
    replay_resolved = Path(replay_file).resolve() if replay_file is not None else None
    # replay 在构造时读一次（CLI 已确认文件存在）；live 不预读任何数据
    replay_payload = load_replay(replay_resolved) if replay_resolved is not None else None
    runner_factory = runner_factory or _default_runner_factory

    app = FastAPI(title="bci-sys feedback-loop", docs_url=None, redoc_url=None)
    app.state.mode = mode
    app.state.runner = None

    def _static_run_info():
        # 确认页在引擎不存在时就要能回显 subj/数据文件/靶点（C7：参数先确认后启动）
        info = {
            "mode": mode, "subj": subj,
            "target": {"channel": config.TARGET.channel,
                       "band_hz": list(config.TARGET.band_hz),
                       "direction": config.TARGET.direction},
        }
        if mode == "replay":
            info.update(data_vhdr=replay_payload["meta"].get("data_vhdr"),
                        replay_file=replay_resolved.name)
        else:
            info["data_vhdr"] = io_bids.vhdr_path(subj, data_root).name  # 纯路径拼接，不读盘
        return info

    @app.get("/")
    def index():
        return FileResponse(_WEB_DIR / "index.html", media_type="text/html")

    @app.get("/app.js")
    def app_js():
        return FileResponse(_WEB_DIR / "app.js", media_type="text/javascript")

    @app.get("/api/run")
    def api_run():
        info = _static_run_info()
        r = app.state.runner
        if r is None:
            return {**info, "status": None, "reason": None, "run_dir": None}
        outcome = getattr(r, "outcome", None)
        if outcome is None:
            # JSON 出口统一正斜杠，保证 Windows 下 endswith("sub-001/fake") 类断言可移植；
            # 磁盘回读仍用原始 r.run_dir（下面 Path 拼接），归一化只作用于响应字符串
            return {**info, "status": None, "reason": None,
                    "run_dir": r.run_dir.replace("\\", "/")}
        # 终态后回读 run.json：数据/墙钟时长只从这里进浏览器（不进 WS、不进 jsonl）
        run_json_path = Path(r.run_dir) / "run.json"
        extra = {}
        try:
            extra = json.loads(run_json_path.read_text("utf-8"))
        except OSError:
            pass
        return {**info, "status": outcome["status"],
                "reason": outcome.get("reason"),
                "run_dir": r.run_dir.replace("\\", "/"), **extra}

    if mode == "live":
        @app.post("/api/start")
        def api_start():
            if app.state.runner is not None:
                raise HTTPException(409, "会话已存在（一次性，不可二次启动）")
            runner = runner_factory(subj=subj, root=data_root, out_root=out_root)
            app.state.runner = runner
            runner.start()
            return {"run_dir": runner.run_dir.replace("\\", "/")}

        @app.post("/api/abort")
        def api_abort(response: Response):
            runner = app.state.runner
            if runner is None:
                raise HTTPException(409, "尚无运行中的会话")
            if getattr(runner, "outcome", None) is not None:
                raise HTTPException(409, "会话已终态")
            runner.abort.set()
            response.status_code = 202  # Accepted：已受理，终态由引擎线程异步落定
            return {"accepted": True}

        @app.websocket("/ws")
        async def ws_bridge(ws: WebSocket):
            # 单连接：第二连接 accept 后立刻以 429 关闭（跨 Starlette 版本可移植）
            if getattr(app.state, "ws_taken", False):
                await ws.accept()
                await ws.close(code=429)
                return
            app.state.ws_taken = True
            await ws.accept()
            try:
                runner = app.state.runner
                if runner is not None and runner.bus.snapshot is not None:
                    await ws.send_json({"seq": runner.bus.seq,
                                        "msg": runner.bus.snapshot})
                else:
                    await ws.send_json({"seq": 0, "msg": eng.initial_snapshot()})
                while True:
                    runner = app.state.runner  # start 可能发生在连接之后
                    if runner is None:
                        await asyncio.sleep(0.1)  # 引擎未启动：只等不占队列
                        continue
                    # q.get 是阻塞调用，经线程池桥接（单连接，最多占一个 worker）
                    env = await run_in_threadpool(runner.bus.q.get)
                    await ws.send_json(env)
                    # 终态帧送完后连接挂起等前端关闭；引擎线程已结束，不影响落盘
            except WebSocketDisconnect:
                pass
            finally:
                app.state.ws_taken = False  # 断开不中止引擎，只释放连接名额

    else:  # replay
        @app.get("/api/replay")
        def api_replay(file: str = Query(..., min_length=1)):
            target = (replay_resolved.parent / file).resolve()
            if target != replay_resolved:
                # 白名单 = 启动参数解析出的单一文件；越界/穿越/换名全部 400。
                # 比较在一切读盘之前：不存在/无权限的文件名同样只得到 400，无差异侧信道
                raise HTTPException(400, "file 不在 replay 白名单内")
            # 构造时已读一次，端点直接复用内存快照（不复读、不解析；坏文件在构造期
            # /T8 CLI 预检就失败，端点层不存在 KeyError→500 路径）
            return replay_payload

        @app.post("/api/start")
        async def replay_start():
            raise HTTPException(403, "replay 模式为只读")

        @app.post("/api/abort")
        async def replay_abort():
            raise HTTPException(403, "replay 模式为只读")

        @app.websocket("/ws")
        async def ws_replay_reject(ws: WebSocket):
            await ws.accept()
            await ws.close(code=1008)  # 1008=policy violation：replay 无 live 流

    return app
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_webapp.py -q`
Expected: 9 passed（replay 两个用例需 ds007169，无数据 skip；第 9 个 `test_reducer_counts_only_valid_finite_segments_like_engine` 为质量门新增的口径守卫）。`test_static_*` 在 T7 创建文件前会 404——本任务先建最小静态占位：`src/bci_sys/web/index.html`（内容 `<!doctype html><html><head><meta charset="utf-8"></head><body>IT4</body></html>`）与 `src/bci_sys/web/app.js`（内容 `// IT4 placeholder`），T7 整体替换。

- [ ] **Step 5: 全量回归 + AST 守卫**

Run: `uv run pytest -q`
Expected: `144 passed`（135+9，含本任务新用例与 T7 占位文件；replay 2 用例无数据 skip）。
另跑：`uv run pytest tests/test_boundary.py -q`——webapp.py 位于 replay/，自动守卫（无 offline import、无 filtfilt）。

- [ ] **Step 6: Commit**

```bash
git add src/bci_sys/replay/webapp.py src/bci_sys/web/ tests/test_webapp.py   # pyproject.toml 仅在缺包时
git commit -m "feat(it4): webapp——单连接 WS 桥接/start-abort/replay 白名单与 reducer

Co-Authored-By: Claude <noreply@anthropic.com>"
```

（pyproject.toml 仅当 hatchling 需要显式声明 web/ 包内资源时才改：先构建检查
`uv build` 产物 wheel 里是否含 bci_sys/web/index.html；**实测 hatchling 默认已打入，
pyproject 未改，dist/ 检查后删除**——若产物缺文件，再加
`[tool.hatch.build.targets.wheel] packages = [\"src/bci_sys\"]` 与 include 配置。）

> **执行记录（2026-09-17）**：首提交 1f887b8（4 文件 +515：8 测试 → 143 passed）；
> 质量门裁定后修复提交 99a475d（reducer finite 口径 + 新测试 8→9、/api/replay 复用
> 构造期 replay_payload、_CARD_MS/_REST_MS 常量化、import 移顶、abort 加 alive 断言、
> 删 return pl）→ **144 passed / 0 skipped / boundary 3**。计划嵌入代码已同步为最终形态。
---

### Task 7: 前端（八相位单页 + WS 重连 + replay 播放器）

spec §6.2/6.3。无构建、无外链、无框架：替换 T6 的两个占位文件。前端**不发明相位**，
严格镜像引擎帧；本地行为仅三种：无效帧保持上值、rest 倒计时、冻结卡 3s 兜底动画
（实际切换以后端 block_start 帧为准）。

**Files:**
- Replace: `src/bci_sys/web/index.html`
- Replace: `src/bci_sys/web/app.js`
- Test: `tests/test_web_frontend.py`（新建，静态契约；视觉在 T9 人工截图验收）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_web_frontend.py`：

```python
"""IT4 前端静态契约（视觉行为由 T9 人工回放截图验收，不引入浏览器测试栈）。"""
from pathlib import Path

from fastapi.testclient import TestClient

from bci_sys.replay import webapp

WEB = Path(__file__).resolve().parents[1] / "bci_sys" / "web"


def test_index_contains_all_eight_phase_screens_and_controls():
    html = (WEB / "index.html").read_text("utf-8").lower()
    for sid in ("screen-confirm", "screen-calibration", "screen-freeze",
                "screen-block", "screen-rest", "screen-finished", "screen-aborted",
                "replay-bar"):
        assert f'id="{sid}"' in html
    for token in ("btn-start", "btn-abort", "cal-dots", "block-dots",
                  "hit-bar", "hit-tick", "theta-gauge", "theta-thr",
                  "invalid-banner", "rest-countdown", "rates-body",
                  "replay-seek", "btn-replay-play"):
        assert token in html
    assert 'src="/app.js"' in html
    assert "http://" not in html and "https://" not in html  # 无外链/CDN


def test_app_js_protocol_contract():
    js = (WEB / "app.js").read_text("utf-8")
    for token in ("/ws", "/api/start", "/api/abort", "/api/replay",
                  '"snapshot"', "calib_tick", "calib_seg", "block_start",
                  "rest_start", "block_summary", "final", "aborted",
                  "lastSeq", "boundaries"):
        assert token in js, token
    assert "http://" not in js and "https://" not in js
    # 关页/断流不允许调用 abort
    assert "beforeunload" not in js and "onbeforeunload" not in js


def test_static_served_via_app():
    with TestClient(webapp.create_app()) as c:
        assert c.get("/").headers["content-type"].startswith("text/html")
        assert "javascript" in c.get("/app.js").headers["content-type"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_web_frontend.py -q`
Expected: FAIL（占位文件缺所有 id/契约串）

- [ ] **Step 3: 写 index.html**

完整替换 `src/bci_sys/web/index.html`：

```html
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IT4 feedback-loop</title>
<style>
  :root {
    --bg:#0d1117; --panel:#161b22; --line:#30363d; --fg:#e6edf3; --mut:#8b98a5;
    --green:#2ea043; --green2:#4ec94e; --red:#d62728; --red2:#ff7676;
    --yellow:#e5c07b; --blue:#58a6ff;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family:"Segoe UI","Microsoft YaHei",sans-serif; }
  #topbar { display:flex; align-items:center; gap:14px; padding:10px 18px;
            border-bottom:1px solid var(--line); font-size:13px; }
  #title { font-weight:700; }
  #badge-mode { padding:2px 10px; border-radius:20px; background:#3d3412;
                color:var(--yellow); font-size:12px; font-weight:600;
                display:none; }
  body.replay #badge-mode { display:inline-block; }
  .dot { margin-left:auto; width:9px; height:9px; border-radius:50%;
         background:#6e7681; }
  .dot.live { background:var(--green2); } .dot.off { background:var(--red2); }
  main { max-width:860px; margin:0 auto; padding:22px 18px 60px; }
  .screen.hidden, #replay-bar.hidden { display:none; }
  .panel { background:var(--panel); border:1px solid var(--line);
           border-radius:10px; padding:20px 24px; }
  h2 { font-size:17px; margin:0 0 14px; }
  .kv { font-size:13.5px; line-height:2.0; color:#c9d1d9; } .kv b { color:var(--fg); }
  .note { font-size:12px; color:var(--mut); margin:10px 0 0; }
  button { cursor:pointer; border-radius:8px; font-weight:700; }
  .bigbtn { background:var(--green); color:#fff; border:0; padding:12px 30px;
            font-size:15px; margin-top:16px; }
  .bigbtn:disabled { background:#30363d; color:#6e7681; cursor:default; }
  .abortbtn { background:#21262d; color:var(--red2); border:1px solid #7a4040;
              padding:8px 16px; font-size:12px; }
  .count { font-size:42px; font-weight:700; text-align:center; margin:8px 0;
           font-variant-numeric:tabular-nums; }
  .sub { font-size:12px; color:var(--mut); text-align:center; }
  .cal-dots { display:flex; gap:3px; flex-wrap:wrap; margin:16px 0; }
  .cal-dots i { width:9px; height:9px; border-radius:2px; background:#30363d;
                font-style:normal; }
  .cal-dots i.on { background:#8b98a5; }
  .top { display:flex; justify-content:space-between; align-items:center;
         font-size:13px; color:#9aa7b4; margin-bottom:12px; }
  .dots6 { font-size:17px; letter-spacing:3px; color:#303942; }
  .dots6 b { color:var(--green2); font-weight:700; }
  .chip { padding:2px 11px; border-radius:20px; font-size:12px; font-weight:600; }
  .chip.ok { background:#123d1f; color:var(--green2); }
  .chip.bad { background:#4a1d1d; color:var(--red2); }
  .bar-wrap { background:#21262d; border-radius:8px; height:32px; position:relative;
              margin:14px 0 6px; }
  .bar-fill { height:100%; border-radius:8px; width:0%;
              background:linear-gradient(90deg,var(--green),var(--green2)); }
  .bar-wrap.paused .bar-fill {
    background:repeating-linear-gradient(45deg,#5a3a3a,#5a3a3a 8px,#3a2424 8px,#3a2424 16px); }
  .bar-label { font-size:11.5px; color:var(--mut); text-align:right; }
  .tick { position:absolute; top:0; bottom:0; width:2px; background:#e6edf3;
          opacity:.85; }
  .gauge-track { background:#21262d; height:14px; border-radius:7px;
                 position:relative; margin:16px 0 4px; }
  .gauge-fill { position:absolute; inset:0 auto 0 0; width:0%; border-radius:7px;
                background:var(--blue); }
  .gauge-thr { position:absolute; top:-5px; bottom:-5px; width:2px;
               background:var(--yellow); }
  .gauge-label { font-size:11.5px; color:var(--mut); }
  .banner { margin-top:14px; border-radius:8px; padding:9px 12px; font-size:13.5px;
            text-align:center; background:#3a2424; border:1px solid #7a4040;
            color:#ffb3b3; display:none; }
  body.invalid .banner { display:block; }
  body.invalid .bar-wrap { border:1px solid #7a4040; }
  .rest-title { color:var(--yellow); font-size:20px; font-weight:700;
                text-align:center; }
  table.rates { width:100%; border-collapse:collapse; font-size:13.5px; margin-top:12px; }
  .rates td,.rates th { padding:6px; border-bottom:1px solid #21262d; text-align:center; }
  .rates th { color:var(--mut); font-weight:600; }
  .warn { color:var(--red2); font-size:20px; font-weight:700; }
  .disclaimer { font-size:12px; color:var(--mut); margin-top:14px; }
  #replay-bar { display:flex; align-items:center; gap:12px; margin-bottom:14px;
                background:var(--panel); border:1px solid var(--line);
                border-radius:10px; padding:10px 16px; font-size:13px; }
  #replay-seek { flex:1; }
  .ghostbtn { background:#21262d; color:#c9d1d9; border:1px solid var(--line);
              border-radius:8px; padding:8px 18px; font-size:13px; }
</style>
</head>
<body>
<div id="topbar">
  <span id="title">IT4 反馈训练 · feedback-loop</span>
  <span id="badge-mode">回放模式（只读，不落盘）</span>
  <span id="ws-dot" class="dot"></span>
</div>
<main>
  <div id="replay-bar">
    <button id="btn-replay-play" class="ghostbtn">▶ 播放</button>
    <input id="replay-seek" type="range" min="0" max="0" value="0" step="1">
    <span id="replay-time" class="sub" style="min-width:120px">00:00 / 00:00</span>
  </div>

  <section id="screen-confirm" class="screen panel">
    <h2>启动前参数确认</h2>
    <div class="kv" id="confirm-kv">正在读取冻结参数…</div>
    <p class="note">参数确认后冻结，运行期无任何修改入口；要改只能中止重来。</p>
    <button id="btn-start" class="bigbtn">确认参数并开始 S3 校准</button>
  </section>

  <section id="screen-calibration" class="screen panel hidden">
    <h2>S3 校准中 · 阈值将在 02:00 冻结，冻结后不可更改</h2>
    <div class="count"><span id="cal-time">00:00</span>
      <span style="font-size:18px;color:#8b98a5"> / 02:00</span></div>
    <div id="cal-dots" class="cal-dots" aria-hidden="true"></div>
    <div style="text-align:right"><button id="btn-abort-cal" class="abortbtn">中止会话</button></div>
  </section>

  <section id="screen-freeze" class="screen panel hidden">
    <h2 style="text-align:center">参数已冻结（frozen）</h2>
    <div class="kv" style="text-align:center;font-size:15px;line-height:2.2"
         id="freeze-kv"></div>
    <div class="sub">即将进入 block 1…</div>
    <div style="text-align:right;margin-top:10px">
      <button id="btn-abort-freeze" class="abortbtn">中止会话</button></div>
  </section>

  <section id="screen-block" class="screen panel hidden">
    <div class="top">
      <span><span id="block-dots" class="dots6"></span>
            <b id="block-label" style="margin-left:10px"></b></span>
      <span id="block-timeleft"></span>
      <span id="qc-chip" class="chip ok"></span>
    </div>
    <div id="hit-bar" class="bar-wrap">
      <div id="hit-fill" class="bar-fill"></div>
      <div id="hit-tick" class="tick" style="left:80%"></div>
    </div>
    <div class="bar-label" id="hit-label"></div>
    <div id="theta-gauge" class="gauge-track">
      <div id="theta-fill" class="gauge-fill"></div>
      <div id="theta-thr" class="gauge-thr" style="left:50%"></div>
    </div>
    <div class="gauge-label" id="theta-label">θ 平滑功率 —（黄线＝S3 冻结阈值）</div>
    <div id="invalid-banner" class="banner"></div>
    <div style="text-align:right;margin-top:12px">
      <button id="btn-abort-block" class="abortbtn">中止会话</button></div>
  </section>

  <section id="screen-rest" class="screen panel hidden">
    <div class="rest-title">休息中 · 数据钟暂停</div>
    <div class="count" id="rest-countdown">00:30</div>
    <div class="sub" id="rest-prev"></div>
    <div style="text-align:right;margin-top:14px">
      <button id="btn-abort-rest" class="abortbtn">中止会话</button></div>
  </section>

  <section id="screen-finished" class="screen panel hidden">
    <h2>会话完成</h2>
    <div class="kv" id="finished-kv"></div>
    <table class="rates">
      <tbody id="rates-body">
      <tr id="rates-head"><th>block</th></tr>
      <tr id="rates-rate"><th>达标率</th></tr>
      <tr id="rates-invalid"><th>无效段</th></tr>
      </tbody>
    </table>
    <div class="disclaimer">描述性结果，不是疗效结论。产物：feedback.jsonl / run.json
      （命令行打印的目录路径）。</div>
  </section>

  <section id="screen-aborted" class="screen panel hidden">
    <div class="warn">会话已中止</div>
    <div class="kv" id="aborted-kv" style="margin-top:10px"></div>
    <div class="disclaimer">本次数据标记为中止（aborted），不删除（D3）。</div>
  </section>
</main>
<script src="/app.js"></script>
</body>
</html>
```
- [ ] **Step 4: 写 app.js**

完整替换 `src/bci_sys/web/app.js`：

```javascript
/* IT4 单页：live 走 WS（snapshot+帧、断档自动重连），replay 走
 * GET /api/replay 一次取全量帧，本地按 offsets_ms 以 1x 调度。
 * 两种传输共用 applyEvent——浏览器所见严格镜像引擎，前端不发明相位。 */
"use strict";

const $ = (id) => document.getElementById(id);

const REASON_CN = {
  user_abort: "用户中止（POST /api/abort 或 Ctrl+C）",
  incomplete_data: "可用整段不足 4200（数据不完整）",
  wall_cap: "墙钟超过 25 分钟硬上限",
  freeze_failed: "S3 θ 冻结失败（成熟有效段不足或阈值非有限）",
  engine_error: "引擎内部错误（详见终端 traceback）",
};

function freshState() {
  return {
    mode: "live", started: false, terminal: false,
    phase: null, params: null,
    subj: null, dataVhdr: null, runDir: null,
    calK: -1, blockIdx: null, segInBlock: 0,
    blockValid: 0, blockHit: 0,
    totals: { valid: 0, hit: 0 },
    blockRates: [], blockInvalid: [],
    fr: null, smoothed: null, verdict: null,
    status: null, reason: null, k: null, t1: null,
    runInfo: null,
  };
}
let S = freshState();
let restEndsAt = 0, restTimer = null;

// ---------- WS（live） ----------
let ws = null, lastSeq = 0, wsShouldRun = true, reconnectMs = 300;

function connectWs() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  $("ws-dot").className = "dot live";
  ws.onmessage = (ev) => {
    let env;
    try { env = JSON.parse(ev.data); } catch (e) { scheduleReconnect(); return; }
    if (!env || !env.msg) { scheduleReconnect(); return; }
    if (env.msg.type === "snapshot") {
      applySnapshot(env.msg);
      lastSeq = env.seq;
      render();
      return;
    }
    // seq 断档（含丢帧）：不插值、不猜，重连凭 snapshot 追平
    if (env.seq !== lastSeq + 1) { scheduleReconnect(); return; }
    lastSeq = env.seq;
    applyEvent(env.msg);
    render();
    if (env.msg.type === "final" || env.msg.type === "aborted") {
      wsShouldRun = false; S.terminal = true;
    }
  };
  ws.onclose = () => {
    $("ws-dot").className = "off dot";
    if (wsShouldRun) setTimeout(connectWs, reconnectMs);
  };
  ws.onerror = () => { try { ws.close(); } catch (e) {} };
}
function scheduleReconnect() {
  if (!wsShouldRun) return;
  try { ws.close(); } catch (e) {}
  setTimeout(connectWs, reconnectMs);
}

// ---------- 状态应用（live 帧与 replay 帧共用） ----------
function applySnapshot(m) {
  S.phase = m.phase;
  S.blockIdx = m.block_idx;
  S.segInBlock = m.seg_in_phase || 0;
  S.k = m.k; S.t1 = m.t1;
  S.params = m.params;
  S.fr = (m.b_fp1 == null) ? null : {
    b_fp1: m.b_fp1, b_fp2: m.b_fp2, theta_thr_uV2: m.theta_thr,
  };
  S.totals = { valid: m.n_valid || 0, hit: m.n_hit || 0 };
  S.blockRates = m.block_rates || [];
  S.smoothed = m.last_smoothed;
  S.verdict = m.last_verdict;
  S.status = m.status; S.reason = m.reason;
  if (m.k != null) S.started = true;
}

function applyEvent(f) {
  S.started = true;
  switch (f.type) {
    case "calib_tick":
    case "calib_seg":
      S.phase = "calibration"; S.calK = f.k; S.t1 = f.t1; break;
    case "freeze":
      S.fr = f; S.phase = "freeze_card";
      S.k = null; S.t1 = f.t_sec; break;
    case "phase":
      if (f.phase === "block_start") {
        S.phase = "block"; S.blockIdx = f.block_idx;
        S.segInBlock = 0; S.blockValid = 0; S.blockHit = 0;
        S.verdict = null; document.body.classList.remove("invalid");
        S.t1 = f.t1;
      } else if (f.phase === "rest_start") {
        S.phase = "rest";
        restEndsAt = performance.now() + S.params.rest_sec * 1000;
        startRestTicker();
      } else if (f.phase === "rest_end") {
        /* 相位以随后的 block_start 为准，这里不动 */
      }
      break;
    case "seg":
      S.phase = "block"; S.blockIdx = f.block_idx;
      S.segInBlock = f.seg_in_block + 1; S.k = f.k; S.t1 = f.t1;
      S.verdict = { valid: f.valid, reasons: f.reasons,
                    fz_clean_p2p_uv: f.fz_clean_p2p_uv,
                    eog_p2p_uv: f.eog_p2p_uv };
      if (f.valid) {
        S.blockValid += 1; S.totals.valid += 1;
        S.smoothed = f.smoothed_uV2;
        if (f.hit) { S.blockHit += 1; S.totals.hit += 1; }
        // 无效类统一由 renderBlock 按 verdict 切换，此处不动 body 类
      }
      break;
    case "block_summary":
      S.blockRates[f.block_idx] = f.hit_rate;
      S.blockInvalid[f.block_idx] = f.n_invalid;
      break;
    case "final":
      S.phase = "finished"; S.status = "ok"; S.terminal = true;
      S.blockRates = f.block_rates;
      if (S.mode === "live") fetchRunInfo();
      break;
    case "aborted":
      S.phase = "aborted"; S.status = "aborted";
      S.reason = f.reason; S.k = f.k; S.t1 = f.t1;
      S.terminal = true;
      if (S.mode === "live") fetchRunInfo();
      break;
  }
}

async function fetchRunInfo() {
  try { S.runInfo = await (await fetch("/api/run")).json(); } catch (e) {}
}

// ---------- 渲染 ----------
function show(id) {
  for (const s of document.querySelectorAll(".screen"))
    s.classList.toggle("hidden", s.id !== id);
}
function mmss(x) {
  x = Math.max(0, Math.round(x));
  return `${String(Math.floor(x / 60)).padStart(2, "0")}:` +
         `${String(x % 60).padStart(2, "0")}`;
}
function pct(x) { return (x * 100).toFixed(1) + "%"; }

function render() {
  renderConfirm();
  if (!S.started && !(S.phase === "finished" || S.phase === "aborted")) {
    show("screen-confirm"); return;
  }
  switch (S.phase) {
    case "calibration": renderCalibration(); break;
    case "freeze_card": renderFreeze(); break;
    case "block": renderBlock(); break;
    case "rest": break; // 倒计时由 ticker 自己画，相位切换已 show
    case "finished": renderFinished(); break;
    case "aborted": renderAborted(); break;
  }
  if (S.phase === "rest") show("screen-rest");
}

function renderConfirm() {
  const kv = $("confirm-kv");
  if (!S.params && S.mode === "replay" && RP.meta) S.params = RP.meta.params;
  if (!S.params) return;
  const p = S.params;
  kv.innerHTML =
    `被试：<b>sub-${S.subj ?? "001"}</b>　数据：<b>${S.dataVhdr ?? "—"}</b><br>` +
    `靶点：<b>θ ${p.theta_band_hz[0]}–${p.theta_band_hz[1]} Hz @ Fz，` +
    `direction = ${targetDir()}</b><br>` +
    `S3 校准 <b>${p.qc_calibration_sec}s</b>　S4 <b>${p.n_blocks}×${p.block_sec}s</b>` +
    `　休息 <b>${p.rest_sec}s ×${p.n_blocks - 1}</b>（数据钟暂停）<br>` +
    `θ 平滑 <b>W=${p.theta_smooth_segs} 段（仅有效段）</b>　` +
    `阈值 <b>S3 P${p.theta_quantile * 100}</b>` +
    `（目标达标率 ${p.theta_target_rate}，仅冻结用）<br>` +
    `QC：L1 残差 p2p &gt; ${p.l1_p2p_uv}µV　` +
    `L3 FP1/FP2 任一 p2p &gt; ${p.l3_eog_p2p_uv}µV<br>` +
    `墙钟硬上限 ${p.session_wall_cap_sec}s`;
  $("btn-start").textContent =
    S.mode === "replay" ? "开始回放（1x）" : "确认参数并开始 S3 校准";
}
function targetDir() {
  return S.runInfo?.target?.direction ?? (RP.meta?.target?.direction) ?? "up";
}

let dotsBuilt = false;
function renderCalibration() {
  show("screen-calibration");
  if (!dotsBuilt) {
    const box = $("cal-dots");
    for (let i = 0; i < 600; i++) {
      const d = document.createElement("i");
      box.appendChild(d);
    }
    dotsBuilt = true;
  }
  const kids = $("cal-dots").children;
  for (let i = 0; i < 600; i++)
    kids[i].classList.toggle("on", i <= S.calK);
  const sec = Math.min((S.calK + 1) * (S.params?.segment_sec ?? 0.2),
                       S.params?.qc_calibration_sec ?? 120);
  $("cal-time").textContent = mmss(sec);
}

function renderFreeze() {
  show("screen-freeze");
  const f = S.fr;
  $("freeze-kv").innerHTML =
    `b_FP1 = <b>${f.b_fp1.toFixed(4)}</b>　b_FP2 = <b>${f.b_fp2.toFixed(4)}</b><br>` +
    `θ 阈值 = <b>${f.theta_thr_uV2.toFixed(2)} µV²</b>` +
    `（S3 有效段 P${(f.quantile ?? S.params.theta_quantile) * 100}，` +
    `n=${f.n_thr_segs}，实测校准达标率 ${(f.calib_hit_rate * 100).toFixed(1)}%）<br>` +
    `direction = <b>${f.direction ?? targetDir()}</b>`;
}

function renderBlock() {
  show("screen-block");
  const b = S.blockIdx ?? 0;
  $("block-dots").innerHTML =
    `<b>${"●".repeat(b + 1)}</b>${"○".repeat(Math.max(0, 5 - b))}`;
  $("block-label").textContent = `BLOCK ${b + 1} / ${S.params.n_blocks}`;
  const left = S.params.block_sec - S.segInBlock * S.params.segment_sec;
  $("block-timeleft").textContent = `剩余 ${mmss(left)}`;

  const invalid = !!(S.verdict && !S.verdict.valid);
  document.body.classList.toggle("invalid", invalid);
  const chip = $("qc-chip");
  if (invalid) {
    chip.className = "chip bad";
    chip.textContent = `✗ 无效段 · 反馈暂停（${(S.verdict.reasons || []).join("/") || "?"}）`;
  } else {
    chip.className = "chip ok";
    chip.textContent = "● 有效 · 反馈运行";
  }

  const rate = S.blockValid ? S.blockHit / S.blockValid : 0;
  $("hit-fill").style.width = pct(rate);
  $("hit-bar").classList.toggle("paused", invalid);
  $("hit-label").textContent =
    `本块达标 ${S.blockHit} / ${S.blockValid}（白线＝80% 目标刻线；仅有效段计分）`;

  if (S.smoothed != null && S.fr) {
    const w = Math.max(0, Math.min(100,
      S.smoothed / (2 * S.fr.theta_thr_uV2) * 100));
    $("theta-fill").style.width = w + "%";
  }
  $("theta-label").textContent =
    "θ 平滑功率 " + (S.smoothed == null ? "—（窗未成熟）"
      : S.smoothed.toFixed(1) + " µV²") + "　黄线＝S3 冻结阈值";

  $("invalid-banner").textContent = invalid
    ? `本段无效（${(S.verdict.reasons || []).join("、") || "未知原因"}）` +
      " · 反馈暂停 · 不计入达标率"
    : "";
}

function startRestTicker() {
  clearInterval(restTimer);
  show("screen-rest");
  const prev = S.blockRates.length ? S.blockRates[S.blockRates.length - 1] : null;
  $("rest-prev").textContent = prev == null ? ""
    : `上一块达标率 ${pct(prev)}（仅记录，阈值不变）`;
  restTimer = setInterval(() => {
    const left = (restEndsAt - performance.now()) / 1000;
    $("rest-countdown").textContent = mmss(left);
    if (left <= 0) clearInterval(restTimer);
  }, 200);
  $("rest-countdown").textContent = mmss(S.params.rest_sec);
}

function renderFinished() {
  show("screen-finished");
  const n = S.params.n_blocks;
  $("rates-head").innerHTML = "<th>block</th>" +
    Array.from({ length: n }, (_, i) => `<th>${i + 1}</th>`).join("");
  $("rates-rate").innerHTML = "<th>达标率</th>" +
    Array.from({ length: n }, (_, i) =>
      `<td>${S.blockRates[i] == null ? "—" : pct(S.blockRates[i])}</td>`).join("");
  $("rates-invalid").innerHTML = "<th>无效段</th>" +
    Array.from({ length: n }, (_, i) =>
      `<td>${S.blockInvalid[i] ?? "—"}</td>`).join("");
  let wall, data;
  if (S.runInfo && S.runInfo.wall_elapsed_s != null) {
    wall = S.runInfo.wall_elapsed_s; data = S.runInfo.data_elapsed_s;
    S.runDir = S.runInfo.run_dir;
  } else if (S.mode === "replay") {
    wall = RP.offs[RP.offs.length - 1] / 1000;
    data = S.params.qc_calibration_sec + S.params.n_blocks * S.params.block_sec;
  } else { wall = "—"; data = S.t1 ?? "—"; }
  $("finished-kv").innerHTML =
    `会话完成 · sub-${S.subj ?? "001"} · 数据 <b>${data}s</b>` +
    ` / 墙钟 <b>${typeof wall === "number" ? wall.toFixed(1) : wall}s</b><br>` +
    (S.runDir ? `产物目录：<b>${S.runDir}</b>` : "");
}

function renderAborted() {
  show("screen-aborted");
  $("aborted-kv").innerHTML =
    `原因：<b>${REASON_CN[S.reason] ?? S.reason}</b><br>` +
    `已完成到：block ${S.blockIdx == null ? "—" : S.blockIdx + 1}` +
    `（k=${S.k ?? "—"}，t1=${S.t1 ?? "—"}s）<br>` +
    `run.json status=aborted；本次数据已留痕，标记为中止不删除。`;
}

// ---------- replay 播放器 ----------
const RP = { meta: null, frames: [], offs: [], boundaries: [],
             idx: -1, playing: false, timer: null,
             anchorWall: 0, anchorOff: 0 };

async function bootReplay(file) {
  document.body.classList.add("replay");
  $("replay-bar").classList.remove("hidden");
  wsShouldRun = false;
  $("ws-dot").className = "dot";
  S.mode = "replay";
  const pl = await (await fetch(`/api/replay?file=${encodeURIComponent(file)}`)).json();
  RP.meta = pl.meta; RP.frames = pl.frames; RP.offs = pl.offsets_ms;
  RP.boundaries = pl.boundaries;
  S.params = pl.meta.params;
  const seek = $("replay-seek");
  seek.max = String(RP.offs[RP.offs.length - 1] || 0);
  renderReplayTime();
  render();
}

function replayStep() {
  if (!RP.playing) return;
  if (RP.idx + 1 >= RP.frames.length) {
    RP.playing = false; $("btn-replay-play").textContent = "▶ 播放";
    return;
  }
  const next = RP.idx + 1;
  const wait = RP.offs[next] - RP.anchorOff - (performance.now() - RP.anchorWall);
  RP.timer = setTimeout(() => {
    RP.idx = next;
    applyEvent(RP.frames[RP.idx]);
    render();
    $("replay-seek").value = String(RP.offs[RP.idx]);
    renderReplayTime();
    replayStep();
  }, Math.max(0, wait));
}
function replayPlay() {
  if (RP.playing) { // 暂停
    RP.playing = false; clearTimeout(RP.timer);
    $("btn-replay-play").textContent = "▶ 继续";
    return;
  }
  RP.playing = true;
  if (RP.idx + 1 >= RP.frames.length) { RP.idx = -1; resetReplayState(); }
  if (RP.idx < 0) {  // 首帧 200ms 后才到，先定格校准屏，避免所有 section 同闪
    S.phase = "calibration"; S.calK = -1; render();
  }
  RP.anchorWall = performance.now();
  RP.anchorOff = RP.idx >= 0 ? RP.offs[RP.idx] : 0;
  $("btn-replay-play").textContent = "⏸ 暂停";
  replayStep();
}
function resetReplayState() {
  const subj = S.subj, vhdr = S.dataVhdr, mode = S.mode;
  S = freshState();
  S.mode = mode; S.subj = subj; S.dataVhdr = vhdr; S.params = RP.meta.params;
}
function replaySeekTo(wallMs) {
  // 吸附到该时刻或之前最近相位边界，从头 reduce 重建 snapshot（与后端口径一致）
  RP.playing = false; clearTimeout(RP.timer);
  $("btn-replay-play").textContent = "▶ 继续";
  let bi = -1;
  for (const bIdx of RP.boundaries) {
    if (RP.offs[bIdx] <= wallMs) bi = bIdx; else break;
  }
  resetReplayState();
  for (let i = 0; i <= bi; i++) applyEvent(RP.frames[i]);
  RP.idx = bi;
  RP.anchorOff = bi >= 0 ? RP.offs[bi] : 0;
  render();
  renderReplayTime();
}
function renderReplayTime() {
  const cur = RP.idx >= 0 ? RP.offs[RP.idx] / 1000 : 0;
  const end = (RP.offs[RP.offs.length - 1] || 0) / 1000;
  $("replay-time").textContent = `${mmss(cur)} / ${mmss(end)}`;
}

// ---------- 启动 ----------
async function boot() {
  const info = await (await fetch("/api/run")).json();
  S.subj = info.subj; S.dataVhdr = info.data_vhdr; S.runInfo = info;
  for (const id of ["btn-abort-cal", "btn-abort-freeze", "btn-abort-block",
                    "btn-abort-rest"])
    $(id).addEventListener("click", doAbort);
  $("btn-start").addEventListener("click", onStart);
  $("btn-replay-play").addEventListener("click", replayPlay);
  $("replay-seek").addEventListener("input", (e) =>
    replaySeekTo(Number(e.target.value)));
  if (info.mode === "replay") {
    await bootReplay(info.replay_file);
  } else {
    connectWs();
  }
  render();
}

function onStart() {
  if (S.mode === "replay") { replayPlay(); return; }
  $("btn-start").disabled = true;
  fetch("/api/start", { method: "POST" })
    .then((r) => { if (!r.ok) throw new Error(String(r.status)); })
    .then(() => { S.started = true; render(); })
    .catch((e) => { $("btn-start").disabled = false;
                    alert("启动失败：" + e.message); });
}
async function doAbort() {
  const r = await fetch("/api/abort", { method: "POST" });
  if (!r.ok && r.status !== 202) alert("中止请求失败：" + r.status);
}

boot();
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_web_frontend.py -q`
Expected: 3 passed。

- [ ] **Step 6: 全量回归**

Run: `uv run pytest -q`
Expected: `147 passed`（144+3；数据用例无 ds007169 时 skip）。
手动冒烟（不阻塞提交，T9 系统验收）：`uv run uvicorn bci_sys.replay.webapp:create_app --factory --port 8000` 仅验证工厂可导入；真实 live/replay 走 T8 CLI。

- [ ] **Step 7: Commit**

```bash
git add src/bci_sys/web/index.html src/bci_sys/web/app.js tests/test_web_frontend.py
git commit -m "feat(it4): 八相位单页——WS snapshot 重连/无效暂停/六块仪表/replay 播放器

Co-Authored-By: Claude <noreply@anthropic.com>"
```
---

### Task 8: feedback_cli（入口/退出码/console script）

spec §5.2、§6、§8。顺序铁律：**preflight + 读盘在 uvicorn 启动之前**（rc=2 不可能
发生在服务器起来之后）；Ctrl-C 经 abort 走 user_abort=130（引擎线程要落 aborted 行）；
引擎自然终态后 watcher 置 `server.should_exit` 退出。

**Files:**
- Create: `src/bci_sys/replay/feedback_cli.py`
- Modify: `pyproject.toml`（加 console script）
- Test: `tests/test_feedback_cli.py`（新建；子进程只跑 --help/rc=2，rc=0/1/3/130
  用注入的 FakeServer+FakeClock 在进程内快放）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_feedback_cli.py`：

```python
"""IT4 CLI：argparse/help、rc 映射、preflight 先于服务器、快放全跑 rc=0、
abort=130、截断=3、engine_error=1、replay 只读零写入。"""
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bci_sys import io_bids
from bci_sys.replay import feedback_cli as cli
from bci_sys.replay import feedback_engine as eng


class FakeClock:
    def __init__(self): self.t = 0.0
    def now(self): return self.t
    def sleep(self, dt): self.t += dt


class _Server:
    """替身：不绑端口，用 TestClient 完成 POST /api/start 后等引擎收尾。"""
    def __init__(self, app, port, action=None):
        self.app, self.port, self.action = app, port, action

    def run(self):
        with TestClient(self.app) as c:
            r = c.post("/api/start")
            assert r.status_code == 200, r.text
            runner = self.app.state.runner
            if self.action == "abort":
                runner.abort.set()
            runner.join(30)


class _ReplayServer:
    def __init__(self, app, port, **k): self.app = app
    def run(self):
        with TestClient(self.app) as c:
            info = c.get("/api/run").json()
            r = c.get(f"/api/replay?file={info['replay_file']}")
            assert r.status_code == 200 and r.json()["frames"]


def _live_args(tmp, data_root):
    return ["--subj", "001", "--data-root", str(data_root), "--out-dir", str(tmp)]


def test_help_subprocess():
    r = subprocess.run([sys.executable, "-m", "bci_sys.replay.feedback_cli", "--help"],
                       capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert r.returncode == 0
    assert "feedback-loop" in r.stdout and "--replay" in r.stdout and "--port" in r.stdout


def test_bad_subj_rc2_before_server_and_zero_artifacts(tmp_path, data_root):
    before = sorted(p.name for p in tmp_path.glob("**/*"))
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.replay.feedback_cli",
         "--subj", "999", "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 2
    assert "PREFLIGHT FAIL" in r.stderr
    assert not (tmp_path / "feedback").exists()
    assert sorted(p.name for p in tmp_path.glob("**/*")) == before


def test_rc_map():
    assert cli._rc("ok", None) == 0
    assert cli._rc("aborted", "user_abort") == 130
    for reason in ("incomplete_data", "wall_cap", "freeze_failed"):
        assert cli._rc("aborted", reason) == 3
    assert cli._rc("aborted", "engine_error") == 1
    assert cli._rc(None, None) == 1  # 无终态（被强杀兜底）


def test_live_full_run_rc0(data_root, tmp_path, capsys):
    clk = FakeClock()
    rc = cli.run(_live_args(tmp_path, data_root),
                 _server_cls=_Server, _sleep_fn=clk.sleep, _clock=clk.now)
    assert rc == 0
    out = capsys.readouterr().out
    assert "http://127.0.0.1:" in out  # spec §6：只绑本机且终端打印 URL
    runs = list((tmp_path / "feedback" / "sub-001").iterdir())
    rd = runs[0]
    run = __import__("json").loads((rd / "run.json").read_text("utf-8"))
    assert run["status"] == "ok" and run["n_running_seg"] == 3600


def test_user_abort_rc130(data_root, tmp_path):
    clk = FakeClock()
    rc = cli.run(_live_args(tmp_path, data_root),
                 _server_cls=lambda app, port: _Server(app, port, action="abort"),
                 _sleep_fn=clk.sleep, _clock=clk.now)
    assert rc == 130


def test_incomplete_data_rc3(data_root, tmp_path, monkeypatch):
    orig = io_bids.load_channels_uV
    def truncated(subj, names, root=None):
        return {c: a[:2000 * 50] for c, a in orig(subj, names, root).items()}
    monkeypatch.setattr(cli.io_bids, "load_channels_uV", truncated)
    clk = FakeClock()
    rc = cli.run(_live_args(tmp_path, data_root),
                 _server_cls=_Server, _sleep_fn=clk.sleep, _clock=clk.now)
    assert rc == 3


def test_engine_error_rc1(tmp_path):
    # 不走数据：fault 注入替身 runner，start 即终态 engine_error
    rc = cli.run(["--subj", "001", "--out-dir", str(tmp_path)],
                 _server_cls=_Server, _fault="engine_error")
    assert rc == 1


def test_replay_rc0_and_writes_nothing(data_root, tmp_path):
    # 先快放产出一次 run，再对它的 jsonl 起 replay 服务
    clk = FakeClock()
    out = eng.run_feedback(subj="001", root=data_root, out_root=tmp_path,
                           run_id="rr", sleep_fn=clk.sleep, clock=clk.now)
    jf = Path(out["run_dir"]) / "feedback.jsonl"
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    rc = cli.run(["--replay", str(jf), "--port", "8123"], _server_cls=_ReplayServer)
    assert rc == 0
    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert before == after  # replay 零写入


def test_bad_replay_path_rc2(tmp_path):
    rc = cli.run(["--replay", str(tmp_path / "nope.jsonl")], _server_cls=_ReplayServer)
    assert rc == 2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_feedback_cli.py -q`
Expected: FAIL（`ModuleNotFoundError: feedback_cli`）

- [ ] **Step 3: 实现**

创建 `src/bci_sys/replay/feedback_cli.py`：

```python
"""feedback-loop 命令行入口（spec §5.2/§6/§8）。

live：preflight+读盘（rc=2 在起服务器前）→ 绑 127.0.0.1 的 uvicorn →
点确认后引擎线程起跑；引擎自然终态由 watcher 关服务器；Ctrl-C → abort 后 rc=130。
replay：不起引擎、不写任何文件。
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from .. import config, io_bids
from . import feedback_engine as eng
from . import webapp

_ALL_CH = (config.TARGET.channel, *config.EOG_PROXY_CHANNELS)


def _rc(status: str | None, reason: str | None) -> int:
    """spec §5.2 退出码映射。"""
    if status == "ok":
        return 0
    if status == "aborted":
        if reason == "user_abort":
            return 130
        if reason in ("incomplete_data", "wall_cap", "freeze_failed"):
            return 3
        return 1  # engine_error / 未知
    return 1  # 无终态兜底


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="feedback-loop",
        description="IT4 神经反馈会话（S3 冻结 → 6×120s block；web 可视化）",
    )
    ap.add_argument("--subj", default="001", help="被试编号（默认 001）")
    ap.add_argument("--data-root", default=None, help="ds007169 根目录（默认项目内）")
    ap.add_argument("--out-dir", default="outputs", help="产物根目录")
    ap.add_argument("--port", type=int, default=8000, help="本机端口（默认 8000）")
    ap.add_argument("--replay", default=None, metavar="PATH",
                    help="回放模式：run 目录或 feedback.jsonl 路径（只读，不起引擎）")
    return ap


class _FaultRunner:
    """测试专用替身（_fault 注入）：start 即 engine_error 终态，不读盘不写盘。"""

    def __init__(self, *_a, **_k):
        self.abort = threading.Event()
        self.outcome = None
        self.run_dir = ""

    def start(self, **_k):
        self.outcome = {"status": "aborted", "reason": "engine_error",
                        "run_dir": self.run_dir}

    def join(self, timeout=None):
        pass


def run(argv=None, *, _server_cls=None, _sleep_fn=None, _clock=None,
        _fault=None) -> int:
    args = _build_parser().parse_args(argv)

    # ---------------- replay：无引擎、无写入 ----------------
    if args.replay is not None:
        p = Path(args.replay)
        if p.is_dir():
            p = p / "feedback.jsonl"
        try:
            p = p.resolve(strict=True)
            webapp.load_replay(p)  # 先验证可读/首行 meta
        except (OSError, ValueError) as e:
            print(f"PREFLIGHT FAIL: replay 文件不可用：{e}", file=sys.stderr)
            return 2
        app = webapp.create_app(mode="replay", replay_file=p)
        server = _make_server(app, args.port, _server_cls)
        print(f"[feedback-loop] replay（只读） http://127.0.0.1:{args.port}  {p}")
        server.run()
        return 0  # 回放查看器退出一律 0（没有可中止的会话）

    # ---------------- live：preflight/读盘先于服务器（rc=2） ----------------
    chans = None
    if _fault != "engine_error":  # 测试替身路径不读盘（验证 rc=1 映射用）
        try:
            io_bids.preflight(args.subj, args.data_root)
            chans = io_bids.load_channels_uV(args.subj, list(_ALL_CH), args.data_root)
        except (FileNotFoundError, ValueError, KeyError, OSError) as e:
            print(f"PREFLIGHT FAIL: {e}", file=sys.stderr)
            return 2

    registry: dict = {}

    def runner_factory(*, subj, root, out_root):
        if _fault == "engine_error":
            r = _FaultRunner()
        else:
            r = eng.FeedbackRunner(subj=subj, root=root, out_root=out_root,
                                   chans=chans)
            # 测试注入快放时钟（生产路径两个都是 None → 真墙钟）
            r._sleep_fn, r._clock = _sleep_fn, _clock
        registry["r"] = r
        def watch():
            r.join()
            server.should_exit = True  # 引擎终态后关服务器（守护线程，不碰落盘）
        threading.Thread(target=watch, daemon=True).start()
        return r

    app = webapp.create_app(subj=args.subj, data_root=args.data_root,
                            out_root=args.out_dir, runner_factory=runner_factory)
    server = _make_server(app, args.port, _server_cls)
    print(f"[feedback-loop] http://127.0.0.1:{args.port}  "
          f"（仅本机；浏览器关闭不中止会话）")
    server.run()

    runner = registry.get("r")
    if runner is not None and runner.outcome is None:
        # uvicorn 自己吃掉了 Ctrl-C（serve 正常返回但引擎还活着）：
        # 必须显式 abort 并等它落 aborted 行，不能让 daemon 线程被进程带走
        print("\n[中止] 等待引擎落 aborted 行…", file=sys.stderr)
        runner.abort.set()
        runner.join(timeout=10)
    outcome = getattr(runner, "outcome", None)
    if outcome is None:
        return _rc(None, None)
    if outcome["status"] == "ok":
        print(f"[完成] 产物在 {outcome['run_dir']}")
    else:
        print(f"\n[中止] reason={outcome.get('reason')}；数据已留痕不删除（D3）。",
              file=sys.stderr)
    return _rc(outcome["status"], outcome.get("reason"))


def _make_server(app, port: int, server_cls):
    if server_cls is not None:
        return server_cls(app, port)
    import uvicorn
    config_uv = uvicorn.Config(app, host="127.0.0.1", port=port,
                               log_level="warning")
    return uvicorn.Server(config_uv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows GBK 控制台 µ/中文
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 加 console script**

在 `pyproject.toml` 的 `[project.scripts]` 表（与 replay-monitor / qc-monitor /
theta-probe 同处）追加一行：

```toml
feedback-loop = "bci_sys.replay.feedback_cli:main"
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_feedback_cli.py -q`
Expected: 9 passed（数据用例无 ds007169 时 skip）。
另验：`uv run feedback-loop --help` 实进程输出含 `usage: feedback-loop`。

- [ ] **Step 6: 全量回归**

Run: `uv run pytest -q`
Expected: `156 passed`（147+9；skip 取决于数据）。

- [ ] **Step 7: Commit**

```bash
git add src/bci_sys/replay/feedback_cli.py tests/test_feedback_cli.py pyproject.toml
git commit -m "feat(it4): feedback-loop CLI——preflight 先于服务器/rc 0,2,130,3,1/replay 只读

Co-Authored-By: Claude <noreply@anthropic.com>"
```
---

### Task 9: 全量验收与留证（spec §10）

无新代码。自动化部分由实现者完成；**人工浏览器走查由用户执行**（subagent 不替点、
不伪造截图）。截图是证据不是程序产物（spec §10：程序产物只有 jsonl/run.json/网页）。

**Files:**
- Modify: `evidence/README.md`（新增 IT4 节，沿 IT3 形态）
- Evidence（人工）: `evidence/it4/` 下 PNG 截图

- [ ] **Step 1: 全量回归 + 隔离守卫**

Run:
```bash
uv run pytest -q
uv run pytest tests/test_boundary.py -q
```
Expected: `156 passed`（无 ds007169 时数据用例 skip，数字相应下降且无 failure）；
boundary 全绿（webapp/feedback_engine/theta_metric/feedback 均在 replay/，
无 offline import、无 filtfilt）。

- [ ] **Step 2: 自动验收数值落表（快放、可重复）**

用注入时钟跑两次全量（不占 17 分钟真墙钟；真墙钟体验在 Step 4 人工走查）。新建
`e:/tmp/it4_acceptance.py`（**不进仓库**，沿 IT2/IT3 probe 脚本习惯）：

```python
# -*- coding: utf-8 -*-
"""IT4 验收：双跑 A4 sha256 + 实测数值表（快放时钟；单位 µV，严禁 ×1e6）。"""
import hashlib, json, sys
sys.path.insert(0, r"e:\trae_projects\bci-sys\src")
from bci_sys.replay import feedback_engine as eng

class Clk:
    def __init__(self): self.t = 0.0
    def now(self): return self.t
    def sleep(self, dt): self.t += dt

bs, runs = [], []
for rid, root in (("a", r"e:\tmp\it4_a"), ("b", r"e:\tmp\it4_b")):
    clk = Clk()
    out = eng.run_feedback(subj="001", out_root=root, run_id="fixed",
                           sleep_fn=clk.sleep, clock=clk.now)
    p = root + r"\feedback\sub-001\fixed\feedback.jsonl"
    raw = open(p, "rb").read()
    bs.append(hashlib.sha256(raw).hexdigest())
    runs.append((p, root + r"\feedback\sub-001\fixed\run.json"))
print("status:", out)
print("sha256 equal:", bs[0] == bs[1], bs[0])
rows = [json.loads(x) for x in open(runs[0][0], encoding="utf-8").read().splitlines()]
fz = next(r for r in rows if r["type"] == "freeze")
print("b:", fz["b_fp1"], fz["b_fp2"], "thr:", fz["theta_thr_uV2"],
      "n:", fz["n_thr_segs"], "cal_rate:", fz["calib_hit_rate"])
sums = [r for r in rows if r["type"] == "block_summary"]
print("rates:", [s["hit_rate"] for s in sums])
print("invalid:", [s["n_invalid"] for s in sums])
print("valid:", [s["n_valid"] for s in sums])
print("run.json:", open(runs[0][1], encoding="utf-8").read())
```

Run: `uv run python e:/tmp/it4_acceptance.py`
Expected（把**实际输出**誊入 evidence README；下列为预估值，不预填）：
- `sha256 equal: True`（A4）
- b 约 0.248x / 0.378x；θ_thr 量级参照 ≈16.7 µV²（以实际为准）
- invalid 必须是 `[43, 41, 57, 71, 85, 108]`（QC-only，已锁）
- 六块率在 0.5–0.99；n_valid 合计 + 405 = 3600
- run.json：status=ok、data_elapsed_s=840.0、wall_elapsed_s≈993.0（快放）、
  rate_ratio≈0.846、n_calib_seg=600、n_running_seg=3600

- [ ] **Step 3: 字节级复核（独立命令）**

Run:
```bash
cmp e:/tmp/it4_a/feedback/sub-001/fixed/feedback.jsonl \
    e:/tmp/it4_b/feedback/sub-001/fixed/feedback.jsonl && echo BYTE-IDENTICAL
```
Expected: 打印 `BYTE-IDENTICAL`。

- [ ] **Step 4: 人工 live 走查（用户执行，约 17 分钟真墙钟）**

Run（终端只打印 URL，不自动开浏览器）：
```bash
uv run feedback-loop --subj 001
# 浏览器手动打开 http://127.0.0.1:8000
```

核对单（spec §10）并截图存 `evidence/it4/`：
| # | 时刻 | 截图文件 | 核对点 |
|---|---|---|---|
| 1 | 启动后 | `01-confirm.png` | 参数齐全（up/6×120/rest30/W5/P20/QC 阈值）；未点开始无任何数据推进（C7） |
| 2 | 校准中 | `02-calibration.png` | 灰格推进、mm:ss 计时、无 θ 表/进度条 |
| 3 | t=120s | `03-freeze.png` | b≈0.248/0.378、θ_thr=Step2 实际值、n、cal_rate、direction=up |
| 4 | block 正常段 | `04-block-valid.png` | 六圆点/BLOCK i/剩余时间/QC 绿/条+80% 刻线/θ 表+黄刻线 |
| 5 | 遇无效段 | `05-block-invalid.png` | 斜纹冻结、红横幅带 L1/L3 原因、分子分母不动 |
| 6 | 块间 | `06-rest.png` | 30s 倒计时、数据钟暂停、上一块率仅记录 |
| 7 | 结束 | `07-finished.png` | 六块率/无效段表与 jsonl 一致；840.0/≈993.x；产物路径；非疗效声明 |

另起一次会话在 block 中段点「中止会话」截 `08-aborted.png`（reason=user_abort、
k/t1 定格、run.json status=aborted、文件仍在）。

关页不中止：block 中直接关标签页，终端进程与 jsonl 继续增长；重开页面经 snapshot
追平（截图或终端行数口述记录）。

- [ ] **Step 5: 人工 replay 走查（用户执行）**

```bash
uv run feedback-loop --replay outputs/feedback/sub-001/<Step4的run_id>
```
- 顶栏「回放模式」；确认页数据来自 meta；播放后画面与 live 同构；
- 暂停/继续、拖动吸附相位边界（block 起点/冻结/rest）后画面状态正确；
- 截图 `09-replay.png`；确认 replay 期间 `outputs/` 无新目录产生。

- [ ] **Step 6: 更新 evidence/README.md 并提交**

IT4 节包含：命令、Step2 实测数值表（b/thr/n/cal_rate/六块率/invalid/n_valid）、
A4 sha256 与 cmp 结果、run.json 关键字段、9 张截图引用、replay 说明
（含「描述性结果，不是疗效结论」声明、无硬件/开发者本人为被试两条边界声明）。

截图是否入库沿 IT3 先例：先 `git ls-files evidence | head` 看 IT3 截图是否被跟踪，
照做；`.gitignore` 若忽略 PNG 则 README 仍必须入库。

```bash
git add evidence/README.md evidence/it4/   # 按 IT3 先例调整截图是否 add
git commit -m "evidence(it4): 全量留证——八相位截图/实测数值表/A4 sha256/replay 说明

Co-Authored-By: Claude <noreply@anthropic.com>"
```

- [ ] **Step 7: 完成判据对照（spec §10）**

- [x] 程序可启动、浏览器看得到 S3→六块全程实时反馈与 QC 暂停
- [x] jsonl 可独立重算（T5 test_independent_recompute）并驱动网页回放（T6/T7）
- [x] A4 字节一致（T5 + Step2/3）
- [x] 不下疗效结论；S5/任务/六件套/ABABA 留 IT5–IT7
<!-- APPEND-MARKER -->
