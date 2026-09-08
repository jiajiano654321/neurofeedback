# IT1 + IT2 设计：replay_monitor 与 theta_probe

> 日期：2026-09-08
> 范围：迭代计划第 1 周的两个可运行程序（IT1 `replay_monitor`、IT2 `theta_probe`）
> 依据：[BACKGROUND.md](../../../BACKGROUND.md)（七条硬约束 C1–C7、留痕铁律 D1–D3、成功判据 A1–A5）
> 状态：已与需求方逐条确认。所有"实测"数值均于 2026-09-08 用 sub-001 等真实数据核验。

---

## 1. 范围与目标

本期只做两件事，各一个能启动、能看到输出的程序：

| 程序 | 一句话 | 成果物 |
|---|---|---|
| IT1 `replay_monitor` | 按 1x 节奏把 ds007169 的 200 ms 段吐到终端，同时落盘 JSONL | 终端截图（含速率比）+ `replay.jsonl` |
| IT2 `theta_probe` | 离线算 Fz theta，按 n-back 负荷分组出折线图 | PNG 折线图 |

**IT2 是完全独立的离线程序**（决策记录）：直接读 BIDS 文件、按试次切 epoch、用 multitaper 算 theta 功率。
它不消费 IT1 的回放流。理由：IT2 的唯一职责是当"真值检验"——验证 theta 计算与通道映射正确；
若挂在流式路径上，门控失败时分不清是 theta 算法错还是流机制错。这也符合 **C3 物理隔离**：
离线可用零相位滤波，实时代码与离线代码不共用任何处理逻辑。

**不做（YAGNI）**：IT1 不做 QC（IT3）、不做反馈（IT4）；IT2 不做伪迹剔除（QC 是 IT3 的事）、
不做 50 Hz 陷波（theta 4–8 Hz 离工频远）、不消费回放流。

---

## 2. 已核实的关键数据事实（2026-09-08）

以下数值均用真实数据实测，是设计的硬事实：

1. **格式**：BrainVision（.vhdr/.vmrk/.eeg），float32，24 导 = 19 EEG + 1 ECG + 4 MISC，250 Hz。
   每人记录 **239628 点 = 958.512 s**（~16 分钟；eeg.json 写的 958.508 s 是四舍五入值，
   **不可反推点数**，差 1 个点）。mne 原生可读。
2. **单位（关键陷阱）**：数据集由 pybv 0.7.6 写出。mne 读出的 Fz 带 ~1.44×10⁶ 的直流偏移；
   1–30 Hz 带通后 **std：sub-001=42.3、sub-002=47.4、sub-006=112.8 µV**，全部落在正常 EEG
   范围（10–120 µV）。**结论：mne 输出的数值直接就是 µV，严禁再乘 1e6。**
   若错乘：theta 功率大 10¹² 倍但曲线形状不变、单调性照样 PASS；IT3 的 L1（100 µV 峰峰值）
   会全部超标。属于 C1 同类"不报错、不崩溃、数据看着正常"的错。
3. **FIR 滤波器长度**：1–30 Hz 零相位 FIR = **825 点 = 3.30 s**，是 epoch（375 点/1.5 s）的 2.2 倍。
   **处理顺序必须固定为：读全程 → 全程滤波 → 再切 epoch。** 先切后滤会因滤波器长于 epoch
   而大量补零，theta 估计全废且不报错。
4. **试次间隔**：非练习试次共 400 个（1/2/3/4-back 各 100），间隔 median 1.701 s、min 1.700 s
   （4 个试次正好卡 1.700 s）。窗口取刺激 onset 起 **[0, 1.5 s]**：最小间隙下也留 0.2 s，不重叠。
5. **事件字段**：`events.tsv` 含 `onset / trial_type / nback_level / istutorial / outcome /
   dropped_samples` 等；练习段以 `istutorial=true` 标记，须排除；`dropped_samples` 事件单独记录。
6. **通道名以 channels.tsv 为准**：EOG 代理通道是 **FP1/FP2**（大写），靶点单通道 **Fz**。
   无 FCz、无专用 EOG。

---

## 3. 技术栈与项目结构

**环境**：Python 3.11，uv 管理（`pyproject.toml`）。运行期依赖 `mne`、`numpy`、`scipy`、
`matplotlib`、`pandas`；测试 `pytest`。初始化 git 仓库（D2 要求记录代码版本）。

**数据根目录**：数据已复制到项目内 `ds007169/`（955 MB），默认数据根指向它；
可用环境变量 `BCI_DATA_ROOT` 覆盖。`ds007169/`、`outputs/`、`.venv/`、`__pycache__/` 进 `.gitignore`；
`evidence/` 提交存档。原始 BIDS 文件只读不复制、不修改（D1）。

**结构（方案 A：单包双入口，处理逻辑物理隔离）**：

```
bci-sys/
  pyproject.toml
  .gitignore
  src/bci_sys/
    __init__.py
    config.py                 # 唯一常量出处
    io_bids.py                # 纯 I/O：读 BrainVision / events、通道查找、preflight
    replay/                   # ← 实时路径（IT1；IT3/IT4 继续在此生长）
      __init__.py
      segments.py             #   因果分段器（只向前切，不碰未来样本）
      pace.py                 #   1x 节拍器
      monitor.py              #   终端显示 + JSONL/run.json 落盘
      cli.py                  #   replay-monitor 入口
    offline/                  # ← 离线路径（IT2；C3 隔离，可用零相位滤波）
      __init__.py
      theta.py                #   全程滤波 → epoch → multitaper theta 功率
      probe.py                #   负荷分组、门控判定、出图出表
      cli.py                  #   theta-probe 入口
  tests/
  outputs/                    # gitignore
  evidence/                   # 提交：截图/PNG/GIF
  docs/superpowers/specs/
```

**C3 隔离的强制执行**：`replay/` 与 `offline/` 互不 import；共享的只有 `config.py`（常量）和
`io_bids.py`（纯文件读取，I/O 不存在"用未来数据"风险）。用一条 AST 扫描测试断言这个边界
（见 §7）。通道清单只在 `config.py` 一处定义，配合 IT2 门控抓通道映射错误。

**两个命令行入口**：
- `replay-monitor --subj 001 [--duration-sec 0] [--data-root ...]`
- `theta-probe --subj 001 | --all [--data-root ...]`

`--duration-sec` 默认 **0 = 全量回放**；显式传 N 则在段边界截断到 N 秒（开发/截图用）。

---

## 4. 共享层

### 4.1 `config.py`（唯一常量出处）

- `DATA_ROOT`：默认项目内 `ds007169/`，可被 `BCI_DATA_ROOT` 覆盖。
- `SFREQ = 250.0`；`SEGMENT_SEC = 0.2` → `SEGMENT_SAMPLES = 50`。
- `RECOMMENDED_SUBS = ["001","002","006","014","015","018","019"]`；`EXTREME_SUB = "012"`。
- **靶点定义（C1）**：`TARGET = Target(channel="Fz", band_hz=(4.0, 8.0), direction="up")`。
  `direction` 是**必填参数、无默认值**，构造时不写就抛错。本期不用于反馈，但从第一行代码起
  就把模式立对——任何涉及靶点的地方显式 `direction="up"`。
- `EOG_PROXY_CHANNELS = ("FP1", "FP2")`（IT3 才用，先登记，名字大小写以此为准）。
- theta 谱参数：`THETA_BAND_HZ = (4.0, 8.0)`、`EPOCH_SEC = 1.5`、`MULTITAPER_NW = 2.0`。
- 速率颜色阈值：`RATE_GREEN = 0.98`、`RATE_YELLOW = 0.95`。

### 4.2 `io_bids.py`（纯 I/O，无信号处理逻辑）

- `load_fz_uV(subj) -> np.ndarray`：mne 读 .vhdr，返回 Fz 单通道全程数据，**单位 µV**。
  docstring 必须写明："本数据集由 pybv 写出，mne 输出数值经实测已是 µV（1–30 Hz 带通后
  std≈42 µV），**不要乘 1e6**；信号带 ~1.44e6 直流偏移属正常，由后续滤波去除。"
- `load_events(subj) -> pandas.DataFrame`：读 `*_events.tsv`。
- `channel_index(raw, name) -> int`：按名字**显式查找**通道下标，找不到直接抛错
  （通道映射安全靠它，不靠位置假设）。
- `preflight(subj) -> None`：S1 质检，任何一项不过 = **硬停**（C7），非零退出、明确报错、
  **不写任何文件**。检查项：
  1. .vhdr/.vmrk/.eeg 与 events.tsv 存在且可读；
  2. 采样率 == 250 Hz；
  3. Fz 通道存在；EEG 通道数 == 19；
  4. events.tsv 含 `nback_level` 列；
  5. **单位硬检查**：全程 1–30 Hz 带通后 Fz std 必须 ∈ [5, 300] µV，否则报
     "极可能是单位换算错误（检查是否误乘 1e6）"。这是唯一能挡住单位错误的闸门，
     在启动时跑、不在实时环路内。

---

## 5. IT1 `replay_monitor`（实时路径）

### 5.1 流程

S1 preflight → 建运行目录 `outputs/replay/sub-001/<墙钟时间戳>/` → 写 meta 行 →
分段回放循环（5 行/秒）→ 写确定性 summary 行 + `run.json`。

### 5.2 JSONL 结构（`replay.jsonl`，每行一条，UTF-8）

1. **首行 meta**：`{type:"meta", schema_version, code_version, subj, data_file, sfreq,
   segment_sec, fz_channel, n_segments_planned}`。
   `code_version` = git hash + dirty 标记（D2）。**不含墙钟时间**。
2. **每段一行**：`{type:"seg", seg, t0, t1, fz_uV:[50 个 float]}`。`t0/t1` 是**数据时间**（秒）。
3. **事件行**（onset 落在本段区间 **[t0, t1)** 时插入，紧跟该段段行之后；左闭右开避免边界
   重复计数）：`{type:"event", t, trial_type, nback_level, istutorial, kind}`；
   `dropped_samples` 也作为事件行（`kind:"dropped"`）。
4. **末行 summary**：`{type:"summary", n_segments, data_elapsed_s, tail_s, n_events,
   n_dropped}`。**只含数据侧量，不含任何墙钟量。**

### 5.3 确定性（A4）

数据侧产物完全确定：`replay.jsonl` 里零墙钟信息。同被试、同 `--duration-sec` 两次运行，
两个 `replay.jsonl` **字节级一致**（运行目录名不同无妨）。墙钟量只写进独立的 `run.json`：
`{wall_elapsed_s, rate_ratio, wall_started_iso, status}`，允许每次不同。

### 5.4 节拍

循环开始记 `wall0 = perf_counter()`；第 k 段处理完睡到 `wall0 + (k+1)*0.2 s`。
deadline 锚定起点、自校正，不累加漂移。速率比 = 数据已过秒数 / 墙钟已过秒数。

### 5.5 终端显示（每段一行）

```
00412 │ t= 82.4s │ Fz p2p 12.4µV │ rate 1.00x │ ◆ 1-back L1
```

段号 │ 数据时间 │ Fz **段内峰峰值 p2p**（不显示瞬时绝对值——原始信号带直流偏移；
p2p 不受直流影响，且正是 IT3 L1 要用的量）│ 滚动速率比 │ 事件标记。

**速率比颜色**（ANSI，终端支持时）：**绿** ≥0.98（跟得上 1x）／**黄** 0.95–0.98（轻微落后）／
**红** <0.95（跟不上节拍——IT1 要抓的故障）。`dropped_samples` 事件红色 `✗`。速率比是
截图验收的核心证据。

### 5.6 边界处理

239628 点（958.512 s）→ **4792 个整段**（958.4 s）+ 余 28 点 = **0.112 s** 尾巴；
尾巴不足一段不发段，记入 summary 的 `tail_s`（从实际点数算，不用 json 时长反推）。
`--duration-sec N`（N>0）在段边界截断。

### 5.7 中止处理

Ctrl+C：`run.json` 写 `status:"aborted"`，已写内容保留不删（D3 精神），不写 summary 行。
正常结束 `status:"ok"`。

---

## 6. IT2 `theta_probe`（离线路径，`offline/`）

### 6.1 流程

preflight → `load_fz_uV` 读 Fz 全程 → **全程零相位滤波** → 切 epoch → 逐 epoch multitaper
算 theta 功率 → 按 n-back 负荷分组取中位数 → 门控判定 → 出图出表。

### 6.2 硬约束：处理顺序

**读全程 → 全程 1–30 Hz 零相位 FIR 带通（filtfilt，离线允许）→ 再切 epoch → 谱估计。**
禁止先切 epoch 再逐段滤波（滤波器 825 点 > epoch 375 点，先切后滤补零废结果，且不报错）。

### 6.3 Epoch 与谱估计

- 只取非练习试次：`istutorial != true` 且 `nback_level ∈ {1,2,3,4}`（每级 100，共 400）。
- 窗口 = 刺激 onset 起 **[0, 1.5 s]**（375 点），不重叠（实测最小间隔 1.700 s）。
- **本期不做伪迹剔除**：每级 100 trial 取中位数，眨眼等宽带噪声不决定单调性。若门控过不了，
  先查 theta 算法与通道映射，**不许靠加剔除"修"数据**。
- 每 epoch multitaper 谱（NW=2），theta 功率 = 4–8 Hz 带内积分，单位 µV²。

### 6.4 分组与门控

- 每级取**中位数**（抗离群）得 4 个点。
- **硬门控（仅默认 `--subj` 单被试，验收用 sub-001）**：
  `median(L1) < median(L2) < median(L3) < median(L4)` 严格成立 → 打印 `PASS` + 四个值，
  退出码 0；否则打印 `FAIL` + 实际四个值，**退出码 1，不许进 IT3**。
- `--all`：跑 7 个推荐被试（001/002/006/014/015/018/019），出多线汇总图 + 逐人 PASS/FAIL
  表，纯参考、不作硬门控（退出码恒 0）。sub-012 留作以后极端测试，本期不跑。

### 6.5 产物

单被试：`outputs/theta/<subj>/theta_by_load.csv`（列：load、n_epochs、median_theta_uV2）+
`theta_by_load.png`（x = 1–4 back，y = theta 中位数，四点连线）。
`--all`：`outputs/theta/all/summary.csv`（逐人四级中位数 + PASS/FAIL）+ `summary.png`
（7 人多线汇总）。验收时关键 PNG 存入 `evidence/`。

---

## 7. 错误处理与测试

### 7.1 错误处理（C7 硬拦风格）

- preflight 任一不过 → 非零退出、明确报错、不写任何文件。
- 运行中 Ctrl+C → `run.json` 写 `aborted`，已写内容保留（D3）。
- 通道按名查找失败、单位 std 越界 → 启动即硬停。

### 7.2 测试（pytest）

1. **C3 边界测试**：AST 扫描断言 `replay/` 不 import `offline.*`、`offline/` 不 import `replay.*`。
2. **分段器**：sub-001 全量 = 4792 段、每段 50 点、段间无重叠无空隙、余 28 点（0.112 s）不发段；
   `--duration-sec` 在段边界截断。
3. **确定性（A4）**：注入 noop sleep（假时钟，不走真睡眠）跑两次限时回放，两个 `replay.jsonl`
   字节级一致。
4. **theta 合成信号**：6 Hz 正弦的 theta 功率 ≫ 20 Hz 正弦；幅值-功率关系符合理论 A²/2（容差内）。
5. **门控逻辑**：对合成的严格单调序列判 PASS、对非单调序列判 FAIL。
6. **通道映射**：`channel_index` 按名查找 Fz 成功；错名抛错。
7. **集成测试（真数据）**：preflight 通过 sub-001；**theta_probe 门控在 sub-001 上 PASS**——
   这条测试本身就是 IT2 的验收判据。

---

## 8. 入口与验收（"定入口"）

**IT1 `replay-monitor --subj 001`**
- 屏幕：5 行/秒滚动段信息，速率比带颜色（正常绿色）。
- 跑完：`outputs/replay/sub-001/<ts>/replay.jsonl` + `run.json`。
- 验收：限时跑一段（如 `--duration-sec 60`），截终端图（含速率比）存
  `evidence/IT1-replay-速率比.png`；核对 JSONL 段数 = 时长/0.2。

**IT2 `theta-probe --subj 001`**
- 屏幕：打印 PASS/FAIL + 四级 theta 中位数。
- 产物：`outputs/theta/001/theta_by_load.csv` + `.png`。
- 验收：退出码 0（PASS）且 PNG 四点单调上升，存 `evidence/IT2-theta-单调曲线.png`。
  **算不出单调上升 = theta 计算或通道映射有错，不许进 IT3。**

每期收工必须截图存 `evidence/`，不截图 = 本期未验收。

---

## 9. 决策记录（为什么这么定）

| 决策 | 选择 | 理由 |
|---|---|---|
| IT2 theta 怎么算 | 完全离线独立，不消费回放流 | 门控只检验 theta/通道正确性；隔离流式机制的干扰；符合 C3 |
| 回放时长 | 默认全量 1x，`--duration-sec 0`=全量，可限时 | 速率锁 1x 是 IT1 要验的，不提供加速；限时方便开发/截图 |
| JSONL 内容 | 元数据 + Fz 原始 50 点 + 事件行 | 可事后重算（A4 精神）；IT3 再加 FP1/FP2；全程约 2.5 MB |
| 确定性 | 数据侧 `replay.jsonl` 字节一致，墙钟量隔离到 `run.json` | A4 可重算；墙钟本就每次不同 |
| IT2 被试 | sub-001 硬门控 + `--all` 7 人参考 | 门控验代码正确性，一人足够；多人是顺手的信心检查，不作硬门控 |
| 代码组织 | 单包双入口，replay/offline 物理隔离，共享 config+io | C3 靠目录边界 + AST 测试强制；共享面压到无风险的 I/O 与常量 |
| 单位 | mne 输出直接当 µV，禁止乘 1e6 + preflight std 硬检查 | 实测 std 42–113 µV 正常；错乘不报错，靠闸门挡 |
| 滤波顺序 | 全程滤波后切 epoch | FIR 825 点 > epoch 375 点，先切后滤补零废结果 |
| epoch 窗口 | [0, 1.5 s]，不做伪迹剔除 | 实测不重叠；1.5 s 给 theta 带 6 个频点；剔除是 IT3 的事 |
