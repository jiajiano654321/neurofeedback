# IT1 + IT2 + IT3 + IT4 验收证据

- IT1/IT2：日期 2026-09-09 ｜ 代码版本（git）：`7d4751c`
- IT3：日期 2026-09-10 ｜ 分支 `feature/it3-qc-monitor`（测试 92 passed，0 skip）
- IT4：日期 2026-09-17/18 ｜ 分支 `feature/it4-feedback-loop`（测试 **164 passed**，0 skip，boundary 3；走查时代码 `6b1ed93`）

本阶段目标是**端到端跑通管线**（公开数据 1x 回放 + 离线 theta + 实时三级 QC），**不评估疗效**。所有有效率/系数均为公开数据上的结果留痕，不是疗效结论。

## IT1：replay-monitor（1x 实时回放）

命令与实测结果：

| 运行 | 命令 | wall(s) | rate_ratio | 段数 | status |
|---|---|---|---|---|---|
| 60 s | `uv run replay-monitor --subj 001 --duration-sec 60` | 60.001 | **1.0033** | 300（=60/0.2） | ok |
| 252 s | `uv run replay-monitor --subj 001 --duration-sec 252` | 252.001 | **1.0008** | 1260 | ok |

- `IT1-replay-速率比.png`：252 s 运行在 t≈71–79 s 的稳态终端窗口（由真实 stdout 捕获忠实渲染，仅把 ANSI 颜色转成像素色，**未改任何文本**）。
  - 每行 `rate 1.00x` 均为**绿色**（绿阈 ≥0.98）；红色 `✗ dropped` 与 `◆ 1-back L1` 标记可见。
  - 60 s 运行 300 行里有 233 行速率落在 1.00–1.02x；前 ~12 行（<2.5 s）是节拍器首次 sleep 前的启动瞬态（1.2–1.98x，仍绿），随后锁定 1.00x。run.json 的 `rate_ratio` 取稳态末值。
- 全量（默认 `--duration-sec 0`）= 239628 点 / 250 Hz = 958.512 s → 4792 个整段 + 28 点（0.112 s）尾巴不发射。

### A4 字节级确定性（人工核对）

两次独立 30 s 运行（不同墙钟、独立时间戳目录）：

```
uv run replay-monitor --subj 001 --duration-sec 30 --out-dir outputs/det1
uv run replay-monitor --subj 001 --duration-sec 30 --out-dir outputs/det2
diff det1/.../replay.jsonl det2/.../replay.jsonl   → IDENTICAL
```

- `replay.jsonl` 均为 164 行，sha256 完全一致：
  `737c296abb7e6c8688192daf8271efa504663a711f647a6dd64e2b1993e2e632`
- summary 行 `"tail_s": 0.0`：限时回放在段边界停止，没有不足一段的残尾（`tail_s` 只表示“本次纳入切分的样本中不足一段的部分”；只有全量回放才是源文件末尾 0.112 s）。
- 墙钟量（`wall_started_iso` / `wall_elapsed_s` / `rate_ratio`）只写在 `run.json`，不进数据侧 `replay.jsonl`，故数据侧跨运行字节一致。

## IT2：theta-probe（离线 Fz theta 负荷检验）

- `IT2-theta-单调曲线.png`：`uv run theta-probe --subj 001` → 打印 `PASS`、退出码 **0**。
  每档 100 个真实（非 tutorial）试次 epoch，中位 Fz theta 功率（µV²）：

  | 1-back | 2-back | 3-back | 4-back |
  |---|---|---|---|
  | 107.12 | 152.09 | 161.80 | 233.06 |

  **1→4-back 严格单调上升，硬门控通过**，方可进入 IT3。

- `IT2-theta-多被试汇总.png`：`uv run theta-probe --all`（纯参考，退出码恒 0）。

  逐被试中位数（µV²）：

  | sub | 结果 | 1 | 2 | 3 | 4 |
  |---|---|---|---|---|---|
  | 001 | PASS | 107.1 | 152.1 | 161.8 | 233.1 |
  | 002 | PASS | 98.5 | 161.7 | 204.3 | 272.6 |
  | 006 | FAIL | 149.6 | 108.7 | 125.3 | 161.9 |
  | 015 | FAIL | 249.9 | 240.9 | 287.9 | 235.8 |
  | 018 | FAIL | 198.3 | 207.3 | 302.1 | 201.8 |
  | 019 | FAIL | 202.5 | 209.2 | 207.6 | 184.3 |
  | 014 | **QC 排除** | — | — | — | — |

  硬门控只对 sub-001；其余数值仅作异质性参考，不影响本阶段验收。`--all` 对每个被试先跑 preflight，单个失败只标记 `QC FAIL，跳过` 并继续，整批恒退出 0。**sub-014 被单位/幅值硬门拦下**：Fz 1–30Hz 带通后 std=1888.4 µV（正常窗 5–300）；同数据集其余 6 人 std 在 41.9–112.8 µV，单位（µV）本身确认无误，故 014 是极端/异常被试（早先在绕过 preflight 的版本里它的 1-back 中位数曾显示为 0 µV²），其取舍留待 IT3+（本阶段不证疗效）。

## IT3：qc-monitor（1x 回放 + 实时 L1/L3 质量判定）

命令（全量，真实 1x 墙钟）：`uv run qc-monitor --subj 001 --out-dir outputs/it3-full`，退出码 0。

| 项 | 实测值 |
|---|---|
| 数据时长 / 段数 | 958.512 s → 4792 整段（末尾 28 点 = 0.112 s 残尾不发射，与 IT1 一致） |
| 校准 / 运行段 | 600 段 CAL（0–120 s）/ 4192 段运行裁决 |
| 冻结点 | t=120.0 s 一次性冻结，拟合窗 [10,120] s、27500 样本 |
| 冻结系数 | **b_FP1=0.2482，b_FP2=0.3781**（设计前实测 0.25/0.38，吻合） |
| 墙钟 / 速率比 | wall 958.682 s，稳态 rate_ratio **1.0002** |
| 有效 / 无效 | 3666 / 526，总体有效率 **87.45%**（summary 记 0.874523） |
| 原因计数 | n_l1=41、n_l3=521、n_l1_and_l3=36（即 L1-only 5、L3-only 485、双中 36） |

计数自洽：41 + 521 − 36 = 526；有效 + 无效 = 4192。无效段**只标记并写 reasons，不删除**（C5）。否决主要由 L3 眼动项驱动（占 526/521 口径中的绝大多数），L1 去眼电残差门仅命中 41 段——与设计阶段"去眼电后残差 p2p 中位约 37µV、L1 约 1%"的量级预估一致。

证据图：

- `IT3-qc-终端.png`：全量运行真实 stdout 的 t=119.4–125.4 s 窗口（ANSI 颜色忠实转像素，未改文本）。可见 t=120.0 的冻结行 `[校准] b_FP1=0.248 b_FP2=0.378 (frozen)`、末三行灰色 `CAL`、之后绿色 `✓`，以及 t=124.0/124.2 s 两个红色 `✗ L3`（EOG p2p 213/394µV，Fz 残差仍仅 28/53µV）；rate 全程绿 1.00x。
- `IT3-qc-终端-L1L3.png`：t=172.6–174.8 s 窗口，展示运行期第一次 `✗ L1 L3`（t=173.6 s：残差 105.4µV 且 EOG 521.5µV，两门同中）与紧随的 `✗ L3`，reasons 并列时标红全记录。
- `IT3-qc-有效率时间轴.png`：事后读 qc.jsonl 生成（不回流因果路径）。上轨逐段裁决色带（灰=校准，其后以绿为主、红/紫为否决段）；下轨运行期 30s 分箱有效率：前段约 0.90，约 450s 后缓慢下行，最低箱约 0.72（~900s，眼动伪迹增多的时段），末箱回到 0.88；标题 overall 87.5% 与 summary 一致。

### A4 字节级确定性（人工核对）

两次独立 300 s 运行（不同墙钟、独立时间戳目录）：

```
uv run qc-monitor --subj 001 --duration-sec 300 --out-dir outputs/det1
uv run qc-monitor --subj 001 --duration-sec 300 --out-dir outputs/det2
cmp qc.jsonl qc.jsonl   → IDENTICAL
```

sha256 两次均为 `1029a0680cdae99bdad026a4dfc4b40e699bbe1dfb34ea841aa21c95bd942470`。qc.jsonl 锁 LF、无墙钟量；墙钟只在 run.json；非有限 p2p 落 null 不产生裸 `NaN` token。

## IT4：feedback-loop（S3 冻结 → 八相位 Web 反馈，1x 回放）

**边界声明**：原型阶段，仅回放公共数据 ds007169 sub-001；**无硬件接入、无实时采集**；
走查被试为开发者本人；下列率值/系数全是公开数据上的**描述性结果，不是疗效结论**。
S5 任务协议、六件套、ABABA 等留 IT5–IT7。

命令（终端只打印 URL，不自动开浏览器）：

```bash
uv run feedback-loop --subj 001                     # live：http://127.0.0.1:8000
uv run feedback-loop --replay outputs/feedback/sub-001/<run_id>   # 只读回放，不落盘
```

### 快放验收数值表（注入时钟双跑，单位 µV/µV²，不 ×1e6）

| 项 | 实测值 |
|---|---|
| EOG 冻结系数 | b_FP1=0.24818284，b_FP2=0.37812954 |
| θ 阈值（S3 P20，n=496 有效段） | 16.770595 µV²；实测校准达标率 0.8004 |
| 六块达标率 | 0.752244 / 0.872987 / 0.799263 / 0.831758 / 0.809709 / 0.882114（即 75.2/87.3/79.9/83.2/81.0/88.2%） |
| 无效段（QC-only，锁死值） | 43 / 41 / 57 / 71 / 85 / 108（合计 405） |
| 有效段 | 557 / 559 / 543 / 529 / 515 / 492（合计 3195；3195+405=3600，每块 600 段） |
| 时长 | 数据 840.0 s（600 校准段 + 3600 运行段）｜快放墙钟 993.0 s（5×30s rest 数据钟暂停、墙钟照走），rate_ratio 0.8459 |
| jsonl 规模 | **1,024,560 字节 / 4225 行**：meta 1 / phase 16 / calib_seg 600 / freeze 1 / seg 3600 / block_summary 6 / final 1 |

### A4 字节级确定性（sha256 链）

同代码版本双跑（不同墙钟、独立目录 run_id=fixed）整文件 sha256 相等。随提交推进的链：

| 代码版本 | meta.git_commit | feedback.jsonl sha256 |
|---|---|---|
| 9726c48（T9 锁定） | 9726c48-dirty | `10e6ac5df73a…b07bcb` |
| 7052b70 | 7052b70-dirty | `5ee8086c…ffaa9d` |
| 35b8e63（当前） | 35b8e63-dirty | `43a1925319411cdb6e27354e293083a6e5d7b636951fbe98cd5cf0d5e10952f0` |

跨提交整文件 sha 必变，**唯一差异是 meta 行的设计字段 `git_commit`**（记录产出版本，
沿 IT1「墙钟/版本只进 meta/run.json、不进数据行」的决策）；去掉 meta 行
（`tail -n +2`）后 4224 个数据行在三个版本间 `cmp` **逐字节相同**。A4 契约本身是
同版本双跑字节一致，当前版本双跑相等已复验。

### 真墙钟无头走查（2026-09-18 04:07–04:26，三会话全过）

方法：playwright-core 驱动系统 **无头 Edge 153** 自动点击/截图，Node spawn 真生产入口
`.venv python -m bci_sys.replay.feedback_cli`（detached/windowsHide），真墙钟、真 HTTP/WS、
真页面；脚本 `e:/tmp/it4_pw/walk.mjs` 不进仓库。**下列截图均为该自动走查产物，非人工摆拍。**

- **session A**（live 全程，run `20260918-120725`，04:07:21→04:23:59，约 16.6 分钟）：
  八相位与关页/重连全部符合；block 中关标签页 15 s，jsonl 625→700 行继续增长
  （关页不中止），重开页面经 snapshot 追平（剩余 01:56→01:39）。结束屏显示
  数据 840 s / **墙钟 993.0 s**、产物目录、六率+无效段表；浏览器收到 final 帧后
  **服务器秒退 rc=0**。run.json：`status=ok, wall_elapsed_s=993.001, data_elapsed_s=840,
  rate_ratio=0.8459, n_calib_seg=600, n_running_seg=3600, block_rates` 齐。
- **session B**（block 中中止，run `20260918-122404`）：block 1（k=700，t1=140.2 s）点
  「中止会话」→ 进程退出码 **130**；run.json `status=aborted, reason=user_abort`
  （wall 143.201 s / data 140.2 s / n_running_seg=101）；feedback.jsonl + run.json 均保留不删除（D3）。
- **session C**（--replay runA，只读）：金色徽标「回放模式（只读，不落盘）」+ 控制条可见；
  seek 吸附三相位（120000 ms→冻结卡、250000 ms→rest、200000 ms→block，02:03/16:33）；
  播放按钮 ⏸ 暂停 ↔ ▶ 继续切换正确；**replay 前后 outputs/ 目录零差异**。

### run.json 字段

`subj / status / mode / wall_started_iso / wall_elapsed_s / data_elapsed_s / rate_ratio /
n_calib_seg / n_running_seg / block_rates / git_commit`，aborted 另加 `reason`。
**写入时机契约**：终态时 run.json 在 publish final/aborted WebSocket 帧**之前**落盘，
浏览器收到终态帧再 GET /api/run 必读到完整文件（终态帧与 HTTP 刷新存在竞争，见计划 T9 记录）。

### 截图（`evidence/it4/`，10 张）

| 文件 | 相位/核对点 |
|---|---|
| `01-confirm.png` | 确认页参数齐（up / 6×120s / rest 30s / W=5 / P20 / QC L1>100µV、L3>200µV / 墙钟上限 1500s）；未开始不推进 |
| `02-calibration.png` | S3 校准灰格推进、mm:ss 计时、无 θ 表/进度条 |
| `03-freeze.png` | t=120s 冻结卡：b=0.2482/0.3781、θ_thr=16.77 µV²、n=496、校准率 80.0%、direction=up |
| `04-block-valid.png` | BLOCK 1/6：六圆点、剩余时间、QC 绿 chip、达标条+80% 刻线、θ 平滑功率+黄阈值刻线 |
| `05-block-invalid.png` | 斜纹冻结、红 chip「✕ 无效段·反馈暂停（L3）」+红横幅、分子分母定格 15/19、不计入达标率 |
| `06-rest.png` | 块间 30s 倒计时、数据钟暂停、上一块率 75.2% 仅记录（阈值不变） |
| `07-finished.png` | 结束屏：数据 840s/墙钟 993.0s、产物目录、六率与无效段表（同锁死值）、非疗效声明 |
| `08-aborted.png` | 中止屏：user_abort、定格 k=700/t1=140.2s、aborted 不删除（D3） |
| `09-replay.png` | 回放模式：金徽标（只读不落盘）+ 控制条 + seek 吸附 BLOCK 相位、与 live 同构渲染 |
| `10-reconnect.png` | block 中关页后重连，snapshot 追平，会话继续（剩余时间已推进） |

## 经验性发现（实测，非文档反推）

- **第一个真实（`istutorial≠true`）n-back 试次 onset = 246.016 s**；前 ~246 s 为教程/指导段。
- 回放终端的 `◆ N-back` 标记对“带 nback_level 的刺激”显示（含教程级刺激），最早出现在 **t=77.6 s（seg 387）**；而 theta 分析的 `real_trial_events` 严格只取非教程试次。两者用途不同、各自正确。因此 60 s 窗口只见 `✗ dropped` 与教程起始 marker，看不到 `◆`；截图改用 252 s 稳态窗口。
- **IT3 全量有效率 87.45%，与设计阶段预估的约 88% 吻合**；时间轴上有效率非平稳：前 ~450 s 各 30 s 箱约 0.90，~900 s 附近降到约 0.72（眼动伪迹在该时段聚集），末箱回到 0.88。否决结构高度偏向 L3（521 次命中）vs L1（41 次），说明冻结 EOG 回归后 Fz 残差普遍远低于 100µV，门主要在抓眼动/大伪迹——这是单被试公开数据的描述性结果，不外推。
