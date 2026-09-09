# IT3 `qc_monitor` 设计：实时三级质量判定（L1/L2/L3）

日期：2026-09-09
前置：IT1 `replay_monitor`、IT2 `theta_probe` 已完成并验收（sub-001 theta 1→4-back 硬门控通过）。
依据：`BACKGROUND.md` 第 4 节 C2/C3/C4/C5/C7、第 5 节 D 系、第 9 节 IT3 行。

## 1. 目标与非目标

**目标**：在 **1x 回放的实时因果路径**上，对每 200 ms 段做三级质量判定，把无效段标红并记录原因（C5），产出一个能启动、能看到输出的程序 `qc-monitor`，以及：

- 终端截图（含速率比、逐段 ✓/✗/CAL 裁决）；
- 有效率时间轴图 PNG；
- 确定性逐段记录 qc.jsonl。

**非目标（本期明确不做）**：

- 不算 theta、不与阈值比、不给反馈（IT4）；
- 不做行为任务、不做会话六阶段编排（IT5/IT6）；
- **不调优算法、不因“有效率不好看”改阈值**（BACKGROUND §8）。阈值在本设计冻结；
- 不对有效率设“通过线”，有效率数字只作结果留痕；
- 不做疗效或跨被试结论。

## 2. 经验性前提（均已在 ds007169 sub-001 实测）

这些实测决定了设计，不能照手册字面反推：

1. **L1/L3 的 100µV 级峰峰值门槛只对带限信号成立**。原始宽带信号被直流偏置/慢漂移主导：
   - 原始 Fz 200ms p2p 中位 205.7µV，**4785/4792 段（99.9%）>100µV**；FP1/FP2 原始 p2p 中位也约 187µV。照字面在原始信号上判，几乎全灭。
   - 仅因果 1Hz 高通仍 99.9% 超阈，且此时 OLS 回归系数异常。
2. **在 1–30Hz 带限信号上量级才合理**。因果 4 阶 Butterworth（1Hz 高通串 30Hz 低通，单向 `lfilter`）稳态后：Fz p2p 中位约 72µV、约 24% 段 >100µV；群延迟峰值约 16ms（< 一段 200ms）。
3. **L2 回归可行，但校准必须剔除滤波器启动瞬态**。校准窗内若包含启动样本，OLS 会被污染（b 异常到约 −4.8/6.7，去眼电后反而 99% 超阈）。剔除前 10s 启动样本后，前 120s 估出的冻结系数合理：
   - 因果带通：b(FP1,FP2) ≈ **0.25 / 0.38**；去眼电后残差 p2p 中位约 37µV、约 1% 段 >100µV。
   - 零相位参照值 0.25/0.37（实时禁用，仅作量级对照）。
4. **L3 阈值手册未给数值**。带通后 FP1/FP2 两路取大的 200ms p2p：>150µV 约 17.3%、>200µV 约 12.1%、>300µV 约 6.0%。FP1/FP2 是额极头皮 EEG（非专用 EOG），平静时 p2p 中位约 60–70µV。定 **L3=200µV**（约 P90，只抓明显眼动/大伪迹），与 L1 同为绝对阈值、冻结、不调优。
5. **L1 判在 L2 去眼电之后的残差上**（实测综合有效率约 88%，主要由 L3 否决）；若判在去眼电之前则约 75%。采用前者：L2 能扣掉的眼电不应浪费，L3 专拦“眼电大到回归不可信”的段。
6. 第一个正式（非教程）n-back 试次在 t=246s，与本程序无关（QC 不按试次切分），仅记录在此以免误解。

## 3. 架构与物理隔离（C3）

- 新 console script **`qc-monitor`** → `bci_sys.replay.qc_cli:main`。
- 全部新代码位于 **`src/bci_sys/replay/`**（实时因果侧）。复用 `segments.py`、`pace.py`、`gitmeta.py`。
- **禁止 import `bci_sys.offline.*`**（零相位滤波在那边，物理隔离）。既有 AST 边界测试扩展覆盖新文件。offline 的 theta/滤波器一个都不复用。
- 共享层只允许 `config.py`、`io_bids.py`。

新增/修改文件：

| 文件 | 职责 |
|---|---|
| `replay/causal_filter.py` | 流式因果 1–30Hz 带通：对 Fz/FP1/FP2 三路，各维护高通、低通两套 `lfilter` 的 `zi` 状态，按 50 点一块推进 |
| `replay/qc.py` | 纯逻辑、无 I/O：冻结 OLS 系数、残差、p2p、三级判定、校准→运行状态机 |
| `replay/qc_monitor.py` | 编排：preflight→读三通道→1x 循环→终端标色→收尾写 qc.jsonl/run.json/时间轴 PNG |
| `replay/qc_cli.py` | argparse 入口与退出码 |
| `io_bids.py` | 新增 `load_channels_uV(subj, names, root) -> dict[str, np.ndarray]`（值**已是 µV，禁止 ×1e6**，docstring 沿用单位陷阱警告）；preflight 增加 FP1/FP2 按名存在检查 |
| `config.py` | 新增冻结常量（见 §4） |

## 4. 冻结参数（S2 语义，运行期无修改入口，C2）

新增 config 常量：

- `QC_CALIBRATION_SEC = 120.0`（校准窗 = S3 两分钟）
- `QC_FILTER_WARMUP_SEC = 10.0`（估 b 时剔除的滤波器启动样本）
- `QC_L1_P2P_UV = 100.0`
- `QC_L3_EOG_P2P_UV = 200.0`
- 复用 `FILTER_BAND_HZ = (1.0, 30.0)`、`EOG_PROXY_CHANNELS = ("FP1","FP2")`、`SFREQ=250`、`SEGMENT_SAMPLES=50`。

滤波器（写死，即 §2 实测所用）：`scipy.signal.butter(4, 1.0/(sf/2), "high")` 与 `butter(4, 30.0/(sf/2), "low")` 两节，**各用单向 `scipy.signal.lfilter`（严禁 filtfilt）**，先高通后低通串联，每节、每通道独立携带 `zi`。频带锁定 1–30Hz。分块流式结果必须与“整段一次性同参数单向 lfilter”数值一致（测试锁定）。

## 5. 每段因果数据流

三路原始信号（µV）持续进入流式因果带通，输出 `fz_f, fp1_f, fp2_f`（每路各自独立的滤波器状态，跨段延续）。

- **校准期 `t1 <= 120s`（phase=calibration）**：
  - 仅推进滤波、累积三路**滤波后**样本；不做有效/无效裁决，段记录 `phase="calibration"`, `valid=null`, `reasons=[]`，且 `fz_clean_p2p=null`、`eog_p2p=null`。
  - 终端灰色 `CAL`。
  - 在到达 120s 边界时一次性估计冻结系数：取滤波后样本区间 **[10s, 120s]**（剔除前 10s=2500 点启动样本，拟合样本 110s=27500 点），对三路**去均值**后做无截距 OLS
    `b = lstsq([FP1, FP2], Fz)`，冻结 `b1,b2`，写一条 `calib` 记录（系数、拟合区间、样本数）。
- **运行期 `t1 > 120s`（phase=running）**，每段：
  - `fz_clean = fz_f − b1·fp1_f − b2·fp2_f`（逐样本，使用冻结系数，不再更新，C2）；
  - `l1 = p2p(fz_clean 段内50点) > 100`；
  - `l3 = max(p2p(fp1_f), p2p(fp2_f)) > 200`（**两路任一超阈即否决**，L3 是否决项）；
  - `valid = (not l1) and (not l3)`；`reasons` 收集所有命中项（可同时为 `["L1","L3"]`，C5 全记录）。
  - 终端：有效绿 `✓`；无效红 `✗` 加原因（如 `✗ L3`、`✗ L1 L3`）。

有效率分母**只含运行期段**（校准段不计入）。无效只标记、不删除、反馈在 IT4 才暂停（C5）。

## 6. 终端输出

每行：`段号 │ t=…s │ Fz_clean p2p …µV │ EOG p2p …µV │ rate <色>x │ <裁决标记>`。

- rate 着色复用 pace 阈值（绿 ≥0.98 / 黄 ≥0.95 / 红更低）；开头启动瞬态与 IT1 同理。
- 裁决色：`✓` 绿、`✗ …` 红、`CAL` 灰。
- 启动时打印一行固定参数回显（含 L1/L3/校准秒数/EOG 通道/direction 不涉及本期）；冻结时打印一行 `b_FP1=… b_FP2=… (frozen)`。
- Windows 控制台 UTF-8 处理沿用 replay CLI 的 `sys.stdout.reconfigure`。

## 7. 产物与确定性（A4 / D2）

目录 `outputs/qc/sub-<id>/<run_id>/`：

- **qc.jsonl**（同 subj、两次运行逐字节一致；不含任何墙钟量）：
  - `meta`：`type, schema_version, code_version, subj, sfreq, segment_sec, channels{Fz,FP1,FP2}, filter_band_hz, calibration_sec, filter_warmup_sec, l1_p2p_uv, l3_eog_p2p_uv, n_segments_planned`；
  - `calib`（仅一条，冻结时）：`type, t_sec, b_fp1, b_fp2, fit_window_sec:[10,120], n_fit_samples`；
  - `seg`：`type, seg, t0, t1, phase, fz_clean_p2p, eog_p2p, valid, reasons`；运行期 `eog_p2p = max(p2p(FP1), p2p(FP2))`，校准段四个度量字段为 null、`reasons=[]`；
  - `summary`（仅 status=ok）：`n_running, n_valid, n_invalid, n_l1, n_l3, n_l1_and_l3, valid_rate`。
- **run.json**（唯一墙钟载体）：`subj, status, wall_started_iso, wall_elapsed_s, rate_ratio, n_segments`。
- **valid_rate_timeline.png（有效率时间轴图）**：
  - 上轨：逐段裁决色带（有效绿 / L1 红 / L3 深红或可分色 / 两者皆中 / 校准灰）；
  - 下轨：仅运行期段、**30s 分箱**有效率折线；
  - 出图在跑完后对“已产生的因果标签”做事后汇总，不回流任何裁决；绘图代码与纯因果判定函数分离，故不违反 C3。PNG 不纳入字节一致性比对。
- KeyboardInterrupt：`status="aborted"`、不写 summary、保留已写文件（同 IT1）。

## 8. 错误处理（C7 硬拦，失败不写文件）

- preflight 在现有检查（文件可读、sfreq=250、Fz 存在、EEG=19 导、nback 列、Fz 1–30Hz std∈[5,300]µV）之上，**增加 FP1、FP2 按名存在**，缺任一则 raise。
- 冻结时 b 含非有限值（NaN/Inf）→ raise。
- 不为有效率/系数另设任意数值门槛（单位 gross 错误已由 preflight std 门挡）。
- CLI 退出码：ok=0；preflight/数据错误=2；中止=130（与 replay-monitor 一致）。

## 9. 测试（TDD）

不依赖数据的为主，真数据只做结构验收：

- **流式滤波**：50 点分块携带 `zi` 的输出 == 整段一次性单向 `lfilter`（数值一致）。
- **因果性（守 C3）**：同一前缀先跑到 k 段记录输出；喂入更长数据重跑，前 k 段输出逐点不变（追加未来样本不改变过去）。
- **QC 纯逻辑**：
  - 合成 `Fz = b1·FP1 + b2·FP2 + 已知独立残差`，校准后精确回收 b；
  - L1 残差超阈→invalid(reason L1)；FP1 或 FP2 任一 p2p 超 L3→invalid(L3)；两者皆中→reasons 合并；
  - 状态机：120s 前全部 calibration 且无 valid 判定；恰在边界产出一条 calib；冻结后系数不变；warmup 区间样本被排除（构造启动瞬态，验证不剔除会估错、剔除后正确）。
- **monitor/CLI**：qc.jsonl schema/计数/summary 自洽（n_valid+n_invalid=n_running，valid_rate=比值）；**A4 两次运行字节一致**；aborted 无 summary；preflight 失败不写任何文件；缺 FP1 时报错；退出码 0/2/130。
- **边界**：AST 测试扩展，新 replay 模块不得 import offline。
- **真数据（sub-001，结构性）**：程序跑通；冻结 b 有限且落在实测合理区间（每个系数约 0.1–0.7）；产出时间轴 PNG 与 summary。**不对有效率设断言**。

## 10. 验收（亲手跑、截图存 evidence）

1. `uv run qc-monitor --subj 001`：前 120s 灰色 `CAL` → 打印冻结 b → 之后绿 `✓` / 红 `✗ L1|L3`，rate 绿 1.00x。
2. 全量 1x 跑完（约 16 分钟，可后台），产 valid_rate_timeline.png；
   - 终端稳态截图 → `evidence/IT3-qc-终端.png`；
   - 时间轴图 → `evidence/IT3-qc-有效率时间轴.png`；
   - qc.jsonl 两次运行人工 `diff` 字节一致。
3. 全量测试通过；evidence README 记录 sub-001 冻结 b、有效率、L1/L3 计数（结果留痕，非疗效结论）。

## 11. 风险与备注

- 绝对阈值（100/200µV）来自 ds007169 回放与手册；硬件到货后阻抗/参考不同可能需要在新的 S2 重新确认——属以后事项，本期冻结不调。
- 10s warmup 是针对所选 4 阶因果滤波器启动的经验值；实现时以因果性/回收精度测试锁定，不随手改。
- FP1/FP2 非专用眼电，L3 是保守否决；L2 残差门兜底其余大伪迹。
