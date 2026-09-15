# IT4 `feedback_loop` 设计：S3 冻结阈值 → S4 六 block 实时 θ 反馈（Web 版）

日期：2026-09-12
前置：IT1 `replay_monitor`、IT2 `theta_probe`、IT3 `qc_monitor` 已完成并合并 main（92 passed；IT3 全量有效率 87.45%，b_FP1=0.2482 / b_FP2=0.3781）。
依据：`BACKGROUND.md` §2 会话六阶段、§3 关键取值、§4 C1–C7、§5 D 系、§6（ABABA 属 IT7）、§7 A 系、§9 IT4 行。

## 1. 目标与非目标

**目标**：在 **1x 回放的实时因果路径**上，做出一个能启动、浏览器里看得到输出的程序 `feedback-loop`：

- S3 校准 120s：与 IT3 同一冻结时刻冻结 EOG 回归系数，并**由数据算出并冻结 θ 阈值**（S2/C2，运行期无修改入口）；
- S4 六个 block × 120s，block 间 30s 休息（休息只走墙钟、数据钟暂停）；
- 每个 200ms 段：L2 去眼电后的 Fz_clean → 因果 4–8Hz → θ 功率平滑 → 与冻结阈值比；**QC 无效段反馈暂停**（不正向不负向、不计入达标率、记录原因，C4/C5）；
- 浏览器单页实时显示：确认页（C7 硬门）→ 校准屏 → 冻结卡 → block 进度条 + θ 表 → 无效暂停 → 休息倒计时 → 汇总屏；
- 落盘确定性 `feedback.jsonl` + `run.json`；同一网页支持读历史 jsonl 做 1x **回放（replay 模式）**，替代手册原定的 GIF 录屏。

**非目标（本期明确不做）**：

- 不做 n-back 任务刺激/按键/正确率（IT5）；页面无任务区；
- 不做 S1 阻抗质检、S5 后测、S6 完整会话编排与六件套（IT6）；
- 不做 ABABA 伪反馈与揭盲（IT7）；
- 不做难度自适应（手册：只留接口，固定 2-back 语义）；
- 不调优算法、不按运行结果改阈值/分位/窗长（§8）；block 达标率无「通过线」，只记录留痕；
- 不落原始/滤波波形（见 §7）；
- 不做任何疗效结论；所有结果表述带「在回放模式下」。

### 1.1 与数据集协议的关系（2026-09-12 实测确认）

ds007169 自己的任务协议是 **4 个难度级（1/2/3/4-back）各 100 个真实试次、SOA 实测 1.701s、组间约 8.7s**，单组约 177s（sub-001：组起点 246.0/423.1/600.2/777.3s，末试次止 945.7s）。这是**数据集的协议**，不约束本系统的会话协议。

本系统 S4 固定为手册 §3 的 **6 × 120s block**：

- 真人会话预算：S3 120 + S4 720 + 休息 150 + S5 120 = 1110s（18.5min），在 25min 硬上限内；
- 回放数据预算：S3 120 + S4 720 = 840s（4200 段）≤ 文件 958.512s，尾 118.5s（592 段）不用；
- **IT5 约束（留痕）**：未来 2-back 任务块必须适配 120s——6s 指令 + 67 trials × 1.7s = 119.9s；不沿用数据集的 100 trials/176s；
- 120s block 边界会切穿数据集难度组（240/360/480/600/720/840s vs 组起点 246/423/600/777s）。IT4 无任务、不评估疗效，仅作描述性事实记录，不做跨难度解释。

## 2. 经验性前提（均已在 ds007169 sub-001 实测，2026-09-12）

脚本为一次性分析（不进仓库）；单位 µV，未乘 1e6。

1. **冻结系数与 IT3 一致**：1–30Hz 因果带通、[10s,120s] 去均值无截距 OLS 得 b_FP1=0.248183、b_FP2=0.378130（IT3 全量冻结值 0.2482/0.3781），Fz_clean = Fz_f − b1·FP1_f − b2·FP2_f。
2. **θ 链路**：clean 上再串因果 4 阶 Butterworth hp(4)+lp(8)（单向 lfilter，与 IT3 滤波器同构）；段功率 = 段内 50 点平方均值（µV²）。
3. **平滑窗实测选型 W=5 段（1s）**：在 S3 有效段平滑值上取 P20 作阈值时，六 block 达标率（同一份冻结阈值）：
   - W=5：0.76 / 0.88 / 0.80 / 0.83 / 0.81 / 0.88（全 ≥0.75，四个落在 0.75–0.85）；
   - W=10（2s）：0.72 / 0.89 / 0.84 / 0.89 / 0.87 / 0.91；W=20（4s）：0.59 / 0.91 / 0.68 / 0.86 / 0.84 / 0.95——窗越长慢漂移主导、block 间摆幅越大。
   - 采用 **W=5、仅有效段入窗**（全段平滑与仅有效段数值差异 <0.01，但语义上伪迹段不得进入反馈指标）。
4. **阈值不硬编码**：sub-001 上 W=5 仅有效段平滑的 S3 P20 实测 ≈ 16.74 µV²（n=491，calib_hit_rate=0.800）。运行时由校准数据现算冻结；16.74 仅作验收时的量级参照，不写进 config/不断言。
5. **C4 实证**：不去眼电时无效段上 θ 平滑功率中位 202 µV²，去眼电 + QC 门控后同段 35 µV²（约 6 倍）——眨眼确实抬高 θ 带功率，L2/L3 是必要环节，不是装饰。
6. S4 区间（k=600…4199）实测无效段 43/41/57/71/85/108（7%→18%，逐 block 增多），与 IT3 全量时间轴的非平稳现象一致；如实留痕，不调阈值。
7. n-back 真实试次时序与 IT4 无关（IT4 不按试次切分），§1.1 已记录。

## 3. 架构与物理隔离（C3）

- 新 console script **`feedback-loop`** → `bci_sys.replay.feedback_cli:main`。
- 全部新代码位于 **`src/bci_sys/replay/`**；**禁止 import `bci_sys.offline.*`**，既有 replay 目录 AST 守卫（`tests/test_boundary.py`）自动覆盖新文件。共享层只允许 `config.py`、`io_bids.py`。
- 新增依赖：`fastapi`、`uvicorn`（标准库 `http.server` 方案被否决；WebSocket 双向帧 + 静态托管）。前端无构建链、无外部 CDN（离线可用）。
- 复用：`causal_filter.CausalBandpass`（加可选 `zi` 注入，默认行为不变）、`qc.QcJudge/fit_eog_coeffs/p2p_uv`、`segments.iter_segments`、`pace`、`gitmeta`、`io_bids.load_channels_uV/preflight`。

| 文件 | 职责 |
|---|---|
| `replay/theta_metric.py` | 纯逻辑无 I/O：因果 4–8Hz 滤波节构造、段功率、W=5 仅有效段尾窗平滑器、阈值分位、达标比较（C1 direction 显式） |
| `replay/feedback.py` | 纯逻辑会话状态机：相位推进、block 计数、暂停语义、freeze 失败/incomplete/wall_cap 裁决 |
| `replay/feedback_engine.py` | 引擎线程编排：preflight→读三通道→1x 循环→休息挂数据钟→有界队列广播→写 feedback.jsonl/run.json |
| `replay/webapp.py` | FastAPI：静态单页、WS snapshot+推流、`/api/start`、`/api/abort`、`/api/run`、replay 取 jsonl |
| `replay/web/index.html`、`replay/web/app.js` | 无构建单页（§6 八相位） |
| `replay/feedback_cli.py` | argparse、uvicorn 启动、live/replay 两模式、退出码 0/2/130 |
| `config.py` | 新增冻结常量（§5.3） |
| `tests/` | §9 测试矩阵 |

**进程模型（已定方案 A）**：uvicorn/FastAPI 在主线程；`POST /api/start` 后引擎在单个后台 daemon 线程跑同步实时循环；每段结果 dict 经 `queue.Queue(maxsize=256)` 交给 asyncio 桥接协程广播 WS。引擎**永不阻塞在 UI 上**：`put_nowait`，满则丢最旧帧并累加 `dropped`；消息带连续 `seq`，客户端凭 seq 断档重取 snapshot。中止信号为 `threading.Event`，段等待与休息 sleep 均以 ≤200ms 粒度检查。

## 4. 每段因果数据流

```
Fz/FP1/FP2 ──因果1–30Hz(CausalBandpass, 跨段zi)──▶ QcJudge（冻结b / L1 / L3）
                                                         │ clean=Fz_f−b1·FP1_f−b2·FP2_f
                                            因果4–8Hz（hp4串lp8, 跨段zi, 校准zf续接）
                                                         │ 段功率 mean(θ²)
                                    W=5 仅有效段尾窗均值 ──hit=(smoothed≥θ_thr), direction="up"
```

### 4.1 校准期（k=0…599，t1 0.2→120.0s）

- 逐段推进三路 1–30Hz；引擎同时累积三路**滤波后**样本（list 拼接，30000×3×8B=720,000B≈0.72MB 内存，测试断言常驻 < 10MB，不写盘）；QcJudge 处 calibration 相位（valid=null，不裁决，页面灰格）。
- **θ 相关数值在校准期不存在**：live WS 只推 UI 帧 `calib_tick {k,t1}` 驱动灰格（该帧不落盘）；θ 功率在冻结瞬间重建后一次性补算（见 §4.2），600 条完整 `calib_seg` 数据行在 freeze 行之前写入 jsonl。replay 模式直接用 jsonl 的完整 `calib_seg` 行按 1x 时刻推帧（页面同样只取 k/t1 画灰格）。
- 校准期中止：无 calib_seg 行、无 freeze，落 aborted（reason=user_abort）。

### 4.2 冻结瞬间（t1=120.0s，一次完成，全部失败硬停 C7）

1. b1/b2 直接取自 **QcJudge 在本段 freeze 时产出的 calib 记录**（QcJudge 内部对同一批校准块完成 `fit_eog_coeffs(..., warmup=2500)`）；引擎不二次调用拟合，保证 θ 链路与 QC 用同一对系数；
2. 重建 `cal_clean = Fz_cal − b1·FP1_cal − b2·FP2_cal`（30000 点）；
3. **一次性单向** 4–8Hz lfilter（以 `zi=np.zeros(...)` 调用并取回 zf）滤波 cal_clean，保存末端滤波器状态 **zf**（hp/lp 两节各自的 zf）；
4. 600 段功率 `mean(θ²)`；校准段有效性按运行期同一规则在 cal_clean/FP1/FP2 上重算（L1 残差 p2p、L3 EOG p2p，阈值同冻结常量）；
5. W=5 仅有效段尾窗平滑（deque），取 t1>10s 且平滑已成熟段的平滑值集合 E（sub-001 实测 n=491）；
6. `theta_thr = np.quantile(E, 0.20, method="linear")`（numpy 2.0 线性插值，调用方式在测试中锁定）；记录 `calib_hit_rate = mean(E≥thr)`；
7. **zf 注入**运行期 θ 滤波器两节作为初始 zi。不变量（测试锁定）：运行期逐块流式 θ 输出 == 对 clean[0:840s] 以 zi=0 整段一次性单向滤波的对应样本，逐点误差为 0（同机同 BLAS）；
8. 冻结成功：写 freeze 行、广播、页面 3s 冻结卡（纯墙钟，不耗数据，不可跳过，C7）；随后 block 1 从 k=600 开始。

**冻结失败硬停**（不进入 S4）：校准有效且平滑成熟段 < `THETA_MIN_CALIB_SEGS=50`，或阈值非有限，或 b 拟合抛错 → `status=aborted, reason=freeze_failed`，run.json 照写，页面中止屏显示原因。

### 4.3 运行期（k=600…4199）

每段：

1. 三路 1–30Hz 推进 → QcJudge running verdict（fz_clean_p2p、eog_p2p、valid、reasons）；
2. clean 50 点 → 4–8Hz 两节（zi 跨段）→ `theta_pow = mean(θ²)`；
3. **valid=true**：功率入 W=5 deque（满则弹出最旧），`smoothed=mean(deque)`，`hit = smoothed >= theta_thr`（direction="up" 由构造注入并校验，C1）；n_valid+=1，n_hit+=int(hit)；
4. **valid=false 或 theta_pow 非有限**：`smoothed=null, hit=null`，功率**不入窗**、deque 不动；n_valid/n_hit 不变（C5：暂停，既不正向也不负向）；前端仪表保持上一值 + 暂停横幅显示 reasons；
5. deque 跨 block、跨 rest 连续不重置（rest 无数据进入，自然挂起）；滤波器 zi、QcJudge 状态同样跨 rest 保持；
6. 每 600 段一条 `block_summary`，计数归零进 rest（最后一块后直接 finished）。

### 4.4 休息（5 × 30s，纯墙钟）

不推进段游标/任何 zi/deque；rest 开始/结束各一条 phase 行与 WS 帧（行内只有 rest_idx 与冻结时长 30.0，**不含墙钟**）；倒计时前端按服务器帧时间戳本地倒数，结束由引擎 Event 计时驱动（前端计时器只负责显示）。rest 中可中止（≤200ms 响应）。

### 4.5 达标率口径

- 进度条填充（block 内）= `n_hit / n_valid`；n_valid=0 时为 0（不除零）；
- block_summary.hit_rate = block 内 `(valid 且 hit)` 段数 / **valid** 段数；无效段分子分母都不进；
- overall_hit_rate（final 行）= 六块 n_hit 之和 / 六块 n_valid 之和。

## 5. 会话状态机与冻结参数

### 5.1 相位与定时（数据钟，1 段=0.2s）

| 相位 | 段范围 | 段数 | 墙钟 |
|---|---|---|---|
| calibration | k=0…599 | 600 | 1x |
| freeze_card | — | 0 | 3s |
| block 1…6 | k=600…4199，每块 600 | 6×600 | 1x |
| rest 1…5 | — | 0 | 各 30s |
| finished | 不拉取（尾 592 段不用） | — | 汇总 |

数据 840.0s；墙钟 ≈ 840 + 150 + 3 + 确认等待。相位推进权只在引擎同步循环，按已发射段数硬推进；页面无继续/延长/跳过控件（C7）。

### 5.2 异常终态

- `user_abort`：POST /api/abort 或 Ctrl+C；已写行不删（D3），无 final、无汇总；
- `incomplete_data`：可用整段不足 4200（换被试/截断文件）；在最后一段正常收尾，aborted 行列出 `blocks_completed`；禁止「第 6 块缩水跑完标 ok」；
- `wall_cap`：墙钟超过 `SESSION_WALL_CAP_SEC=1500`（25min 硬上限兜底）；
- `freeze_failed`：§4.2。

CLI 退出码：ok=0；preflight/数据错误=2；user_abort=130（Ctrl+C 惯例）；incomplete_data/wall_cap/freeze_failed=3（数据/会话条件不满足，非用户中断）；engine_error=1（§8）。

### 5.3 config.py 新增（C2，运行期无修改入口）

```python
THETA_SMOOTH_SEGS = 5        # 仅有效段尾窗（1s）
THETA_QUANTILE = 0.20        # S3 阈值分位（numpy linear）
THETA_TARGET_RATE = 0.80     # 设计目标达标率（=1−分位）；仅冻结时使用，不参与运行期控制
THETA_MIN_CALIB_SEGS = 50    # 冻结所需最小合格校准段
FEEDBACK_N_BLOCKS = 6
FEEDBACK_BLOCK_SEC = 120.0
FEEDBACK_REST_SEC = 30.0
FEEDBACK_FREEZE_CARD_SEC = 3.0
SESSION_WALL_CAP_SEC = 1500.0
```

复用 `THETA_BAND_HZ=(4.0,8.0)`、`QC_CALIBRATION_SEC/QC_FILTER_WARMUP_SEC/QC_L1_P2P_UV/QC_L3_EOG_P2P_UV`、`SFREQ/SEGMENT_SAMPLES`、`TARGET(channel=Fz, band=4–8, direction="up")`。

## 6. Web 层

仅绑 `127.0.0.1`；启动后终端打印 URL（不自动开浏览器）；默认端口可用 `--port` 覆盖。静态资源无外链。

### 6.1 HTTP/WS

| 路由 | 语义 |
|---|---|
| `GET /` | 单页 |
| `WS /ws` | 连接先发一帧 `snapshot`（当前完整状态），之后推 event 帧；最大连接数 1，第二连接 429 |
| `POST /api/start` | 一次性；创建引擎线程；二次 → 409；未确认前不存在引擎状态（C7） |
| `POST /api/abort` | 置 Event；可中止相位 202，finished/aborted 后 409 |
| `GET /api/run` | 当前 run 目录、status、模式 |
| replay：`GET /api/replay?file=<白名单相对路径>` | 读 jsonl → 同一事件帧协议 → 同一前端，1x 重放；live 专有 POST 在 replay 模式返回 403；白名单外 400 |

snapshot 字段：`{mode, phase, block_idx, seg_in_phase, k, t1, b_fp1, b_fp2, theta_thr, params 回显, n_valid, n_hit, block_rates[], last_smoothed, last_verdict, seq, dropped}`。
event 帧类型：`calib_tick {k,t1}`（**live 专属 UI 帧，不落盘**——校准期 θ 尚不存在）、`freeze`（同 jsonl freeze 行）、`seg`（同 jsonl seg 行）、`phase`（仅 block_start/rest_start/rest_end）、`block_summary`、`final`、`aborted`；replay 模式用 jsonl 完整 `calib_seg` 行替代 calib_tick（页面两种帧都按 k/t1 画灰格）。除 calib_tick 外，WS 数据帧字段与 jsonl 同名行**一字不差**（「浏览器所见」可由落盘重算）。会话完成/中止只经 final/aborted 帧传达，不另发 finished phase 帧。每帧带 `seq`；数据字段与 jsonl 同名行**一字不差**（「浏览器所见」必须可由落盘重算）。

### 6.2 页面八相位（草图已确认，布局 A）

1. **确认页**：subj、数据文件名、靶点 direction=up、全部冻结参数回显；「确认参数并开始 S3 校准」按钮（C7 唯一启动入口）；
2. **校准屏**：「S3 校准中」+ mm:ss/02:00 倒计时 + 600 灰格推进 + 中止按钮；无进度条/θ 表；
3. **冻结卡**：b_FP1/b_FP2、θ_thr（实测值）、n_thr_segs、calib_hit_rate、direction=up；3s 自动切 block 1；
4. **block 运行**：六块点阵（●○）、BLOCK i/6、本块剩余时间、QC 状态灯；主进度条（n_hit/n_valid，80% 白色目标刻线）；θ 表（当前 smoothed 位置 + 黄色阈值刻线）；
5. **无效暂停**：进度条斜纹冻结、仪表冻结、红色横幅「本段无效（原因 L1/L3）· 反馈暂停 · 不计入达标率」；
6. **休息**：黄字「休息中 · 数据钟暂停」+ 30s 倒计时 + 上一块达标率（仅记录，阈值不变）；
7. **finished 汇总**：数据/墙钟时长、六块达标率与无效段表、产物路径；附「不是疗效结论」声明；
8. **aborted**：reason、已完成到的 block/k/t1、「数据只标记不删除」。

前端不发明相位，严格镜像引擎；本地行为仅限：无效帧保持上值、rest 倒计时显示、冻结卡 3s 动画兜底（实际切换以后端 phase 帧为准）。

**关页不中止**：浏览器关闭/断开 WS 不影响引擎与落盘（防止误触浪费一次会话；本机单被试）；重连经 snapshot 追平。

### 6.3 replay 模式

`feedback-loop --replay <run目录或feedback.jsonl路径>`：不起引擎、不写任何文件；白名单 = 该参数解析出的单一文件（拒绝目录外/`..`）；顶栏「回放模式」标识；控件：暂停/继续、拖动到任意 t1（跳到该 t1 前最近相位边界重建 snapshot）。测试锁定：同一 jsonl 重放出的事件序列数据字段与 live 帧完全一致。

## 7. 落盘与 A4 确定性

目录：`outputs/feedback/sub-001/<YYYYMMDD-HHMMSS>/feedback.jsonl` + `run.json`。无 PNG/GIF；replay 只读不写。

### 7.1 feedback.jsonl（LF 锁定、`newline="\n"`、逐行 flush、仅引擎线程持有写柄）

1. **meta**（首行，目录创建前先过 preflight，沿 IT3 顺序）：
   `type, schema_version="1.0", subj, data_vhdr, git_commit, target{channel,band_hz,direction}, params{sfreq,segment_sec,qc_calibration_sec,qc_filter_warmup_sec,l1_p2p_uv,l3_eog_p2p_uv,theta_band_hz,theta_smooth_segs,theta_quantile,theta_target_rate,theta_min_calib_segs,n_blocks,block_sec,rest_sec,freeze_card_sec,session_wall_cap_sec}`；
2. **calib_seg ×≤600**：`{type:"calib_seg",k,t1,theta_pow_uV2,valid,smoothed_uV2:number|null}`（冻结瞬间补算补写；前 10s/未成熟段 smoothed=null）；
3. **freeze ×1**：`{type:"freeze",t_sec:120.0,b_fp1,b_fp2,theta_thr_uV2,smooth_win_seg,quantile,target_rate,direction:"up",n_thr_segs,calib_hit_rate,n_fit_samples:27500,fit_window_sec:[10.0,120.0]}`；
4. **seg ×3600**：`{type:"seg",k,t1,block_idx,seg_in_block,theta_pow_uV2,smoothed_uV2:number|null,hit:boolean|null,fz_clean_p2p_uv,eog_p2p_uv,valid,reasons:[...]}`；
5. **phase ×16**：block_start ×6（带 t1）、rest_start/rest_end ×各 5（rest_idx、rest_sec=30.0，无墙钟）、finished 状态在 final 行；
6. **block_summary ×6**：`{block_idx,t1_start,t1_end,n_valid,n_invalid,n_hit,hit_rate}`；
7. 终态：**final** `{status:"ok",n_valid,n_invalid,n_hit,overall_hit_rate,block_rates[6]}` 或 **aborted** `{status:"aborted",reason,k,t1,blocks_completed}`。

非有限值一律经 `_finite_or_none` 落 `null`（沿 IT3），禁止裸 NaN/Infinity；numpy 标量落盘前 `float()/int()/bool()`。

### 7.2 run.json（墙钟只在这里）

`{subj,status,reason?,mode:"live",wall_started_iso,wall_elapsed_s,data_elapsed_s:840.0|实际,rate_ratio,n_calib_seg,n_running_seg,block_rates,git_commit}`；aborted 无 final 行但 run.json 字段照写（n_* 取实际已落盘计数，二者相等由测试锁定）。

### 7.3 A4

- 两次独立全跑（不同墙钟/目录）`feedback.jsonl` **字节一致**（cmp + sha256）；WS 线程、队列、dropped、seq、墙钟量一律不进 jsonl；
- 流式 θ（zf 续接）与整段一次性单向滤波逐点一致（§4.2-7）；
- **独立重算测试**：只读 jsonl + meta 参数 + ds007169 数据，重算 freeze 阈值与各块 hit_rate，与落盘值一致；
- 跨机 BLAS/LAPACK ULP 只要求数值容差，字节一致同机验证（沿 IT3 声明）。

## 8. 错误处理与硬拦（C7）

- preflight 失败（文件/采样率/通道/单位硬门）：CLI rc=2，不建任何目录；
- 冻结失败 / 数据不足 / 墙钟超限：硬停到对应 aborted reason，不提供「继续跑」「忽略」入口；
- 引擎线程内未预期异常：置 aborted(reason=engine_error, error 类型名)、jsonl 落 aborted 行、run.json 照写、WS 广播后 CLI 以 **rc=1** 退出；traceback 只进终端不进浏览器；
- WS 断开不中止引擎；第二连接 429；start 二次 409；replay 下写操作 403；白名单外文件 400。

## 9. 测试矩阵（pytest，沿现有形态）

| 层 | 测试要点 |
|---|---|
| theta_metric | 4/8Hz 系数冻结；50 点分块流式 == 整段单向；**zf 续接逐点一致**；解析信号段功率；W=5 只吃有效段、无效不入窗且输出保持；非有限不入窗；direction 缺失/非法构造即报错（C1）；构造分布上 P20 阈值给出 ≈0.80 达标率、`method="linear"` 锁定 |
| feedback 状态机 | 相位/段游标全序列；rest 不推进任何状态；计数只在有效段动（C5）；n_valid=0 不除零；block_summary 数值；incomplete_data→blocks_completed；wall_cap；freeze_failed（注入 <50 合格段）；freeze_card 3s 不耗数据 |
| 引擎 | 短跑 schema；A4 双跑 sha256 一致且 jsonl 无墙钟词；abort 段中与 rest 中两落点（jsonl seg 数 == run.json）；preflight 零产物；非有限落 null；LF/无 \r；filtfilt AST 守卫覆盖新文件；**jsonl 独立重算一致**；校准缓冲内存上限 |
| webapp | TestClient：未 start 的 snapshot；start 幂等 409；第二 WS 429；abort 202/终态后 409；replay 写操作 403、越界路径 400；replay 帧数据 == 同 jsonl 的 live 字段；seq 断档 snapshot 重同步；断 WS 引擎继续 |
| CLI | 子进程：`--help`；短跑 rc=0 且打印 127.0.0.1 URL；坏被试 rc=2 零产物；user_abort rc=130；截断数据 rc=3（incomplete_data）；engine_error rc=1；`--replay` 不起引擎不写文件 |

时间相关测试注入步进/快放 pacer，不依赖真墙钟；web 测试用 FastAPI TestClient。

## 10. 人工验收（回放模式下）与留证

1. `uv run feedback-loop --subj 001` → 确认页参数齐全；未点开始无任何数据推进（C7）；
2. 校准倒计时到 120s → 冻结卡 b≈0.248/0.378、θ_thr 为实测值（量级参照 ≈16.7 µV²，不断言）、direction=up；
3. 六块进度条逐段跳；无效段斜纹暂停带原因；休息 30s 数据钟挂起；无延长/跳过入口；
4. 汇总屏数值与 jsonl 一致（预估达标率 0.76/0.88/0.80/0.83/0.81/0.88、无效 43/41/57/71/85/108，验收以实际运行为准）；
5. 第二次全跑 cmp 字节一致；
6. `--replay` 1x 重放、暂停、拖到任意 t1，画面与 live 同构。

**evidence/**（每期截图铁律照做，形态更新）：README 增 IT4 节 + 浏览器各相位截图（确认/冻结卡/运行/暂停/休息/汇总）+ 命令、实测数值表、A4 sha256、replay 说明。GIF/PNG 不是程序产物；程序产物只有 jsonl、run.json 与可重放网页。

**完成判据**：程序可启动、浏览器看得到 S3→六块全程实时反馈与 QC 暂停；jsonl 可独立重算并驱动网页回放；A4 字节一致。不下疗效结论。S5/任务/六件套/ABABA 留 IT5–IT7。
