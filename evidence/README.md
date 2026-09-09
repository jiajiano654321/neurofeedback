# IT1 + IT2 验收证据

日期：2026-09-09 ｜ 代码版本（git）：`460f2b5` ｜ 数据：ds007169（sub-001，250 Hz，Fz）

本阶段目标是**端到端跑通管线**（公开数据 1x 回放 + 离线 theta），**不评估疗效**。

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
  `aea38d842b280579018b57f35a288768ea98804a0a5452a730fd8a807ea29e01`
- 墙钟量（`wall_started_iso` / `wall_elapsed_s` / `rate_ratio`）只写在 `run.json`，不进数据侧 `replay.jsonl`，故数据侧跨运行字节一致。

## IT2：theta-probe（离线 Fz theta 负荷检验）

- `IT2-theta-单调曲线.png`：`uv run theta-probe --subj 001` → 打印 `PASS`、退出码 **0**。
  每档 100 个真实（非 tutorial）试次 epoch，中位 Fz theta 功率（µV²）：

  | 1-back | 2-back | 3-back | 4-back |
  |---|---|---|---|
  | 107.12 | 152.09 | 161.80 | 233.06 |

  **1→4-back 严格单调上升，硬门控通过**，方可进入 IT3。

- `IT2-theta-多被试汇总.png`：`uv run theta-probe --all`（纯参考，退出码恒 0）。

  | sub | 结果 | 1 | 2 | 3 | 4 |
  |---|---|---|---|---|---|
  | 001 | PASS | 107.1 | 152.1 | 161.8 | 233.1 |
  | 002 | PASS | 98.5 | 161.7 | 204.3 | 272.6 |
  | 006 | FAIL | 149.6 | 108.7 | 125.3 | 161.9 |
  | 014 | FAIL | **0.0** | 215.4 | 205.7 | 218.9 |
  | 015 | FAIL | 249.9 | 240.9 | 287.9 | 235.8 |
  | 018 | FAIL | 198.3 | 207.3 | 302.1 | 201.8 |
  | 019 | FAIL | 202.5 | 209.2 | 207.6 | 184.3 |

  硬门控只对 sub-001；其余仅作异质性参考，不影响本阶段验收。sub-014 的 1-back 中位数恰为 0 µV² 是被试数据特性，留待 IT3+ 排查（本阶段不证疗效）。

## 经验性发现（实测，非文档反推）

- **第一个真实（`istutorial≠true`）n-back 试次 onset = 246.016 s**；前 ~246 s 为教程/指导段。
- 回放终端的 `◆ N-back` 标记对“带 nback_level 的刺激”显示（含教程级刺激），最早出现在 **t=77.6 s（seg 387）**；而 theta 分析的 `real_trial_events` 严格只取非教程试次。两者用途不同、各自正确。因此 60 s 窗口只见 `✗ dropped` 与教程起始 marker，看不到 `◆`；截图改用 252 s 稳态窗口。
