"""实时质量判定（QC）——整条实时链路的「门卫」，每 200ms 段必过。

它在管线中的位置（每段 = 50 个采样点，按顺序走一遍）：

    Fz / FP1 / FP2 三路信号
        │  上游 causal_filter.py 已做【因果】1–30Hz 带通（只用过去数据，C3）
        ▼
      本文件：
        ① 校准期（前 120s）：只攒数据、不裁决；第 120s 一次性估出眼电回归系数并冻结（C2）
        ② 运行期（120s 之后）：每段先扣眼电（L2），再过两道门（L1 残差门 / L3 眼电门）
        ▼
      valid=True  好段 → 交给 theta_metric 算 θ、给反馈
      valid=False 坏段 → 反馈暂停、原因留痕；数据本身不删不改（C4/C5）

三级处置（BACKGROUND §C4）：

    L2（每段都做的「清洗」，不是门）：Fz_clean = Fz − b1·FP1 − b2·FP2
        用最小二乘把额极通道（FP1/FP2 当眼电代理）里的眼动成分，从 Fz 里扣掉。
    L1（第一道门）：扣完眼电的 Fz 残差，段内峰峰值 > 100µV → 坏段。
        抓 L2 扣不掉的大伪迹（肌电、动作）。
    L3（否决门）：FP1/FP2 任一路段内峰峰值 > 200µV → 坏段。
        眼动大到「回归扣减本身都不可信」时直接否决，不抱侥幸。L3 是否决项，不是参考项。

为什么要这么较真：眨眼会让 θ 频段功率虚高。若不拦，系统会在眨眼时给奖励，
被试最后练出来的是「眨眼技能」而不是上调 θ——而且全程不报错、数据看着完全正常。

本文件的三条铁律：
    - 纯逻辑：无文件 I/O、无墙钟、无随机数，同一输入永远同一输出（A4 字节确定性的一环）。
    - 只用过去：系数只在第 120s 估一次，冻结后永不更新（C2/C3）。
    - 坏段只打标签：输出 {valid: False, reasons: [...]}，绝不删除、绝不「修好再用」（C5）。
"""
from __future__ import annotations

import numpy as np

from .. import config


def p2p_uv(x) -> float:
    """段内峰峰值 = max−min（µV），即这一段信号「上下摆动有多大」。

    用峰峰值而不是标准差：它对瞬时尖峰（眨眼、肌电爆点）更敏感，
    而且不受直流偏置影响（原始信号带 ~10⁶ 量级的直流偏移，标准差会被主导）。
    L1、L3 两道门都拿它当判据。
    """
    x = np.asarray(x, dtype=float)
    return float(x.max() - x.min())


def fit_eog_coeffs(fz_f, fp1_f, fp2_f, start_sample: int) -> tuple[float, float]:
    """在 [start_sample, 末尾] 区间上估计眼电回归系数 b1、b2。

    全系统只在 S3 校准结束（第 120s）调用这一次，估完立即冻结，运行期永不重估。

    模型（去均值 + 无截距最小二乘）：

        Fz_去均值 = b1 · FP1_去均值 + b2 · FP2_去均值

    系数的物理意义：b1=0.25 表示 FP1 每摆动 1µV，Fz 里约有 0.25µV 来自同一个
    眼动源，运行期就按这个比例扣掉（Fz_clean = Fz − b1·FP1 − b2·FP2）。
    sub-001 实测冻结值 b1=0.248、b2=0.378，即 Fz 里约 1/4~1/3 的眼动成分可实时扣除。

    两个理想化假设（OLS 的统计前提，知道即可，现实都不完全成立）：
      ① FP1/FP2 能抓住全部眼动成分；② 脑电与眼电互不相关。
      正因为假设不完美，才需要 L1 拦「扣不干净」、L3 拦「扣得不可信」。

    start_sample 是 warmup 剔除点（默认第 10s = 第 2500 点）：因果滤波器启动
    瞬态会把系数估歪（实测能歪到 −4.8/6.7），所以前 10s 不进拟合窗。

    以下情况一律 ValueError——校准失败必须硬停，不许「凑合冻结」（C7）：
    - 三路长度不一致
    - start_sample 越界
    - 校准窗内含非有限输入（NaN / Inf）
    - 设计矩阵零秩（两路 EOG 校准窗内全恒定，去均值后全零）
    - 回归系数非有限（NaN / Inf）
    """
    fz = np.asarray(fz_f, dtype=float)
    fp1 = np.asarray(fp1_f, dtype=float)
    fp2 = np.asarray(fp2_f, dtype=float)
    if not (fz.size == fp1.size == fp2.size):
        raise ValueError(f"三路长度不一致: {fz.size}, {fp1.size}, {fp2.size}")
    if start_sample < 0 or start_sample >= fz.size:
        raise ValueError(f"warmup 起点越界: {start_sample} / {fz.size}")

    y0 = fz[start_sample:]
    X0 = np.column_stack([fp1[start_sample:], fp2[start_sample:]])
    # 先在原始切片上拦非有限：若先做 inf−mean(inf) 去均值会先抛 RuntimeWarning。
    # 也防住「y 全 NaN、X 全零」时 lstsq 静默返回有限 [0,0] 绕过后续检查。
    if not (np.all(np.isfinite(y0)) and np.all(np.isfinite(X0))):
        raise ValueError("EOG 回归输入含非有限值（NaN/Inf），校准失败")
    y = y0 - y0.mean()
    X = X0 - X0.mean(axis=0, keepdims=True)
    # 零秩（两路去均值后全 0＝全程恒定）时 lstsq 静默返回有限 [0,0]，会伪装成
    # 「冻结成功但完全不去眼电」，必须硬拦（C7）。秩 1（两通道共线、或一路恒定）
    # 仍给出唯一有效的去眼电线性组合（实测最小范数解的残差相同），可接受。
    if np.linalg.matrix_rank(X) == 0:
        raise ValueError("EOG 回归设计矩阵零秩（FP1/FP2 校准窗内恒定？），校准失败")
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    if not np.all(np.isfinite(b)):
        raise ValueError("EOG 回归系数非有限值（NaN/Inf），校准失败")
    return float(b[0]), float(b[1])


def segment_verdict(
    fz_f, fp1_f, fp2_f, b_fp1: float, b_fp2: float,
    l1_thr: float, l3_thr: float,
) -> dict:
    """对【一段】50 点信号做 L1/L3 裁决。纯函数，无状态。

    IT3 的 qc_monitor 和 IT4 的 feedback_engine（冻结瞬间重建 600 段校准时）
    都调用它——同一段数据在两处必须得到同一个判决，所以抽成纯函数共用、禁止各写一份。

    返回（不含 phase；phase 由 QcJudge 状态机在外层补）：
      fz_clean_p2p : 扣眼电后 Fz 残差的段内峰峰值（落盘/画图用）
      eog_p2p      : 两路眼电代理取大后的段内峰峰值
      valid        : True=好段可给反馈；False=坏段，反馈暂停
      reasons      : 命中的门，取值 [] / ["L1"] / ["L3"] / ["L1","L3"]（C5：全记录）
    """
    fz = np.asarray(fz_f, dtype=float)
    fp1 = np.asarray(fp1_f, dtype=float)
    fp2 = np.asarray(fp2_f, dtype=float)

    # L2：扣眼电（每段都做的清洗步骤，本身不是门）。
    clean = fz - b_fp1 * fp1 - b_fp2 * fp2
    clean_p2p = p2p_uv(clean)
    # L3 取两路 EOG 的较大者。必须用 np.maximum：内建 max 遇 NaN 有顺序依赖
    # （max(nan,0) 和 max(0,nan) 结果不同），会让一路坏通道时灵时不灵地漏判。
    eog_p2p = float(np.maximum(p2p_uv(fp1), p2p_uv(fp2)))
    # L1 / L3 判定用 not(<=) 而不是 (>)：NaN 比较大小永远是 False，
    # 若写成 (>)，NaN 段会变成「没超阈」的好段——坏数据被静默放行。
    # not(<=) 让 NaN 段一律命中，绝不把坏段标成有效。
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


class QcJudge:
    """校准 → 运行 两段式状态机。逐段喂入【已经过因果带通】的三路等长数据块。

    数据钟时间线（t1 = 当前段尾的时间）::

        t1 ∈ (0, 120s]   calibration：600 段只攒样本，每段返回 valid=None（页面灰格）
        t1 = 120s 那一段  一次性 _freeze()：拟合并冻结 b1/b2，该段返回值附带 calib 记录
        t1 > 120s        running：每段 L1/L3 裁决，返回 valid=True/False

    数据短于 120s（限时回放/测试）：永不冻结、不报错，调用方循环自然结束。
    冻结后系数和阈值不再改变，运行期没有任何修改入口（C2）。
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
        # 参数一律 float 化：防止调用方传进 numpy 标量或 int 污染后续序列化。
        sfreq = float(sfreq)
        self.calibration_sec = float(calibration_sec)
        self.warmup_sec = float(warmup_sec)
        self.l1_thr = float(l1_thr)
        self.l3_thr = float(l3_thr)
        self.fz_name = fz_name
        self.eog_names = tuple(eog_names)

        # 秒 → 采样点：校准窗 120s×250Hz = 30000 点；warmup 10s = 2500 点。
        self.calib_samples = int(round(self.calibration_sec * sfreq))
        self.warmup_samples = int(round(self.warmup_sec * sfreq))
        self._names = (self.fz_name, *self.eog_names)
        self._buf = {c: [] for c in self._names}  # 校准期按块攒滤波后样本
        self._n_accum = 0
        self.b_fp1: float | None = None           # 冻结前为 None
        self.b_fp2: float | None = None
        self.frozen = False
        self.calib: dict | None = None            # 冻结那一段附带的 calib 记录

    def _calibration_verdict(self) -> dict:
        """校准期段裁决：不判好坏（valid=None），四个度量字段全为 None。"""
        return {
            "phase": "calibration",
            "fz_clean_p2p": None,
            "eog_p2p": None,
            "valid": None,
            "reasons": [],
        }

    def push(self, t1: float, filtered: dict[str, np.ndarray]) -> dict:
        """推进一段。t1=本段尾的数据时间（秒）；filtered 含 Fz/FP1/FP2 的本段样本。

        前置条件：各通道块等长、非空、一维（上游 CausalBandpass / iter_segments 保证），
        judge 不重复校验形状。
        """
        # 冻结后是纯透传：每段直接裁决，状态机不再积累任何东西。
        if self.frozen:
            return self._running_verdict(filtered)

        # —— 校准期：攒样本，不裁决 ——
        block_len = np.asarray(filtered[self.fz_name]).size
        for c in self._names:
            self._buf[c].append(np.asarray(filtered[c], dtype=float))
        self._n_accum += block_len

        verdict = self._calibration_verdict()
        # 可能跨边界（最后一块使累计样本 ≥30000）：冻结时只取前 calib_samples 点。
        if self._n_accum >= self.calib_samples:
            self._freeze(t1)
            verdict["calib"] = self.calib  # 仅冻结这一段带 calib，下游据此写 jsonl
        return verdict

    def _freeze(self, t1: float) -> None:
        """第 120s 调用一次：拼接校准缓冲 → 估 b1/b2 → 冻结 → 生成 calib 记录。"""
        # 截到整 calib_samples：块长不整除校准窗时，多出的尾巴不进拟合。
        arrs = {
            c: np.concatenate(self._buf[c])[: self.calib_samples]
            for c in self._names
        }
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
            "n_fit_samples": self.calib_samples - self.warmup_samples,  # 默认 27500
        }
        self._buf = {}  # 校准原始缓冲不再需要，释放内存（约 0.7MB）

    def _running_verdict(self, filtered: dict[str, np.ndarray]) -> dict:
        """运行期：状态机只负责补 phase，裁决逻辑全部在纯函数 segment_verdict。"""
        return {
            "phase": "running",
            **segment_verdict(
                filtered[self.fz_name],
                filtered[self.eog_names[0]],
                filtered[self.eog_names[1]],
                self.b_fp1, self.b_fp2, self.l1_thr, self.l3_thr,
            ),
        }
