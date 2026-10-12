"""流式因果带通滤波器——实时路径上最底层的一块地基（C3 的物理实现）。

什么叫「因果」：滤波器在第 t 个样本上的输出，只允许依赖 t 以及 t 之前的样本。
本模块只用单向的 scipy.signal.lfilter，【严禁 filtfilt / 零相位滤波】——零相位
滤波要把信号倒过来再滤一遍，等于偷看了未来样本。实时反馈里一旦偷看未来，
反馈就是假的，而且这个错不报错、不崩溃、数据看着完全正常。offline/ 离线
分析可以用 filtfilt（反正数据全在手里），两边物理隔离、互不 import。

滤波器结构（文章里说的「8 阶 Butterworth + IIR」）：
    原始信号 → 4 阶高通(1Hz) → 4 阶低通(30Hz) → 1–30Hz 带通信号
    两节都是 IIR（无限冲激响应），串联，各自单向 lfilter。

为什么是 1–30Hz：
    - 原始 Fz 带着约 10⁶ µV 量级的直流偏移和慢漂移，段内 p2p 中位 205µV，
      99.9% 的段直接超过 QC 的 100µV 门——在原始信号上判好坏等于全灭。
    - 带限到 1–30Hz 后 p2p 中位约 72µV，绝对阈值（L1=100/L3=200µV）才有区分度。
    - 4 阶 IIR 的群延迟峰值约 16ms，远小于一个反馈段 200ms，实时上可接受。

流式 == 整段一次性（本模块最重要的不变量，有测试锁死）：
    IIR 滤波器有「记忆」（zi，延迟线状态）。每滤完一块，lfilter 返回新状态 zf；
    把 zf 当下一块的 zi 传下去，50 点一块流式滤出来的结果，与把全部样本
    一次性喂进去逐点相等。初始 zi=0（零初始状态），开机瞬间的启动瞬态由
    QC 校准拟合时剔除前 10s（warmup）解决。

另外 IT4 的 θ 链路（4–8Hz）复用同一套机制：校准 120s 整段滤完后导出末端
zf，运行期从该 zf 续接，保证流式 θ 与整段单向滤波逐点相等。
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .. import config


def design_filter_coeffs(
    sfreq: float = config.SFREQ,
    band_hz: tuple[float, float] = config.FILTER_BAND_HZ,
):
    """设计两节 Butterworth IIR 系数，返回 ((高通 b,a), (低通 b,a))。

    butter 的截止频率要按奈奎斯特频率（采样率的一半 = 125Hz）归一化。
    默认频带 (1, 30)Hz 是冻结值（C2）；IT4 的 θ 链路传 (4, 8) 复用本函数。
    """
    lo, hi = band_hz
    b_hp, a_hp = signal.butter(4, lo / (sfreq / 2.0), btype="high")
    b_lp, a_lp = signal.butter(4, hi / (sfreq / 2.0), btype="low")
    return (b_hp, a_hp), (b_lp, a_lp)


def bandpass_oneshot(
    x: np.ndarray,
    sfreq: float = config.SFREQ,
    band_hz: tuple[float, float] = config.FILTER_BAND_HZ,
) -> np.ndarray:
    """整段一次性单向带通（零初始 zi，不分块）。

    两个用途：① 流式实现的「参照基准」——测试断言分块流式结果与它逐点一致；
    ② 非实时场景（如 preflight 诊断）。实时循环里不用它，用 CausalBandpass。
    """
    (b_hp, a_hp), (b_lp, a_lp) = design_filter_coeffs(sfreq, band_hz)
    y = signal.lfilter(b_hp, a_hp, np.asarray(x, dtype=float))
    return signal.lfilter(b_lp, a_lp, y)


class _CausalSection:
    """单个 IIR 节（高通或低通）的流式封装：一次 lfilter + 状态续接。

    状态 zi 是长度 = max(len(a),len(b))-1 的延迟线，装着「过去样本对未来的
    影响」。push 一块返回这一块的输出，并把新状态 zf 存下来供下一块使用。
    构造时可注入 zi（IT4 θ 链路用校准末端 zf 续接）；注入时拷贝一份，防止
    外部数组别名导致状态被意外改写。
    """

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
    """一个通道的完整带通：高通节串低通节，两节状态各自延续。

    zi=(zi_hp, zi_lp)：任一元为 None 表示该节零初始启动。
    空块原样返回且不碰 zi——scipy 的 lfilter 喂空输入会返回一个错误的非零
    zf，存下去会把 IIR 状态永久污染，之后每一段输出都是错的（还不报错）。
    非线程安全：push 只能由唯一的回放线程串行调用。
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
        """推进一块（默认 50 点 = 200ms）。先高通，其输出立刻进低通。"""
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"通道块必须一维，收到 ndim={arr.ndim}")
        if arr.size == 0:
            return np.empty(0, dtype=float)
        return self.lp.push(self.hp.push(arr))

    def states(self) -> tuple[np.ndarray, np.ndarray]:
        """导出两节当前状态的拷贝（IT4 冻结 θ 链路时取校准末端 zf 用）。"""
        return self.hp.zi.copy(), self.lp.zi.copy()


class CausalBandpass:
    """多个通道并行的因果带通：每通道一个独立 CausalChannel，状态互不干扰。

    用法（qc_monitor / feedback_engine 每 200ms 调一次）::

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
        """三路等长一维块同时推进一段，返回同结构的滤波后块（单位不变，µV）。

        形状检查在推进任何滤波器状态【之前】完成：非法输入直接抛错，所有 zi
        保持原样，调用方失败重试不会污染滤波器（「分块==整段」不变量不能被
        一个边界输入击穿）。空块同样直接返回、不碰 zi（理由见 CausalChannel）。
        """
        arrays = {c: np.asarray(block[c], dtype=float) for c in self.channels}
        sizes = {a.size for a in arrays.values()}
        ndims = {a.ndim for a in arrays.values()}
        if len(sizes) != 1 or ndims != {1}:
            raise ValueError(f"各通道块必须等长且一维: sizes={sizes}, ndims={ndims}")
        if sizes == {0}:
            return {c: np.empty(0, dtype=float) for c in self.channels}
        return {c: self._ch[c].push(arrays[c]) for c in self.channels}
