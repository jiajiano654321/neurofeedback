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
    """返回 ((b_hp, a_hp), (b_lp, a_lp))。默认 band_hz 取自 config.FILTER_BAND_HZ（频带冻结 1–30Hz，C2）。"""
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
    非线程安全：push 必须由单线程串行调用（引擎模型：唯一回放线程）。
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
