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
