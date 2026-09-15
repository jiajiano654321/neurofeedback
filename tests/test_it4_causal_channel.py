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
    ch.push(x[:50])
    z_before = ch.states()[0].copy()
    out_empty = ch.push(np.empty(0))
    assert out_empty.size == 0
    np.testing.assert_array_equal(ch.states()[0], z_before)
