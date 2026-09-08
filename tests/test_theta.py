import numpy as np

from bci_sys import config
from bci_sys.offline import theta


def _sine(freq, amp=10.0, sec=30.0, sfreq=250.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * sfreq)) / sfreq
    return amp * np.sin(2 * np.pi * freq * t) + 0.5 * rng.standard_normal(t.size)


def test_theta_power_6hz_far_exceeds_20hz():
    sfreq = 250.0
    eps6 = np.stack([_sine(6.0)[:375] for _ in range(5)])
    eps20 = np.stack([_sine(20.0)[:375] for _ in range(5)])
    p6 = theta.theta_power_uV2(eps6, sfreq).mean()
    p20 = theta.theta_power_uV2(eps20, sfreq).mean()
    assert p6 > 50 * p20


def test_theta_power_matches_theoretical_amp_squared_half():
    # 纯正弦 A*sin(2πft) 全带功率 ≈ A²/2；theta 带含 6Hz，应接近 A²/2
    sfreq = 250.0
    t = np.arange(375) / sfreq
    epoch = 10.0 * np.sin(2 * np.pi * 6 * t)
    p = theta.theta_power_uV2(epoch[None, :], sfreq)[0]
    assert abs(p - (10.0 ** 2) / 2) / (10.0 ** 2 / 2) < 0.25  # 容差 25%


def test_slice_epochs_shape_and_no_overlap_counts():
    # 4 个 onset，间隔 1.7s，窗口 1.5s → 4 个 375 点 epoch
    sfreq = 250.0
    x = np.zeros(int(20 * sfreq))
    onsets = np.array([1.0, 2.7, 4.4, 6.1])
    eps = theta.slice_epochs(x, onsets, sfreq)
    assert eps.shape == (4, 375)


def test_slice_epochs_drops_incomplete_tail():
    sfreq = 250.0
    x = np.zeros(int(10.0 * sfreq))  # 10 秒信号
    onsets = np.array([1.0, 9.5])    # 9.5s 起的窗口(到 11.0s)超出末尾 → 丢弃
    eps = theta.slice_epochs(x, onsets, sfreq)
    assert eps.shape == (1, 375)


def test_pipeline_on_sub001_produces_400_epochs(data_root):
    from bci_sys import io_bids

    fz = io_bids.load_fz_uV("001", data_root)
    ev = io_bids.load_events("001", data_root)
    tr = io_bids.real_trial_events(ev)
    ff = theta.filter_whole(fz)
    eps = theta.slice_epochs(ff, tr["onset"].values)
    powers = theta.theta_power_uV2(eps)
    assert eps.shape[1] == 375
    assert 395 <= eps.shape[0] <= 400  # 末尾不完整 epoch 可能被丢
    assert powers.shape[0] == eps.shape[0]
    assert np.all(np.isfinite(powers)) and np.all(powers > 0)
