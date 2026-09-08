import numpy as np
import pytest

from bci_sys import io_bids


def test_unit_std_check_passes_normal_eeg():
    rng = np.random.default_rng(0)
    t = np.arange(250 * 30) / 250.0
    # 30 µV 量级的 6 Hz 振荡 + 噪声 → 带通后 std 在正常范围
    x = 30.0 * np.sin(2 * np.pi * 6 * t) + 5.0 * rng.standard_normal(t.size)
    io_bids.check_fz_unit_std(x, sfreq=250.0)  # 不抛错即通过


def test_unit_std_check_catches_1e6_mistake():
    rng = np.random.default_rng(1)
    t = np.arange(250 * 30) / 250.0
    x = 30.0 * np.sin(2 * np.pi * 6 * t) + 5.0 * rng.standard_normal(t.size)
    with pytest.raises(ValueError, match="单位换算"):
        io_bids.check_fz_unit_std(x * 1e6, sfreq=250.0)  # 模拟误乘 1e6


def test_unit_std_check_catches_flat_signal():
    x = np.zeros(250 * 30)
    with pytest.raises(ValueError, match="单位换算"):
        io_bids.check_fz_unit_std(x, sfreq=250.0)


def test_preflight_passes_on_sub001(data_root):
    io_bids.preflight("001", data_root)  # 不抛错即通过


def test_preflight_fails_on_missing_subject(data_root):
    with pytest.raises(FileNotFoundError):
        io_bids.preflight("999", data_root)
