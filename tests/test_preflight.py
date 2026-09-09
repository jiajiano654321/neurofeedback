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


def test_preflight_fails_when_raw_missing_fp2(tmp_path, monkeypatch):
    """mne 侧缺 FP2：channel_index 直接 KeyError（硬停）。"""
    from types import SimpleNamespace

    rng = np.random.default_rng(2)
    n = 250 * 30
    names = ["Fz", "FP1"] + [f"E{i:02d}" for i in range(17)]  # 19 导，无 FP2
    data = np.stack(
        [30.0 * np.sin(2 * np.pi * 6 * np.arange(n) / 250.0) + 5.0 * rng.standard_normal(n)]
        + [rng.standard_normal(n) for _ in names[1:]]
    )
    fake_raw = SimpleNamespace(
        ch_names=names, info={"sfreq": 250.0}, get_data=lambda: data
    )
    monkeypatch.setattr(io_bids, "load_raw", lambda subj, root=None: fake_raw)

    eeg_dir = tmp_path / "sub-001" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-001_task-nback_events.tsv").write_text("onset\tnback_level\n", encoding="utf-8")

    with pytest.raises(KeyError, match="FP2"):
        io_bids.preflight("001", tmp_path)


def test_preflight_fails_when_channels_tsv_missing_fp2(tmp_path, monkeypatch):
    """mne 侧有 FP2 但 channels.tsv 没有：按名检查仍 KeyError。"""
    from types import SimpleNamespace
    import pandas as pd

    rng = np.random.default_rng(3)
    n = 250 * 30
    base = ["Fz", "FP1"] + [f"E{i:02d}" for i in range(17)]  # tsv 19 导无 FP2
    names = base + ["FP2"]
    data = np.stack(
        [30.0 * np.sin(2 * np.pi * 6 * np.arange(n) / 250.0) + 5.0 * rng.standard_normal(n)]
        + [rng.standard_normal(n) for _ in names[1:]]
    )
    fake_raw = SimpleNamespace(
        ch_names=names, info={"sfreq": 250.0}, get_data=lambda: data
    )
    monkeypatch.setattr(io_bids, "load_raw", lambda subj, root=None: fake_raw)

    eeg_dir = tmp_path / "sub-001" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-001_task-nback_events.tsv").write_text("onset\tnback_level\n", encoding="utf-8")
    pd.DataFrame({"name": base, "type": ["EEG"] * 19}).to_csv(
        eeg_dir / "sub-001_task-nback_channels.tsv", sep="\t", index=False
    )

    with pytest.raises(KeyError, match="FP2"):
        io_bids.preflight("001", tmp_path)
