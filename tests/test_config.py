import os

import pytest

from bci_sys import config


def test_target_direction_is_required_and_up():
    # C1：direction 必填、无默认；本项目靶点显式上调
    with pytest.raises(TypeError):
        config.Target(channel="Fz", band_hz=(4.0, 8.0))  # 缺 direction
    with pytest.raises(ValueError):
        config.Target(channel="Fz", band_hz=(4.0, 8.0), direction="sideways")
    t = config.Target(channel="Fz", band_hz=(4.0, 8.0), direction="up")
    assert t.channel == "Fz"
    assert t.direction == "up"


def test_module_target_constant():
    assert config.TARGET.channel == "Fz"
    assert config.TARGET.direction == "up"
    assert config.TARGET.band_hz == (4.0, 8.0)


def test_segment_constants():
    assert config.SFREQ == 250.0
    assert config.SEGMENT_SEC == 0.2
    assert config.SEGMENT_SAMPLES == 50


def test_data_root_default_and_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("BCI_DATA_ROOT", raising=False)
    assert config.data_root().name == "ds007169"
    monkeypatch.setenv("BCI_DATA_ROOT", str(tmp_path))
    assert config.data_root() == tmp_path


def test_eog_proxy_channel_names_uppercase():
    # 通道名以 channels.tsv 为准：FP1/FP2 大写
    assert config.EOG_PROXY_CHANNELS == ("FP1", "FP2")
