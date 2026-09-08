import pandas as pd
import pytest

from bci_sys import config, io_bids


class _FakeRaw:
    """最小 mne Raw 替身：只需要 ch_names。"""

    def __init__(self, ch_names):
        self.ch_names = ch_names


def test_channel_index_finds_fz():
    raw = _FakeRaw(["Pz", "Fz", "Cz"])
    assert io_bids.channel_index(raw, "Fz") == 1


def test_channel_index_raises_on_missing():
    raw = _FakeRaw(["Pz", "Cz"])
    with pytest.raises(KeyError, match="Fz"):
        io_bids.channel_index(raw, "Fz")


def test_paths_point_to_bids_files(data_root):
    vhdr = io_bids.vhdr_path("001", data_root)
    assert vhdr.name == "sub-001_task-nback_eeg.vhdr"
    assert vhdr.exists()
    assert io_bids.events_path("001", data_root).exists()


def test_load_events_and_trial_filter(data_root):
    ev = io_bids.load_events("001", data_root)
    assert "nback_level" in ev.columns
    assert "istutorial" in ev.columns
    tr = io_bids.real_trial_events(ev)
    # 非练习试次 400 个，1/2/3/4-back 各 100
    assert len(tr) == 400
    counts = tr["nback_level"].value_counts().sort_index()
    assert list(counts.index) == [1, 2, 3, 4]
    assert (counts == 100).all()
